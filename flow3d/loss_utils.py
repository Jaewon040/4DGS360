import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from .transforms import transform_rigid, homogenize_points, rt_to_mat4


def masked_mse_loss(pred, gt, mask=None, normalize=True, quantile: float = 1.0):
    if mask is None:
        return trimmed_mse_loss(pred, gt, quantile)
    else:
        sum_loss = F.mse_loss(pred, gt, reduction="none").mean(dim=-1, keepdim=True)
        quantile_mask = (
            (sum_loss < torch.quantile(sum_loss, quantile)).squeeze(-1)
            if quantile < 1
            else torch.ones_like(sum_loss, dtype=torch.bool).squeeze(-1)
        )
        ndim = sum_loss.shape[-1]
        if normalize:
            return torch.sum((sum_loss * mask)[quantile_mask]) / (
                ndim * torch.sum(mask[quantile_mask]) + 1e-8
            )
        else:
            return torch.mean((sum_loss * mask)[quantile_mask])


def masked_l1_loss(pred, gt, mask=None, normalize=True, quantile: float = 1.0):
    if mask is None:
        return trimmed_l1_loss(pred, gt, quantile)
    else:
        sum_loss = F.l1_loss(pred, gt, reduction="none").mean(dim=-1, keepdim=True)
        if quantile < 1:
            num = sum_loss.numel()
            if num < 16_000_000:
                threshold = torch.quantile(sum_loss, quantile)
            else:
                sorted, _ = torch.sort(sum_loss.reshape(-1))
                idxf = quantile * num
                idxi = int(idxf)
                threshold = sorted[idxi] + (sorted[idxi + 1] - sorted[idxi]) * (idxf - idxi)
            quantile_mask = (sum_loss < threshold).squeeze(-1)
        else: 
            quantile_mask = torch.ones_like(sum_loss, dtype=torch.bool).squeeze(-1)

        ndim = sum_loss.shape[-1]
        if normalize:
            return torch.sum((sum_loss * mask)[quantile_mask]) / (
                ndim * torch.sum(mask[quantile_mask]) + 1e-8
            )
        else:
            return torch.mean((sum_loss * mask)[quantile_mask])


def masked_huber_loss(pred, gt, delta, mask=None, normalize=True):
    if mask is None:
        return F.huber_loss(pred, gt, delta=delta)
    else:
        sum_loss = F.huber_loss(pred, gt, delta=delta, reduction="none")
        ndim = sum_loss.shape[-1]
        if normalize:
            return torch.sum(sum_loss * mask) / (ndim * torch.sum(mask) + 1e-8)
        else:
            return torch.mean(sum_loss * mask)


def trimmed_mse_loss(pred, gt, quantile=0.9):
    loss = F.mse_loss(pred, gt, reduction="none").mean(dim=-1)
    loss_at_quantile = torch.quantile(loss, quantile)
    trimmed_loss = loss[loss < loss_at_quantile].mean()
    return trimmed_loss


def trimmed_l1_loss(pred, gt, quantile=0.9):
    loss = F.l1_loss(pred, gt, reduction="none").mean(dim=-1)
    loss_at_quantile = torch.quantile(loss, quantile)
    trimmed_loss = loss[loss < loss_at_quantile].mean()
    return trimmed_loss


def compute_gradient_loss(pred, gt, mask, quantile=0.98):
    """
    Compute gradient loss
    pred: (batch_size, H, W, D) or (batch_size, H, W)
    gt: (batch_size, H, W, D) or (batch_size, H, W)
    mask: (batch_size, H, W), bool or float
    """
    # NOTE: messy need to be cleaned up
    mask_x = mask[:, :, 1:] * mask[:, :, :-1]
    mask_y = mask[:, 1:, :] * mask[:, :-1, :]
    pred_grad_x = pred[:, :, 1:] - pred[:, :, :-1]
    pred_grad_y = pred[:, 1:, :] - pred[:, :-1, :]
    gt_grad_x = gt[:, :, 1:] - gt[:, :, :-1]
    gt_grad_y = gt[:, 1:, :] - gt[:, :-1, :]
    loss = masked_l1_loss(
        pred_grad_x[mask_x][..., None], gt_grad_x[mask_x][..., None], quantile=quantile
    ) + masked_l1_loss(
        pred_grad_y[mask_y][..., None], gt_grad_y[mask_y][..., None], quantile=quantile
    )
    return loss


def knn(x: torch.Tensor, k: int) -> tuple[np.ndarray, np.ndarray]:
    x = x.cpu().numpy()
    knn_model = NearestNeighbors(
        n_neighbors=k + 1, algorithm="auto", metric="euclidean"
    ).fit(x)
    distances, indices = knn_model.kneighbors(x)
    return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)


def get_weights_for_procrustes(clusters, visibilities=None, invisible_weight=0.5):
    """
    Compute weights for Procrustes alignment.

    Args:
        clusters: (T, P, 3) - point positions
        visibilities: (T, P) - boolean visibility mask
        invisible_weight: float - weight multiplier for invisible points (default 0.5)
                         0.0 = ignore invisible points (original behavior)
                         0.5 = half weight for invisible points
                         1.0 = equal weight for invisible and visible points

    Returns:
        weights: (T, P) - weights where invisible points get reduced weight
    """
    clusters_median = clusters.median(dim=-2, keepdim=True)[0]
    dists2clusters_center = torch.norm(clusters - clusters_median, dim=-1)
    dists2clusters_center /= dists2clusters_center.median(dim=-1, keepdim=True)[0]
    weights = torch.exp(-dists2clusters_center)
    weights /= weights.mean(dim=-1, keepdim=True) + 1e-6
    if visibilities is not None:
        # visible: weight *= 1.0, invisible: weight *= invisible_weight
        visibility_multiplier = visibilities.float() + (~visibilities).float() * invisible_weight
        weights *= visibility_multiplier + 1e-6
    invalid = dists2clusters_center > np.quantile(
        dists2clusters_center.cpu().numpy(), 0.9
    )
    invalid |= torch.isnan(weights)
    weights[invalid] = 0
    return weights


def compute_z_acc_loss(means_ts_nb: torch.Tensor, w2cs: torch.Tensor):
    """
    :param means_ts (G, 3, B, 3)
    :param w2cs (B, 4, 4)
    return (float)
    """
    camera_center_t = torch.linalg.inv(w2cs)[:, :3, 3]  # (B, 3)
    ray_dir = F.normalize(
        means_ts_nb[:, 1] - camera_center_t, p=2.0, dim=-1
    )  # [G, B, 3]
    # acc = 2 * means[:, 1] - means[:, 0] - means[:, 2]  # [G, B, 3]
    # acc_loss = (acc * ray_dir).sum(dim=-1).abs().mean()
    acc_loss = (
        ((means_ts_nb[:, 1] - means_ts_nb[:, 0]) * ray_dir).sum(dim=-1) ** 2
    ).mean() + (
        ((means_ts_nb[:, 2] - means_ts_nb[:, 1]) * ray_dir).sum(dim=-1) ** 2
    ).mean()
    return acc_loss


def compute_se3_smoothness_loss(
    rots: torch.Tensor,
    transls: torch.Tensor,
    weight_rot: float = 1.0,
    weight_transl: float = 2.0,
):
    """
    central differences
    :param motion_transls (K, T, 3)
    :param motion_rots (K, T, 6)
    """
    r_accel_loss = compute_accel_loss(rots)
    t_accel_loss = compute_accel_loss(transls)
    return r_accel_loss * weight_rot + t_accel_loss * weight_transl


def compute_accel_loss(transls):
    accel = 2 * transls[:, 1:-1] - transls[:, :-2] - transls[:, 2:]
    loss = accel.norm(dim=-1).mean()
    return loss


def curve_distance_knn(
    points: torch.Tensor,
    k : int
):
    """
    : points: [p, T, 3]
    """
    distance = torch.norm(points.unsqueeze(1) - points.unsqueeze(0), dim=-1) 
    curve_distance, _ = torch.max(distance, dim=-1) 
    _, idx = curve_distance.topk(k + 1, largest=False, dim=-1)
    return idx[..., 1:]


def compute_arap_loss(
    ts: torch.Tensor,
    positions: torch.Tensor,
    curve_transforms: torch.Tensor,
    transforms:torch.Tensor,
    k: int,
    weight1: float,
    weight2: float,
):
    positions = positions.unsqueeze(-2).detach()
    curve_transforms = rt_to_mat4(curve_transforms[..., :3, :3], curve_transforms[..., :3, -1])
    with torch.no_grad():
        curve_positions = transform_rigid(homogenize_points(positions), curve_transforms)[...,:-1]
        indices = curve_distance_knn(curve_positions, 2*k)
        rand_indices = torch.stack([torch.randperm(indices.shape[1]) for _ in range(indices.shape[0])])
        indices = indices[torch.arange(indices.shape[0]).unsqueeze(1), rand_indices][...,:k]

    transforms = rt_to_mat4(transforms[..., :3, :3], transforms[..., :3, -1])
    positions = transform_rigid(homogenize_points(positions), transforms)[..., :-1]
    positions_nb = positions[indices]
    transforms_nb = transforms[indices]
    positions = positions.unsqueeze(1)

    # Check if we have enough time steps (need at least 2)
    num_time_steps = positions.shape[2]
    if num_time_steps < 2:
        return torch.tensor(0.0, device=positions.device, requires_grad=True)

    positions_t = positions[:, :, :-1, :]
    positions_dt = positions[:, :, 1:, :]
    positions_nb_t = positions_nb[:, :, :-1, :]
    positions_nb_dt = positions_nb[:, :, 1:, :]
    transforms_nb_t = transforms_nb[:, :, :-1, :, :]
    transforms_nb_dt = transforms_nb[:, :, 1:, :, :]

    first_term = torch.norm(positions_t - positions_nb_t, dim=-1)-torch.norm(positions_dt- positions_nb_dt, dim=-1)
    first_term = torch.abs(first_term)

    second_term = torch.norm(transform_rigid(homogenize_points(positions_t), transforms_nb_t.inverse())-transform_rigid(homogenize_points(positions_dt), transforms_nb_dt.inverse()), dim=-1)

    return torch.mean(weight1 * first_term + weight2 * second_term)


def compute_radius_loss(
    positions: torch.Tensor,
    radius: torch.Tensor,
    k: int,
):
    dist, _ = knn(positions.detach(),k)
    dist = torch.from_numpy(dist).to(positions.device)
    dist = torch.mean(dist, dim=-1)

    loss = torch.clamp(radius - dist, min=0.)
    loss = loss**2
    return torch.mean(loss)


def compute_node_visibility(
    fg_visibilities: torch.Tensor,  # [G, T]
    fg_means: torch.Tensor,  # [G, 3]
    motion_tree,
    t: int,
    level: int = 0,
) -> torch.Tensor:
    """
    Compute node visibility at time t by aggregating Gaussian visibilities.

    Args:
        fg_visibilities: Gaussian visibilities [G, T]
        fg_means: Gaussian means in canonical space [G, 3]
        motion_tree: MotionTree object
        t: Time step
        level: Motion tree level

    Returns:
        node_vis: Node visibility at time t [num_nodes], values in [0, 1]
    """
    device = fg_means.device
    num_nodes = motion_tree.motion_nodes[level].num_nodes

    # Get Gaussian visibility at time t [G]
    gaussian_vis_t = fg_visibilities[:, t].float()  # [G]

    # Compute KNN: which nodes affect which Gaussians [G, k]
    nn_idx, nn_weight = motion_tree.compute_knn_nodes(fg_means, k=3, level=level, no_softmax=False)

    # Aggregate visibility to nodes using weighted average
    node_vis = torch.zeros(num_nodes, device=device)
    node_count = torch.zeros(num_nodes, device=device)

    # For each Gaussian, distribute its visibility to its neighboring nodes
    for k in range(nn_idx.shape[1]):  # k=3
        node_indices = nn_idx[:, k]  # [G]
        weights = nn_weight[:, k]  # [G]
        weighted_vis = gaussian_vis_t * weights  # [G]

        node_vis.index_add_(0, node_indices, weighted_vis)
        node_count.index_add_(0, node_indices, weights)

    # Normalize by total weight
    node_vis = node_vis / (node_count + 1e-8)

    return node_vis


def compute_structural_loss(
    motion_tree,
    node_positions_cano: torch.Tensor,  # [N, 3]
    ts: torch.Tensor,  # [B]
    w2cs: torch.Tensor,  # [B, 4, 4]
    Ks: torch.Tensor,  # [B, 3, 3]
    img_wh: tuple,  # (W, H)
    num_frames: int,
    num_sample_rays: int = 50,
    patch_size: int = 5,
    max_nodes_per_patch: int = 20,
    curve_sampling_interval: int = 15,
    time_weight_decay: str = "linear",
    level: int = 0,
) -> torch.Tensor:
    """
    Compute structural loss to preserve front/back node distances across time.

    Args:
        motion_tree: MotionTree object
        node_positions_cano: Canonical node positions [N, 3]
        ts: Batch times [B]
        w2cs: World-to-camera matrices [B, 4, 4]
        Ks: Camera intrinsics [B, 3, 3]
        img_wh: Image size (W, H)
        num_frames: Total number of frames
        num_sample_rays: Number of rays to sample
        patch_size: Patch size for gathering nodes
        max_nodes_per_patch: Max nodes per patch
        curve_sampling_interval: Sampling interval for curve (15 like ARAP)
        time_weight_decay: "linear" or "exponential"
        level: Motion tree level

    Returns:
        loss: Structural loss value
    """
    device = node_positions_cano.device
    N = node_positions_cano.shape[0]
    B = ts.shape[0]
    W, H = img_wh

    # Step 1: Compute curve positions and speeds (全体궤적)
    curve_times = torch.arange(0, num_frames, curve_sampling_interval, device=device)
    curve_transforms = motion_tree.compute_node_world_transforms(curve_times, level=level)
    # curve_transforms: [N, T_curve, 4, 4]
    curve_transforms_mat = rt_to_mat4(curve_transforms[..., :3, :3], curve_transforms[..., :3, -1])
    curve_positions = transform_rigid(
        homogenize_points(node_positions_cano.unsqueeze(-2)),
        curve_transforms_mat
    )[..., :3]  # [N, T_curve, 3]

    # Compute speeds: ||pos(t+1) - pos(t)||
    velocities = curve_positions[:, 1:] - curve_positions[:, :-1]  # [N, T_curve-1, 3]
    speeds = torch.norm(velocities, dim=-1)  # [N, T_curve-1]

    # Step 2: Compute batch-time node positions for rasterization
    batch_transforms = motion_tree.compute_node_world_transforms(ts, level=level)
    batch_transforms_mat = rt_to_mat4(batch_transforms[..., :3, :3], batch_transforms[..., :3, -1])
    batch_positions = transform_rigid(
        homogenize_points(node_positions_cano.unsqueeze(-2)),
        batch_transforms_mat
    )[..., :3]  # [N, B, 3]

    total_loss = 0.0
    num_pairs = 0

    # Process each batch camera
    for b in range(B):
        # Step 3: Project nodes to image plane
        uvs, depths = _project_nodes_to_image(
            batch_positions[:, b, :], Ks[b], w2cs[b], img_wh
        )  # [N, 2], [N]

        # Skip if no valid projections
        if depths.numel() == 0:
            continue

        # Step 4-5: Sample rays and define patches
        valid_node_ids = torch.where((depths > 0) & (uvs[:, 0] >= 0) & (uvs[:, 0] < W) &
                                      (uvs[:, 1] >= 0) & (uvs[:, 1] < H))[0]
        if len(valid_node_ids) < num_sample_rays:
            num_sample_rays_actual = len(valid_node_ids)
        else:
            num_sample_rays_actual = num_sample_rays

        if num_sample_rays_actual == 0:
            continue

        sampled_ids = valid_node_ids[torch.randperm(len(valid_node_ids))[:num_sample_rays_actual]]
        ray_centers = uvs[sampled_ids]  # [num_rays, 2]

        # Step 6-8: Gather nodes in patches and sort by depth
        patch_nodes, patch_valid = _gather_patch_nodes(
            uvs, depths, ray_centers, patch_size, max_nodes_per_patch
        )  # [num_rays, max_nodes], [num_rays, max_nodes]

        # Step 9-11: Split into front/back and sample
        front_sampled, back_sampled = _sample_front_back_nodes(
            patch_nodes, patch_valid, num_front=2, num_back=5
        )  # [num_rays, 2], [num_rays, 5]

        # Step 12-14: Speed pattern matching
        pairs = _match_pairs_by_speed_pattern(
            front_sampled, back_sampled, speeds
        )  # [num_valid_pairs, 2]

        if pairs.shape[0] == 0:
            continue

        # Step 15: Compute distance loss
        # Use consecutive times from batch
        if b < B - 1:
            t1, t2 = ts[b], ts[b + 1]
        else:
            t1, t2 = ts[b], ts[b]

        loss_b = _compute_distance_preservation_loss(
            pairs, node_positions_cano, motion_tree, t1, t2,
            num_frames, time_weight_decay, level
        )

        total_loss += loss_b
        num_pairs += pairs.shape[0]

    if num_pairs > 0:
        return total_loss / B
    else:
        return torch.tensor(0.0, device=device, requires_grad=True)


def _project_nodes_to_image(
    positions: torch.Tensor,  # [N, 3]
    K: torch.Tensor,  # [3, 3]
    w2c: torch.Tensor,  # [4, 4]
    img_wh: tuple,  # (W, H)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project 3D positions to 2D image coordinates."""
    # Transform to camera space
    positions_cam = transform_rigid(
        homogenize_points(positions.unsqueeze(-2)),
        w2c.unsqueeze(0).expand(positions.shape[0], -1, -1).unsqueeze(-3)
    ).squeeze(-2)[..., :3]  # [N, 3]

    depths = positions_cam[:, 2]  # [N]

    # Project to image plane
    positions_2d = torch.matmul(K, positions_cam.T).T  # [N, 3]
    uvs = positions_2d[:, :2] / (positions_2d[:, 2:3] + 1e-6)  # [N, 2]

    return uvs, depths


def _gather_patch_nodes(
    uvs: torch.Tensor,  # [N, 2]
    depths: torch.Tensor,  # [N]
    ray_centers: torch.Tensor,  # [num_rays, 2]
    patch_size: int,
    max_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather nodes within patches around ray centers."""
    num_rays = ray_centers.shape[0]
    N = uvs.shape[0]
    device = uvs.device

    # Define patch boundaries (half_size on each side)
    half_size = patch_size // 2
    patch_min = ray_centers - half_size  # [num_rays, 2]
    patch_max = ray_centers + half_size + 1  # [num_rays, 2]

    # Check which nodes are in which patches: [N, num_rays]
    in_patch_x = (uvs[:, 0:1] >= patch_min[:, 0:1].T) & (uvs[:, 0:1] < patch_max[:, 0:1].T)
    in_patch_y = (uvs[:, 1:2] >= patch_min[:, 1:2].T) & (uvs[:, 1:2] < patch_max[:, 1:2].T)
    in_patch = in_patch_x & in_patch_y  # [N, num_rays]

    # For each ray, gather node ids and depths
    patch_nodes = torch.full((num_rays, max_nodes), -1, dtype=torch.long, device=device)
    patch_depths = torch.full((num_rays, max_nodes), float('inf'), device=device)

    for r in range(num_rays):
        node_ids_in_patch = torch.where(in_patch[:, r])[0]
        if len(node_ids_in_patch) == 0:
            continue

        # Get depths for these nodes
        depths_in_patch = depths[node_ids_in_patch]

        # Sort by depth and limit to max_nodes
        sorted_indices = torch.argsort(depths_in_patch)
        num_to_keep = min(len(sorted_indices), max_nodes)

        # Random sampling if more than max_nodes
        if len(sorted_indices) > max_nodes:
            keep_indices = sorted_indices[torch.randperm(len(sorted_indices))[:max_nodes]]
        else:
            keep_indices = sorted_indices[:num_to_keep]

        patch_nodes[r, :num_to_keep] = node_ids_in_patch[keep_indices]
        patch_depths[r, :num_to_keep] = depths_in_patch[keep_indices]

    # Sort by depth within each patch
    sorted_depth_indices = torch.argsort(patch_depths, dim=-1)
    patch_nodes_sorted = torch.gather(patch_nodes, 1, sorted_depth_indices)

    # Valid mask
    valid_mask = (patch_nodes_sorted != -1)

    return patch_nodes_sorted, valid_mask


def _sample_front_back_nodes(
    patch_nodes: torch.Tensor,  # [num_rays, max_nodes]
    valid_mask: torch.Tensor,  # [num_rays, max_nodes]
    num_front: int = 2,
    num_back: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample front and back nodes from each patch."""
    num_rays, max_nodes = patch_nodes.shape
    device = patch_nodes.device
    mid_idx = max_nodes // 2

    front_half = patch_nodes[:, :mid_idx]  # [num_rays, max_nodes//2]
    front_valid = valid_mask[:, :mid_idx]

    # Changed: Sample back nodes from entire ray (not just back half)
    back_all = patch_nodes  # [num_rays, max_nodes]
    back_valid = valid_mask

    # Sample from front
    front_sampled = torch.full((num_rays, num_front), -1, dtype=torch.long, device=device)
    for r in range(num_rays):
        valid_front = front_half[r][front_valid[r]]
        if len(valid_front) > 0:
            num_sample = min(num_front, len(valid_front))
            indices = torch.randperm(len(valid_front))[:num_sample]
            front_sampled[r, :num_sample] = valid_front[indices]

    # Sample from back (now from entire ray)
    back_sampled = torch.full((num_rays, num_back), -1, dtype=torch.long, device=device)
    for r in range(num_rays):
        valid_back = back_all[r][back_valid[r]]
        if len(valid_back) > 0:
            num_sample = min(num_back, len(valid_back))
            indices = torch.randperm(len(valid_back))[:num_sample]
            back_sampled[r, :num_sample] = valid_back[indices]

    return front_sampled, back_sampled


def _match_pairs_by_speed_pattern(
    front_sampled: torch.Tensor,  # [num_rays, num_front]
    back_sampled: torch.Tensor,  # [num_rays, num_back]
    speeds: torch.Tensor,  # [N, T_curve-1]
) -> torch.Tensor:
    """Match front-back pairs based on speed pattern similarity."""
    num_rays = front_sampled.shape[0]
    num_front = front_sampled.shape[1]
    num_back = back_sampled.shape[1]
    device = speeds.device

    all_pairs = []

    for r in range(num_rays):
        front_ids = front_sampled[r]  # [num_front]
        back_ids = back_sampled[r]  # [num_back]

        # Filter out invalid (-1) nodes
        valid_front = front_ids[front_ids >= 0]
        valid_back = back_ids[back_ids >= 0]

        if len(valid_front) == 0 or len(valid_back) == 0:
            continue

        # Get speed patterns
        speeds_front = speeds[valid_front]  # [num_valid_front, T_curve-1]
        speeds_back = speeds[valid_back]  # [num_valid_back, T_curve-1]

        # Compute relative difference: |s1 - s2| / (s1 + s2)
        # Broadcasting: [num_valid_front, 1, T] and [1, num_valid_back, T]
        rel_diff = torch.abs(
            speeds_front[:, None, :] - speeds_back[None, :, :]
        ) / (
            speeds_front[:, None, :] + speeds_back[None, :, :] + 1e-6
        )  # [num_valid_front, num_valid_back, T_curve-1]

        # Average over time (pattern similarity)
        pattern_sim = rel_diff.mean(dim=-1)  # [num_valid_front, num_valid_back]

        # For each front node, select top-2 most similar back nodes
        num_select = min(2, len(valid_back))
        top_indices = torch.topk(pattern_sim, k=num_select, dim=-1, largest=False).indices

        # Create pairs
        for f_idx, f_id in enumerate(valid_front):
            for b_idx in range(num_select):
                b_id = valid_back[top_indices[f_idx, b_idx]]
                all_pairs.append([f_id.item(), b_id.item()])

    if len(all_pairs) == 0:
        return torch.empty((0, 2), dtype=torch.long, device=device)

    return torch.tensor(all_pairs, dtype=torch.long, device=device)


def _compute_distance_preservation_loss(
    pairs: torch.Tensor,  # [num_pairs, 2]
    node_positions_cano: torch.Tensor,  # [N, 3]
    motion_tree,
    t1: torch.Tensor,
    t2: torch.Tensor,
    num_frames: int,
    time_weight_decay: str,
    level: int,
) -> torch.Tensor:
    """Compute distance preservation loss for pairs."""
    if pairs.shape[0] == 0:
        return torch.tensor(0.0, device=pairs.device, requires_grad=True)

    device = pairs.device

    # Get node positions at t1 and t2
    times = torch.stack([t1, t2])  # [2]
    transforms = motion_tree.compute_node_world_transforms(times, level=level)
    transforms_mat = rt_to_mat4(transforms[..., :3, :3], transforms[..., :3, -1])

    positions_t = transform_rigid(
        homogenize_points(node_positions_cano.unsqueeze(-2)),
        transforms_mat
    )[..., :3]  # [N, 2, 3]

    # Get pair positions
    node_A = pairs[:, 0]  # [num_pairs]
    node_B = pairs[:, 1]  # [num_pairs]

    pos_A_t1 = positions_t[node_A, 0, :]  # [num_pairs, 3]
    pos_B_t1 = positions_t[node_B, 0, :]
    pos_A_t2 = positions_t[node_A, 1, :]
    pos_B_t2 = positions_t[node_B, 1, :]

    # Compute distances
    dist_t1 = torch.norm(pos_A_t1 - pos_B_t1, dim=-1)  # [num_pairs]
    dist_t2 = torch.norm(pos_A_t2 - pos_B_t2, dim=-1)

    # Time-based weight
    time_diff = torch.abs(t1 - t2).float()
    if time_weight_decay == "linear":
        weight = 1.0 - time_diff / num_frames
    else:  # exponential
        weight = torch.exp(-2.0 * time_diff / num_frames)

    weight = torch.clamp(weight, min=0.0, max=1.0)

    # Distance preservation loss
    loss = weight * torch.abs(dist_t1 - dist_t2).mean()

    return loss