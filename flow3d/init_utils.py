import time
from typing import Literal

import cupy as cp
import numpy as np

import roma
import torch
import torch.nn.functional as F
from cuml import HDBSCAN, KMeans
from loguru import logger as guru
from tqdm import tqdm

from flow3d.loss_utils import (
    compute_accel_loss,
    compute_se3_smoothness_loss,
    compute_z_acc_loss,
    get_weights_for_procrustes,
    knn,
    masked_l1_loss,
)
from flow3d.params import GaussianParams
from flow3d.tensor_dataclass import StaticObservations, TrackObservations
from flow3d.transforms import solve_procrustes, cont_6d_to_rmat
from flow3d.vis.utils import  get_server, project_2d_tracks



def init_fg_from_tracks_3d(
    cano_t: int, tracks_3d: TrackObservations
) -> GaussianParams:
    """
    using dataclasses individual tensors so we know they're consistent
    and are always masked/filtered together
    """
    num_fg = tracks_3d.xyz.shape[0]

    # Initialize gaussian colors.
    colors = torch.logit(tracks_3d.colors)
    # Initialize gaussian scales: find the average of the three nearest
    # neighbors in the first frame for each point and use that as the
    # scale.
    dists, _ = knn(tracks_3d.xyz[:, cano_t], 3)
    dists = torch.from_numpy(dists)
    scales = dists.mean(dim=-1, keepdim=True)
    scales = scales.clamp(torch.quantile(scales, 0.05), torch.quantile(scales, 0.95))
    scales = torch.log(scales.repeat(1, 3))
    # Initialize gaussian means.
    means = tracks_3d.xyz[:, cano_t]
    # Initialize gaussian orientations as random.
    quats = torch.rand(num_fg, 4)
    # Initialize gaussian opacities.
    opacities = torch.logit(torch.full((num_fg,), 0.7))

    # Initialize visibilities from tracks_3d.visibles [G, T]
    # Convert boolean to float: 0.0 = invisible, 1.0 = visible
    visibilities = tracks_3d.visibles.float()

    gaussians = GaussianParams(means, quats, scales, colors, opacities, visibilities=visibilities)
    return gaussians


def init_node_params_from_tracks_3d(
    cano_t: int,
    num_nodes: int,
    tracks_3d: TrackObservations,
    motion_coefs: torch.Tensor,
    velocity_weight: float = 0.0,  # 0.0 = spatial only (default), 0.5 = balanced
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    # Extract the 3D positions at the canonical time frame
    canonical_positions = tracks_3d.xyz[:, cano_t]  # [num_tracks, 3]

    # Perform weighted sampling to select node indices
    # Pass full trajectories if velocity weighting is enabled
    selected_indices = knn_weighted_sampling(
        canonical_positions,
        k=3,
        num_samples=num_nodes,
        tracks_3d=tracks_3d.xyz if velocity_weight > 0 else None,
        velocity_weight=velocity_weight
    )

    selected_means = canonical_positions[selected_indices]
    selected_motion_coefs = motion_coefs[selected_indices]

    # calculate distances and compute node radius
    dists, _ = knn(selected_means, k=3)
    radius = torch.from_numpy(dists).mean(dim=-1)
    radius = (radius).clamp(
        min=torch.quantile(radius, 0.05), max=torch.quantile(radius, 0.95)
    )
    radius = torch.log(radius)

    return selected_means, selected_motion_coefs, radius


def knn_weighted_sampling(
    points: torch.Tensor,
    k: int,
    num_samples: int,
    tracks_3d: torch.Tensor = None,  # [N, T, 3] optional full trajectories
    velocity_weight: float = 0.0,     # 0.0 = spatial only, 0.5 = balanced, 1.0 = velocity only
) -> torch.Tensor:

    dists, _ = knn(points, k)
    dists = torch.from_numpy(dists).mean(
        dim=-1
    )  # Average distance for each track point

    # Compute global mean of the distance
    global_average_dists = torch.mean(dists)

    # Compute pairwise distance between tracks (using chunks to avoid OOM)
    N = points.shape[0]
    chunk_size = 10000  # Process 1000 points at a time
    num_neighbors_in_range = torch.zeros(N, device=points.device, dtype=torch.long)

    for i in range(0, N, chunk_size):
        chunk_end = min(i + chunk_size, N)
        # Compute distances from all points to this chunk
        dists_chunk = torch.norm(
            points.unsqueeze(1) - points[i:chunk_end].unsqueeze(0),
            dim=-1
        )  # [N, chunk_size]
        # Count neighbors within global average distance for this chunk
        num_neighbors_in_range += torch.sum(dists_chunk < global_average_dists, dim=-1)

    # Subtract 1 to exclude self-distance
    num_neighbors_in_range = num_neighbors_in_range - 1
    num_neighbors_in_range = num_neighbors_in_range.float()

    spatial_weight = torch.exp(-num_neighbors_in_range)

    # Add velocity-based weighting if enabled
    if tracks_3d is not None and velocity_weight > 0:
        # Compute velocities: [N, T-1, 3]
        velocities = tracks_3d[:, 1:] - tracks_3d[:, :-1]

        # Average speed for each point: [N]
        speeds = torch.norm(velocities, dim=-1).mean(dim=-1)

        # Normalize speed (mean speed = 1.0)
        speed_normalized = speeds / (speeds.mean() + 1e-6)

        # Convert to importance weight (faster points get higher weight)
        velocity_importance = torch.exp(speed_normalized - 1.0)

        # Combine using geometric mean for balance
        combined_weight = (
            spatial_weight ** (1 - velocity_weight) *
            velocity_importance ** velocity_weight
        )

        guru.info(f"Velocity-based node sampling enabled (weight={velocity_weight:.2f})")
        guru.info(f"  Speed stats: min={speeds.min():.4f}, mean={speeds.mean():.4f}, max={speeds.max():.4f}")
    else:
        combined_weight = spatial_weight

    selected_indices = torch.multinomial(combined_weight, num_samples=num_samples)

    return selected_indices


def init_bg(
    points: StaticObservations,
) -> GaussianParams:
    """
    using dataclasses instead of individual tensors so we know they're consistent
    and are always masked/filtered together
    """
    num_init_bg_gaussians = points.xyz.shape[0]
    bg_scene_center = points.xyz.mean(0)
    bg_points_centered = points.xyz - bg_scene_center
    bg_min_scale = bg_points_centered.quantile(0.05, dim=0)
    bg_max_scale = bg_points_centered.quantile(0.95, dim=0)
    bg_scene_scale = torch.max(bg_max_scale - bg_min_scale).item() / 2.0
    bkdg_colors = torch.logit(points.colors)

    # Initialize gaussian scales: find the average of the three nearest
    # neighbors in the first frame for each point and use that as the
    # scale.
    dists, _ = knn(points.xyz, 3)
    dists = torch.from_numpy(dists)
    bg_scales = dists.mean(dim=-1, keepdim=True)
    bkdg_scales = torch.log(bg_scales.repeat(1, 3))

    bg_means = points.xyz

    # Initialize gaussian orientations by normals.
    local_normals = points.normals.new_tensor([[0.0, 0.0, 1.0]]).expand_as(
        points.normals
    )
    bg_quats = roma.rotvec_to_unitquat(
        F.normalize(torch.linalg.cross(local_normals, points.normals, dim=-1), dim=-1)
        * (local_normals * points.normals).sum(-1, keepdim=True).acos_()
    ).roll(1, dims=-1)
    bg_opacities = torch.logit(torch.full((num_init_bg_gaussians,), 0.7))
    gaussians = GaussianParams(
        bg_means,
        bg_quats,
        bkdg_scales,
        bkdg_colors,
        bg_opacities,
        scene_center=bg_scene_center,
        scene_scale=bg_scene_scale,
    )
    return gaussians


def init_motion_params_with_procrustes(
    tracks_3d: TrackObservations,
    num_bases: int,
    rot_type: Literal["quat", "6d"],
    cano_t: int,
    cluster_init_method: str = "kmeans",
    min_mean_weight: float = 0.1,
    vis: bool = False,
    port: int | None = None,
    invisible_weight: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, TrackObservations]:
    device = tracks_3d.xyz.device
    num_frames = tracks_3d.xyz.shape[1]
    # sample centers and get initial se3 motion bases by solving procrustes
    means_cano = tracks_3d.xyz[:, cano_t].clone()  # [num_gaussians, 3]

    # remove outliers
    scene_center = means_cano.median(dim=0).values
    # print(f"{scene_center=}")
    dists = torch.norm(means_cano - scene_center, dim=-1)
    dists_th = torch.quantile(dists, 0.95)
    valid_mask = dists < dists_th

    # remove tracks that are not visible in any frame
    valid_mask = valid_mask & tracks_3d.visibles.any(dim=1)
    # print(f"{valid_mask.sum()=}")

    tracks_3d = tracks_3d.filter_valid(valid_mask)  # valid tracks [Gaussian, T, 3]

    if vis and port is not None:
        server = get_server(port)
        try:
            pts = tracks_3d.xyz.cpu().numpy()
            clrs = tracks_3d.colors.cpu().numpy()
            while True:
                for t in range(num_frames):
                    server.scene.add_point_cloud("points", pts[:, t], clrs)
                    time.sleep(0.3)
        except KeyboardInterrupt:
            pass

    means_cano = means_cano[valid_mask]  # [gaussian, 3]

    sampled_centers, num_bases, labels = sample_initial_bases_centers(
        cluster_init_method, cano_t, tracks_3d, num_bases
    )  

    # assign each point to the label to compute the cluster weight
    ids, counts = labels.unique(return_counts=True)
    ids = ids[counts > 100]
    num_bases = len(ids)
    sampled_centers = sampled_centers[:, ids]
    # print(f"{num_bases=} {sampled_centers.shape=}")

    # compute basis weights from the distance to the cluster centers
    dists2centers = torch.norm(
        means_cano[:, None] - sampled_centers, dim=-1
    )  # [gaussian, sampled_centters] [37548, 10]
    motion_coefs = 10 * torch.exp(-dists2centers)

    init_rots, init_ts = [], []

    if rot_type == "quat":
        id_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        rot_dim = 4
    else:
        id_rot = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=device)
        rot_dim = 6

    init_rots = id_rot.reshape(1, 1, rot_dim).repeat(num_bases, num_frames, 1)
    init_ts = torch.zeros(num_bases, num_frames, 3, device=device)
    errs_before = np.full((num_bases, num_frames), -1.0)
    errs_after = np.full((num_bases, num_frames), -1.0)


    tgt_ts = list(range(cano_t - 1, -1, -1)) + list(range(cano_t, num_frames))
    # print(f"{tgt_ts=}")
    skipped_ts = {}
    for n, cluster_id in enumerate(ids):
        mask_in_cluster = labels == cluster_id  # for gaussian in each cluster
        cluster = tracks_3d.xyz[mask_in_cluster].transpose(
            0, 1
        )  # [num_frames, n_pts, 3]
        visibilities = tracks_3d.visibles[mask_in_cluster].swapaxes(
            0, 1
        )  # [num_frames, n_pts]
        confidences = tracks_3d.confidences[mask_in_cluster].swapaxes(
            0, 1
        )  # [num_frames, n_pts]
        weights = get_weights_for_procrustes(
            cluster, visibilities, invisible_weight=invisible_weight
        )  #[timesteps, gaussian in this cluster]
        prev_t = cano_t
        cluster_skip_ts = []
        for cur_t in tgt_ts:
            # compute pairwise transform from cano_t
            procrustes_weights = (
                weights[cano_t]
                * weights[cur_t]
                * (confidences[cano_t] + confidences[cur_t])
                / 2
            )
            if procrustes_weights.sum() < min_mean_weight * num_frames:
                init_rots[n, cur_t] = init_rots[n, prev_t]
                init_ts[n, cur_t] = init_ts[n, prev_t]
                cluster_skip_ts.append(cur_t)
            else:
                se3, (err, err_before) = solve_procrustes(
                    cluster[cano_t],
                    cluster[cur_t],
                    weights=procrustes_weights,
                    enforce_se3=True,
                    rot_type=rot_type,
                )
                init_rot, init_t, _ = se3
                assert init_rot.shape[-1] == rot_dim
                # double cover
                if rot_type == "quat" and torch.linalg.norm(
                    init_rot - init_rots[n][prev_t]
                ) > torch.linalg.norm(-init_rot - init_rots[n][prev_t]):
                    init_rot = -init_rot
                init_rots[n, cur_t] = init_rot
                init_ts[n, cur_t] = init_t
                errs_after[n, cur_t] = err
                errs_before[n, cur_t] = err_before
            prev_t = cur_t
        skipped_ts[cluster_id.item()] = cluster_skip_ts

    guru.info(f"{skipped_ts=}")
    guru.info(
        "procrustes init median error: {:.5f} => {:.5f}".format(
            np.median(errs_before[errs_before > 0]),
            np.median(errs_after[errs_after > 0]),
        )
    )
    guru.info(
        "procrustes init mean error: {:.5f} => {:.5f}".format(
            np.mean(errs_before[errs_before > 0]), np.mean(errs_after[errs_after > 0])
        )
    )
    guru.info(f"{init_rots.shape=}, {init_ts.shape=}, {motion_coefs.shape=}")

    return init_rots, init_ts , motion_coefs, tracks_3d



def run_initial_optim(
    fg: GaussianParams,
    init_rots: torch.Tensor,
    init_ts: torch.Tensor,
    tracks_3d: TrackObservations,
    motion_coefs: torch.Tensor,
    Ks: torch.Tensor,
    w2cs: torch.Tensor,
    num_iters: int = 1000,
    use_depth_range_loss: bool = False,
):
    """
    :param motion_rots: [num_bases, num_frames, 4|6]
    :param motion_transls: [num_bases, num_frames, 3]
    :param motion_coefs: [num_bases, num_frames]
    :param means: [num_gaussians, 3]
    """
    init_rots.requires_grad = True
    init_ts.requires_grad = True
    motion_coefs.requires_grad = True
    optimizer = torch.optim.Adam(
        [
            {"params": init_rots, "lr": 1e-2},
            {"params": init_ts, "lr": 3e-2},
            {"params": motion_coefs, "lr": 1e-2},
            {"params": fg.params["means"], "lr": 1e-3},
        ],
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=0.1 ** (1 / num_iters)
    )

    num_frames = init_rots.shape[1]
    device = init_rots.device

    w_smooth_func = lambda i, min_v, max_v, th: (
        min_v if i <= th else (max_v - min_v) * (i - th) / (num_iters - th) + min_v
    )

    gt_2d, gt_depth = project_2d_tracks(
        tracks_3d.xyz.swapaxes(0, 1), Ks, w2cs, return_depth=True
    )
    # (G, T, 2)
    gt_2d = gt_2d.swapaxes(0, 1)
    # (G, T)
    gt_depth = gt_depth.swapaxes(0, 1)

    ts = torch.arange(0, num_frames, device=device)
    ts_clamped = torch.clamp(ts, min=1, max=num_frames - 2)
    ts_neighbors = torch.cat((ts_clamped - 1, ts_clamped, ts_clamped + 1))  # i (3B,)

    def compute_transforms(ts, coefs, rots, transls):
        rots = rots[:, ts]
        transls = transls[:, ts]

        rots = torch.einsum(
            "pk, kni -> pni", coefs, rots
        )
        transls = torch.einsum(
            "pk, kni -> pni", coefs, transls
        )
        rotmats = cont_6d_to_rmat(rots)
        return torch.cat(
            [rotmats, transls[..., None]], dim=-1
        )

    pbar = tqdm(range(0, num_iters))
    for i in pbar:
        coefs = F.softmax(motion_coefs, dim=-1)
        transfms = compute_transforms(ts, coefs, init_rots, init_ts)
        positions = torch.einsum(
            "pnij,pj->pni",
            transfms,
            F.pad(fg.params["means"], (0, 1), value=1.0),
        )

        loss = 0.0
        # Use 3D loss for all points (visible and invisible)
        # Confidence is already set to 1.0 for visible, 0.5 for invisible
        track_3d_loss = masked_l1_loss(
            positions,
            tracks_3d.xyz,
            tracks_3d.confidences[..., None],
        )
        loss += track_3d_loss * 1.0

        if use_depth_range_loss:
            # Compute predicted depth for depth range loss
            pred_2d, pred_depth = project_2d_tracks(
                positions.swapaxes(0, 1), Ks, w2cs, return_depth=True
            )
            pred_depth = pred_depth.swapaxes(0, 1)
            near_depths = torch.quantile(gt_depth, 0.0, dim=0, keepdim=True)
            far_depths = torch.quantile(gt_depth, 0.98, dim=0, keepdim=True)
            loss_depth_in_range = 0
            if (pred_depth < near_depths).any():
                loss_depth_in_range += (near_depths - pred_depth)[
                    pred_depth < near_depths
                ].mean()
            if (pred_depth > far_depths).any():
                loss_depth_in_range += (pred_depth - far_depths)[
                    pred_depth > far_depths
                ].mean()

            loss += loss_depth_in_range * w_smooth_func(i, 0.05, 0.5, 400)

        motion_coef_sparse_loss = 1 - (coefs**2).sum(dim=-1).mean()
        loss += motion_coef_sparse_loss * 0.01

        # motion basis should be smooth.
        w_smooth = w_smooth_func(i, 0.01, 0.1, 400)
        small_acc_loss = compute_se3_smoothness_loss(
            init_rots, init_ts
        )
        loss += small_acc_loss * w_smooth

        small_acc_loss_tracks = compute_accel_loss(positions)
        loss += small_acc_loss_tracks * w_smooth * 0.5

        transfms_nbs = compute_transforms(ts_neighbors, coefs, init_rots, init_ts)
        means_nbs = torch.einsum(
            "pnij,pj->pni", transfms_nbs, F.pad(fg.params["means"], (0, 1), value=1.0)
        )  # (G, 3n, 3)
        means_nbs = means_nbs.reshape(means_nbs.shape[0], 3, -1, 3)  # [G, 3, n, 3]
        z_accel_loss = compute_z_acc_loss(means_nbs, w2cs)
        loss += z_accel_loss * 0.1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        pbar.set_description(
            f"{loss.item():.3f} "
            f"{track_3d_loss.item():.3f} "
            f"{motion_coef_sparse_loss.item():.3f} "
            f"{small_acc_loss.item():.3f} "
            f"{small_acc_loss_tracks.item():.3f} "
            f"{z_accel_loss.item():.3f} "
        )


def random_quats(N: int) -> torch.Tensor:
    u = torch.rand(N, 1)
    v = torch.rand(N, 1)
    w = torch.rand(N, 1)
    quats = torch.cat(
        [
            torch.sqrt(1.0 - u) * torch.sin(2.0 * np.pi * v),
            torch.sqrt(1.0 - u) * torch.cos(2.0 * np.pi * v),
            torch.sqrt(u) * torch.sin(2.0 * np.pi * w),
            torch.sqrt(u) * torch.cos(2.0 * np.pi * w),
        ],
        -1,
    )
    return quats


def sample_initial_bases_centers(
    mode: str, cano_t: int, tracks_3d: TrackObservations, num_bases: int
):
    """
    :param mode: "farthest" | "hdbscan" | "kmeans"
    :param tracks_3d: [G, T, 3]
    :param cano_t: canonical index
    :param num_bases: number of SE3 bases
    """
    assert mode in ["farthest", "hdbscan", "kmeans"]
    means_canonical = tracks_3d.xyz[:, cano_t].clone()

    # linearly interpolate missing 3d points
    xyz = cp.asarray(tracks_3d.xyz)  # [gaussian, T, 3]
    # print(f"{xyz.shape=}")
    visibles = cp.asarray(tracks_3d.visibles)

    num_tracks = xyz.shape[0]
    xyz_interp = batched_interp_masked(xyz, visibles)

    velocities = xyz_interp[:, 1:] - xyz_interp[:, :-1]
    vel_dirs = (
        velocities / (cp.linalg.norm(velocities, axis=-1, keepdims=True) + 1e-5)
    ).reshape(
        (num_tracks, -1)
    )  # [gaussian, 3 * T]

    # [num_bases, num_gaussians]
    if mode == "kmeans":
        model = KMeans(n_clusters=num_bases)
    else:
        model = HDBSCAN(min_cluster_size=20, max_cluster_size=num_tracks // 4)
    model.fit(vel_dirs)
    labels = model.labels_
    num_bases = labels.max().item() + 1
    sampled_centers = torch.stack(
        [
            means_canonical[torch.tensor(labels == i)].median(dim=0).values
            for i in range(num_bases)
        ]
    )[
        None
    ]  # the cluster center at the canonical timestep
    return sampled_centers, num_bases, torch.tensor(labels)


def interp_masked(vals: cp.ndarray, mask: cp.ndarray, pad: int = 1) -> cp.ndarray:
    """
    hacky way to interpolate batched with cupy
    by concatenating the batches and pad with dummy values
    :param vals: [B, M, *]
    :param mask: [B, M]
    """
    assert mask.ndim == 2
    assert vals.shape[:2] == mask.shape

    B, M = mask.shape

    # get the first and last valid values for each track
    sh = vals.shape[2:]
    vals = vals.reshape((B, M, -1))
    D = vals.shape[-1]
    first_val_idcs = cp.argmax(mask, axis=-1)
    last_val_idcs = M - 1 - cp.argmax(cp.flip(mask, axis=-1), axis=-1)
    bidcs = cp.arange(B)

    v0 = vals[bidcs, first_val_idcs][:, None]
    v1 = vals[bidcs, last_val_idcs][:, None]
    m0 = mask[bidcs, first_val_idcs][:, None]
    m1 = mask[bidcs, last_val_idcs][:, None]
    if pad > 1:
        v0 = cp.tile(v0, [1, pad, 1])
        v1 = cp.tile(v1, [1, pad, 1])
        m0 = cp.tile(m0, [1, pad])
        m1 = cp.tile(m1, [1, pad])

    vals_pad = cp.concatenate([v0, vals, v1], axis=1)
    mask_pad = cp.concatenate([m0, mask, m1], axis=1)

    M_pad = vals_pad.shape[1]
    vals_flat = vals_pad.reshape((B * M_pad, -1))
    mask_flat = mask_pad.reshape((B * M_pad,))
    idcs = cp.where(mask_flat)[0]

    cx = cp.arange(B * M_pad)
    out = cp.zeros((B * M_pad, D), dtype=vals_flat.dtype)
    for d in range(D):
        out[:, d] = cp.interp(cx, idcs, vals_flat[idcs, d])

    out = out.reshape((B, M_pad, *sh))[:, pad:-pad]
    return out


def batched_interp_masked(
    vals: cp.ndarray, mask: cp.ndarray, batch_num: int = 4096, batch_time: int = 64
):
    assert mask.ndim == 2
    B, M = mask.shape
    out = cp.zeros_like(vals)
    for b in tqdm(range(0, B, batch_num), leave=False):
        for m in tqdm(range(0, M, batch_time), leave=False):
            x = interp_masked(
                vals[b : b + batch_num, m : m + batch_time],
                mask[b : b + batch_num, m : m + batch_time],
            )  # (batch_num, batch_time, *)
            out[b : b + batch_num, m : m + batch_time] = x
    return out
