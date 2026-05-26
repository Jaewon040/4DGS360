
from typing import Tuple, Literal
import torch
from pathlib import Path
from third_party.cotracker.model_utils import get_points_on_a_grid
import models
import av
import cv2
import numpy as np
from einops import repeat, rearrange

def get_foreground_queries(
    num_points: int,
    fg_mask: np.ndarray,
    depths: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    frame_idx: int = 0,
    sampling_strategy: Literal["uniform", "random"] = "uniform"
) -> torch.Tensor:
    """
    Sample query points from foreground mask region

    Args:
        num_points: Number of points to sample (e.g., 1024)
        fg_mask: (H, W) binary mask where 255 or >128 = foreground
        depths: (T, H, W) or (B, T, H, W) depth maps
        intrinsics: (T, 3, 3) or (B, T, 3, 3) intrinsic matrices
        extrinsics: (T, 4, 4) or (B, T, 4, 4) extrinsic matrices
        frame_idx: Query frame index
        sampling_strategy: "uniform" for grid-based or "random" for random sampling

    Returns:
        queries: (1, P, 4) tensor - (frame_idx, x, y, z) where P <= num_points
    """
    if len(depths.shape) == 3:
        return get_foreground_queries(
            num_points=num_points,
            fg_mask=fg_mask,
            depths=depths.unsqueeze(0),
            intrinsics=intrinsics.unsqueeze(0),
            extrinsics=extrinsics.unsqueeze(0),
            frame_idx=frame_idx,
            sampling_strategy=sampling_strategy
        ).squeeze(0)

    H, W = fg_mask.shape
    device = depths.device

    # Get foreground pixel coordinates
    fg_binary = (fg_mask > 128).astype(np.uint8)
    fg_coords = np.argwhere(fg_binary > 0)  # (N, 2) - (y, x) format

    if len(fg_coords) == 0:
        # No foreground pixels, fall back to center point
        center_y, center_x = H // 2, W // 2
        fg_coords = np.array([[center_y, center_x]])

    # Sample num_points from foreground
    if sampling_strategy == "uniform":
        # Grid-based sampling within foreground bounding box
        y_min, x_min = fg_coords.min(axis=0)
        y_max, x_max = fg_coords.max(axis=0)

        # Create grid within FG bounding box
        grid_size = int(np.ceil(np.sqrt(num_points)))
        y_coords = np.linspace(y_min, y_max, grid_size).astype(np.int32)
        x_coords = np.linspace(x_min, x_max, grid_size).astype(np.int32)
        yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
        grid_coords = np.stack([yy.ravel(), xx.ravel()], axis=-1)  # (grid_size^2, 2)

        # Filter to keep only FG points
        valid_mask = fg_binary[grid_coords[:, 0], grid_coords[:, 1]] > 0
        sampled_coords = grid_coords[valid_mask]

        # If not enough points, add random FG points
        if len(sampled_coords) < num_points:
            remaining = num_points - len(sampled_coords)
            random_indices = np.random.choice(len(fg_coords), size=min(remaining, len(fg_coords)), replace=False)
            sampled_coords = np.concatenate([sampled_coords, fg_coords[random_indices]], axis=0)
        elif len(sampled_coords) > num_points:
            # Subsample uniformly
            indices = np.linspace(0, len(sampled_coords) - 1, num_points).astype(np.int32)
            sampled_coords = sampled_coords[indices]

    elif sampling_strategy == "random":
        # Random sampling
        num_sample = min(num_points, len(fg_coords))
        indices = np.random.choice(len(fg_coords), size=num_sample, replace=False)
        sampled_coords = fg_coords[indices]

    else:
        raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")

    # Convert to (x, y) format for consistency with get_points_on_a_grid
    xy = torch.from_numpy(sampled_coords[:, [1, 0]]).float().to(device)  # (P, 2) - (x, y)
    xy = xy.unsqueeze(0)  # (1, P, 2)

    # Get depth values
    ji = torch.round(xy).to(torch.int32)
    d = depths[:, frame_idx][torch.arange(depths.shape[0])[:, None], ji[..., 1], ji[..., 0]]

    # Filter points with valid depth
    assert d.shape[0] == 1, "batch size must be 1"
    mask = d[0] > 0
    d = d[:, mask]
    xy = xy[:, mask]
    ji = ji[:, mask]

    if d.shape[1] == 0:
        # No valid depth points, return empty
        return torch.zeros(1, 0, 4, device=device)

    # Unproject to 3D
    inv_intrinsics0 = torch.linalg.inv(intrinsics[0, frame_idx])
    inv_extrinsics0 = torch.linalg.inv(extrinsics[0, frame_idx])

    xy_homo = torch.cat([xy, torch.ones_like(xy[..., :1])], dim=-1)
    xy_homo = torch.einsum('ij,bnj->bni', inv_intrinsics0, xy_homo)
    local_coords = xy_homo * d[..., None]
    local_coords_homo = torch.cat([local_coords, torch.ones_like(local_coords[..., :1])], dim=-1)
    world_coords = torch.einsum('ij,bnj->bni', inv_extrinsics0, local_coords_homo)
    world_coords = world_coords[..., :3]

    queries = torch.cat([torch.full_like(xy[:, :, :1], frame_idx), world_coords], dim=-1).to(device)
    return queries

def get_grid_queries(grid_size: int, depths: torch.Tensor, intrinsics: torch.Tensor, extrinsics: torch.Tensor, frame_idx: int = 0):
    if len (depths.shape) == 3:
        return get_grid_queries(
            grid_size=grid_size,
            depths=depths.unsqueeze(0),
            intrinsics=intrinsics.unsqueeze(0),
            extrinsics=extrinsics.unsqueeze(0),
            frame_idx=frame_idx
        ).squeeze(0)

    image_size = depths.shape[-2:]
    xy = get_points_on_a_grid(grid_size, image_size).to(intrinsics.device) # type: ignore
    ji = torch.round(xy).to(torch.int32)
    d = depths[:, frame_idx][torch.arange(depths.shape[0])[:, None], ji[..., 1], ji[..., 0]]

    assert d.shape[0] == 1, "batch size must be 1"
    mask = d[0] > 0
    d = d[:, mask]
    xy = xy[:, mask]
    ji = ji[:, mask]

    inv_intrinsics0 = torch.linalg.inv(intrinsics[0, frame_idx])
    inv_extrinsics0 = torch.linalg.inv(extrinsics[0, frame_idx])

    xy_homo = torch.cat([xy, torch.ones_like(xy[..., :1])], dim=-1)
    xy_homo = torch.einsum('ij,bnj->bni', inv_intrinsics0, xy_homo)
    local_coords = xy_homo * d[..., None]
    local_coords_homo = torch.cat([local_coords, torch.ones_like(local_coords[..., :1])], dim=-1)
    world_coords = torch.einsum('ij,bnj->bni', inv_extrinsics0, local_coords_homo)
    world_coords = world_coords[..., :3]

    queries = torch.cat([torch.full_like(xy[:, :, :1], frame_idx), world_coords], dim=-1).to(depths.device)  # type: ignore
    return queries

@torch.inference_mode()
def _inference_with_grid(
    *,
    model: torch.nn.Module,
    video: torch.Tensor,
    depths: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    query_point: torch.Tensor,
    num_iters: int = 6,
    grid_size: int = 8,
    grid_query_frame: int = 0,
    guidance_2d = None,
    track_vis_threshold: float = 0.5,
    masks: torch.Tensor = None,
    **kwargs,
):
    if grid_size != 0:
        additional_queries = get_grid_queries(grid_size, depths=depths, intrinsics=intrinsics, extrinsics=extrinsics, frame_idx=grid_query_frame)
        query_point = torch.cat([query_point, additional_queries], dim=1)
        N_supports = additional_queries.shape[1]
    else:
        N_supports = 0

    # If guidance_2d is provided, extend it to match the query_point size (including support grid)
    if guidance_2d is not None and grid_size != 0:
        # Pad guidance with zeros for support points
        B, T, P, _ = guidance_2d['coords_3d'].shape
        device = guidance_2d['coords_3d'].device

        # Create zero guidance for support points
        support_coords = torch.zeros(B, T, N_supports, 3, device=device)
        support_visibility = torch.zeros(B, T, N_supports, device=device)

        # Concatenate
        guidance_2d = {
            'coords_3d': torch.cat([guidance_2d['coords_3d'], support_coords], dim=2),
            'visibility': torch.cat([guidance_2d['visibility'], support_visibility], dim=2),
        }

    preds, train_data_list = model(
        rgb_obs=video,
        depth_obs=depths,
        num_iters=num_iters,
        query_point=query_point,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        mode="inference",
        guidance_2d=guidance_2d,
        track_vis_threshold=track_vis_threshold,
        masks=masks,
        **kwargs
    )
    N_total = preds.coords.shape[2]
    preds = preds.query_slice(slice(0, N_total - N_supports))
    return preds, train_data_list

def load_model(checkpoint_path: str):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    model, cfg = models.from_pretrained(checkpoint_path)
    if hasattr(model, "eval_mode"):
        model.set_eval_mode("raw")
    model.eval()

    return model

def read_video(video_path: str) -> np.ndarray:
    container = av.open(video_path)
    frames = []
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format="rgb24"))
    container.close()
    return np.stack(frames)

def resize_depth_bilinear(depth: np.ndarray, new_shape: Tuple[int, int]) -> np.ndarray:
    is_valid = (depth > 0).astype(np.float32)
    depth_resized = cv2.resize(depth, new_shape, interpolation=cv2.INTER_LINEAR)
    is_valid_resized = cv2.resize(is_valid, new_shape, interpolation=cv2.INTER_LINEAR)
    depth_resized = depth_resized / (is_valid_resized + 1e-6)
    depth_resized[is_valid_resized <= 1e-6] = 0.0
    return depth_resized

@torch.no_grad()
def inference(
    *,
    model: torch.nn.Module,
    video: torch.Tensor,
    depths: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    query_point: torch.Tensor,
    num_iters: int = 6,
    grid_size: int = 8,
    grid_query_frame: int = 0,
    bidrectional: bool = True,
    vis_threshold = 0.5,
    guidance_2d = None,
    track_vis_threshold: float = 0.5,
    masks: torch.Tensor = None,
    synthetic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _DEPTH_ROI_LOWER = 1e-7
    _depths = depths.clone()
    # Synthetic depth pipelines (e.g. 1/(disp+1e-10)) leave tiny sentinel values
    # in background pixels. ">0" keeps those sentinels and collapses q25==q75,
    # producing depth_roi=[1e-7, 1e-10] (lower>upper) which masks every pixel to
    # NaN downstream. With --synthetic, filter at the same lower bound used by
    # depth_roi so the IQR is computed only over real depth values.
    if synthetic:
        _depths = _depths[_depths > _DEPTH_ROI_LOWER].reshape(-1)
        if _depths.numel() < 4:
            raise RuntimeError(
                f"--synthetic: only {_depths.numel()} pixels have depth > {_DEPTH_ROI_LOWER}; "
                f"cannot compute depth_roi. Check that input depths are in metric units."
            )
    else:
        _depths = _depths[_depths > 0].reshape(-1)
    q25 = torch.kthvalue(_depths, max(1, int(0.25 * len(_depths)))).values
    q75 = torch.kthvalue(_depths, max(1, int(0.75 * len(_depths)))).values
    iqr = q75 - q25
    _depth_roi = torch.tensor(
        [_DEPTH_ROI_LOWER, (q75 + 1.5 * iqr).item()],
        dtype=torch.float32,
        device=video.device
    )

    T, C, H, W = video.shape
    assert depths.shape == (T, H, W)
    N = query_point.shape[0]

    model.set_image_size((H, W))

    preds, _ = _inference_with_grid(
        model=model,
        video=video[None],
        depths=depths[None],
        intrinsics=intrinsics[None],
        extrinsics=extrinsics[None],
        query_point=query_point[None],
        num_iters=num_iters,
        depth_roi=_depth_roi,
        grid_size=grid_size,
        grid_query_frame=grid_query_frame,
        guidance_2d=guidance_2d,
        track_vis_threshold=track_vis_threshold,
        masks=masks[None] if masks is not None else None
    )

    if bidrectional and not model.bidirectional and (query_point[..., 0] > 0).any():
        # Flip guidance_2d for backward pass if provided
        guidance_2d_backward = None
        if guidance_2d is not None:
            guidance_2d_backward = {
                'coords_3d': guidance_2d['coords_3d'].flip(dims=(1,)),
                'visibility': guidance_2d['visibility'].flip(dims=(1,)),
            }

        preds_backward, _ = _inference_with_grid(
            model=model,
            video=video[None].flip(dims=(1,)),
            depths=depths[None].flip(dims=(1,)),
            intrinsics=intrinsics[None].flip(dims=(1,)),
            extrinsics=extrinsics[None].flip(dims=(1,)),
            query_point=torch.cat([T - 1 - query_point[..., :1], query_point[..., 1:]], dim=-1)[None],
            num_iters=num_iters,
            depth_roi=_depth_roi,
            grid_size=grid_size,
            grid_query_frame=T - 1 - grid_query_frame,
            guidance_2d=guidance_2d_backward,
            track_vis_threshold=track_vis_threshold,
            masks=masks[None].flip(dims=(1,)) if masks is not None else None
        )
        preds.coords = torch.where(
            repeat(torch.arange(T, device=video.device), 't -> b t n 3', b=1, n=N) < repeat(query_point[..., 0][None], 'b n -> b t n 3', t=T, n=N),
            preds_backward.coords.flip(dims=(1,)),
            preds.coords
        )
        preds.visibs = torch.where(
            repeat(torch.arange(T, device=video.device), 't -> b t n', b=1, n=N) < repeat(query_point[..., 0][None], 'b n -> b t n', t=T, n=N),
            preds_backward.visibs.flip(dims=(1,)),
            preds.visibs
        )

    coords, visib_logits = preds.coords, preds.visibs
    visibs = torch.sigmoid(visib_logits) >= vis_threshold
    return coords.squeeze(), visibs.squeeze()