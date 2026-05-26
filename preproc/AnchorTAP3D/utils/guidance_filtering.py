import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import logging
import cv2

logger = logging.getLogger(__name__)


def filter_guidance_by_mask_and_depth(
    guidance_2d: dict,
    track2d_dir: Path,
    query_frame_name: str,
    image_names: list,
    depths: torch.Tensor,
    masks: torch.Tensor = None,
    sampled_indices: np.ndarray = None,
    depth_jump_threshold: float = 2.0,
    mask_erosion_iterations: int = 0
) -> dict:
    """
    Filter 2D track guidance by mask and depth consistency

    This function filters out guidance points that:
    1. Fall outside the mask in each frame (if masks provided)
       - Masks are eroded by mask_erosion_iterations to remove uncertain boundary regions
    2. Have sudden depth jumps between consecutive frames (gradient-based filtering)
       - When depth jumps, the point with larger depth is considered background and filtered

    Args:
        guidance_2d: Dict with 'coords_3d' [1, T, P, 3] and 'visibility' [1, T, P]
        track2d_dir: Directory containing 2D track files
        query_frame_name: Query frame name (without extension)
        image_names: List of all frame names (without extension)
        depths: [T, H, W] depth maps
        masks: [T, H, W] mask tensor where >128 = foreground (optional)
        sampled_indices: Optional indices if queries were pruned
        depth_jump_threshold: Maximum allowed depth change between consecutive frames (meters)
        mask_erosion_iterations: Number of erosion iterations to apply to masks (0 = no erosion)

    Returns:
        guidance_2d: Filtered guidance dict (modified in-place and returned)
    """
    device = guidance_2d['visibility'].device
    num_frames = len(image_names)
    num_points = guidance_2d['visibility'].shape[2]

    # Load 2D track coordinates for all frames
    xy_2d_all_frames = []

    for target_frame_name in image_names:
        track_file = track2d_dir / f"{query_frame_name}_{target_frame_name}.npy"

        if not track_file.exists():
            # No track file: assume all points are invalid (outside mask)
            xy_2d_all_frames.append(np.zeros((num_points, 2), dtype=np.float32))
            continue

        # Load 2D track: [P, 4] (x, y, occlusions, expected_dist)
        track_2d = np.load(track_file)

        # Apply sampled indices if provided (for pruned queries)
        if sampled_indices is not None:
            track_2d = track_2d[sampled_indices]

        if track_2d.shape[0] != num_points:
            logger.warning(f"2D track has {track_2d.shape[0]} points, expected {num_points}. Slicing to match.")
            track_2d = track_2d[:num_points]

        xy_2d = track_2d[:, :2]  # [P, 2] - (x, y) pixel coordinates
        xy_2d_all_frames.append(xy_2d)

    # Stack to [T, P, 2]
    xy_2d_all_frames = np.stack(xy_2d_all_frames, axis=0)
    xy_2d_tensor = torch.from_numpy(xy_2d_all_frames).float().to(device)

    # Normalize coordinates to [-1, 1] for grid_sample
    # Get H, W from either depths or masks (at least one must be provided)
    if depths is not None:
        H, W = depths.shape[-2:]
    elif masks is not None:
        H, W = masks.shape[-2:]
    else:
        raise ValueError("Either depths or masks must be provided for filtering")

    xy_norm = xy_2d_tensor.clone()
    xy_norm[:, :, 0] = 2.0 * xy_2d_tensor[:, :, 0] / (W - 1) - 1.0  # x
    xy_norm[:, :, 1] = 2.0 * xy_2d_tensor[:, :, 1] / (H - 1) - 1.0  # y

    # Save original visibility before filtering
    original_visibility = guidance_2d['visibility'][0].clone()

    # === Step 1: Mask filtering (if provided) ===
    inside_mask = None
    if masks is not None:
        # Check if masks need to be subset to match image_names
        if masks.shape[0] != num_frames:
            logger.warning(f"Masks shape {masks.shape[0]} != num_frames {num_frames}. Attempting to subset masks based on track file existence.")
            # This can happen when masks contain all frames but image_names is a subset (e.g., train split)
            # We cannot reliably map without frame indices, so we'll skip mask filtering
            logger.warning(f"Skipping mask filtering due to frame count mismatch")
            masks = None

    if masks is not None:
        # Apply erosion to masks if requested
        if mask_erosion_iterations > 0:
            logger.info(f"Applying {mask_erosion_iterations} erosion iterations to masks")
            kernel = np.ones((3, 3), np.uint8)
            eroded_masks = []
            for t in range(masks.shape[0]):
                mask_np = masks[t].cpu().numpy().astype(np.uint8)
                eroded_mask = cv2.erode(mask_np, kernel, iterations=mask_erosion_iterations)
                eroded_masks.append(eroded_mask)
            masks = torch.from_numpy(np.stack(eroded_masks)).to(device)
            logger.info(f"Mask erosion completed")

        # Sample mask values at 2D track locations (vectorized)
        mask_values = F.grid_sample(
            masks.unsqueeze(1).float(),  # [T, 1, H, W]
            xy_norm.unsqueeze(2),         # [T, P, 1, 2]
            mode='nearest',
            padding_mode='zeros',
            align_corners=True
        ).squeeze(1).squeeze(2)  # [T, P]

        # Check if points are inside mask (foreground)
        inside_mask = mask_values > 128  # [T, P]

        # Update visibility: points outside mask get visibility=0
        guidance_2d['visibility'][0] = guidance_2d['visibility'][0] * inside_mask.float()

        # Log mask filtering statistics
        high_vis_filtered_mask = ((original_visibility > 0.5) & (~inside_mask)).sum().item()
        logger.info(f"Mask filtering: {high_vis_filtered_mask} high-confidence points filtered")

    # === Step 2: Depth gradient filtering ===
    if depths is not None:
        # Sample depth values at 2D track locations
        depth_values = F.grid_sample(
            depths.unsqueeze(1).float(),  # [T, 1, H, W]
            xy_norm.unsqueeze(2),          # [T, P, 1, 2]
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        ).squeeze(1).squeeze(2)  # [T, P]

        # Compute depth change between consecutive frames
        depth_diff = torch.abs(depth_values[1:] - depth_values[:-1])  # [T-1, P]

        # Detect sudden depth jumps
        sudden_jump = depth_diff > depth_jump_threshold  # [T-1, P]

        # When there's a jump, filter the frame with LARGER depth (background)
        depth_larger_next = depth_values[1:] > depth_values[:-1]  # [T-1, P] - True if frame t+1 has larger depth

        # Filter frame t+1 if it has larger depth and there's a jump
        filter_next_frame = sudden_jump & depth_larger_next  # [T-1, P]
        # Filter frame t if it has larger depth and there's a jump
        filter_prev_frame = sudden_jump & (~depth_larger_next)  # [T-1, P]

        # Apply filtering: set visibility=0 for filtered frames
        # For frames 1~T (skip frame 0 as it has no previous frame)
        for t in range(1, num_frames):
            if t < num_frames:  # Filter based on t-1 -> t jump
                guidance_2d['visibility'][0, t] = guidance_2d['visibility'][0, t] * (~filter_next_frame[t-1]).float()
            if t > 0:  # Filter based on t -> t+1 jump
                guidance_2d['visibility'][0, t-1] = guidance_2d['visibility'][0, t-1] * (~filter_prev_frame[t-1]).float()

        # Log depth filtering statistics
        high_vis_filtered_depth = (filter_next_frame.sum() + filter_prev_frame.sum()).item()
        logger.info(f"Depth gradient filtering: {high_vis_filtered_depth} points filtered due to depth jumps")

    # Log final statistics
    high_vis_final = (guidance_2d['visibility'][0] > 0.5).sum().item()
    logger.info(f"Total high-confidence points after filtering: {high_vis_final}")

    return guidance_2d
