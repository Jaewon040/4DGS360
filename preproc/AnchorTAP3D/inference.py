# Copyright (c) TAPIP3D team(https://tapip3d.github.io/)

from concurrent.futures import ThreadPoolExecutor
import shlex
import tap
import torch
from typing import Optional, Tuple
from pathlib import Path
from datetime import datetime
from einops import repeat
from utils.common_utils import setup_logger
import logging
from annotation.megasam import MegaSAMAnnotator
import numpy as np
import cv2
from datasets.data_ops import _filter_one_depth

from utils.inference_utils import load_model, read_video, inference, get_grid_queries, get_foreground_queries, resize_depth_bilinear
from utils.track_filtering import filter_outlier_tracks_by_velocity, filter_outlier_tracks_by_velocity_adaptive
from utils.guidance_filtering import filter_guidance_by_mask_and_depth
from datasets.utils.colmap import get_colmap_camera_params
import json

logger = logging.getLogger(__name__)

DEFAULT_QUERY_GRID_SIZE = 32

def sigmoid(x):
    """Sigmoid function for numpy arrays"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def unproject_2d_points_to_3d(
    xy_2d: np.ndarray,
    depth_map: torch.Tensor,
    intrinsic: torch.Tensor,
    extrinsic: torch.Tensor,
    frame_idx: int,
    device: str = "cuda"
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Unproject 2D points to 3D world coordinates using depth map

    Args:
        xy_2d: (P, 2) array of 2D points (x, y)
        depth_map: (H, W) depth map
        intrinsic: (3, 3) intrinsic matrix
        extrinsic: (4, 4) extrinsic matrix
        frame_idx: Frame index for query point format
        device: Device to place tensors

    Returns:
        query_points: (P, 4) tensor (frame_idx, x, y, z) in world coordinates (invalid points set to 0)
        valid_mask: (P,) boolean array indicating which points have valid depth
    """
    num_points = xy_2d.shape[0]
    H, W = depth_map.shape

    # Sample depth at 2D locations
    xy_int = np.round(xy_2d).astype(np.int32)
    xy_int[:, 0] = np.clip(xy_int[:, 0], 0, W - 1)
    xy_int[:, 1] = np.clip(xy_int[:, 1], 0, H - 1)

    sampled_depths = depth_map[xy_int[:, 1], xy_int[:, 0]].cpu().numpy()  # [P]

    # Create valid mask for points with valid depth
    valid_mask = sampled_depths > 0

    if valid_mask.sum() == 0:
        logger.warning(f"No valid depth points found for 2D track query")
        return torch.zeros(num_points, 4, device=device), valid_mask

    # Initialize output with zeros
    query_points = np.zeros((num_points, 4), dtype=np.float32)

    # Unproject only valid points
    xy_2d_valid = xy_2d[valid_mask]
    sampled_depths_valid = sampled_depths[valid_mask]

    # Unproject to camera coordinates
    xy_homo = np.concatenate([xy_2d_valid, np.ones((valid_mask.sum(), 1))], axis=-1)  # [P_valid, 3]
    K_inv = np.linalg.inv(intrinsic.cpu().numpy())
    camera_coords = (K_inv @ xy_homo.T).T * sampled_depths_valid[:, None]  # [P_valid, 3]

    # Transform to world coordinates
    camera_coords_homo = np.concatenate([camera_coords, np.ones((valid_mask.sum(), 1))], axis=-1)  # [P_valid, 4]
    inv_extrinsic = np.linalg.inv(extrinsic.cpu().numpy())
    world_coords = (inv_extrinsic @ camera_coords_homo.T).T[:, :3]  # [P_valid, 3]

    # Fill in valid points
    query_points[valid_mask, 0] = frame_idx
    query_points[valid_mask, 1:] = world_coords.astype(np.float32)

    return torch.from_numpy(query_points).to(device), valid_mask

def load_and_process_2dtrack(
    track2d_dir: Path,
    query_frame_name: str,
    image_names: list,
    num_points: int,
    depths: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    device: str = "cuda",
    sampled_indices: np.ndarray = None,
    masks: torch.Tensor = None,
    filter_2dtrack_with_mask: bool = False,
    filter_2dtrack_with_depth: bool = False,
    depth_jump_threshold: float = 2.0,
    mask_erosion_iterations: int = 0,
    track_vis_threshold: float = 0.5
) -> dict:
    """
    Load 2D track data and convert to 3D guidance

    Args:
        track2d_dir: Directory containing 2D track files
        query_frame_name: Query frame image name (without extension)
        image_names: List of all image names (without extension)
        num_points: Number of query points
        depths: (T, H, W) depth maps
        intrinsics: (T, 3, 3) intrinsic matrices
        extrinsics: (T, 4, 4) extrinsic matrices
        device: Device to place tensors
        sampled_indices: Optional indices for pruned queries
        masks: (T, H, W) mask tensor for filtering (optional)

    Returns:
        guidance_2d: Dict with 'coords_3d' (1, T, P, 3) and 'visibility' (1, T, P)
    """
    coords_3d_list = []
    visibility_list = []

    num_frames = len(image_names)

    for target_t in range(num_frames):
        target_frame_name = image_names[target_t]
        track_file = track2d_dir / f"{query_frame_name}_{target_frame_name}.npy"

        if not track_file.exists():
            logger.warning(f"2D track file not found: {track_file}, using zeros")
            coords_3d_list.append(np.zeros((num_points, 3), dtype=np.float32))
            visibility_list.append(np.zeros(num_points, dtype=np.float32))
            continue

        # Load 2D track: [P, 4] (x, y, occlusions, expected_dist)
        track_2d = np.load(track_file)

        # Apply sampled indices if provided (for pruned queries)
        if sampled_indices is not None:
            track_2d = track_2d[sampled_indices]

        if track_2d.shape[0] != num_points:
            logger.warning(f"2D track has {track_2d.shape[0]} points, expected {num_points}. Slicing to match.")
            track_2d = track_2d[:num_points]

        xy_2d = track_2d[:, :2]  # [P, 2]
        occlusions = track_2d[:, 2]  # [P]
        expected_dist = track_2d[:, 3]  # [P]

        # Calculate visibility: (1 - sigmoid(occlusions)) * (1 - sigmoid(expected_dist))
        visibility = (1 - sigmoid(occlusions)) * (1 - sigmoid(expected_dist))  # [P]

        # Unproject 2D to 3D using depth
        depth_map = depths[target_t].cpu().numpy()  # (H, W)
        intrinsic = intrinsics[target_t].cpu().numpy()  # (3, 3)
        extrinsic = extrinsics[target_t].cpu().numpy()  # (4, 4)

        H, W = depth_map.shape

        # Sample depth at 2D locations
        xy_int = np.round(xy_2d).astype(np.int32)
        xy_int[:, 0] = np.clip(xy_int[:, 0], 0, W - 1)
        xy_int[:, 1] = np.clip(xy_int[:, 1], 0, H - 1)

        sampled_depths = depth_map[xy_int[:, 1], xy_int[:, 0]]  # [P]

        # Create valid mask and set visibility to 0 for invalid depth points
        valid_depth_mask = sampled_depths > 0
        visibility = visibility * valid_depth_mask  # Set visibility to 0 for invalid depth points

        # Unproject to camera coordinates (invalid points will have coords at origin)
        xy_homo = np.concatenate([xy_2d, np.ones((num_points, 1))], axis=-1)  # [P, 3]
        K_inv = np.linalg.inv(intrinsic)
        camera_coords = (K_inv @ xy_homo.T).T * sampled_depths[:, None]  # [P, 3]

        # Transform to world coordinates
        camera_coords_homo = np.concatenate([camera_coords, np.ones((num_points, 1))], axis=-1)  # [P, 4]
        inv_extrinsic = np.linalg.inv(extrinsic)
        world_coords = (inv_extrinsic @ camera_coords_homo.T).T[:, :3]  # [P, 3]

        coords_3d_list.append(world_coords.astype(np.float32))
        visibility_list.append(visibility.astype(np.float32))

    # Stack to [T, P, 3] and [T, P]
    coords_3d = np.stack(coords_3d_list, axis=0)  # [T, P, 3]
    visibility = np.stack(visibility_list, axis=0)  # [T, P]

    # Convert to torch and add batch dimension
    guidance_2d = {
        'coords_3d': torch.from_numpy(coords_3d).unsqueeze(0).to(device),  # [1, T, P, 3]
        'visibility': torch.from_numpy(visibility).unsqueeze(0).to(device),  # [1, T, P]
    }

    logger.info(f"Loaded 2D track guidance for query frame '{query_frame_name}'")
    logger.info(f"Guidance shape: coords_3d={guidance_2d['coords_3d'].shape}, visibility={guidance_2d['visibility'].shape}")
    logger.info(f"Visibility stats: mean={guidance_2d['visibility'].mean().item():.3f}, max={guidance_2d['visibility'].max().item():.3f}, min={guidance_2d['visibility'].min().item():.3f}")
    logger.info(f"Points with visibility > {track_vis_threshold}: {(guidance_2d['visibility'] > track_vis_threshold).sum().item()} / {guidance_2d['visibility'].numel()}")

    # Apply mask and depth filtering if enabled
    filter_masks = masks if filter_2dtrack_with_mask else None
    filter_depths = depths if filter_2dtrack_with_depth else None

    if filter_masks is not None or filter_depths is not None:
        logger.info(f"Applying guidance filtering (mask={filter_2dtrack_with_mask}, depth={filter_2dtrack_with_depth})")
        guidance_2d = filter_guidance_by_mask_and_depth(
            guidance_2d=guidance_2d,
            track2d_dir=track2d_dir,
            query_frame_name=query_frame_name,
            image_names=image_names,
            depths=filter_depths,
            masks=filter_masks,
            sampled_indices=sampled_indices,
            depth_jump_threshold=depth_jump_threshold,
            mask_erosion_iterations=mask_erosion_iterations
        )

    return guidance_2d

class Arguments(tap.Tap):
    work_dir: str
    device: str = "cuda"
    num_iters: int = 6
    support_grid_size: int = 16
    use_support_grid: bool = False  # If set, keep support_grid_size as-is in prepare_inputs (otherwise it is forced to 0 when query_point is None)
    num_threads: int = 8
    img_res: int = 1  # Image resolution downscale factor (1 = original, 2 = 1/2, 4 = 1/4, etc.)
    resolution_factor: int = 2  # Model resolution scale factor (only used with use_model_resolution)
    use_model_resolution: bool = False  # Use model resolution instead of image resolution
    vis_threshold: Optional[float] = 0.9
    checkpoint: Optional[str] = None
    depth_model: str = "moge"
    query_frame: int = 0  # Frame index to use for query points
    scale_factor: str = "1x"  # Scale factor for camera/depth files (1x or 2x)
    depth_type: str = "vda_lidar"  # Depth type for DyCheck format (vda_lidar, depth_anything_lidar, etc.)
    iphone: bool = False  # Use DyCheck dataset format (rgb, depth, camera, masks in scale_factor subdirs)
    colmap: bool = False  # Use COLMAP sparse reconstruction for camera parameters
    depth: Optional[str] = None  # Path to depth directory containing .npy files
    camera_parameter: Optional[str] = None  # Path to camera parameter JSON file
    fg_only: bool = False  # Sample query points only from foreground mask region
    use_2dtrack: bool = False  # Use 2D track guidance for visible points
    fg_only_2dquery: bool = False  # Use 2D track query points but NO 2D track guidance (only for query initialization)
    track_vis_threshold: float = 0.5  # Visibility threshold for 2D track guidance (points with visibility > threshold use 2D track)
    prune_query: int = 0  # Randomly sample up to N query points with valid depth from 2D tracks (0 = no pruning)
    use_mask: bool = False  # Use masks to filter KNN candidates (only pixels inside mask are used for support/context tokens)
    filter_2dtrack_with_mask: bool = False  # Filter 2D track guidance by checking if points are inside masks
    filter_2dtrack_with_depth: bool = False  # Filter 2D track guidance by checking depth consistency between frames
    depth_jump_threshold: float = 2.0  # Maximum allowed depth change between consecutive frames (meters)
    mask_erosion_iterations: int = 5  # Number of erosion iterations to apply to masks before filtering (0 = no erosion)
    filter_outliers: bool = False  # Filter outlier tracks based on velocity consistency
    velocity_threshold: float = 3.0  # Velocity change threshold multiplier for outlier detection
    use_adaptive_filter: bool = False  # Use adaptive percentile-based filtering instead of median-based
    start_query: int = 0  # Starting frame index for query point inference
    synthetic: bool = False  # Treat depth as synthetic-pipeline output: tiny background sentinels (e.g. 1e-10 from 1/(disp+eps)) are filtered out of the IQR-based depth_roi calculation. Without this flag, sentinel pixels collapse q25==q75 and yield depth_roi=[1e-7, 1e-10] (lower>upper), silently NaN-ing every track and giving all-zero visibility.
    use_megasam: bool = False  # Override all depth/camera loading and use MegaSAM (with --depth_model, e.g. moge) to estimate both depth and camera parameters from RGB only. Image dir/mask filtering from --iphone still applies.
    megasam_resolution: int = 384 * 512  # Internal pixel budget for MegaSAM (DROID-SLAM camera tracking + RAFT flow). Larger values overflow DROID-SLAM int32 buffers; depths are resized back to inference_res afterwards regardless.

def prepare_inputs(img_dir: str, inference_res: Tuple[int, int], support_grid_size: int, num_threads: int = 8, device: str = "cpu", query_frame: int = 0, work_dir: Optional[Path] = None, scale_factor: str = "1x", depth_type: str = "vda_lidar", depth_model: str = "moge", iphone: bool = False, colmap: bool = False, depth_dir: Optional[str] = None, camera_param_path: Optional[str] = None, use_megasam: bool = False, megasam_resolution: Optional[int] = None, use_support_grid: bool = False):
    video, depths, intrinsics, extrinsics, query_point = None, None, None, None, None

    img_dir_path = Path(img_dir)
    if not img_dir_path.is_dir():
        raise ValueError(f"Image directory not found: {img_dir}")

    # Load all images from directory
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    all_image_paths = [p for p in img_dir_path.iterdir() if p.suffix.lower() in image_extensions]

    # Filter for images starting with "0_" if iphone is enabled
    if iphone:
        image_paths = sorted([p for p in all_image_paths if p.stem.startswith("0_")])
        logger.info(f"iPhone mode: filtering for images starting with '0_'")
    else:
        image_paths = sorted(all_image_paths)

    if len(image_paths) == 0:
        raise ValueError(f"No images found in directory: {img_dir}")

    logger.info(f"Loading {len(image_paths)} images from {img_dir}")
    video = np.stack([cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in image_paths])

    # Load depth and camera parameters separately to allow mixing sources
    # Depth priority: --depth > --iphone > MegaSAM
    # Camera priority: --camera_parameter > --colmap > --iphone > MegaSAM

    if use_megasam:
        logger.info("--use_megasam set: skipping all depth/camera file loading; MegaSAM will estimate depth and cameras from RGB.")

    # Load depths
    if use_megasam:
        depths = None
    elif depth_dir is not None:
        logger.info(f"Loading depths from custom path: {depth_dir}")
        depth_dir_path = Path(depth_dir)
        if not depth_dir_path.exists():
            raise ValueError(f"Depth directory not found: {depth_dir}")

        depths_list = []
        for img_path in image_paths:
            img_name = img_path.stem
            depth_path = depth_dir_path / f"{img_name}.npy"
            if not depth_path.exists():
                raise ValueError(f"Depth file not found: {depth_path}")
            depth = np.load(depth_path)
            if depth.ndim == 3 and depth.shape[2] == 1:
                depth = depth[:, :, 0]  # Remove channel dimension
            depths_list.append(depth)

        depths = np.stack(depths_list)
        _original_res = depths.shape[1:3]
        logger.info(f"Loaded {len(depths_list)} depth maps from {depth_dir}")
    elif iphone and work_dir is not None:
        depth_dir_path = work_dir / "flow3d_preprocessed" / f"aligned_{depth_type}" / scale_factor
        logger.info(f'depth_dir = {depth_dir_path}, depth_type = {depth_type}')

        if depth_dir_path.exists():
            logger.info(f"Loading DyCheck depth maps from {scale_factor} folder with depth_type={depth_type}")

            depths_list = []
            # Determine if depth_type uses inverse depth (disparity) or direct depth
            is_inverse_depth = depth_type == "vda_lidar"
            # breakpoint()

            for img_path in image_paths:
                img_name = img_path.stem
                depth_path = depth_dir_path / f"{img_name}.npy"
                if not depth_path.exists():
                    raise ValueError(f"Depth file not found: {depth_path}")

                depth_data = np.load(depth_path)
                if depth_data.ndim == 3 and depth_data.shape[2] == 1:
                    depth_data = depth_data[:, :, 0]  # Remove channel dimension

                if is_inverse_depth:
                    # For vda_lidar: stored as inverse depth, convert to depth
                    depth_data = np.clip(depth_data, 1e-3, None)
                    depth = 1 / depth_data
                    # breakpoint()
                else:
                    # For depth_anything_lidar and others: stored as direct depth
                    depth = depth_data

                depths_list.append(depth.astype(np.float32))

            depths = np.stack(depths_list)

            _original_res = depths.shape[1:3]
            logger.info(f"Loaded {len(depths_list)} depth maps from DyCheck (is_inverse_depth={is_inverse_depth})")
        else:
            logger.warning(f"DyCheck depth directory not found: {depth_dir_path}")
            depths = None
    else:
        depths = None

    # Load camera parameters from custom path if provided
    if use_megasam:
        intrinsics = None
        extrinsics = None
    elif camera_param_path is not None:
        logger.info(f"Loading camera parameters from: {camera_param_path}")
        camera_param_file = Path(camera_param_path)
        if not camera_param_file.exists():
            raise ValueError(f"Camera parameter file not found: {camera_param_path}")

        # Check if it's a directory (DROID reconstruction format) or a file (JSON format)
        if camera_param_file.is_dir():
            # DROID reconstruction format: load {scale_factor}.npy from directory
            droid_recon_file = camera_param_file / f"{scale_factor}/droid_recon.npy"
            if not droid_recon_file.exists():
                raise ValueError(f"DROID reconstruction file not found: {droid_recon_file}")

            logger.info(f"Loading DROID reconstruction from: {droid_recon_file}")
            recon = np.load(droid_recon_file, allow_pickle=True).item()

            # Extract camera parameters
            traj_c2w = recon["traj_c2w"]  # (N, 4, 4) camera to world
            h, w = recon["img_shape"]  # Original image shape

            # Get current video shape for scaling
            H, W = video.shape[1:3]
            sy, sx = H / h, W / w

            # Convert to world to camera (extrinsics)
            traj_w2c = np.linalg.inv(traj_c2w)

            # Build intrinsics matrix
            fx, fy, cx, cy = recon["intrinsics"]  # (4,)
            K = np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]], dtype=np.float32)

            # Get keyframe timestamps
            kf_tstamps = recon["tstamps"].astype(int)

            # Match camera parameters to image frames
            num_frames = len(image_paths)
            intrinsics_list = []
            extrinsics_list = []

            for i in range(num_frames):
                # Find closest keyframe timestamp
                if i < len(kf_tstamps):
                    kf_idx = kf_tstamps[i] if i < len(kf_tstamps) else kf_tstamps[-1]
                    if kf_idx >= len(traj_w2c):
                        kf_idx = len(traj_w2c) - 1
                else:
                    kf_idx = len(traj_w2c) - 1

                intrinsics_list.append(K)
                extrinsics_list.append(traj_w2c[kf_idx])

            intrinsics = np.stack(intrinsics_list)
            extrinsics = np.stack(extrinsics_list)

            logger.info(f"Loaded DROID camera parameters: {len(kf_tstamps)} keyframes for {num_frames} frames")
        else:
            # JSON format
            with open(camera_param_file, 'r') as f:
                camera_params = json.load(f)

            # Build intrinsics matrix from fx, fy, cx, cy format
            intrinsics_dict = camera_params['intrinsics']
            fx = intrinsics_dict['fx']
            fy = intrinsics_dict['fy']
            cx = intrinsics_dict['cx']
            cy = intrinsics_dict['cy']

            K = np.array([
                [fx, 0.0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float32)

            # Apply intrinsics to all frames
            num_frames = len(image_paths)
            intrinsics = np.stack([K] * num_frames)

            # Build extrinsics matrix from w2c (world to camera)
            if 'w2c' in camera_params:
                w2c = np.array(camera_params['w2c'], dtype=np.float32)  # 4x4
                extrinsics = np.stack([w2c] * num_frames)
            else:
                extrinsics = None

            logger.info(f"Loaded camera parameters for {num_frames} frames")
    elif colmap and work_dir is not None:
        colmap_dir = work_dir / "colmap" / "sparse"
        if colmap_dir.exists():
            logger.info(f"Loading camera parameters from COLMAP: {colmap_dir}")

            # Get image file names with original extension
            img_files = [str(p) for p in image_paths]

            # Load COLMAP camera parameters
            K_all, extrinsics_all = get_colmap_camera_params(str(colmap_dir), img_files)

            intrinsics = K_all[:, :3, :3].astype(np.float32)  # Extract 3x3 from 4x4
            extrinsics = extrinsics_all.astype(np.float32)  # Already 4x4

            logger.info(f"Loaded {len(intrinsics)} camera parameters from COLMAP")
        else:
            logger.warning(f"COLMAP directory not found: {colmap_dir}")
            intrinsics = None
            extrinsics = None
    elif iphone and work_dir is not None:
        camera_dir = work_dir / "camera"
        if camera_dir.exists():
            logger.info(f"Loading DyCheck camera parameters")

            intrinsics_list = []
            extrinsics_list = []

            for img_path in image_paths:
                img_name = img_path.stem

                # Load camera parameters
                camera_path = camera_dir / f"{img_name}.json"
                if not camera_path.exists():
                    raise ValueError(f"Camera file not found: {camera_path}")

                with open(camera_path, 'r') as f:
                    camera_params = json.load(f)

                # Build intrinsics matrix
                focal_length = camera_params['focal_length']
                cx, cy = camera_params['principal_point']
                fx = fy = focal_length * camera_params.get('pixel_aspect_ratio', 1.0)

                K = np.array([
                    [fx, camera_params.get('skew', 0.0), cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ], dtype=np.float32)
                intrinsics_list.append(K)

                # Build extrinsics matrix (world to camera transform)
                R = np.array(camera_params['orientation'], dtype=np.float32)  # 3x3
                position = np.array(camera_params['position'], dtype=np.float32)  # Camera position in world
                t = -R @ position  # Translation vector

                extrinsic = np.eye(4, dtype=np.float32)
                extrinsic[:3, :3] = R
                extrinsic[:3, 3] = t
                extrinsics_list.append(extrinsic)

            intrinsics = np.stack(intrinsics_list)
            extrinsics = np.stack(extrinsics_list)

            logger.info(f"Loaded {len(intrinsics_list)} camera parameters from DyCheck")
        else:
            logger.warning(f"DyCheck camera directory not found: {camera_dir}")
            intrinsics = None
            extrinsics = None
    else:
        intrinsics = None
        extrinsics = None

    if depths is None:
        logger.info(f"No depth provided, running MegaSAM to get depths")
        megasam_res = megasam_resolution if megasam_resolution is not None else inference_res[0] * inference_res[1]
        megasam = MegaSAMAnnotator(
            script_path=Path(__file__).parent / "third_party" / "megasam" / "inference.py",
            depth_model=depth_model,
            resolution=megasam_res,
        )
        megasam.to(device)
        depths, intrinsics, extrinsics = megasam.process_video(video, gt_intrinsics=intrinsics, return_raw_depths=True)
        _original_res = video.shape[1:3]
    else:
        _original_res = depths.shape[1:3]

    if intrinsics is None:
        raise ValueError("Intrinsics must be provided if depth is provided")
    if extrinsics is None:
        logger.info(f"No extrinsics provided, using identity matrix for all frames")
        extrinsics = repeat(np.eye(4), "i j -> t i j", t=len(video))
    
    intrinsics[:, 0, :] *= (inference_res[1] - 1) / (_original_res[1] - 1)
    intrinsics[:, 1, :] *= (inference_res[0] - 1) / (_original_res[0] - 1)

    # resize & remove edges
    with ThreadPoolExecutor(num_threads) as executor:
        video_futures = [executor.submit(cv2.resize, rgb, (inference_res[1], inference_res[0]), interpolation=cv2.INTER_LINEAR) for rgb in video]
        depths_futures = [executor.submit(resize_depth_bilinear, depth, (inference_res[1], inference_res[0])) for depth in depths]
        
        video = np.stack([future.result() for future in video_futures])
        depths = np.stack([future.result() for future in depths_futures])

        depths_futures = [executor.submit(_filter_one_depth, depth, 0.08, 15, intrinsic) for depth, intrinsic in zip(depths, intrinsics)]
        depths = np.stack([future.result() for future in depths_futures])

    video = (torch.from_numpy(video).permute(0, 3, 1, 2).float() / 255.0).to(device)
    depths = torch.from_numpy(depths).float().to(device)
    intrinsics = torch.from_numpy(intrinsics).float().to(device)
    extrinsics = torch.from_numpy(extrinsics).float().to(device)
    # breakpoint()
    if query_point is None:
        if not use_support_grid:
            support_grid_size = 0
            logger.info("--use_support_grid not set: forcing support_grid_size=0 (no auxiliary support grid)")
        else:
            logger.info(f"--use_support_grid set: keeping support_grid_size={support_grid_size}")
        query_point = get_grid_queries(grid_size=DEFAULT_QUERY_GRID_SIZE, depths=depths, intrinsics=intrinsics, extrinsics=extrinsics, frame_idx=query_frame)
        logger.info(f"No queries provided, using a grid at frame {query_frame} as queries")
    else:
        query_point = torch.from_numpy(query_point).float().to(device)

    return video, depths, intrinsics, extrinsics, query_point, support_grid_size

if __name__ == "__main__":
    setup_logger()
    args = Arguments().parse_args()

    # Setup paths
    work_dir = Path(args.work_dir)

    # Use DyCheck dataset format if enabled
    if args.iphone:
        img_dir = work_dir / "rgb" / args.scale_factor
        mask_dir = work_dir / "flow3d_preprocessed" / "track_anything" / args.scale_factor
        if not mask_dir.exists():
            mask_dir = work_dir / "masks" / args.scale_factor
        logger.info(f"Using DyCheck format: images from {img_dir}, masks from {mask_dir}")
    else:
        img_dir = work_dir / "images" / args.scale_factor
        if not img_dir.exists():
            img_dir = work_dir / "rgb" / args.scale_factor
        mask_dir = work_dir / "masks" / args.scale_factor

    tapip3d_dir = work_dir / "anchortap3d"

    tracks_dir = tapip3d_dir / "tracks"
    vid_dir = tapip3d_dir / "vid" / f"{args.img_res}x"
    depth_dir = tapip3d_dir / "depth" / f"{args.img_res}x"
    refined_depth_dir = tapip3d_dir / "refined_depth" / f"{args.img_res}x"
    intrinsic_dir = tapip3d_dir / "intrinsic"
    extrinsic_dir = tapip3d_dir / "extrinsic"
    mask_out_dir = tapip3d_dir / "masks"

    # Create output directories
    tracks_dir.mkdir(parents=True, exist_ok=True)
    vid_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    refined_depth_dir.mkdir(parents=True, exist_ok=True)
    intrinsic_dir.mkdir(parents=True, exist_ok=True)
    extrinsic_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(args.checkpoint)
    model.to(args.device)

    # Determine inference resolution
    if args.use_model_resolution:
        # Use model resolution with resolution_factor
        inference_res = (int(model.image_size[0] * np.sqrt(args.resolution_factor)), int(model.image_size[1] * np.sqrt(args.resolution_factor)))
        logger.info(f"Using model resolution with factor {args.resolution_factor}: {inference_res}")
    else:
        # Use image resolution (default)
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
        image_paths = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in image_extensions])
        if len(image_paths) == 0:
            raise ValueError(f"No images found in directory: {img_dir}")
        first_img = cv2.imread(str(image_paths[0]))
        orig_h, orig_w = first_img.shape[0], first_img.shape[1]

        # Apply resolution downscaling
        if args.img_res > 1:
            # Round to nearest integer if not divisible
            inference_h = round(orig_h / args.img_res)
            inference_w = round(orig_w / args.img_res)
            inference_res = (inference_h, inference_w)
            logger.info(f"Using downscaled resolution (1/{args.img_res}): {inference_res} from original ({orig_h}, {orig_w})")
        else:
            inference_res = (orig_h, orig_w)
            logger.info(f"Using original image resolution: {inference_res}")

    model.set_image_size(inference_res)

    # Prepare inputs
    video, depths, intrinsics, extrinsics, query_point, support_grid_size = prepare_inputs(
        img_dir=str(img_dir),
        inference_res=inference_res,
        support_grid_size=args.support_grid_size,
        num_threads=args.num_threads,
        device=args.device,
        query_frame=0,
        work_dir=work_dir,
        scale_factor=args.scale_factor,
        depth_type=args.depth_type,
        depth_model=args.depth_model,
        iphone=args.iphone,
        colmap=args.colmap,
        depth_dir=args.depth,
        camera_param_path=args.camera_parameter,
        use_megasam=args.use_megasam,
        megasam_resolution=args.megasam_resolution,
        use_support_grid=args.use_support_grid,
    )

    # Save video, depths, intrinsics, extrinsics separately
    video_np = video.cpu().numpy()
    depths_np = depths.cpu().numpy()
    intrinsics_np = intrinsics.cpu().numpy()
    extrinsics_np = extrinsics.cpu().numpy()

    np.save(vid_dir / "video.npy", video_np)
    np.save(depth_dir / "depths.npy", depths_np)
    np.save(intrinsic_dir / "intrinsics.npy", intrinsics_np)
    np.save(extrinsic_dir / "extrinsics.npy", extrinsics_np)

    logger.info(f"Saved video to {vid_dir}")
    logger.info(f"Saved depths to {depth_dir}")
    logger.info(f"Saved intrinsics to {intrinsic_dir}")
    logger.info(f"Saved extrinsics to {extrinsic_dir}")

    # Get image filenames for track naming
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    all_image_paths = [p for p in img_dir.iterdir() if p.suffix.lower() in image_extensions]
    # Filter for images starting with "0_" if iphone is enabled
    if args.iphone:
        image_paths = sorted([p for p in all_image_paths if p.stem.startswith("0_")])
    else:
        image_paths = sorted(all_image_paths)
    image_names = [p.stem for p in image_paths]  # Get filename without extension

    # Load mask images and resize to inference resolution
    masks = []
    masks_tensor = None
    masks_tensor_for_guidance = None
    if mask_dir.exists():
        logger.info(f"Loading masks from {mask_dir} matching image names")
        # Load masks corresponding to image_names (respects iphone filtering)
        for img_name in image_names:
            # Try to find mask with same name but potentially different extension
            mask_found = False
            for ext in image_extensions:
                mask_path = mask_dir / f"{img_name}{ext}"
                if mask_path.exists():
                    mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    # Resize mask to inference resolution
                    mask_img = cv2.resize(mask_img, (inference_res[1], inference_res[0]), interpolation=cv2.INTER_NEAREST)
                    masks.append(mask_img)
                    mask_found = True
                    break

            if not mask_found:
                logger.warning(f"Mask not found for image: {img_name}")
                # Add empty mask as placeholder
                masks.append(np.zeros((inference_res[0], inference_res[1]), dtype=np.uint8))

        if len(masks) > 0:
            logger.info(f"Loaded {len(masks)} mask images matching {len(image_names)} images")
            masks = np.stack(masks)

            # Convert to GPU tensor once if --use_mask is enabled
            if args.use_mask:
                masks_tensor = torch.from_numpy(masks).float().to(args.device)
                logger.info(f"Converted masks to GPU tensor for KNN filtering: {masks_tensor.shape}")
            if args.filter_2dtrack_with_mask:
                masks_tensor_for_guidance = torch.from_numpy(masks).float().to(args.device)
                logger.info(f"Converted masks to GPU tensor for 2D track guidance filtering: {masks_tensor_for_guidance.shape}")
        else:
            logger.warning(f"No images loaded, cannot load masks")
            masks = None
    else:
        logger.warning(f"Mask directory not found: {mask_dir}")
        masks = None

    # Run inference for all frames start_query~T
    T = video.shape[0]
    query_masks_list = []  # Store (GRID_SIZE^2,) boolean arrays for each query time t

    # Validate start_query
    if args.start_query < 0 or args.start_query >= T:
        logger.warning(f"Invalid start_query={args.start_query}, must be in range [0, {T-1}]. Using 0.")
        start_query = 0
    else:
        start_query = args.start_query
        logger.info(f"Starting inference from query frame {start_query}")

    for t in range(start_query, T):
        logger.info(f"Running inference for query frame {t}/{T-1}")

        # Initialize valid_mask variable for 2D track filtering
        valid_mask_2dtrack = None
        sampled_indices_original = None

        # Get query points for current frame
        # Priority 1: Use 2D track query points if --use_2dtrack or --fg_only_2dquery is enabled
        if args.use_2dtrack or args.fg_only_2dquery:
            track2d_dir = work_dir / "bootstapir" / f"{args.scale_factor}"
            if not track2d_dir.exists():
                track2d_dir = work_dir / "flow3d_preprocessed" / "2d_tracks" / f"{args.scale_factor}"
            query_frame_name = image_names[t]
            track_query_file = track2d_dir / f"{query_frame_name}_{query_frame_name}.npy"

            if track_query_file.exists():
                # Load 2D track query points
                track_2d_query = np.load(track_query_file)  # [P, 4] (x, y, occlusion, expected_dist)
                xy_2d = track_2d_query[:, :2]  # [P, 2]

                # Unproject to 3D
                query_point_t, valid_mask_2dtrack = unproject_2d_points_to_3d(
                    xy_2d=xy_2d,
                    depth_map=depths[t],
                    intrinsic=intrinsics[t],
                    extrinsic=extrinsics[t],
                    frame_idx=t,
                    device=args.device
                )

                # Prune query points if requested
                if args.prune_query > 0:
                    num_valid = valid_mask_2dtrack.sum()
                    if num_valid > args.prune_query:
                        # Get indices of valid points in the original array
                        valid_indices = np.where(valid_mask_2dtrack)[0]
                        # Randomly sample N from valid indices
                        sampled_valid_indices = np.random.choice(valid_indices, args.prune_query, replace=False)
                        sampled_valid_indices = np.sort(sampled_valid_indices)  # Sort for consistency

                        # Create new mask: only sampled points are valid
                        new_valid_mask = np.zeros_like(valid_mask_2dtrack, dtype=bool)
                        new_valid_mask[sampled_valid_indices] = True

                        # Store original indices for 2D track guidance
                        sampled_indices_original = sampled_valid_indices

                        # Filter query points and mask
                        query_point_t = query_point_t[sampled_valid_indices]
                        valid_mask_2dtrack = new_valid_mask

                        logger.info(f"Pruned query points from {num_valid} valid points to {args.prune_query}")

                logger.info(f"Using 2D track query points: {query_point_t.shape[0]} points ({valid_mask_2dtrack.sum()}/{len(valid_mask_2dtrack)} valid) from {track_query_file.name}")
            else:
                logger.warning(f"2D track query file not found: {track_query_file}, falling back to grid sampling")
                # Fallback to grid sampling
                query_point_t = get_grid_queries(
                    grid_size=DEFAULT_QUERY_GRID_SIZE,
                    depths=depths,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    frame_idx=t
                )
        # Priority 2: Use FG mask-based sampling if --fg_only flag is set
        elif args.fg_only and masks is not None:
            query_point_t = get_foreground_queries(
                num_points=DEFAULT_QUERY_GRID_SIZE * DEFAULT_QUERY_GRID_SIZE,
                fg_mask=masks[t],
                depths=depths,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                frame_idx=t,
                sampling_strategy="uniform"
            )
            logger.info(f"FG-only mode: Sampled {query_point_t.shape[0]} query points from foreground mask")
        # Priority 3: Default grid sampling
        else:
            query_point_t = get_grid_queries(
                grid_size=DEFAULT_QUERY_GRID_SIZE,
                depths=depths,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                frame_idx=t
            )
            if args.fg_only and masks is None:
                logger.warning(f"--fg_only flag set but no masks found, falling back to grid sampling")

        # Create foreground mask for query points
        # When using FG-based sampling, all query points are from FG by design
        # When using grid sampling, we need to check which points are in FG
        if masks is not None:
            if query_point_t.shape[0] > 0:
                # Since we sampled from FG mask, all points should be foreground
                # But still verify by projecting back to image
                world_coords = query_point_t[:, 1:]  # (P_valid, 3)
                world_coords_homo = torch.cat([world_coords, torch.ones_like(world_coords[:, :1])], dim=-1)  # (P_valid, 4)

                # Transform to camera coordinates
                extrinsic_t = extrinsics[t]  # (4, 4)
                camera_coords_homo = torch.matmul(extrinsic_t, world_coords_homo.T)  # (4, P_valid)
                camera_coords = camera_coords_homo[:3]  # (3, P_valid)

                # Project to image coordinates
                intrinsic_t = intrinsics[t]  # (3, 3)
                image_coords = torch.matmul(intrinsic_t, camera_coords)  # (3, P_valid)
                xy = image_coords[:2] / (image_coords[2:3] + 1e-8)  # (2, P_valid)
                xy = xy.T.cpu().numpy()  # (P_valid, 2)

                # Sample mask at query point locations
                mask_t = masks[t]  # (H, W)
                H_mask, W_mask = mask_t.shape

                # Round to integer coordinates and clip to image bounds
                xy_int = np.round(xy).astype(np.int32)
                xy_int[:, 0] = np.clip(xy_int[:, 0], 0, W_mask - 1)
                xy_int[:, 1] = np.clip(xy_int[:, 1], 0, H_mask - 1)

                # Check if points are in foreground (white=255 or >128)
                query_is_foreground = mask_t[xy_int[:, 1], xy_int[:, 0]] > 128
            else:
                query_is_foreground = np.array([], dtype=bool)
        else:
            # No mask, assume all points are valid
            query_is_foreground = np.ones(query_point_t.shape[0], dtype=bool)

        query_masks_list.append(query_is_foreground)

        # Load 2D track guidance if enabled (but NOT if fg_only_2dquery is used)
        guidance_2d = None
        if args.use_2dtrack and not args.fg_only_2dquery:
            track2d_dir = work_dir / "bootstapir" / f"{args.scale_factor}"
            if not track2d_dir.exists():
                track2d_dir = work_dir / "flow3d_preprocessed" / "2d_tracks" / f"{args.scale_factor}"
                
            if not track2d_dir.exists():
                logger.warning(f"2D track directory not found: {track2d_dir}, skipping 2D guidance")
            elif query_point_t.shape[0] == 0:
                logger.warning(f"No valid query points, skipping 2D guidance")
            else:
                query_frame_name = image_names[t]  # Get query frame name
                guidance_2d = load_and_process_2dtrack(
                    track2d_dir=track2d_dir,
                    query_frame_name=query_frame_name,
                    image_names=image_names,
                    num_points=query_point_t.shape[0],
                    depths=depths,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    device=args.device,
                    sampled_indices=sampled_indices_original,
                    masks=masks_tensor_for_guidance,
                    filter_2dtrack_with_mask=args.filter_2dtrack_with_mask,
                    filter_2dtrack_with_depth=args.filter_2dtrack_with_depth,
                    depth_jump_threshold=args.depth_jump_threshold,
                    mask_erosion_iterations=args.mask_erosion_iterations,
                    track_vis_threshold=args.track_vis_threshold
                )

        # Run inference
        with torch.autocast("cuda", dtype=torch.bfloat16):
            coords, visibs = inference(
                model=model,
                video=video,
                depths=depths,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                query_point=query_point_t,
                num_iters=args.num_iters,
                grid_size=support_grid_size,
                grid_query_frame=t,
                vis_threshold=args.vis_threshold,
                guidance_2d=guidance_2d,
                track_vis_threshold=args.track_vis_threshold,
                masks=masks_tensor,
                synthetic=args.synthetic,
            )

        # Convert to numpy
        coords_np = coords.cpu().numpy()  # T x P x 3
        visibs_np = visibs.cpu().numpy()  # T x P

        # Filter out invalid depth points if using 2D track queries
        # Note: if prune_query was used (sampled_indices_original is not None), filtering is already done
        if valid_mask_2dtrack is not None and sampled_indices_original is None:
            coords_np = coords_np[:, valid_mask_2dtrack, :]  # [T, N, 3] where N <= P
            visibs_np = visibs_np[:, valid_mask_2dtrack]      # [T, N]
            logger.info(f"Filtered to {valid_mask_2dtrack.sum()}/{len(valid_mask_2dtrack)} valid points")

        # Filter outlier tracks based on velocity consistency
        if args.filter_outliers and coords_np.shape[1] > 0:
            logger.info(f"Filtering outlier tracks for frame {t}")
            if args.use_adaptive_filter:
                coords_np, visibs_np, valid_track_mask = filter_outlier_tracks_by_velocity_adaptive(
                    coords_np, visibs_np,
                    velocity_threshold_percentile=95.0,
                    min_valid_frames=3
                )
            else:
                coords_np, visibs_np, valid_track_mask = filter_outlier_tracks_by_velocity(
                    coords_np, visibs_np,
                    velocity_threshold=args.velocity_threshold,
                    min_valid_frames=3
                )

        # Combine coords and visibs into T x P x 4 (or T x N x 4 if filtered)
        track3d = np.concatenate([coords_np, visibs_np[..., None]], axis=-1)  # T x P x 4

        # Save as track3d_{image_name}.npy
        image_name = image_names[t]
        track3d_path = tracks_dir / f"track3d_{image_name}.npy"
        np.save(track3d_path, track3d)
        logger.info(f"Saved tracking results to {track3d_path} with shape {track3d.shape}")

        # Extract refined depth for the query frame t
        # Project 3D coords back to camera space to get depth values
        H, W = inference_res
        refined_depth_map = np.zeros((H, W), dtype=np.float32)

        # Convert world coordinates to camera coordinates
        world_coords = torch.from_numpy(coords_np[t]).to(args.device)  # (P, 3)
        world_coords_homo = torch.cat([world_coords, torch.ones_like(world_coords[:, :1])], dim=-1)  # (P, 4)

        # Transform to camera coordinates
        extrinsic_t = extrinsics[t]  # (4, 4)
        camera_coords_homo = torch.matmul(extrinsic_t, world_coords_homo.T)  # (4, P)
        camera_coords = camera_coords_homo[:3]  # (3, P)

        # Extract depth (z-coordinate in camera space)
        refined_depths = camera_coords[2].cpu().numpy()  # (P,)

        # Project to pixel coordinates to know where to place depth values
        intrinsic_t = intrinsics[t]  # (3, 3)
        image_coords = torch.matmul(intrinsic_t, camera_coords)  # (3, P)
        xy = image_coords[:2] / (image_coords[2:3] + 1e-8)  # (2, P)
        xy = xy.T.cpu().numpy()  # (P, 2)

        # Round to integer coordinates and clip to image bounds
        xy_int = np.round(xy).astype(np.int32)
        valid_mask = (xy_int[:, 0] >= 0) & (xy_int[:, 0] < W) & (xy_int[:, 1] >= 0) & (xy_int[:, 1] < H) & (refined_depths > 0)

        # Fill depth map at valid locations (use max depth if multiple points project to same pixel)
        for i in range(len(xy_int)):
            if valid_mask[i]:
                x, y = xy_int[i]
                refined_depth_map[y, x] = max(refined_depth_map[y, x], refined_depths[i])

        # Save refined depth map
        refined_depth_path = refined_depth_dir / f"depth_{image_name}.npy"
        np.save(refined_depth_path, refined_depth_map)
        logger.info(f"Saved refined depth to {refined_depth_path}")

    # # Save query foreground masks as (T, P) boolean array
    # if len(query_masks_list) > 0:
    #     # breakpoint()
    #     query_masks_array = np.stack(query_masks_list)  # (T, P)
    #     mask_save_path = mask_out_dir / "query_foreground_masks.npy"
    #     np.save(mask_save_path, query_masks_array)
    #     logger.info(f"Saved query foreground masks ({query_masks_array.shape}) to {mask_save_path}")

    logger.info(f"All results saved to {tapip3d_dir.resolve()}")