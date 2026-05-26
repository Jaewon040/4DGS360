import os
import os.path as osp
from dataclasses import dataclass
from functools import partial
from typing import Literal, cast
from pathlib import Path

import json
import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tyro
from loguru import logger as guru
from roma import roma
from tqdm import tqdm

from flow3d.data.base_dataset import BaseDataset
from flow3d.data.utils import (
    UINT16_MAX,
    SceneNormDict,
    median_filter_2d,
    normal_from_depth_image,
    normalize_coords,
    parse_tapir_track_info,
)
from flow3d.transforms import rt_to_mat4


@dataclass
class TapipDataConfig:
    data_dir: str
    start: int = 0
    end: int = -1
    mask_erosion_radius: int = 3
    scene_norm_dict: tyro.conf.Suppress[SceneNormDict | None] = None
    num_targets_per_frame: int = 4
    load_from_cache: bool = False
    track_2d_type: Literal["bootstapir", "tapir"] = "bootstapir"
    res: int = 1
    dycheck: bool = False


class TapipDataset(BaseDataset):
    def __init__(
        self,
        data_dir: str,
        start: int = 0,
        end: int = -1,
        mask_erosion_radius: int = 3,
        scene_norm_dict: SceneNormDict | None = None,
        num_targets_per_frame: int = 4,
        load_from_cache: bool = False,
        track_2d_type: Literal["bootstapir", "tapir"] = "bootstapir",
        res: int = 1,
        dycheck: bool = False,
        **_,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.num_targets_per_frame = num_targets_per_frame
        self.load_from_cache = load_from_cache
        self.has_validation = False
        self.mask_erosion_radius = mask_erosion_radius
        self.res = res

        # Directory setup
        # Try images/{res}x first, then fallback to rgb/{res}x
        img_dir_images = f"{data_dir}/images/{res}x"
        img_dir_rgb = f"{data_dir}/rgb/{res}x"
        if os.path.exists(img_dir_images):
            self.img_dir = img_dir_images
        else:
            self.img_dir = img_dir_rgb
        self.img_ext = os.path.splitext(os.listdir(self.img_dir)[0])[1]
        self.mask_dir = f"{data_dir}/masks/{res}x"
        self.tapip3d_dir = f"{data_dir}/Tapip3d"
        self.cache_dir = f"{data_dir}/cache"

        # 2D track setup
        self.tracks_dir = f"{data_dir}/{track_2d_type}"
        # dycheck 데이터셋의 경우 res가 필요할 수 있음
        if not os.path.exists(self.tracks_dir):
            # 대안 경로들 시도
            alt_paths = [
                f"{data_dir}/{track_2d_type}/",
                f"{data_dir}/bootstapir",
                f"{data_dir}/flow3d_preprocessed/2d_tracks/",
            ]
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    self.tracks_dir = alt_path
                    break
        guru.info(f"Using 2D tracks from {self.tracks_dir}")

        frame_names = [os.path.splitext(p)[0] for p in sorted(os.listdir(self.img_dir))]

        # Filter for dycheck dataset (only use frames starting with "0_")
        if dycheck:
            frame_names = [name for name in frame_names if name.startswith("0_")]
            guru.info(f"DyCheck mode enabled: filtered to {len(frame_names)} frames with '0_' prefix")

        if end == -1:
            end = len(frame_names)
        self.start = start
        self.end = end
        self.frame_names = frame_names[start:end]

        self.imgs: list[torch.Tensor | None] = [None for _ in self.frame_names]
        self.masks: list[torch.Tensor | None] = [None for _ in self.frame_names]

        # Load pre-computed depths (T, 1, H, W) or (T, H, W)
        depth_path = f"{self.tapip3d_dir}/depth/{res}x/depths.npy"
        guru.info(f"Loading depths from {depth_path}")
        depths_all = np.load(depth_path)
        guru.info(f"Loaded depths with shape: {depths_all.shape}")

        # Handle both (T, 1, H, W) and (T, H, W) formats
        if depths_all.ndim == 4:
            depths_all = depths_all[:, 0, :, :]  # (T, H, W)
        elif depths_all.ndim != 3:
            raise ValueError(f"Expected depths to be 3D or 4D, got shape {depths_all.shape}")

        self.depths_array = torch.from_numpy(depths_all).float()[start:end]  # (T, H, W)

        # Load camera parameters
        guru.info(f"Loading camera parameters")
        Ks_all = np.load(f"{self.tapip3d_dir}/intrinsic/intrinsics.npy")  # (T, 3, 3)
        extrinsics_all = np.load(f"{self.tapip3d_dir}/extrinsic/extrinsics.npy")  # (T, 4, 4)

        self.Ks = torch.from_numpy(Ks_all).float()[start:end]
        # Extrinsics are w2c (world to camera)
        self.w2cs = torch.from_numpy(extrinsics_all).float()[start:end]

        # Keyframe indices
        tstamps = torch.from_numpy(np.arange(0, len(frame_names)))
        tmask = (tstamps >= start) & (tstamps < end)
        self._keyframe_idcs = tstamps[tmask] - start
        self.scale = 1

        guru.info(f"Loaded {len(self.frame_names)} frames from {start} to {end}")

        # Scene normalization
        if scene_norm_dict is None:
            cached_scene_norm_dict_path = os.path.join(
                self.cache_dir, "scene_norm_dict.pth"
            )
            if os.path.exists(cached_scene_norm_dict_path) and self.load_from_cache:
                guru.info("loading cached scene norm dict...")
                scene_norm_dict = torch.load(cached_scene_norm_dict_path)
            else:
                # Get 3D tracks for scene normalization
                tracks_3d = self.get_tracks_3d_raw(
                    num_samples=5000, step=max(1, self.num_frames // 10)
                )[0]
                scale, transfm = self.compute_scene_norm(tracks_3d, self.w2cs)
                scene_norm_dict = SceneNormDict(scale=scale, transfm=transfm)
                os.makedirs(self.cache_dir, exist_ok=True)
                torch.save(scene_norm_dict, cached_scene_norm_dict_path)

        # Apply scene normalization
        self.scene_norm_dict = cast(SceneNormDict, scene_norm_dict)
        self.scale = self.scene_norm_dict["scale"]
        transform = self.scene_norm_dict["transfm"]
        self.transform = transform
        guru.info(f"scene norm {self.scale=}, {transform=}")

        self.w2cs = torch.einsum("nij,jk->nik", self.w2cs.float(), torch.linalg.inv(transform.float()))
        self.w2cs[:, :3, 3] /= self.scale

        # Scale depths
        self.depths_array /= self.scale

    @property
    def num_frames(self) -> int:
        return len(self.frame_names)

    @property
    def keyframe_idcs(self) -> torch.Tensor:
        return self._keyframe_idcs

    def __len__(self):
        return len(self.frame_names)

    def get_w2cs(self) -> torch.Tensor:
        return self.w2cs

    def get_Ks(self) -> torch.Tensor:
        return self.Ks

    def get_img_wh(self) -> tuple[int, int]:
        return self.get_image(0).shape[1::-1]

    def get_image(self, index) -> torch.Tensor:
        if self.imgs[index] is None:
            self.imgs[index] = self.load_image(index)
        img = cast(torch.Tensor, self.imgs[index])
        return img

    def get_mask(self, index) -> torch.Tensor:
        if self.masks[index] is None:
            self.masks[index] = self.load_mask(index)
        mask = cast(torch.Tensor, self.masks[index])
        return mask

    def get_depth(self, index) -> torch.Tensor:
        return self.depths_array[index]

    def load_image(self, index) -> torch.Tensor:
        path = f"{self.img_dir}/{self.frame_names[index]}{self.img_ext}"
        img = imageio.imread(path)
        img = torch.from_numpy(img).float() / 255.0
        # Handle RGBA images (4 channels) - take only RGB
        if img.ndim == 3 and img.shape[-1] == 4:
            img = img[..., :3]
        return img

    def load_mask(self, index) -> torch.Tensor:
        path = f"{self.mask_dir}/{self.frame_names[index]}.png"
        r = self.mask_erosion_radius

        # Try to load from the specified resolution
        mask = None
        if os.path.exists(path):
            mask = imageio.imread(path)
        else:
            # Try jpg extension
            path_jpg = f"{self.mask_dir}/{self.frame_names[index]}.jpg"
            if os.path.exists(path_jpg):
                mask = imageio.imread(path_jpg)
            else:
                # Try to load from 1x and resize
                mask_dir_1x = self.mask_dir.replace(f"/{self.res}x", "/1x") if f"/{self.res}x" in self.mask_dir else f"{self.data_dir}/masks/1x"
                path_1x_png = f"{mask_dir_1x}/{self.frame_names[index]}.png"
                path_1x_jpg = f"{mask_dir_1x}/{self.frame_names[index]}.jpg"

                if os.path.exists(path_1x_png):
                    mask_1x = imageio.imread(path_1x_png)
                elif os.path.exists(path_1x_jpg):
                    mask_1x = imageio.imread(path_1x_jpg)
                else:
                    raise FileNotFoundError(f"Mask not found: {path}, {path_jpg}, {path_1x_png}, {path_1x_jpg}")

                # Get target size from image
                img = self.get_image(index)
                target_h, target_w = img.shape[:2]

                # Resize mask
                mask = cv2.resize(mask_1x, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

                # Save resized mask for future use
                os.makedirs(self.mask_dir, exist_ok=True)
                imageio.imwrite(path, mask)

        fg_mask = mask.reshape((*mask.shape[:2], -1)).max(axis=-1) > 0
        bg_mask = ~fg_mask
        fg_mask_erode = cv2.erode(
            fg_mask.astype(np.uint8), np.ones((r, r), np.uint8), iterations=1
        )
        bg_mask_erode = cv2.erode(
            bg_mask.astype(np.uint8), np.ones((r, r), np.uint8), iterations=1
        )
        out_mask = np.zeros_like(fg_mask, dtype=np.float32)
        out_mask[bg_mask_erode > 0] = -1
        out_mask[fg_mask_erode > 0] = 1
        return torch.from_numpy(out_mask).float()

    def load_target_tracks(
        self,
        query_index: int,
        target_indices: list[int],
        dim: int = 1,
    ):
        """
        Load 2D tracks (bootstapir/tapir format).
        :param dim (int), default 1: dimension to stack the time axis
        return (N, T, 4) if dim=1, (T, N, 4) if dim=0
        """
        q_name = self.frame_names[query_index]
        all_tracks = []
        for ti in target_indices:
            t_name = self.frame_names[ti]
            path = f"{self.tracks_dir}/{self.res}x/{q_name}_{t_name}.npy"
            tracks = np.load(path).astype(np.float32)
            # Scale coordinates by resolution factor
            tracks[:, :2] = tracks[:, :2] / self.res
            all_tracks.append(tracks)
        return torch.from_numpy(np.stack(all_tracks, axis=dim))

    def load_tracks_3d(self, query_index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load pre-computed 3D tracks for a query frame.
        Returns:
            tracks_3d: (P, T, 3) - 3D positions of P points across T frames
            fg_mask: (P,) - boolean mask indicating foreground points
            visibles: (P, T) - boolean visibility mask for each point at each frame
        """
        query_name = self.frame_names[query_index]
        track_path = f"{self.tapip3d_dir}/tracks/track3d_{query_name}.npy"

        # Load tracks: (T, P, 4) where 4 = (x, y, z, visibility)
        tracks_full = np.load(track_path).astype(np.float32)  # (T, P, 4)

        # CRITICAL: Validate track dimensions
        T_track, P, _ = tracks_full.shape
        if T_track != self.num_frames:
            guru.warning(f"Track frame count mismatch for query {query_index}: "
                        f"track has {T_track} frames but dataset has {self.num_frames} frames. "
                        f"This will cause index out of bounds errors!")
            # Pad or truncate to match dataset frames
            if T_track < self.num_frames:
                # Pad with zeros and mark as invisible
                pad_frames = self.num_frames - T_track
                padding = np.zeros((pad_frames, P, 4), dtype=np.float32)
                tracks_full = np.concatenate([tracks_full, padding], axis=0)
                guru.info(f"Padded track from {T_track} to {self.num_frames} frames")
            else:
                # Truncate to dataset length
                tracks_full = tracks_full[:self.num_frames]
                guru.info(f"Truncated track from {T_track} to {self.num_frames} frames")

        tracks_3d = torch.from_numpy(tracks_full[:, :, :3])  # (T, P, 3)
        tracks_3d = tracks_3d.permute(1, 0, 2)  # (P, T, 3)

        # Extract visibility information
        visibility = torch.from_numpy(tracks_full[:, :, 3])  # (T, P)
        visibility = visibility.permute(1, 0)  # (P, T)
        visibles = visibility > 0.5  # Convert to boolean (assuming visibility is 0 or 1)

        # Compute foreground mask from mask image
        # Project query points to 2D and check against mask
        query_tracks_3d = tracks_3d[:, query_index, :]  # (P, 3)
        K = self.Ks[query_index]
        w2c = self.w2cs[query_index]

        # If scene is normalized, we must normalize the points before projection
        if hasattr(self, 'transform'):
            query_tracks_3d_homo = F.pad(query_tracks_3d, (0, 1), value=1.0)
            query_tracks_3d_transformed = torch.einsum("ij,pj->pi", self.transform[:3], query_tracks_3d_homo)
            query_tracks_3d = query_tracks_3d_transformed / self.scale

        # Transform to camera coordinates and project to 2D
        points_cam = torch.einsum("ij,pj->pi", w2c[:3, :3], query_tracks_3d) + w2c[:3, 3]
        points_2d = torch.einsum("ij,pj->pi", K, points_cam)
        points_2d = points_2d[:, :2] / points_2d[:, 2:].clamp(min=1e-6)  # (P, 2)

        # Load mask for this frame
        mask = self.get_mask(query_index)  # (H, W)
        H, W = mask.shape

        # Convert to integer coordinates and clip to image bounds
        xy_int = torch.round(points_2d).long()
        xy_int[:, 0] = torch.clamp(xy_int[:, 0], 0, W - 1)
        xy_int[:, 1] = torch.clamp(xy_int[:, 1], 0, H - 1)

        # Check if points are in foreground (mask > 0 means foreground)
        fg_mask = mask[xy_int[:, 1], xy_int[:, 0]] > 0  # (P,)

        return tracks_3d, fg_mask, visibles

    def get_tracks_3d_raw(
        self,
        num_samples: int,
        start: int = 0,
        end: int = -1,
        step: int = 1,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get 3D tracks for scene normalization (without foreground filtering).
        Returns tracks from all query frames combined.
        """
        num_frames = self.num_frames
        if end < 0:
            end = num_frames + 1 + end
        query_idcs = list(range(start, end, step))

        num_per_query_frame = int(np.ceil(num_samples / len(query_idcs)))
        cur_num = 0

        all_tracks_3d = []
        all_colors = []
        all_visibles = []
        all_invisibles = []
        all_confidences = []

        for q_idx in query_idcs:
            tracks_3d, fg_mask, tracks_visibles = self.load_tracks_3d(q_idx)  # (P, T, 3), (P,), (P, T)

            # Sample tracks (and corresponding visibility)
            num_sel = min(num_per_query_frame, num_samples - cur_num, len(tracks_3d))
            if num_sel < len(tracks_3d):
                sel_idcs = np.random.choice(len(tracks_3d), num_sel, replace=False)
                tracks_3d = tracks_3d[sel_idcs]
                tracks_visibles = tracks_visibles[sel_idcs]
            cur_num += tracks_3d.shape[0]

            # Apply scene transformation
            tracks_3d_homo = F.pad(tracks_3d, (0, 1), value=1.0)  # (P, T, 4)
            if hasattr(self, 'transform'):
                tracks_3d_transformed = torch.einsum(
                    "ij,ptj->pti", self.transform[:3], tracks_3d_homo
                )
                tracks_3d = tracks_3d_transformed[..., :3]

            # Get colors from query frame
            img = self.get_image(q_idx)
            H, W = img.shape[:2]
            K = self.Ks[q_idx]
            w2c = self.w2cs[q_idx] if hasattr(self, 'w2cs') else torch.eye(4)

            # Project to get colors
            tracks_query = tracks_3d[:, q_idx, :]  # (P, 3)
            points_cam = torch.einsum("ij,pj->pi", w2c[:3, :3], tracks_query) + w2c[:3, 3]
            points_2d = torch.einsum("ij,pj->pi", K, points_cam)
            points_2d = points_2d[:, :2] / points_2d[:, 2:].clamp(min=1e-6)

            # Sample colors
            points_2d_norm = normalize_coords(points_2d, H, W)
            colors = F.grid_sample(
                img.permute(2, 0, 1)[None],
                points_2d_norm[None, None, :, :],
                align_corners=True,
                padding_mode="border",
            )[0, :, 0, :].T  # (P, 3)

            # Use actual visibility from tracks
            visibles = tracks_visibles
            invisibles = ~tracks_visibles
            confidences = tracks_visibles.float()  # Use visibility as confidence

            all_tracks_3d.append(tracks_3d)
            all_colors.append(colors)
            all_visibles.append(visibles)
            all_invisibles.append(invisibles)
            all_confidences.append(confidences)

        tracks_3d = torch.cat(all_tracks_3d, dim=0)  # (N, T, 3)
        colors = torch.cat(all_colors, dim=0)  # (N, 3)
        visibles = torch.cat(all_visibles, dim=0)  # (N, T)
        invisibles = torch.cat(all_invisibles, dim=0)  # (N, T)
        confidences = torch.cat(all_confidences, dim=0)  # (N, T)

        return tracks_3d, visibles, invisibles, confidences, colors

    def _compute_canonical_time(
        self,
        start: int = 0,
        end: int = -1,
        step: int = 1,
    ) -> int:
        """
        Compute canonical time as the frame with the most foreground visible points.
        """
        num_frames = self.num_frames
        if end < 0:
            end = num_frames + 1 + end
        query_idcs = list(range(start, end, step))

        # Count visible foreground points for each query frame
        max_visible_count = 0
        best_cano_t = query_idcs[len(query_idcs) // 2]  # Default to middle frame

        for q_idx in query_idcs:
            _, fg_mask, tracks_visibles = self.load_tracks_3d(q_idx)
            # Count foreground points visible at the query frame
            fg_visible_count = (fg_mask & tracks_visibles[:, q_idx]).sum().item()

            if fg_visible_count > max_visible_count:
                max_visible_count = fg_visible_count
                best_cano_t = q_idx

        guru.info(f"Selected canonical time {best_cano_t} with {max_visible_count} visible foreground points")
        return best_cano_t

    def get_tracks_3d(
        self,
        num_samples: int,
        start: int = 0,
        end: int = -1,
        step: int = 1,
        depth_scale: float = 1.0,
        return_cano_t: bool = False,
        cano_t_only: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Get 3D tracks with foreground filtering for initialization.
        This is used for foreground object initialization.

        Args:
            return_cano_t: If True, return canonical time as 6th element
            cano_t_only: If True, only compute and return canonical time (fast path)
        """
        num_frames = self.num_frames
        if end < 0:
            end = num_frames + 1 + end

        # Compute canonical time
        cano_t = self._compute_canonical_time(start, end, step)

        # Fast path: only return canonical time
        if cano_t_only:
            # Return dummy values for the first 5 elements since they won't be used
            dummy = torch.empty(0)
            if return_cano_t:
                return dummy, dummy, dummy, dummy, dummy, cano_t
            else:
                return dummy, dummy, dummy, dummy, dummy

        query_idcs = list(range(start, end, step))

        num_per_query_frame = int(np.ceil(num_samples / len(query_idcs)))
        cur_num = 0

        all_tracks_3d = []
        all_colors = []
        all_visibles = []
        all_invisibles = []
        all_confidences = []

        for q_idx in query_idcs:
            tracks_3d, fg_mask, tracks_visibles = self.load_tracks_3d(q_idx)  # (P, T, 3), (P,), (P, T)

            # Filter for foreground only
            tracks_3d = tracks_3d[fg_mask]  # (P_fg, T, 3)
            tracks_visibles = tracks_visibles[fg_mask]  # (P_fg, T)

            if len(tracks_3d) == 0:
                continue

            # Sample tracks (and corresponding visibility)
            num_sel = min(num_per_query_frame, num_samples - cur_num, len(tracks_3d))
            if num_sel < len(tracks_3d):
                sel_idcs = np.random.choice(len(tracks_3d), num_sel, replace=False)
                tracks_3d = tracks_3d[sel_idcs]
                tracks_visibles = tracks_visibles[sel_idcs]
            cur_num += tracks_3d.shape[0]

            # Apply scene transformation
            tracks_3d_homo = F.pad(tracks_3d, (0, 1), value=1.0)  # (P, T, 4)
            tracks_3d_transformed = torch.einsum(
                "ij,ptj->pti", self.transform[:3], tracks_3d_homo
            )
            tracks_3d = tracks_3d_transformed[..., :3]
            tracks_3d /= self.scale

            # Get colors from query frame
            img = self.get_image(q_idx)
            H, W = img.shape[:2]
            K = self.Ks[q_idx]
            w2c = self.w2cs[q_idx]

            # Project to get colors
            tracks_query = tracks_3d[:, q_idx, :]  # (P, 3)
            points_cam = torch.einsum("ij,pj->pi", w2c[:3, :3], tracks_query) + w2c[:3, 3]
            points_2d = torch.einsum("ij,pj->pi", K, points_cam)
            points_2d = points_2d[:, :2] / points_2d[:, 2:].clamp(min=1e-6)

            # Sample colors
            points_2d_norm = normalize_coords(points_2d, H, W)
            colors = F.grid_sample(
                img.permute(2, 0, 1)[None],
                points_2d_norm[None, None, :, :],
                align_corners=True,
                padding_mode="border",
            )[0, :, 0, :].T  # (P, 3)

            # Use actual visibility from tracks
            visibles = tracks_visibles
            invisibles = ~tracks_visibles
            confidences = tracks_visibles.float()  # Use visibility as confidence

            all_tracks_3d.append(tracks_3d)
            all_colors.append(colors)
            all_visibles.append(visibles)
            all_invisibles.append(invisibles)
            all_confidences.append(confidences)

        tracks_3d = torch.cat(all_tracks_3d, dim=0)  # (N, T, 3)
        colors = torch.cat(all_colors, dim=0)  # (N, 3)
        visibles = torch.cat(all_visibles, dim=0)  # (N, T)
        invisibles = torch.cat(all_invisibles, dim=0)  # (N, T)
        confidences = torch.cat(all_confidences, dim=0)  # (N, T)

        if return_cano_t:
            return tracks_3d, visibles, invisibles, confidences, colors, cano_t
        else:
            return tracks_3d, visibles, invisibles, confidences, colors

    def get_bkgd_points(
        self,
        num_samples: int,
        use_kf_tstamps: bool = True,
        stride: int = 8,
        down_rate: int = 8,
        min_per_frame: int = 64,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get background points from depth maps."""
        start = 0
        end = self.num_frames
        H, W = self.get_image(0).shape[:2]
        grid = torch.stack(
            torch.meshgrid(
                torch.arange(0, W, dtype=torch.float32),
                torch.arange(0, H, dtype=torch.float32),
                indexing="xy",
            ),
            dim=-1,
        )

        if use_kf_tstamps:
            query_idcs = self.keyframe_idcs.tolist()
        else:
            num_query_frames = self.num_frames // stride
            query_endpts = torch.linspace(start, end, num_query_frames + 1)
            query_idcs = ((query_endpts[:-1] + query_endpts[1:]) / 2).long().tolist()

        bg_geometry = []
        guru.info(f"Loading bkgd points from {len(query_idcs)} frames")

        for query_idx in tqdm(query_idcs, desc="Loading bkgd points", leave=False):
            img = self.get_image(query_idx)
            depth = self.get_depth(query_idx)
            bg_mask = self.get_mask(query_idx) < 0
            bool_mask = (bg_mask * (depth > 0)).to(torch.bool)
            w2c = self.w2cs[query_idx]
            K = self.Ks[query_idx]

            # Get the bounding box of previous points that reproject into frame
            bmax_x, bmax_y, bmin_x, bmin_y = 0, 0, W, H
            for p3d, _, _ in bg_geometry:
                if len(p3d) < 1:
                    continue
                # Reproject into current frame
                p2d = torch.einsum(
                    "ij,jk,pk->pi", K, w2c[:3], F.pad(p3d, (0, 1), value=1.0)
                )
                p2d = p2d[:, :2] / p2d[:, 2:].clamp(min=1e-6)
                xmin, xmax = p2d[:, 0].min().item(), p2d[:, 0].max().item()
                ymin, ymax = p2d[:, 1].min().item(), p2d[:, 1].max().item()

                bmin_x = min(bmin_x, int(xmin))
                bmin_y = min(bmin_y, int(ymin))
                bmax_x = max(bmax_x, int(xmax))
                bmax_y = max(bmax_y, int(ymax))

            # Don't include points that are covered by previous points
            bmin_x = max(0, bmin_x)
            bmin_y = max(0, bmin_y)
            bmax_x = min(W, bmax_x)
            bmax_y = min(H, bmax_y)
            overlap_mask = torch.ones_like(bool_mask)
            overlap_mask[bmin_y:bmax_y, bmin_x:bmax_x] = 0

            bool_mask &= overlap_mask
            if bool_mask.sum() < min_per_frame:
                guru.debug(f"skipping {query_idx=}")
                continue

            points = (
                torch.einsum(
                    "ij,pj->pi",
                    torch.linalg.inv(K),
                    F.pad(grid[bool_mask], (0, 1), value=1.0),
                )
                * depth[bool_mask][:, None]
            )
            points = torch.einsum(
                "ij,pj->pi", torch.linalg.inv(w2c)[:3], F.pad(points, (0, 1), value=1.0)
            )
            point_normals = normal_from_depth_image(depth, K, w2c)[bool_mask]
            point_colors = img[bool_mask]

            num_sel = max(len(points) // down_rate, min_per_frame)
            sel_idcs = np.random.choice(len(points), num_sel, replace=False)
            points = points[sel_idcs]
            point_normals = point_normals[sel_idcs]
            point_colors = point_colors[sel_idcs]
            guru.debug(f"{query_idx=} {points.shape=}")
            bg_geometry.append((points, point_normals, point_colors))

        bg_points, bg_normals, bg_colors = map(
            partial(torch.cat, dim=0), zip(*bg_geometry)
        )

        if len(bg_points) > num_samples:
            sel_idcs = np.random.choice(len(bg_points), num_samples, replace=False)
            bg_points = bg_points[sel_idcs]
            bg_normals = bg_normals[sel_idcs]
            bg_colors = bg_colors[sel_idcs]

        return bg_points, bg_normals, bg_colors

    def __getitem__(self, index: int):
        """
        Get training data for a single frame.
        Two modes:
        - use_2d_track=False: Use 3D tracks and project to 2D
        - use_2d_track=True: Load 2D tracks from bootstapir/tapir
        """
        index = np.random.randint(0, self.num_frames)

        data = {
            "frame_names": self.frame_names[index],
            "ts": torch.tensor(index),
            "w2cs": self.w2cs[index],
            "Ks": self.Ks[index],
            "imgs": self.get_image(index),
            "depths": self.get_depth(index),
        }

        tri_mask = self.get_mask(index)
        valid_mask = tri_mask != 0
        mask = tri_mask == 1
        data["masks"] = mask.float()
        data["valid_masks"] = valid_mask.float()

        # Select target frames
        target_inds = torch.from_numpy(
            np.random.choice(
                self.num_frames, (self.num_targets_per_frame,), replace=False
            )
        )

        # Load 2D tracks from bootstapir/tapir
        # (P, 2)
        query_tracks = self.load_target_tracks(index, [index])[:, 0, :2]
        # (N, P, 4)
        target_tracks = self.load_target_tracks(index, target_inds.tolist(), dim=0)

        data["query_tracks_2d"] = query_tracks
        data["target_ts"] = target_inds
        data["target_w2cs"] = self.w2cs[target_inds]
        data["target_Ks"] = self.Ks[target_inds]
        data["target_tracks_2d"] = target_tracks[..., :2]

        # Parse visibility info from 2D tracks
        (
            data["target_visibles"],
            data["target_invisibles"],
            data["target_confidences"],
        ) = parse_tapir_track_info(target_tracks[..., 2], target_tracks[..., 3])

        # Sample depth at track locations
        # (N, H, W)
        target_depths = torch.stack([self.get_depth(i) for i in target_inds], dim=0)
        H, W = target_depths.shape[-2:]
        data["target_track_depths"] = F.grid_sample(
            target_depths[:, None],
            normalize_coords(target_tracks[..., None, :2], H, W),
            align_corners=True,
            padding_mode="border",
        )[:, 0, :, 0]

        return data

    def compute_scene_norm(
        self, X: torch.Tensor, w2cs: torch.Tensor
    ) -> tuple[float, torch.Tensor]:
        """
        Compute scene normalization parameters.
        :param X: [N*T, 3] or [N, T, 3]
        :param w2cs: [N, 4, 4]
        """
        X = X.reshape(-1, 3)
        scene_center = X.mean(dim=0)
        X = X - scene_center[None]
        min_scale = X.quantile(0.05, dim=0)
        max_scale = X.quantile(0.95, dim=0)
        scale = (max_scale - min_scale).max().item() / 2.0
        original_up = -F.normalize(w2cs[:, 1, :3].mean(0), dim=-1)
        target_up = original_up.new_tensor([0.0, 0.0, 1.0])
        R = roma.rotvec_to_rotmat(
            F.normalize(original_up.cross(target_up), dim=-1)
            * original_up.dot(target_up).acos_()
        )
        transfm = rt_to_mat4(R, torch.einsum("ij,j->i", -R, scene_center))
        return scale, transfm


if __name__ == "__main__":
    # Example usage
    dataset = TapipDataset(
        data_dir="/path/to/your/data",
        start=0,
        end=-1,
    )
    print(f"Dataset loaded with {len(dataset)} frames")

    # Test loading a sample
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
