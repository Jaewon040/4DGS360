import os
import json
from dataclasses import dataclass
from functools import partial
from typing import Literal, cast
from pathlib import Path

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
    get_tracks_3d_for_query_frame,
    median_filter_2d,
    normal_from_depth_image,
    normalize_coords,
    parse_tapir_track_info,
)
from flow3d.transforms import rt_to_mat4


@dataclass
class DavisDataConfig:
    seq_name: str
    root_dir: str
    data_type: str = "davis"
    start: int = 0
    end: int = -1
    res: str = "480p"
    image_type: str = "JPEGImages"
    mask_type: str = "Annotations"
    depth_type: Literal[
        "aligned_depth_anything",
        "aligned_depth_anything_v2",
        "depth_anything",
        "depth_anything_v2",
        "unidepth_disp",
    ] = "aligned_depth_anything"
    camera_type: Literal["droid_recon"] = "droid_recon"
    track_2d_type: Literal["bootstapir", "tapir"] = "bootstapir"
    mask_erosion_radius: int = 3
    scene_norm_dict: tyro.conf.Suppress[SceneNormDict | None] = None
    num_targets_per_frame: int = 4
    load_from_cache: bool = False


@dataclass
class CustomDataConfig:
    data_dir: str
    data_type: str = "custom"
    start: int = 0
    end: int = -1
    res: str = "1x"
    image_type: str = "images"
    mask_type: str = "masks"
    depth_type: Literal[
        "aligned_depth_anything",
        "aligned_depth_anything_v2",
        "depth_anything",
        "depth_anything_v2",
        "unidepth_disp",
    ] = "aligned_depth_anything"
    camera_type: Literal["droid_recon", "megasam", "panoptic_custom"] = "droid_recon"
    track_2d_type: Literal["bootstapir", "tapir"] = "bootstapir"
    mask_erosion_radius: int = 7
    scene_norm_dict: tyro.conf.Suppress[SceneNormDict | None] = None
    num_targets_per_frame: int = 4
    load_from_cache: bool = False
    # shape-of-motion-back features
    use_pseudo_view: bool = False
    pseudo_track_2d_type: str = "bootstapir"
    pseudo_sequences: list[str] | None = None
    num_pseudo_targets_per_frame: int = 4
    find_reasonable_3d_tracks: bool = False
    find_r3t_knn: bool = False


@dataclass
class PanopticCustomDataConfig:
    data_dir: str
    data_type: str = "custom"
    start: int = 0
    end: int = -1
    res: str = "seq1"
    image_type: str = "images"
    mask_type: str = "masks"
    depth_type: Literal[
        "aligned_depth_anything",
        "aligned_depth_anything_v2",
        "depth_anything",
        "depth_anything_v2",
        "unidepth_disp",
    ] = "aligned_depth_anything"
    camera_type: Literal["droid_recon", "megasam", "panoptic_custom"] = "panoptic_custom"
    track_2d_type: Literal["bootstapir", "tapir"] = "bootstapir"
    mask_erosion_radius: int = 7
    scene_norm_dict: tyro.conf.Suppress[SceneNormDict | None] = None
    num_targets_per_frame: int = 4
    load_from_cache: bool = False
    # shape-of-motion-back features
    use_pseudo_view: bool = False
    pseudo_track_2d_type: str = "bootstapir"
    pseudo_sequences: list[str] | None = None
    num_pseudo_targets_per_frame: int = 4
    find_reasonable_3d_tracks: bool = False
    find_r3t_knn: bool = False


class CasualDataset(BaseDataset):
    def __init__(
        self,
        data_dir: str = None,
        seq_name: str = None,
        root_dir: str = None,
        data_type: str = "custom",
        start: int = 0,
        end: int = -1,
        res: str = "1x",
        image_type: str = "JPEGImages",
        mask_type: str = "Annotations",
        depth_type: Literal[
            "aligned_depth_anything",
            "aligned_depth_anything_v2",
            "depth_anything",
            "depth_anything_v2",
            "unidepth_disp",
        ] = "aligned_depth_anything",
        camera_type: Literal["droid_recon", "megasam", "panoptic_custom"] = "droid_recon",
        track_2d_type: Literal["bootstapir", "tapir"] = "bootstapir",
        mask_erosion_radius: int = 3,
        scene_norm_dict: SceneNormDict | None = None,
        num_targets_per_frame: int = 4,
        load_from_cache: bool = False,
        # shape-of-motion-back features
        use_pseudo_view: bool = False,
        pseudo_track_2d_type: str = "bootstapir",
        pseudo_sequences: list[str] | None = None,
        num_pseudo_targets_per_frame: int = 4,
        find_reasonable_3d_tracks: bool = False,
        find_r3t_knn: bool = False,
        **_,
    ):
        super().__init__()

        self.data_type = data_type
        self.depth_type = depth_type
        self.camera_type = camera_type
        self.num_targets_per_frame = num_targets_per_frame
        self.load_from_cache = load_from_cache
        self.has_validation = False
        self.mask_erosion_radius = mask_erosion_radius
        self.find_reasonable_3d_tracks = find_reasonable_3d_tracks
        self.find_r3t_knn = find_r3t_knn

        # Support both old (seq_name + root_dir) and new (data_dir) style
        if data_dir is not None:
            # New style: shape-of-motion-back compatible
            self.data_dir = data_dir
            self.res = res
            self.img_dir = f"{data_dir}/{image_type}/{res}"
            self.depth_dir = f"{data_dir}/{depth_type}/{res}"
            self.mask_dir = f"{data_dir}/{mask_type}/{res}"
            self.tracks_dir = f"{data_dir}/{track_2d_type}/{res}"
            self.cache_dir = f"{data_dir}/flow3d_preprocessed/{res}"
        else:
            # Old style: himor original
            self.seq_name = seq_name
            self.root_dir = root_dir
            self.res = res
            self.data_dir = f"{root_dir}"
            self.img_dir = f"{root_dir}/{image_type}/{res}/{seq_name}"
            self.depth_dir = f"{root_dir}/{depth_type}/{res}/{seq_name}"
            self.mask_dir = f"{root_dir}/{mask_type}/{res}/{seq_name}"
            self.tracks_dir = f"{root_dir}/{track_2d_type}/{res}/{seq_name}"
            self.cache_dir = f"{root_dir}/flow3d_preprocessed/{res}/{seq_name}"

        self.img_ext = os.path.splitext(os.listdir(self.img_dir)[0])[1]
        frame_names = [os.path.splitext(p)[0] for p in sorted(os.listdir(self.img_dir))]

        # Pseudo view setup
        self.use_pseudo_view = use_pseudo_view
        self.num_pseudo_targets_per_frame = num_pseudo_targets_per_frame
        if use_pseudo_view:
            if pseudo_sequences is None:
                pseudo_sequences = ["seq5", "seq25"]
            self.pseudo_sequences = pseudo_sequences
            self.pseudo_tracks_dirs = {}

            for seq_name in pseudo_sequences:
                if data_dir is not None:
                    self.pseudo_tracks_dirs[seq_name] = f"{data_dir}/{pseudo_track_2d_type}/{seq_name}"
                else:
                    self.pseudo_tracks_dirs[seq_name] = f"{root_dir}/{pseudo_track_2d_type}/{seq_name}"

                if not os.path.exists(self.pseudo_tracks_dirs[seq_name]):
                    guru.warning(f"Pseudo tracks directory not found: {self.pseudo_tracks_dirs[seq_name]}")

            guru.info(f"Loaded pseudo sequences: {pseudo_sequences}")
            guru.info(f"Pseudo tracks directories: {self.pseudo_tracks_dirs}")

        if end == -1:
            end = len(frame_names)
        self.start = start
        self.end = end
        self.frame_names = frame_names[start:end]

        self.imgs: list[torch.Tensor | None] = [None for _ in self.frame_names]
        self.depths: list[torch.Tensor | None] = [None for _ in self.frame_names]
        self.masks: list[torch.Tensor | None] = [None for _ in self.frame_names]

        # Load cameras based on type
        if camera_type == "droid_recon":
            img = self.get_image(0)
            H, W = img.shape[:2]
            if seq_name is not None:
                # Old style
                w2cs, Ks, tstamps = load_cameras(
                    f"{root_dir}/{camera_type}/{seq_name}.npy", H, W
                )
            else:
                # New style - try to infer name from data_dir
                data_name = data_dir.split("/")[-1]
                camera_path = f"{data_dir}/{camera_type}/{res}/droid_recon.npy"
                w2cs, Ks, tstamps = load_cameras(camera_path, H, W)

        elif camera_type == "megasam":
            data_name = data_dir.split("/")[-1] if data_dir else seq_name
            if data_dir is not None:
                cam_path = Path(data_dir) / f"{data_name}.npz"
            else:
                cam_path = Path(root_dir) / f"{data_name}.npz"

            cams = np.load(cam_path)
            c2ws = cams["cam_c2w"][:self.end]
            K = cams["intrinsic"]

            c2ws = torch.from_numpy(c2ws)
            w2cs = torch.linalg.inv(c2ws)
            Ks = torch.from_numpy(K).unsqueeze(0).repeat((c2ws.shape[0], 1, 1))
            tstamps = torch.arange(len(w2cs))

        elif camera_type == "panoptic_custom":
            guru.info("Loading panoptic_custom (fixed camera) parameters")
            img = self.get_image(0)
            H, W = img.shape[:2]

            cam_path = f"{self.data_dir}/{camera_type}/{res}/cam_parameter.json"
            with open(cam_path, "r") as f:
                cam_params = json.load(f)

            intr = cam_params["intrinsics"]
            fx, fy = intr["fx"], intr["fy"]
            cx, cy = intr["cx"], intr["cy"]

            K = np.array([
                [fx,  0.0, cx],
                [0.0, fy,  cy],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)
            w2c = np.array(cam_params["w2c"], dtype=np.float32)

            # Fixed camera: replicate for all frames
            w2cs = torch.from_numpy(w2c).unsqueeze(0).repeat(self.end, 1, 1)
            Ks = torch.from_numpy(K).unsqueeze(0).repeat(self.end, 1, 1)
            tstamps = torch.arange(self.end)
        else:
            raise ValueError(f"Unknown camera type: {camera_type}")

        assert (
            len(self.frame_names) == len(w2cs) == len(Ks)
        ), f"{len(self.frame_names)}, {len(w2cs)}, {len(Ks)}"

        self.w2cs = w2cs[start:end]
        self.Ks = Ks[start:end]

        if camera_type == "droid_recon":
            tmask = (tstamps >= start) & (tstamps < end)
            self._keyframe_idcs = tstamps[tmask] - start
        else:
            self._keyframe_idcs = torch.arange(len(self.frame_names))

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
                tracks_3d = self.get_tracks_3d(5000, step=self.num_frames // 10)[0]
                scale, transfm = compute_scene_norm(tracks_3d, self.w2cs)
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

        # Load pseudo cameras if needed
        if self.use_pseudo_view:
            self.pseudo_cameras = self.load_pseudo_cameras()
            guru.info(f"Loaded pseudo cameras: {list(self.pseudo_cameras.keys())}")

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
        if self.depths[index] is None:
            if self.camera_type == "droid_recon" or self.camera_type == "panoptic_custom":
                self.depths[index] = self.load_depth(index)
            elif self.camera_type == "megasam":
                data_name = self.data_dir.split("/")[-1]
                depth_path = Path(self.data_dir) / f"{data_name}.npz"
                depths = np.load(depth_path)
                self.depths[index] = torch.tensor(depths["depths"][index]).float()
        return self.depths[index] / self.scale

    def load_image(self, index) -> torch.Tensor:
        path = f"{self.img_dir}/{self.frame_names[index]}{self.img_ext}"
        return torch.from_numpy(imageio.imread(path)).float() / 255.0

    def load_mask(self, index) -> torch.Tensor:
        path = f"{self.mask_dir}/{self.frame_names[index]}.png"
        r = self.mask_erosion_radius
        try:
            mask = imageio.imread(path)
        except:
            path = f"{self.mask_dir}/{self.frame_names[index]}.jpg"
            mask = imageio.imread(path)
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

    def load_depth(self, index) -> torch.Tensor:
        path = f"{self.depth_dir}/{self.frame_names[index]}.npy"
        disp = np.load(path)
        depth = 1.0 / np.clip(disp, a_min=1e-6, a_max=1e6)
        depth = torch.from_numpy(depth).float()
        depth = median_filter_2d(depth[None, None], 11, 1)[0, 0]
        return depth

    def load_target_tracks(
        self, query_index: int, target_indices: list[int], dim: int = 1
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
            path = f"{self.tracks_dir}/{q_name}_{t_name}.npy"
            tracks = np.load(path).astype(np.float32)
            all_tracks.append(tracks)
        return torch.from_numpy(np.stack(all_tracks, axis=dim))

    def load_pseudo_target_tracks(
        self, query_index: int, target_indices: list[int], dim: int = 1
    ):
        """Load pseudo view 2D tracks from multiple sequences."""
        q_name = self.frame_names[query_index]
        all_pseudo_tracks = {}

        for seq_name in self.pseudo_sequences:
            seq_tracks = []
            for ti in target_indices:
                t_name = self.frame_names[ti]
                path = f"{self.pseudo_tracks_dirs[seq_name]}/{q_name}_{t_name}.npy"

                try:
                    tracks = np.load(path).astype(np.float32)
                    seq_tracks.append(tracks)
                except FileNotFoundError:
                    guru.warning(f"Pseudo track file not found: {path}")
                    dummy_tracks = np.zeros((0, 4), dtype=np.float32)
                    seq_tracks.append(dummy_tracks)

            all_pseudo_tracks[seq_name] = torch.from_numpy(np.stack(seq_tracks, axis=dim))

        return all_pseudo_tracks

    def load_pseudo_cameras(self):
        """Load camera parameters for each pseudo view sequence."""
        pseudo_cameras = {}
        data_name = self.data_dir.split("/")[-1]
        img = self.get_image(0)
        H, W = img.shape[:2]

        for seq_name in self.pseudo_sequences:
            if self.camera_type == "droid_recon":
                camera_path = f"{self.data_dir}/{self.camera_type}/{seq_name}/{data_name}.npy"

                if os.path.exists(camera_path):
                    try:
                        pseudo_w2cs, pseudo_Ks, pseudo_tstamps = load_cameras(
                            camera_path, H, W
                        )

                        # Apply scene normalization
                        if hasattr(self, 'scene_norm_dict'):
                            transform = self.scene_norm_dict["transfm"]
                            pseudo_w2cs = torch.einsum(
                                "nij,jk->nik",
                                pseudo_w2cs.float(),
                                torch.linalg.inv(transform.float())
                            )
                            pseudo_w2cs[:, :3, 3] /= self.scale

                        pseudo_cameras[seq_name] = {
                            'w2cs': pseudo_w2cs[self.start:self.end],
                            'Ks': pseudo_Ks[self.start:self.end],
                        }

                        guru.info(f"Loaded pseudo cameras for {seq_name}: {camera_path}")
                    except Exception as e:
                        guru.error(f"Error loading pseudo cameras for {seq_name}: {e}")
                        pseudo_cameras[seq_name] = {
                            'w2cs': self.w2cs,
                            'Ks': self.Ks
                        }
                else:
                    guru.warning(f"Pseudo camera file not found: {camera_path}")
                    pseudo_cameras[seq_name] = {
                        'w2cs': self.w2cs,
                        'Ks': self.Ks
                    }

            elif self.camera_type == "megasam":
                camera_path = f"{self.data_dir}/{self.camera_type}/{seq_name}/{data_name}.npz"

                if os.path.exists(camera_path):
                    try:
                        cams = np.load(camera_path)
                        pseudo_c2ws = cams["cam_c2w"][:self.end]
                        pseudo_K = cams["intrinsic"]

                        pseudo_c2ws = torch.from_numpy(pseudo_c2ws)
                        pseudo_w2cs = torch.linalg.inv(pseudo_c2ws)
                        pseudo_Ks = torch.from_numpy(pseudo_K).unsqueeze(0).repeat(
                            (pseudo_c2ws.shape[0], 1, 1)
                        )

                        # Apply scene normalization
                        if hasattr(self, 'scene_norm_dict'):
                            transform = self.scene_norm_dict["transfm"]
                            pseudo_w2cs = torch.einsum(
                                "nij,jk->nik",
                                pseudo_w2cs.float(),
                                torch.linalg.inv(transform.float())
                            )
                            pseudo_w2cs[:, :3, 3] /= self.scale

                        pseudo_cameras[seq_name] = {
                            'w2cs': pseudo_w2cs[self.start:self.end],
                            'Ks': pseudo_Ks[self.start:self.end]
                        }

                        guru.info(f"Loaded pseudo cameras for {seq_name}: {camera_path}")
                    except Exception as e:
                        guru.error(f"Error loading pseudo cameras for {seq_name}: {e}")
                        pseudo_cameras[seq_name] = {
                            'w2cs': self.w2cs,
                            'Ks': self.Ks
                        }
                else:
                    guru.warning(f"Pseudo camera file not found: {camera_path}")
                    pseudo_cameras[seq_name] = {
                        'w2cs': self.w2cs,
                        'Ks': self.Ks
                    }

            elif self.camera_type == "panoptic_custom":
                guru.info(f"Loading panoptic_custom pseudo camera for {seq_name}")

                cam_path = f"{self.data_dir}/{self.camera_type}/{seq_name}/cam_parameter.json"

                if os.path.exists(cam_path):
                    try:
                        with open(cam_path, "r") as f:
                            cam_params = json.load(f)

                        intr = cam_params["intrinsics"]
                        fx, fy = intr["fx"], intr["fy"]
                        cx, cy = intr["cx"], intr["cy"]

                        K = np.array([
                            [fx,  0.0, cx],
                            [0.0, fy,  cy],
                            [0.0, 0.0, 1.0]
                        ], dtype=np.float32)
                        w2c = np.array(cam_params["w2c"], dtype=np.float32)

                        pseudo_w2cs = torch.from_numpy(w2c).unsqueeze(0).repeat(self.end, 1, 1)
                        pseudo_Ks = torch.from_numpy(K).unsqueeze(0).repeat(self.end, 1, 1)

                        # Apply scene normalization
                        transform = self.scene_norm_dict["transfm"]
                        pseudo_w2cs = torch.einsum("nij,jk->nik", pseudo_w2cs.float(), torch.linalg.inv(transform.float()))
                        pseudo_w2cs[:, :3, 3] /= self.scale

                        pseudo_cameras[seq_name] = {
                            'w2cs': pseudo_w2cs[self.start:self.end],
                            'Ks': pseudo_Ks[self.start:self.end],
                        }
                        guru.info(f"Loaded panoptic_custom pseudo camera for {seq_name}")
                    except Exception as e:
                        guru.error(f"Error loading pseudo camera for {seq_name}: {e}")
                        pseudo_cameras[seq_name] = {
                            'w2cs': self.w2cs,
                            'Ks': self.Ks
                        }
                else:
                    guru.warning(f"Pseudo camera file not found: {cam_path}")
                    pseudo_cameras[seq_name] = {
                        'w2cs': self.w2cs,
                        'Ks': self.Ks
                    }
            else:
                guru.warning(f"Unknown camera type for pseudo views: {self.camera_type}")
                pseudo_cameras[seq_name] = {
                    'w2cs': self.w2cs,
                    'Ks': self.Ks
                }

        return pseudo_cameras

    def get_tracks_3d(
        self, num_samples: int, start: int = 0, end: int = -1, step: int = 1, depth_scale: float = 1.0, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_frames = self.num_frames
        if end < 0:
            end = num_frames + 1 + end
        query_idcs = list(range(start, end, step))
        target_idcs = list(range(start, end, step))
        masks = torch.stack([self.get_mask(i) for i in target_idcs], dim=0)
        fg_masks = (masks == 1).float()
        depths = torch.stack([self.get_depth(i) * depth_scale for i in target_idcs], dim=0)
        inv_Ks = torch.linalg.inv(self.Ks[target_idcs])
        c2ws = torch.linalg.inv(self.w2cs[target_idcs])

        num_per_query_frame = int(np.ceil(num_samples / len(query_idcs)))
        cur_num = 0

        tracks_all_queries = []
        for q_idx in query_idcs:
            # (N, T, 4)
            tracks_2d = self.load_target_tracks(q_idx, target_idcs)
            num_sel = int(
                min(num_per_query_frame, num_samples - cur_num, len(tracks_2d))
            )

            if num_sel < len(tracks_2d):
                num_sel = num_per_query_frame
                sel_idcs = np.random.choice(len(tracks_2d), num_sel, replace=False)
                tracks_2d = tracks_2d[sel_idcs]
            cur_num += tracks_2d.shape[0]
            img = self.get_image(q_idx)
            tidx = target_idcs.index(q_idx)
            tracks_tuple = get_tracks_3d_for_query_frame(
                tidx, img, tracks_2d, depths, fg_masks, inv_Ks, c2ws
            )

            if self.find_reasonable_3d_tracks:
                from flow3d.init_utils import complete_3d_tracks
                from flow3d.tensor_dataclass import TrackObservations

                initial_tracks = TrackObservations(
                    xyz=tracks_tuple[0],
                    visibles=tracks_tuple[2],
                    invisibles=tracks_tuple[3],
                    confidences=tracks_tuple[4],
                    colors=tracks_tuple[1],
                )
                completed_tracks = complete_3d_tracks(
                    initial_tracks,
                    q_idx=tidx,
                    use_knn=self.find_r3t_knn,
                    debug_init_completion=kwargs.get("debug_init_completion", False),
                )
                tracks_tuple = (
                    completed_tracks.xyz,
                    completed_tracks.colors,
                    completed_tracks.visibles,
                    completed_tracks.invisibles,
                    completed_tracks.confidences,
                )

            tracks_all_queries.append(tracks_tuple)
        tracks_3d, colors, visibles, invisibles, confidences = map(
            partial(torch.cat, dim=0), zip(*tracks_all_queries)
        )

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

        # Query tracks
        query_tracks = self.load_target_tracks(index, [index])[:, 0, :2]
        target_inds = torch.from_numpy(
            np.random.choice(
                self.num_frames, (self.num_targets_per_frame,), replace=False
            )
        )
        target_tracks = self.load_target_tracks(index, target_inds.tolist(), dim=0)
        data["query_tracks_2d"] = query_tracks
        data["target_ts"] = target_inds
        data["target_w2cs"] = self.w2cs[target_inds]
        data["target_Ks"] = self.Ks[target_inds]
        data["target_tracks_2d"] = target_tracks[..., :2]

        (
            data["target_visibles"],
            data["target_invisibles"],
            data["target_confidences"],
        ) = parse_tapir_track_info(target_tracks[..., 2], target_tracks[..., 3])

        target_depths = torch.stack([self.get_depth(i) for i in target_inds], dim=0)
        H, W = target_depths.shape[-2:]
        data["target_track_depths"] = F.grid_sample(
            target_depths[:, None],
            normalize_coords(target_tracks[..., None, :2], H, W),
            align_corners=True,
            padding_mode="border",
        )[:, 0, :, 0]

        # Pseudo view data
        if self.use_pseudo_view:
            pseudo_target_inds = torch.from_numpy(
                np.random.choice(
                    self.num_frames,
                    (self.num_pseudo_targets_per_frame,),
                    replace=False
                )
            )

            try:
                all_pseudo_query_tracks = self.load_pseudo_target_tracks(index, [index])
                all_pseudo_target_tracks = self.load_pseudo_target_tracks(
                    index, pseudo_target_inds.tolist(), dim=0
                )
            except:
                all_pseudo_query_tracks = {seq: torch.zeros((0, 2)) for seq in self.pseudo_sequences}
                all_pseudo_target_tracks = {seq: torch.zeros((len(pseudo_target_inds), 0, 4)) for seq in self.pseudo_sequences}

            for seq_idx, seq_name in enumerate(self.pseudo_sequences):
                try:
                    pseudo_query_tracks = all_pseudo_query_tracks[seq_name][:, 0, :2]
                except (KeyError, IndexError):
                    pseudo_query_tracks = torch.zeros((0, 2))

                try:
                    pseudo_target_tracks = all_pseudo_target_tracks[seq_name]
                except KeyError:
                    pseudo_target_tracks = torch.zeros((len(pseudo_target_inds), 0, 4))

                if hasattr(self, 'pseudo_cameras') and seq_name in self.pseudo_cameras:
                    pseudo_w2cs = self.pseudo_cameras[seq_name]['w2cs'][pseudo_target_inds]
                    pseudo_Ks = self.pseudo_cameras[seq_name]['Ks'][pseudo_target_inds]
                    pseudo_current_w2c = self.pseudo_cameras[seq_name]['w2cs'][index]
                    pseudo_current_K = self.pseudo_cameras[seq_name]['Ks'][index]
                else:
                    guru.warning(f"Pseudo cameras not found for {seq_name}, using original cameras")
                    pseudo_w2cs = self.w2cs[pseudo_target_inds]
                    pseudo_Ks = self.Ks[pseudo_target_inds]
                    pseudo_current_w2c = self.w2cs[index]
                    pseudo_current_K = self.Ks[index]

                if pseudo_target_tracks.shape[1] > 0:
                    (
                        pseudo_visibles,
                        pseudo_invisibles,
                        pseudo_confidences
                    ) = parse_tapir_track_info(
                        pseudo_target_tracks[..., 2],
                        pseudo_target_tracks[..., 3]
                    )
                else:
                    pseudo_visibles = torch.zeros((len(pseudo_target_inds), 0), dtype=torch.bool)
                    pseudo_invisibles = torch.zeros((len(pseudo_target_inds), 0), dtype=torch.bool)
                    pseudo_confidences = torch.zeros((len(pseudo_target_inds), 0))

                seq_suffix = f"_{seq_idx}" if len(self.pseudo_sequences) > 1 else ""

                data.update({
                    f"pseudo_query_tracks_2d{seq_suffix}": pseudo_query_tracks,
                    f"pseudo_target_ts{seq_suffix}": pseudo_target_inds,
                    f"pseudo_target_w2cs{seq_suffix}": pseudo_w2cs,
                    f"pseudo_target_Ks{seq_suffix}": pseudo_Ks,
                    f"pseudo_target_tracks_2d{seq_suffix}": pseudo_target_tracks[..., :2],
                    f"pseudo_target_visibles{seq_suffix}": pseudo_visibles,
                    f"pseudo_target_invisibles{seq_suffix}": pseudo_invisibles,
                    f"pseudo_target_confidences{seq_suffix}": pseudo_confidences,
                    f"pseudo_sequence_name{seq_suffix}": seq_name,
                    f"pseudo_current_w2c{seq_suffix}": pseudo_current_w2c,
                    f"pseudo_current_K{seq_suffix}": pseudo_current_K,
                })

        return data


def load_cameras(
    path: str, H: int, W: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert os.path.exists(path), f"Camera file {path} does not exist."
    recon = np.load(path, allow_pickle=True).item()
    guru.debug(f"{recon.keys()=}")
    traj_c2w = recon["traj_c2w"]  # (N, 4, 4)
    h, w = recon["img_shape"]
    sy, sx = H / h, W / w
    traj_w2c = np.linalg.inv(traj_c2w)
    fx, fy, cx, cy = recon["intrinsics"]  # (4,)
    K = np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]])  # (3, 3)
    Ks = np.tile(K[None, ...], (len(traj_c2w), 1, 1))  # (N, 3, 3)
    kf_tstamps = recon["tstamps"].astype("int")
    return (
        torch.from_numpy(traj_w2c).float(),
        torch.from_numpy(Ks).float(),
        torch.from_numpy(kf_tstamps),
    )


def compute_scene_norm(
    X: torch.Tensor, w2cs: torch.Tensor
) -> tuple[float, torch.Tensor]:
    """
    :param X: [N*T, 3]
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
    d = CasualDataset(
        seq_name="bear",
        root_dir="/shared/vye/datasets/DAVIS",
        camera_type="droid_recon"
    )
