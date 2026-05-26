import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Annotated, Callable

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
import tyro
import yaml
from loguru import logger as guru
from tqdm import tqdm

from flow3d.data import get_train_val_datasets, iPhoneDataConfig, iPhoneTapip3dDataConfig
from flow3d.renderer import Renderer
from flow3d.trajectories import (
    get_arc_w2cs,
    get_avg_w2c,
    get_lemniscate_w2cs,
    get_lookat,
    get_orbit_w2cs,
    get_spiral_w2cs,
    get_wander_w2cs,
)
from flow3d.vis.utils import make_video_divisble

torch.set_float32_matmul_precision("high")


@dataclass
class BaseTrajectoryConfig:
    num_frames: int = tyro.MISSING
    ref_t: int = -1
    _fn: tyro.conf.SuppressFixed[Callable] = tyro.MISSING

    def get_w2cs(self, **kwargs) -> torch.Tensor:
        cfg_kwargs = asdict(self)
        _fn = cfg_kwargs.pop("_fn")
        cfg_kwargs.update(kwargs)
        return _fn(**cfg_kwargs)


@dataclass
class ArcTrajectoryConfig(BaseTrajectoryConfig):
    num_frames: int = 250
    degree: float = 40
    _fn: tyro.conf.SuppressFixed[Callable] = get_arc_w2cs


@dataclass
class LemniscateTrajectoryConfig(BaseTrajectoryConfig):
    num_frames: int = 240
    degree: float = 15.0
    _fn: tyro.conf.SuppressFixed[Callable] = get_lemniscate_w2cs


@dataclass
class SpiralTrajectoryConfig(BaseTrajectoryConfig):
    num_frames: int = 180
    rads: float = 0.5
    zrate: float = 0.3
    rots: int = 3
    _fn: tyro.conf.SuppressFixed[Callable] = get_spiral_w2cs


@dataclass
class WanderTrajectoryConfig(BaseTrajectoryConfig):
    num_frames: int = 120
    _fn: tyro.conf.SuppressFixed[Callable] = get_wander_w2cs


@dataclass
class OrbitTrajectoryConfig(BaseTrajectoryConfig):
    """360-degree orbit around the object in one direction."""
    num_frames: int = 240
    _fn: tyro.conf.SuppressFixed[Callable] = get_orbit_w2cs


@dataclass
class FixedTrajectoryConfig(BaseTrajectoryConfig):
    _fn: tyro.conf.SuppressFixed[Callable] = lambda ref_w2c, **_: ref_w2c[None]


@dataclass
class CustomCameraTrajectoryConfig(BaseTrajectoryConfig):
    """Load camera from test_camera directory."""
    num_frames: int = -1  # Will be set to dataset num_frames
    camera_json: str = "1.json"  # filename in test_camera directory
    _fn: tyro.conf.SuppressFixed[Callable] = lambda **_: None  # Will be handled separately


@dataclass
class ValidationCameraTrajectoryConfig(BaseTrajectoryConfig):
    """Load cameras from validation split."""
    num_frames: int = -1  # Will be set to validation dataset num_frames
    _fn: tyro.conf.SuppressFixed[Callable] = lambda **_: None  # Will be handled separately


@dataclass
class TrainCameraTrajectoryConfig(BaseTrajectoryConfig):
    """Render from training camera viewpoints."""
    num_frames: int = -1  # Will be set to train dataset num_frames
    _fn: tyro.conf.SuppressFixed[Callable] = lambda **_: None  # Will be handled separately


@dataclass
class FreezeOrbitTrajectoryConfig(BaseTrajectoryConfig):
    """Phase 1: train cameras 0..freeze_t with matching time.
    Transition: 15 frames alpha-blend full->fg_only at freeze_t camera.
    Phase 2: fixed time=freeze_t, orbit 360 from train_camera[freeze_t]."""
    freeze_t: int = tyro.MISSING
    orbit_frames: int = 180
    transition_frames: int = 15
    zoom_out: float = 1.0  # scale camera distance from lookat (e.g. 1.05 = 5% further)
    num_frames: int = -1  # set automatically
    _fn: tyro.conf.SuppressFixed[Callable] = lambda **_: None  # handled separately


@dataclass
class BaseTimeConfig:
    _fn: tyro.conf.SuppressFixed[Callable] = tyro.MISSING

    def get_ts(self, **kwargs) -> torch.Tensor:
        cfg_kwargs = asdict(self)
        _fn = cfg_kwargs.pop("_fn")
        return _fn(**kwargs, **cfg_kwargs)


@dataclass
class ReplayTimeConfig(BaseTimeConfig):
    _fn: tyro.conf.SuppressFixed[Callable] = (
        lambda num_frames, traj_frames, device, **_: F.pad(
            torch.arange(num_frames, device=device)[30:traj_frames+30],
            (0, max(traj_frames - num_frames, 0)),
            value=num_frames - 1,
        )
    )


@dataclass
class FixedTimeConfig(BaseTimeConfig):
    t: int = 0
    _fn: tyro.conf.SuppressFixed[Callable] = (
        lambda t, num_frames, traj_frames, device, **_: torch.tensor(
            [min(t, num_frames - 1)], device=device
        ).expand(traj_frames)
    )


@dataclass
class AllTimeConfig(BaseTimeConfig):
    """Render all timesteps."""
    _fn: tyro.conf.SuppressFixed[Callable] = (
        lambda num_frames, traj_frames, device, **_: torch.arange(num_frames, device=device)
    )


@dataclass
class VideoConfig:
    work_dir: str
    data: (
        Annotated[
            iPhoneDataConfig,
            tyro.conf.subcommand(
                name="iphone",
                default=iPhoneDataConfig(
                    data_dir=tyro.MISSING,
                    load_from_cache=True,
                    skip_load_imgs=True,
                ),
            ),
        ]
        | Annotated[
            iPhoneTapip3dDataConfig,
            tyro.conf.subcommand(
                name="iphone-tapip3d",
                default=iPhoneTapip3dDataConfig(
                    data_dir=tyro.MISSING,
                    load_from_cache=True,
                    skip_load_imgs=True,
                ),
            ),
        ]
    )
    trajectory: (
        Annotated[ArcTrajectoryConfig, tyro.conf.subcommand(name="arc")]
        | Annotated[OrbitTrajectoryConfig, tyro.conf.subcommand(name="orbit")]
        | Annotated[LemniscateTrajectoryConfig, tyro.conf.subcommand(name="lemniscate")]
        | Annotated[SpiralTrajectoryConfig, tyro.conf.subcommand(name="spiral")]
        | Annotated[WanderTrajectoryConfig, tyro.conf.subcommand(name="wander")]
        | Annotated[FixedTrajectoryConfig, tyro.conf.subcommand(name="fixed")]
        | Annotated[CustomCameraTrajectoryConfig, tyro.conf.subcommand(name="custom")]
        | Annotated[ValidationCameraTrajectoryConfig, tyro.conf.subcommand(name="val")]
        | Annotated[TrainCameraTrajectoryConfig, tyro.conf.subcommand(name="train")]
        | Annotated[FreezeOrbitTrajectoryConfig, tyro.conf.subcommand(name="freeze-orbit")]
    )
    time: (
        Annotated[ReplayTimeConfig, tyro.conf.subcommand(name="replay")]
        | Annotated[FixedTimeConfig, tyro.conf.subcommand(name="fixed")]
        | Annotated[AllTimeConfig, tyro.conf.subcommand(name="all")]
    )
    fps: float = 15.0
    port: int = 8890
    fg_only: bool = False


def main(cfg: VideoConfig):
    train_dataset = get_train_val_datasets(cfg.data, load_val=False)[0]
    guru.info(f"Training dataset has {train_dataset.num_frames} frames")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = f"{cfg.work_dir}/checkpoints/last.ckpt"
    assert os.path.exists(ckpt_path)

    renderer = Renderer.init_from_checkpoint(
        ckpt_path,
        device,
        work_dir=cfg.work_dir,
        port=None,
    )
    assert train_dataset.num_frames == renderer.num_frames

    guru.info(f"Rendering video from {renderer.global_step=}")

    train_w2cs = train_dataset.get_w2cs().to(device)
    avg_w2c = get_avg_w2c(train_w2cs)
    # avg_w2c = train_w2cs[0]
    train_c2ws = torch.linalg.inv(train_w2cs)
    lookat = get_lookat(train_c2ws[:, :3, -1], train_c2ws[:, :3, 2])
    up = torch.tensor([0.0, 0.0, 1.0], device=device)
    K = train_dataset.get_Ks()[0].to(device)
    img_wh = train_dataset.get_img_wh()

    # get the radius of the bounding sphere of training cameras
    rc_train_c2ws = torch.einsum("ij,njk->nik", torch.linalg.inv(avg_w2c), train_c2ws)
    rc_pos = rc_train_c2ws[:, :3, -1]
    rads = (rc_pos.amax(0) - rc_pos.amin(0)) * 1

    # Handle freeze-orbit trajectory
    if isinstance(cfg.trajectory, FreezeOrbitTrajectoryConfig):
        freeze_t = cfg.trajectory.freeze_t
        orbit_frames = cfg.trajectory.orbit_frames
        transition_frames = cfg.trajectory.transition_frames
        train_Ks = train_dataset.get_Ks().to(device)
        ref_idx = min(freeze_t, train_dataset.num_frames - 1)
        ref_K = train_Ks[ref_idx]

        # Phase 1: train cameras 0..freeze_t (inclusive), time = 0..freeze_t
        # Apply zoom_out: scale each camera's distance from lookat
        zoom = cfg.trajectory.zoom_out
        phase1_c2ws = torch.linalg.inv(train_w2cs[:freeze_t + 1]).clone()
        phase1_c2ws[:, :3, 3] = lookat + zoom * (phase1_c2ws[:, :3, 3] - lookat)
        phase1_w2cs = torch.linalg.inv(phase1_c2ws)
        phase1_ts = torch.arange(freeze_t + 1, device=device)
        phase1_Ks = train_Ks[:freeze_t + 1]

        # Build orbit w2cs: rotate full c2w around up-axis through lookat.
        # At theta=0, camera is exactly train_camera[freeze_t] (with zoom applied).
        ref_w2c = phase1_w2cs[freeze_t]
        ref_c2w = torch.linalg.inv(ref_w2c)
        ref_pos = ref_c2w[:3, 3]
        ref_R = ref_c2w[:3, :3]
        import roma
        thetas = torch.linspace(0.0, torch.pi * 2.0, orbit_frames + 1, device=device)[:-1]
        rot_mats = roma.rotvec_to_rotmat(thetas[:, None] * up[None])  # (N, 3, 3)
        orbit_positions = lookat + torch.einsum("nij,j->ni", rot_mats, ref_pos - lookat)
        orbit_Rs = torch.einsum("nij,jk->nik", rot_mats, ref_R)
        orbit_c2ws = torch.eye(4, device=device)[None].expand(orbit_frames, -1, -1).clone()
        orbit_c2ws[:, :3, :3] = orbit_Rs
        orbit_c2ws[:, :3, 3] = orbit_positions
        orbit_w2cs = torch.linalg.inv(orbit_c2ws)

        # Transition: stay at orbit[0] camera (= train[freeze_t]), time = freeze_t
        trans_w2cs = orbit_w2cs[:1].expand(transition_frames, -1, -1)
        trans_ts = torch.full((transition_frames,), freeze_t, device=device)
        trans_Ks = ref_K[None].expand(transition_frames, -1, -1)

        # Phase 2: full orbit, time = freeze_t
        phase2_w2cs = orbit_w2cs
        phase2_ts = torch.full((orbit_frames,), freeze_t, device=device)
        phase2_Ks = ref_K[None].expand(orbit_frames, -1, -1)

        w2cs = torch.cat([phase1_w2cs, trans_w2cs, phase2_w2cs], dim=0)
        ts = torch.cat([phase1_ts, trans_ts, phase2_ts], dim=0)
        Ks_per_frame = torch.cat([phase1_Ks, trans_Ks, phase2_Ks], dim=0)
        cfg.trajectory.num_frames = len(w2cs)
        guru.info(
            f"freeze-orbit: {freeze_t + 1} train + {transition_frames} transition "
            f"+ {orbit_frames} orbit = {cfg.trajectory.num_frames} total"
        )

    # Handle train camera trajectory
    elif isinstance(cfg.trajectory, TrainCameraTrajectoryConfig):
        train_Ks = train_dataset.get_Ks().to(device)
        w2cs = train_w2cs
        Ks_per_frame = train_Ks
        cfg.trajectory.num_frames = train_dataset.num_frames
        guru.info(f"Using {cfg.trajectory.num_frames} train cameras")

    # Handle validation camera trajectory
    elif isinstance(cfg.trajectory, ValidationCameraTrajectoryConfig):
        guru.info("Loading validation cameras")
        # Load validation dataset to get validation cameras
        _, _, val_dataset, _ = get_train_val_datasets(cfg.data, load_val=True)
        if val_dataset is None:
            raise ValueError("No validation dataset found. Make sure validation split exists in splits/val.json")
        guru.info(f"Validation dataset has {val_dataset.num_frames} frames")

        # Get validation cameras and intrinsics
        val_w2cs = val_dataset.get_w2cs().to(device)
        val_Ks = val_dataset.get_Ks().to(device)

        # Use all validation cameras
        w2cs = val_w2cs
        K = val_Ks[0]  # Set K to first validation camera's intrinsics for img_wh compatibility
        Ks_per_frame = val_Ks  # Store per-frame intrinsics
        cfg.trajectory.num_frames = val_dataset.num_frames

        # For time config, we'll use the renderer's num_frames (training frames)
        # but render from validation camera viewpoints
        guru.info(f"Using {cfg.trajectory.num_frames} validation cameras")

    # Handle custom camera trajectory
    elif isinstance(cfg.trajectory, CustomCameraTrajectoryConfig):
        import json
        camera_json_path = os.path.join(cfg.data.data_dir, "test_camera", cfg.trajectory.camera_json)
        guru.info(f"Loading custom camera from {camera_json_path}")

        with open(camera_json_path) as f:
            camera_dict = json.load(f)

        # Parse camera parameters (same format as original iPhone camera)
        focal_length = camera_dict["focal_length"]
        principal_point = camera_dict["principal_point"]
        K_custom = torch.tensor([
            [focal_length, 0.0, principal_point[0]],
            [0.0, focal_length, principal_point[1]],
            [0.0, 0.0, 1.0],
        ], device=device, dtype=torch.float32)

        orientation = np.array(camera_dict["orientation"])
        position = np.array(camera_dict["position"])
        w2c_raw = torch.from_numpy(
            np.block([
                [orientation, -orientation @ position[:, None]],
                [np.zeros((1, 3)), np.ones((1, 1))],
            ]).astype(np.float32)
        ).to(device)

        # Apply scene normalization (same as in iphone_tapip3d_dataset.py)
        if hasattr(train_dataset, 'scene_norm_dict'):
            scale = train_dataset.scene_norm_dict["scale"]
            transform = train_dataset.scene_norm_dict["transfm"].to(device)
            w2c_normalized = torch.mm(w2c_raw, torch.linalg.inv(transform))
            w2c_normalized[:3, 3] /= scale
        else:
            w2c_normalized = w2c_raw

        # Repeat for all frames
        w2cs = w2c_normalized[None].repeat(renderer.num_frames, 1, 1)
        K = K_custom
        cfg.trajectory.num_frames = renderer.num_frames
    else:
        w2cs = cfg.trajectory.get_w2cs(
            ref_w2c=(
                avg_w2c
                if cfg.trajectory.ref_t < 0
                else train_w2cs[min(cfg.trajectory.ref_t, train_dataset.num_frames - 1)]
            ),
            lookat=lookat,
            up=up,
            focal_length=K[0, 0].item(),
            rads=rads,
        )

    if not isinstance(cfg.trajectory, FreezeOrbitTrajectoryConfig):
        ts = cfg.time.get_ts(
            num_frames=renderer.num_frames,
            traj_frames=cfg.trajectory.num_frames,
            device=device,
        )

    import viser.transforms as vt
    from flow3d.vis.utils import get_server

    server = get_server(port=8890)
    for i, train_w2c in enumerate(train_w2cs):
        train_c2w = torch.linalg.inv(train_w2c).cpu().numpy()
        server.scene.add_camera_frustum(
            f"/train_camera/{i:03d}",
            np.pi / 4,
            1.0,
            0.02,
            (0, 0, 0),
            wxyz=vt.SO3.from_matrix(train_c2w[:3, :3]).wxyz,
            position=train_c2w[:3, -1],
        )
    for i, w2c in enumerate(w2cs):
        c2w = torch.linalg.inv(w2c).cpu().numpy()
        server.scene.add_camera_frustum(
            f"/camera/{i:03d}",
            np.pi / 4,
            1.0,
            0.02,
            (255, 0, 0),
            wxyz=vt.SO3.from_matrix(c2w[:3, :3]).wxyz,
            position=c2w[:3, -1],
        )
        avg_c2w = torch.linalg.inv(avg_w2c).cpu().numpy()
        server.scene.add_camera_frustum(
            f"/ref_camera",
            np.pi / 4,
            1.0,
            0.02,
            (0, 0, 255),
            wxyz=vt.SO3.from_matrix(avg_c2w[:3, :3]).wxyz,
            position=avg_c2w[:3, -1],
        )
    # num_frames = len(train_w2cs)
    # w2cs = train_w2cs[:1].repeat(num_frames, 1, 1)
    # ts = torch.arange(num_frames, device=device)
    # assert len(w2cs) == len(ts)

    video = []
    # Check if we have per-frame intrinsics (validation cameras)
    use_per_frame_K = isinstance(cfg.trajectory, (ValidationCameraTrajectoryConfig, TrainCameraTrajectoryConfig, FreezeOrbitTrajectoryConfig))

    # Precompute freeze-orbit phase boundaries
    if isinstance(cfg.trajectory, FreezeOrbitTrajectoryConfig):
        _phase1_end = cfg.trajectory.freeze_t + 1          # exclusive
        _trans_end = _phase1_end + cfg.trajectory.transition_frames  # exclusive

    for idx, (w2c, t) in enumerate(zip(tqdm(w2cs), ts)):
        with torch.inference_mode():
            current_K = Ks_per_frame[idx] if use_per_frame_K else K
            if isinstance(cfg.trajectory, FreezeOrbitTrajectoryConfig):
                if idx < _phase1_end:
                    # Phase 1: full scene
                    img = renderer.model.render(int(t.item()), w2c[None], current_K[None], img_wh, fg_only=False)["img"][0]
                elif idx < _trans_end:
                    # Transition: alpha blend full → fg_only
                    alpha = (idx - _phase1_end) / max(cfg.trajectory.transition_frames - 1, 1)
                    img_full = renderer.model.render(int(t.item()), w2c[None], current_K[None], img_wh, fg_only=False)["img"][0]
                    img_fg = renderer.model.render(int(t.item()), w2c[None], current_K[None], img_wh, fg_only=cfg.fg_only)["img"][0]
                    img = (1 - alpha) * img_full + alpha * img_fg
                else:
                    # Phase 2: orbit
                    img = renderer.model.render(int(t.item()), w2c[None], current_K[None], img_wh, fg_only=cfg.fg_only)["img"][0]
            else:
                img = renderer.model.render(int(t.item()), w2c[None], current_K[None], img_wh, fg_only=cfg.fg_only)["img"][0]
        img = (img.cpu().numpy() * 255.0).astype(np.uint8)
        video.append(img)
    video = np.stack(video, 0)

    video_dir = f"{cfg.work_dir}/videos/{datetime.now().strftime('%Y-%m-%d-%H%M%S')}"
    frames_dir = f"{video_dir}/frames"
    os.makedirs(frames_dir, exist_ok=True)
    for idx, frame in enumerate(video):
        iio.imwrite(f"{frames_dir}/{idx:04d}.png", frame)
    iio.imwrite(f"{video_dir}/video.mp4", make_video_divisble(video), fps=cfg.fps)
    with open(f"{video_dir}/cfg.yaml", "w") as f:
        yaml.dump(asdict(cfg), f, default_flow_style=False)


if __name__ == "__main__":
    main(tyro.cli(VideoConfig))
