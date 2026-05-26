import os
import os.path as osp
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Annotated

import numpy as np
import torch
import tyro
import yaml
from loguru import logger as guru
from torch.utils.data import DataLoader
from tqdm import tqdm

from flow3d.configs import LossesConfig, OptimizerConfig, SceneLRConfig, load_yaml, update_dataclass

from flow3d.data import (
    BaseDataset,
    get_train_val_datasets,
    iPhoneDataConfig,
    NvidiaDataConfig,
    iPhoneTapip3dDataConfig,
    CustomDataConfig,
    PanopticCustomDataConfig,
    TapipDataConfig,
)
from flow3d.data.utils import to_device
from flow3d.init_utils import (
    init_bg,
    init_fg_from_tracks_3d,
    init_motion_params_with_procrustes,
    run_initial_optim,
    init_node_params_from_tracks_3d,
)
from flow3d.scene_model import SceneModel
from flow3d.tensor_dataclass import StaticObservations, TrackObservations
from flow3d.trainer import Trainer
from flow3d.validator import Validator

from flow3d.params import MotionTree, MotionBasesPerLevel, MotionNodesPerLevel, ParentIndicesPerLevel

# import wandb

torch.set_float32_matmul_precision("high")


def set_seed(seed):
    # Set the seed for generating random numbers
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_seed(42)


@dataclass
class TrainConfig:
    work_dir: str
    data: (
        Annotated[iPhoneDataConfig, tyro.conf.subcommand(name="iphone")]
        | Annotated[NvidiaDataConfig, tyro.conf.subcommand(name="nvidia")]
        | Annotated[iPhoneTapip3dDataConfig, tyro.conf.subcommand(name="iphone360")]
        | Annotated[CustomDataConfig, tyro.conf.subcommand(name="custom")]
        | Annotated[PanopticCustomDataConfig, tyro.conf.subcommand(name="mycasual")]
        | Annotated[TapipDataConfig, tyro.conf.subcommand(name="tapip3d")]
    )
    lr: SceneLRConfig
    loss: LossesConfig
    optim: OptimizerConfig
    num_fg: int = 50_000
    num_bg: int = 100_000
    num_init_nodes: int = 50
    num_nodes_second: int = 10
    num_motion_bases: int = 10
    num_motion_bases_second: int = 5
    num_epochs: int = 600
    port: int | None = None
    vis_debug: bool = False
    batch_size: int = 8
    num_dl_workers: int = 4
    validate_every: int = 200
    save_videos_every: int = 200
    wandb_project: str | None = None
    wandb_name: str | None = None
    invisible_weight: float = 0.5  # Weight for invisible points (0.0=ignore, 0.5=half, 1.0=equal to visible)
    debug_node_visibility: int = 0  # Debug node visibility every N steps (0=disabled)
    use_structural_loss: bool = False  # Enable structural loss for front/back distance preservation
    init_velocity_node: float = 0.0  # Velocity-based node initialization weight (0.0=spatial only, 0.5=balanced, 1.0=velocity only)


def main(cfg: TrainConfig):
    backup_code(cfg.work_dir)
    train_dataset, train_video_view, val_img_dataset, val_kpt_dataset = (
        get_train_val_datasets(cfg.data, load_val=True)
    )
    guru.info(f"Training dataset has {train_dataset.num_frames} frames")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if isinstance(cfg.data,NvidiaDataConfig):
        guru.info("Overriding configuration with Nvidia-specific settings")
        override_cfg = load_yaml("./flow3d/configs/nvidia.yaml")
        cfg.loss = update_dataclass(cfg.loss, override_cfg["LossesConfig"])
        cfg.optim = update_dataclass(cfg.optim, override_cfg["OptimizerConfig"])

    # Set debug_node_visibility from top-level config
    cfg.optim.debug_node_visibility = cfg.debug_node_visibility

    # Set structural loss weight if enabled
    if cfg.use_structural_loss:
        if cfg.loss.w_structural == 0.0:
            cfg.loss.w_structural = 1.0
            guru.info("Enabling structural loss with default weight 1.0")

    # save config
    os.makedirs(cfg.work_dir, exist_ok=True)
    with open(f"{cfg.work_dir}/cfg.yaml", "w") as f:
        yaml.dump(asdict(cfg), f, default_flow_style=False)

    # Initialize wandb if project name is provided
    writer = None
    if cfg.wandb_project is not None:
        import wandb
        writer = wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_name,
            dir=cfg.work_dir,
            config=asdict(cfg)
        )

    # Initialize model and save as init.ckpt
    init_ckpt_path = f"{cfg.work_dir}/checkpoints/init.ckpt"
    guru.info("Start initialize_and_checkpoint_model")
    initialize_and_checkpoint_model(
        cfg,
        train_dataset,
        device,
        init_ckpt_path,
        vis=cfg.vis_debug,
        port=cfg.port,
    )

    # Load from init.ckpt or last.ckpt if it exists
    last_ckpt_path = f"{cfg.work_dir}/checkpoints/last.ckpt"
    ckpt_path = last_ckpt_path if os.path.exists(last_ckpt_path) else init_ckpt_path

    trainer, start_epoch = Trainer.init_from_checkpoint(
        ckpt_path,
        device,
        cfg.lr,
        cfg.loss,
        cfg.optim,
        work_dir=cfg.work_dir,
        port=cfg.port,
        writer=writer,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_dl_workers,
        persistent_workers=True,
        shuffle=True,
        collate_fn=BaseDataset.train_collate_fn,
    )

    validator = None
    if (
        train_video_view is not None
        or val_img_dataset is not None
        or val_kpt_dataset is not None
    ):
        validator = Validator(
            model=trainer.model,
            device=device,
            data_type=cfg.data.data_type,
            train_loader=(
                DataLoader(train_video_view, batch_size=1) if train_video_view else None
            ),
            val_img_loader=(
                DataLoader(val_img_dataset, batch_size=1) if val_img_dataset else None
            ),
            val_kpt_loader=(
                DataLoader(val_kpt_dataset, batch_size=1) if val_kpt_dataset else None
            ),
            save_dir=cfg.work_dir,
        )

    guru.info(f"Starting training from {trainer.global_step=}")
    for epoch in (
        pbar := tqdm(
            range(start_epoch, cfg.num_epochs),
            initial=start_epoch,
            total=cfg.num_epochs,
        )
    ):
        trainer.set_epoch(epoch)
        for batch in train_loader:
            batch = to_device(batch, device)
            loss = trainer.train_step(batch)
            pbar.set_description(f"Loss: {loss:.6f}")

        with torch.no_grad():
            if validator is not None:
                if (epoch > 0 and epoch % cfg.validate_every == 0) or (
                    epoch == cfg.num_epochs - 1
                ):
                    val_logs = validator.validate()
                    trainer.log_dict(val_logs)
                    # Also save train images during validation
                    validator.validate_train_imgs()
                if (epoch > 0 and epoch % cfg.save_videos_every == 0) or (
                    epoch == cfg.num_epochs - 1):
                    validator.save_train_videos(epoch)

    if writer is not None:
        writer.finish()
        

def initialize_and_checkpoint_model(
    cfg: TrainConfig,
    train_dataset: BaseDataset,
    device: torch.device,
    ckpt_path: str,
    vis: bool = False,
    port: int | None = None,
):
    if os.path.exists(ckpt_path):
        guru.info(f"model checkpoint exists at {ckpt_path}")
        return

    fg_params, init_rots, init_ts, motion_coefs, bg_params, tracks_3d, cano_t = init_model_from_tracks(
        train_dataset,
        cfg.num_fg,
        cfg.num_bg,
        cfg.num_motion_bases,
        vis=vis,
        port=port,
        invisible_weight=cfg.invisible_weight,
        velocity_weight=cfg.init_velocity_node,
    )

    Ks = train_dataset.get_Ks().to(device)
    w2cs = train_dataset.get_w2cs().to(device)

    guru.info("Start run_initial_optim")
    run_initial_optim(fg_params, init_rots, init_ts, tracks_3d, motion_coefs, Ks, w2cs, num_iters=1000)
    guru.info("End run_initial_optim")


    # init motion bases
    rots, transls = init_rots, init_ts
    guru.info(f"motion bases rots.shape: {rots.shape}, transls.shape: {transls.shape}")

    # init motion nodes
    num_nodes = cfg.num_init_nodes
    node_means, node_motion_coefs, node_radius = init_node_params_from_tracks_3d(
        cano_t, num_nodes, tracks_3d, motion_coefs, velocity_weight=cfg.init_velocity_node
    )

    guru.info(f"node_means.shape: {node_means.shape}, node_motion_coefs.shape: {node_motion_coefs.shape}, node_radius.shape: {node_radius.shape}") 

    # params for MotionTree
    child_nodes_per_level = [cfg.num_init_nodes, cfg.num_nodes_second]
    motion_bases_per_level = [cfg.num_motion_bases, cfg.num_motion_bases_second]
    guru.info("motion_bases_per_level:", motion_bases_per_level)

    motion_tree = MotionTree(child_nodes_per_level, motion_bases_per_level)

    # generate first level bases, nodes and parent indices
    motion_bases = MotionBasesPerLevel(rots[None], transls[None]) # (1, 10, T, 6),  (1, 10, T, 3)
    motion_nodes = MotionNodesPerLevel(node_means, node_radius, node_motion_coefs)
    indices = torch.arange(1, device=device)[:, None].expand(-1, child_nodes_per_level[0]).reshape(-1) # e.g., [0, 0, 0, 1, 1, 1, ...]
    parent_indices = ParentIndicesPerLevel(indices=indices)

    # update
    motion_tree.set_init_tree([motion_bases], [motion_nodes], [parent_indices])

    model = SceneModel(Ks, w2cs, fg_params, motion_tree, bg_params)

    guru.info(f"Saving initialization to {ckpt_path}")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({"model": model.state_dict(), "epoch": 0, "global_step": 0}, ckpt_path)


def init_model_from_tracks(
    train_dataset,
    num_fg: int,
    num_bg: int,
    num_motion_bases: int,
    vis: bool = False,
    port: int | None = None,
    invisible_weight: float = 0.5,
    velocity_weight: float = 0.0,
):
    tracks_3d = TrackObservations(*train_dataset.get_tracks_3d(num_fg))
    # one track per gaussian. tracks_3d.shape[gaussian, t, 3]
    if not tracks_3d.check_sizes():
        import ipdb

        ipdb.set_trace()

    rot_type = "6d"
    cano_t = int(
        tracks_3d.visibles.sum(dim=0).argmax().item()
    )  # the time step that the most gaussian is visible
    guru.info(f"{cano_t=} {num_fg=} {num_bg=} {num_motion_bases=}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    init_rots, init_ts, motion_coefs, tracks_3d = init_motion_params_with_procrustes(
        tracks_3d, num_motion_bases, rot_type, cano_t, vis=vis, port=port, invisible_weight=invisible_weight
    )

    init_rots = init_rots.to(device)
    init_ts = init_ts.to(device)
    motion_coefs = motion_coefs.to(device)

    fg_params = init_fg_from_tracks_3d(cano_t, tracks_3d)
    fg_params = fg_params.to(device)

    bg_params = None
    if num_bg > 0:
        bg_points = StaticObservations(*train_dataset.get_bkgd_points(num_bg))
        assert bg_points.check_sizes()
        bg_params = init_bg(bg_points)
        bg_params = bg_params.to(device)

    tracks_3d = tracks_3d.to(device)
    return fg_params, init_rots, init_ts, motion_coefs, bg_params, tracks_3d, cano_t


def backup_code(work_dir):
    root_dir = osp.abspath(osp.join(osp.dirname(__file__)))
    tracked_dirs = [osp.join(root_dir, dirname) for dirname in ["flow3d", "scripts"]]
    dst_dir = osp.join(work_dir, "code", datetime.now().strftime("%Y-%m-%d-%H%M%S"))
    for tracked_dir in tracked_dirs:
        if osp.exists(tracked_dir):
            shutil.copytree(tracked_dir, osp.join(dst_dir, osp.basename(tracked_dir)))


if __name__ == "__main__":
    main(tyro.cli(TrainConfig))