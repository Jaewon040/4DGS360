import os
import time
from dataclasses import dataclass
from typing import Annotated

import numpy as np
import torch
import torch.nn.functional as F
import tyro
from loguru import logger as guru

from flow3d.renderer import Renderer
from flow3d.data import (
    iPhoneDataConfig,
    NvidiaDataConfig,
    iPhoneTapip3dDataConfig,
    CustomDataConfig,
    PanopticCustomDataConfig,
    TapipDataConfig,
)

torch.set_float32_matmul_precision("high")


@dataclass
class RenderConfig:
    ckpt_path: str
    port: int = 8890
    data: (
        Annotated[iPhoneDataConfig, tyro.conf.subcommand(name="iphone")]
        | Annotated[NvidiaDataConfig, tyro.conf.subcommand(name="nvidia")]
        | Annotated[iPhoneTapip3dDataConfig, tyro.conf.subcommand(name="iphone360")]
        | Annotated[CustomDataConfig, tyro.conf.subcommand(name="custom")]
        | Annotated[PanopticCustomDataConfig, tyro.conf.subcommand(name="mycasual")]
        | Annotated[TapipDataConfig, tyro.conf.subcommand(name="tapip3d")]
        | None
    ) = None
    show_cameras: bool = True


def main(cfg: RenderConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = cfg.ckpt_path
    assert os.path.exists(ckpt_path)

    renderer = Renderer.init_from_checkpoint(
        ckpt_path,
        device,
        work_dir="./",
        port=cfg.port,
    )

    guru.info(f"Starting rendering from {renderer.global_step=}")

    # Load datasets if data is provided
    train_dataset = None
    val_dataset = None
    if cfg.data is not None and cfg.show_cameras:
        from dataclasses import replace
        from flow3d.data import iPhoneDataset, iPhoneTapip3dDataset
        from dataclasses import asdict

        guru.info("Loading datasets for camera visualization...")

        # Try to load cached scene_norm_dict
        scene_norm_dict = None
        cache_dir = os.path.join(cfg.data.data_dir, "flow3d_preprocessed/cache")

        # Different cache file names for different dataset types
        if isinstance(cfg.data, iPhoneTapip3dDataConfig):
            scene_norm_dict_path = os.path.join(cache_dir, "scene_norm_dict_train.pth")
        else:
            scene_norm_dict_path = os.path.join(cache_dir, "scene_norm_dict.pth")

        if os.path.exists(scene_norm_dict_path):
            guru.info(f"Loading cached scene_norm_dict from {scene_norm_dict_path}")
            scene_norm_dict = torch.load(scene_norm_dict_path)
        else:
            guru.warning(f"No cached scene_norm_dict found at {scene_norm_dict_path}")
            guru.warning("Camera visualization will be skipped. Please run training first to generate cache.")

        if scene_norm_dict is not None:
            # Set skip_load_imgs and scene_norm_dict for camera-only loading
            data_cfg_with_skip = replace(
                cfg.data,
                skip_load_imgs=True,
                scene_norm_dict=scene_norm_dict
            )

            # Load train dataset
            try:
                if isinstance(cfg.data, (iPhoneDataConfig, NvidiaDataConfig)):
                    train_dataset = iPhoneDataset(**asdict(data_cfg_with_skip))
                elif isinstance(cfg.data, iPhoneTapip3dDataConfig):
                    train_dataset = iPhoneTapip3dDataset(**asdict(data_cfg_with_skip))
                elif isinstance(cfg.data, TapipDataConfig):
                    from flow3d.data import TapipDataset
                    train_dataset = TapipDataset(**asdict(data_cfg_with_skip))
                else:
                    guru.warning(f"Camera visualization not supported for {type(cfg.data).__name__}")
                    train_dataset = None

                if train_dataset is not None:
                    guru.info(f"Loaded train dataset with {train_dataset.num_frames} frames")

                    # Load val dataset if it exists
                    if train_dataset.has_validation:
                        guru.info("Loading validation dataset...")
                        val_cfg = replace(
                            data_cfg_with_skip,
                            split="val",
                            scene_norm_dict=train_dataset.scene_norm_dict,
                            load_from_cache=True
                        )

                        if isinstance(cfg.data, (iPhoneDataConfig, NvidiaDataConfig)):
                            val_dataset = iPhoneDataset(**asdict(val_cfg))
                        elif isinstance(cfg.data, iPhoneTapip3dDataConfig):
                            val_dataset = iPhoneTapip3dDataset(**asdict(val_cfg))
                        elif isinstance(cfg.data, TapipDataConfig):
                            val_dataset = TapipDataset(**asdict(val_cfg))

                        if val_dataset is not None:
                            guru.info(f"Loaded val dataset with {val_dataset.num_frames} frames")

                    # Set datasets for camera visualization
                    guru.info("Setting up camera visualizations...")
                    renderer.set_datasets(train_dataset, val_dataset)

            except Exception as e:
                guru.error(f"Failed to load datasets for camera visualization: {e}")
                guru.info("Continuing without camera visualization...")

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main(tyro.cli(RenderConfig))

