import os
from dataclasses import asdict, dataclass
from typing import Annotated

import torch
import tyro
from loguru import logger as guru
from torch.utils.data import DataLoader

from flow3d.data import (
    BaseDataset,
    get_train_val_datasets,
    iPhoneDataConfig,
    NvidiaDataConfig,
    iPhoneTapip3dDataConfig
)
from flow3d.scene_model import SceneModel
from flow3d.validator import Validator
from run_training import set_seed
set_seed(42)


torch.set_float32_matmul_precision("high")


@dataclass
class TrainConfig:
    work_dir: str
    ckpt_path: str
    data: (
        Annotated[iPhoneDataConfig, tyro.conf.subcommand(name="iphone")]
        | Annotated[NvidiaDataConfig, tyro.conf.subcommand(name="nvidia")]
        | Annotated[iPhoneTapip3dDataConfig, tyro.conf.subcommand(name="iphone360")]
    )
    


def main(cfg: TrainConfig):
    _, train_video_view, val_img_dataset, val_kpt_dataset = (
        get_train_val_datasets(cfg.data, load_val=True)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(cfg.work_dir, exist_ok=True)

    # if checkpoint exists
    ckpt_path = cfg.ckpt_path
    guru.info(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path)
    state_dict = ckpt["model"]
    model = SceneModel.init_from_state_dict(state_dict)
    model = model.to(device)


    validator = None
    if (
        train_video_view is not None
        or val_img_dataset is not None
        or val_kpt_dataset is not None
    ):
        validator = Validator(
            model=model,
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

    with torch.no_grad():
        validator.validate()


if __name__ == "__main__":
    main(tyro.cli(TrainConfig))