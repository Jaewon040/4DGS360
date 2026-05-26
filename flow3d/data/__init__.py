from dataclasses import asdict, replace

from torch.utils.data import Dataset

from .base_dataset import BaseDataset
from .iphone_dataset import (
    iPhoneDataConfig,
    NvidiaDataConfig,
    iPhoneDataset,
    iPhoneDatasetKeypointView,
    iPhoneDatasetVideoView,
)
from .iphone360_dataset import (
    iPhoneTapip3dDataConfig,
    iPhoneTapip3dDataset,
)
from .casual_dataset import (
    CustomDataConfig,
    PanopticCustomDataConfig,
    CasualDataset,
)
from .tapip_dataset import (
    TapipDataConfig,
    TapipDataset,
)


def get_train_val_datasets(
    data_cfg: iPhoneDataConfig | NvidiaDataConfig | iPhoneTapip3dDataConfig | CustomDataConfig | PanopticCustomDataConfig | TapipDataConfig, load_val: bool
) -> tuple[BaseDataset, Dataset | None, Dataset | None, Dataset | None]:
    train_video_view = None
    val_img_dataset = None
    val_kpt_dataset = None
    if isinstance(data_cfg, iPhoneDataConfig):
        train_dataset = iPhoneDataset(**asdict(data_cfg))
        train_video_view = iPhoneDatasetVideoView(train_dataset)
        if load_val:
            val_img_dataset = (
                iPhoneDataset(
                    **asdict(replace(data_cfg, split="val", load_from_cache=True))
                )
                if train_dataset.has_validation
                else None
            )
            val_kpt_dataset = iPhoneDatasetKeypointView(train_dataset)
    elif isinstance(data_cfg, NvidiaDataConfig):
        train_dataset = iPhoneDataset(**asdict(data_cfg))
        if load_val:
            val_img_dataset = (
                iPhoneDataset(
                    **asdict(replace(data_cfg, split="val", load_from_cache=True))
                )
                if train_dataset.has_validation
                else None
            )
    elif isinstance(data_cfg, iPhoneTapip3dDataConfig):
        train_dataset = iPhoneTapip3dDataset(**asdict(data_cfg))
        train_video_view = train_dataset.get_video_dataset()
        if load_val:
            val_img_dataset = (
                iPhoneTapip3dDataset(
                    **asdict(replace(data_cfg, split="val", scene_norm_dict=train_dataset.scene_norm_dict, load_from_cache=True))
                )
                if train_dataset.has_validation
                else None
            )
    elif isinstance(data_cfg, CustomDataConfig):
        train_dataset = CasualDataset(**asdict(data_cfg))
        if load_val:
            val_img_dataset = (
                CasualDataset(
                    **asdict(replace(data_cfg, split="val", load_from_cache=True))
                )
                if train_dataset.has_validation
                else None
            )
    elif isinstance(data_cfg, PanopticCustomDataConfig):
        train_dataset = CasualDataset(**asdict(data_cfg))
        if load_val:
            val_img_dataset = (
                CasualDataset(
                    **asdict(replace(data_cfg, split="val", load_from_cache=True))
                )
                if train_dataset.has_validation
                else None
            )
    elif isinstance(data_cfg, TapipDataConfig):
        train_dataset = TapipDataset(**asdict(data_cfg))
        if load_val:
            val_img_dataset = (
                TapipDataset(
                    **asdict(replace(data_cfg, split="val", scene_norm_dict=train_dataset.scene_norm_dict, load_from_cache=True))
                )
                if train_dataset.has_validation
                else None
            )
    else:
        raise ValueError(f"Unknown data config: {data_cfg}")
    return train_dataset, train_video_view, val_img_dataset, val_kpt_dataset
