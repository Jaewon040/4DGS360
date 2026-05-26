# AnchorTAP3D

This code is based on [TAPIP3D](https://github.com/zbw001/TAPIP3D).

## Installation
Please refer to the setup instructions in [TAPIP3D](https://github.com/zbw001/TAPIP3D).


## Usage
```bash
python inference.py \
    --work_dir <path-to-4dgs360>/data/tapip3d/iphone360/jelly/ \
    --img_res 1 \
    --checkpoint checkpoints/tapip3d_final.pth \
    --iphone \
    --use_2dtrack \
    --track_vis_threshold 0.5 \
    --filter_2dtrack_with_mask \
    --prune_query 1000
```
Use `--use_support_grid` for an extra support grid.

## Visualization
```bash
python visualize.py \
    <path-to-4dgs360>/data/tapip3d/iphone360/jelly/ \
    --query_time 1
```
