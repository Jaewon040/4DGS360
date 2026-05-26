#!/bin/bash

python run_training.py \
    --work-dir ./outputs/iphone360/block2 \
    --port 8888 \
    data:iphone360 \
    --data.data-dir ./data/iphone360/block2/ \
    --data.camera_type original &&
python run_training.py \
    --work-dir ./outputs/iphone360/jacket \
    --port 8888 \
    data:iphone360 \
    --data.data-dir ./data/iphone360/jacket/ \
    --data.camera_type original &&\
python run_training.py \
    --work-dir ./outputs/iphone360/jelly \
    --port 8888 \
    --invisible_weight 1.0 \
    --loss.invisible_weight 0.5 \
    --use_structural_loss \
    --loss.structural_start_iter 5000 \
    --loss.w_structural 0.3 --loss.structural_patch_size 100 \
    data:iphone360 \
    --data.data-dir ./data/iphone360/jelly/ \
    --data.camera_type original &&\
python run_training.py \
    --work-dir ./outputs/iphone360/pull-up \
    --port 8888 \
    data:iphone360 \
    --data.data-dir ./data/iphone360/pull-up/ \
    --data.camera_type original &&\
python run_training.py \
    --work-dir ./outputs/iphone360/goat \
    --port 8888 \
    data:iphone360 \
    --data.data-dir ./data/iphone360/goat/ \
    --data.camera_type original &&\
python run_training.py \
    --work-dir ./outputs/iphone360/walk-around \
    --port 8888 \
    data:iphone360 \
    --data.data-dir ./data/iphone360/walk-around/ \
    --data.camera_type original 
