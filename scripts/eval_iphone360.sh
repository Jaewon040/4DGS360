#!/bin/bash

for scene in block2 goat jacket jelly pull-up walk-around; do
    PYTHONPATH="." python scripts/evaluate_iphone360.py \
        --data_dir ./data/iphone360/${scene}/ \
        --result_dir ./outputs/iphone360/${scene}
done