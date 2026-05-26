python inference.py     \
    --work_dir ../../data/tapip3d/iphone360/jelly/     \
    --img_res 1     \
    --checkpoint checkpoints/tapip3d_final.pth \
    --iphone \
    --use_2dtrack \
    --track_vis_threshold 0.5 \
    --filter_2dtrack_with_mask \
    --prune_query 1000 
