import argparse
import os
import os.path as osp
from glob import glob

import imageio.v2 as iio
import numpy as np
import torch
from tqdm import tqdm


UINT16_MAX = 65535

def align_monodepth_with_lidar_depth(
    metric_depth_dir: str,
    input_monodepth_dir: str,
    output_monodepth_dir: str,
    matching_pattern: str = "*",
):
    print(
        f"Aligning monodepth in {input_monodepth_dir} with lidar depth in {metric_depth_dir}"
    )
    mono_paths = sorted(glob(f"{input_monodepth_dir}/{matching_pattern}"))
    img_files = [osp.basename(p) for p in mono_paths]
    os.makedirs(output_monodepth_dir, exist_ok=True)
    print(output_monodepth_dir)
    if len(os.listdir(output_monodepth_dir)) == len(img_files):
        print(f"Founds {len(img_files)} files in {output_monodepth_dir}, skipping")
        return

    for f in tqdm(img_files):
        imname = os.path.splitext(f)[0]
        metric_path = osp.join(metric_depth_dir, imname + ".npy")
        mono_path = osp.join(input_monodepth_dir, imname + ".png")
        
        mono_disp_map = (iio.imread(mono_path) + 1e-8) / UINT16_MAX

        metric_disp_map = np.load(metric_path)[:,:,0]
        mask = metric_disp_map > 0
        metric_disp_map = 1.0 / np.clip(metric_disp_map, a_min=1e-6, a_max=1e6)
        
        mono_disp = mono_disp_map[mask]
        metric_disp = metric_disp_map[mask]

        ms_colmap_disp = metric_disp - np.median(metric_disp) + 1e-8
        ms_mono_disp = mono_disp - np.median(mono_disp) + 1e-8

        scale = np.median(ms_colmap_disp/ ms_mono_disp)
        shift = np.median(metric_disp - scale * mono_disp)

        aligned_disp = scale * mono_disp_map + shift

        min_thre = min(1e-6, np.quantile(aligned_disp, 0.01))
        # set depth values that are too small to invalid (0)
        aligned_disp[aligned_disp < min_thre] = 0.0
        min_thre = np.quantile(metric_disp,0.01)
        aligned_disp[aligned_disp < min_thre] = min_thre
        out_file = osp.join(output_monodepth_dir, imname + ".npy")
        np.save(out_file, aligned_disp)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lidar_depth_dir", 
        type=str, 
        default=None, 
        help="Input lidar depth dir")
    parser.add_argument(
        "--input_monodepth_dir", 
        type=str, default=None, 
        help="Input monodepth dir"
    )
    parser.add_argument(
        "--output_monodepth_dir",
        type=str,
        default=None,
        help="Output monodepth dir",
    )
    parser.add_argument(
        "--matching_pattern",
        type=str,
        default="*",
        help="Matching pattern for images to align",
    )
    args = parser.parse_args()
    align_monodepth_with_lidar_depth(
        args.lidar_depth_dir,
        args.input_monodepth_dir,
        args.output_monodepth_dir,
        args.matching_pattern,
    )

if __name__ == "__main__":
    main()