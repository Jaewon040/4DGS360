import argparse
import json
import os.path as osp

import cv2
import imageio.v3 as iio
import numpy as np
import torch
from tqdm import tqdm

from flow3d.metrics import mLPIPS, mPSNR, mSSIM, CLIP
from flow3d.data.colmap import get_colmap_camera_params


parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir",
    type=str,
    help="Path to the data directory that contains all the sequences.",
)
parser.add_argument(
    "--result_dir",
    type=str,
    help="Path to the result directory that contains the results."
    "result_dir should contain results directly (result_dir/results)",
)

args = parser.parse_args()


def generate_background_masks(mask, val_names, threshold=0.8, kernel_size=(3, 3), erode_iterations=3):
    prefixes = list(set(int(name.split("_")[0]) for name in val_names))
    
    # Initialize masks for each unique prefix
    aggregated_masks = {prefix: np.zeros_like(mask[0], dtype=np.float64) for prefix in prefixes}

    # Aggregate masks based on the unique prefixs
    for i, name in enumerate(val_names):
        prefix = int(name.split("_")[0])
        aggregated_masks[prefix] += mask[i]
    
    processed_masks = {}
    kernel = np.ones(kernel_size, dtype=np.uint8)

    for prefix, agg_mask in aggregated_masks.items():
        binary_mask = (agg_mask > agg_mask.max() * threshold).astype(np.float64)
        eroded_mask = cv2.erode(binary_mask, kernel, iterations=erode_iterations)
        processed_masks[prefix] = eroded_mask
    
    # Update original masks using the processed masks
    for i, name in enumerate(val_names):
        prefix = int(name.split("_")[0])
        mask[i] = np.clip(mask[i] * processed_masks[prefix], 0., 255.0)

    return mask


def load_data_dict(data_dir, train_names, val_names, camera_type="original"):
    val_imgs = np.array(
        [iio.imread(osp.join(data_dir, "rgb/1x", f"{name}.png")) for name in val_names]
    )
    val_covisibles = np.array(
        [
            iio.imread(
                osp.join(
                    data_dir, "flow3d_preprocessed/covisible/1x/val/", f"{name}.png"
                )
            )
            for name in tqdm(val_names, desc="Loading val covisibles")
        ]
    )
    
    val_foregrounds = np.array(
        [
            iio.imread(
                osp.join(
                    data_dir, "flow3d_preprocessed/track_anything/1x", f"{name}.png"
                )
            )
            for name in tqdm(val_names, desc="Loading val foreground masks")
        ]
    )
    val_bkgd_masks= generate_background_masks(val_covisibles, val_names)

    kernel = np.ones((3,3), np.uint8)
    for i in range(len(val_names)):
        val_foregrounds[i] = cv2.dilate(val_foregrounds[i],kernel,  iterations=5)

    val_masks = val_bkgd_masks + val_foregrounds
    val_masks = np.clip(val_masks, 0., 255.0)
    val_masks = val_masks.astype(val_bkgd_masks.dtype)
    for i in range(len(val_names)):
        val_masks[i] = cv2.dilate(val_masks[i],kernel,  iterations=5)
        val_masks[i] = cv2.erode(val_masks[i],kernel,  iterations=5)
    
    val_times = np.array(
        [int(t.split("_")[1]) for t in val_names]
    )

    train_depths = np.array(
        [
            np.load(osp.join(data_dir, "depth/1x", f"{name}.npy"))[..., 0]
            for name in train_names
        ]
    )
    if camera_type == "original":
        train_Ks, train_w2cs = [], []
        for frame_name in train_names:
            with open(osp.join(data_dir, "camera", f"{frame_name}.json"), "r") as f:
                camera_dict = json.load(f)
            focal_length = camera_dict["focal_length"]
            principal_point = camera_dict["principal_point"]
            train_Ks.append(
                [
                    [focal_length, 0.0, principal_point[0]],
                    [0.0, focal_length, principal_point[1]],
                    [0.0, 0.0, 1.0],
                ]
            )
            orientation = np.array(camera_dict["orientation"])
            position = np.array(camera_dict["position"])
            train_w2cs.append(
                np.block(
                    [
                        [orientation, -orientation @ position[:, None]],
                        [np.zeros((1, 3)), np.ones((1, 1))],
                    ]
                ).astype(np.float32)
            )
    elif camera_type == "refined":
        train_Ks, train_w2cs = get_colmap_camera_params(

        )
    else:
        print("error")


    return {
        "val_imgs": val_imgs,
        "val_times": val_times,
        "val_masks": val_masks,
    }


def load_result_dict(result_dir, val_names):
    try:
        pred_val_imgs = np.array(
            [
                iio.imread(osp.join(result_dir, "rgb", f"{name}.png"))
                for name in val_names
            ]
        )
    except:
        pred_val_imgs = None
    

    return {
        "pred_val_imgs": pred_val_imgs,
    }


def evaluate_nv(data_dict, result_dict):
    device = "cuda"
    psnr_metric = mPSNR().to(device)
    ssim_metric = mSSIM().to(device)
    lpips_metric = mLPIPS().to(device)
    clip_metric = CLIP().to(device)
    clipt_metric = CLIP().to(device)

    val_imgs = torch.from_numpy(data_dict["val_imgs"])[..., :3].to(device)
    val_masks = torch.from_numpy(data_dict["val_masks"]).to(device)
    pred_val_imgs = torch.from_numpy(result_dict["pred_val_imgs"]).to(device)
    val_times = torch.from_numpy(data_dict["val_times"]).to(device)

    for i in tqdm(range(len(val_imgs))):
        val_img = val_imgs[i] / 255.0
        pred_val_img = pred_val_imgs[i] / 255.0
        val_mask = val_masks[i] / 255.0
        if i + 5 < len(val_imgs):
            if val_times[i+5] - val_times[i] == 5:
                next_pred_val_img = pred_val_imgs[i+5] / 255.0
                next_val_mask = val_masks[i+5] / 255.0
                # This is a bug, but we keep it unchanged to align with the results in the paper
                # clipt_metric.update(pred_val_img[None]*val_mask[None][...,None], next_pred_val_img[None]*next_val_mask[None][...,None])
                clipt_metric.update(pred_val_img[None]*val_mask[None][...,None], next_pred_val_img[None]*val_mask[None][...,None])
        clip_metric.update(val_img[None]*val_mask[None][...,None], pred_val_img[None]*val_mask[None][...,None])
        psnr_metric.update(val_img, pred_val_img, val_mask)
        ssim_metric.update(val_img[None], pred_val_img[None], val_mask[None])
        lpips_metric.update(val_img[None], pred_val_img[None], val_mask[None])
    
    mpsnr = psnr_metric.compute().item()
    mssim = ssim_metric.compute().item()
    mlpips = lpips_metric.compute().item()
    clip = clip_metric.compute().item()
    clipt = clipt_metric.compute().item()
    print(f"NV mPSNR: {mpsnr:.4f}")
    print(f"NV mSSIM: {mssim:.4f}")
    print(f"NV mLPIPS: {mlpips:.4f}")
    print(f"NV mCLIP-I: {clip:.4f}")
    print(f"NV mCLIP-T: {clipt:.4f}")
    return mpsnr, mssim, mlpips, clip, clipt


if __name__ == "__main__":
    seq_name = args.result_dir.split("/")[-2]

    print("=========================================")
    print(f"Evaluating {seq_name}")
    print("=========================================")
    data_dir = osp.join(args.data_dir, seq_name)
    if not osp.exists(data_dir):
        data_dir = args.data_dir
    if not osp.exists(data_dir):
        raise ValueError(f"Data directory {data_dir} not found.")
    result_dir = osp.join(args.result_dir, seq_name, "results/")
    if not osp.exists(result_dir):
        result_dir = osp.join(args.result_dir, "results/")
    if not osp.exists(result_dir):
        raise ValueError(f"Result directory {result_dir} not found.")

    with open(osp.join(data_dir, "splits/train.json")) as f:
        train_names = json.load(f)["frame_names"]
    with open(osp.join(data_dir, "splits/val.json")) as f:
        val_names = json.load(f)["frame_names"]

    data_dict = load_data_dict(data_dir, train_names, val_names)
    result_dict = load_result_dict(result_dir, val_names)
    print(f"Number of val images: {len(data_dict['val_imgs'])}")
    if len(data_dict["val_imgs"]) > 0:
        if result_dict["pred_val_imgs"] is None:
            print("No NV results found.")
        mpsnr, mssim, mlpips, clip, clipt= evaluate_nv(data_dict, result_dict)