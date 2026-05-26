import argparse
import json
import os.path as osp

import imageio.v3 as iio
import numpy as np
import torch
from tqdm import tqdm

from flow3d.metrics import mLPIPS, mPSNR, mSSIM, CLIP


def compute_fg_bbox_with_margin(fg_mask, margin_ratio_w=1 / 8, margin_ratio_h=1 / 8):
    """Bounding box of fg_mask with margins relative to image dimensions."""
    if fg_mask.max() == 0:
        return None
    rows = np.any(fg_mask > 0, axis=1)
    cols = np.any(fg_mask > 0, axis=0)
    if not rows.any() or not cols.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    H, W = fg_mask.shape
    margin_h = int(H * margin_ratio_h)
    margin_w = int(W * margin_ratio_w)
    rmin = max(0, rmin - margin_h)
    rmax = min(H - 1, rmax + margin_h)
    cmin = max(0, cmin - margin_w)
    cmax = min(W - 1, cmax + margin_w)
    return (rmin, rmax, cmin, cmax)


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
parser.add_argument(
    "--per_frame_score",
    action="store_true",
    help="If set, save per-frame scores to per_frame_scores.json.",
)
args = parser.parse_args()


def load_data_dict(data_dir, val_names):
    val_imgs = np.array(
        [iio.imread(osp.join(data_dir, "rgb/1x", f"{name}.png")) for name in val_names]
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
    if val_foregrounds.ndim == 4 and val_foregrounds.shape[-1] == 3:
        val_foregrounds = val_foregrounds.max(axis=-1)

    val_times = np.array([int(t.split("_")[1]) for t in val_names])

    return {
        "val_imgs": val_imgs,
        "val_times": val_times,
        "val_fg_masks": val_foregrounds,
    }


def load_result_dict(result_dir, val_names):
    try:
        pred_fgonly_imgs = np.array(
            [
                iio.imread(osp.join(result_dir, "rgb_fgonly", f"{name}.png"))
                for name in val_names
            ]
        )
    except Exception:
        print("FG-only images not found, BBFG_LPIPS / BBFG_CLIP-I / BBFG_CLIP-T will be skipped")
        pred_fgonly_imgs = None

    return {"pred_fgonly_imgs": pred_fgonly_imgs}


def evaluate(data_dict, result_dict, val_names=None, per_frame_score=False):
    device = "cuda"

    # BBFG_LPIPS uses full-frame fg-only render vs GT with white background.
    bbfg_lpips_metric = mLPIPS().to(device)
    # BBFG_CLIP-I / BBFG_CLIP-T use bbox-cropped fg-only render.
    bbfg_clip_metric = CLIP().to(device)
    bbfg_clipt_metric = CLIP().to(device)

    val_imgs = torch.from_numpy(data_dict["val_imgs"])[..., :3].to(device)
    val_fg_masks = torch.from_numpy(data_dict["val_fg_masks"]).to(device)
    val_times = torch.from_numpy(data_dict["val_times"]).to(device)
    pred_fgonly_imgs = torch.from_numpy(result_dict["pred_fgonly_imgs"]).to(device)

    bbfg_crops_info = []
    bbfg_frame_indices = []

    for i in tqdm(range(len(val_imgs))):
        val_img = val_imgs[i] / 255.0
        val_fg_mask = val_fg_masks[i] / 255.0
        pred_fgonly_img = pred_fgonly_imgs[i] / 255.0

        # ----- BBFG_LPIPS (= old masked_lpips) -----
        fg_mask_white_bg = torch.where(
            val_fg_mask > 0.5, val_fg_mask, torch.ones_like(val_fg_mask)
        )
        gt_masked = (
            val_img * val_fg_mask[..., None] + (1 - val_fg_mask[..., None]) * 1.0
        )
        bbfg_lpips_metric.update(pred_fgonly_img[None], gt_masked[None], fg_mask_white_bg[None])

        # ----- BBFG_CLIP-I (bbox crop + fg-only render) -----
        fg_mask_np = val_fg_mask.cpu().numpy()
        bbox = compute_fg_bbox_with_margin(fg_mask_np, margin_ratio_w=1 / 8, margin_ratio_h=1 / 8)
        if bbox is None:
            continue
        rmin, rmax, cmin, cmax = bbox

        gt_crop = val_img[rmin : rmax + 1, cmin : cmax + 1]
        fg_mask_crop = val_fg_mask[rmin : rmax + 1, cmin : cmax + 1]
        pred_fgonly_crop = pred_fgonly_img[rmin : rmax + 1, cmin : cmax + 1]

        gt_crop_masked = (
            gt_crop * fg_mask_crop[..., None] + (1 - fg_mask_crop[..., None]) * 1.0
        )
        bbfg_clip_metric.update(pred_fgonly_crop[None], gt_crop_masked[None])

        bbfg_crops_info.append(
            {
                "idx": i,
                "t": val_times[i].item(),
                "pred_fgonly_crop": pred_fgonly_crop,
            }
        )
        bbfg_frame_indices.append(i)

    # ----- BBFG_CLIP-T (temporal: frame at t vs t+5) -----
    for i in range(len(bbfg_crops_info)):
        curr_info = bbfg_crops_info[i]
        curr_t = curr_info["t"]
        for j in range(i + 1, len(bbfg_crops_info)):
            next_info = bbfg_crops_info[j]
            next_t = next_info["t"]
            if next_t - curr_t == 5:
                bbfg_clipt_metric.update(
                    curr_info["pred_fgonly_crop"][None],
                    next_info["pred_fgonly_crop"][None],
                )
                break

    bbfg_lpips = bbfg_lpips_metric.compute().item()
    bbfg_clip = bbfg_clip_metric.compute().item()
    bbfg_clipt = bbfg_clipt_metric.compute().item()

    per_frame_scores = None
    if per_frame_score and val_names is not None:
        def _lpips_per_frame(metric):
            ss = torch.stack(metric.sum_scores).float()
            tot = torch.stack(metric.total).float()
            return (ss / tot.clamp(min=1)).tolist()

        def _clip_per_frame(metric):
            return torch.cat(metric.sum_scores).tolist()

        per_frame_scores = {}
        bbfg_lpips_pf = _lpips_per_frame(bbfg_lpips_metric)
        for i, name in enumerate(val_names):
            per_frame_scores[name] = {"BBFG_LPIPS": bbfg_lpips_pf[i]}

        bbfg_clip_pf = _clip_per_frame(bbfg_clip_metric)
        for j, frame_idx in enumerate(bbfg_frame_indices):
            per_frame_scores[val_names[frame_idx]]["BBFG_CLIP-I"] = bbfg_clip_pf[j]

    print(f"BBFG_LPIPS:  {bbfg_lpips:.4f}")
    print(f"BBFG_CLIP-I: {bbfg_clip:.4f}")
    print(f"BBFG_CLIP-T: {bbfg_clipt:.4f}")

    return bbfg_lpips, bbfg_clip, bbfg_clipt, per_frame_scores


def compute_simple_bbfg_metrics(result_dir, val_names):
    """BBFG_PSNR / BBFG_SSIM from pre-saved {result_dir}/rgb_bbox/{name}_gt.png and _render.png."""
    bbox_dir = osp.join(result_dir, "rgb_bbox")
    if not osp.exists(bbox_dir):
        print("rgb_bbox directory not found, BBFG_PSNR / BBFG_SSIM will be skipped.")
        return None, None, None

    psnr_metric = mPSNR().to("cuda")
    ssim_metric = mSSIM().to("cuda")
    valid_names = []

    for name in tqdm(val_names, desc="Computing BBFG_PSNR / BBFG_SSIM"):
        gt_path = osp.join(bbox_dir, f"{name}_gt.png")
        render_path = osp.join(bbox_dir, f"{name}_render.png")
        if not osp.exists(gt_path) or not osp.exists(render_path):
            continue
        gt = torch.from_numpy(iio.imread(gt_path)[..., :3]).float().to("cuda") / 255.0
        render = (
            torch.from_numpy(iio.imread(render_path)[..., :3]).float().to("cuda") / 255.0
        )
        psnr_metric.update(gt, render)
        ssim_metric.update(gt[None], render[None])
        valid_names.append(name)

    if not valid_names:
        print("No valid rgb_bbox image pairs found.")
        return None, None, None

    avg_psnr = psnr_metric.compute().item()
    avg_ssim = ssim_metric.compute().item()

    pf_psnr = (
        -10.0
        * torch.log(
            torch.stack(psnr_metric.sum_squared_error).float()
            / torch.stack(psnr_metric.total).float().clamp(min=1)
        )
        / np.log(10.0)
    ).tolist()
    pf_ssim = torch.cat(ssim_metric.similarity).tolist()
    per_frame = {
        name: {"BBFG_PSNR": p, "BBFG_SSIM": s}
        for name, p, s in zip(valid_names, pf_psnr, pf_ssim)
    }

    print(f"BBFG_PSNR:   {avg_psnr:.4f}")
    print(f"BBFG_SSIM:   {avg_ssim:.4f}")
    return avg_psnr, avg_ssim, per_frame


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

    with open(osp.join(data_dir, "splits/val.json")) as f:
        val_names = json.load(f)["frame_names"]

    metrics_json_path = osp.join(result_dir, "metrics.json")
    metrics_dict = {}
    per_frame_scores = None

    data_dict = load_data_dict(data_dir, val_names)
    result_dict = load_result_dict(result_dir, val_names)
    print(f"Number of val images: {len(data_dict['val_imgs'])}")

    if len(data_dict["val_imgs"]) > 0 and result_dict["pred_fgonly_imgs"] is not None:
        bbfg_lpips, bbfg_clip, bbfg_clipt, per_frame_scores = evaluate(
            data_dict,
            result_dict,
            val_names=val_names,
            per_frame_score=args.per_frame_score,
        )
        metrics_dict["BBFG_LPIPS"] = bbfg_lpips
        metrics_dict["BBFG_CLIP-I"] = bbfg_clip
        metrics_dict["BBFG_CLIP-T"] = bbfg_clipt

    simple_psnr, simple_ssim, simple_per_frame = compute_simple_bbfg_metrics(
        result_dir, val_names
    )
    if simple_psnr is not None:
        metrics_dict["BBFG_PSNR"] = simple_psnr
        metrics_dict["BBFG_SSIM"] = simple_ssim

    with open(metrics_json_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"\nSaved metrics to {metrics_json_path}")

    if args.per_frame_score:
        if per_frame_scores is None:
            per_frame_scores = {}
        if simple_per_frame is not None:
            for name, vals in simple_per_frame.items():
                per_frame_scores.setdefault(name, {}).update(vals)
        per_frame_json_path = osp.join(result_dir, "per_frame_scores.json")
        with open(per_frame_json_path, "w") as f:
            json.dump(per_frame_scores, f, indent=2)
        print(f"Saved per-frame scores to {per_frame_json_path}")
