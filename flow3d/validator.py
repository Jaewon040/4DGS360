import os
import os.path as osp
from typing import cast
import json

import imageio as iio
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger as guru
from pytorch_msssim import SSIM
from torch.utils.data import DataLoader
from tqdm import tqdm

from flow3d.data.utils import normalize_coords, to_device
from flow3d.metrics import PCK, mLPIPS, mPSNR, mSSIM, CLIP
from flow3d.scene_model import SceneModel
from flow3d.vis.utils import (
    apply_depth_colormap,
    make_video_divisble,
    plot_correspondences,
)


from .transforms import rt_to_mat4, homogenize_points, transform_rigid
import cv2


def compute_fg_bbox_with_margin(fg_mask, margin_ratio_w=1/8, margin_ratio_h=1/8):
    """
    Compute bounding box from foreground mask with generous margins.

    Args:
        fg_mask: (H, W) tensor or numpy array
        margin_ratio_w: margin ratio relative to image width (applied to left and right)
        margin_ratio_h: margin ratio relative to image height (applied to top and bottom)

    Returns:
        (rmin, rmax, cmin, cmax) tuple, or None if no foreground found
    """
    if isinstance(fg_mask, torch.Tensor):
        fg_mask_np = fg_mask.cpu().numpy()
    else:
        fg_mask_np = fg_mask

    # Find bounding box
    if fg_mask_np.max() == 0:
        return None

    rows = np.any(fg_mask_np > 0, axis=1)
    cols = np.any(fg_mask_np > 0, axis=0)

    if not rows.any() or not cols.any():
        return None

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Add margins based on image dimensions
    H, W = fg_mask_np.shape
    margin_h = int(H * margin_ratio_h)
    margin_w = int(W * margin_ratio_w)

    rmin = max(0, rmin - margin_h)
    rmax = min(H - 1, rmax + margin_h)
    cmin = max(0, cmin - margin_w)
    cmax = min(W - 1, cmax + margin_w)

    return (rmin, rmax, cmin, cmax)


class Validator:
    def __init__(
        self,
        model: SceneModel,
        device: torch.device,
        data_type: str,
        train_loader: DataLoader | None,
        val_img_loader: DataLoader | None,
        val_kpt_loader: DataLoader | None,
        save_dir: str,
    ):
        self.model = model
        self.device = device
        self.data_type = data_type
        self.train_loader = train_loader
        self.val_img_loader = val_img_loader
        self.val_kpt_loader = val_kpt_loader
        self.save_dir = save_dir
        self.has_bg = self.model.has_bg

        # metrics
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.psnr_metric = mPSNR()
        self.ssim_metric = mSSIM()
        self.lpips_metric = mLPIPS().to(device)
        self.fg_psnr_metric = mPSNR()
        self.fg_ssim_metric = mSSIM()
        self.fg_lpips_metric = mLPIPS().to(device)
        self.bg_psnr_metric = mPSNR()
        self.bg_ssim_metric = mSSIM()
        self.bg_lpips_metric = mLPIPS().to(device)
        self.clip_metric = CLIP().to(device)
        self.pck_metric = PCK()

        # New foreground-focused metrics
        self.masked_lpips_metric = mLPIPS().to(device)
        self.bbfg_psnr_metric = mPSNR()
        self.bbfg_ssim_metric = mSSIM()
        self.bbfg_lpips_metric = mLPIPS().to(device)
        self.bbfg_clip_metric = CLIP().to(device)
        self.bbfg_clipt_metric = CLIP().to(device)

    def _save_depth(self, depth: torch.Tensor, save_dir: str, frame_name: str,
                    acc: torch.Tensor | None = None):
        """Save depth map in raw (.npy) and colored (.png) formats."""
        os.makedirs(save_dir, exist_ok=True)

        # Save raw depth (H, W)
        np.save(
            osp.join(save_dir, f"{frame_name}.npy"),
            depth[0, ..., 0].cpu().numpy()
        )

        # Save colored depth visualization
        # Slice acc to match depth[0] shape: (H, W, 1)
        acc_sliced = acc[0] if acc is not None else None
        depth_colored = apply_depth_colormap(depth[0], acc=acc_sliced)
        iio.imwrite(
            osp.join(save_dir, f"{frame_name}.png"),
            (depth_colored.cpu().numpy() * 255).astype(np.uint8)
        )

    def reset_metrics(self):
        self.psnr_metric.reset()
        self.ssim_metric.reset()
        self.lpips_metric.reset()
        self.fg_psnr_metric.reset()
        self.fg_ssim_metric.reset()
        self.fg_lpips_metric.reset()
        self.bg_psnr_metric.reset()
        self.bg_ssim_metric.reset()
        self.bg_lpips_metric.reset()
        self.pck_metric.reset()
        self.clip_metric.reset()

        # Reset new metrics
        self.masked_lpips_metric.reset()
        self.bbfg_psnr_metric.reset()
        self.bbfg_ssim_metric.reset()
        self.bbfg_lpips_metric.reset()
        self.bbfg_clip_metric.reset()
        self.bbfg_clipt_metric.reset()

    @torch.no_grad()
    def validate(self):
        self.reset_metrics()
        metric_imgs = self.validate_imgs() or {}
        metric_kpts = self.validate_keypoints() or {}
        metric_train_imgs = self.validate_train_imgs() or {}
        return {**metric_imgs, **metric_kpts, **metric_train_imgs}

    @torch.no_grad()
    def validate_imgs(self, num_wandb_images=4):
        guru.info("rendering validation images...")
        if self.val_img_loader is None:
            return

        # Check if we should render only foreground gaussians or with white background
        val_fgonly = self.val_img_loader.dataset.is_val_fgonly()
        val_fgonly_bgwhite = self.val_img_loader.dataset.is_val_fgonly_bgwhite()

        if val_fgonly:
            guru.info("Rendering validation with foreground gaussians only (val_fgonly=True)")
        if val_fgonly_bgwhite:
            guru.info("Rendering validation with white background gaussians (val_fgonly_bgwhite=True)")

        # Calculate indices to sample evenly across the dataset
        total_frames = len(self.val_img_loader)
        if total_frames <= num_wandb_images:
            wandb_indices = set(range(total_frames))
        else:
            # Sample evenly across the full range
            wandb_indices = set(np.linspace(0, total_frames - 1, num_wandb_images, dtype=int).tolist())

        wandb_images = []
        wandb_bbox_images = []  # For bbox cropped results

        # Store all batches info for CLIP-T computation
        all_batches_info = []

        for idx, batch in enumerate(tqdm(self.val_img_loader, desc="render val images")):
            batch = to_device(batch, self.device)
            frame_name = batch["frame_names"][0]
            t = batch["ts"][0]
            # (1, 4, 4).
            w2c = batch["w2cs"]
            # (1, 3, 3).
            K = batch["Ks"]
            # (1, H, W, 3).
            img = batch["imgs"]
            # (1, H, W).
            valid_mask = batch.get(
                "valid_masks", torch.ones_like(batch["imgs"][..., 0])
            )
            # (1, H, W).
            fg_mask = batch.get("masks", None)

            bkgd_mask = batch.get(
                "bkgd_masks",
                torch.ones_like(valid_mask)[None],
            )
            W, H = img_wh = img[0].shape[-2::-1]
            rendered = self.model.render(t, w2c, K, img_wh, return_depth=True, return_mask=True,
                                        fg_only=val_fgonly, bg_white=val_fgonly_bgwhite)

            # Save depth for validation images
            depth_dir = osp.join(self.save_dir, "results", "depth")
            self._save_depth(rendered["depth"], depth_dir, frame_name, acc=rendered.get("acc"))

            # Compute metrics.

            if self.data_type in ("iphone", "iphone360"):
                valid_mask *= bkgd_mask
                fg_valid_mask = fg_mask*valid_mask
                valid_mask = torch.clamp(valid_mask + fg_valid_mask, min=0.0, max=1.0)
                bg_valid_mask = (1 - fg_mask) * valid_mask
                main_valid_mask = valid_mask if self.has_bg else fg_valid_mask
            elif self.data_type == "nvidia":
                valid_mask = rendered["acc"][..., 0]
                valid_mask = (valid_mask >= 0.5).float()
                valid_mask_np = valid_mask[0].cpu().numpy()
                kernel = np.ones((3, 3), np.uint8)
                valid_mask_np = cv2.dilate(valid_mask_np, kernel, iterations=1)
                valid_mask_np = cv2.erode(valid_mask_np, kernel, iterations=1)
                valid_mask = torch.tensor(valid_mask_np, device=valid_mask.device).unsqueeze(0)
                fg_valid_mask = rendered["mask"][..., 0]
                fg_valid_mask = (fg_valid_mask >= 0.5).float()
                fg_valid_mask_np = fg_valid_mask[0].cpu().numpy()
                kernel = np.ones((3, 3), np.uint8)
                fg_valid_mask_np = cv2.dilate(fg_valid_mask_np, kernel, iterations=1)
                fg_valid_mask_np = cv2.erode(fg_valid_mask_np, kernel, iterations=1)
                fg_valid_mask = torch.tensor(fg_valid_mask_np, device=fg_valid_mask.device).unsqueeze(0)
                bg_valid_mask = (1 - fg_valid_mask) * valid_mask
                main_valid_mask = valid_mask if self.has_bg else fg_valid_mask

            rendered["img"] = torch.clamp(rendered["img"], min=0., max=1.)
            self.psnr_metric.update(rendered["img"], img, main_valid_mask)
            self.ssim_metric.update(rendered["img"], img, main_valid_mask)
            self.lpips_metric.update(rendered["img"], img, main_valid_mask)
            self.clip_metric.update(rendered["img"]*main_valid_mask[...,None], img*main_valid_mask[..., None])

            if self.has_bg:
                self.fg_psnr_metric.update(rendered["img"], img, fg_valid_mask)
                self.fg_ssim_metric.update(rendered["img"], img, fg_valid_mask)
                self.fg_lpips_metric.update(rendered["img"], img, fg_valid_mask)

                self.bg_psnr_metric.update(rendered["img"], img, bg_valid_mask)
                self.bg_ssim_metric.update(rendered["img"], img, bg_valid_mask)
                self.bg_lpips_metric.update(rendered["img"], img, bg_valid_mask)

            # ===== New foreground-focused metrics =====

            # Initialize variables for scoping
            bbox = None
            rendered_fgonly = None
            rendered_fgonly_crop = None
            gt_crop_masked = None

            # 1. masked_lpips: GT with fg mask (bg→white) vs FG-only render (bg→white)
            if fg_mask is not None:
                # Render with fg_only=True (background will be white)
                rendered_fgonly = self.model.render(t, w2c, K, img_wh, return_depth=True, return_mask=False,
                                                   fg_only=True, bg_white=False)
                rendered_fgonly["img"] = torch.clamp(rendered_fgonly["img"], min=0., max=1.)

                # Save foreground-only depth
                depth_fgonly_dir = osp.join(self.save_dir, "results", "depth_fgonly")
                self._save_depth(rendered_fgonly["depth"], depth_fgonly_dir, frame_name)

                # Convert GT mask: foreground regions keep fg_mask value, background→1.0 (white)
                fg_mask_white_bg = torch.where(fg_mask > 0.5, fg_mask, torch.ones_like(fg_mask))

                # Apply mask: GT image with white background
                gt_masked = img * fg_mask[..., None] + (1 - fg_mask[..., None]) * 1.0

                # Compute masked LPIPS (both have white background now)
                self.masked_lpips_metric.update(rendered_fgonly["img"], gt_masked, fg_mask_white_bg)

            # 2. bbfg_* metrics: Bbox crop + FG-only rendering
            if fg_mask is not None and rendered_fgonly is not None:
                bbox = compute_fg_bbox_with_margin(fg_mask[0], margin_ratio_w=1/8, margin_ratio_h=1/8)

                if bbox is not None:
                    rmin, rmax, cmin, cmax = bbox

                    # Crop GT and mask
                    gt_crop = img[:, rmin:rmax+1, cmin:cmax+1, :]
                    fg_mask_crop = fg_mask[:, rmin:rmax+1, cmin:cmax+1]

                    # Render fg-only and crop
                    rendered_fgonly_crop = rendered_fgonly["img"][:, rmin:rmax+1, cmin:cmax+1, :]

                    # Save bbox-cropped depth
                    depth_fgonly_crop = rendered_fgonly["depth"][:, rmin:rmax+1, cmin:cmax+1, :]
                    depth_bbox_dir = osp.join(self.save_dir, "results", "depth_bbox")
                    self._save_depth(depth_fgonly_crop, depth_bbox_dir, frame_name)

                    # Create white background mask for cropped region
                    fg_mask_crop_white_bg = torch.where(fg_mask_crop > 0.5, fg_mask_crop, torch.ones_like(fg_mask_crop))

                    # Apply mask to GT (background→white)
                    gt_crop_masked = gt_crop * fg_mask_crop[..., None] + (1 - fg_mask_crop[..., None]) * 1.0

                    # Compute bbfg metrics
                    self.bbfg_psnr_metric.update(rendered_fgonly_crop, gt_crop_masked, fg_mask_crop_white_bg)
                    self.bbfg_ssim_metric.update(rendered_fgonly_crop, gt_crop_masked, fg_mask_crop_white_bg)
                    self.bbfg_lpips_metric.update(rendered_fgonly_crop, gt_crop_masked, fg_mask_crop_white_bg)
                    self.bbfg_clip_metric.update(rendered_fgonly_crop, gt_crop_masked)

                    # Store for CLIP-T computation and wandb visualization
                    all_batches_info.append({
                        "idx": idx,
                        "t": t.item(),
                        "rendered_fgonly_crop": rendered_fgonly_crop,
                        "gt_crop_masked": gt_crop_masked,
                        "fg_mask_crop": fg_mask_crop,
                        "frame_name": frame_name,
                    })

                    # Collect bbox images for wandb
                    if idx in wandb_indices:
                        gt_bbox_img = gt_crop_masked[0].cpu().numpy()
                        pred_bbox_img = rendered_fgonly_crop[0].cpu().numpy()
                        bbox_comparison = np.concatenate([gt_bbox_img, pred_bbox_img], axis=1)
                        wandb_bbox_images.append({
                            "frame": f"{frame_name}_bbox",
                            "image": bbox_comparison,
                        })

            # Dump results.
            results_dir = osp.join(self.save_dir, "results", "rgb")
            os.makedirs(results_dir, exist_ok=True)
            if self.data_type == "nvidia":
                # Combine RGB and alpha channel
                rgba_image = torch.cat(
                    [rendered["img"][0], rendered["acc"][0]], dim=-1
                ).cpu().numpy()
                iio.imwrite(
                    osp.join(results_dir, f"{frame_name}.png"),
                    (rgba_image * 255).astype(np.uint8),
                )
            else:
                iio.imwrite(
                    osp.join(results_dir, f"{frame_name}.png"),
                    (rendered["img"][0].cpu().numpy() * 255).astype(np.uint8),
                )

            # Save FG-only full image (for evaluate_iphone.py)
            if fg_mask is not None and rendered_fgonly is not None:
                fgonly_results_dir = osp.join(self.save_dir, "results", "rgb_fgonly")
                os.makedirs(fgonly_results_dir, exist_ok=True)
                iio.imwrite(
                    osp.join(fgonly_results_dir, f"{frame_name}.png"),
                    (rendered_fgonly["img"][0].cpu().numpy() * 255).astype(np.uint8),
                )

            # Save bbox crop results if available
            if fg_mask is not None and bbox is not None:
                bbox_results_dir = osp.join(self.save_dir, "results", "rgb_bbox")
                os.makedirs(bbox_results_dir, exist_ok=True)

                # Save rendered bbox crop
                iio.imwrite(
                    osp.join(bbox_results_dir, f"{frame_name}_render.png"),
                    (rendered_fgonly_crop[0].cpu().numpy() * 255).astype(np.uint8),
                )

                # Save GT bbox crop (with mask applied)
                iio.imwrite(
                    osp.join(bbox_results_dir, f"{frame_name}_gt.png"),
                    (gt_crop_masked[0].cpu().numpy() * 255).astype(np.uint8),
                )

                # Save comparison side-by-side
                bbox_comparison = np.concatenate([
                    (gt_crop_masked[0].cpu().numpy() * 255).astype(np.uint8),
                    (rendered_fgonly_crop[0].cpu().numpy() * 255).astype(np.uint8)
                ], axis=1)
                iio.imwrite(
                    osp.join(bbox_results_dir, f"{frame_name}_comparison.png"),
                    bbox_comparison,
                )

            # Collect images for wandb logging (sampled evenly across dataset)
            if idx in wandb_indices:
                gt_img = img[0].cpu().numpy()
                pred_img = rendered["img"][0].cpu().numpy()
                # Create side-by-side comparison
                comparison = np.concatenate([gt_img, pred_img], axis=1)
                wandb_images.append({
                    "frame": frame_name,
                    "image": comparison,
                })

        # ===== Compute CLIP-T (temporal consistency) =====
        # Compare frames with 5 timestep difference
        for i in range(len(all_batches_info)):
            curr_info = all_batches_info[i]
            curr_t = curr_info["t"]

            # Find frame with t+5
            for j in range(i+1, len(all_batches_info)):
                next_info = all_batches_info[j]
                next_t = next_info["t"]

                if next_t - curr_t == 5:
                    # Compare rendered images at t and t+5
                    self.bbfg_clipt_metric.update(
                        curr_info["rendered_fgonly_crop"],
                        next_info["rendered_fgonly_crop"]
                    )
                    break

        metrics = {
            "val/psnr": self.psnr_metric.compute(),
            "val/ssim": self.ssim_metric.compute(),
            "val/lpips": self.lpips_metric.compute(),
            "val/CLIP-I": self.clip_metric.compute(),
            "val/fg_psnr": self.fg_psnr_metric.compute(),
            "val/fg_ssim": self.fg_ssim_metric.compute(),
            "val/fg_lpips": self.fg_lpips_metric.compute(),
            "val/bg_psnr": self.bg_psnr_metric.compute(),
            "val/bg_ssim": self.bg_ssim_metric.compute(),
            "val/bg_lpips": self.bg_lpips_metric.compute(),
        }

        # Add new foreground-focused metrics
        metrics["val/masked_lpips"] = self.masked_lpips_metric.compute()
        metrics["val/bbfg_psnr"] = self.bbfg_psnr_metric.compute()
        metrics["val/bbfg_ssim"] = self.bbfg_ssim_metric.compute()
        metrics["val/bbfg_lpips"] = self.bbfg_lpips_metric.compute()
        metrics["val/bbfg_clip"] = self.bbfg_clip_metric.compute()
        metrics["val/bbfg_clipt"] = self.bbfg_clipt_metric.compute()

        # Add wandb images if any were collected
        if len(wandb_images) > 0:
            metrics["val/images"] = wandb_images
        if len(wandb_bbox_images) > 0:
            metrics["val/bbox_images"] = wandb_bbox_images

        return metrics

    @torch.no_grad()
    def validate_train_imgs(self, num_wandb_images=4, save_all=True):
        guru.info("rendering train images...")
        if self.train_loader is None:
            return

        # Calculate indices to sample evenly across the dataset for wandb
        total_frames = len(self.train_loader)
        if total_frames <= num_wandb_images:
            wandb_indices = set(range(total_frames))
        else:
            # Sample evenly across the full range
            wandb_indices = set(np.linspace(0, total_frames - 1, num_wandb_images, dtype=int).tolist())

        wandb_images = []

        # Create output directory for train images if saving all
        if save_all:
            train_results_dir = osp.join(self.save_dir, "results", "train_rgb")
            train_depth_dir = osp.join(self.save_dir, "results", "train_depth")
            os.makedirs(train_results_dir, exist_ok=True)
            os.makedirs(train_depth_dir, exist_ok=True)

        for idx, batch in enumerate(tqdm(self.train_loader, desc="render train images")):
            batch = to_device(batch, self.device)
            frame_name = batch["frame_names"][0] if "frame_names" in batch else f"frame_{idx:04d}"
            t = batch["ts"][0]
            # (1, 4, 4).
            w2c = batch["w2cs"]
            # (1, 3, 3).
            K = batch["Ks"]
            # (1, H, W, 3).
            img = batch["imgs"]

            W, H = img_wh = img[0].shape[-2::-1]
            rendered = self.model.render(t, w2c, K, img_wh, return_depth=True, return_mask=True)

            rendered["img"] = torch.clamp(rendered["img"], min=0., max=1.)

            # Save all train images if requested
            if save_all:
                if self.data_type == "nvidia":
                    # Combine RGB and alpha channel
                    rgba_image = torch.cat(
                        [rendered["img"][0], rendered["acc"][0]], dim=-1
                    ).cpu().numpy()
                    iio.imwrite(
                        osp.join(train_results_dir, f"{frame_name}.png"),
                        (rgba_image * 255).astype(np.uint8),
                    )
                else:
                    iio.imwrite(
                        osp.join(train_results_dir, f"{frame_name}.png"),
                        (rendered["img"][0].cpu().numpy() * 255).astype(np.uint8),
                    )

                # Save training depth
                self._save_depth(
                    rendered["depth"],
                    train_depth_dir,
                    frame_name,
                    acc=rendered.get("acc")
                )

            # Create wandb images for sampled indices
            if idx in wandb_indices:
                gt_img = img[0].cpu().numpy()
                pred_img = rendered["img"][0].cpu().numpy()
                # Create side-by-side comparison
                comparison = np.concatenate([gt_img, pred_img], axis=1)
                wandb_images.append({
                    "frame": frame_name,
                    "image": comparison,
                })

        # Return wandb images if any were collected
        if len(wandb_images) > 0:
            return {"train/images": wandb_images}

        return None

    @torch.no_grad()
    def validate_keypoints(self):
        if self.val_kpt_loader is None:
            return
        pred_keypoints_3d_all = []
        time_ids = self.val_kpt_loader.dataset.time_ids.tolist()
        h, w = self.val_kpt_loader.dataset.dataset.imgs.shape[1:3]
        pred_train_depths = np.zeros((len(time_ids), h, w))

        for batch in tqdm(self.val_kpt_loader, desc="render val keypoints"):
            batch = to_device(batch, self.device)
            # (2,).
            ts = batch["ts"][0]
            # (2, 4, 4).
            w2cs = batch["w2cs"][0]
            # (2, 3, 3).
            Ks = batch["Ks"][0]
            # (2, H, W, 3).
            imgs = batch["imgs"][0]
            # (2, P, 3).
            keypoints = batch["keypoints"][0]
            # (P,)
            keypoint_masks = (keypoints[..., -1] > 0.5).all(dim=0)
            src_keypoints, target_keypoints = keypoints[:, keypoint_masks, :2]
            W, H = img_wh = imgs.shape[-2:0:-1]
            rendered = self.model.render(
                ts[0].item(),
                w2cs[:1],
                Ks[:1],
                img_wh,
                target_ts=ts[1:],
                target_w2cs=w2cs[1:],
                return_depth=True,
            )
            pred_tracks_3d = rendered["tracks_3d"][0, ..., 0, :]
            pred_tracks_2d = torch.einsum("ij,hwj->hwi", Ks[1], pred_tracks_3d)
            pred_tracks_2d = pred_tracks_2d[..., :2] / torch.clamp(
                pred_tracks_2d[..., -1:], min=1e-6
            )
            pred_keypoints = F.grid_sample(
                pred_tracks_2d[None].permute(0, 3, 1, 2),
                normalize_coords(src_keypoints, H, W)[None, None],
                align_corners=True,
            ).permute(0, 2, 3, 1)[0, 0]

            # Compute metrics.
            self.pck_metric.update(pred_keypoints, target_keypoints, max(img_wh) * 0.05)

            padded_keypoints_3d = torch.zeros_like(keypoints[0])
            pred_keypoints_3d = F.grid_sample(
                pred_tracks_3d[None].permute(0, 3, 1, 2),
                normalize_coords(src_keypoints, H, W)[None, None],
                align_corners=True,
            ).permute(0, 2, 3, 1)[0, 0]
            # Transform 3D keypoints back to world space.
            pred_keypoints_3d = torch.einsum(
                "ij,pj->pi",
                torch.linalg.inv(w2cs[1])[:3],
                F.pad(pred_keypoints_3d, (0, 1), value=1.0),
            )
            padded_keypoints_3d[keypoint_masks] = pred_keypoints_3d
            # Cache predicted keypoints.
            pred_keypoints_3d_all.append(padded_keypoints_3d.cpu().numpy())
            pred_train_depths[time_ids.index(ts[0].item())] = (
                rendered["depth"][0, ..., 0].cpu().numpy()
            )

        # Dump unified results.
        all_Ks = self.val_kpt_loader.dataset.dataset.Ks
        all_w2cs = self.val_kpt_loader.dataset.dataset.w2cs

        keypoint_result_dict = {
            "Ks": all_Ks[time_ids].cpu().numpy(),
            "w2cs": all_w2cs[time_ids].cpu().numpy(),
            "pred_keypoints_3d": np.stack(pred_keypoints_3d_all, 0),
            "pred_train_depths": pred_train_depths,
        }

        results_dir = osp.join(self.save_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        np.savez(
            osp.join(results_dir, "keypoints.npz"),
            **keypoint_result_dict,
        )
        guru.info(
            f"Dumped keypoint results to {results_dir=} {keypoint_result_dict['pred_keypoints_3d'].shape=}"
        )
        pck_value = self.pck_metric.compute().item()
        pck_json_path = osp.join(results_dir, "pck.json")
        with open(pck_json_path, "w") as json_file:
            json.dump({"val/pck": pck_value}, json_file)
        guru.info(f"Saved PCK value to {pck_json_path}")

        return {"val/pck": pck_value}

    @torch.no_grad()
    def save_train_videos(self, epoch: int):
        if self.train_loader is None:
            return
        video_dir = osp.join(self.save_dir, "videos", f"epoch_{epoch:04d}")
        os.makedirs(video_dir, exist_ok=True)
        fps = getattr(self.train_loader.dataset.dataset, "fps", 15.0)
        # Render video.
        video = []
        ref_pred_depths = []
        masks = []
        depth_min, depth_max = 1e6, 0
        for batch_idx, batch in enumerate(
            tqdm(self.train_loader, desc="Rendering video", leave=False)
        ):
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            # ().
            t = batch["ts"][0]
            # (4, 4).
            w2c = batch["w2cs"][0]
            # (3, 3).
            K = batch["Ks"][0]
            # (H, W, 3).
            img = batch["imgs"][0]
            # (H, W).
            depth = batch["depths"][0]

            img_wh = img.shape[-2::-1]
            rendered = self.model.render(
                t, w2c[None], K[None], img_wh, return_depth=True, return_mask=True
            )
            # Putting results onto CPU since it will consume unnecessarily
            # large GPU memory for long sequence OW.
            video.append(torch.cat([img, rendered["img"][0]], dim=1).cpu())
            ref_pred_depth = torch.cat(
                (depth[..., None], rendered["depth"][0]), dim=1
            ).cpu()
            ref_pred_depths.append(ref_pred_depth)
            depth_min = min(depth_min, ref_pred_depth.min().item())
            depth_max = max(depth_max, ref_pred_depth.quantile(0.99).item())
            if rendered["mask"] is not None:
                masks.append(rendered["mask"][0].cpu().squeeze(-1))

        # rgb video
        video = torch.stack(video, dim=0)
        num_frames = video.shape[0]
        iio.mimwrite(
            osp.join(video_dir, "rgbs.mp4"),
            make_video_divisble((video.numpy() * 255).astype(np.uint8)),
            fps=fps,
        )
        # depth video
        depth_video = torch.stack(
            [
                apply_depth_colormap(
                    ref_pred_depth, near_plane=depth_min, far_plane=depth_max
                )
                for ref_pred_depth in ref_pred_depths
            ],
            dim=0,
        )
        iio.mimwrite(
            osp.join(video_dir, "depths.mp4"),
            make_video_divisble((depth_video.numpy() * 255).astype(np.uint8)),
            fps=fps,
        )
        if len(masks) > 0:
            # mask video
            mask_video = torch.stack(masks, dim=0)
            iio.mimwrite(
                osp.join(video_dir, "masks.mp4"),
                make_video_divisble((mask_video.numpy() * 255).astype(np.uint8)),
                fps=fps,
            )

        # Render 2D track video.
        tracks_2d, target_imgs = [], []
        sample_interval = 10
        batch0 = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in self.train_loader.dataset[0].items()
        }
        # ().
        t = batch0["ts"]
        # (4, 4).
        w2c = batch0["w2cs"]
        # (3, 3).
        K = batch0["Ks"]
        # (H, W, 3).
        img = batch0["imgs"]
        # (H, W).
        bool_mask = batch0["masks"] > 0.5
        img_wh = img.shape[-2::-1]
        for batch in tqdm(
            self.train_loader, desc="Rendering 2D track video", leave=False
        ):
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            # Putting results onto CPU since it will consume unnecessarily
            # large GPU memory for long sequence OW.
            # (1, H, W, 3).
            target_imgs.append(batch["imgs"].cpu())
            # (1,).
            target_ts = batch["ts"]
            # (1, 4, 4).
            target_w2cs = batch["w2cs"]
            # (1, 3, 3).
            target_Ks = batch["Ks"]
            rendered = self.model.render(
                t,
                w2c[None],
                K[None],
                img_wh,
                target_ts=target_ts,
                target_w2cs=target_w2cs,
            )
            pred_tracks_3d = rendered["tracks_3d"][0][
                ::sample_interval, ::sample_interval
            ][bool_mask[::sample_interval, ::sample_interval]].swapaxes(0, 1)
            pred_tracks_2d = torch.einsum("bij,bpj->bpi", target_Ks, pred_tracks_3d)
            pred_tracks_2d = pred_tracks_2d[..., :2] / torch.clamp(
                pred_tracks_2d[..., 2:], min=1e-6
            )
            tracks_2d.append(pred_tracks_2d.cpu())
        tracks_2d = torch.cat(tracks_2d, dim=0)
        target_imgs = torch.cat(target_imgs, dim=0)
        track_2d_video = plot_correspondences(
            target_imgs.numpy(),
            tracks_2d.numpy(),
            query_id=cast(int, t),
        )
        iio.mimwrite(
            osp.join(video_dir, "tracks_2d.mp4"),
            make_video_divisble(np.stack(track_2d_video, 0)),
            fps=fps,
        )

        #Render 2D first layer node video
        nodes_2d, target_imgs = [], []
        batch0 = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in self.train_loader.dataset[0].items()
        }
        # ().
        t = batch0["ts"]
        # (4, 4)
        w2c = batch0["w2cs"]
        # (3, 3)
        K = batch0["Ks"]
        # (H, W, 3)
        img = batch0["imgs"]
        # (H, W)
        bool_mask = batch0["masks"] > 0.5
        img_wh = img.shape[-2::-1]
        
        node_positions_cano = self.model.motion_tree.motion_nodes[0].get_positions()

        transfms = self.model.motion_tree.compute_node_world_transforms(torch.tensor([i for i in range(num_frames)]), level=0)

        node_positions_cano = node_positions_cano.unsqueeze(-2)
        transfms = rt_to_mat4(transfms[..., :3, :3], transfms[..., :3, -1])
        node_positions_cano = transform_rigid(homogenize_points(node_positions_cano), transfms)
        for batch in tqdm(
            self.train_loader, desc="Rendering 2d node video", leave=False
        ):
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            # Putting results onto CPU since it will ocnsume unnecessarily
            # large GPU memory for long sequence OW.
            # (1, H, W, 3)
            target_imgs.append(batch["imgs"].cpu())
            # (1, )
            target_ts = batch["ts"]
            # (1, 4, 4)
            target_w2cs = batch["w2cs"]
            # (1, 3, 3)
            target_Ks = batch["Ks"]
            
            #node at this timestep
            target_node_positions_cano = node_positions_cano[..., target_ts, :].squeeze(-2).unsqueeze(0)
            target_node_positions_cano = transform_rigid(target_node_positions_cano, target_w2cs)[..., :-1]
            target_node_2d = torch.einsum("bij, bpj -> bpi", target_Ks, target_node_positions_cano)
            target_node_2d = target_node_2d[..., :2] / torch.clamp(
                target_node_2d[..., 2:], min=1e-6
            )
            nodes_2d.append(target_node_2d.cpu())
        nodes_2d = torch.cat(nodes_2d, dim=0)
        target_imgs = torch.cat(target_imgs, dim=0)
        node_2d_video = plot_correspondences(
            target_imgs.numpy(),
            nodes_2d.numpy(),
            query_id=cast(int, t),
        )
        iio.mimwrite(
            osp.join(video_dir, "nodes_2d.mp4"),
            make_video_divisble(np.stack(node_2d_video, 0)),
            fps=fps
        )
