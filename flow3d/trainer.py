import functools
import time
from dataclasses import asdict
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger as guru
from nerfview import CameraState
from fused_ssim import fused_ssim
from lpips import LPIPS
from torch import nn
from torch.utils.tensorboard import SummaryWriter  # type: ignore

from flow3d.configs import LossesConfig, OptimizerConfig, SceneLRConfig
from flow3d.loss_utils import (
    compute_gradient_loss,
    compute_se3_smoothness_loss,
    compute_z_acc_loss,
    masked_l1_loss,
    compute_arap_loss,
    compute_radius_loss,
    compute_node_visibility,
    curve_distance_knn,
)
from flow3d.init_utils import knn_weighted_sampling
from flow3d.metrics import PCK, mLPIPS, mPSNR, mSSIM
from flow3d.scene_model import SceneModel
from flow3d.transforms import homogenize_points, transform_rigid, rt_to_mat4
from flow3d.vis.utils import get_server
from flow3d.vis.viewer import DynamicViewer


from einops import rearrange

class Trainer:
    def __init__(
        self,
        model: SceneModel,
        device: torch.device,
        lr_cfg: SceneLRConfig,
        losses_cfg: LossesConfig,
        optim_cfg: OptimizerConfig,
        # Logging.
        work_dir: str,
        port: int | None = None,
        writer: None = None,
        log_every: int = 10,
        checkpoint_every: int = 200,
        validate_every: int = 500,
        validate_video_every: int = 1000,
        validate_viewer_assets_every: int = 100,
    ):
        self.device = device
        self.log_every = log_every
        self.checkpoint_every = checkpoint_every
        self.validate_every = validate_every
        self.validate_video_every = validate_video_every
        self.validate_viewer_assets_every = validate_viewer_assets_every

        self.model = model
        self.num_frames = model.num_frames

        self.lr_cfg = lr_cfg
        self.losses_cfg = losses_cfg
        self.optim_cfg = optim_cfg

        self.reset_opacity_every = (
            self.optim_cfg.reset_opacity_every_n_controls * self.optim_cfg.control_every
        )
        self.optimizers, self.schedulers = self.configure_optimizers()

        # running stats for adaptive density control
        self.running_stats = {
            "xys_grad_norm_acc": torch.zeros(self.model.num_gaussians, device=device),
            "vis_count": torch.zeros(
                self.model.num_gaussians, device=device, dtype=torch.int64
            ),
            "max_radii": torch.zeros(self.model.num_gaussians, device=device),
        }

        self.work_dir = work_dir
        self.tb_writer = SummaryWriter(log_dir=work_dir)
        self.wandb_writer = writer

        self.global_step = 0
        self.epoch = 0

        self.viewer = None
        if port is not None:
            server = get_server(port=port)
            self.viewer = DynamicViewer(
                server, self.render_fn, model.num_frames, work_dir, mode="training"
            )

        # metrics
        self.lpips = LPIPS(net="alex", spatial=True).to(self.device)
        convert_to_buffer(self.lpips, persistent=False)
        self.psnr_metric = mPSNR()
        self.ssim_metric = mSSIM()
        self.lpips_metric = mLPIPS()
        self.pck_metric = PCK()
        self.bg_psnr_metric = mPSNR()
        self.fg_psnr_metric = mPSNR()
        self.bg_ssim_metric = mSSIM()
        self.fg_ssim_metric = mSSIM()
        self.bg_lpips_metric = mLPIPS()
        self.fg_lpips_metric = mLPIPS()

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def save_checkpoint(self, path: str):
        model_dict = self.model.state_dict()
        optimizer_dict = {k: v.state_dict() for k, v in self.optimizers.items()}
        scheduler_dict = {k: v.state_dict() for k, v in self.schedulers.items()}
        ckpt = {
            "model": model_dict,
            "optimizers": optimizer_dict,
            "schedulers": scheduler_dict,
            "global_step": self.global_step,
            "epoch": self.epoch,
        }
        torch.save(ckpt, path)
        guru.info(f"Saved checkpoint at {self.global_step=} to {path}")

    @staticmethod
    def init_from_checkpoint(
        path: str, device: torch.device, *args, **kwargs
    ) -> tuple["Trainer", int]:
        guru.info(f"Loading checkpoint from {path}")
        ckpt = torch.load(path)
        state_dict = ckpt["model"]

        model = SceneModel.init_from_state_dict(state_dict)
        model = model.to(device)
        trainer = Trainer(model, device, *args, **kwargs)
        if "optimizers" in ckpt:
            trainer.load_checkpoint_optimizers(ckpt["optimizers"])
        if "schedulers" in ckpt:
            trainer.load_checkpoint_schedulers(ckpt["schedulers"])
        trainer.global_step = ckpt.get("global_step", 0)
        start_epoch = ckpt.get("epoch", 0)
        trainer.set_epoch(start_epoch)
        return trainer, start_epoch

    def load_checkpoint_optimizers(self, opt_ckpt):
        for k, v in self.optimizers.items():
            v.load_state_dict(opt_ckpt[k])

    def load_checkpoint_schedulers(self, sched_ckpt):
        for k, v in self.schedulers.items():
            v.load_state_dict(sched_ckpt[k])

    @torch.inference_mode()
    def render_fn(self, camera_state: CameraState, img_wh: tuple[int, int]):
        W, H = img_wh

        focal = 0.5 * H / np.tan(0.5 * camera_state.fov).item()
        K = torch.tensor(
            [[focal, 0.0, W / 2.0], [0.0, focal, H / 2.0], [0.0, 0.0, 1.0]],
            device=self.device,
        )
        w2c = torch.linalg.inv(
            torch.from_numpy(camera_state.c2w.astype(np.float32)).to(self.device)
        )
        t = 0
        if self.viewer is not None:
            t = (
                int(self.viewer._playback_guis[0].value)
                if not self.viewer._canonical_checkbox.value
                else None
            )
        self.model.training = False
        img = self.model.render(t, w2c[None], K[None], img_wh)["img"][0]
        img = torch.clamp(img, min=0., max=1.)
        return (img.cpu().numpy() * 255.0).astype(np.uint8)

    def train_step(self, batch):
        if self.viewer is not None:
            while self.viewer.state.status == "paused":
            # while self.viewer.state == "paused":
                time.sleep(0.1)
            self.viewer.lock.acquire()

        loss, stats, num_rays_per_step, num_rays_per_sec= self.compute_losses(batch)
        if loss.isnan():
            guru.info(f"Loss is NaN at step {self.global_step}!!")
            import ipdb

            ipdb.set_trace()
        loss.backward()


        for opt in self.optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        for sched in self.schedulers.values():
            sched.step()


        self.log_dict(stats)
        self.global_step += 1
        self.run_control_steps()

        # Debug node visibility if enabled
        if self.optim_cfg.debug_node_visibility > 0:
            if self.global_step % self.optim_cfg.debug_node_visibility == 0:
                self._debug_node_visibility(batch)

        if self.viewer is not None:
            self.viewer.lock.release()
            self.viewer.state.num_train_rays_per_sec = num_rays_per_sec
            if self.viewer.mode == "training":
                self.viewer.update(self.global_step, num_rays_per_step)

        if self.global_step % self.checkpoint_every == 0:
            self.save_checkpoint(f"{self.work_dir}/checkpoints/last.ckpt")

        return loss.item()

    def compute_losses(self, batch):
        self.model.training = True
        B = batch["imgs"].shape[0]
        W, H = img_wh = batch["imgs"].shape[2:0:-1]
        N = batch["target_ts"][0].shape[0]

        # (B,).
        ts = batch["ts"]
        # (B, 4, 4).
        w2cs = batch["w2cs"]
        # (B, 3, 3).
        Ks = batch["Ks"]
        # (B, H, W, 3).
        imgs = batch["imgs"]
        # (B, H, W).
        valid_masks = batch.get("valid_masks", torch.ones_like(batch["imgs"][..., 0]))
        # (B, H, W).
        masks = batch["masks"]
        masks *= valid_masks
        # (B, H, W).
        depths = batch["depths"]
        # [(P, 2), ...].
        query_tracks_2d = batch["query_tracks_2d"]
        # [(N,), ...].
        target_ts = batch["target_ts"]
        # [(N, 4, 4), ...].
        target_w2cs = batch["target_w2cs"]
        # [(N, 3, 3), ...].
        target_Ks = batch["target_Ks"]
        # [(N, P, 2), ...].
        target_tracks_2d = batch["target_tracks_2d"]
        # [(N, P), ...].
        target_visibles = batch["target_visibles"]
        # [(N, P), ...].
        target_invisibles = batch["target_invisibles"]
        # [(N, P), ...].
        target_confidences = batch["target_confidences"]
        # [(N, P), ...].
        target_track_depths = batch["target_track_depths"]

        _tic = time.time()
        # (B, G, 3).
        means, quats = self.model.compute_poses_all(ts)  # (G, B, 3), (G, B, 4)
        device = means.device
        means = means.transpose(0, 1)
        quats = quats.transpose(0, 1)
        # [(N, G, 3), ...].
        target_ts_vec = torch.cat(target_ts)
        # (B * N, G, 3).
        target_means, _ = self.model.compute_poses_all(target_ts_vec)
        target_means = target_means.transpose(0, 1)
        target_mean_list = target_means.split(N)
        num_frames = self.model.num_frames

        loss = 0.0

        bg_colors = []
        rendered_all = []
        self._batched_xys = []
        self._batched_radii = []
        self._batched_img_wh = []
        for i in range(B):
            bg_color = torch.ones(1, 3, device=device)
            rendered = self.model.render(
                ts[i].item(),
                w2cs[None, i],
                Ks[None, i],
                img_wh,
                target_ts=target_ts[i],
                target_w2cs=target_w2cs[i],
                bg_color=bg_color,
                means=means[i],
                quats=quats[i],
                target_means=target_mean_list[i].transpose(0, 1),
                return_depth=True,
                return_mask=self.model.has_bg,
            )
            rendered_all.append(rendered)
            bg_colors.append(bg_color)
            if (
                self.model._current_xys is not None
                and self.model._current_radii is not None
                and self.model._current_img_wh is not None
            ):
                self._batched_xys.append(self.model._current_xys)
                self._batched_radii.append(self.model._current_radii)
                self._batched_img_wh.append(self.model._current_img_wh)

        # Necessary to make viewer work.
        num_rays_per_step = H * W * B
        num_rays_per_sec = num_rays_per_step / (time.time() - _tic)

        # (B, H, W, N, *).
        rendered_all = {
            key: (
                torch.cat([out_dict[key] for out_dict in rendered_all], dim=0)
                if rendered_all[0][key] is not None
                else None
            )
            for key in rendered_all[0]
        }
        bg_colors = torch.cat(bg_colors, dim=0)

        # Compute losses.
        # (B * N).
        frame_intervals = (ts.repeat_interleave(N) - target_ts_vec).abs()
        if not self.model.has_bg:
            imgs = (
                imgs * masks[..., None]
                + (1.0 - masks[..., None]) * bg_colors[:, None, None]
            )
        else:
            imgs = (
                imgs * valid_masks[..., None]
                + (1.0 - valid_masks[..., None]) * bg_colors[:, None, None]
            )
        # (P_all, 2).
        tracks_2d = torch.cat([x.reshape(-1, 2) for x in target_tracks_2d], dim=0)
        # (P_all,)
        visibles = torch.cat([x.reshape(-1) for x in target_visibles], dim=0)
        # (P_all,)
        confidences = torch.cat([x.reshape(-1) for x in target_confidences], dim=0)

        # RGB loss.
        rendered_imgs = cast(torch.Tensor, rendered_all["img"])

        # Use separate FG/BG RGB loss if flag is enabled
        if self.losses_cfg.rgb_loss_fg_bg and self.model.has_bg:
            # Separate FG and BG learning
            # FG: only learn inside mask region
            # BG: only learn outside mask region
            fg_region_mask = masks[..., None]  # (B, H, W, 1)
            bg_region_mask = (1.0 - masks[..., None]) * valid_masks[..., None]  # (B, H, W, 1)

            # FG RGB loss - only inside mask
            fg_rendered = rendered_imgs * fg_region_mask
            fg_target = imgs * fg_region_mask

            # BG RGB loss - only outside mask
            bg_rendered = rendered_imgs * bg_region_mask
            bg_target = imgs * bg_region_mask

            if self.global_step < self.optim_cfg.stop_control_steps:
                # FG loss
                if fg_region_mask.sum() > 0:
                    fg_rgb_loss = 0.8 * F.l1_loss(fg_rendered, fg_target) + 0.2 * (
                        1 - fused_ssim(
                            fg_rendered.permute(0, 3, 1, 2),
                            fg_target.permute(0, 3, 1, 2),
                            padding="valid"
                        )
                    )
                else:
                    fg_rgb_loss = torch.tensor(0.0, device=device)

                # BG loss
                if bg_region_mask.sum() > 0:
                    bg_rgb_loss = 0.8 * F.l1_loss(bg_rendered, bg_target) + 0.2 * (
                        1 - fused_ssim(
                            bg_rendered.permute(0, 3, 1, 2),
                            bg_target.permute(0, 3, 1, 2),
                            padding="valid"
                        )
                    )
                else:
                    bg_rgb_loss = torch.tensor(0.0, device=device)

                rgb_loss = fg_rgb_loss + bg_rgb_loss
                lpips_loss = torch.zeros_like(rgb_loss)
            else:
                # FG loss with LPIPS
                if fg_region_mask.sum() > 0:
                    fg_lpips = self.lpips.forward(
                        F.interpolate(fg_rendered.permute(0, 3, 1, 2), scale_factor=0.25, mode="bilinear", align_corners=True),
                        F.interpolate(fg_target.permute(0, 3, 1, 2), scale_factor=0.25, mode="bilinear", align_corners=True),
                        normalize=True
                    ).mean()
                    fg_rgb_loss = 0.8 * F.l1_loss(fg_rendered, fg_target) + 0.2 * (
                        1 - fused_ssim(
                            fg_rendered.permute(0, 3, 1, 2),
                            fg_target.permute(0, 3, 1, 2),
                            padding="valid"
                        )
                    ) + 0.01 * fg_lpips
                else:
                    fg_rgb_loss = torch.tensor(0.0, device=device)
                    fg_lpips = torch.tensor(0.0, device=device)

                # BG loss with LPIPS
                if bg_region_mask.sum() > 0:
                    bg_lpips = self.lpips.forward(
                        F.interpolate(bg_rendered.permute(0, 3, 1, 2), scale_factor=0.25, mode="bilinear", align_corners=True),
                        F.interpolate(bg_target.permute(0, 3, 1, 2), scale_factor=0.25, mode="bilinear", align_corners=True),
                        normalize=True
                    ).mean()
                    bg_rgb_loss = 0.8 * F.l1_loss(bg_rendered, bg_target) + 0.2 * (
                        1 - fused_ssim(
                            bg_rendered.permute(0, 3, 1, 2),
                            bg_target.permute(0, 3, 1, 2),
                            padding="valid"
                        )
                    ) + 0.01 * bg_lpips
                else:
                    bg_rgb_loss = torch.tensor(0.0, device=device)
                    bg_lpips = torch.tensor(0.0, device=device)

                rgb_loss = fg_rgb_loss + bg_rgb_loss
                lpips_loss = fg_lpips + bg_lpips
        else:
            # Original RGB loss (applies to all pixels)
            if self.model.has_bg:
                rendered_imgs = (
                    rendered_imgs * valid_masks[..., None]
                    + (1.0 - valid_masks[..., None]) * bg_colors[:, None, None]
                )

            if self.global_step < self.optim_cfg.stop_control_steps:
                rgb_loss = 0.8 * F.l1_loss(rendered_imgs, imgs) + 0.2 * (
                    1 - fused_ssim(rendered_imgs.permute(0, 3, 1, 2), imgs.permute(0, 3, 1, 2), padding="valid")
                )
                lpips_loss = torch.zeros_like(rgb_loss)
            else:
                lpips_loss = self.lpips.forward(
                F.interpolate(rendered_imgs.permute(0, 3, 1, 2), scale_factor=0.25, mode="bilinear",align_corners=True),
                F.interpolate(imgs.permute(0, 3, 1, 2), scale_factor=0.25, mode="bilinear",align_corners=True),
                normalize=True
                ).mean()
                rgb_loss = 0.8 * F.l1_loss(rendered_imgs, imgs) + 0.2 * (
                    1 - fused_ssim(rendered_imgs.permute(0, 3, 1, 2), imgs.permute(0, 3, 1, 2), padding="valid")
                ) + 0.01*lpips_loss

        loss += rgb_loss * self.losses_cfg.w_rgb

        # Mask loss.
        if not self.model.has_bg:
            mask_loss = F.mse_loss(rendered_all["acc"], masks[..., None])  # type: ignore
        else:
            # Check if masks are completely empty or too small
            mask_threshold = 50  # minimum number of pixels
            if masks.sum() < mask_threshold:
                mask_loss = F.mse_loss(
                    rendered_all["acc"], torch.ones_like(rendered_all["acc"])  # type: ignore
                )
            else:
                mask_loss = F.mse_loss(
                    rendered_all["acc"], torch.ones_like(rendered_all["acc"])  # type: ignore
                ) + masked_l1_loss(
                    rendered_all["mask"],
                    masks[..., None],
                    quantile=0.98,  # type: ignore
                )
        loss += mask_loss * self.losses_cfg.w_mask

        loss
        # (B * N, H * W, 3).
        pred_tracks_3d = (
            rendered_all["tracks_3d"].permute(0, 3, 1, 2, 4).reshape(-1, H * W, 3)  # type: ignore
        )
        pred_tracks_2d = torch.einsum(
            "bij,bpj->bpi", torch.cat(target_Ks), pred_tracks_3d
        )
        # (B * N, H * W, 1).
        mapped_depth = torch.clamp(pred_tracks_2d[..., 2:], min=1e-6)
        # (B * N, H * W, 2).
        pred_tracks_2d = pred_tracks_2d[..., :2] / mapped_depth

        # (B * N).
        w_interval = torch.exp(-2 * frame_intervals / num_frames)
        # w_track_loss = min(1, (self.max_steps - self.global_step) / 6000)
        track_weights = confidences[..., None] * w_interval

        # (B, H, W).
        masks_flatten = torch.zeros_like(masks)
        for i in range(B):
            # This takes advantage of the fact that the query 2D tracks are
            # always on the grid.
            query_pixels = query_tracks_2d[i].to(torch.int64)
            masks_flatten[i, query_pixels[:, 1], query_pixels[:, 0]] = 1.0
        # (B * N, H * W).
        masks_flatten = (
            masks_flatten.reshape(-1, H * W).tile(1, N).reshape(-1, H * W) > 0.5
        )

        # Check if there are any visible tracks to compute loss on
        if visibles.sum() > 0:
            track_2d_loss = masked_l1_loss(
                pred_tracks_2d[masks_flatten][visibles],
                tracks_2d[visibles],
                mask=track_weights[visibles],
                quantile=0.98,
            ) / max(H, W)
        else:
            track_2d_loss = torch.tensor(0.0, device=device)
        loss += track_2d_loss * self.losses_cfg.w_track

        depth_masks = (
            masks[..., None] if not self.model.has_bg else valid_masks[..., None]
        )

        pred_depth = cast(torch.Tensor, rendered_all["depth"])
        pred_disp = 1.0 / (pred_depth + 1e-5)
        tgt_disp = 1.0 / (depths[..., None] + 1e-5)
        depth_loss = masked_l1_loss(
            pred_disp,
            tgt_disp,
            mask=depth_masks,
            quantile=0.98,
        )
       
        loss += depth_loss * self.losses_cfg.w_depth_reg

        # mapped depth loss (using cached depth with EMA)
        #  mapped_depth_loss = 0.0
        mapped_depth_gt = torch.cat([x.reshape(-1) for x in target_track_depths], dim=0)
        if visibles.sum() > 0:
            mapped_depth_loss = masked_l1_loss(
                1 / (mapped_depth[masks_flatten][visibles] + 1e-5),
                1 / (mapped_depth_gt[visibles, None] + 1e-5),
                track_weights[visibles],
                quantile=0.95
            )
        else:
            mapped_depth_loss = torch.tensor(0.0, device=device)

        loss += mapped_depth_loss * self.losses_cfg.w_depth_const

        #  depth_gradient_loss = 0.0
        depth_gradient_loss = compute_gradient_loss(
            pred_disp,
            tgt_disp,
            mask=depth_masks > 0.5,
            quantile=0.95,
        )

        loss += depth_gradient_loss * self.losses_cfg.w_depth_grad

  
        small_accel_loss = 0
        for i in range(self.model.motion_tree.leaf_level+1):
            small_accel_loss += compute_se3_smoothness_loss(
                rearrange(self.model.motion_tree.motion_bases[i].params["rots"], "np k t dim -> (np k) t dim"),     
                rearrange(self.model.motion_tree.motion_bases[i].params["transls"], "np k t dim -> (np k) t dim") 
            )
        loss += small_accel_loss * self.losses_cfg.w_smooth_bases

        # tracks should be smooth
        ts = torch.clamp(ts, min=1, max=num_frames - 2)
        ts_neighbors = torch.cat((ts - 1, ts, ts + 1))
        transfms_nbs = self.model.motion_tree.compute_transforms_from_nodes(ts_neighbors, self.model.fg.params["means"])

        means_fg_nbs = torch.einsum(
            "pnij,pj->pni",
            transfms_nbs,
            F.pad(self.model.fg.params["means"], (0, 1), value=1.0),
        )
        means_fg_nbs = means_fg_nbs.reshape(
            means_fg_nbs.shape[0], 3, -1, 3
        )  # [G, 3, n, 3]
        if self.losses_cfg.w_smooth_tracks > 0:
            small_accel_loss_tracks = 0.5 * (
                (2 * means_fg_nbs[:, 1:-1] - means_fg_nbs[:, :-2] - means_fg_nbs[:, 2:])
                .norm(dim=-1)
                .mean()
            )
            loss += small_accel_loss_tracks * self.losses_cfg.w_smooth_tracks
        
        arap = 0.
        num_levels = self.model.motion_tree.leaf_level+1
        for l in range(num_levels):
            positions = self.model.motion_tree.motion_nodes[l].get_positions()
            curve_transfms = self.model.motion_tree.compute_node_world_transforms(torch.tensor([i for i in range(0,self.num_frames,15)]), level=l)
            transfms = self.model.motion_tree.compute_node_world_transforms(ts, level=l)
            arap += self.losses_cfg.w_arap*compute_arap_loss(ts, positions, curve_transfms,transfms ,5, 1., 0.5) * 5**(num_levels-1-l)

        loss += arap

        # Structural loss (front/back distance preservation)
        structural_loss = torch.tensor(0., device=device)
        if self.losses_cfg.w_structural > 0 and self.global_step >= self.losses_cfg.structural_start_iter:
            from flow3d.loss_utils import compute_structural_loss
            for l in range(num_levels):
                positions = self.model.motion_tree.motion_nodes[l].get_positions()
                structural_loss += compute_structural_loss(
                    self.model.motion_tree,
                    positions,
                    ts,
                    w2cs,
                    Ks,
                    img_wh,
                    num_frames,
                    num_sample_rays=self.losses_cfg.structural_num_rays,
                    patch_size=self.losses_cfg.structural_patch_size,
                    max_nodes_per_patch=self.losses_cfg.structural_max_nodes_per_patch,
                    curve_sampling_interval=self.losses_cfg.structural_curve_interval,
                    time_weight_decay=self.losses_cfg.structural_weight_decay,
                    level=l,
                )
            loss += structural_loss * self.losses_cfg.w_structural

        radius_loss = 0
        num_levels = self.model.motion_tree.leaf_level+1
        for l in range(num_levels):
            positions = self.model.motion_tree.motion_nodes[l].get_positions()
            radius = self.model.motion_tree.motion_nodes[l].get_radius()
            radius_loss += compute_radius_loss(positions, radius, 3)
        radius_loss = radius_loss / num_levels
        
        loss += self.losses_cfg.w_radius_reg * radius_loss


        # Constrain the std of scales.
        # Apply visibility-based weighting if available
        if self.model.fg.visibilities is not None and B > 0:
            # Get visibility for current batch frames [B, G]
            fg_vis_batch = self.model.fg.visibilities[:, ts].T  # [G, B] -> [B, G]
            # Average visibility across batch
            fg_vis_weight = fg_vis_batch.mean(dim=0)  # [G]
            # Apply invisible_weight: visible=1.0, invisible=invisible_weight
            invisible_weight = self.losses_cfg.invisible_weight if hasattr(self.losses_cfg, 'invisible_weight') else 0.5
            fg_vis_weight = fg_vis_weight + (1 - fg_vis_weight) * invisible_weight
            # Compute weighted scale variance
            scale_var_per_gaussian = torch.var(self.model.fg.params["scales"], dim=-1)  # [G]
            weighted_scale_var = (scale_var_per_gaussian * fg_vis_weight).sum() / (fg_vis_weight.sum() + 1e-8)
            loss += self.losses_cfg.w_scale_var * torch.nan_to_num(weighted_scale_var, nan=0.0)
        else:
            # Fallback to original behavior if visibilities not available
            loss += (
                self.losses_cfg.w_scale_var
                * torch.nan_to_num(torch.var(self.model.fg.params["scales"], dim=-1).mean(), nan=0.0)
            )
        if self.model.bg is not None:
            loss += (
                self.losses_cfg.w_scale_var
                * torch.nan_to_num(torch.var(self.model.bg.params["scales"], dim=-1).mean(), nan=0.0)
            )


        # Acceleration along ray direction should be small.
        z_accel_loss = compute_z_acc_loss(means_fg_nbs, w2cs)
        loss += self.losses_cfg.w_z_accel * z_accel_loss

        # Prepare stats for logging.
        stats = {
            "train/loss": loss.item(),
            "train/rgb_loss": rgb_loss.item(),
            "train/lpips_loss": lpips_loss.item(),
            "train/mask_loss": mask_loss.item(),
            "train/depth_loss": depth_loss.item(),
            "train/depth_gradient_loss": depth_gradient_loss.item(),
            "train/mapped_depth_loss": mapped_depth_loss.item(),
            "train/track_2d_loss": track_2d_loss.item(),
            "train/small_accel_loss": small_accel_loss.item(),
            "train/z_acc_loss": z_accel_loss.item(),
            "train/arap_loss": arap.item(),
            "train/radius_loss": radius_loss.item(),
            "train/num_gaussians": self.model.num_gaussians,
            "train/num_fg_gaussians": self.model.num_fg_gaussians,
            "train/num_bg_gaussians": self.model.num_bg_gaussians,
            "train/num_nodes": self.model.motion_tree.num_nodes
        }

        if self.losses_cfg.w_structural > 0:
            stats["train/structural_loss"] = structural_loss.item()

        # Compute metrics.
        with torch.no_grad():
            psnr = self.psnr_metric(
                rendered_imgs, imgs, masks if not self.model.has_bg else valid_masks
            )
            self.psnr_metric.reset()
            stats["train/psnr"] = psnr
            if self.model.has_bg:
                bg_psnr = self.bg_psnr_metric(rendered_imgs, imgs, 1.0 - masks)
                fg_psnr = self.fg_psnr_metric(rendered_imgs, imgs, masks)
                self.bg_psnr_metric.reset()
                self.fg_psnr_metric.reset()
                stats["train/bg_psnr"] = bg_psnr
                stats["train/fg_psnr"] = fg_psnr

        stats.update(
            **{
                "train/num_rays_per_sec": num_rays_per_sec,
                "train/num_rays_per_step": float(num_rays_per_step),
            }
        )

        return loss, stats, num_rays_per_step, num_rays_per_sec

    def log_dict(self, stats: dict):
        # Separate images from scalar metrics
        scalar_stats = {}
        image_stats = {}

        for k, v in stats.items():
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict) and "image" in v[0]:
                # This is a list of images for wandb
                image_stats[k] = v
            else:
                scalar_stats[k] = v

        # Log scalars to TensorBoard
        for k, v in scalar_stats.items():
            self.tb_writer.add_scalar(k, v, self.global_step)

        # Log to wandb if available
        if self.wandb_writer is not None:
            wandb_log = scalar_stats.copy()

            # Add images to wandb log
            if len(image_stats) > 0:
                import wandb
                for k, image_list in image_stats.items():
                    wandb_log[k] = [
                        wandb.Image(
                            img_dict["image"],
                            caption=img_dict.get("frame", "")
                        )
                        for img_dict in image_list
                    ]

            self.wandb_writer.log(wandb_log, step=self.global_step)


    def run_control_steps(self):
        global_step = self.global_step
        # Adaptive gaussian control.
        cfg = self.optim_cfg
        num_frames = self.model.num_frames
        ready = self._prepare_control_step()


        if (
            ready
            and global_step > cfg.warmup_steps
            and global_step % cfg.control_every == 0
            and global_step < cfg.stop_control_steps
        ):  
            if (
                global_step < cfg.stop_densify_steps
                and global_step % self.reset_opacity_every > num_frames
            ):
                self._densify_control_step(global_step)
            if global_step % self.reset_opacity_every > min(3 * num_frames, 1000):
                self._cull_control_step(global_step)
            if global_step % self.reset_opacity_every == 0:
                self._reset_opacity_control_step()

            if (global_step >= 3 * cfg.warmup_steps
                and global_step% cfg.node_gaussian_densify_steps ==0 
                and global_step <= (cfg.warmup_steps + cfg.node_gaussian_densify_times * cfg.node_gaussian_densify_steps)):
                self._densify_node_step_with_gaussian(num_sample=20)

            if (cfg.start_grad_densify_steps <= global_step
                and global_step <= cfg.stop_node_grad_densify
                and global_step % cfg.node_grad_densify_steps == 0):
                self._densify_node_control_step(global_step)

            if global_step == cfg.start_second_level:
                self._cull_node_step(threshold=30)
                self._deepen_motion_tree()


            # Reset stats after every control.
            for k in self.running_stats:
                self.running_stats[k].zero_()

    def _deepen_motion_tree(self):
        # deepen motion tree
        gaussian_means = self.model.fg.params["means"]
        parent_indices_to_remove = self.model.motion_tree.deepen(gaussian_means)
        parent_level = self.model.motion_tree.leaf_level - 1

        # get cull mask
        cull_mask = torch.zeros(self.model.motion_tree.motion_nodes[parent_level].num_nodes, dtype=torch.bool, device=self.device)
        cull_mask[parent_indices_to_remove] = True
        
        self.model.motion_tree.parent_indices[parent_level].cull(cull_mask)
        motion_node_param_map = self.model.motion_tree.motion_nodes[parent_level].cull(cull_mask)
        for param_name, new_params in motion_node_param_map.items():
            full_param_name = f"motion_tree.motion_nodes.{parent_level}.params.{param_name}"
            optimizer = self.optimizers[full_param_name]
            remove_from_optim(optimizer, [new_params], cull_mask)
            
        # identify new params
        param_name_exists = self.optimizers.keys()
        lr_dict = asdict(self.lr_cfg)

        # update optimizers and schedulers
        for name, params in self.model.named_parameters():

            # for already existing params
            if name in param_name_exists:
                continue

            split = name.split(".")
            assert len(split) == 5
            _, part, _, _, field = split # e.g., motion_tree.motion_bases.0.params.transls

            lr = lr_dict[part][field]
            optim = torch.optim.Adam([{"params": params, "lr": lr, "name": name}])

            fnc = lambda _, **__: 1.0

            self.optimizers[name] = optim
            self.schedulers[name] = torch.optim.lr_scheduler.LambdaLR(
                optim, functools.partial(fnc, lr_init=lr)
            )
        
        guru.info(">>>> motion tree deepened!!!!! added new params to schedulers and optimizers!!!")


    @torch.no_grad()
    def _prepare_control_step(self) -> bool:
        # Prepare for adaptive gaussian control based on the current stats.
        if not (
            self.model._current_radii is not None
            and self.model._current_xys is not None
        ):
            guru.warning("Model not training, skipping control step preparation")
            return False

        batch_size = len(self._batched_xys)
        # these quantities are for each rendered view and have shapes (C, G, *)
        # must be aggregated over all views
        for _current_xys, _current_radii, _current_img_wh in zip(
            self._batched_xys, self._batched_radii, self._batched_img_wh
        ):  
            _current_radii, _ = torch.max(_current_radii, dim=-1)
            sel = _current_radii > 0
            gidcs = torch.where(sel)[1]
            # normalize grads to [-1, 1] screen space
            xys_grad = _current_xys.grad.clone()
            xys_grad[..., 0] *= _current_img_wh[0] / 2.0 * batch_size
            xys_grad[..., 1] *= _current_img_wh[1] / 2.0 * batch_size
            self.running_stats["xys_grad_norm_acc"].index_add_(
                0, gidcs, xys_grad[sel].norm(dim=-1)
            )
            self.running_stats["vis_count"].index_add_(
                0, gidcs, torch.ones_like(gidcs, dtype=torch.int64)
            )
            max_radii = torch.maximum(
                self.running_stats["max_radii"].index_select(0, gidcs),
                _current_radii[sel] / max(_current_img_wh),
            )
            self.running_stats["max_radii"].index_put((gidcs,), max_radii)
        return True


    @torch.no_grad()
    def _get_node_importance(self, level=0):
        grad = self.running_stats["xys_grad_norm_acc"] / self.running_stats[
            "vis_count"
        ].clamp_min(1)
        grad = grad[:self.model.num_fg_gaussians]
        nn_idx, nn_weight = self.model.motion_tree.compute_knn_nodes(self.model.fg.params["means"], k=3, no_softmax=True, level=level)  #[G, K]
        node_importance = torch.zeros_like(self.model.motion_tree.motion_nodes[level].get_positions()[:, 0])
        node_edge_count = torch.zeros_like(self.model.motion_tree.motion_nodes[level].get_positions()[:, 0])
        avg_affected_x = torch.zeros_like(self.model.motion_tree.motion_nodes[level].get_positions())
        node_importance.index_add_(dim=0, index=nn_idx.view(-1), source=(nn_weight * grad[:, None]).view(-1))
        node_edge_count.index_add_(dim=0, index=nn_idx.view(-1), source=nn_weight.view(-1))
        avg_x = (nn_weight * grad[:, None]).view(-1, 1) #[G, K]
        avg_x = avg_x * self.model.fg.params["means"][:,None].expand(*nn_weight.shape, 3).reshape(-1, 3) #[GK, 3]
        avg_affected_x.index_add_(dim=0, index=nn_idx.view(-1), source=avg_x)
        avg_affected_x = avg_affected_x / node_importance[:, None]
        node_importance = node_importance / (node_edge_count + 1e-7)
        return node_importance, avg_affected_x, node_edge_count


    @torch.no_grad()
    def _densify_node_control_step(self, global_step):
        assert (self.running_stats["vis_count"] > 0).any()

        level = 0

        cfg = self.optim_cfg
        node_avg_xgradnorm, node_avg_x, node_edge_count = self._get_node_importance(level=level)

        selected_pts_mask = torch.logical_and(node_avg_xgradnorm > cfg.densify_node_grad_threshold, node_avg_x.isnan().logical_not().all(dim=-1))


        pruned_pts_mask = node_edge_count == 0
        if selected_pts_mask.sum() > 0 or pruned_pts_mask.sum() > 0:
            guru.info(f"Add {selected_pts_mask.sum()} nodes and prune {pruned_pts_mask.sum()} node. ", end='')
        else:
            return
        
        # Densify
        if selected_pts_mask.sum() > 0:
            new_motion_node = {}
            new_motion_node["positions"]= node_avg_x[selected_pts_mask]
            new_motion_node["radius"] = self.model.motion_tree.motion_nodes[level].params["radius"][selected_pts_mask]
            new_motion_node["motion_coefs"] = self.model.motion_tree.motion_nodes[level].params["motion_coefs"][selected_pts_mask]

            motion_node_param_map = self.model.motion_tree.motion_nodes[level].add(new_motion_node)
            for param_name, new_params in motion_node_param_map.items():
                full_param_name = f"motion_tree.motion_nodes.{level}.params.{param_name}"                
                optimizer = self.optimizers[full_param_name]

                add_in_optim(optimizer, [new_params], num_add=selected_pts_mask.sum().item())

            # update parent indices
            parent_indices = self.model.motion_tree.parent_indices[level].get_indices()
            parent_indices_added = parent_indices[selected_pts_mask]
            self.model.motion_tree.parent_indices[level].add(parent_indices_added)
        
        # Prune
        if pruned_pts_mask.shape[0] < self.model.motion_tree.motion_nodes[level].num_nodes:
            pruned_pts_mask = torch.cat([pruned_pts_mask, torch.zeros([self.model.motion_tree.motion_nodes[level].num_nodes - pruned_pts_mask.shape[0]]).to(pruned_pts_mask.device).to(pruned_pts_mask.dtype)])
        if pruned_pts_mask.sum() > 0:
            
            self.model.motion_tree.parent_indices[level].cull(pruned_pts_mask)

            motion_node_param_map = self.model.motion_tree.motion_nodes[level].cull(pruned_pts_mask)
            for param_name, new_params in motion_node_param_map.items():
                full_param_name = f"motion_tree.motion_nodes.{level}.params.{param_name}"
                optimizer = self.optimizers[full_param_name]

                remove_from_optim(optimizer, [new_params], pruned_pts_mask)


    @torch.no_grad()
    def _densify_control_step(self, global_step):
        assert (self.running_stats["vis_count"] > 0).any()

        cfg = self.optim_cfg
        xys_grad_avg = self.running_stats["xys_grad_norm_acc"] / self.running_stats[
            "vis_count"
        ].clamp_min(1)
        is_grad_too_high = xys_grad_avg > cfg.densify_xys_grad_threshold
        # Split gaussians.
        scales = self.model.get_scales_all()
        is_scale_too_big = scales.amax(dim=-1) > cfg.densify_scale_threshold
        if global_step < cfg.stop_control_by_screen_steps:
            is_radius_too_big = (
                self.running_stats["max_radii"] > cfg.densify_screen_threshold
            )
        else:
            is_radius_too_big = torch.zeros_like(is_grad_too_high, dtype=torch.bool)

        should_split = is_grad_too_high & (is_scale_too_big | is_radius_too_big)
        should_dup = is_grad_too_high & ~is_scale_too_big

        num_fg = self.model.num_fg_gaussians
        should_fg_split = should_split[:num_fg]
        num_fg_splits = int(should_fg_split.sum().item())
        should_fg_dup = should_dup[:num_fg]
        num_fg_dups = int(should_fg_dup.sum().item())

        should_bg_split = should_split[num_fg:]
        num_bg_splits = int(should_bg_split.sum().item())
        should_bg_dup = should_dup[num_fg:]
        num_bg_dups = int(should_bg_dup.sum().item())

        fg_param_map = self.model.fg.densify_params(should_fg_split, should_fg_dup)
        for param_name, new_params in fg_param_map.items():
            full_param_name = f"fg.params.{param_name}"
            optimizer = self.optimizers[full_param_name]
            dup_in_optim(
                optimizer,
                [new_params],
                should_fg_split,
                num_fg_splits * 2 + num_fg_dups,
            )

        if self.model.bg is not None:
            bg_param_map = self.model.bg.densify_params(should_bg_split, should_bg_dup)
            for param_name, new_params in bg_param_map.items():
                full_param_name = f"bg.params.{param_name}"
                optimizer = self.optimizers[full_param_name]
                dup_in_optim(
                    optimizer,
                    [new_params],
                    should_bg_split,
                    num_bg_splits * 2 + num_bg_dups,
                )

        # update running stats
        for k, v in self.running_stats.items():
            v_fg, v_bg = v[:num_fg], v[num_fg:]
            new_v = torch.cat(
                [
                    v_fg[~should_fg_split],
                    v_fg[should_fg_dup],
                    v_fg[should_fg_split].repeat(2),
                    v_bg[~should_bg_split],
                    v_bg[should_bg_dup],
                    v_bg[should_bg_split].repeat(2),
                ],
                dim=0,
            )
            self.running_stats[k] = new_v
        guru.info(
            f"Split {should_split.sum().item()} gaussians, "
            f"Duplicated {should_dup.sum().item()} gaussians, "
            f"{self.model.num_gaussians} gaussians left"
        )


    @torch.no_grad()
    def _cull_control_step(self, global_step):
        # Cull gaussians.
        cfg = self.optim_cfg
        opacities = self.model.get_opacities_all()
        device = opacities.device
        is_opacity_too_small = opacities < cfg.cull_opacity_threshold
        is_radius_too_big = torch.zeros_like(is_opacity_too_small, dtype=torch.bool)
        is_scale_too_big = torch.zeros_like(is_opacity_too_small, dtype=torch.bool)
        cull_scale_threshold = (
            torch.ones(len(is_scale_too_big), device=device) * cfg.cull_scale_threshold
        )
        num_fg = self.model.num_fg_gaussians
        cull_scale_threshold[num_fg:] *= self.model.bg_scene_scale
        if global_step > self.reset_opacity_every:
            scales = self.model.get_scales_all()
            is_scale_too_big = scales.amax(dim=-1) > cull_scale_threshold
            if global_step < cfg.stop_control_by_screen_steps:
                is_radius_too_big = (
                    self.running_stats["max_radii"] > cfg.cull_screen_threshold
                )
        should_cull = is_opacity_too_small | is_radius_too_big | is_scale_too_big
        should_fg_cull = should_cull[:num_fg]
        should_bg_cull = should_cull[num_fg:]

        fg_param_map = self.model.fg.cull_params(should_fg_cull)
        for param_name, new_params in fg_param_map.items():
            full_param_name = f"fg.params.{param_name}"
            optimizer = self.optimizers[full_param_name]
            remove_from_optim(optimizer, [new_params], should_fg_cull)

        if self.model.bg is not None:
            bg_param_map = self.model.bg.cull_params(should_bg_cull)
            for param_name, new_params in bg_param_map.items():
                full_param_name = f"bg.params.{param_name}"
                optimizer = self.optimizers[full_param_name]
                remove_from_optim(optimizer, [new_params], should_bg_cull)

        # update running stats
        for k, v in self.running_stats.items():
            self.running_stats[k] = v[~should_cull]

        guru.info(
            f"Culled {should_cull.sum().item()} gaussians, "
            f"{self.model.num_gaussians} gaussians left"
        )


    @torch.no_grad()
    def _cull_node_step(self, threshold=30):
        # currently only support cull 0-th level nodes...
        level = 0

        neighbor_gaussian_indices = self.model.find_gaussian_near_node(level=level)
        neighbor_gaussian_num = torch.sum(neighbor_gaussian_indices, dim=-1)
        should_cull = neighbor_gaussian_num < threshold

        # update parent_indices
        self.model.motion_tree.parent_indices[level].cull(should_cull)

        # cull and update optimizer
        param_dicts = self.model.motion_tree.motion_nodes[level].cull(should_cull)
        for param_name, new_params in param_dicts.items():
            full_param_name = f"motion_tree.motion_nodes.{level}.params.{param_name}"
            optimizer = self.optimizers[full_param_name]
            remove_from_optim(optimizer, [new_params], should_cull)
    
    @torch.no_grad()
    def _densify_node_step_with_gaussian(self, gaussian_interval=10, time_interval=10, num_sample=10):
        """
        Densify motion nodes by sampling new nodes baed on Gaussian positions.

        Args:
            gaussian_interval (int): Interval for subsampling Gaussians.
            time_interval (int): Interval for subsampling time steps.
            num_sample (int): Number of new nodes to sample.
        """
        level = 0

        # Compute positions of subsampled Gaussian at subsampled time steps
        sampled_time_steps = torch.tensor([i for i in range(0, self.num_frames, time_interval)])
        sampled_gaussian_means_cano = self.model.fg.params["means"][::gaussian_interval]
        gaussian_transforms = self.model.motion_tree.compute_transforms_from_nodes(
            sampled_time_steps, sampled_gaussian_means_cano, level=level 
        )
        gaussian_means = torch.einsum(
            "pnij, pj -> pni",
            gaussian_transforms,
            F.pad(sampled_gaussian_means_cano, (0, 1), value=1.0),
        )

        # Compute node positions at subsampled time steps
        node_positions_cano = self.model.motion_tree.motion_nodes[level].get_positions()
        node_transforms = self.model.motion_tree.compute_node_world_transforms(
            sampled_time_steps, level=level
        )
        node_positions = torch.einsum(
            "pnij, pj -> pni",
            node_transforms,
            F.pad(node_positions_cano, (0, 1), value=1.0),
        )

        # Retrieve parent indices for nodes
        parent_indices = self.model.motion_tree.parent_indices[level].get_indices()

        # Find 3 nearest neighbor nodes for each Gaussian in the canonical frame
        dist_gaussian2nodes_cano = torch.norm(
            sampled_gaussian_means_cano.unsqueeze(1) - node_positions_cano.unsqueeze(0), dim=-1
        )
        _, neighbor_node_indices = dist_gaussian2nodes_cano.topk(k=3, largest=False)
        neighbor_node_positions = node_positions[neighbor_node_indices]
        neighbor_node_indices = parent_indices[neighbor_node_indices]

        # Find Gaussians with large curve distance to their neighbor nodes
        dist_gaussian2neighbornodes = torch.norm(
            gaussian_means.unsqueeze(1) - neighbor_node_positions, dim=-1
        )
        curve_gaussian2neighbornodes, curve_neighbor_node_indices  = torch.min(
            torch.max(dist_gaussian2neighbornodes, dim=-1)[0], dim=-1
        )

        neighbor_parent_indices = neighbor_node_indices[
            torch.arange(len(curve_neighbor_node_indices)), curve_neighbor_node_indices
        ]

        # Select Gaussians with distances above the 90th percentile as new node candidates
        distance_threshold = torch.quantile(curve_gaussian2neighbornodes, 0.9)
        candidate_mask = curve_gaussian2neighbornodes > distance_threshold
        new_node_candidates = sampled_gaussian_means_cano[candidate_mask]
        candidate_parent_indices = neighbor_parent_indices[candidate_mask]

        # Sample new nodes from the candidates
        sampled_indices = knn_weighted_sampling(new_node_candidates, 3, num_sample)
        sampled_node_positions = new_node_candidates[sampled_indices]
        sampled_node_parent_indices = candidate_parent_indices[sampled_indices]
        sampled_node_radius = torch.mean(
            self.model.motion_tree.motion_nodes[level].params["radius"]
        ).repeat(num_sample)

        # Compute motion coefficients for the new nodes
        knn_indices, knn_weights = self.model.motion_tree.compute_knn_nodes(
            sampled_node_positions, k=3, level=level
        )
        motion_coefficients = self.model.motion_tree.motion_nodes[level].params["motion_coefs"][knn_indices]
        motion_coefficients = torch.sum(knn_weights.unsqueeze(-1) * motion_coefficients, dim=-2)

        # Create a dictionary for the new nodes
        new_nodes_dict = {
            "positions": sampled_node_positions,
            "radius": sampled_node_radius,
            "motion_coefs": motion_coefficients,
        }

        # Update parent indices
        self.model.motion_tree.parent_indices[level].add(sampled_node_parent_indices)

        # Add new nodes to the motio tree and update the optimizer
        param_dicts = self.model.motion_tree.motion_nodes[level].add(new_nodes_dict)
        for param_name, new_params in param_dicts.items():
            full_param_name = f"motion_tree.motion_nodes.{level}.params.{param_name}"
            optimizer = self.optimizers[full_param_name]
            add_in_optim(optimizer, [new_params], num_add=num_sample)


    @torch.no_grad()
    def _reset_opacity_control_step(self):
        # Reset gaussian opacities.
        new_val = torch.logit(torch.tensor(0.8 * self.optim_cfg.cull_opacity_threshold))
        for part in ["fg", "bg"]:
            part_params = getattr(self.model, part).reset_opacities(new_val)
            # Modify optimizer states by new assignment.
            for param_name, new_params in part_params.items():
                full_param_name = f"{part}.params.{param_name}"
                optimizer = self.optimizers[full_param_name]
                reset_in_optim(optimizer, [new_params])
        guru.info("Reset opacities")

    @torch.no_grad()
    def _debug_node_visibility(self, batch):
        """
        Debug node visibility by logging distributions and relationships to wandb.
        """
        if self.model.fg.visibilities is None:
            guru.warning("No visibility information available, skipping debug")
            return

        # Use first time in batch
        t = int(batch["ts"][0].item())
        w2c = batch["w2cs"][0]  # [4, 4]

        num_levels = self.model.motion_tree.leaf_level + 1
        num_frames = self.model.num_frames

        for level in range(num_levels):
            # 1. Compute node visibility at time t
            node_vis = compute_node_visibility(
                self.model.fg.visibilities,
                self.model.fg.params["means"],
                self.model.motion_tree,
                t,
                level=level
            )  # [num_nodes]

            # 2. Compute node depth (camera-relative)
            node_positions_cano = self.model.motion_tree.motion_nodes[level].get_positions()  # [num_nodes, 3]

            # Transform to world space at time t
            node_transforms_world = self.model.motion_tree.compute_node_world_transforms(
                torch.tensor([t], device=self.device), level=level
            )  # [num_nodes, 1, 4, 4]
            node_transforms_t = rt_to_mat4(
                node_transforms_world[..., :3, :3],
                node_transforms_world[..., :3, -1]
            )[:, 0, :, :]  # [num_nodes, 4, 4]

            node_positions_world = transform_rigid(
                homogenize_points(node_positions_cano.unsqueeze(-2)),
                node_transforms_t.unsqueeze(-3)
            ).squeeze(-2)[..., :3]  # [num_nodes, 3]

            # Transform to camera space
            node_positions_cam = transform_rigid(
                homogenize_points(node_positions_world.unsqueeze(-2)),
                w2c.unsqueeze(0).expand(node_positions_world.shape[0], -1, -1).unsqueeze(-3)
            ).squeeze(-2)[..., :3]  # [num_nodes, 3]

            node_depth = node_positions_cam[:, 2]  # [num_nodes] - Z depth

            # 3. Compute neighbor visibility average
            # Get curve positions for KNN
            curve_times = torch.arange(0, num_frames, 15, device=self.device)
            curve_transforms = self.model.motion_tree.compute_node_world_transforms(
                curve_times, level=level
            )  # [num_nodes, T_curve, 4, 4]
            curve_transforms_mat = rt_to_mat4(
                curve_transforms[..., :3, :3],
                curve_transforms[..., :3, -1]
            )

            curve_positions = transform_rigid(
                homogenize_points(node_positions_cano.unsqueeze(-2)),
                curve_transforms_mat
            )[..., :3]  # [num_nodes, T_curve, 3]

            # Find neighbors using curve distance
            neighbor_indices = curve_distance_knn(curve_positions, k=10)  # [num_nodes, 10]

            # Get neighbor visibilities
            neighbor_vis = node_vis[neighbor_indices]  # [num_nodes, 10]
            neighbor_vis_avg = neighbor_vis.mean(dim=-1)  # [num_nodes]

            # 4. Log to wandb
            if self.wandb_writer is not None:
                import wandb

                # Convert to numpy for logging
                node_vis_np = node_vis.cpu().numpy()
                node_depth_np = node_depth.cpu().numpy()
                neighbor_vis_avg_np = neighbor_vis_avg.cpu().numpy()

                # 1. Histogram: visibility distribution
                self.wandb_writer.log({
                    f"debug_vis/level{level}_distribution": wandb.Histogram(node_vis_np),
                }, step=self.global_step)

                # 2. Scatter: visibility vs depth
                vis_depth_data = [[float(d), float(v)] for d, v in zip(node_depth_np, node_vis_np)]
                self.wandb_writer.log({
                    f"debug_vis/level{level}_vis_vs_depth": wandb.plot.scatter(
                        wandb.Table(data=vis_depth_data, columns=["depth", "visibility"]),
                        "depth", "visibility",
                        title=f"Level {level}: Visibility vs Depth (t={t})"
                    ),
                }, step=self.global_step)

                # 3. Scatter: self vis vs neighbor avg vis
                vis_neighbor_data = [[float(s), float(n)] for s, n in zip(node_vis_np, neighbor_vis_avg_np)]
                self.wandb_writer.log({
                    f"debug_vis/level{level}_self_vs_neighbor": wandb.plot.scatter(
                        wandb.Table(data=vis_neighbor_data, columns=["self_vis", "neighbor_avg_vis"]),
                        "self_vis", "neighbor_avg_vis",
                        title=f"Level {level}: Self vs Neighbor Vis (t={t})"
                    ),
                }, step=self.global_step)

                guru.info(f"Logged node visibility debug for level {level} at t={t}")


    def configure_optimizers(self):
        def _exponential_decay(step, *, lr_init, lr_final):
            t = np.clip(step / self.optim_cfg.max_steps, 0.0, 1.0)
            lr = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
            return lr / lr_init

        lr_dict = asdict(self.lr_cfg)
        optimizers = {}
        schedulers = {}
        # named parameters will be [part].params.[field]
        # e.g. fg.params.means
        # lr config is a nested dict for each fg/bg part
        for name, params in self.model.named_parameters():

            split = name.split(".")
            if len(split) == 3:
                part, _, field = split
            elif len(split) == 5:
                _, part, _, _, field = split # e.g., motion_tree.motion_bases.0.params.transls
            else:
                raise ValueError("wrong split!")

            lr = lr_dict[part][field]
            optim = torch.optim.Adam([{"params": params, "lr": lr, "name": name}])

            if "scales" in name:
                fnc = functools.partial(_exponential_decay, lr_final=0.1 * lr)
            else:
                fnc = lambda _, **__: 1.0

            optimizers[name] = optim
            schedulers[name] = torch.optim.lr_scheduler.LambdaLR(
                optim, functools.partial(fnc, lr_init=lr)
            )
        return optimizers, schedulers
    

def compact_indices(indices):
    """ return compact indices (from 0 to max_nodes-1)
        e.g. Input:  [0, 0, 1, 1, 3, 4, 6, 6, 7]
            Output: [0, 0, 1, 1, 2, 3, 4, 4, 5]
    """

    output, inverse_indices = torch.unique(indices, return_inverse=True)
    indices_range = torch.arange(len(output), device=output.device)
    compact_indices = indices_range[inverse_indices]
    return compact_indices


def dup_in_optim(optimizer, new_params: list, should_dup: torch.Tensor, num_dups: int):
    for i, p_new in enumerate(new_params):
        old_params = optimizer.param_groups[i]["params"][0]
        param_state = optimizer.state[old_params]
        if len(param_state) == 0:
            return
        for key in param_state:
            if key == "step":
                continue
            p = param_state[key]
            param_state[key] = torch.cat(
                [p[~should_dup], p.new_zeros(num_dups, *p.shape[1:])],
                dim=0,
            )
        del optimizer.state[old_params]
        optimizer.state[p_new] = param_state
        optimizer.param_groups[i]["params"] = [p_new]
        del old_params
        torch.cuda.empty_cache()


def remove_from_optim(optimizer, new_params: list, _should_cull: torch.Tensor):
    assert len(optimizer.param_groups) == len(new_params)
    for i, p_new in enumerate(new_params):
        old_params = optimizer.param_groups[i]["params"][0]
        param_state = optimizer.state[old_params]
        if len(param_state) == 0:
            return
        for key in param_state:
            if key == "step":
                continue
            param_state[key] = param_state[key][~_should_cull]
        del optimizer.state[old_params]
        optimizer.state[p_new] = param_state
        optimizer.param_groups[i]["params"] = [p_new]
        del old_params
        torch.cuda.empty_cache()


def reset_in_optim(optimizer, new_params: list):
    assert len(optimizer.param_groups) == len(new_params)
    for i, p_new in enumerate(new_params):
        old_params = optimizer.param_groups[i]["params"][0]
        param_state = optimizer.state[old_params]
        if len(param_state) == 0:
            return
        for key in param_state:
            param_state[key] = torch.zeros_like(param_state[key])
        del optimizer.state[old_params]
        optimizer.state[p_new] = param_state
        optimizer.param_groups[i]["params"] = [p_new]
        del old_params
        torch.cuda.empty_cache()


def add_in_optim(optimizer, new_params: list, num_add: int):
    assert len(optimizer.param_groups) == len(new_params)
    for i, p_new in enumerate(new_params):
        old_params = optimizer.param_groups[i]["params"][0]
        param_state = optimizer.state[old_params]
        if len(param_state) == 0:
            return
        for key in param_state:
            if key == "step":
                continue
            param_state[key] = torch.cat([param_state[key], torch.zeros(num_add, *param_state[key].shape[1:], device=param_state[key].device)])
        del optimizer.state[old_params]
        optimizer.state[p_new] = param_state
        optimizer.param_groups[i]["params"] = [p_new]
        del old_params
        torch.cuda.empty_cache()


def convert_to_buffer(module: nn.Module, persistent: bool = True):
    # Recurse over child modules.
    for name, child in list(module.named_children()):
        convert_to_buffer(child, persistent)

    # Also re-save buffers to change persistence.
    for name, parameter_or_buffer in (
        *module.named_parameters(recurse=False),
        *module.named_buffers(recurse=False),
    ):
        value = parameter_or_buffer.detach().clone()
        delattr(module, name)
        module.register_buffer(name, value, persistent=persistent)
