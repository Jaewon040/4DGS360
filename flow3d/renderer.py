import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger as guru
from nerfview import CameraState

from flow3d.scene_model import SceneModel
from flow3d.vis.utils import get_server
from flow3d.vis.viewer import DynamicViewer


class Renderer:
    def __init__(
        self,
        model: SceneModel,
        device: torch.device,
        work_dir: str,
        port: int | None = None,
    ):
        self.device = device

        self.model = model
        self.num_frames = model.num_frames

        self.work_dir = work_dir
        self.global_step = 0
        self.epoch = 0

        self.viewer = None
        if port is not None:
            server = get_server(port=port)
            self.viewer = DynamicViewer(
                server, self.render_fn, model.num_frames, work_dir, mode="rendering"
            )

        # Camera visualization state
        self.train_dataset = None
        self.val_dataset = None
        self.camera_update_callback = None

    def set_datasets(self, train_dataset, val_dataset=None):
        """Set datasets for camera visualization."""
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        if self.viewer is not None:
            self._setup_camera_visualization()

    def _setup_camera_visualization(self):
        """Setup camera frustums in viser."""
        if self.train_dataset is None:
            return

        import numpy as np
        from scipy.spatial.transform import Rotation

        def c2w_to_quaternion(R):
            """Convert rotation matrix to quaternion (w, x, y, z)."""
            rot = Rotation.from_matrix(R)
            quat = rot.as_quat()  # Returns (x, y, z, w)
            return np.array([quat[3], quat[0], quat[1], quat[2]])  # Convert to (w, x, y, z)

        server = self.viewer.server

        # Get train cameras
        train_w2cs = self.train_dataset.get_w2cs()
        train_Ks = self.train_dataset.get_Ks()
        train_c2ws = torch.linalg.inv(train_w2cs).cpu().numpy()

        # Select 4 evenly spaced train cameras
        num_train_cams = len(train_c2ws)
        selected_indices = np.linspace(0, num_train_cams - 1, 4, dtype=int)

        # Add selected train cameras (blue)
        for idx in selected_indices:
            c2w = train_c2ws[idx]
            server.scene.add_camera_frustum(
                f"/cameras/train/{idx}",
                fov=2 * np.arctan(train_Ks[idx, 0, 2].cpu().numpy() / train_Ks[idx, 0, 0].cpu().numpy()),
                aspect=1.0,
                scale=0.1,
                color=(0, 0, 255),  # Blue for train
                wxyz=c2w_to_quaternion(c2w[:3, :3]),
                position=c2w[:3, 3],
                line_width=1.0,
            )

        # Add val cameras (red) if available - only every other camera (1_, 2_)
        if self.val_dataset is not None and self.val_dataset.has_validation:
            val_w2cs = self.val_dataset.get_w2cs()
            val_Ks = self.val_dataset.get_Ks()
            val_c2ws = torch.linalg.inv(val_w2cs).cpu().numpy()

            for i, c2w in enumerate(val_c2ws):
                # Only show cameras at indices 1, 3, 5, 7, ... (every other one)
                if i % 2 == 1:
                    server.scene.add_camera_frustum(
                        f"/cameras/val/{i}",
                        fov=2 * np.arctan(val_Ks[i, 0, 2].cpu().numpy() / val_Ks[i, 0, 0].cpu().numpy()),
                        aspect=1.0,
                        scale=0.1,
                        color=(255, 0, 0),  # Red for val
                        wxyz=c2w_to_quaternion(c2w[:3, :3]),
                        position=c2w[:3, 3],
                        line_width=1.0,
                    )

        # Setup callback to update current frame highlight
        def update_current_camera(event):
            current_frame = int(event.target.value)
            # Update selected train camera colors
            for idx in selected_indices:
                color = (0, 255, 255) if idx == current_frame else (0, 0, 255)  # Cyan for current, blue otherwise
                server.scene.add_camera_frustum(
                    f"/cameras/train/{idx}",
                    fov=2 * np.arctan(train_Ks[idx, 0, 2].cpu().numpy() / train_Ks[idx, 0, 0].cpu().numpy()),
                    aspect=1.0,
                    scale=0.1,
                    color=color,
                    wxyz=c2w_to_quaternion(train_c2ws[idx, :3, :3]),
                    position=train_c2ws[idx, :3, 3],
                    line_width=1.0,
                )

        self.viewer._playback_guis[0].on_update(update_current_camera)

    @staticmethod
    def init_from_checkpoint(
        path: str, device: torch.device, *args, **kwargs
    ) -> "Renderer":
        guru.info(f"Loading checkpoint from {path}")
        ckpt = torch.load(path)
        state_dict = ckpt["model"]
        model = SceneModel.init_from_state_dict(state_dict)
        model = model.to(device)
        renderer = Renderer(model, device, *args, **kwargs)
        renderer.global_step = ckpt.get("global_step", 0)
        renderer.epoch = ckpt.get("epoch", 0)
        return renderer

    @torch.inference_mode()
    def render_fn(self, camera_state: CameraState, img_wh: tuple[int, int]):
        if self.viewer is None:
            return np.full((img_wh[1], img_wh[0], 3), 255, dtype=np.uint8)

        W, H = img_wh

        focal = 0.5 * H / np.tan(0.5 * camera_state.fov).item()
        K = torch.tensor(
            [[focal, 0.0, W / 2.0], [0.0, focal, H / 2.0], [0.0, 0.0, 1.0]],
            device=self.device,
        )
        w2c = torch.linalg.inv(
            torch.from_numpy(camera_state.c2w.astype(np.float32)).to(self.device)
        )
        t = (
            int(self.viewer._playback_guis[0].value)
            if not self.viewer._canonical_checkbox.value
            else None
        )
        self.model.training = False

        # Get rendering mode options from viewer
        fg_only = self.viewer._fg_only_checkbox.value
        visible_only = self.viewer._visible_only_checkbox.value
        invisible_only = self.viewer._invisible_only_checkbox.value
        show_nodes = self.viewer._show_nodes_checkbox.value
        show_viewpoints = self.viewer._show_viewpoints_checkbox.value
        num_viewpoints = int(self.viewer._num_viewpoints_slider.value)

        # Create filter mask for visibility filtering
        filter_mask = None
        if (visible_only or invisible_only) and t is not None:
            # visibility filtering only makes sense when we have a specific time t
            # and when we have visibility information (fg gaussians only)
            if self.model.fg.visibilities is not None:
                # Get visibility at time t [G]
                vis_at_t = self.model.fg.visibilities[:, t] > 0.5

                # If both are checked, invisible_only takes priority
                if invisible_only:
                    vis_at_t = ~vis_at_t  # Invert to get invisible points

                if fg_only:
                    # Only filter FG gaussians
                    filter_mask = vis_at_t
                else:
                    # If rendering both FG and BG, only filter FG part
                    # Create a mask for all gaussians [num_fg + num_bg]
                    all_mask = torch.ones(
                        self.model.num_gaussians, dtype=torch.bool, device=self.device
                    )
                    all_mask[: self.model.num_fg_gaussians] = vis_at_t
                    filter_mask = all_mask

        img = self.model.render(
            t, w2c[None], K[None], img_wh, fg_only=fg_only, filter_mask=filter_mask,
            show_nodes=show_nodes
        )["img"][0]

        img = (img.cpu().numpy() * 255.0).astype(np.uint8)

        # Add viewpoint visualization if enabled
        if show_viewpoints:
            # Get foreground Gaussian centers
            fg_means = self.model.fg.params["means"]  # (G, 3)
            num_fg = fg_means.shape[0]

            # Sample num_viewpoints gaussians
            if num_viewpoints < num_fg:
                # Random sampling
                indices = torch.randperm(num_fg, device=self.device)[:num_viewpoints]
                sampled_means = fg_means[indices]
            else:
                sampled_means = fg_means

            # Transform to current time if t is not None
            if t is not None:
                transfms = self.model.motion_tree.compute_transforms_from_nodes(
                    torch.tensor([t], device=self.device), sampled_means
                )
                # Apply transformation: [N, 1, 4, 4] x [N, 4] -> [N, 1, 3]
                sampled_means = torch.einsum(
                    "nij,nj->ni",
                    transfms[:, 0],  # [N, 4, 4]
                    F.pad(sampled_means, (0, 1), value=1.0),  # [N, 4]
                )[..., :3]

            # Project to 2D
            points_3d_homo = F.pad(sampled_means, (0, 1), value=1.0)  # [N, 4]
            points_2d_homo = torch.einsum("ij,jk,nk->ni", K, w2c[:3], points_3d_homo)  # [N, 3]
            points_2d = points_2d_homo[:, :2] / points_2d_homo[:, 2:3]  # [N, 2]

            # Filter points that are in front of camera and within image bounds
            valid_mask = (
                (points_2d_homo[:, 2] > 0) &
                (points_2d[:, 0] >= 0) & (points_2d[:, 0] < W) &
                (points_2d[:, 1] >= 0) & (points_2d[:, 1] < H)
            )
            valid_points_2d = points_2d[valid_mask]

            # Convert to numpy if tensor
            if torch.is_tensor(img):
                img = (img.cpu().numpy() * 255.0).astype(np.uint8)

            # Draw red points
            import cv2
            for point in valid_points_2d.cpu().numpy():
                x, y = int(point[0]), int(point[1])
                cv2.circle(img, (x, y), radius=3, color=(255, 0, 0), thickness=-1)

        return img
