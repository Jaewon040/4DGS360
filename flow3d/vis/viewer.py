from pathlib import Path
from typing import Callable, Literal, Optional, Tuple, Union

import numpy as np
from jaxtyping import Float32, UInt8
from nerfview import CameraState, Viewer
from viser import Icon, ViserServer

from flow3d.vis.playback_panel import add_gui_playback_group
from flow3d.vis.render_panel import populate_render_tab


class DynamicViewer(Viewer):
    def __init__(
        self,
        server: ViserServer,
        render_fn: Callable[
            [CameraState, Tuple[int, int]],
            Union[
                UInt8[np.ndarray, "H W 3"],
                Tuple[UInt8[np.ndarray, "H W 3"], Optional[Float32[np.ndarray, "H W"]]],
            ],
        ],
        num_frames: int,
        work_dir: str,
        mode: Literal["rendering", "training"] = "rendering",
    ):
        self.num_frames = num_frames
        self.work_dir = Path(work_dir)
        super().__init__(server, render_fn, mode)

    def _define_guis(self):
        super()._define_guis()
        server = self.server
        self._time_folder = server.gui.add_folder("Time")
        with self._time_folder:
            self._playback_guis = add_gui_playback_group(
                server,
                num_frames=self.num_frames,
                initial_fps=15.0,
            )
            self._playback_guis[0].on_update(self.rerender)
            self._canonical_checkbox = server.gui.add_checkbox("Canonical", False)
            self._canonical_checkbox.on_update(self.rerender)

            _cached_playback_disabled = []

            def _toggle_gui_playing(event):
                if event.target.value:
                    nonlocal _cached_playback_disabled
                    _cached_playback_disabled = [
                        gui.disabled for gui in self._playback_guis
                    ]
                    target_disabled = [True] * len(self._playback_guis)
                else:
                    target_disabled = _cached_playback_disabled
                for gui, disabled in zip(self._playback_guis, target_disabled):
                    gui.disabled = disabled

            self._canonical_checkbox.on_update(_toggle_gui_playing)

        # Add rendering mode checkboxes
        self._fg_only_checkbox = server.gui.add_checkbox("FG Only", False)
        self._fg_only_checkbox.on_update(self.rerender)

        self._visible_only_checkbox = server.gui.add_checkbox("Visible Only", False)
        self._visible_only_checkbox.on_update(self.rerender)

        self._invisible_only_checkbox = server.gui.add_checkbox("Invisible Only", False)
        self._invisible_only_checkbox.on_update(self.rerender)

        self._show_nodes_checkbox = server.gui.add_checkbox("Show Nodes", False)
        self._show_nodes_checkbox.on_update(self.rerender)

        # Add viewpoint visualization controls
        self._show_viewpoints_checkbox = server.gui.add_checkbox("View Points", False)
        self._show_viewpoints_checkbox.on_update(self.rerender)

        self._num_viewpoints_slider = server.gui.add_slider(
            "Num View Points",
            min=10,
            max=1000,
            step=10,
            initial_value=100
        )
        self._num_viewpoints_slider.on_update(self.rerender)

        tabs = server.gui.add_tab_group()
        with tabs.add_tab("Render", Icon.CAMERA):
            self.render_tab_state = populate_render_tab(
                server, Path(self.work_dir) / "camera_paths", self._playback_guis[0]
            )
