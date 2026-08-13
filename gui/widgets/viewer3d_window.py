"""Pop-out 3-D viewer window — one volume, full controls, freed on close.

Each window owns its own lazy PvCanvas (GL context) and one
:class:`~dfxm.common.render3d.Scene3D`; closing the window closes the plotter
and drops the volume reference, returning the memory. All rendering setup goes
through ``render3d.populate`` so the view matches the stage's exports exactly.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

from dfxm.common import render3d as R3

from ..viewers import LoadedVolume, VolumeSourceSpec
from .pv_canvas import PvCanvas


class Viewer3DWindow(QWidget):
    """Interactive 3-D view of ONE volume with ParaView-style controls."""

    def __init__(self, spec: VolumeSourceSpec, stage_name: str, style_json: str = "") -> None:
        super().__init__(None, Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle(f"{stage_name} — {spec.name} (3D)")
        self.resize(1100, 750)
        self._spec = spec
        self._stage_name = stage_name
        self._style_json = style_json
        self.loaded: LoadedVolume | None = None
        self.scene: R3.Scene3D | None = None
        self._canvas = PvCanvas()
        self._status = QLabel("")
        self._status.setWordWrap(True)

        self._controls = QWidget()  # Task 9 fills this
        controls_scroll = QScrollArea()
        controls_scroll.setWidget(self._controls)
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFixedWidth(280)

        centre = QVBoxLayout()
        centre.addWidget(self._canvas, 1)
        centre.addWidget(self._status)
        lay = QHBoxLayout(self)
        row = QHBoxLayout()
        row.addLayout(centre, 1)
        row.addWidget(controls_scroll)
        lay.addLayout(row)

    # -- lifecycle --------------------------------------------------------
    def load_and_render(self) -> None:
        """Load the volume (heavy) and do the first render."""
        self.loaded = self._spec.load()
        self.scene = R3.Scene3D(
            volume=self.loaded.volume,
            spacing=self.loaded.spacing,
            cmap=self.loaded.cmap,
            clim=self.loaded.clim,
        )
        self.rebuild()
        if self._canvas.available:
            R3.apply_camera(self._canvas.plotter, R3.CameraSpec(preset="front"))

    def rebuild(self) -> None:
        """Clear and re-populate the plotter from the current scene."""
        if self.scene is None or not self._canvas.ensure():
            if not self._canvas.available:
                self._status.setText("3-D unavailable (no OpenGL context) — controls disabled")
            return
        pl = self._canvas.plotter
        pl.clear()
        ok = R3.populate(pl, self.scene, scalar_bar_title=self.loaded.cbar_label)
        self._status.setText(
            f"{self._spec.name}: shape {tuple(self.scene.volume.shape)}"
            if ok
            else "nothing to show (no finite voxels after threshold/clip)"
        )
        pl.render()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self._canvas.plotter is not None:
            self._canvas.plotter.close()
        self.loaded = None
        self.scene = None
        super().closeEvent(event)
