"""Pop-out 3-D viewer window — one volume, full controls, freed on close.

Each window owns its own lazy PvCanvas (GL context) and one
:class:`~dfxm.common.render3d.Scene3D`; closing the window closes the plotter
and drops the volume reference, returning the memory. All rendering setup goes
through ``render3d.populate`` so the view matches the stage's exports exactly.
"""

from __future__ import annotations

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from dfxm.common import cmaps as _cmaps
from dfxm.common import render3d as R3

from ..theme import ThemeController
from ..viewers import LoadedVolume, VolumeSourceSpec
from .pv_canvas import PvCanvas
from .wheel_guard import install_wheel_guard

# matplotlib names + the pipeline's registered ParaView-Fast cmap.
_cmaps.register()
_CMAP_NAMES = ("magma", "viridis", "plasma", "inferno", "RdBu_r", "gray", "fast")
_BACKGROUNDS = ("theme", "white", "black")
_CLIM_RANGE = (-1.0e12, 1.0e12)


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
        self._clip_flipped = False

        self._controls = QWidget()
        self._build_controls()
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

    # -- controls -----------------------------------------------------------
    def _build_controls(self) -> None:
        form = QFormLayout(self._controls)

        self._mode_combo = QComboBox()
        self._mode_combo.addItems(list(R3.RENDER_MODES))
        install_wheel_guard(self._mode_combo)
        self._mode_combo.currentTextChanged.connect(self._on_mode)
        form.addRow("Render mode", self._mode_combo)

        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(list(_CMAP_NAMES))
        install_wheel_guard(self._cmap_combo)
        self._cmap_combo.currentTextChanged.connect(self._on_cmap)
        form.addRow("Colormap", self._cmap_combo)

        self._clim_min = QDoubleSpinBox()
        self._clim_min.setRange(*_CLIM_RANGE)
        self._clim_min.setDecimals(6)
        install_wheel_guard(self._clim_min)
        self._clim_min.valueChanged.connect(self._on_clim)
        form.addRow("Colour min", self._clim_min)

        self._clim_max = QDoubleSpinBox()
        self._clim_max.setRange(*_CLIM_RANGE)
        self._clim_max.setDecimals(6)
        install_wheel_guard(self._clim_max)
        self._clim_max.valueChanged.connect(self._on_clim)
        form.addRow("Colour max", self._clim_max)

        self._clim_auto_btn = QPushButton("Auto colour range")
        self._clim_auto_btn.clicked.connect(self._auto_clim)
        form.addRow("", self._clim_auto_btn)

        self._log_check = QCheckBox("Log colour scale")
        self._log_check.toggled.connect(self._on_log)
        form.addRow("", self._log_check)

        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(0, 100)
        install_wheel_guard(self._opacity_slider)
        self._opacity_slider.valueChanged.connect(self._on_opacity)
        form.addRow("Opacity", self._opacity_slider)

        self._mapping_combo = QComboBox()
        self._mapping_combo.addItems(list(R3.OPACITY_MAPPINGS))
        install_wheel_guard(self._mapping_combo)
        self._mapping_combo.currentTextChanged.connect(self._on_mapping)
        form.addRow("Opacity mapping", self._mapping_combo)

        self._bg_combo = QComboBox()
        self._bg_combo.addItems(list(_BACKGROUNDS))
        install_wheel_guard(self._bg_combo)
        self._bg_combo.currentTextChanged.connect(self._on_background)
        form.addRow("Background", self._bg_combo)

        self._thresh_check = QCheckBox("Value threshold")
        self._thresh_check.toggled.connect(self._on_threshold)
        form.addRow("", self._thresh_check)

        self._thresh_min = QDoubleSpinBox()
        self._thresh_min.setRange(*_CLIM_RANGE)
        self._thresh_min.setDecimals(6)
        install_wheel_guard(self._thresh_min)
        self._thresh_min.valueChanged.connect(self._on_threshold_value)
        form.addRow("Threshold min", self._thresh_min)

        self._thresh_max = QDoubleSpinBox()
        self._thresh_max.setRange(*_CLIM_RANGE)
        self._thresh_max.setDecimals(6)
        install_wheel_guard(self._thresh_max)
        self._thresh_max.valueChanged.connect(self._on_threshold_value)
        form.addRow("Threshold max", self._thresh_max)

        self._downsample_spin = QSpinBox()
        self._downsample_spin.setRange(1, 16)
        install_wheel_guard(self._downsample_spin)
        self._downsample_spin.valueChanged.connect(self._on_downsample)
        form.addRow("Downsample", self._downsample_spin)

        self._clip_check = QCheckBox("Clip plane")
        self._clip_check.toggled.connect(self._on_clip)
        form.addRow("", self._clip_check)

        self._clip_axis_combo = QComboBox()
        self._clip_axis_combo.addItems(["X", "Y", "Z"])
        install_wheel_guard(self._clip_axis_combo)
        self._clip_axis_combo.currentTextChanged.connect(self._on_clip)
        form.addRow("Clip axis", self._clip_axis_combo)

        self._clip_flip_btn = QPushButton("Flip clip direction")
        self._clip_flip_btn.clicked.connect(self._on_clip_flip)
        form.addRow("", self._clip_flip_btn)

        cam_row = QHBoxLayout()
        self._cam_front = QPushButton("Front")
        self._cam_top = QPushButton("Top")
        self._cam_side = QPushButton("Side")
        self._cam_iso = QPushButton("Iso")
        for btn, preset in (
            (self._cam_front, "front"),
            (self._cam_top, "top"),
            (self._cam_side, "side"),
            (self._cam_iso, "iso"),
        ):
            btn.clicked.connect(lambda _checked=False, p=preset: self._apply_camera_preset(p))
            cam_row.addWidget(btn)
        cam_widget = QWidget()
        cam_widget.setLayout(cam_row)
        form.addRow("Camera preset", cam_widget)

        self._az_spin = QDoubleSpinBox()
        self._az_spin.setRange(-360.0, 360.0)
        install_wheel_guard(self._az_spin)
        form.addRow("Azimuth (°)", self._az_spin)

        self._el_spin = QDoubleSpinBox()
        self._el_spin.setRange(-90.0, 90.0)
        install_wheel_guard(self._el_spin)
        form.addRow("Elevation (°)", self._el_spin)

        self._zoom_spin = QDoubleSpinBox()
        self._zoom_spin.setRange(0.01, 100.0)
        self._zoom_spin.setValue(1.0)
        self._zoom_spin.setSingleStep(0.1)
        install_wheel_guard(self._zoom_spin)
        form.addRow("Zoom", self._zoom_spin)

        self._cam_apply_btn = QPushButton("Apply camera pose")
        self._cam_apply_btn.setToolTip(
            "Azimuth/elevation/zoom apply on top of the 'front' preset. These fields "
            "show the last APPLIED pose, not the live mouse-orbited view — exports use "
            "the live camera, not these fields."
        )
        self._cam_apply_btn.clicked.connect(self._apply_camera_fields)
        form.addRow("", self._cam_apply_btn)

        self._bounds_check = QCheckBox("Show bounds axes (µm)")
        self._bounds_check.toggled.connect(self._on_bounds)
        form.addRow("", self._bounds_check)

    def _init_controls_from_scene(self) -> None:
        """Sync every control's displayed value from ``self.scene`` (no signals)."""
        scene = self.scene
        with QSignalBlocker(self._mode_combo):
            self._mode_combo.setCurrentText(scene.mode)
        with QSignalBlocker(self._cmap_combo):
            if scene.cmap in _CMAP_NAMES:
                self._cmap_combo.setCurrentText(scene.cmap)
        vmin, vmax = scene.resolved_clim()
        with QSignalBlocker(self._clim_min):
            self._clim_min.setValue(vmin)
        with QSignalBlocker(self._clim_max):
            self._clim_max.setValue(vmax)
        with QSignalBlocker(self._opacity_slider):
            self._opacity_slider.setValue(round(scene.opacity * 100))
        with QSignalBlocker(self._mapping_combo):
            self._mapping_combo.setCurrentText(scene.opacity_mapping)
        # Default to the app theme's background (matches the canvas's own
        # themed background at creation, instead of Scene3D's plain "white").
        scene.background = ThemeController.instance().palette.pv_background
        with QSignalBlocker(self._bg_combo):
            self._bg_combo.setCurrentText("theme")
        with QSignalBlocker(self._log_check):
            self._log_check.setChecked(scene.log_scale)
        self._sync_log_enabled()
        with QSignalBlocker(self._thresh_check):
            self._thresh_check.setChecked(scene.threshold is not None)
        if scene.threshold is not None:
            with QSignalBlocker(self._thresh_min):
                self._thresh_min.setValue(scene.threshold[0])
            with QSignalBlocker(self._thresh_max):
                self._thresh_max.setValue(scene.threshold[1])
        with QSignalBlocker(self._downsample_spin):
            self._downsample_spin.setValue(int(scene.downsample))
        self._clip_flipped = False
        with QSignalBlocker(self._clip_axis_combo):
            self._clip_axis_combo.setCurrentText("X")
        with QSignalBlocker(self._clip_check):
            self._clip_check.setChecked(scene.clip is not None)
        with QSignalBlocker(self._bounds_check):
            self._bounds_check.setChecked(False)

    def _on_mode(self, text: str) -> None:
        self.scene.mode = text
        self.rebuild()

    def _on_cmap(self, text: str) -> None:
        self.scene.cmap = text
        self.rebuild()

    def _on_clim(self, _value: float) -> None:
        self.scene.clim = (self._clim_min.value(), self._clim_max.value())
        self._sync_log_enabled()
        self.rebuild()

    def _auto_clim(self) -> None:
        lo, hi = R3.auto_clim(self.scene.volume)
        self.scene.clim = (lo, hi)
        with QSignalBlocker(self._clim_min):
            self._clim_min.setValue(lo)
        with QSignalBlocker(self._clim_max):
            self._clim_max.setValue(hi)
        self._sync_log_enabled()
        self.rebuild()

    def _sync_log_enabled(self) -> None:
        ok = R3.log_valid(self.scene.resolved_clim())
        self._log_check.setEnabled(ok)
        self._log_check.setToolTip("" if ok else "log needs an all-positive colour range")
        if not ok and self._log_check.isChecked():
            self._log_check.setChecked(False)  # emits -> scene.log_scale=False + rebuild

    def _on_log(self, checked: bool) -> None:
        self.scene.log_scale = checked
        self.rebuild()

    def _on_opacity(self, value: int) -> None:
        self.scene.opacity = value / 100.0
        self.rebuild()

    def _on_mapping(self, text: str) -> None:
        self.scene.opacity_mapping = text
        self.rebuild()

    def _on_background(self, text: str) -> None:
        if text == "theme":
            self.scene.background = ThemeController.instance().palette.pv_background
        else:
            self.scene.background = text
        self.rebuild()

    def _on_threshold(self, checked: bool) -> None:
        self.scene.threshold = (
            (self._thresh_min.value(), self._thresh_max.value()) if checked else None
        )
        self.rebuild()

    def _on_threshold_value(self, _value: float) -> None:
        if self._thresh_check.isChecked():
            self.scene.threshold = (self._thresh_min.value(), self._thresh_max.value())
            self.rebuild()

    def _on_downsample(self, value: int) -> None:
        self.scene.downsample = value
        self.rebuild()

    def _current_clip(self) -> tuple | None:
        """Axis-aligned plane through the volume centre (v1 — no live vtk widget)."""
        if not self._clip_check.isChecked():
            return None
        vol, (sx, sy, sz) = self.scene.volume, self.scene.spacing
        z, y, x = vol.shape
        centre = (x * sx / 2.0, y * sy / 2.0, z * sz / 2.0)
        axis = {"X": 0, "Y": 1, "Z": 2}[self._clip_axis_combo.currentText()]
        normal = [0.0, 0.0, 0.0]
        normal[axis] = -1.0 if self._clip_flipped else 1.0
        return (centre, tuple(normal))

    def _on_clip(self, *_args) -> None:
        self.scene.clip = self._current_clip()
        self.rebuild()

    def _on_clip_flip(self) -> None:
        self._clip_flipped = not self._clip_flipped
        self.scene.clip = self._current_clip()
        self.rebuild()

    def _apply_camera(self, cam: R3.CameraSpec) -> None:
        if not self._canvas.available:
            return
        R3.apply_camera(self._canvas.plotter, cam)
        self._canvas.plotter.render()

    def _set_camera_fields(self, cam: R3.CameraSpec) -> None:
        with QSignalBlocker(self._az_spin):
            self._az_spin.setValue(cam.azimuth)
        with QSignalBlocker(self._el_spin):
            self._el_spin.setValue(cam.elevation)
        with QSignalBlocker(self._zoom_spin):
            self._zoom_spin.setValue(cam.zoom)

    def _apply_camera_preset(self, preset: str) -> None:
        cam = R3.CameraSpec(preset=preset)
        self._set_camera_fields(cam)
        self._apply_camera(cam)

    def _apply_camera_fields(self) -> None:
        """Apply the az/el/zoom fields on top of the 'front' preset (fields = last-applied pose)."""
        cam = R3.CameraSpec(
            preset="front",
            azimuth=self._az_spin.value(),
            elevation=self._el_spin.value(),
            zoom=self._zoom_spin.value(),
        )
        self._apply_camera(cam)

    def _on_bounds(self, checked: bool) -> None:
        if not self._canvas.available:
            return
        pl = self._canvas.plotter
        if checked:
            pl.show_bounds(xtitle="X (µm)", ytitle="Y (µm)", ztitle="Z (µm)")
        else:
            pl.remove_bounds_axes()
        pl.render()

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
        self._init_controls_from_scene()
        self.rebuild()
        if self._canvas.available:
            cam = R3.CameraSpec(preset="front")
            R3.apply_camera(self._canvas.plotter, cam)
            self._set_camera_fields(cam)

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
        # pl.clear() above also drops the bounds axes actor — re-apply it here
        # so toggling structural controls doesn't silently hide the bounds.
        if self._bounds_check.isChecked():
            pl.show_bounds(xtitle="X (µm)", ytitle="Y (µm)", ztitle="Z (µm)")
        pl.render()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self._canvas.plotter is not None:
            self._canvas.plotter.close()
        self.loaded = None
        self.scene = None
        super().closeEvent(event)
