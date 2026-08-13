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
