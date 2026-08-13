"""Lazy 3-D volume panel: a volume selector + a Render button over a PvCanvas.

Nothing is loaded or rendered (and pyvista is never imported) until the user
picks a volume and clicks "Render 3-D" — the sources handed in by the stage view
are VolumeSourceSpecs whose ``load`` callable is invoked only then.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..viewers import VolumeSourceSpec
from .pv_canvas import PvCanvas


class Volume3DPanel(QWidget):
    """Choose one of the run's volumes and render it interactively in 3-D."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sources: dict[str, VolumeSourceSpec] = {}

        self._combo = QComboBox()
        self._render_btn = QPushButton("Render 3-D")
        self._render_btn.clicked.connect(self._on_render)
        self._status = QLabel("(run the stage, then pick a volume to render)")
        self._status.setWordWrap(True)
        self._canvas = PvCanvas()

        top = QHBoxLayout()
        top.addWidget(QLabel("Volume:"))
        top.addWidget(self._combo, 1)
        top.addWidget(self._render_btn)

        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self._status)
        lay.addWidget(self._canvas, 1)
        self._set_enabled(False)

    def set_sources(self, sources: dict[str, VolumeSourceSpec]) -> None:
        """Install lazy volume sources (name -> VolumeSourceSpec). Does not render."""
        self._sources = dict(sources)
        self._combo.clear()
        self._combo.addItems(list(self._sources))
        has = bool(self._sources)
        self._set_enabled(has)
        self._status.setText(
            "pick a volume and click Render 3-D" if has else "(no 3-D volume from this run)"
        )

    def _set_enabled(self, on: bool) -> None:
        self._combo.setEnabled(on)
        self._render_btn.setEnabled(on)

    def _on_render(self) -> None:
        name = self._combo.currentText()
        if name not in self._sources:
            return
        self._status.setText(f"loading '{name}' …")
        self._render_btn.setEnabled(False)
        try:
            spec = self._sources[name]
            lv = spec.load()
            ok = self._canvas.show_volume(lv.volume, lv.spacing, cmap=lv.cmap, clim=lv.clim)
            self._status.setText(
                f"{name}: shape {tuple(lv.volume.shape)}"
                if ok
                else f"{name}: 3-D unavailable (no OpenGL context)"
            )
        except Exception as exc:  # noqa: BLE001 - surface any load/render error in the UI
            self._status.setText(f"render failed: {exc}")
        finally:
            self._render_btn.setEnabled(True)
