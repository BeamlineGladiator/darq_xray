"""Lazy 3-D volume launcher: a volume selector + a button that opens a pop-out window.

Nothing is loaded or rendered (and pyvista is never imported) until the user
picks a volume and clicks "Open 3D viewer…" — the sources handed in by the
stage view are VolumeSourceSpecs whose ``load`` callable is invoked only then,
inside the freshly opened :class:`~gui.widgets.viewer3d_window.Viewer3DWindow`.
Each window owns its own GL context and volume; closing it frees both, so
opening several volumes never accumulates memory in the main window.
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
from .viewer3d_window import Viewer3DWindow


class Volume3DPanel(QWidget):
    """Choose one of the run's volumes and open it in an independent 3-D viewer window."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sources: dict[str, VolumeSourceSpec] = {}
        self._windows: list[Viewer3DWindow] = []
        self._stage_name = ""

        self._combo = QComboBox()
        self._open_btn = QPushButton("Open 3D viewer…")
        self._open_btn.clicked.connect(self._on_open)
        self._status = QLabel("(run the stage, then open a volume in the 3-D viewer)")
        self._status.setWordWrap(True)

        top = QHBoxLayout()
        top.addWidget(QLabel("Volume:"))
        top.addWidget(self._combo, 1)
        top.addWidget(self._open_btn)

        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self._status)
        lay.addStretch(1)
        self._set_enabled(False)

    def set_sources(self, sources: dict[str, VolumeSourceSpec], stage_name: str = "") -> None:
        """Install lazy volume sources (name -> VolumeSourceSpec). Does not open anything."""
        self._sources = dict(sources)
        self._stage_name = stage_name
        self._combo.clear()
        self._combo.addItems(list(self._sources))
        has = bool(self._sources)
        self._set_enabled(has)
        self._status.setText(
            "pick a volume and click Open 3D viewer…" if has else "(no 3-D volume from this run)"
        )

    def _set_enabled(self, on: bool) -> None:
        self._combo.setEnabled(on)
        self._open_btn.setEnabled(on)

    def _on_open(self) -> None:
        spec = self._sources.get(self._combo.currentText())
        if spec is None:
            return
        w = Viewer3DWindow(spec, self._stage_name or "stage")
        self._windows.append(w)
        w.destroyed.connect(
            lambda *_a, _w=w: self._windows.remove(_w) if _w in self._windows else None
        )
        w.show()
        w.load_and_render()
