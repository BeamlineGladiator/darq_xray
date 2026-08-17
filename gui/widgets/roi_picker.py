"""Interactive ROI picker: drag a rectangle on a preview plane, get pixel bounds.

Source-agnostic and dumb — it imports nothing from ``dfxm`` beyond numpy-level
types. Each call site supplies previews as ``(label, thunk)`` pairs where
``thunk() -> (array2d, sx, sy)`` (lazy). The plane is drawn exactly like the
map exports (``origin="lower"``, physical aspect via ``set_aspect(sy/sx)``), so
the crop you draw is the crop you get. On accept, :attr:`result` is the
half-open ``(r0, r1, c0, c1)`` pixel-index tuple; otherwise ``None``.
"""

from __future__ import annotations

import math

import numpy as np


def rect_to_indices(xmin, xmax, ymin, ymax, w, h) -> tuple[int, int, int, int]:
    """Map a selector rectangle (data coords on pixel-edge extents) to half-open
    ``(r0, r1, c0, c1)`` pixel indices, clamped to ``[0, w]`` / ``[0, h]``.

    ``x`` is columns (X), ``y`` is rows (Y). floor(min)/ceil(max) on pixel-edge
    extents gives inclusive-of-touched-pixels behaviour with no ±0.5 fencepost.
    """
    c0 = max(0, min(int(math.floor(min(xmin, xmax))), w))
    c1 = max(0, min(int(math.ceil(max(xmin, xmax))), w))
    r0 = max(0, min(int(math.floor(min(ymin, ymax))), h))
    r1 = max(0, min(int(math.ceil(max(ymin, ymax))), h))
    return r0, r1, c0, c1


from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.widgets import RectangleSelector  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from .busy import busy_cursor  # noqa: E402


class ROIPickerDialog(QDialog):
    """Drag a rectangle on a preview plane; read back half-open pixel bounds."""

    def __init__(self, previews, initial=None, parent=None) -> None:
        super().__init__(parent)
        self._previews = list(previews)
        self._initial = initial
        self.result: tuple[int, int, int, int] | None = None
        self._arr: np.ndarray | None = None
        self._sx = self._sy = 1.0
        self._rect: tuple[float, float, float, float] | None = None  # xmin,xmax,ymin,ymax

        self.setWindowTitle("Pick ROI")
        self._combo = QComboBox()
        for label, _thunk in self._previews:
            self._combo.addItem(label)
        self._combo.currentIndexChanged.connect(self._load_current)

        self._fig = Figure(figsize=(5, 6), layout="tight")
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._ax = self._fig.add_subplot(111)
        self._selector = None
        self._readout = QLabel("drag a rectangle")

        self._reset = QPushButton("Reset")
        self._use = QPushButton("Use")
        self._cancel = QPushButton("Cancel")
        self._use.setEnabled(False)
        self._reset.clicked.connect(self._on_reset)
        self._use.clicked.connect(self._accept)
        self._cancel.clicked.connect(self.reject)

        top = QHBoxLayout()
        top.addWidget(QLabel("Preview:"))
        top.addWidget(self._combo, 1)
        btns = QHBoxLayout()
        btns.addWidget(self._readout, 1)
        btns.addWidget(self._reset)
        btns.addWidget(self._use)
        btns.addWidget(self._cancel)
        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self._canvas, 1)
        lay.addLayout(btns)

        if self._previews:
            self._load_current()

    def _current_shape(self) -> tuple[int, int] | None:
        return None if self._arr is None else (self._arr.shape[0], self._arr.shape[1])

    def _load_current(self) -> None:
        prev_shape = self._current_shape()
        idx = max(0, self._combo.currentIndex())
        try:
            with busy_cursor("loading preview…", widget=self._readout):
                arr, sx, sy = self._previews[idx][1]()
        except Exception as exc:  # noqa: BLE001 — bad path/dataset: show, don't crash
            self._arr = None
            self._readout.setText(f"cannot preview: {exc}")
            return
        self._arr, self._sx, self._sy = np.asarray(arr, dtype=float), float(sx), float(sy)
        h, w = self._arr.shape[:2]
        self._ax.clear()
        self._ax.imshow(self._arr, origin="lower", extent=[0, w, 0, h], cmap="magma", aspect="auto")
        self._ax.set_aspect(self._sy / self._sx)  # physical aspect (Y stretched by sy/sx)
        self._ax.set_xlabel("cols / X (px)")
        self._ax.set_ylabel("rows / Y (px)")
        self._selector = RectangleSelector(
            self._ax,
            onselect=lambda e_press, e_release: self._on_rect_change(
                e_press.xdata, e_release.xdata, e_press.ydata, e_release.ydata
            ),
            useblit=False,
            interactive=True,
            button=[1],
        )
        # keep the rect only if the new preview has the same shape
        if prev_shape is not None and prev_shape != (h, w):
            self._rect = None
            self._use.setEnabled(False)
            self._readout.setText(f"shape {h}×{w} px — previous selection cleared")
        elif self._rect is None and self._initial is not None:
            r0, r1, c0, c1 = self._initial
            if self._selector is not None:
                self._selector.extents = (c0, c1, r0, r1)
            self._on_rect_change(c0, c1, r0, r1)
        elif self._rect is not None:
            xmin, xmax, ymin, ymax = self._rect
            if self._selector is not None:
                self._selector.extents = (xmin, xmax, ymin, ymax)
            self._on_rect_change(xmin, xmax, ymin, ymax)
        self._canvas.draw_idle()

    def _on_rect_change(self, xmin, xmax, ymin, ymax) -> None:
        if self._arr is None or None in (xmin, xmax, ymin, ymax):
            return
        self._rect = (xmin, xmax, ymin, ymax)
        h, w = self._arr.shape[:2]
        r0, r1, c0, c1 = rect_to_indices(xmin, xmax, ymin, ymax, w, h)
        dr, dc = r1 - r0, c1 - c0
        self._use.setEnabled(dr >= 1 and dc >= 1)
        self._readout.setText(
            f"{dr}×{dc} px  =  {dr * self._sy:.1f} × {dc * self._sx:.1f} µm (Y×X)"
        )

    def _on_reset(self) -> None:
        self._rect = None
        self._use.setEnabled(False)
        self._readout.setText("drag a rectangle")
        if self._selector is not None:
            self._selector.set_visible(False)
        self._canvas.draw_idle()

    def _accept(self) -> None:
        if self._arr is None or self._rect is None:
            return
        h, w = self._arr.shape[:2]
        self.result = rect_to_indices(*self._rect, w=w, h=h)
        self.accept()
