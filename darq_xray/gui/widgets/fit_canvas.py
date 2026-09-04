"""WYSIWYG figure preview host.

A plain ``FigureCanvasQTAgg`` dropped into a stretch layout lets Qt's resize
rewrite the *figure size in inches* to whatever the widget happens to be — a
56-inch composed figure gets squeezed into a 13-inch canvas while every font
and line width (in points) stays put, so the preview shows 4× oversized,
overlapping text that the export never has. :class:`FitFigureHost` does the
opposite: it keeps the figure's true size in inches and scales its **dpi** so
the whole page fits the available area, then centres the fixed-size canvas.
Points therefore shrink with the page and the preview is a faithful thumbnail
of the export.
"""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QVBoxLayout, QWidget


def fit_dpi(size_in: tuple[float, float], avail_px: tuple[float, float]) -> float:
    """Largest dpi at which a ``size_in`` (w, h inches) page fits ``avail_px``
    (w, h device pixels); the tighter axis wins, floor 1.0."""
    w_in, h_in = float(size_in[0]), float(size_in[1])
    w_px, h_px = float(avail_px[0]), float(avail_px[1])
    cands = []
    if w_in > 0:
        cands.append(w_px / w_in)
    if h_in > 0:
        cands.append(h_px / h_in)
    return max(1.0, min(cands)) if cands else 1.0


class FitFigureHost(QWidget):
    """Hosts one composed ``Figure`` at its true size in inches, dpi-fitted to
    the widget. ``self.canvas`` is the live ``FigureCanvasQTAgg`` (for
    ``mpl_connect``); the figure's inches are never changed here."""

    def __init__(self, figure, parent=None):
        super().__init__(parent)
        self._size_in = tuple(float(v) for v in figure.get_size_inches())
        self.canvas = FigureCanvasQTAgg(figure)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas, 0, Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(1, 1)
        self._fit()

    @property
    def figure(self):
        return self.canvas.figure

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        self._fit()

    def _fit(self) -> None:
        w_in, h_in = self._size_in
        ratio = float(self.canvas.device_pixel_ratio or 1.0)
        # logical widget pixels available -> logical dpi
        dpi = fit_dpi((w_in, h_in), (max(1, self.width()), max(1, self.height())))
        fig = self.canvas.figure
        # Keep matplotlib's own DPR bookkeeping consistent: the canvas resets
        # figure.dpi to _original_dpi * ratio whenever the screen ratio changes.
        fig._original_dpi = dpi
        fig.set_dpi(dpi * ratio)
        # A FIXED canvas size makes FigureCanvasQT.resizeEvent's
        # set_size_inches(w/dpi, h/dpi) reproduce the true inches (rounded to
        # whole pixels) instead of the stretch layout's arbitrary box.
        w_px = max(1, int(round(w_in * dpi)))
        h_px = max(1, int(round(h_in * dpi)))
        self.canvas.setFixedSize(w_px, h_px)
        self.canvas.draw_idle()
