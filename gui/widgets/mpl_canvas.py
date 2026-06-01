"""Embedded matplotlib canvas with a click-pick signal.

Used by the output viewers and, later, by the interactive line-profile picker
(it replaces the legacy TkAgg click loop). Emits :attr:`clicked` with the data
coordinates of a left-click inside the axes.
"""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget


class MplCanvas(QWidget):
    """A Figure + navigation toolbar in a Qt widget, with click picking."""

    clicked = Signal(float, float)  # (xdata, ydata) of a click inside the axes

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=(5, 4), layout="tight")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)

        self.canvas.mpl_connect("button_press_event", self._on_click)

    def _on_click(self, event) -> None:
        if event.inaxes is self.ax and event.xdata is not None and event.ydata is not None:
            self.clicked.emit(float(event.xdata), float(event.ydata))

    # -- convenience ------------------------------------------------------
    def clear(self) -> None:
        self.ax.clear()
        self.canvas.draw_idle()

    def show_image(self, data, **imshow_kw):
        """Replace the axes content with ``imshow(data)`` and redraw."""
        self.ax.clear()
        im = self.ax.imshow(data, **imshow_kw)
        self.canvas.draw_idle()
        return im

    def draw(self) -> None:
        self.canvas.draw_idle()
