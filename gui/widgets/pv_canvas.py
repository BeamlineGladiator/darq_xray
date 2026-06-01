"""Embedded pyvista 3-D view, degrading gracefully when unavailable.

``pyvistaqt`` needs a working OpenGL/VTK context, which may be missing on a
headless box or over plain X forwarding. Rather than crash the whole app, this
widget catches the failure and shows a placeholder; callers check
:attr:`available` before driving :attr:`plotter`.
"""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class PvCanvas(QWidget):
    """A pyvistaqt ``QtInteractor`` if one can be created, else a placeholder."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._plotter = None
        self._available = False
        self._init_plotter()

    def _init_plotter(self) -> None:
        try:
            from pyvistaqt import QtInteractor

            self._plotter = QtInteractor(self)
            self._layout.addWidget(self._plotter.interactor)
            self._available = True
        except Exception as exc:  # noqa: BLE001 - any import/GL failure degrades to a label
            label = QLabel(f"3-D view unavailable:\n{exc}")
            label.setWordWrap(True)
            self._layout.addWidget(label)
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    @property
    def plotter(self):
        return self._plotter

    def clear(self) -> None:
        if self._plotter is not None:
            self._plotter.clear()
