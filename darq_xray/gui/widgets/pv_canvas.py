"""Embedded pyvista 3-D view — created lazily, degrading gracefully if unavailable.

``pyvistaqt`` needs a working OpenGL/VTK context (often missing on a headless box
or plain X forwarding) and importing it is not cheap. So the ``QtInteractor`` is
built only on the first :meth:`ensure` call, never at construction — nothing
about pyvista is touched until the user actually asks for a 3-D render. Any
import/GL failure degrades to a placeholder label.
"""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ..theme import ThemeController


class PvCanvas(QWidget):
    """A pyvistaqt ``QtInteractor`` built on demand, else a placeholder label."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._plotter = None
        self._placeholder: QLabel | None = None
        self._tried = False
        self._available = False
        ThemeController.instance().themeChanged.connect(self.apply_theme)

    # -- lazy initialisation ---------------------------------------------
    def ensure(self) -> bool:
        """Create the QtInteractor on first use. Returns True if 3-D is usable."""
        if self._tried:
            return self._available
        self._tried = True
        try:
            from pyvistaqt import QtInteractor

            self._plotter = QtInteractor(self)
            self._plotter.set_background(ThemeController.instance().palette.pv_background)
            self._layout.addWidget(self._plotter.interactor)
            self._available = True
        except Exception as exc:  # noqa: BLE001 - any import/GL failure -> label
            self._placeholder = QLabel(f"3-D view unavailable:\n{exc}")
            self._placeholder.setWordWrap(True)
            self._layout.addWidget(self._placeholder)
            self._available = False
        return self._available

    @property
    def available(self) -> bool:
        return self._available

    @property
    def plotter(self):
        return self._plotter

    def apply_theme(self, palette) -> None:
        """Recolour the 3-D background; no-op until the plotter exists."""
        if self._plotter is not None:
            self._plotter.set_background(palette.pv_background)

    def clear(self) -> None:
        if self._plotter is not None:
            self._plotter.clear()
