"""Embedded pyvista 3-D view — created lazily, degrading gracefully if unavailable.

``pyvistaqt`` needs a working OpenGL/VTK context (often missing on a headless box
or plain X forwarding) and importing it is not cheap. So the ``QtInteractor`` is
built only on the first :meth:`ensure` / :meth:`show_volume` call, never at
construction — nothing about pyvista is touched until the user actually asks for
a 3-D render. Any import/GL failure degrades to a placeholder label.
"""

from __future__ import annotations

import numpy as np
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

    # -- rendering --------------------------------------------------------
    def show_volume(self, volume, spacing, cmap="magma", clim=None, opacity="linear") -> bool:
        """Volume-render a (Z, Y, X) array; NaN voxels are thresholded out.

        Returns True if rendered, False if 3-D is unavailable (placeholder shown).
        """
        if not self.ensure():
            return False
        import pyvista as pv

        self._plotter.clear()
        dt = np.transpose(np.asarray(volume, dtype=float), (2, 1, 0))  # -> (X, Y, Z)
        finite = dt[np.isfinite(dt)]
        sentinel = (
            (float(np.min(finite)) - 1000.0 * (float(np.ptp(finite)) + 1.0))
            if finite.size
            else -1e30
        )
        dc = np.where(np.isfinite(dt), dt, sentinel)
        grid = pv.ImageData()
        grid.dimensions = np.array(dc.shape) + 1
        grid.spacing = tuple(float(s) for s in spacing)
        grid.origin = (0.0, 0.0, 0.0)
        grid.cell_data["values"] = dc.flatten(order="F")
        thresh = sentinel * 0.5 if sentinel < 0 else sentinel + 1.0
        mesh = grid.threshold(value=thresh, scalars="values")
        if mesh.n_cells:
            self._plotter.add_mesh(
                mesh,
                scalars="values",
                cmap=cmap,
                clim=list(clim) if clim is not None else None,
                opacity=opacity,
                show_edges=False,
            )
            self._plotter.reset_camera()
        self._plotter.render()
        return True
