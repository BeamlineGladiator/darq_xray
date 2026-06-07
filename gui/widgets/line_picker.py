"""Interactive line picker for the profiles stage (built lazily on demand).

Replaces the legacy TkAgg click-loop with an embedded matplotlib-Qt canvas:
scroll the planes of one slice, click two endpoints, and read back
``(start_uv, end_uv, offset_um)`` in the slice's (u, v) frame. Opened only when
the user clicks "Pick line…", so the consolidated file is read and the canvas
built on demand, never at stage-view construction.
"""

from __future__ import annotations

import matplotlib.colors as mcolors
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from dfxm.common import render as _rnd
from dfxm.stages import profiles as _pr


class LinePickerDialog(QDialog):
    """Modal picker over one slice of an oblique_slices.h5 file.

    On accept, :attr:`result` is ``(start_uv, end_uv, offset_um)``; otherwise None.
    """

    def __init__(self, h5_path, slice_name, init_offset=0.0, ref_pref="", parent=None) -> None:
        super().__init__(parent)
        import h5py

        pr = _pr
        self.result = None
        self._pr = pr
        self._rnd = _rnd
        self._slice_name = slice_name
        self._pts: list[tuple[float, float]] = []

        self._f = h5py.File(h5_path, "r")
        present = pr.volume_ids_with_slice(self._f, slice_name)
        if not present:
            self._f.close()
            raise KeyError(f"slice {slice_name!r} not present in {h5_path}")
        self._ref_id = pr._pick_reference_id(present, ref_pref)
        self._sg = self._f[f"{self._ref_id}/{slice_name}"]
        self._u, self._v, self._offsets = pr.read_axes(self._sg)
        self._attrs = pr.read_volume_attrs(self._f, self._ref_id)
        self._idx, _ = pr.resolve_plane_index(self._offsets, init_offset)

        self.setWindowTitle(f"Pick line — {slice_name} ({self._ref_id})")
        self._fig = Figure(figsize=(7, 6), layout="tight")
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._ax = self._fig.add_subplot(111)
        self._im = None
        self._line_artist = None
        self._info = QLabel()

        self._prev = QPushButton("◀ plane")
        self._next = QPushButton("plane ▶")
        self._use = QPushButton("Use line")
        self._cancel = QPushButton("Cancel")
        self._use.setEnabled(False)
        self._prev.clicked.connect(lambda: self._step(-1))
        self._next.clicked.connect(lambda: self._step(+1))
        self._use.clicked.connect(self._accept)
        self._cancel.clicked.connect(self.reject)

        nav = QHBoxLayout()
        nav.addWidget(self._prev)
        nav.addWidget(self._next)
        nav.addStretch(1)
        nav.addWidget(self._use)
        nav.addWidget(self._cancel)

        lay = QVBoxLayout(self)
        lay.addWidget(self._canvas, 1)
        lay.addWidget(self._info)
        lay.addLayout(nav)

        self._canvas.mpl_connect("button_press_event", self._on_click)
        self._draw_plane()

    # -- plane display ----------------------------------------------------
    def _plane(self):
        return self._sg["slices"][self._idx].astype(float)

    def _draw_plane(self) -> None:
        self._ax.clear()
        extent = [float(self._u[0]), float(self._u[-1]), float(self._v[0]), float(self._v[-1])]
        vmin, vmax = self._attrs["vmin"], self._attrs["vmax"]
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax) if vmin is not None else None
        self._im = self._ax.imshow(
            self._plane(),
            cmap=self._rnd.cmap_nan_transparent(self._attrs["cmap"]),
            norm=norm,
            extent=extent,
            origin="lower",
            aspect="equal",
        )
        self._ax.set_xlabel("u (µm)")
        self._ax.set_ylabel("v (µm)")
        self._redraw_line()
        self._update_info()
        self._canvas.draw_idle()

    def _redraw_line(self) -> None:
        if len(self._pts) == 2:
            (u0, v0), (u1, v1) = self._pts
            self._ax.plot([u0, u1], [v0, v1], "-o", color="cyan", lw=2, ms=6, zorder=6)

    def _update_info(self) -> None:
        off = float(self._offsets[self._idx])
        pts = " -> ".join(f"({u:.2f}, {v:.2f})" for u, v in self._pts) or "click two points"
        self._info.setText(
            f"plane {self._idx + 1}/{len(self._offsets)}  offset {off:+.3f} µm   |   line: {pts}"
        )

    # -- interaction ------------------------------------------------------
    def _step(self, d: int) -> None:
        self._idx = max(0, min(self._idx + d, len(self._offsets) - 1))
        self._draw_plane()

    def _on_click(self, event) -> None:
        if event.inaxes is not self._ax or event.xdata is None or event.ydata is None:
            return
        if len(self._pts) >= 2:
            self._pts = []
        self._pts.append((float(event.xdata), float(event.ydata)))
        self._use.setEnabled(len(self._pts) == 2)
        self._draw_plane()

    def _accept(self) -> None:
        if len(self._pts) != 2:
            return
        self.result = (self._pts[0], self._pts[1], float(self._offsets[self._idx]))
        self.accept()

    # -- cleanup ----------------------------------------------------------
    def done(self, code) -> None:  # noqa: D401 - Qt override
        try:
            self._f.close()
        except Exception:  # noqa: BLE001 - already closed / never opened
            pass
        super().done(code)
