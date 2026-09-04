"""Shared read-only plane browser over one oblique_slices.h5.

Owns the open h5 handle, the (slice, group, plane) cursor, and one matplotlib
canvas that redraws the current plane with the group's stored cmap/clim.
LinePickerDialog and MarkPlanesDialog compose it and add their own controls;
owner overlays go through ``post_draw`` and owners resync labels on
``viewChanged``. Built lazily on dialog open — never at stage-view
construction.
"""

from __future__ import annotations

import h5py
import matplotlib.colors as mcolors
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from darq_xray.common import render as _rnd
from darq_xray.stages import profiles as _pr


class PlaneBrowser(QWidget):
    """One plane of one field group of one slice, with stepping and switching."""

    viewChanged = Signal()  # emitted after every redraw

    def __init__(self, h5_path, parent=None) -> None:
        super().__init__(parent)
        self._path = str(h5_path)
        self._f = h5py.File(self._path, "r")
        self.post_draw = None  # callable(ax) for owner overlays, or None
        self.slice_name: str | None = None
        self.group_id: str | None = None
        self.plane_index = 0
        self.present: list[str] = []
        self.u = self.v = self.offsets = None
        self.attrs: dict | None = None
        self._sg = None

        self._fig = Figure(figsize=(7, 6), layout="tight")
        self.canvas = FigureCanvasQTAgg(self._fig)
        self.ax = self._fig.add_subplot(111)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)

    # -- file ---------------------------------------------------------------
    @property
    def file(self):
        """The open h5py.File (read-only)."""
        return self._f

    def close_file(self) -> None:
        try:
            self._f.close()
        except Exception:  # noqa: BLE001 - already closed / never opened
            pass

    def reopen(self) -> None:
        """Re-open after an external write; re-binds the current view."""
        self._f = h5py.File(self._path, "r")
        if self.slice_name and self.group_id:
            self._sg = self._f[f"{self.group_id}/{self.slice_name}"]

    # -- navigation -----------------------------------------------------------
    def slice_names(self) -> list[str]:
        """All slice-group names in the file (union across field groups)."""
        names: set[str] = set()
        for vid in _pr.list_volume_ids(self._f):
            g = self._f[vid]
            for k in g:
                if isinstance(g[k], h5py.Group) and "slices" in g[k]:
                    names.add(str(k))
        return sorted(names)

    def open_slice(self, slice_name, *, ref_pref="", init_offset=0.0) -> None:
        present = _pr.volume_ids_with_slice(self._f, slice_name)
        if not present:
            raise KeyError(f"slice {slice_name!r} not present in {self._path}")
        self.slice_name = slice_name
        self.present = present
        self._bind_group(_pr._pick_reference_id(present, ref_pref))
        self.plane_index, _ = _pr.resolve_plane_index(self.offsets, init_offset)
        self.redraw()

    def _bind_group(self, vid) -> None:
        self.group_id = vid
        self._sg = self._f[f"{vid}/{self.slice_name}"]
        self.u, self.v, self.offsets = _pr.read_axes(self._sg)
        self.attrs = _pr.read_volume_attrs(self._f, vid)
        self.plane_index = max(0, min(self.plane_index, len(self.offsets) - 1))

    def set_group(self, vid) -> None:
        self._bind_group(vid)
        self.redraw()

    def set_plane(self, idx) -> None:
        self.plane_index = max(0, min(int(idx), len(self.offsets) - 1))
        self.redraw()

    def step(self, d) -> None:
        self.set_plane(self.plane_index + d)

    def current_offset(self) -> float:
        return float(self.offsets[self.plane_index])

    # -- drawing --------------------------------------------------------------
    def redraw(self) -> None:
        self.ax.clear()
        extent = [float(self.u[0]), float(self.u[-1]), float(self.v[0]), float(self.v[-1])]
        vmin, vmax = self.attrs["vmin"], self.attrs["vmax"]
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax) if vmin is not None else None
        self.ax.imshow(
            self._sg["slices"][self.plane_index].astype(float),
            cmap=_rnd.cmap_nan_transparent(self.attrs["cmap"]),
            norm=norm,
            extent=extent,
            origin="lower",
            aspect="equal",
        )
        self.ax.set_xlabel("u (µm)")
        self.ax.set_ylabel("v (µm)")
        if self.post_draw is not None:
            self.post_draw(self.ax)
        self.canvas.draw_idle()
        self.viewChanged.emit()
