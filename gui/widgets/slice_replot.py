"""Slice replot dialog (built lazily on demand).

Reads an oblique_slices.h5 straight from disk and re-renders selected planes
with the current publication style + an optional colour-limit override — no
resampling, and no prior stage run required (works from a cold start). All the
figure work happens in the Qt-free core (dfxm.stages.slices); this dialog is a
thin shell around slices.replot_catalog / slices.render_replot.
"""

from __future__ import annotations

import os
import time

from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from dfxm.stages import slices as _sl

from .clim_section import ClimGroupSection
from .clim_section import volume_label as _volume_label
from .plane_selection import PlaneSelectionPanel
from .plane_selection_model import build_slice_rows, slice_selections


class SliceReplotDialog(QDialog):
    """Pick volumes/slices/planes from an oblique_slices.h5 and re-render PNGs."""

    def __init__(self, h5_path, style=None, out_default="", parent=None) -> None:
        super().__init__(parent)
        self._h5_path = h5_path
        self._style = style
        self.written: list[str] = []
        self._ts = time.strftime("%Y%m%d-%H%M%S")
        # Explicit out_default pins the field; otherwise it tracks the loaded h5.
        self._out_pinned = bool(out_default)

        self.setWindowTitle(f"Replot slices — {os.path.basename(h5_path)}")

        # file field (browsable; defaults to the passed h5, editable for a cold start)
        self._file_edit = QLineEdit(h5_path)
        file_browse = QPushButton("Browse…")
        file_browse.clicked.connect(self._on_browse_h5)
        file_reload = QPushButton("Load")
        file_reload.clicked.connect(self._reload)
        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("Slices file:"))
        file_row.addWidget(self._file_edit, 1)
        file_row.addWidget(file_browse)
        file_row.addWidget(file_reload)

        # planes-first selection panel (left: planes, listed once; right: quantities)
        self._panel = PlaneSelectionPanel(show_quantities=True)

        # per-kind clim override (one vmin/vmax row per kind present in the file)
        self._clim = ClimGroupSection()
        clim_header = QLabel("Colour limits (per quantity; blank = stored):")

        # ROI crop override
        self._r0, self._r1 = QLineEdit(), QLineEdit()
        self._c0, self._c1 = QLineEdit(), QLineEdit()
        for e, ph in ((self._r0, "r0"), (self._r1, "r1"), (self._c0, "c0"), (self._c1, "c1")):
            e.setPlaceholderText(ph)
        self._pick_roi_btn = QPushButton("Pick ROI…")
        self._pick_roi_btn.clicked.connect(self._on_pick_roi)
        roi_row = QHBoxLayout()
        roi_row.addWidget(QLabel("ROI crop (px, blank=full) — r=rows(Y), c=cols(X):"))
        for e in (self._r0, self._r1, self._c0, self._c1):
            roi_row.addWidget(e)
        roi_row.addWidget(self._pick_roi_btn)

        # output dir (defaults to a subfolder beside the loaded slices file)
        self._out_edit = QLineEdit(out_default or self._default_out_for(h5_path))
        self._out_edit.textEdited.connect(self._on_out_edited)
        out_browse = QPushButton("Browse…")
        out_browse.clicked.connect(self._on_browse_out)
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output dir:"))
        out_row.addWidget(self._out_edit, 1)
        out_row.addWidget(out_browse)

        self._status = QLabel("")
        self._render_btn = QPushButton("Render")
        self._render_btn.setProperty("role", "primary")
        self._render_btn.clicked.connect(self._on_render)
        self._panel.selectionChanged.connect(
            lambda: self._render_btn.setEnabled(self._panel.has_selection())
        )
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btn_row = QHBoxLayout()
        btn_row.addWidget(self._status, 1)
        btn_row.addWidget(self._render_btn)
        btn_row.addWidget(close_btn)

        layout = QVBoxLayout(self)
        layout.addLayout(file_row)
        layout.addWidget(self._panel, 1)
        layout.addWidget(clim_header)
        layout.addWidget(self._clim)
        layout.addLayout(roi_row)
        layout.addLayout(out_row)
        layout.addLayout(btn_row)

        self._catalog: list = []
        self._skipped: list[str] = []
        self._reload()

    def _default_out_for(self, h5_path: str) -> str:
        """Default output dir = a timestamped subfolder beside the loaded h5."""
        if not h5_path:
            return ""
        return os.path.join(os.path.dirname(os.path.abspath(h5_path)), "replots", self._ts)

    def _on_out_edited(self, _text: str) -> None:
        self._out_pinned = True

    # -- population -----------------------------------------------------------
    def _reload(self) -> None:
        self._h5_path = self._file_edit.text().strip()
        if not self._out_pinned:
            self._out_edit.setText(self._default_out_for(self._h5_path))
        if not self._h5_path or not os.path.exists(self._h5_path):
            self._catalog = []
            self._panel.set_rows([])
            self._panel.set_quantities([])
            self._clim.set_groups([])
            self._status.setText("no such file")
            return
        try:
            self._catalog = _sl.replot_catalog(self._h5_path)
        except Exception as exc:  # noqa: BLE001 — GUI reload: show status, never crash
            self._catalog = []
            self._panel.set_rows([])
            self._panel.set_quantities([])
            self._clim.set_groups([])
            self._status.setText(f"cannot read: {exc}")
            return
        self._clim.set_groups(self._clim_groups(self._catalog))
        self._panel.set_rows(
            build_slice_rows(self._catalog), section_labels=self._section_labels(self._catalog)
        )
        vids = list(dict.fromkeys(e.volume_id for e in self._catalog))
        self._panel.set_quantities([(vid, _volume_label(vid)) for vid in vids])
        self._status.setText(f"{len(self._catalog)} slice group(s)")

    def select_all(self) -> None:  # kept for smoke/back-compat
        self._panel.set_all_checked(True)

    @staticmethod
    def _section_labels(catalog) -> dict[str, str]:
        """Annotate each slice-group section with its stored plane pixel size.

        A slice_name shared by several volumes normally stores the same (nv, nu)
        shape everywhere; when it doesn't (a mixed-grid file), the ROI-crop bound
        isn't a single number, so the picker is the source of truth instead.
        """
        shapes_by_sname: dict[str, set] = {}
        for e in catalog:
            if e.shape is not None:
                shapes_by_sname.setdefault(e.slice_name, set()).add(tuple(e.shape))
        labels: dict[str, str] = {}
        for sname, shapes in shapes_by_sname.items():
            if len(shapes) == 1:
                (nv, nu) = next(iter(shapes))
                labels[sname] = f"{sname}   ·   {nv}×{nu} px (Y×X)"
            else:
                labels[sname] = f"{sname}   ·   mixed grids — see Pick ROI…"
        return labels

    # -- selection → core -----------------------------------------------------
    def _selections(self):
        sels, self._skipped = slice_selections(
            self._catalog,
            self._panel.checked_plane_keys(),
            self._panel.checked_quantity_keys(),
        )
        return sels

    @staticmethod
    def _clim_groups(catalog):
        """One (volume_id, label) row per distinct quantity, in first-seen order."""
        vids = list(dict.fromkeys(e.volume_id for e in catalog))
        return [(vid, _volume_label(vid)) for vid in vids]

    def _roi(self):
        def _i(edit):
            t = edit.text().strip()
            return int(t) if t else None

        vals = (_i(self._r0), _i(self._r1), _i(self._c0), _i(self._c1))
        if all(v is None for v in vals):
            return None
        if any(v is None for v in vals):
            return None  # partial ROI ignored; keep parity with the four-box contract
        return vals

    def _on_pick_roi(self) -> None:
        if not self._h5_path or not os.path.exists(self._h5_path):
            self._status.setText("load a slices file first to pick an ROI")
            return
        try:
            catalog = _sl.replot_catalog(self._h5_path)
        except Exception as exc:  # noqa: BLE001
            self._status.setText(f"cannot read: {exc}")
            return
        previews = [
            (
                f"{e.volume_id} · {e.slice_name}",
                (lambda v=e.volume_id, s=e.slice_name: _sl.plane_preview(self._h5_path, v, s)),
            )
            for e in catalog
        ]
        if not previews:
            self._status.setText("nothing to preview")
            return
        import sys

        _mod = sys.modules[__name__]
        if not hasattr(_mod, "ROIPickerDialog"):
            from .roi_picker import ROIPickerDialog  # imported on demand

            _mod.ROIPickerDialog = ROIPickerDialog
        dlg = _mod.ROIPickerDialog(previews, initial=self._roi(), parent=self)
        if dlg.exec() and dlg.result:
            r0, r1, c0, c1 = dlg.result
            for edit, val in ((self._r0, r0), (self._r1, r1), (self._c0, c0), (self._c1, c1)):
                edit.setText(str(val))

    def render_selection(self, out_dir):
        """Render currently-checked planes into *out_dir*; returns written paths."""
        self.written = _sl.render_replot(
            self._h5_path,
            self._selections(),
            self._style,
            self._clim.clim_by_group(),
            out_dir,
            roi=self._roi(),
        )
        return self.written

    # -- slots ----------------------------------------------------------------
    def _on_render(self) -> None:
        out_dir = self._out_edit.text().strip()
        if not out_dir:
            self._status.setText("set an output dir")
            return
        sels = self._selections()
        if not sels:
            self._status.setText("nothing selected")
            return
        # Validate colour-limit boxes before rendering: non-empty but unparseable → error
        err = self._clim.validate()
        if err:
            self._status.setText(err)
            return
        try:
            written = self.render_selection(out_dir)
        except Exception as exc:  # noqa: BLE001 — surface render errors in the status bar
            self._status.setText(f"render failed: {exc}")
            return
        self._status.setText(f"wrote {len(written)} PNG(s) → {out_dir}")
        if self._skipped:
            self._status.setText(self._status.text() + f"; skipped {len(self._skipped)} combo(s)")

    def _on_browse_h5(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open oblique_slices.h5", "", "HDF5 (*.h5)")
        if path:
            self._file_edit.setText(path)
            self._reload()

    def _on_browse_out(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Output directory")
        if path:
            self._out_pinned = True
            self._out_edit.setText(path)
