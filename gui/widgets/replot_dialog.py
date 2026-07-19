"""Generic 2-level replot dialog (group → layer), built lazily on demand.

Consumes a catalog_fn + render callback from the Qt-free core, so it serves
strain/mosaicity/rocking uniformly. clim + pixel-index ROI crop override the
stored render; nothing is recomputed. The file field keeps cold-start working.
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

from .clim_section import ClimGroupSection
from .plane_selection import PlaneSelectionPanel
from .plane_selection_model import build_layer_rows, layer_selections


class ReplotDialog(QDialog):
    """Pick groups/layers from a replot catalog (via catalog_fn) and re-render PNGs."""

    def __init__(
        self,
        h5_default,
        catalog_fn,
        render_fn,
        style=None,
        out_default="",
        preview_fn=None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._h5_path = h5_default or ""
        self._catalog_fn = catalog_fn
        self._render_fn = render_fn
        self._style = style
        self._preview_fn = preview_fn
        self.written: list[str] = []
        self._ts = time.strftime("%Y%m%d-%H%M%S")
        # Explicit out_default pins the field; otherwise it tracks the loaded h5.
        self._out_pinned = bool(out_default)
        self.setWindowTitle("Replot")

        self._file_edit = QLineEdit(self._h5_path)
        file_browse = QPushButton("Browse…")
        file_browse.clicked.connect(self._on_browse_h5)
        file_reload = QPushButton("Load")
        file_reload.clicked.connect(self._reload)
        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("Volume file:"))
        file_row.addWidget(self._file_edit, 1)
        file_row.addWidget(file_browse)
        file_row.addWidget(file_reload)

        # planes-first selection panel (left: layers, listed once; right: quantities)
        self._panel = PlaneSelectionPanel(show_quantities=True)

        self._clim = ClimGroupSection()
        clim_header = QLabel("Colour limits (per plot kind; blank = stored):")

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

        self._out_edit = QLineEdit(out_default or self._default_out_for(self._h5_path))
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
            self._catalog = self._catalog_fn(self._h5_path)
        except Exception as exc:  # noqa: BLE001 — GUI reload: show status, never crash
            self._catalog = []
            self._panel.set_rows([])
            self._panel.set_quantities([])
            self._clim.set_groups([])
            self._status.setText(f"cannot read: {exc}")
            return
        self._clim.set_groups([(grp.key, grp.label) for grp in self._catalog])
        self._panel.set_rows(build_layer_rows(self._catalog))
        self._panel.set_quantities(
            [
                (
                    grp.key,
                    grp.label
                    if grp.shape is None
                    else f"{grp.label}   ·   {grp.shape[0]}×{grp.shape[1]} px (Y×X)",
                )
                for grp in self._catalog
            ]
        )
        self._status.setText(f"{len(self._catalog)} group(s)")

    def select_all(self) -> None:  # kept for smoke/back-compat
        self._panel.set_all_checked(True)

    def _selections(self):
        sels, self._skipped = layer_selections(
            self._catalog,
            self._panel.checked_plane_keys(),
            self._panel.checked_quantity_keys(),
        )
        return sels

    @staticmethod
    def _i(edit):
        t = edit.text().strip()
        return int(t) if t else None

    def _roi(self):
        vals = (self._i(self._r0), self._i(self._r1), self._i(self._c0), self._i(self._c1))
        return vals if all(v is not None for v in vals) else None  # all-four-or-none

    def _on_pick_roi(self) -> None:
        if not self._preview_fn or not self._h5_path:
            self._status.setText("load a volume file first to pick an ROI")
            return
        try:
            catalog = self._catalog_fn(self._h5_path)
        except Exception as exc:  # noqa: BLE001
            self._status.setText(f"cannot read: {exc}")
            return
        previews = [
            (grp.label, (lambda k=grp.key: self._preview_fn(self._h5_path, k))) for grp in catalog
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
        self.written = self._render_fn(
            self._h5_path,
            self._selections(),
            self._style,
            self._clim.clim_by_group(),
            self._roi(),
            out_dir,
        )
        return self.written

    def _on_render(self) -> None:
        out_dir = self._out_edit.text().strip()
        if not out_dir:
            self._status.setText("set an output dir")
            return
        if not self._selections():
            self._status.setText("nothing selected")
            return
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
        path, _ = QFileDialog.getOpenFileName(self, "Open volume h5", "", "HDF5 (*.h5)")
        if path:
            self._file_edit.setText(path)
            self._reload()

    def _on_browse_out(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Output directory")
        if path:
            self._out_pinned = True
            self._out_edit.setText(path)
