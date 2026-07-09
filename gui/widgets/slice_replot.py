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

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from dfxm.stages import slices as _sl

from .clim_section import ClimGroupSection

# Friendly labels for the per-quantity colour-limit rows, keyed by volume_id.
# volume_id is f"{kind}{suffix}" where suffix is ""/"_chi"/"_mu" (slices.py:_axis_suffix).
_KIND_LABELS = {
    "mosa_com": "Mosaicity COM",
    "mosa_fwhm": "Mosaicity FWHM",
    "strain": "Strain",
    "raw_sum": "Raw sum intensity",
    "raw_specific": "Raw frame",
    "raw_mosa_sum": "Raw mosa-sum intensity",
    "raw_mosa_specific": "Raw mosa frame",
}


def _volume_label(volume_id: str) -> str:
    """Human label for a clim row, e.g. 'mosa_com_chi' -> 'Mosaicity COM (χ)'."""
    for comp, sym in (("_chi", "χ"), ("_mu", "μ")):
        if volume_id.endswith(comp):
            base = volume_id[: -len(comp)]
            return f"{_KIND_LABELS.get(base, base)} ({sym})"
    return _KIND_LABELS.get(volume_id, volume_id)


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

        # volume → slice → plane tree (checkable)
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Volume / slice / plane"])
        self._tree.setColumnCount(1)

        # tree toolbar: Select all / Deselect all
        select_all_btn = QPushButton("Select all")
        select_all_btn.clicked.connect(self.select_all)
        deselect_btn = QPushButton("Deselect all")
        deselect_btn.clicked.connect(self._deselect_all)
        tree_toolbar = QHBoxLayout()
        tree_toolbar.addWidget(select_all_btn)
        tree_toolbar.addWidget(deselect_btn)
        tree_toolbar.addStretch(1)

        # per-kind clim override (one vmin/vmax row per kind present in the file)
        self._clim = ClimGroupSection()
        clim_header = QLabel("Colour limits (per plot kind; blank = stored):")

        # ROI crop override
        self._r0, self._r1 = QLineEdit(), QLineEdit()
        self._c0, self._c1 = QLineEdit(), QLineEdit()
        for e, ph in ((self._r0, "r0"), (self._r1, "r1"), (self._c0, "c0"), (self._c1, "c1")):
            e.setPlaceholderText(ph)
        roi_row = QHBoxLayout()
        roi_row.addWidget(QLabel("ROI crop (px, blank=full) — r=rows(Y), c=cols(X):"))
        for e in (self._r0, self._r1, self._c0, self._c1):
            roi_row.addWidget(e)

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
        render_btn = QPushButton("Render")
        render_btn.setProperty("role", "primary")
        render_btn.clicked.connect(self._on_render)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btn_row = QHBoxLayout()
        btn_row.addWidget(self._status, 1)
        btn_row.addWidget(render_btn)
        btn_row.addWidget(close_btn)

        layout = QVBoxLayout(self)
        layout.addLayout(file_row)
        layout.addWidget(self._tree, 1)
        layout.addLayout(tree_toolbar)
        layout.addWidget(clim_header)
        layout.addWidget(self._clim)
        layout.addLayout(roi_row)
        layout.addLayout(out_row)
        layout.addLayout(btn_row)

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
        self._tree.clear()
        if not self._out_pinned:
            self._out_edit.setText(self._default_out_for(self._h5_path))
        if not self._h5_path or not os.path.exists(self._h5_path):
            self._clim.set_groups([])
            self._status.setText("no such file")
            return
        try:
            catalog = _sl.replot_catalog(self._h5_path)
        except Exception as exc:  # noqa: BLE001 — GUI reload: show status, never crash
            self._clim.set_groups([])
            self._status.setText(f"cannot read: {exc}")
            return
        self._clim.set_groups(self._clim_groups(catalog))
        by_vid: dict[str, QTreeWidgetItem] = {}
        for entry in catalog:
            vtop = by_vid.get(entry.volume_id)
            if vtop is None:
                vtop = QTreeWidgetItem(self._tree, [entry.volume_id])
                vtop.setFlags(
                    vtop.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsAutoTristate
                )
                vtop.setCheckState(0, Qt.CheckState.Unchecked)
                by_vid[entry.volume_id] = vtop
            snode_label = entry.slice_name
            if entry.shape is not None:
                snode_label = f"{entry.slice_name}   ·   {entry.shape[0]}×{entry.shape[1]} px (Y×X)"
            snode = QTreeWidgetItem(vtop, [snode_label])
            snode.setFlags(
                snode.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsAutoTristate
            )
            snode.setCheckState(0, Qt.CheckState.Unchecked)
            snode.setData(0, Qt.ItemDataRole.UserRole, (entry.volume_id, entry.slice_name))
            for k, off in enumerate(entry.offsets_um):
                leaf = QTreeWidgetItem(snode, [f"plane {k}  ({off:+.2f} µm)"])
                leaf.setFlags(leaf.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                leaf.setCheckState(0, Qt.CheckState.Unchecked)
                leaf.setData(0, Qt.ItemDataRole.UserRole, (entry.volume_id, entry.slice_name, k))
        self._tree.expandAll()
        self.select_all()  # default: remake everything; user unticks to subset
        self._status.setText(f"{len(catalog)} slice group(s)")

    # -- bulk selection -------------------------------------------------------
    def select_all(self) -> None:
        """Check every volume node; auto-tristate cascades to all slices + planes."""
        for i in range(self._tree.topLevelItemCount()):
            self._tree.topLevelItem(i).setCheckState(0, Qt.CheckState.Checked)

    def _deselect_all(self) -> None:
        for i in range(self._tree.topLevelItemCount()):
            self._tree.topLevelItem(i).setCheckState(0, Qt.CheckState.Unchecked)

    # -- selection → core -----------------------------------------------------
    def _selections(self):
        """Collect checked (vid, slice, plane_idxs|None) tuples from the tree.

        Auto-tristate cascades a Checked parent down to all leaf (plane) children,
        so checking a volume or slice checks every leaf beneath it. Leaf states are
        read directly: all-checked → explicit index list; snode Checked with no
        leaves (zero-plane edge case) → None (all planes). Partially-checked slices
        yield only the checked leaf indices.
        """
        sels = []
        for i in range(self._tree.topLevelItemCount()):
            vtop = self._tree.topLevelItem(i)
            for j in range(vtop.childCount()):
                snode = vtop.child(j)
                vid, sname = snode.data(0, Qt.ItemDataRole.UserRole)
                checked = [
                    snode.child(k).data(0, Qt.ItemDataRole.UserRole)[2]
                    for k in range(snode.childCount())
                    if snode.child(k).checkState(0) == Qt.CheckState.Checked
                ]
                if snode.checkState(0) == Qt.CheckState.Checked and not checked:
                    sels.append((vid, sname, None))  # whole slice = all planes
                elif checked:
                    sels.append((vid, sname, checked))
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
