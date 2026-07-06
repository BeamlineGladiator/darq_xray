"""Slice replot dialog (built lazily on demand).

Reads an oblique_slices.h5 straight from disk and re-renders selected planes
with the current publication style + an optional colour-limit override — no
resampling, and no prior stage run required (works from a cold start). All the
figure work happens in the Qt-free core (dfxm.stages.slices); this dialog is a
thin shell around slices.replot_catalog / slices.render_replot.
"""

from __future__ import annotations

import os

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


class SliceReplotDialog(QDialog):
    """Pick volumes/slices/planes from an oblique_slices.h5 and re-render PNGs."""

    def __init__(self, h5_path, style=None, out_default="", parent=None) -> None:
        super().__init__(parent)
        self._h5_path = h5_path
        self._style = style
        self.written: list[str] = []

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

        # clim override
        self._vmin = QLineEdit()
        self._vmin.setPlaceholderText("vmin (blank = stored)")
        self._vmax = QLineEdit()
        self._vmax.setPlaceholderText("vmax (blank = stored)")
        clim_row = QHBoxLayout()
        clim_row.addWidget(QLabel("Colour limits:"))
        clim_row.addWidget(self._vmin)
        clim_row.addWidget(self._vmax)

        # output dir
        self._out_edit = QLineEdit(out_default)
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
        layout.addLayout(clim_row)
        layout.addLayout(out_row)
        layout.addLayout(btn_row)

        self._reload()

    # -- population -----------------------------------------------------------
    def _reload(self) -> None:
        self._h5_path = self._file_edit.text().strip()
        self._tree.clear()
        if not self._h5_path or not os.path.exists(self._h5_path):
            self._status.setText("no such file")
            return
        try:
            catalog = _sl.replot_catalog(self._h5_path)
        except Exception as exc:  # noqa: BLE001 — GUI reload: show status, never crash
            self._status.setText(f"cannot read: {exc}")
            return
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
            snode = QTreeWidgetItem(vtop, [entry.slice_name])
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

    def _clim(self):
        def _f(edit):
            t = edit.text().strip()
            return float(t) if t else None

        try:
            vmin, vmax = _f(self._vmin), _f(self._vmax)
        except ValueError:
            return None
        if vmin is None and vmax is None:
            return None
        return (vmin, vmax)

    def render_selection(self, out_dir):
        """Render currently-checked planes into *out_dir*; returns written paths."""
        self.written = _sl.render_replot(
            self._h5_path, self._selections(), self._style, self._clim(), out_dir
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
        for label, edit in (("vmin", self._vmin), ("vmax", self._vmax)):
            t = edit.text().strip()
            if t:
                try:
                    float(t)
                except ValueError:
                    self._status.setText(f"invalid colour limit ({label}): {t!r}")
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
            self._out_edit.setText(path)
