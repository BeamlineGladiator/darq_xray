"""Generic 2-level replot dialog (group → layer), built lazily on demand.

Consumes a catalog_fn + render callback from the Qt-free core, so it serves
strain/mosaicity/rocking uniformly. clim + pixel-index ROI crop override the
stored render; nothing is recomputed. The file field keeps cold-start working.
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


class ReplotDialog(QDialog):
    """Pick groups/layers from a replot catalog (via catalog_fn) and re-render PNGs."""

    def __init__(
        self, h5_default, catalog_fn, render_fn, style=None, out_default="", parent=None
    ) -> None:
        super().__init__(parent)
        self._h5_path = h5_default or ""
        self._catalog_fn = catalog_fn
        self._render_fn = render_fn
        self._style = style
        self.written: list[str] = []
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

        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Group / layer"])
        select_all_btn = QPushButton("Select all")
        select_all_btn.clicked.connect(self.select_all)
        deselect_btn = QPushButton("Deselect all")
        deselect_btn.clicked.connect(self._deselect_all)
        toolbar = QHBoxLayout()
        toolbar.addWidget(select_all_btn)
        toolbar.addWidget(deselect_btn)
        toolbar.addStretch(1)

        self._vmin = QLineEdit()
        self._vmin.setPlaceholderText("vmin")
        self._vmax = QLineEdit()
        self._vmax.setPlaceholderText("vmax")
        clim_row = QHBoxLayout()
        clim_row.addWidget(QLabel("Colour limits:"))
        clim_row.addWidget(self._vmin)
        clim_row.addWidget(self._vmax)

        self._r0, self._r1 = QLineEdit(), QLineEdit()
        self._c0, self._c1 = QLineEdit(), QLineEdit()
        for e, ph in ((self._r0, "r0"), (self._r1, "r1"), (self._c0, "c0"), (self._c1, "c1")):
            e.setPlaceholderText(ph)
        roi_row = QHBoxLayout()
        roi_row.addWidget(QLabel("ROI crop (px, blank=full):"))
        for e in (self._r0, self._r1, self._c0, self._c1):
            roi_row.addWidget(e)

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
        layout.addLayout(toolbar)
        layout.addLayout(clim_row)
        layout.addLayout(roi_row)
        layout.addLayout(out_row)
        layout.addLayout(btn_row)
        self._reload()

    def _reload(self) -> None:
        self._h5_path = self._file_edit.text().strip()
        self._tree.clear()
        if not self._h5_path or not os.path.exists(self._h5_path):
            self._status.setText("no such file")
            return
        try:
            catalog = self._catalog_fn(self._h5_path)
        except Exception as exc:  # noqa: BLE001 — GUI reload: show status, never crash
            self._status.setText(f"cannot read: {exc}")
            return
        for grp in catalog:
            top = QTreeWidgetItem(self._tree, [grp.label])
            top.setFlags(
                top.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsAutoTristate
            )
            top.setCheckState(0, Qt.CheckState.Unchecked)
            top.setData(0, Qt.ItemDataRole.UserRole, grp.key)
            for z, lab in enumerate(grp.item_labels):
                leaf = QTreeWidgetItem(top, [lab])
                leaf.setFlags(leaf.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                leaf.setCheckState(0, Qt.CheckState.Unchecked)
                leaf.setData(0, Qt.ItemDataRole.UserRole, z)
        self._tree.expandAll()
        self._status.setText(f"{self._tree.topLevelItemCount()} group(s)")

    def select_all(self) -> None:
        for i in range(self._tree.topLevelItemCount()):
            self._tree.topLevelItem(i).setCheckState(0, Qt.CheckState.Checked)

    def _deselect_all(self) -> None:
        for i in range(self._tree.topLevelItemCount()):
            self._tree.topLevelItem(i).setCheckState(0, Qt.CheckState.Unchecked)

    def _selections(self):
        sels = []
        for i in range(self._tree.topLevelItemCount()):
            top = self._tree.topLevelItem(i)
            key = top.data(0, Qt.ItemDataRole.UserRole)
            checked = [
                top.child(k).data(0, Qt.ItemDataRole.UserRole)
                for k in range(top.childCount())
                if top.child(k).checkState(0) == Qt.CheckState.Checked
            ]
            if top.checkState(0) == Qt.CheckState.Checked and len(checked) == top.childCount():
                sels.append((key, None))  # whole group = all layers
            elif checked:
                sels.append((key, checked))
        return sels

    @staticmethod
    def _f(edit):
        t = edit.text().strip()
        return float(t) if t else None

    @staticmethod
    def _i(edit):
        t = edit.text().strip()
        return int(t) if t else None

    def _clim(self):
        vmin, vmax = self._f(self._vmin), self._f(self._vmax)
        return None if (vmin is None and vmax is None) else (vmin, vmax)

    def _roi(self):
        vals = (self._i(self._r0), self._i(self._r1), self._i(self._c0), self._i(self._c1))
        return vals if all(v is not None for v in vals) else None  # all-four-or-none

    def render_selection(self, out_dir):
        self.written = self._render_fn(
            self._h5_path, self._selections(), self._style, self._clim(), self._roi(), out_dir
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
        try:
            written = self.render_selection(out_dir)
        except Exception as exc:  # noqa: BLE001 — surface render errors in the status bar
            self._status.setText(f"render failed: {exc}")
            return
        self._status.setText(f"wrote {len(written)} PNG(s) → {out_dir}")

    def _on_browse_h5(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open volume h5", "", "HDF5 (*.h5)")
        if path:
            self._file_edit.setText(path)
            self._reload()

    def _on_browse_out(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Output directory")
        if path:
            self._out_edit.setText(path)
