"""Pin planes… dialog: pick sweep planes to pin, emit pinned_slices_json.

Loads a swept oblique_slices.h5, lists slice group -> planes with the shared
planes-first panel (same number/offset filter as the replot dialogs), and on
OK builds exact-geometry pinned specs via the Qt-free
``dfxm.stages.slices.build_pinned_spec``. The caller (slices StageView) writes
the JSON into the form and ticks 'Run pinned planes only'.
"""

from __future__ import annotations

import json
import os

from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from dfxm.common.errors import StageUserError
from dfxm.stages import slices as _sl

from .plane_selection import PlaneSelectionPanel
from .plane_selection_model import build_slice_rows


class PinPlanesDialog(QDialog):
    """Pick planes from a swept oblique_slices.h5; OK yields result_json."""

    def __init__(self, h5_default: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Pin planes")
        self.result_json: str | None = None
        self._h5_path = h5_default or ""

        self._file_edit = QLineEdit(self._h5_path)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._on_browse)
        load = QPushButton("Load")
        load.clicked.connect(self._reload)
        frow = QHBoxLayout()
        frow.addWidget(QLabel("Slices file:"))
        frow.addWidget(self._file_edit, 1)
        frow.addWidget(browse)
        frow.addWidget(load)

        self._panel = PlaneSelectionPanel(show_quantities=False)
        self._status = QLabel("")
        ok = QPushButton("OK")
        ok.setProperty("role", "primary")
        ok.clicked.connect(self._on_ok)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        brow = QHBoxLayout()
        brow.addWidget(self._status, 1)
        brow.addWidget(ok)
        brow.addWidget(cancel)

        layout = QVBoxLayout(self)
        layout.addLayout(frow)
        layout.addWidget(self._panel, 1)
        layout.addLayout(brow)
        self._reload()

    def _reload(self) -> None:
        self._h5_path = self._file_edit.text().strip()
        if not self._h5_path or not os.path.exists(self._h5_path):
            self._panel.set_rows([])
            self._status.setText("no such file")
            return
        try:
            catalog = _sl.replot_catalog(self._h5_path)
        except Exception as exc:  # noqa: BLE001 — show inline, never crash
            self._panel.set_rows([])
            self._status.setText(f"cannot read: {exc}")
            return
        self._panel.set_rows(build_slice_rows(catalog))
        self._panel.set_all_checked(False)  # pinning = explicit picks, not all
        self._status.setText(f"{len(self._panel._rows)} plane(s) — check the ones to pin")

    def _on_ok(self) -> None:
        keys = self._panel.checked_plane_keys()
        if not keys:
            self._status.setText("no planes checked")
            return
        row_by_key = {r.key: r for r in self._panel._rows}
        by_slice: dict[str, list[float]] = {}
        for sname, _idx in keys:
            by_slice.setdefault(sname, []).append(row_by_key[(sname, _idx)].offset)
        specs: list[dict] = []
        try:
            for sname, offs in by_slice.items():
                specs.extend(_sl.build_pinned_spec(self._h5_path, sname, offs))
        except StageUserError as exc:
            self._status.setText(f"{exc}\n\n{exc.hint}" if exc.hint else str(exc))
            return
        self.result_json = json.dumps(specs, indent=2)
        self.accept()

    def _on_browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open oblique_slices.h5", "", "HDF5 (*.h5)")
        if path:
            self._file_edit.setText(path)
            self._reload()
