"""Profiles replot dialog (built lazily on demand).

Re-renders profile jobs cold from an oblique_slices.h5 with per-quantity
colour-limit overrides — overviews, companion and traces only, never CSVs.
Jobs come from the profiles form's Jobs (JSON); all figure work happens in the
Qt-free core (dfxm.stages.profiles.render_replot); this dialog is a thin shell.
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
    QSpinBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from dfxm.stages import profiles as _pr

from .clim_section import ClimGroupSection, volume_label


class ProfilesReplotDialog(QDialog):
    """Pick jobs/fields from the form's jobs list and re-render profile figures."""

    def __init__(self, h5_path, jobs, style=None, out_default="", parent=None) -> None:
        super().__init__(parent)
        self._h5_path = h5_path
        self._jobs = [j for j in (jobs or []) if isinstance(j, dict) and "name" in j]
        self._style = style
        self.written: list[str] = []
        self._ts = time.strftime("%Y%m%d-%H%M%S")
        self._out_pinned = bool(out_default)

        self.setWindowTitle(f"Replot profiles — {os.path.basename(h5_path or '(no file)')}")

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

        # job → fields checkbox tree
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Job / field"])
        self._tree.itemChanged.connect(self._on_item_changed)

        self._clim = ClimGroupSection()
        clim_header = QLabel("Colour limits (per quantity; blank = stored):")

        self._out_edit = QLineEdit(out_default or self._default_out_for(h5_path))
        self._out_edit.textEdited.connect(lambda _t: setattr(self, "_out_pinned", True))
        out_browse = QPushButton("Browse…")
        out_browse.clicked.connect(self._on_browse_out)
        self._dpi = QSpinBox()
        self._dpi.setRange(50, 1200)
        self._dpi.setValue(int(_pr.STAGE.defaults()["fig_dpi"]))
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output dir:"))
        out_row.addWidget(self._out_edit, 1)
        out_row.addWidget(out_browse)
        out_row.addWidget(QLabel("DPI:"))
        out_row.addWidget(self._dpi)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._render_btn = QPushButton("Render")
        self._render_btn.setProperty("role", "primary")
        self._render_btn.clicked.connect(self._on_render)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btn_row = QHBoxLayout()
        btn_row.addWidget(self._status, 1)
        btn_row.addWidget(self._render_btn)
        btn_row.addWidget(close_btn)

        layout = QVBoxLayout(self)
        layout.addLayout(file_row)
        layout.addWidget(self._tree, 1)
        layout.addWidget(clim_header)
        layout.addWidget(self._clim)
        layout.addLayout(out_row)
        layout.addLayout(btn_row)

        self._catalog: list = []
        self._reload()

    def _default_out_for(self, h5_path: str) -> str:
        if not h5_path:
            return ""
        return os.path.join(os.path.dirname(os.path.abspath(h5_path)), "replots", self._ts)

    # -- population -----------------------------------------------------------
    def _reload(self) -> None:
        self._h5_path = self._file_edit.text().strip()
        if not self._out_pinned:
            self._out_edit.setText(self._default_out_for(self._h5_path))
        self._tree.clear()
        self._catalog = []
        if not self._h5_path or not os.path.exists(self._h5_path):
            self._clim.set_groups([])
            self._status.setText("no such file")
            self._update_render_enabled()
            return
        if not self._jobs:
            self._clim.set_groups([])
            self._status.setText("no jobs — fill Jobs (JSON) on the form (e.g. via Pick line…)")
            self._update_render_enabled()
            return
        try:
            self._catalog = _pr.replot_catalog(self._h5_path, self._jobs)
        except Exception as exc:  # noqa: BLE001 — GUI reload: show status, never crash
            self._clim.set_groups([])
            self._status.setText(f"cannot read: {exc}")
            self._update_render_enabled()
            return
        vids = list(dict.fromkeys(v for e in self._catalog for v in e.fields))
        self._clim.set_groups([(vid, volume_label(vid)) for vid in vids])
        self._tree.blockSignals(True)
        for e in self._catalog:
            top = QTreeWidgetItem([e.label + ("   · pinned" if e.note else "")])
            top.setData(0, Qt.ItemDataRole.UserRole, e.job_index)
            top.setFlags(
                top.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsAutoTristate
            )
            job_fields = self._jobs[e.job_index].get("fields") or None
            for vid in e.fields:
                child = QTreeWidgetItem([vid])
                child.setFlags(child.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                checked = job_fields is None or vid in job_fields
                child.setCheckState(
                    0, Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
                )
                top.addChild(child)
            self._tree.addTopLevelItem(top)
            top.setExpanded(True)
        self._tree.blockSignals(False)
        n_missing = len(self._jobs) - len(self._catalog)
        msg = f"{len(self._catalog)} job(s)"
        if n_missing:
            msg += f"; {n_missing} job(s) reference a slice not in this file"
        self._status.setText(msg)
        self._update_render_enabled()

    def select_all(self) -> None:
        for i in range(self._tree.topLevelItemCount()):
            top = self._tree.topLevelItem(i)
            for j in range(top.childCount()):
                top.child(j).setCheckState(0, Qt.CheckState.Checked)

    def _on_item_changed(self, _item, _col) -> None:
        self._update_render_enabled()

    def _update_render_enabled(self) -> None:
        self._render_btn.setEnabled(bool(self._checked_jobs()))

    # -- selection → core -----------------------------------------------------
    def _checked_jobs(self) -> list[dict]:
        """Jobs to render: checked fields become the job's 'fields' override."""
        out = []
        for i in range(self._tree.topLevelItemCount()):
            top = self._tree.topLevelItem(i)
            vids = [
                top.child(j).text(0)
                for j in range(top.childCount())
                if top.child(j).checkState(0) == Qt.CheckState.Checked
            ]
            if not vids:
                continue
            ji = top.data(0, Qt.ItemDataRole.UserRole)
            out.append({**self._jobs[ji], "fields": vids})
        return out

    def render_selection(self, out_dir):
        """Render the checked jobs into *out_dir*; returns written paths."""
        res = _pr.render_replot(
            self._h5_path,
            self._checked_jobs(),
            self._style,
            self._clim.clim_by_group(),
            out_dir,
            dpi=int(self._dpi.value()),
        )
        self.written = [
            p
            for jr in res.jobs
            for p in ([jr.figure] if jr.figure else []) + list(jr.overviews) + list(jr.traces)
        ]
        self._last_result = res
        return self.written

    # -- slots ----------------------------------------------------------------
    def _on_render(self) -> None:
        out_dir = self._out_edit.text().strip()
        if not out_dir:
            self._status.setText("set an output dir")
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
        res = self._last_result
        msg = f"wrote {len(written)} PNG(s) → {out_dir}"
        if res.skipped:
            msg += f"; skipped: {'; '.join(res.skipped)}"
        if res.notes:
            msg += f"; notes: {'; '.join(res.notes)}"
        self._status.setText(msg)

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
