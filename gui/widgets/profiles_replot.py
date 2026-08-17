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

from .busy import DialogBatchRunner
from .clim_section import ClimGroupSection, volume_label


class ProfilesReplotDialog(QDialog):
    """Pick jobs/fields from the form's jobs list and re-render profile figures."""

    def __init__(self, h5_path, jobs, style=None, out_default="", parent=None, params=None) -> None:
        super().__init__(parent)
        self._h5_path = h5_path
        self._jobs = [j for j in (jobs or []) if isinstance(j, dict) and "name" in j]
        self._style = style
        self._params = params
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
        dpi_default = (self._params or {}).get("fig_dpi", _pr.STAGE.defaults()["fig_dpi"])
        self._dpi.setValue(int(dpi_default))
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
        self._close_btn = QPushButton("Close")
        self._close_btn.clicked.connect(self.reject)
        btn_row = QHBoxLayout()
        btn_row.addWidget(self._status, 1)
        btn_row.addWidget(self._render_btn)
        btn_row.addWidget(self._close_btn)

        layout = QVBoxLayout(self)
        layout.addLayout(file_row)
        layout.addWidget(self._tree, 1)
        layout.addWidget(clim_header)
        layout.addWidget(self._clim)
        layout.addLayout(out_row)
        layout.addLayout(btn_row)

        self._catalog: list = []
        self._batch = DialogBatchRunner(self, (self._render_btn, self._close_btn))
        self._reload()

    @staticmethod
    def _fmt_error(exc: Exception) -> str:
        """Render *exc* for a single status line, appending a StageUserError hint if present."""
        hint = getattr(exc, "hint", None)
        return f"{exc} — {hint}" if hint else str(exc)

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
            self._status.setText(f"cannot read: {self._fmt_error(exc)}")
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
        """Jobs to render: checked fields become the job's 'fields' override.

        Exception: a job whose original spec had no 'fields' key and whose
        every child is checked is passed through unchanged (no 'fields' key
        added), so render_replot's run-default ordering (``[ref] + sorted
        others``) applies instead of tree order.
        """
        out = []
        for i in range(self._tree.topLevelItemCount()):
            top = self._tree.topLevelItem(i)
            n_children = top.childCount()
            n_checked = sum(
                1 for j in range(n_children) if top.child(j).checkState(0) == Qt.CheckState.Checked
            )
            if n_checked == 0:
                continue
            ji = top.data(0, Qt.ItemDataRole.UserRole)
            job = self._jobs[ji]
            if "fields" not in job and n_checked == n_children:
                out.append(job)
                continue
            vids = [
                top.child(j).text(0)
                for j in range(n_children)
                if top.child(j).checkState(0) == Qt.CheckState.Checked
            ]
            out.append({**job, "fields": vids})
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
            params=self._params,
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
        jobs = self._checked_jobs()
        h5, style, params = self._h5_path, self._style, self._params
        clims, dpi = self._clim.clim_by_group(), int(self._dpi.value())
        self._last_out_dir = out_dir
        result_box: list = []

        def _whole_batch(_jobs):
            # ONE item: profiles' stem dedup + shared trace margins are
            # per-call state — never split this batch (plan Task 7 note).
            res = _pr.render_replot(h5, _jobs, style, clims, out_dir, dpi=dpi, params=params)
            result_box.append(res)  # plain attr/list append: GIL-safe, read only after finish
            return [
                p
                for jr in res.jobs
                for p in ([jr.figure] if jr.figure else []) + list(jr.overviews) + list(jr.traces)
            ]

        self._result_box = result_box
        self._batch.start(
            [jobs], _whole_batch, self._on_batch_done, text=f"Rendering {len(jobs)} job(s)…"
        )

    def _on_batch_done(self, written: list, error: str, cancelled: bool) -> None:
        self.written = written
        if error:
            self._status.setText(f"render failed: {error}")
            return
        res = self._result_box[0] if self._result_box else None
        self._last_result = res
        msg = f"wrote {len(written)} PNG(s) → {self._last_out_dir}"
        if cancelled:
            msg = "cancelled — " + msg
        if res is not None and res.skipped:
            msg += f"; skipped: {'; '.join(res.skipped)}"
        if res is not None and res.notes:
            msg += f"; notes: {'; '.join(res.notes)}"
        self._status.setText(msg)

    def reject(self) -> None:  # noqa: D401 — Qt override
        """Close gates on a running batch: first Esc/Close requests cancel."""
        if self._batch.running:
            self._batch.request_cancel()
            return
        super().reject()

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
