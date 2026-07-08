"""Generic stage panel: parameter form + run/cancel + log + results.

One :class:`StageView` drives any stage: it renders the stage's schema with
:class:`~gui.widgets.param_form.ParamForm`, launches the run in a child
process via :class:`~dfxm.runner.StageRunner`, and polls it from a
:class:`QTimer` so the UI stays responsive and cancellable.
"""

from __future__ import annotations

import html
import os
from typing import NamedTuple

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from dfxm.common.figures import FigureSpec, figures_for
from dfxm.config.models import Experiment, StageSpec
from dfxm.runner import Done, Failed, Log, Progress, StageRunner
from dfxm.stages.registry import STAGE_TARGETS

from .bindings import experiment_overrides
from .viewers import inject_line_into_jobs, volume_sources
from .widgets.help_panel import HelpPanel
from .widgets.log_console import LogConsole
from .widgets.param_form import ParamForm
from .widgets.volume3d import Volume3DPanel
from .window_state import DEFAULT_STAGE_SIZES

_POLL_MS = 50

# Stages whose run yields an aligned 3-D volume worth viewing interactively.
_VOLUME_STAGES = ("visualize", "rocking")


class ExportResult(NamedTuple):
    """Outcome of one figure in an :meth:`StageView.export_all` batch."""

    figure_id: str
    ok: bool
    error: str | None


class StageView(QWidget):
    """Form + controls + log/results for a single stage."""

    runFinished = Signal(str, bool)  # (stage_name, ok)
    runStarted = Signal(str)  # stage_name

    def __init__(
        self,
        stage_name: str,
        spec: StageSpec,
        experiment: Experiment,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._stage_name = stage_name
        self._spec = spec
        self._experiment = experiment
        self._runner: StageRunner | None = None
        self._last_params: dict = {}
        self._last_result = None

        self._timer = QTimer(self)
        self._timer.setInterval(_POLL_MS)
        self._timer.timeout.connect(self._poll)

        # --- left: parameter form + run/cancel ---
        self._form = ParamForm(spec.params, self._initial_values())
        self._run_btn = QPushButton("Run")
        self._run_btn.setProperty("role", "primary")
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._on_run)
        self._cancel_btn.clicked.connect(self._on_cancel)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self._run_btn)
        btn_row.addWidget(self._cancel_btn)
        # profiles: an interactive line picker (built lazily on click)
        self._pick_btn: QPushButton | None = None
        if stage_name == "profiles":
            self._pick_btn = QPushButton("Pick line…")
            self._pick_btn.clicked.connect(self._on_pick_line)
            btn_row.addWidget(self._pick_btn)
        # slices/strain/mosaicity/rocking: re-render layers from an existing h5
        self._replot_btn: QPushButton | None = None
        if stage_name in ("slices", "strain", "mosaicity", "rocking"):
            self._replot_btn = QPushButton("Replot…")
            self._replot_btn.clicked.connect(self._on_replot)
            btn_row.addWidget(self._replot_btn)
        btn_row.addStretch(1)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress_text = QLabel("")
        self._progress_text.setWordWrap(True)
        progress_row = QHBoxLayout()
        progress_row.addWidget(self._progress, 1)
        progress_row.addWidget(self._progress_text, 2)

        self._help = HelpPanel()
        self._help.set_idle(spec.label, spec.description)
        self._form.focusedParamChanged.connect(self._help.show_param)
        self._form.focusCleared.connect(self._help.show_idle)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self._form)
        left_layout.addLayout(btn_row)
        left_layout.addLayout(progress_row)
        left_layout.addWidget(self._help)
        left_layout.addStretch(1)

        # --- right: log + results tabs ---
        self._log = LogConsole()
        self._results = QPlainTextEdit()
        self._results.setReadOnly(True)
        self._image = QLabel("(no preview)")
        self._image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_scroll = QScrollArea()
        self._image_scroll.setWidgetResizable(True)
        self._image_scroll.setWidget(self._image)

        # Export buttons — disabled until a successful run populates _last_result.
        self._export_btn = QPushButton("Export…")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._on_export_clicked)
        self._export_all_btn = QPushButton("Export all…")
        self._export_all_btn.setEnabled(False)
        self._export_all_btn.clicked.connect(self._on_export_all_clicked)
        export_row = QHBoxLayout()
        export_row.addStretch(1)
        export_row.addWidget(self._export_all_btn)
        export_row.addWidget(self._export_btn)

        self._output_tab = QWidget()
        output_layout = QVBoxLayout(self._output_tab)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.addWidget(self._image_scroll, 1)
        output_layout.addLayout(export_row)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._log, "Log")
        self._tabs.addTab(self._results, "Results")
        self._tabs.addTab(self._output_tab, "Output")
        # interactive 3-D tab for volume-producing stages (PvCanvas stays lazy)
        self._vol3d: Volume3DPanel | None = None
        if stage_name in _VOLUME_STAGES:
            self._vol3d = Volume3DPanel()
            self._tabs.addTab(self._vol3d, "3D")

        self._banner = QLabel("")
        self._banner.setWordWrap(True)
        self._banner.setTextFormat(Qt.TextFormat.RichText)
        self._banner.setVisible(False)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self._banner)
        right_layout.addWidget(self._tabs, 1)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left)

        splitter = QSplitter()
        splitter.addWidget(left_scroll)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes(list(DEFAULT_STAGE_SIZES))
        self.inner_splitter = splitter

        outer = QVBoxLayout(self)
        outer.addWidget(splitter)

    def showEvent(self, event) -> None:  # Qt hook
        super().showEvent(event)
        self._help.show_idle()  # every stage opens on its description

    # -- experiment wiring ------------------------------------------------
    def _initial_values(self) -> dict:
        values = self._spec.defaults()
        values.update(experiment_overrides(self._stage_name, self._experiment))
        return values

    def set_experiment(self, experiment: Experiment) -> None:
        """Re-apply experiment-derived defaults when the preset changes."""
        self._experiment = experiment
        self._form.set_values(experiment_overrides(self._stage_name, experiment))

    # -- banner / validation ------------------------------------------------
    def _show_banner(self, html_text: str, *, error: bool) -> None:
        self._banner.setProperty("role", "banner-error" if error else "banner-success")
        self._banner.style().unpolish(self._banner)
        self._banner.style().polish(self._banner)
        self._banner.setText(html_text)
        self._banner.setVisible(True)

    def _hide_banner(self) -> None:
        self._banner.setVisible(False)

    def _validate_inputs(self, params: dict) -> tuple[str, str] | None:
        """First (param_name, message) whose must_exist path is set but absent.

        Mode-gated folders are checked only in their active mode: ``single`` mode
        uses ``input_folder`` and ``batch`` mode uses ``root_folder``, so a stale
        pre-filled value for the inactive mode never blocks a run.
        """
        mode = params.get("mode")
        skip: set[str] = set()
        if mode == "single":
            skip = {"root_folder"}
        elif mode == "batch":
            skip = {"input_folder"}
        for p in self._spec.params:
            if not p.must_exist or p.name in skip:
                continue
            value = params.get(p.name)
            if value and not os.path.exists(str(value)):
                return p.name, f"{p.label}: path does not exist: {value}"
        return None

    # -- run lifecycle ----------------------------------------------------
    def _on_run(self) -> None:
        if self._runner is not None and self._runner.is_alive():
            return
        params = self._form.values()
        problem = self._validate_inputs(params)
        if problem is not None:
            name, message = problem
            self._show_banner(f"✗ {html.escape(message)}", error=True)
            self._form.focus_param(name)
            return
        self._hide_banner()
        self._last_params = dict(params)
        run_params = dict(params)
        window = self.window()
        if hasattr(window, "global_plot_style"):
            from dataclasses import asdict

            # Snapshot the CURRENT session publication style so every new run
            # renders with whatever the style dialog says right now.
            run_params["plot_style"] = asdict(window.global_plot_style())
        target = STAGE_TARGETS[self._stage_name]
        self._log.clear()
        self._results.clear()
        self._progress.setValue(0)
        self._progress_text.setText("")
        self._log.append(f"Running stage '{self._stage_name}'…")
        self._export_btn.setEnabled(False)
        self._export_all_btn.setEnabled(False)
        self._set_running(True)
        self.runStarted.emit(self._stage_name)
        self._runner = StageRunner(target, run_params, start_method="spawn")
        self._runner.start()
        self._timer.start()

    def _on_cancel(self) -> None:
        if self._runner is not None:
            self._runner.cancel()
        self._timer.stop()
        self._log.set_status("Cancelled.", error=True)
        self._set_running(False)

    # -- profiles interactive line picker (lazy) --------------------------
    def _on_pick_line(self) -> None:
        import json

        vals = self._form.values()
        h5 = vals.get("consolidated_h5", "")
        if not h5 or not os.path.exists(h5):
            self._log.append("Pick line: set a valid 'consolidated_h5' (run slices first).")
            self._tabs.setCurrentWidget(self._log)
            return
        try:
            jobs = json.loads(vals.get("jobs_json", "") or "[]")
        except json.JSONDecodeError:
            jobs = []
        first = jobs[0] if jobs and isinstance(jobs[0], dict) else {}
        slice_name = first.get("name", "oblique_full")
        offset = float(first.get("offset_um", 0.0))

        from .widgets.line_picker import LinePickerDialog  # imported on demand

        try:
            dlg = LinePickerDialog(
                h5,
                slice_name,
                init_offset=offset,
                ref_pref=vals.get("reference_volume_id", ""),
                parent=self,
            )
        except Exception as exc:  # noqa: BLE001 - missing slice / unreadable file
            self._log.append(f"Pick line failed: {exc}")
            self._tabs.setCurrentWidget(self._log)
            return
        if dlg.exec() and dlg.result:
            start, end, off, fields = dlg.result
            new_jobs = inject_line_into_jobs(
                vals.get("jobs_json", "") or "[]", slice_name, start, end, off, fields=fields
            )
            self._form.set_values({"jobs_json": new_jobs})
            self._log.append(
                f"Picked line on '{slice_name}' @ {off:+.3f} µm -> jobs_json updated; Run to profile."
            )
            self._tabs.setCurrentWidget(self._log)

    # -- interactive replot (lazy) ----------------------------------------
    def _on_replot(self) -> None:
        from dataclasses import replace

        vals = self._form.values()
        window = self.window()
        style = window.global_plot_style() if hasattr(window, "global_plot_style") else None
        style = replace(style) if style is not None else None

        if self._stage_name == "slices":
            self._replot_slices(vals, style)
            return

        from dfxm.stages import mosaicity as _mo
        from dfxm.stages import rocking as _ro
        from dfxm.stages import strain as _st

        module = {"strain": _st, "mosaicity": _mo, "rocking": _ro}[self._stage_name]
        # Best-effort default h5 = the exact path the last run wrote (if any); the
        # dialog's file field lets the user Browse/Load a different one (cold start).
        res = self._last_result
        h5_default = ""
        for attr in ("stacked_path", "aligned_path"):
            p = getattr(res, attr, "") if res is not None else ""
            if p:
                h5_default = p
                break

        from .widgets.replot_dialog import ReplotDialog  # imported on demand

        def render_fn(h5, selections, st, clim, roi, out, _m=module, _p=dict(vals)):
            return _m.render_replot(h5, selections, st, clim, out, roi=roi, params=_p)

        # out_default="" lets the dialog default the output beside the loaded h5.
        dlg = ReplotDialog(
            h5_default,
            module.replot_catalog,
            render_fn,
            style=style,
            out_default="",
            parent=self,
        )
        dlg.exec()
        if dlg.written:
            self._log.append(
                f"Replotted {len(dlg.written)} PNG(s) → {os.path.dirname(dlg.written[0])}"
            )
            self._tabs.setCurrentWidget(self._log)

    def _replot_slices(self, vals: dict, style) -> None:
        """Open the slices-specific replot dialog (SliceReplotDialog)."""
        out_dir = vals.get("output_dir", "") or os.path.join(
            os.path.dirname(
                vals.get("mosa_volume_file", "") or vals.get("strain_volume_file", "") or "."
            ),
            "oblique_slices",
        )
        h5 = os.path.join(out_dir, vals.get("output_h5_name", "") or "oblique_slices.h5")

        from .widgets.slice_replot import SliceReplotDialog  # imported on demand

        # out_default="" lets the dialog default the output beside the loaded h5.
        dlg = SliceReplotDialog(
            h5,
            style=style,
            out_default="",
            parent=self,
        )
        dlg.exec()
        if dlg.written:
            self._log.append(
                f"Replotted {len(dlg.written)} PNG(s) → {os.path.dirname(dlg.written[0])}"
            )
            self._tabs.setCurrentWidget(self._log)

    def _poll(self) -> None:
        runner = self._runner
        if runner is None:
            self._timer.stop()
            return
        for msg in runner.poll():
            self._handle(msg)
        # Once concluded, _finish_* stops the timer; an inactive timer means we
        # are done. If the worker died but the timer is still active, drain once
        # more and, failing that, report an abnormal exit.
        if self._timer.isActive() and not runner.is_alive():
            for msg in runner.poll():
                self._handle(msg)
            if self._timer.isActive():
                self._timer.stop()
                self._log.set_status("Worker exited without a result.", error=True)
                self._set_running(False)

    def _handle(self, msg) -> None:
        if isinstance(msg, Progress):
            self._log.set_progress(msg.frac, msg.text)
            self._progress.setValue(max(0, min(100, int(round(msg.frac * 100)))))
            if msg.text:
                self._progress_text.setText(msg.text)
                self._log.append(f"  [{msg.frac * 100:5.1f}%] {msg.text}")
        elif isinstance(msg, Log):
            self._log.append(msg.text)
        elif isinstance(msg, Done):
            self._finish_ok(msg.result)
        elif isinstance(msg, Failed):
            self._finish_failed(msg)

    def _finish_ok(self, result) -> None:
        self._timer.stop()
        self._last_result = result
        self._log.set_progress(1.0, "Done.")
        self._progress.setValue(100)
        summary = _summarize(self._stage_name, result)
        first_line = summary.splitlines()[0] if summary else "done"
        self._show_banner(f"✓ {html.escape(first_line)}", error=False)
        self._results.setPlainText(summary)
        img = _representative_image(self._stage_name, result)
        shown = False
        if img and os.path.exists(img):
            pix = QPixmap(img)
            if not pix.isNull():
                self._image.setPixmap(
                    pix.scaledToWidth(700, Qt.TransformationMode.SmoothTransformation)
                )
                self._tabs.setCurrentWidget(self._output_tab)
                shown = True
        if not shown:
            self._tabs.setCurrentWidget(self._results)
        if self._vol3d is not None:
            # lazy: install source callables only; nothing loads/renders until
            # the user picks a volume and clicks Render 3-D.
            self._vol3d.set_sources(volume_sources(self._stage_name, result, self._last_params))
        self._export_btn.setEnabled(True)
        self._export_all_btn.setEnabled(True)
        self._set_running(False)
        self.runFinished.emit(self._stage_name, True)

    # -- export ---------------------------------------------------------------

    def _figures(self) -> list[FigureSpec]:
        """Return the list of FigureSpecs for the last successful run (or [] if none)."""
        if self._last_result is None:
            return []
        return figures_for(self._stage_name, self._last_result, self._last_params)

    def export_all(self, out_dir: str) -> list[ExportResult]:
        """Build and save every figure in this stage's catalog to *out_dir*.

        Returns a summary list of :class:`ExportResult` — one entry per spec.
        A per-figure build failure is recorded (``ok=False, error=str(exc)``)
        and the batch continues; one bad figure never aborts the rest.
        Per-format savefig failures are handled inside :func:`save_spec` (the
        format is skipped; the summary reflects whether any formats were written).
        """
        from .widgets.export_dialog import save_spec

        specs = self._figures()
        style = self.window().global_plot_style()
        os.makedirs(out_dir, exist_ok=True)
        summary: list[ExportResult] = []
        for spec in specs:
            try:
                written = save_spec(spec, out_dir, style)
                if not written:
                    summary.append(ExportResult(spec.figure_id, False, "no formats written"))
                elif len(written) < len(style.formats):
                    summary.append(
                        ExportResult(
                            spec.figure_id,
                            True,
                            f"wrote {len(written)}/{len(style.formats)} formats",
                        )
                    )
                else:
                    summary.append(ExportResult(spec.figure_id, True, None))
            except Exception as exc:  # noqa: BLE001 — record + continue, never abort the batch
                summary.append(ExportResult(spec.figure_id, False, str(exc)))
        return summary

    def _on_export_all_clicked(self) -> None:
        try:
            specs = self._figures()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Export all", f"Could not list figures:\n{exc}")
            return
        if not specs:
            QMessageBox.information(
                self, "Export all", "This stage produced no exportable figures."
            )
            return
        folder = QFileDialog.getExistingDirectory(self, "Export all figures to folder")
        if not folder:
            return
        try:
            summary = self.export_all(folder)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Export all", f"Export failed:\n{exc}")
            return
        n_ok = sum(1 for r in summary if r.ok)
        n_fail = len(summary) - n_ok
        failures = "\n".join(f"  {r.figure_id}: {r.error}" for r in summary if not r.ok)
        note = f"\n\nFailed ({n_fail}):\n{failures}" if n_fail else ""
        glyph = "✓" if n_fail == 0 else "⚠"
        self._show_banner(
            f"{glyph} Exported {n_ok}/{len(summary)} figures to {html.escape(folder)}"
            + (f" — {n_fail} failed" if n_fail else ""),
            error=n_fail > 0,
        )
        if n_fail:
            QMessageBox.warning(
                self,
                "Export all — partial failure",
                f"Exported {n_ok} of {len(summary)} figures.{note}",
            )

    def _on_export_clicked(self) -> None:
        try:
            specs = self._figures()
        except Exception as exc:  # noqa: BLE001 — output file may have moved/changed since the run
            QMessageBox.warning(self, "Export", f"Could not list figures:\n{exc}")
            return
        if not specs:
            QMessageBox.information(self, "Export", "This stage produced no exportable figures.")
            return
        from .widgets.export_dialog import ExportDialog

        # Use the session global style held on MainWindow so edits via
        # "Publication style…" carry through to every future export dialog.
        session_style = self.window().global_plot_style()
        dlg = ExportDialog(specs, 0, session_style, parent=self)
        # Delete on close so its MplCanvas's themeChanged subscription doesn't
        # accumulate across repeated Export… opens during a session.
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dlg.exec()

    def _finish_failed(self, failure: Failed) -> None:
        self._timer.stop()
        self._log.set_status(f"Failed: {failure.error}", error=True)
        self._log.append(failure.traceback)
        text = f"✗ {html.escape(failure.error)}"
        hint = getattr(failure, "hint", "")
        if hint:
            text += f"<br><i>{html.escape(hint)}</i>"
        self._show_banner(text, error=True)
        self._set_running(False)
        self.runFinished.emit(self._stage_name, False)

    def _set_running(self, running: bool) -> None:
        self._run_btn.setEnabled(not running)
        self._cancel_btn.setEnabled(running)
        self._form.setEnabled(not running)


def _summarize(stage_name: str, result) -> str:
    """Human-readable summary of a stage result, keyed by stage name."""
    formatter = _SUMMARIZERS.get(stage_name)
    return formatter(result) if formatter is not None else repr(result)


def _summarize_concat(result) -> str:
    lines = [f"{result.n_ok} ok, {result.n_skipped} skipped, {result.n_failed} failed", ""]
    for fr in result.files:
        status = "OK" if fr.ok else ("SKIP" if fr.skipped else "FAIL")
        detail = (
            f"{fr.n_entries} scans, {fr.total_frames} frames, "
            f"{fr.n_motors} motors ({fr.n_varying} varying), "
            f"{'copy' if fr.copied else 'vds'}"
            if fr.ok
            else (fr.error or "")
        )
        lines.append(f"[{status}] {fr.output_path}")
        if detail:
            lines.append(f"        {detail}")
    return "\n".join(lines)


def _summarize_strain(result) -> str:
    if result.layers:
        lines = [
            f"layers: {result.n_layers}   volume: {result.volume_shape}",
            f"stacked: {result.stacked_path}",
        ]
        for layer in result.layers:
            lines.append(
                f"  {layer.name}: shape={layer.shape} mean={layer.mean:.3e} std={layer.std:.3e}"
            )
    else:
        lines = ["no strain layers produced"]
    lines += [f"skipped: {s}" for s in result.skipped]
    return "\n".join(lines)


def _summarize_mosaicity(result) -> str:
    if result.layers:
        lines = [f"layers: {result.n_layers}", f"stacked: {result.stacked_path}"]
        for key, shape in result.datasets.items():
            lines.append(f"  {key}: {shape}")
    else:
        lines = ["no mosaicity layers produced"]
    lines += [f"skipped: {s}" for s in result.skipped]
    return "\n".join(lines)


def _dataset_lines(datasets, with_shape: bool) -> list[str]:
    lines = []
    for d in datasets:
        made = [
            n for n, v in (("layers", d.layers_dir), ("anim", d.animation), ("3d", d.top_view)) if v
        ]
        shape = f"shape={d.shape} " if with_shape else ""
        lines.append(f"  {d.name}: {shape}[{', '.join(made)}]")
        for note in d.notes:
            lines.append(f"      {note}")
    return lines


def _summarize_rocking(result) -> str:
    if result.volume_shape is None:
        lines = ["no rocking volumes produced"]
    else:
        lines = [f"output: {result.output_dir}"]
        if result.aligned_path:
            lines.append(f"aligned: {result.aligned_path}")
        lines += [
            f"layers used: {result.n_layers_used}   volume: {result.volume_shape}",
            f"specific frame: {result.specific_frame_idx}   z-span: {result.z_span_um:.2f} µm",
        ]
        lines += _dataset_lines(result.datasets, with_shape=False)
    lines += [f"skipped: {s}" for s in result.skipped]
    return "\n".join(lines)


def _summarize_visualize(result) -> str:
    lines = [f"output: {result.output_dir}", f"datasets: {len(result.datasets)}"]
    lines += _dataset_lines(result.datasets, with_shape=True)
    lines += [f"skipped: {s}" for s in result.skipped]
    return "\n".join(lines)


def _summarize_paraview(result) -> str:
    lines = [f"output: {result.output_dir}", f"exports: {len(result.exports)}"]
    for e in result.exports:
        lines.append(f"  {e.name}: {e.pvti_path}")
        lines.append(
            f"      dims={e.dimensions_xyz} spacing={e.spacing_um_xyz} "
            f"pieces={e.n_pieces} fields={e.fields}"
        )
    lines += [f"skipped: {s}" for s in result.skipped]
    if result.info_path:
        lines.append(f"info: {result.info_path}")
    return "\n".join(lines)


def _summarize_slices(result) -> str:
    if result.output_h5 is None:
        lines = ["no volumes sliced"]
    else:
        lines = [
            f"output: {result.output_h5}",
            f"volumes: {len(result.volume_ids)}   slices: {len(result.slice_names)}   "
            f"planes: {result.n_planes_total}   pngs: {len(result.pngs)}",
        ]
        lines += [f"  {vid}" for vid in result.volume_ids]
    lines += [f"  {n}" for n in getattr(result, "notes", [])]
    lines += [f"skipped: {s}" for s in result.skipped]
    return "\n".join(lines)


def _summarize_profiles(result) -> str:
    lines = [
        f"mode: {result.mode}",
        f"output: {result.output_dir}",
        f"jobs: {len(result.jobs)}",
    ]
    for j in result.jobs:
        bits = []
        if j.csvs:
            bits.append(f"csv={len(j.csvs)}")
        if j.overviews:
            bits.append(f"overviews={len(j.overviews)}")
        if j.traces:
            bits.append(f"traces={len(j.traces)}")
        extra = (" " + " ".join(bits)) if bits else ""
        fig = j.figure or "(no companion)"
        lines.append(f"  {j.name} @ {j.offset_used_um:+.2f} µm -> {fig}{extra}")
    lines += [f"skipped: {s}" for s in result.skipped]
    return "\n".join(lines)


def _summarize_matched(result) -> str:
    if result.layers_dir is None:
        lines = [
            "no matched layers saved",
            f"matched {result.n_matched}/{result.n_strain}",
        ]
    else:
        clim = f"clim=({result.vmin:.4g}, {result.vmax:.4g})"
        if getattr(result, "vmin_raw", None) is not None:
            clim += f" (rounded from ({result.vmin_raw:.4g}, {result.vmax_raw:.4g}))"
        lines = [
            f"output: {result.layers_dir}",
            f"matched {result.n_matched}/{result.n_strain}, saved {result.n_saved} "
            f"(frame {result.frame_index})",
            f"max match dist: {result.max_match_dist_um:.3f} µm   {clim}",
        ]
    lines += [f"skipped: {s}" for s in result.skipped]
    return "\n".join(lines)


_SUMMARIZERS = {
    "concat": _summarize_concat,
    "strain": _summarize_strain,
    "mosaicity": _summarize_mosaicity,
    "rocking": _summarize_rocking,
    "visualize": _summarize_visualize,
    "paraview": _summarize_paraview,
    "slices": _summarize_slices,
    "profiles": _summarize_profiles,
    "matched": _summarize_matched,
}


def _representative_image(stage_name: str, result) -> str | None:
    """Pick a representative output image to preview, if the stage made one."""
    picker = _IMAGE_PICKERS.get(stage_name)
    return picker(result) if picker is not None else None


def _image_strain(result) -> str | None:
    for layer in result.layers:
        for png in layer.plots:
            if png.endswith("_strain.png"):
                return png
        if layer.plots:
            return layer.plots[0]
    return None


def _image_from_datasets(result) -> str | None:
    for d in result.datasets:
        if d.top_view:
            return d.top_view
    for d in result.datasets:
        if d.layers_dir and os.path.isdir(d.layers_dir):
            pngs = sorted(p for p in os.listdir(d.layers_dir) if p.endswith(".png"))
            if pngs:
                return os.path.join(d.layers_dir, pngs[0])
    return None


def _image_first_png(result) -> str | None:
    return result.pngs[0] if result.pngs else None


def _image_profiles(result) -> str | None:
    for j in result.jobs:
        if j.figure:
            return j.figure
    for j in result.jobs:
        if j.traces:
            return j.traces[0]
    return None


_IMAGE_PICKERS = {
    "strain": _image_strain,
    "visualize": _image_from_datasets,
    "rocking": _image_from_datasets,
    "slices": _image_first_png,
    "matched": _image_first_png,
    "profiles": _image_profiles,
}
