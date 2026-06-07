"""Generic stage panel: parameter form + run/cancel + log + results.

One :class:`StageView` drives any stage: it renders the stage's schema with
:class:`~gui.widgets.param_form.ParamForm`, launches the run in a child
process via :class:`~dfxm.runner.StageRunner`, and polls it from a
:class:`QTimer` so the UI stays responsive and cancellable.
"""

from __future__ import annotations

import os

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from dfxm.config.models import Experiment, StageSpec
from dfxm.runner import Done, Failed, Log, Progress, StageRunner
from dfxm.stages.registry import STAGE_TARGETS

from .bindings import experiment_overrides
from .widgets.log_console import LogConsole
from .widgets.param_form import ParamForm

_POLL_MS = 50


class StageView(QWidget):
    """Form + controls + log/results for a single stage."""

    runFinished = Signal(str, bool)  # (stage_name, ok)

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

        self._timer = QTimer(self)
        self._timer.setInterval(_POLL_MS)
        self._timer.timeout.connect(self._poll)

        # --- left: parameter form + run/cancel ---
        self._form = ParamForm(spec.params, self._initial_values())
        self._run_btn = QPushButton("Run")
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._on_run)
        self._cancel_btn.clicked.connect(self._on_cancel)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self._run_btn)
        btn_row.addWidget(self._cancel_btn)
        btn_row.addStretch(1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self._form)
        left_layout.addLayout(btn_row)
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
        self._tabs = QTabWidget()
        self._tabs.addTab(self._log, "Log")
        self._tabs.addTab(self._results, "Results")
        self._tabs.addTab(self._image_scroll, "Output")

        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(self._tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 600])

        outer = QVBoxLayout(self)
        outer.addWidget(splitter)

    # -- experiment wiring ------------------------------------------------
    def _initial_values(self) -> dict:
        values = self._spec.defaults()
        values.update(experiment_overrides(self._stage_name, self._experiment))
        return values

    def set_experiment(self, experiment: Experiment) -> None:
        """Re-apply experiment-derived defaults when the preset changes."""
        self._experiment = experiment
        self._form.set_values(experiment_overrides(self._stage_name, experiment))

    # -- run lifecycle ----------------------------------------------------
    def _on_run(self) -> None:
        if self._runner is not None and self._runner.is_alive():
            return
        params = self._form.values()
        target = STAGE_TARGETS[self._stage_name]
        self._log.clear()
        self._results.clear()
        self._log.append(f"Running stage '{self._stage_name}'…")
        self._set_running(True)
        self._runner = StageRunner(target, params, start_method="spawn")
        self._runner.start()
        self._timer.start()

    def _on_cancel(self) -> None:
        if self._runner is not None:
            self._runner.cancel()
        self._timer.stop()
        self._log.set_status("Cancelled.", error=True)
        self._set_running(False)

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
            if msg.text:
                self._log.append(f"  [{msg.frac * 100:5.1f}%] {msg.text}")
        elif isinstance(msg, Log):
            self._log.append(msg.text)
        elif isinstance(msg, Done):
            self._finish_ok(msg.result)
        elif isinstance(msg, Failed):
            self._finish_failed(msg)

    def _finish_ok(self, result) -> None:
        self._timer.stop()
        self._log.set_progress(1.0, "Done.")
        self._results.setPlainText(_summarize(result))
        img = _representative_image(result)
        shown = False
        if img and os.path.exists(img):
            pix = QPixmap(img)
            if not pix.isNull():
                self._image.setPixmap(
                    pix.scaledToWidth(700, Qt.TransformationMode.SmoothTransformation)
                )
                self._tabs.setCurrentWidget(self._image_scroll)
                shown = True
        if not shown:
            self._tabs.setCurrentWidget(self._results)
        self._set_running(False)
        self.runFinished.emit(self._stage_name, True)

    def _finish_failed(self, failure: Failed) -> None:
        self._timer.stop()
        self._log.set_status(f"Failed: {failure.error}", error=True)
        self._log.append(failure.traceback)
        self._set_running(False)
        self.runFinished.emit(self._stage_name, False)

    def _set_running(self, running: bool) -> None:
        self._run_btn.setEnabled(not running)
        self._cancel_btn.setEnabled(running)
        self._form.setEnabled(not running)


def _summarize(result) -> str:
    """Human-readable summary of a stage result (duck-typed per stage)."""
    files = getattr(result, "files", None)
    if files is not None:  # ConcatResult
        lines = [f"{result.n_ok} ok, {result.n_skipped} skipped, {result.n_failed} failed", ""]
        for fr in files:
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

    if hasattr(result, "method") and hasattr(result, "volume_shape"):  # StrainResult
        lines = [
            f"method: {result.method}",
            f"layers: {result.n_layers}   volume: {result.volume_shape}",
            f"stacked: {result.stacked_path}",
        ]
        for layer in result.layers:
            lines.append(
                f"  {layer.name}: shape={layer.shape} mean={layer.mean:.3e} std={layer.std:.3e}"
            )
        if result.skipped:
            lines.append(f"skipped: {len(result.skipped)}")
        return "\n".join(lines)

    if hasattr(result, "aligned_path"):  # RockingResult
        lines = [
            f"output: {result.output_dir}",
            f"aligned: {result.aligned_path}",
            f"layers used: {result.n_layers_used}   volume: {result.volume_shape}",
            f"specific frame: {result.specific_frame_idx}   z-span: {result.z_span_um:.2f} µm",
        ]
        for d in result.datasets:
            made = [
                n
                for n, v in (("layers", d.layers_dir), ("anim", d.animation), ("3d", d.top_view))
                if v
            ]
            lines.append(f"  {d.name}: [{', '.join(made)}]")
            for note in d.notes:
                lines.append(f"      {note}")
        lines += [f"skipped: {s}" for s in result.skipped]
        return "\n".join(lines)

    exports = getattr(result, "exports", None)
    if exports is not None:  # ParaviewResult
        lines = [f"output: {result.output_dir}", f"exports: {len(exports)}"]
        for e in exports:
            lines.append(f"  {e.name}: {e.pvti_path}")
            lines.append(
                f"      dims={e.dimensions_xyz} spacing={e.spacing_um_xyz} "
                f"pieces={e.n_pieces} fields={e.fields}"
            )
        lines += [f"skipped: {s}" for s in result.skipped]
        if result.info_path:
            lines.append(f"info: {result.info_path}")
        return "\n".join(lines)

    datasets = getattr(result, "datasets", None)
    if hasattr(result, "output_dir") and isinstance(datasets, list):  # VisualizeResult
        lines = [f"output: {result.output_dir}", f"datasets: {len(datasets)}"]
        for d in datasets:
            made = [
                n
                for n, v in (("layers", d.layers_dir), ("anim", d.animation), ("3d", d.top_view))
                if v
            ]
            lines.append(f"  {d.name}: shape={d.shape} [{', '.join(made)}]")
            for note in d.notes:
                lines.append(f"      {note}")
        lines += [f"skipped: {s}" for s in result.skipped]
        return "\n".join(lines)

    if hasattr(result, "stacked_path") and isinstance(datasets, dict):  # MosaicityResult
        lines = [f"layers: {result.n_layers}", f"stacked: {result.stacked_path}"]
        for key, shape in datasets.items():
            lines.append(f"  {key}: {shape}")
        if result.skipped:
            lines.append(f"skipped: {len(result.skipped)}")
        return "\n".join(lines)

    if hasattr(result, "n_planes_total"):  # SlicesResult
        lines = [
            f"output: {result.output_h5}",
            f"volumes: {len(result.volume_ids)}   slices: {len(result.slice_names)}   "
            f"planes: {result.n_planes_total}   pngs: {len(result.pngs)}",
        ]
        lines += [f"  {vid}" for vid in result.volume_ids]
        lines += [f"skipped: {s}" for s in result.skipped]
        return "\n".join(lines)

    if hasattr(result, "jobs") and hasattr(result, "mode"):  # ProfilesResult
        lines = [
            f"mode: {result.mode}",
            f"output: {result.output_dir}",
            f"jobs: {len(result.jobs)}",
        ]
        for j in result.jobs:
            extra = (
                f" csv={len(j.csvs)} overviews={len(j.overviews)}" if j.csvs or j.overviews else ""
            )
            lines.append(f"  {j.name} @ {j.offset_used_um:+.2f} µm -> {j.figure}{extra}")
        lines += [f"skipped: {s}" for s in result.skipped]
        return "\n".join(lines)

    if hasattr(result, "n_matched"):  # MatchedResult
        lines = [
            f"output: {result.layers_dir}",
            f"matched {result.n_matched}/{result.n_strain}, saved {result.n_saved} "
            f"(frame {result.frame_index})",
            f"max match dist: {result.max_match_dist_um:.3f} µm   clim=({result.vmin:.4g}, "
            f"{result.vmax:.4g})",
        ]
        lines += [f"skipped: {s}" for s in result.skipped[:5]]
        return "\n".join(lines)

    return repr(result)


def _representative_image(result) -> str | None:
    """Pick a representative output image to preview, if the stage made one."""
    layers = getattr(result, "layers", None)
    if layers and hasattr(layers[0], "plots"):  # StrainResult
        for layer in layers:
            for png in layer.plots:
                if png.endswith("_strain.png"):
                    return png
            if layer.plots:
                return layer.plots[0]
    datasets = getattr(result, "datasets", None)
    if isinstance(datasets, list) and datasets and hasattr(datasets[0], "name"):  # VisualizeResult
        for d in datasets:
            if getattr(d, "top_view", None):
                return d.top_view
        for d in datasets:
            ld = getattr(d, "layers_dir", None)
            if ld and os.path.isdir(ld):
                pngs = sorted(p for p in os.listdir(ld) if p.endswith(".png"))
                if pngs:
                    return os.path.join(ld, pngs[0])
    pngs = getattr(result, "pngs", None)  # SlicesResult / MatchedResult
    if isinstance(pngs, list) and pngs:
        return pngs[0]
    jobs = getattr(result, "jobs", None)  # ProfilesResult
    if isinstance(jobs, list) and jobs and getattr(jobs[0], "figure", None):
        return jobs[0].figure
    return None
