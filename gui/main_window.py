"""Main window: compact experiment header + pipeline rail + stage stack.

The left column holds the :class:`~gui.experiment_panel.ExperimentPanel`
(compact header) above a single *pipeline rail*: Overview first, then the
stages in pipeline order with a status glyph each. darfix appears as a
disabled external row after concat; concat is marked optional. The right
side is a stacked set of :class:`~gui.stage_view.StageView` panels behind
an :class:`~gui.overview_page.OverviewPage` landing page.
"""

from __future__ import annotations

import os
from dataclasses import replace

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from dfxm.common.advice import human_bytes
from dfxm.common.plotting import PUBLICATION_STYLE, PlotStyle
from dfxm.config.models import Experiment

from . import advisor
from .bindings import STAGE_ORDER, STAGE_SPECS
from .experiment_panel import ExperimentPanel
from .form_state import FormStateStore
from .overview_page import OverviewPage
from .stage_view import StageView
from .theme import ThemeController
from .window_state import WindowState

_GLYPH_IDLE = "—"
_GLYPH_RUNNING = "▶"
_GLYPH_OK = "✓"
_GLYPH_FAIL = "✗"


class MainWindow(QMainWindow):
    """Top-level window wiring experiment + rail + overview + stages."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DFXM pipeline")
        self.resize(1100, 720)
        self._window_state = WindowState()
        self._form_state = FormStateStore()

        # Session-wide publication style — restored from QSettings when a
        # previous session saved one, else seeded from the module constant
        # (held as an independent copy so mutations never touch PUBLICATION_STYLE).
        self._plot_style: PlotStyle = self._load_plot_style()

        self._experiment_panel = ExperimentPanel()
        experiment = self._experiment_panel.current_experiment()

        # Stage views + overview page (stacked).
        self._stack = QStackedWidget()
        self._overview = OverviewPage(STAGE_ORDER, STAGE_SPECS)
        self._overview.stageSelected.connect(self._show_stage)
        self._stack.addWidget(self._overview)
        self._views: dict[str, StageView] = {}
        for name in STAGE_ORDER:
            view = StageView(name, STAGE_SPECS[name], experiment, store=self._form_state)
            view.runStarted.connect(self._on_run_started)
            view.runFinished.connect(self._on_run_finished)
            self._views[name] = view
            self._stack.addWidget(view)

        # Pipeline rail: one list = navigation + status.
        self._nav = QListWidget()
        self._status_items: dict[str, QListWidgetItem] = {}
        self._item_base: dict[str, str] = {}
        self._row_target: list[str | None] = []

        overview_item = QListWidgetItem("☰  Overview")
        self._nav.addItem(overview_item)
        self._row_target.append("__overview__")
        muted = QBrush(QColor(ThemeController.instance().palette.ink_muted))
        for i, name in enumerate(STAGE_ORDER, start=1):
            label = STAGE_SPECS[name].label
            base = f"{i} {label}" + (" (optional)" if name == "concat" else "")
            item = QListWidgetItem(f"{_GLYPH_IDLE}  {base}")
            if name == "concat":
                item.setForeground(muted)
            self._nav.addItem(item)
            self._row_target.append(name)
            self._status_items[name] = item
            self._item_base[name] = base
            if name == "concat":
                darfix = QListWidgetItem("    ⤷ darfix (external)")
                darfix.setFlags(Qt.ItemFlag.NoItemFlags)
                darfix.setToolTip(
                    "Run darfix outside this app: it turns the concatenated .h5 "
                    "into the maps.h5 files used by strain and mosaicity."
                )
                self._nav.addItem(darfix)
                self._row_target.append(None)

        self._nav.currentRowChanged.connect(self._on_row_changed)
        self._nav.setCurrentRow(0)  # land on Overview

        self._experiment_panel.experimentChanged.connect(self._on_experiment_changed)

        # "Publication style…" button — lives in the left column below the rail.
        self._pub_style_btn = QPushButton("Publication style…")
        self._pub_style_btn.clicked.connect(self._on_pub_style)

        # "Figure builder…" button — non-modal multi-panel composer window,
        # one instance reused across opens (lazy-imported so importing this
        # module never pulls in the compose/matplotlib machinery).
        self._figure_builder = None
        self._figure_builder_btn = QPushButton("Figure builder…")
        self._figure_builder_btn.clicked.connect(self._on_figure_builder)

        # "System check…" — what this machine is and what it implies for
        # settings. The only surface that pays for a GL probe on demand.
        self._system_check_btn = QPushButton("System check…")
        self._system_check_btn.clicked.connect(self._on_system_check)

        # Light/dark theme toggle.
        self._theme_btn = QPushButton()
        self._theme_btn.setCheckable(True)
        self._theme_btn.setChecked(ThemeController.instance().mode == "dark")
        self._theme_btn.setToolTip("Switch between light and dark appearance")
        self._theme_btn.clicked.connect(self._on_theme_toggle)
        self._sync_theme_btn()
        ThemeController.instance().themeChanged.connect(self._on_theme_changed)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self._experiment_panel)
        left_layout.addWidget(self._nav, 1)
        left_layout.addWidget(self._pub_style_btn)
        left_layout.addWidget(self._figure_builder_btn)
        left_layout.addWidget(self._system_check_btn)
        left_layout.addWidget(self._theme_btn)

        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(self._stack)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([380, 720])
        self.setCentralWidget(splitter)

        # Ambient machine readout. Cheap fields only, on a timer — it must never
        # probe GL (that costs a child process) and never block the UI.
        self._machine_label = QLabel("")
        self.statusBar().addPermanentWidget(self._machine_label)
        self._machine_timer = QTimer(self)
        self._machine_timer.setInterval(5000)
        self._machine_timer.timeout.connect(self._refresh_machine_status)
        self._machine_timer.start()
        self._refresh_machine_status()

        self._main_splitter = splitter
        for name in STAGE_ORDER:
            self._window_state.register_stage_splitter(self._views[name].inner_splitter)

        self._window_state.restore(self, self._main_splitter)

    # -- machine status bar --------------------------------------------------

    def _refresh_machine_status(self) -> None:
        """Cores, free disk, RAM and (only if already probed) the GL stack.

        Unmeasured fields are omitted rather than shown as zero: a probe that
        failed is recorded in `probe_errors`, and "0.0 B free" would read as a
        full disk.
        """
        prof = advisor.cached_profile(os.getcwd())
        parts = [f"{prof.cpu_logical} cores"]
        if prof.disk_free:
            parts.append(f"{human_bytes(prof.disk_free)} free")
        if prof.ram_total:
            parts.append(f"{human_bytes(prof.ram_available)}/{human_bytes(prof.ram_total)} RAM")
        if prof.gl is not None:
            parts.append("software GL" if prof.gl.software else "hardware GL")
        self._machine_label.setText(" · ".join(parts))

    # -- global plot style --------------------------------------------------

    def global_plot_style(self) -> PlotStyle:
        """Return the session-wide publication :class:`PlotStyle`.

        This is the default starting style for every :class:`ExportDialog`
        opened from any stage.  It can be edited globally via the
        "Publication style…" button in the left panel.
        """
        return self._plot_style

    @staticmethod
    def _load_plot_style() -> PlotStyle:
        from PySide6.QtCore import QSettings

        from dfxm.common.plotting import style_from_json

        raw = QSettings().value("plot_style", "")
        loaded = style_from_json(raw) if raw else None
        return loaded if loaded is not None else replace(PUBLICATION_STYLE)

    def _save_plot_style(self) -> None:
        from PySide6.QtCore import QSettings

        from dfxm.common.plotting import style_to_json

        QSettings().setValue("plot_style", style_to_json(self._plot_style))

    def _on_pub_style(self) -> None:
        """Open the global publication-style editor."""
        from .widgets.export_dialog import StyleControls

        dlg = QDialog(self)
        dlg.setWindowTitle("Publication style")
        dlg.resize(500, 600)

        # StyleControls mutates self._plot_style in place — no copy needed here
        # because changes should persist in the session style after closing.
        controls = StyleControls(self._plot_style, parent=dlg)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls)

        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btn_box.rejected.connect(dlg.reject)

        layout = QVBoxLayout(dlg)
        layout.addWidget(scroll, 1)
        layout.addWidget(btn_box)

        dlg.exec()
        self._save_plot_style()

    def _on_figure_builder(self) -> None:
        """Open (or raise) the non-modal figure-builder window."""
        from .figure_builder import FigureBuilderWindow

        if self._figure_builder is None:
            self._figure_builder = FigureBuilderWindow(
                self._builder_defaults, replace(self._plot_style), parent=self
            )
        self._figure_builder.show()
        self._figure_builder.raise_()
        self._figure_builder.activateWindow()

    def _on_system_check(self) -> None:
        """Open the System check dialog — the only surface that probes GL
        on demand, and the one place that shows the whole machine at once."""
        from .widgets.system_check import SystemCheckDialog

        SystemCheckDialog(self).exec()

    # Which StageResult attribute holds the stacked/aligned output h5 path,
    # for the stages the panel picker needs a CATALOG file from (not their
    # input directory) — see ``_derive_stage_output_h5``.
    _OUTPUT_H5_RESULT_ATTR = {
        "strain": "stacked_path",
        "mosaicity": "stacked_path",
        "rocking": "aligned_path",
    }

    def _builder_defaults(self) -> dict[str, dict]:
        """Per-stage panel-picker defaults: current h5 path, pixel sizes, jobs.

        Consumed by ``gui.widgets.panel_picker.AddPanelDialog`` (via the
        figure builder's ``defaults_provider``) so a fresh Add panels… dialog
        starts from whatever h5 the matching stage's catalog lives in
        (falling back to the experiment's own output-chaining default), not
        a blank field.

        strain/mosaicity/rocking need the stacked/aligned OUTPUT h5 (the file
        the panel-picker catalog reads), not the input directory field
        (root_folder/raw_root) their form otherwise pre-fills — see
        ``_derive_stage_output_h5``. slices/profiles already point straight
        at their own output h5 field.
        """
        import json

        from .bindings import experiment_overrides

        exp = self._experiment_panel.current_experiment()
        sx, sy = exp.pixel_size_x_um, exp.pixel_size_y_um
        out: dict[str, dict] = {}
        field_for = {
            "slices": "mosa_volume_file",
            "profiles": "consolidated_h5",
        }
        for stage in ("strain", "mosaicity", "rocking", "slices", "profiles"):
            values = self._views[stage]._form.values()
            chained = experiment_overrides(stage, exp)
            if stage in self._OUTPUT_H5_RESULT_ATTR:
                last_result = self._views[stage]._last_result
                attr = self._OUTPUT_H5_RESULT_ATTR[stage]
                h5 = getattr(last_result, attr, "") if last_result is not None else ""
                if not h5:
                    h5 = self._derive_stage_output_h5(stage, values, chained)
            else:
                h5 = values.get(field_for[stage]) or chained.get(field_for[stage]) or ""
            jobs: list = []
            if stage == "profiles":
                try:
                    jobs = json.loads(values.get("jobs_json") or "[]")
                except (TypeError, ValueError):
                    jobs = []
            out[stage] = {
                "h5": h5,
                "sx": sx,
                "sy": sy,
                "jobs": jobs if isinstance(jobs, list) else [],
            }
        return out

    @staticmethod
    def _derive_stage_output_h5(stage: str, values: dict, chained: dict) -> str:
        """Best-effort stacked/aligned output h5, mirroring the stage's own
        ``run()`` path derivation — used only as a fallback when the stage
        hasn't been run yet this session (``_builder_defaults`` always
        prefers the real last-run result first).

        strain/mosaicity: ``os.path.join(default_out_root, stacked_filename)``
        where ``default_out_root`` is ``input_folder`` (single mode) or
        ``root_folder`` (batch mode) — see ``dfxm.stages.strain.run``/
        ``dfxm.stages.mosaicity.run``.
        rocking: ``os.path.join(out_dir, aligned_h5_name)`` where ``out_dir``
        is ``output_dir`` or ``raw_root/<default_dir>`` (the default dir and
        default aligned filename both depend on ``source_scan`` — see
        ``dfxm.stages.rocking.run``).
        """
        defaults = STAGE_SPECS[stage].defaults()

        def get(key: str):
            return values.get(key) or chained.get(key) or defaults.get(key)

        if stage in ("strain", "mosaicity"):
            mode = get("mode")
            root = (get("input_folder") if mode == "single" else get("root_folder")) or ""
            root = root.rstrip("/")
            stacked_filename = get("stacked_filename") or ""
            return os.path.join(root, stacked_filename) if root and stacked_filename else ""

        if stage == "rocking":
            raw_root = (get("raw_root") or "").rstrip("/")
            source = get("source_scan")
            default_dir = (
                "aligned_raw_mosa_volumes"
                if source == "mosaicity"
                else "aligned_raw_rocking_volumes"
            )
            out_dir = get("output_dir") or (os.path.join(raw_root, default_dir) if raw_root else "")
            aligned_name = get("aligned_h5_name") or ""
            if source == "mosaicity" and aligned_name == defaults.get("aligned_h5_name"):
                aligned_name = "aligned_raw_mosa_volumes.h5"
            return os.path.join(out_dir, aligned_name) if out_dir and aligned_name else ""

        return ""

    # -- theme --------------------------------------------------------------
    def _sync_theme_btn(self) -> None:
        dark = ThemeController.instance().mode == "dark"
        self._theme_btn.setText("☾ Dark" if dark else "☀ Light")

    def _on_theme_toggle(self, checked: bool) -> None:
        from PySide6.QtCore import QSettings

        mode = "dark" if checked else "light"
        ThemeController.instance().set_mode(mode)
        QSettings().setValue("theme", mode)

    def _on_theme_changed(self, palette) -> None:
        self._sync_theme_btn()
        # QListWidgetItem foreground is not reachable by QSS — refresh it here.
        item = self._status_items.get("concat")
        if item is not None:
            item.setForeground(QBrush(QColor(palette.ink_muted)))

    # -- navigation ---------------------------------------------------------
    def _on_row_changed(self, row: int) -> None:
        if not 0 <= row < len(self._row_target):
            return
        target = self._row_target[row]
        if target == "__overview__":
            self._stack.setCurrentWidget(self._overview)
        elif target is not None:
            self._stack.setCurrentWidget(self._views[target])

    def _show_stage(self, name: str) -> None:
        view = self._views.get(name)
        if view is None:
            return
        self._nav.setCurrentRow(self._row_target.index(name))
        self._stack.setCurrentWidget(view)

    def closeEvent(self, event) -> None:  # Qt hook
        self._window_state.save(self, self._main_splitter)
        self._save_plot_style()
        for view in self._views.values():
            view.flush()  # write any pending debounced form-state save
        # Join every pinned worker QThread (figure-builder render/export,
        # replot-dialog batches, …) before the app tears down. keep_alive
        # (gui/widgets/busy.py) only stops those threads from being
        # garbage-collected mid-flight; it does not stop the process exiting
        # out from under a still-running one, which aborts with
        # "QThread: Destroyed while thread is still running". This is the one
        # sanctioned place the GUI thread blocks on a worker — see
        # wait_for_workers's docstring.
        from .widgets.busy import busy_cursor, wait_for_workers

        with busy_cursor():
            wait_for_workers()
        super().closeEvent(event)

    # -- slots ----------------------------------------------------------------
    def _on_experiment_changed(self, experiment: Experiment) -> None:
        for view in self._views.values():
            view.set_experiment(experiment)

    def _set_glyph(self, stage_name: str, glyph: str) -> None:
        item = self._status_items.get(stage_name)
        if item is not None:
            item.setText(f"{glyph}  {self._item_base[stage_name]}")
        self._overview.set_status(stage_name, glyph)

    def _on_run_started(self, stage_name: str) -> None:
        self._set_glyph(stage_name, _GLYPH_RUNNING)

    def _on_run_finished(self, stage_name: str, ok: bool) -> None:
        self._set_glyph(stage_name, _GLYPH_OK if ok else _GLYPH_FAIL)
