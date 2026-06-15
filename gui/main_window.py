"""Main window: compact experiment header + pipeline rail + stage stack.

The left column holds the :class:`~gui.experiment_panel.ExperimentPanel`
(compact header) above a single *pipeline rail*: Overview first, then the
stages in pipeline order with a status glyph each. darfix appears as a
disabled external row after concat; concat is marked optional. The right
side is a stacked set of :class:`~gui.stage_view.StageView` panels behind
an :class:`~gui.overview_page.OverviewPage` landing page.
"""

from __future__ import annotations

from dataclasses import replace

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
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

from dfxm.common.plotting import PUBLICATION_STYLE, PlotStyle
from dfxm.config.models import Experiment

from .bindings import STAGE_ORDER, STAGE_SPECS
from .experiment_panel import ExperimentPanel
from .overview_page import OverviewPage
from .stage_view import StageView

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

        # Session-wide publication style — seeded from the module constant but
        # held as an independent copy so mutations never touch PUBLICATION_STYLE.
        self._plot_style: PlotStyle = replace(PUBLICATION_STYLE)

        self._experiment_panel = ExperimentPanel()
        experiment = self._experiment_panel.current_experiment()

        # Stage views + overview page (stacked).
        self._stack = QStackedWidget()
        self._overview = OverviewPage(STAGE_ORDER, STAGE_SPECS)
        self._overview.stageSelected.connect(self._show_stage)
        self._stack.addWidget(self._overview)
        self._views: dict[str, StageView] = {}
        for name in STAGE_ORDER:
            view = StageView(name, STAGE_SPECS[name], experiment)
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
        muted = QBrush(QColor("#888888"))
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

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self._experiment_panel)
        left_layout.addWidget(self._nav, 1)
        left_layout.addWidget(self._pub_style_btn)

        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(self._stack)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([380, 720])
        self.setCentralWidget(splitter)

    # -- global plot style --------------------------------------------------

    def global_plot_style(self) -> PlotStyle:
        """Return the session-wide publication :class:`PlotStyle`.

        This is the default starting style for every :class:`ExportDialog`
        opened from any stage.  It can be edited globally via the
        "Publication style…" button in the left panel.
        """
        return self._plot_style

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
