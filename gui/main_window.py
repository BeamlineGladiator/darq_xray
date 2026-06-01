"""Main window: experiment selector + stage navigation + status panel.

Left column holds the :class:`~gui.experiment_panel.ExperimentPanel`, a stage
navigation list, and a small per-stage run-status panel. The right side is a
stacked set of :class:`~gui.stage_view.StageView` panels, one per stage.

Changing the experiment re-pushes its derived defaults into every stage view;
finishing a run updates that stage's status indicator.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from dfxm.config.models import Experiment

from .bindings import STAGE_ORDER, STAGE_SPECS
from .experiment_panel import ExperimentPanel
from .stage_view import StageView

_STATUS_IDLE = "—"
_STATUS_OK = "✓"
_STATUS_FAIL = "✗"


class MainWindow(QMainWindow):
    """Top-level window wiring experiment + stages + status together."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DFXM pipeline")
        self.resize(1100, 720)

        # Experiment panel (loads the first preset on construction).
        self._experiment_panel = ExperimentPanel()
        experiment = self._experiment_panel.current_experiment()

        # Stage navigation + status.
        self._nav = QListWidget()
        self._status = QListWidget()
        self._status.setMaximumHeight(140)
        self._status.setSelectionMode(QListWidget.SelectionMode.NoSelection)

        # Stage views (stacked).
        self._stack = QStackedWidget()
        self._views: dict[str, StageView] = {}
        self._status_items: dict[str, QListWidgetItem] = {}
        for name in STAGE_ORDER:
            spec = STAGE_SPECS[name]
            view = StageView(name, spec, experiment)
            view.runFinished.connect(self._on_run_finished)
            self._views[name] = view
            self._stack.addWidget(view)
            self._nav.addItem(spec.label)
            item = QListWidgetItem(f"{_STATUS_IDLE}  {spec.label}")
            self._status.addItem(item)
            self._status_items[name] = item

        self._nav.currentRowChanged.connect(self._stack.setCurrentIndex)
        self._nav.setCurrentRow(0)

        # Wire experiment changes into every stage view.
        self._experiment_panel.experimentChanged.connect(self._on_experiment_changed)

        # Layout.
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self._experiment_panel, 1)
        left_layout.addWidget(self._nav)
        left_layout.addWidget(self._status)

        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(self._stack)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([380, 720])
        self.setCentralWidget(splitter)

    # -- slots ------------------------------------------------------------
    def _on_experiment_changed(self, experiment: Experiment) -> None:
        for view in self._views.values():
            view.set_experiment(experiment)

    def _on_run_finished(self, stage_name: str, ok: bool) -> None:
        item = self._status_items.get(stage_name)
        if item is None:
            return
        label = STAGE_SPECS[stage_name].label
        mark = _STATUS_OK if ok else _STATUS_FAIL
        item.setText(f"{mark}  {label}")
