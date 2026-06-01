"""Experiment selector + editor.

Lets the user pick a named preset, edit its fields (calibration constants are
flagged red by :class:`~gui.widgets.param_form.ParamForm`), surface the preset's
notes (e.g. the ``mu_ref`` discrepancy), and Save-as a new preset. Emits
:attr:`experimentChanged` whenever the active experiment changes so the stage
views can re-pull their defaults.
"""

from __future__ import annotations

import os

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from dfxm.config import presets
from dfxm.config.models import EXPERIMENT_SCHEMA, Experiment

from .widgets.param_form import ParamForm


class ExperimentPanel(QWidget):
    """Preset dropdown + experiment field editor + notes + Save-as."""

    experimentChanged = Signal(Experiment)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._presets: dict[str, object] = {}

        self._combo = QComboBox()
        self._combo.currentTextChanged.connect(self._on_select)
        reload_btn = QPushButton("↻")
        reload_btn.setToolTip("Rescan the experiments/ folder")
        reload_btn.setFixedWidth(32)
        reload_btn.clicked.connect(self._reload_presets)

        top = QHBoxLayout()
        top.addWidget(QLabel("Experiment:"))
        top.addWidget(self._combo, 1)
        top.addWidget(reload_btn)

        self._notes = QLabel("")
        self._notes.setWordWrap(True)
        self._notes.setStyleSheet("color: #b00020; font-style: italic;")
        self._notes.setVisible(False)

        # Start from an empty experiment; replaced when a preset loads.
        self._form = ParamForm(EXPERIMENT_SCHEMA, Experiment().to_dict())

        apply_btn = QPushButton("Apply")
        apply_btn.setToolTip("Apply edited fields to the active experiment")
        apply_btn.clicked.connect(self._on_apply)
        save_btn = QPushButton("Save as…")
        save_btn.clicked.connect(self._on_save_as)

        btn_row = QHBoxLayout()
        btn_row.addWidget(apply_btn)
        btn_row.addWidget(save_btn)
        btn_row.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self._notes)
        layout.addWidget(self._form)
        layout.addLayout(btn_row)
        layout.addStretch(1)

        self._reload_presets()

    # -- presets ----------------------------------------------------------
    def _reload_presets(self) -> None:
        self._presets = presets.discover_experiments()
        self._combo.blockSignals(True)
        self._combo.clear()
        self._combo.addItems(list(self._presets.keys()))
        self._combo.blockSignals(False)
        if self._presets:
            # Loads the first preset and emits experimentChanged.
            self._on_select(self._combo.currentText())

    def _on_select(self, name: str) -> None:
        if not name or name not in self._presets:
            return
        exp = presets.load_experiment(self._presets[name])
        self._form.set_values(exp.to_dict())
        self._show_notes(exp.notes)
        self.experimentChanged.emit(exp)

    # -- editing ----------------------------------------------------------
    def current_experiment(self) -> Experiment:
        return Experiment.from_dict(self._form.values())

    def _on_apply(self) -> None:
        exp = self.current_experiment()
        self._show_notes(exp.notes)
        self.experimentChanged.emit(exp)

    def _on_save_as(self) -> None:
        exp = self.current_experiment()
        start = os.fspath(presets.experiments_dir() / f"{exp.name or 'experiment'}.yaml")
        path, _ = QFileDialog.getSaveFileName(self, "Save preset", start, "YAML (*.yaml)")
        if not path:
            return
        presets.save_experiment(exp, path)
        self._reload_presets()

    def _show_notes(self, notes: str) -> None:
        notes = (notes or "").strip()
        self._notes.setText(f"⚠ {notes}" if notes else "")
        self._notes.setVisible(bool(notes))
