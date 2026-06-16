"""Compact experiment header + modal editor dialog.

The header shows: preset dropdown, a one-line calibration summary, the
preset's notes (red, when present), and an **Edit…** button that opens
:class:`ExperimentDialog` — the full schema-driven field form (with help
panel) plus **Save as…**. Emits :attr:`experimentChanged` whenever the
active experiment changes so the stage views re-pull their defaults.
"""

from __future__ import annotations

import os

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from dfxm.config import presets
from dfxm.config.models import EXPERIMENT_SCHEMA, Experiment

from .widgets.help_panel import HelpPanel
from .widgets.param_form import ParamForm


class ExperimentDialog(QDialog):
    """Modal editor for every experiment field (schema-driven)."""

    def __init__(self, experiment: Experiment, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit experiment")
        self.resize(560, 640)

        self._form = ParamForm(EXPERIMENT_SCHEMA, experiment.to_dict())
        help_panel = HelpPanel()
        help_panel.set_idle(
            "Experiment",
            "Shared state every stage inherits: data roots, folder patterns, "
            "calibration constants and beamline HDF5 paths.",
        )
        self._form.focusedParamChanged.connect(help_panel.show_param)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._form)

        save_btn = QPushButton("Save as…")
        save_btn.setToolTip("Write the edited fields to a new preset YAML")
        save_btn.clicked.connect(self._on_save_as)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.addButton(save_btn, QDialogButtonBox.ButtonRole.ActionRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(scroll, 1)
        layout.addWidget(help_panel)
        layout.addWidget(buttons)

    def experiment(self) -> Experiment:
        return Experiment.from_dict(self._form.values())

    def _on_save_as(self) -> None:
        exp = self.experiment()
        start = os.fspath(presets.experiments_dir() / f"{exp.name or 'experiment'}.yaml")
        path, _ = QFileDialog.getSaveFileName(self, "Save preset", start, "YAML (*.yaml)")
        if path:
            presets.save_experiment(exp, path)


class ExperimentPanel(QWidget):
    """Preset dropdown + calibration summary + notes + Edit… dialog."""

    experimentChanged = Signal(Experiment)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._presets: dict[str, object] = {}
        self._experiment = Experiment()

        self._combo = QComboBox()
        self._combo.currentTextChanged.connect(self._on_select)
        reload_btn = QPushButton("↻")
        reload_btn.setToolTip("Rescan the experiments/ folder")
        reload_btn.setFixedWidth(32)
        reload_btn.clicked.connect(self._reload_presets)
        edit_btn = QPushButton("Edit…")
        edit_btn.setToolTip("Open the full experiment editor")
        edit_btn.clicked.connect(self._on_edit)

        top = QHBoxLayout()
        top.addWidget(QLabel("Experiment:"))
        top.addWidget(self._combo, 1)
        top.addWidget(reload_btn)
        top.addWidget(edit_btn)

        self._summary = QLabel("")
        self._summary.setProperty("role", "muted")
        self._summary.setWordWrap(True)

        self._notes = QLabel("")
        self._notes.setWordWrap(True)
        self._notes.setProperty("role", "notes")
        self._notes.setVisible(False)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self._summary)
        layout.addWidget(self._notes)

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
        self._set_experiment(presets.load_experiment(self._presets[name]))

    # -- editing ----------------------------------------------------------
    def current_experiment(self) -> Experiment:
        return self._experiment

    def _make_dialog(self) -> ExperimentDialog:
        return ExperimentDialog(self._experiment, self)

    def _on_edit(self) -> None:
        dlg = self._make_dialog()
        if dlg.exec():
            self._set_experiment(dlg.experiment())
            self._reload_combo_keep_selection()

    def _reload_combo_keep_selection(self) -> None:
        """Pick up presets Save-as may have written, without re-emitting."""
        current = self._combo.currentText()
        self._presets = presets.discover_experiments()
        self._combo.blockSignals(True)
        self._combo.clear()
        self._combo.addItems(list(self._presets.keys()))
        if current in self._presets:
            self._combo.setCurrentText(current)
        self._combo.blockSignals(False)

    def _set_experiment(self, exp: Experiment) -> None:
        self._experiment = exp
        self._summary.setText(
            f"ccmth {exp.ccmth_ref_deg:g}° · {exp.pixel_size_x_um:g}×{exp.pixel_size_y_um:g} µm/px"
        )
        notes = (exp.notes or "").strip()
        self._notes.setText(f"⚠ {notes}" if notes else "")
        self._notes.setVisible(bool(notes))
        self.experimentChanged.emit(exp)
