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
    QMessageBox,
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
        self._form.focusCleared.connect(help_panel.show_idle)

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
        compute_btn = QPushButton("Compute pixel size from scan…")
        compute_btn.setToolTip(
            "Read a raw (pre-darfix) scan .h5 and fill Pixel size X/Y from its motors"
        )
        compute_btn.clicked.connect(self._on_compute_pixel_size)
        buttons.addButton(compute_btn, QDialogButtonBox.ButtonRole.ActionRole)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)

        self._roi_note = QLabel("")
        self._roi_note.setProperty("role", "muted")
        self._roi_note.setWordWrap(True)
        self._form.changed.connect(self._update_roi_note)
        self._update_roi_note()

        layout = QVBoxLayout(self)
        layout.addWidget(scroll, 1)
        layout.addWidget(self._roi_note)
        layout.addWidget(help_panel)
        layout.addWidget(buttons)

    def experiment(self) -> Experiment:
        return Experiment.from_dict(self._form.values())

    def _update_roi_note(self) -> None:
        """Live ROI read-out: darfix window + analysis window in detector px."""
        from dfxm.common import roi as R

        vals = self._form.values()
        try:
            win = R.parse_darfix_roi(vals.get("darfix_roi", ""))
            det_x, det_y = R.analysis_detector_window(
                vals.get("darfix_roi", ""),
                vals.get("analysis_roi_x", ""),
                vals.get("analysis_roi_y", ""),
            )
        except ValueError as exc:
            self._roi_note.setText(f"ROI: {exc}")
            return
        if win is None or det_x is None or det_y is None:
            self._roi_note.setText("")
            return
        self._roi_note.setText(
            f"Detector window: x {win.x0}→{win.x1}, y {win.y0}→{win.y1} · analysis in "
            f"detector px: x {det_x[0]}→{det_x[1]}, y {det_y[0]}→{det_y[1]}"
        )

    def _roi_problems(self) -> list[str]:
        from dfxm.common.roi import validate_rois

        vals = self._form.values()
        return validate_rois(
            vals.get("darfix_roi", ""),
            vals.get("analysis_roi_x", ""),
            vals.get("analysis_roi_y", ""),
        )

    def _warn_roi_problems(self) -> bool:
        """True (and a dialog shown) when the ROI fields are unsaveable."""
        problems = self._roi_problems()
        if not problems:
            return False
        QMessageBox.warning(
            self,
            "Regions of interest",
            "\n".join(problems)
            + "\n\nDarfix ROI is 'x,y,w,h' exactly as darfix shows it (origin+size); "
            "analysis windows are map-frame 'start,end' relative to that window.",
        )
        return True

    def _on_accept(self) -> None:
        if self._warn_roi_problems():
            return
        self.accept()

    def _on_save_as(self) -> None:
        if self._warn_roi_problems():
            return
        exp = self.experiment()
        start = os.fspath(presets.experiments_dir() / f"{exp.name or 'experiment'}.yaml")
        path, _ = QFileDialog.getSaveFileName(self, "Save preset", start, "YAML (*.yaml)")
        if path:
            presets.save_experiment(exp, path)

    def _apply_pixel_size(self, path: str):
        """Compute pixel sizes from *path* and write them into the form.

        No dialogs — raises StageUserError on a user-fixable problem so the
        caller can surface message/hint. Returns the PixelSizeResult.
        """
        from dfxm.common.pixel_size import compute_pixel_size

        vals = self._form.values()
        res = compute_pixel_size(
            path,
            positioners_path=vals.get("positioners_path") or "instrument/positioners",
            entry_suffix=vals.get("entry_suffix") or ".1",
        )
        self._form.set_values(
            {
                "pixel_size_x_um": res.pixel_size_x_um,
                "pixel_size_y_um": res.pixel_size_y_um,
            }
        )
        return res

    def _on_compute_pixel_size(self) -> None:
        from dfxm.common.errors import StageUserError

        vals = self._form.values()
        path, _ = QFileDialog.getOpenFileName(
            self, "Pick a raw scan .h5", vals.get("raw_root") or "", "HDF5 (*.h5 *.hdf5)"
        )
        if not path:
            return
        try:
            res = self._apply_pixel_size(path)
        except StageUserError as exc:
            QMessageBox.warning(self, "Compute pixel size", f"{exc}\n\n{exc.hint}")
            return
        except Exception as exc:  # noqa: BLE001 — unreadable/foreign file
            QMessageBox.warning(self, "Compute pixel size", f"Could not read scan:\n{exc}")
            return
        QMessageBox.information(
            self,
            "Compute pixel size",
            f"Objective {res.objective} (ffsel={res.ffsel:g})\n"
            f"M = {res.magnification:.3f}\n"
            f"2θ = {res.two_theta_deg:.3f}°\n"
            f"condenser {'in' if res.condenser_in else 'out'}\n\n"
            f"Pixel size X = {res.pixel_size_x_um:.4f} µm\n"
            f"Pixel size Y = {res.pixel_size_y_um:.4f} µm",
        )


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
        try:
            if dlg.exec():
                self._set_experiment(dlg.experiment())
                self._reload_combo_keep_selection()
        finally:
            dlg.deleteLater()

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
