"""ExperimentDialog: Initialize from data… wiring + OK-time save prompt."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox  # noqa: E402


def _dlg(**fields):
    from dfxm.config.models import Experiment
    from gui.experiment_panel import ExperimentDialog

    _ = QApplication.instance() or QApplication([])
    return ExperimentDialog(Experiment(**fields))


def _make_raw(tmp_path):
    raw = tmp_path / "RAW"
    d = raw / "s_strain__0"
    d.mkdir(parents=True)
    with h5py.File(d / "s_strain__0.h5", "w") as f:
        pos = f.create_group("1.1/instrument/positioners")
        for k, v in dict(
            mainx=-5000.0, obx=273.0, ffsel=-60.0, ffz=2100.0, lenssel=0.0, ccmth=7.1
        ).items():
            pos.create_dataset(k, data=v)
    return str(raw)


def test_initialize_button_exists():
    from PySide6.QtWidgets import QPushButton

    dlg = _dlg()
    labels = [b.text() for b in dlg.findChildren(QPushButton)]
    assert any("Initialize from data" in t for t in labels)


def test_blank_raw_root_warns_and_aborts(monkeypatch):
    dlg = _dlg()
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a))
    dlg._on_initialize_from_data()
    assert warned and "Raw data root" in warned[0][2]


def test_detect_seam_runs_on_form_values(tmp_path):
    raw = _make_raw(tmp_path)
    dlg = _dlg()
    dlg._form.set_values({"raw_root": raw})  # typed into the form, never saved
    rows = {d.field: d for d in dlg._detect(dlg._form.values())}
    assert rows["folder_pattern"].value == "s_strain__*"
    assert rows["pixel_size_x_um"].value


def test_apply_marks_and_ok_prompts_save(tmp_path, monkeypatch):
    raw = _make_raw(tmp_path)
    dlg = _dlg(raw_root=raw)
    # simulate the review dialog having applied values
    dlg._form.set_values({"folder_pattern": "s_strain__*"})
    dlg._applied_detections = True
    asked = []
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *a, **k: (asked.append(a), QMessageBox.StandardButton.No)[1],
    )
    dlg._on_accept()
    assert asked  # prompted
    assert dlg.result() == QDialog.DialogCode.Accepted  # "No" still accepts
    assert dlg._applied_detections is False  # asks once


def test_ok_without_apply_never_prompts(monkeypatch):
    dlg = _dlg()
    asked = []
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *a, **k: (asked.append(a), QMessageBox.StandardButton.No)[1],
    )
    dlg._on_accept()
    assert not asked
    assert dlg.result() == QDialog.DialogCode.Accepted


def test_save_choice_routes_to_save_as(monkeypatch):
    dlg = _dlg()
    dlg._applied_detections = True
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Save)
    saved = []
    monkeypatch.setattr(dlg, "_on_save_as", lambda: saved.append(True))
    dlg._on_accept()
    assert saved
