"""ExperimentPanel: New… (blank preset saved to experiments/) + Rescan button."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox  # noqa: E402

from darq_xray.config import presets  # noqa: E402
from darq_xray.config.models import Experiment  # noqa: E402


@pytest.fixture
def exp_dir(tmp_path, monkeypatch):
    """An isolated experiments/ folder holding one calibrated preset."""
    d = tmp_path / "experiments"
    d.mkdir()
    presets.save_experiment(
        Experiment(name="Old", raw_root="/data/raw", ccmth_ref_deg=7.1, pixel_size_x_um=0.15),
        d / "Old.yaml",
    )
    monkeypatch.setattr(presets, "experiments_dir", lambda: d)
    return d


def _panel():
    from darq_xray.gui.experiment_panel import ExperimentPanel

    _ = QApplication.instance() or QApplication([])
    return ExperimentPanel()


def _button(widget, text):
    from PySide6.QtWidgets import QPushButton

    matches = [b for b in widget.findChildren(QPushButton) if b.text() == text]
    assert matches, f"no {text!r} button"
    return matches[0]


def _fake_save(monkeypatch, path):
    calls = []

    def fake(*a, **k):
        calls.append(a)
        return (os.fspath(path) if path else "", "")

    monkeypatch.setattr(QFileDialog, "getSaveFileName", fake)
    return calls


def test_rescan_button_is_wide_enough_for_its_label(exp_dir):
    panel = _panel()
    btn = _button(panel, "Rescan")
    assert btn.maximumWidth() >= btn.sizeHint().width()


def test_new_dialog_starts_blank_not_from_the_current_preset(exp_dir):
    panel = _panel()
    assert panel.current_experiment().name == "Old"
    _button(panel, "New…")
    dlg = panel._make_new_dialog()
    assert dlg.windowTitle() == "New experiment"
    assert dlg.experiment() == Experiment(name="")


def test_new_ok_refuses_a_blank_name(exp_dir, monkeypatch):
    panel = _panel()  # keep alive: the dialog is its child
    dlg = panel._make_new_dialog()
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a))
    saves = _fake_save(monkeypatch, exp_dir / "x.yaml")
    dlg._on_accept()
    assert warned and "name" in warned[0][2].lower()
    assert not saves
    assert dlg.result() != QDialog.DialogCode.Accepted


def test_new_ok_refuses_a_name_already_taken(exp_dir, monkeypatch):
    panel = _panel()  # keep alive: the dialog is its child
    dlg = panel._make_new_dialog()
    dlg._form.set_values({"name": "Old"})
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a))
    saves = _fake_save(monkeypatch, exp_dir / "x.yaml")
    dlg._on_accept()
    assert warned and "Old" in warned[0][2]
    assert not saves
    assert dlg.result() != QDialog.DialogCode.Accepted


def test_new_ok_with_save_cancelled_stays_open(exp_dir, monkeypatch):
    panel = _panel()  # keep alive: the dialog is its child
    dlg = panel._make_new_dialog()
    dlg._form.set_values({"name": "Fresh"})
    _fake_save(monkeypatch, None)
    dlg._on_accept()
    assert dlg.result() != QDialog.DialogCode.Accepted
    assert "Fresh" not in presets.discover_experiments()


def test_new_ok_saves_into_experiments_and_accepts(exp_dir, monkeypatch):
    panel = _panel()  # keep alive: the dialog is its child
    dlg = panel._make_new_dialog()
    dlg._form.set_values({"name": "Fresh"})
    saves = _fake_save(monkeypatch, exp_dir / "Fresh.yaml")
    dlg._on_accept()
    assert saves and saves[0][2].endswith(os.path.join("experiments", "Fresh.yaml"))
    assert dlg.result() == QDialog.DialogCode.Accepted
    assert presets.load_experiment(exp_dir / "Fresh.yaml").name == "Fresh"


def test_on_new_selects_the_saved_preset(exp_dir, monkeypatch):
    panel = _panel()
    emitted = []
    panel.experimentChanged.connect(emitted.append)
    fresh = Experiment(name="Fresh", raw_root="/data/fresh")
    presets.save_experiment(fresh, exp_dir / "Fresh.yaml")

    class _Accepted:
        def exec(self):
            return True

        def experiment(self):
            return fresh

        def deleteLater(self):
            pass

    monkeypatch.setattr(panel, "_make_new_dialog", lambda: _Accepted())
    panel._on_new()
    assert panel._combo.currentText() == "Fresh"
    assert panel.current_experiment() == fresh
    assert emitted and emitted[-1] == fresh
