"""StageView per-experiment form-state persistence (save-on-edit / restore)."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from dfxm.config.models import Experiment  # noqa: E402
from gui.bindings import STAGE_SPECS  # noqa: E402
from gui.form_state import FormStateStore  # noqa: E402
from gui.stage_view import StageView  # noqa: E402

_app = QApplication.instance() or QApplication([])


def _store(tmp_path):
    return FormStateStore(QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat))


def test_flush_persists_and_a_fresh_view_restores(tmp_path):
    store = _store(tmp_path)
    exp = Experiment(name="expA")
    view = StageView("strain", STAGE_SPECS["strain"], exp, store=store)
    view._form.set_values({"root_folder": "/data/here"})
    view.flush()
    # a fresh view for the same experiment restores the edited value on construction
    view2 = StageView("strain", STAGE_SPECS["strain"], exp, store=store)
    assert view2._form.values()["root_folder"] == "/data/here"


def test_calibration_params_never_persisted(tmp_path):
    store = _store(tmp_path)
    exp = Experiment(name="expA")
    view = StageView("strain", STAGE_SPECS["strain"], exp, store=store)
    # pixel_size_x_um is calibration=True — it must follow the experiment, not persist
    view._form.set_values({"pixel_size_x_um": 0.999, "root_folder": "/x"})
    view.flush()
    saved = store.load("expA", "strain")
    assert "pixel_size_x_um" not in saved
    assert saved["root_folder"] == "/x"


def test_set_experiment_saves_outgoing_and_loads_incoming(tmp_path):
    store = _store(tmp_path)
    a, b = Experiment(name="expA"), Experiment(name="expB")
    view = StageView("strain", STAGE_SPECS["strain"], a, store=store)
    view._form.set_values({"root_folder": "/from-A"})
    view.set_experiment(b)  # saves A; B has no saved state → B baseline
    assert view._form.values()["root_folder"] != "/from-A"
    view._form.set_values({"root_folder": "/from-B"})
    view.set_experiment(a)  # saves B; restores A's saved value
    assert view._form.values()["root_folder"] == "/from-A"
    view.set_experiment(b)  # B's own saved value comes back
    assert view._form.values()["root_folder"] == "/from-B"


def test_no_store_means_no_persistence(tmp_path):
    # Without a store, StageView keeps the legacy behaviour and writes nothing.
    store = _store(tmp_path)
    exp = Experiment(name="expA")
    view = StageView("strain", STAGE_SPECS["strain"], exp)  # no store
    view._form.set_values({"root_folder": "/nope"})
    view.flush()  # must be a no-op
    assert store.load("expA", "strain") is None
