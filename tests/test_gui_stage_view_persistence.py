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


def test_none_default_path_does_not_leak_across_experiments(tmp_path):
    # F1 regression: output_dir has default=None and is NOT in experiment_overrides,
    # so a plain set_values-based reset would leave A's value behind on a switch.
    store = _store(tmp_path)
    a, b = Experiment(name="expA"), Experiment(name="expB")
    view = StageView("strain", STAGE_SPECS["strain"], a, store=store)
    view._form.set_values({"output_dir": "/data/A/out"})
    view.set_experiment(b)  # full reset to B's baseline — A's path must not survive
    assert view._form.values()["output_dir"] == ""
    view._form.set_values({"root_folder": "/b"})  # touch B so it persists
    view.flush()
    assert (store.load("expB", "strain") or {}).get("output_dir", "") == ""


def test_untouched_stage_is_not_persisted(tmp_path):
    # F2 regression: a stage with no user edit must not freeze a snapshot, so it
    # keeps following the experiment.
    store = _store(tmp_path)
    exp = Experiment(name="expA")
    view = StageView("strain", STAGE_SPECS["strain"], exp, store=store)
    view.flush()
    assert store.load("expA", "strain") is None


def test_restore_skips_uncoercible_value_without_crashing(tmp_path):
    # F3 regression: a foreign/hand-edited payload must not crash construction.
    store = _store(tmp_path)
    exp = Experiment(name="expA")
    store.save("expA", "visualize", {"range_pct": "not_a_number", "raw_root": "/ok"})
    view = StageView("visualize", STAGE_SPECS["visualize"], exp, store=store)  # must not raise
    assert view._form.values()["raw_root"] == "/ok"  # the good key still applied


def test_restore_ignores_calibration_key_in_payload(tmp_path):
    # F4 regression: even if an old payload carries a calibration key, the overlay
    # must not apply it — calibration follows the experiment.
    store = _store(tmp_path)
    exp = Experiment(name="expA")
    store.save("expA", "strain", {"pixel_size_x_um": 0.999, "root_folder": "/x"})
    view = StageView("strain", STAGE_SPECS["strain"], exp, store=store)
    vals = view._form.values()
    assert vals["root_folder"] == "/x"  # non-calibration value restored
    assert vals["pixel_size_x_um"] != 0.999  # calibration NOT overlaid from the payload


def test_no_store_means_no_persistence(tmp_path):
    # Without a store, StageView keeps the legacy behaviour and writes nothing.
    store = _store(tmp_path)
    exp = Experiment(name="expA")
    view = StageView("strain", STAGE_SPECS["strain"], exp)  # no store
    view._form.set_values({"root_folder": "/nope"})
    view.flush()  # must be a no-op
    assert store.load("expA", "strain") is None
