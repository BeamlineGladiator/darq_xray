"""Self-describing ENUM dropdowns: the label is shown, the value is stored.

The whole point of the feature is a dropdown that reads "geom — high
intensities" instead of "geom". The whole *risk* of it is that the stored value
stops being `geom` — which `pyvista.opacity_transfer_function` needs verbatim,
and which every saved form value already is. These tests pin the split.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from dfxm.common import render3d as R3
from dfxm.config.models import Param, ParamType
from dfxm.stages import visualize as V


def test_a_label_table_covers_every_choice_it_describes():
    """A missing entry would silently show a bare value beside labelled ones."""
    assert set(R3.OPACITY_MAPPING_LABELS) == set(R3.OPACITY_MAPPINGS)
    assert set(R3.RENDER_MODE_LABELS) == set(R3.RENDER_MODES)
    for table in (R3.OPACITY_MAPPING_LABELS, R3.RENDER_MODE_LABELS):
        for value, label in table.items():
            assert label.startswith(value), f"{label!r} must still name {value!r}"
            assert label != value, f"{value!r} has no description"


def test_choice_labels_must_match_choices_one_for_one():
    Param("m", ParamType.ENUM, "M", choices=("a", "b"), choice_labels=("a — x", "b — y"))
    Param("m", ParamType.ENUM, "M", choices=("a", "b"))  # unlabelled stays legal
    with pytest.raises(ValueError):
        Param("m", ParamType.ENUM, "M", choices=("a", "b"), choice_labels=("only one",))


def test_visualize_labels_its_three_dimensional_dropdowns():
    spec = V.STAGE
    labelled = {p.name for p in spec.params if p.choice_labels}
    assert "render_mode" in labelled
    assert "opacity_mapping" in labelled
    for key, _ds, _label in V._VOLUME_KEYS:
        assert f"opacity_mapping_{key}" in labelled
    # ...and the per-volume ones keep their inherit choice, described too
    per_volume = spec.get(f"opacity_mapping_{V._VOLUME_KEYS[0][0]}")
    assert per_volume.choices[0] == V._INHERIT
    assert per_volume.choice_labels[0].startswith(V._INHERIT)
    assert per_volume.choice_labels[0] != V._INHERIT


def test_every_labelled_param_still_defaults_to_a_real_value():
    """The default is a VALUE; a label leaking into it would fail at render time."""
    for p in V.STAGE.params:
        if p.choice_labels:
            assert p.default in p.choices, p.name


def test_labelled_enum_stores_the_value_not_the_label():
    """The round trip ParamForm.set_values/values must carry `geom`, not its label."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from gui.widgets.param_form import ParamForm

    _app = QApplication.instance() or QApplication([])
    form = ParamForm(V.STAGE.params)
    form.set_values({"opacity_mapping": "geom", "render_mode": "isosurface"})
    values = form.values()
    assert values["opacity_mapping"] == "geom"
    assert values["render_mode"] == "isosurface"
    # the widget shows the description while holding the bare value
    box = form._editors["opacity_mapping"]
    assert box.currentText() == R3.OPACITY_MAPPING_LABELS["geom"]
    assert box.currentData() == "geom"


def test_an_unlabelled_enum_is_untouched():
    """Only params that opt in change behaviour — every other ENUM keeps currentText."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from gui.widgets.param_form import ParamForm

    _app = QApplication.instance() or QApplication([])
    form = ParamForm(V.STAGE.params)
    form.set_values({"output_format": "gif"})
    assert form.values()["output_format"] == "gif"
    assert form._editors["output_format"].currentText() == "gif"


def test_the_viewer_mapping_combo_shows_labels_and_yields_values():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication, QComboBox

    from gui.widgets.viewer3d_window import fill_mapping_combo

    _app = QApplication.instance() or QApplication([])
    box = QComboBox()
    fill_mapping_combo(box)
    assert [box.itemData(i) for i in range(box.count())] == list(R3.OPACITY_MAPPINGS)
    assert box.itemText(2) == R3.OPACITY_MAPPING_LABELS[R3.OPACITY_MAPPINGS[2]]
