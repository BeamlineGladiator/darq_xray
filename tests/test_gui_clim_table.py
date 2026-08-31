"""Unit tests for the per-volume colour-limit param editor."""

from __future__ import annotations

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from dfxm.stages import visualize as V  # noqa: E402
from gui.widgets.clim_section import ClimGroupSection  # noqa: E402
from gui.widgets.clim_table import ClimTableEditor, summarize_clim  # noqa: E402

_app = QApplication.instance() or QApplication([])


def test_summary_counts_only_the_volumes_with_a_limit():
    assert summarize_clim("") == "all automatic"
    assert summarize_clim("{}") == "all automatic"
    assert summarize_clim('{"strain": [-1, 1]}') == "limits set for 1 of 9 volumes"
    assert summarize_clim('{"strain": [-1, 1], "raw_sum": [null, 5]}').startswith(
        "limits set for 2 of 9"
    )


def test_a_summary_never_raises_on_junk():
    """The summary renders in the form; unparseable text must not crash it.

    The stage's `_clim_overrides` is what rejects bad JSON, with a hint and a
    banner. A widget that throws while painting would hide that message behind
    a traceback.
    """
    for junk in ("{not json}", "[1, 2]", '{"strain": "x"}', "null"):
        assert summarize_clim(junk)


def test_editor_round_trips_its_value():
    ed = ClimTableEditor('{"strain": [-1.0, 1.0]}', "Colour limits")
    assert json.loads(ed.text()) == {"strain": [-1.0, 1.0]}
    ed.setText('{"raw_sum": [0.0, 10.0]}')
    assert json.loads(ed.text()) == {"raw_sum": [0.0, 10.0]}
    assert "1 of 9" in ed._summary.text()


def test_editor_emits_on_set():
    seen = []
    ed = ClimTableEditor("{}", "Colour limits")
    ed.textChanged.connect(seen.append)
    ed.setText('{"strain": [-2, 2]}')
    assert seen == ['{"strain": [-2, 2]}']


def test_the_dialog_offers_every_volume_the_stage_can_render():
    """The rows and the stage's volume list are the same nine, in the same order."""
    section = ClimGroupSection()
    section.set_groups(ClimTableEditor.groups())
    assert list(section._edits) == [key for key, _ds, _label in V._VOLUME_KEYS]


def test_section_seeding_round_trips_through_the_stage_parser():
    """What the dialog writes is what `_clim_overrides` reads back."""
    section = ClimGroupSection()
    section.set_groups(ClimTableEditor.groups())
    section.set_clim_by_group({"strain": (-3e-4, 3e-4), "raw_sum": (None, 12.0)})
    value = json.dumps({k: list(v) for k, v in (section.clim_by_group() or {}).items()})
    assert V._clim_overrides({"volume_clim_json": value}) == {
        "strain": (-3e-4, 3e-4),
        "raw_sum": (None, 12.0),
    }


def test_seeding_leaves_unlisted_volumes_blank():
    section = ClimGroupSection()
    section.set_groups(ClimTableEditor.groups())
    section.set_clim_by_group({"strain": (-1.0, 1.0)})
    assert section.clim_by_group() == {"strain": (-1.0, 1.0)}


def test_reseeding_a_section_drops_the_limits_the_new_mapping_omits():
    """`set_clim_by_group` is a SET: rows the mapping does not name are blanked.

    The one caller today builds a fresh section per dialog, so the two readings
    are indistinguishable there — but the class keeps a persistent `_values`
    cache precisely so a section can be reused, and an update-only seed would
    then collect limits its mapping never contained.
    """
    section = ClimGroupSection()
    section.set_groups(ClimTableEditor.groups())
    section.set_clim_by_group({"strain": (-1.0, 1.0), "raw_sum": (0.0, 9.0)})
    section.set_clim_by_group({"strain": (-2.0, 2.0)})
    assert section.clim_by_group() == {"strain": (-2.0, 2.0)}


def test_param_form_renders_the_clim_editor_for_the_hint():
    """The `editor="clim_table"` hint must actually reach ParamForm."""
    from gui.widgets.param_form import ParamForm

    form = ParamForm(V.STAGE.params)
    assert isinstance(form._editors["volume_clim_json"], ClimTableEditor)
    form.set_values({"volume_clim_json": '{"strain": [-1, 1]}'})
    assert json.loads(form.values()["volume_clim_json"]) == {"strain": [-1, 1]}
