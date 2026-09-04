"""Unit tests for the shared per-kind colour-limit section widget."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from darq_xray.gui.widgets.clim_section import ClimGroupSection  # noqa: E402

_app = QApplication.instance() or QApplication([])


def test_clim_section_builds_rows_and_collects_dict():
    sec = ClimGroupSection()
    sec.set_groups([("strain", "Strain"), ("raw", "Raw")])
    assert set(sec._edits) == {"strain", "raw"}
    # nothing filled → None (cores fall through to stored limits)
    assert sec.clim_by_group() is None
    sec._edits["strain"][0].setText("-3")
    sec._edits["strain"][1].setText("3")
    sec._edits["raw"][1].setText("100")  # vmax only; vmin stays stored
    assert sec.clim_by_group() == {"strain": (-3.0, 3.0), "raw": (None, 100.0)}


def test_clim_section_preserves_values_across_reload():
    sec = ClimGroupSection()
    sec.set_groups([("strain", "Strain"), ("raw", "Raw")])
    sec._edits["strain"][0].setText("-3")
    # reload with the same groups → typed value survives
    sec.set_groups([("strain", "Strain"), ("raw", "Raw")])
    assert sec._edits["strain"][0].text() == "-3"
    # a group that disappears drops its value; a new one starts blank
    sec.set_groups([("mosa_com", "COM")])
    assert set(sec._edits) == {"mosa_com"}
    assert sec._edits["mosa_com"][0].text() == ""


def test_clim_section_survives_empty_intermediate_rebuild():
    # mirrors the old dialog _reload path (set_groups([]) then the real groups);
    # the persistent cache must carry the typed value through the empty call.
    sec = ClimGroupSection()
    sec.set_groups([("strain", "Strain")])
    sec._edits["strain"][0].setText("0.5")
    sec.set_groups([])  # empty intermediate rebuild
    sec.set_groups([("strain", "Strain")])
    assert sec._edits["strain"][0].text() == "0.5"


def test_clim_section_validate_flags_bad_input():
    sec = ClimGroupSection()
    sec.set_groups([("strain", "Strain")])
    assert sec.validate() is None
    sec._edits["strain"][0].setText("abc")
    assert "vmin" in sec.validate()
