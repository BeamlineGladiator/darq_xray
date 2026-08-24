"""The publication style a finished run actually used (offscreen Qt)."""

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QDialog, QLabel  # noqa: E402

_app = QApplication.instance() or QApplication([])

from dfxm.common.plotting import PlotStyle  # noqa: E402
from dfxm.config.models import Experiment  # noqa: E402
from gui.bindings import STAGE_SPECS  # noqa: E402
from gui.main_window import MainWindow  # noqa: E402
from gui.stage_view import StageView, style_stamp  # noqa: E402


def test_no_style_stamps_nothing():
    assert style_stamp(None) == ""


def test_the_stamp_names_all_four_group_colormaps_and_the_font_scale():
    style = PlotStyle(
        cmap_mosa_com="fast",
        cmap_mosa_fwhm="magma",
        cmap_strain="RdBu_r",
        cmap_raw="gray",
        font_scale=1.25,
    )
    stamp = style_stamp(style)
    for expected in ("fast", "magma", "RdBu_r", "gray", "1.25"):
        assert expected in stamp, expected


def test_the_stamp_says_it_is_the_style_the_run_used():
    stamp = style_stamp(PlotStyle())
    assert "rendered with" in stamp.lower()


def test_the_stamp_reflects_the_captured_style_not_the_current_one():
    at_launch = PlotStyle(cmap_strain="turbo")
    edited_since = PlotStyle(cmap_strain="seismic")
    assert "turbo" in style_stamp(at_launch)
    assert "seismic" not in style_stamp(at_launch)  # precondition: they differ
    assert "seismic" in style_stamp(edited_since)


def test_finished_run_records_the_style_it_was_launched_with_not_the_current_one():
    """The whole point of the feature, driven through a real StageView.

    ``_on_run`` captures the session style into ``_last_style``; editing the
    style afterwards must NOT change what a finished run says it rendered with.
    A ``_finish_ok`` that read ``window.global_plot_style()`` instead would pass
    the pure-function tests above and fail here.
    """
    view = StageView("strain", STAGE_SPECS["strain"], Experiment())
    # A parentless StageView is its own window(), so this IS what _finish_ok
    # would reach for if it (wrongly) consulted the live session style.
    view.global_plot_style = lambda: PlotStyle(cmap_strain="seismic")
    view._last_style = PlotStyle(cmap_strain="turbo")  # what the run launched with
    assert view.global_plot_style().cmap_strain != "turbo"  # precondition: they differ

    view._finish_ok(SimpleNamespace(layers=[], skipped=[]))

    text = view._results.toPlainText()
    assert "strain=turbo" in text
    assert "seismic" not in text


def test_a_run_that_captured_no_style_leaves_the_summary_alone():
    view = StageView("strain", STAGE_SPECS["strain"], Experiment())
    assert view._last_style is None  # precondition: nothing captured yet

    view._finish_ok(SimpleNamespace(layers=[], skipped=[]))

    assert view._results.toPlainText() == "no strain layers produced"


def test_the_publication_style_dialog_states_the_timing_rule(monkeypatch):
    """The other half of the signal: say the rule where the style is edited."""
    opened: list[QDialog] = []
    monkeypatch.setattr(QDialog, "exec", lambda self: (opened.append(self), 0)[1])
    monkeypatch.setattr(MainWindow, "_save_plot_style", lambda self: None)

    win = MainWindow()
    win._on_pub_style()

    assert opened, "precondition: _on_pub_style must open a dialog"
    notes = [lbl.text() for lbl in opened[0].findChildren(QLabel) if lbl.property("role") == "hint"]
    assert notes, "the Publication style dialog carries no hint-role note"
    text = " ".join(notes)
    assert "launched with" in text  # the timing rule
    assert "Replot" in text  # where to go instead
