"""The publication style a finished run actually used (offscreen Qt)."""

import os
from dataclasses import asdict, fields
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


def test_the_stamp_scopes_its_claim_to_what_it_actually_names():
    """The stamp names five of PlotStyle's fields — far from all of them.

    Two runs differing only in axes_mode / title_scale / the scale bar / the
    µm-per-cm scale stamp identically, so the headline must not read as "this
    is the whole style". The count is computed, never written down, so this
    cannot go stale when PlotStyle grows a field.
    """
    named = ("cmap_mosa_com", "cmap_mosa_fwhm", "cmap_strain", "cmap_raw", "font_scale")
    unnamed = [f.name for f in fields(PlotStyle) if f.name not in named]
    assert len(unnamed) > len(named)  # precondition: most of the style is unnamed
    # precondition: a field the stamp does NOT name really can differ silently
    assert style_stamp(PlotStyle(axes_mode="none")) == style_stamp(PlotStyle())
    assert "colormaps" in style_stamp(PlotStyle()).lower()


def test_the_stamp_is_a_pure_function_of_the_style_it_is_given():
    """Nothing is hard-coded: a different style in, a different line out.

    (Renamed from test_the_stamp_reflects_the_captured_style_not_the_current_one
    — `style_stamp` has no access to a "current" style, so its body could never
    have told captured from current. That distinction is pinned by
    test_finished_run_records_the_style_it_was_launched_with_not_the_current_one
    below, which is what M6/M7 actually break.)
    """
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


def test_the_capture_survives_the_style_being_edited_mid_run(monkeypatch):
    """`_on_run` must SNAPSHOT the session style, not alias it.

    `MainWindow.global_plot_style()` returns its own `_plot_style` object, and
    "Publication style…" (`StyleControls`) mutates that object **in place** while
    a run is in flight — the button is never disabled. An aliasing capture would
    let such an edit rewrite the stamp of a run that had already rendered from
    the `asdict()` taken at launch, i.e. the stamp would name a style the run
    never used. Unlike the test above this drives the real `_on_run`, so it is
    the only thing that can see the difference.
    """
    view = StageView("strain", STAGE_SPECS["strain"], Experiment())
    live = PlotStyle(cmap_strain="turbo")  # the session's own mutable object
    view.global_plot_style = lambda: live
    started: list[dict] = []
    monkeypatch.setattr(view, "_start_runner", lambda params: started.append(params))
    # never let the pre-flight open a modal (it would hang the suite, not fail it)
    monkeypatch.setattr(view, "_confirm_blocked", lambda reason: True)

    view._on_run()
    assert started, "precondition: the run must actually have launched"
    assert started[0]["plot_style"]["cmap_strain"] == "turbo"  # what the child renders with
    # the child's params and the stamped snapshot must describe the SAME style —
    # a stamp that disagreed with what was rendered would be worse than none
    assert started[0]["plot_style"] == asdict(view._last_style)

    live.cmap_strain = "seismic"  # exactly what StyleControls does, mid-run
    assert view.global_plot_style().cmap_strain == "seismic"  # precondition: it took
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
