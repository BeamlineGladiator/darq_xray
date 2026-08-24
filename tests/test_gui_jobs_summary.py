"""Summary editor for a JSON-list TEXT param (offscreen Qt)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

from dfxm.config.models import Param, ParamType  # noqa: E402
from gui.widgets.jobs_summary import JobsSummaryEditor, summarize_jobs  # noqa: E402
from gui.widgets.param_form import ParamForm  # noqa: E402

_TWO_JOBS = """
[{"name": "oblique_full", "offset_um": 0.0},
 {"name": "ridge", "offset_um": 12.5}]
"""


def test_a_summary_names_each_job_and_its_offset():
    text = summarize_jobs(_TWO_JOBS)
    assert "2 jobs" in text
    assert "oblique_full" in text
    assert "ridge" in text
    assert "12.5" in text


def test_one_job_is_singular():
    assert summarize_jobs('[{"name": "only", "offset_um": 0.0}]').startswith("1 job")


def test_an_empty_list_says_so():
    assert "no jobs" in summarize_jobs("[]").lower()


def test_a_list_without_names_falls_back_to_a_count():
    # The widget must not assume the DFXM job schema — it is a generic
    # JSON-list editor that happens to be used by profiles.
    assert summarize_jobs("[1, 2, 3]") == "3 entries"


def test_malformed_json_reports_rather_than_raising():
    assert "unreadable" in summarize_jobs("{not json").lower()


def test_the_editor_round_trips_the_raw_string_unchanged():
    raw = _TWO_JOBS.strip()
    editor = JobsSummaryEditor(raw, "Jobs (JSON)")
    assert editor.text() == raw


def test_set_text_refreshes_the_summary():
    editor = JobsSummaryEditor("[]", "Jobs (JSON)")
    assert "no jobs" in editor._summary.text().lower()  # precondition
    editor.setText(_TWO_JOBS)
    assert "oblique_full" in editor._summary.text()


def test_set_text_emits_text_changed():
    editor = JobsSummaryEditor("[]", "Jobs (JSON)")
    seen = []
    editor.textChanged.connect(seen.append)
    editor.setText("[1]")
    assert seen


def test_a_form_uses_the_summary_editor_only_when_the_hint_asks_for_it():
    hinted = Param("jobs_json", ParamType.TEXT, "Jobs", default="[]", editor="summary_json")
    plain = Param("notes", ParamType.TEXT, "Notes", default="")
    form = ParamForm((hinted, plain))
    assert isinstance(form._editors["jobs_json"], JobsSummaryEditor)
    assert not isinstance(form._editors["notes"], JobsSummaryEditor)


def test_the_form_reads_and_writes_the_raw_string_through_the_summary_editor():
    # This is the contract that keeps the two picker call sites in stage_view
    # (_on_pick_line, _on_jobs_from_marks) working untouched.
    p = Param("jobs_json", ParamType.TEXT, "Jobs", default="[]", editor="summary_json")
    form = ParamForm((p,))
    form.set_values({"jobs_json": _TWO_JOBS})
    assert form.values()["jobs_json"] == _TWO_JOBS


def test_an_unknown_editor_hint_falls_back_to_the_normal_editor():
    p = Param("notes", ParamType.TEXT, "Notes", default="", editor="nonesuch")
    form = ParamForm((p,))
    assert not isinstance(form._editors["notes"], JobsSummaryEditor)


def _stub_exec(monkeypatch, code, typed):
    """Make the next ``_on_edit`` dialog type *typed* and close with *code*.

    Returns the list of texts the dialog was *seeded* with, one per open. Each
    test asserts on it: it proves the dialog really ran (without which "Cancel
    wrote nothing" would also pass for an ``_on_edit`` that never writes at
    all) and that it opened on the raw JSON rather than the summary.
    """
    from PySide6.QtWidgets import QDialog, QPlainTextEdit

    seeded = []

    def _exec(dlg):
        box = dlg.findChild(QPlainTextEdit)
        seeded.append(box.toPlainText())
        box.setPlainText(typed)
        return code

    monkeypatch.setattr(QDialog, "exec", _exec)
    return seeded


def test_the_raw_dialog_writes_the_typed_json_on_ok(monkeypatch):
    from PySide6.QtWidgets import QDialog

    editor = JobsSummaryEditor("[]", "Jobs (JSON)")
    seeded = _stub_exec(monkeypatch, QDialog.DialogCode.Accepted, _TWO_JOBS)
    editor._on_edit()
    assert seeded == ["[]"]  # precondition: one dialog, opened on the raw JSON
    assert editor.text() == _TWO_JOBS


def test_cancelling_the_raw_dialog_leaves_the_field_untouched(monkeypatch):
    # A Cancel must not overwrite what "Pick line…" just wrote into the field.
    from PySide6.QtWidgets import QDialog

    editor = JobsSummaryEditor(_TWO_JOBS, "Jobs (JSON)")
    seeded = _stub_exec(monkeypatch, QDialog.DialogCode.Rejected, "[]")
    editor._on_edit()
    assert seeded == [_TWO_JOBS]  # precondition: one dialog, opened on the raw JSON
    assert editor.text() == _TWO_JOBS
