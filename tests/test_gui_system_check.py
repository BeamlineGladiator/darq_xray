"""The System check probe table (offscreen Qt)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

from gui.widgets.system_check import SystemCheckDialog  # noqa: E402
from tests.machine_fixtures import tiny_ram, windows_no_vtk, workstation_sw_gl  # noqa: E402


def _labels(dlg):
    return [label for label, _value, _implication in dlg.rows()]


def test_every_probe_gets_a_row():
    dlg = SystemCheckDialog(profile=workstation_sw_gl())
    for expected in ("CPU", "RAM", "Headroom", "Disk", "OpenGL", "ffmpeg"):
        assert expected in _labels(dlg)


def test_software_gl_row_explains_the_consequence():
    prof = workstation_sw_gl()
    assert prof.gl.software is True  # precondition
    dlg = SystemCheckDialog(profile=prof)
    gl_row = next(r for r in dlg.rows() if r[0] == "OpenGL")
    assert "2048" in gl_row[1]
    assert "surface" in gl_row[2].lower()


def test_a_failed_probe_reads_as_unknown_with_its_reason():
    prof = tiny_ram()
    assert prof.gl_status == "crashed" and prof.probe_errors  # precondition
    dlg = SystemCheckDialog(profile=prof)
    gl_row = next(r for r in dlg.rows() if r[0] == "OpenGL")
    assert "unknown" in gl_row[1].lower() or "crashed" in gl_row[1].lower()
    assert "child exited" in dlg.as_text()


def test_missing_ffmpeg_says_what_is_lost():
    prof = windows_no_vtk()
    assert prof.ffmpeg is None  # precondition
    dlg = SystemCheckDialog(profile=prof)
    row = next(r for r in dlg.rows() if r[0] == "ffmpeg")
    assert "GIF" in row[2]


def test_as_text_is_copyable_plain_text():
    dlg = SystemCheckDialog(profile=workstation_sw_gl())
    text = dlg.as_text()
    assert "CPU" in text and "<" not in text
