"""The ambient machine readout in the status bar (offscreen Qt)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

from gui import advisor as A  # noqa: E402
from gui.main_window import MainWindow  # noqa: E402
from tests.machine_fixtures import tiny_ram, windows_no_vtk, workstation_sw_gl  # noqa: E402


def test_status_bar_names_cores_disk_and_ram(monkeypatch):
    monkeypatch.setattr(A, "cached_profile", lambda d: workstation_sw_gl())
    win = MainWindow()
    win._refresh_machine_status()
    text = win._machine_label.text()
    assert "36 cores" in text
    assert "RAM" in text
    assert "free" in text


def test_software_gl_is_called_out(monkeypatch):
    monkeypatch.setattr(A, "cached_profile", lambda d: workstation_sw_gl())
    win = MainWindow()
    win._refresh_machine_status()
    # Precondition: this fixture really is a software renderer.
    assert workstation_sw_gl().gl.software is True
    assert "software GL" in win._machine_label.text()


def test_unmeasured_fields_are_omitted_not_shown_as_zero(monkeypatch):
    monkeypatch.setattr(A, "cached_profile", lambda d: windows_no_vtk())
    win = MainWindow()
    win._refresh_machine_status()
    text = win._machine_label.text()
    assert "GL" not in text  # gl is None on this fixture
    assert "0.0 B" not in text


def test_the_readout_never_probes_gl(monkeypatch):
    monkeypatch.setattr(A, "cached_profile", lambda d: tiny_ram())
    monkeypatch.setattr(
        A.machine,
        "probe_gl",
        lambda **kw: pytest.fail("the status bar must never probe GL"),
    )
    win = MainWindow()
    win._refresh_machine_status()
    assert win._machine_label.text()
