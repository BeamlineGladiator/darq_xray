"""Hardware profile probing (dfxm/common/machine.py, dfxm/common/_glprobe.py)."""

from __future__ import annotations

import json
import subprocess
import sys

from dfxm.common import machine


def test_glprobe_child_prints_one_json_line_and_exits_zero():
    """The probe child must ALWAYS exit 0 with parseable JSON — never a traceback.

    It runs on machines with no GPU, no driver and no vtk; the parent
    distinguishes outcomes by the 'status' field, not by exit code.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "dfxm.common._glprobe"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    assert len(lines) == 1, f"expected exactly one line, got {lines!r}"
    data = json.loads(lines[0])
    assert data["status"] in ("ok", "no-gl", "no-vtk")
    assert set(data) == {"status", "renderer", "vendor", "version", "max_3d_texture", "error"}
    if data["status"] == "ok":
        assert isinstance(data["renderer"], str) and data["renderer"]
        assert data["max_3d_texture"] is None or data["max_3d_texture"] > 0


def test_is_software_renderer_recognises_the_usual_suspects():
    assert machine.is_software_renderer("llvmpipe (LLVM 20.1.2, 256 bits)")
    assert machine.is_software_renderer("Microsoft Basic Render Driver")
    assert machine.is_software_renderer("GDI Generic")
    assert machine.is_software_renderer("SWRast")  # case-insensitive
    assert not machine.is_software_renderer("NVIDIA GeForce RTX 3080/PCIe/SSE2")
    assert not machine.is_software_renderer("")  # unknown is not "software"


def test_probe_memory_returns_plausible_totals():
    total, available = machine.probe_memory()
    assert total > 0
    assert 0 < available <= total


def test_probe_memory_falls_back_when_psutil_missing(monkeypatch):
    """Removing psutil must degrade the probe, never break it."""
    import builtins

    real_import = builtins.__import__

    def no_psutil(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("simulated: psutil not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_psutil)
    total, available = machine.probe_memory()
    assert total > 0  # stdlib fallback carried it


def test_probe_cpu_returns_at_least_one_logical_core():
    logical, physical = machine.probe_cpu()
    assert logical >= 1
    assert physical is None or physical >= 1


def test_profile_never_raises_and_reports_unprobed_gl(tmp_path):
    prof = machine.profile(output_dir=str(tmp_path))
    assert prof.os_name in ("Linux", "Windows", "Darwin")
    assert prof.ram_total > 0
    assert prof.disk_free > 0
    assert prof.gl is None
    assert prof.gl_status == "unprobed"
    assert isinstance(prof.probe_errors, tuple)


def test_profile_survives_every_probe_failing(monkeypatch, tmp_path):
    """The whole point: an unmeasurable machine is described, not fatal."""

    def boom(*a, **k):
        raise OSError("simulated probe failure")

    monkeypatch.setattr(machine, "probe_cpu", boom)
    monkeypatch.setattr(machine, "probe_memory", boom)
    monkeypatch.setattr(machine, "probe_disk", boom)
    monkeypatch.setattr(machine, "probe_ffmpeg", boom)
    prof = machine.profile(output_dir=str(tmp_path))
    assert prof.ram_total == 0
    assert prof.cpu_logical == 1  # documented floor
    assert len(prof.probe_errors) == 4
