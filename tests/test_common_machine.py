"""Hardware profile probing (dfxm/common/machine.py, dfxm/common/_glprobe.py)."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

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


def test_probe_gl_reads_the_child_and_classifies_software(monkeypatch, tmp_path):
    """A well-formed child answer becomes a GLInfo with `software` classified."""
    monkeypatch.setattr(machine, "gl_cache_path", lambda: str(tmp_path / "gl.json"))
    payload = json.dumps(
        {
            "status": "ok",
            "renderer": "llvmpipe (LLVM 20.1.2, 256 bits)",
            "vendor": "Mesa",
            "version": "4.5 (Core Profile)",
            "max_3d_texture": 2048,
            "error": None,
        }
    )

    class _Done:
        returncode = 0
        stdout = payload + "\n"
        stderr = ""

    monkeypatch.setattr(machine.subprocess, "run", lambda *a, **k: _Done())
    info, status = machine.probe_gl(use_cache=False)
    assert status == "ok"
    assert info.software is True
    assert info.max_3d_texture == 2048


def test_probe_gl_treats_a_segfaulting_child_as_crashed(monkeypatch, tmp_path):
    """A driver that kills the child is a normal answer, not an exception.

    This is the whole reason the probe is out of process.
    """
    monkeypatch.setattr(machine, "gl_cache_path", lambda: str(tmp_path / "gl.json"))

    class _Killed:
        returncode = -11  # SIGSEGV
        stdout = ""
        stderr = ""

    monkeypatch.setattr(machine.subprocess, "run", lambda *a, **k: _Killed())
    info, status = machine.probe_gl(use_cache=False)
    assert info is None
    assert status == "crashed"


def test_probe_gl_treats_a_hanging_child_as_crashed(monkeypatch, tmp_path):
    monkeypatch.setattr(machine, "gl_cache_path", lambda: str(tmp_path / "gl.json"))

    def _hang(*a, **k):
        raise subprocess.TimeoutExpired(cmd="glprobe", timeout=1.0)

    monkeypatch.setattr(machine.subprocess, "run", _hang)
    info, status = machine.probe_gl(use_cache=False)
    assert info is None
    assert status == "crashed"


def test_probe_gl_handles_garbage_output(monkeypatch, tmp_path):
    monkeypatch.setattr(machine, "gl_cache_path", lambda: str(tmp_path / "gl.json"))

    class _Garbage:
        returncode = 0
        stdout = "not json at all\n"
        stderr = ""

    monkeypatch.setattr(machine.subprocess, "run", lambda *a, **k: _Garbage())
    info, status = machine.probe_gl(use_cache=False)
    assert info is None
    assert status == "crashed"


def test_probe_gl_caches_to_disk_and_reuses(monkeypatch, tmp_path):
    """Second call must not spawn a child — probing costs a process + a context."""
    cache = tmp_path / "gl.json"
    monkeypatch.setattr(machine, "gl_cache_path", lambda: str(cache))
    calls = []
    payload = json.dumps(
        {
            "status": "ok",
            "renderer": "NVIDIA GeForce RTX 3080/PCIe/SSE2",
            "vendor": "NVIDIA",
            "version": "4.6",
            "max_3d_texture": 16384,
            "error": None,
        }
    )

    class _Done:
        returncode = 0
        stdout = payload + "\n"
        stderr = ""

    def _run(*a, **k):
        calls.append(1)
        return _Done()

    monkeypatch.setattr(machine.subprocess, "run", _run)
    machine._GL_MEMO.clear()
    try:
        first, _ = machine.probe_gl(use_cache=True)
        machine._GL_MEMO.clear()  # drop the in-process memo; disk cache must carry it
        second, _ = machine.probe_gl(use_cache=True)
        assert len(calls) == 1
        assert first == second
        assert first.software is False
    finally:
        machine._GL_MEMO.clear()


def test_probe_gl_memoises_a_crashed_result_in_process(monkeypatch, tmp_path):
    """A crashing driver must not be re-spawned on every subsequent probe call.

    A crashed probe IS memoised (deliberately — hammering a crashing driver on
    every call would be worse than a stale "crashed" answer), unlike a
    successful probe it is NOT written to the disk cache, so this in-process
    memo is the only thing preventing a second child spawn here.
    """
    monkeypatch.setattr(machine, "gl_cache_path", lambda: str(tmp_path / "gl.json"))
    calls = []

    class _Killed:
        returncode = -11  # SIGSEGV
        stdout = ""
        stderr = ""

    def _run(*a, **k):
        calls.append(1)
        return _Killed()

    monkeypatch.setattr(machine.subprocess, "run", _run)
    machine._GL_MEMO.clear()
    try:
        first, first_status = machine.probe_gl(use_cache=True)
        second, second_status = machine.probe_gl(use_cache=True)
        assert len(calls) == 1, "second probe_gl(use_cache=True) spawned a second child"
        assert first is None and second is None
        assert first_status == second_status == "crashed"
    finally:
        machine._GL_MEMO.clear()


def test_profile_does_not_probe_gl_unless_asked(monkeypatch, tmp_path):
    """Default profile() must stay instant — no child process at startup."""
    called = []
    monkeypatch.setattr(machine, "probe_gl", lambda **k: called.append(1) or (None, "ok"))
    machine.profile(output_dir=str(tmp_path))
    assert called == []


# -- the psutil-free fallback must agree with psutil ---------------------------
# The bug this pins: `_memory_stdlib` read `SC_AVPHYS_PAGES`, which is `MemFree`
# — it excludes the reclaimable page cache, so on a machine with a large cache
# it understates obtainable memory by orders of magnitude (measured here:
# 8.4 GB reported against 467.7 GB actually available). Nothing crashed; the
# status bar simply lied, and `advice.headroom_bytes` sized every run off the
# lie. The fallback is not a rare path — it is what runs in any environment
# without psutil, e.g. the darfix venv the GUI is often launched from.


def _stdlib_only_probe(monkeypatch):
    """`probe_memory()` with psutil forced out of reach."""
    import builtins

    real_import = builtins.__import__

    def no_psutil(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("simulated: psutil not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_psutil)
    return machine.probe_memory()


def test_stdlib_fallback_agrees_with_psutil(monkeypatch):
    """Same machine, same instant: dropping psutil must not change the answer."""
    psutil = pytest.importorskip("psutil")
    expected = psutil.virtual_memory().available
    _, available = _stdlib_only_probe(monkeypatch)
    # Generous: the two are sampled microseconds apart and psutil's Linux
    # `available` is /proc/meminfo's MemAvailable, the same number we read.
    assert available == pytest.approx(expected, rel=0.05), (
        f"stdlib fallback says {available} but psutil says {expected} — "
        "the fallback is reading MemFree, not MemAvailable"
    )


def test_parse_meminfo_reads_memavailable_not_memfree():
    """The parse is the whole fix: MemAvailable, in bytes, ignoring MemFree."""
    text = (
        "MemTotal:       526816972 kB\n"
        "MemFree:          8933492 kB\n"
        "MemAvailable:   490585992 kB\n"
        "Buffers:          8264484 kB\n"
    )
    total, available = machine._parse_meminfo(text)
    assert total == 526816972 * 1024
    assert available == 490585992 * 1024


def test_parse_meminfo_declines_when_memavailable_is_absent():
    """Pre-3.14 kernels have no MemAvailable; say so rather than invent one."""
    text = "MemTotal:       526816972 kB\nMemFree:          8933492 kB\n"
    assert machine._parse_meminfo(text) == (0, 0)


def test_memory_stdlib_still_answers_when_meminfo_is_unreadable(monkeypatch):
    """No /proc/meminfo (a non-Linux POSIX, a locked-down container) still probes."""
    monkeypatch.setattr(machine, "_read_meminfo", lambda: "")
    total, available = machine._memory_stdlib()
    assert total > 0
    assert 0 < available <= total
