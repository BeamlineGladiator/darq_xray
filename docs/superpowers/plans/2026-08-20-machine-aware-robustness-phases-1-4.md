# Machine-Aware Robustness (Phases 1–4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Qt-free infrastructure that lets the pipeline measure the machine it runs on, predict a stage's peak memory from HDF5 shapes alone, and read volumes in bounded memory — with no user-visible behaviour change yet.

**Architecture:** Four new modules under `dfxm/common/`. `_glprobe.py` is a leaf run as a child process so a segfaulting GL driver cannot take down the app. `machine.py` builds a frozen `MachineProfile`, caching the expensive GL answer to disk. `advice.py` is pure policy: `(profile, estimate) -> RunPlan`. `volumeio.py` provides chunked, two-pass and disk-backed readers whose results are bit-identical regardless of memory budget.

**Tech Stack:** Python 3.10+, numpy, h5py, psutil (new, with stdlib fallbacks), pytest. No Qt, no pyvista except inside `_glprobe`'s child process.

**Spec:** `docs/superpowers/specs/2026-08-20-machine-aware-robustness-design.md`

## Global Constraints

- **`dfxm/` stays Qt-free.** Never import PySide6 or pyvista at module level in any file this plan touches. `_glprobe.py` imports pyvista *inside* `main()` only.
- **Phases 1–4 change no user-visible behaviour.** No stage's output changes; no GUI file is modified. `docs/Usage.md` is therefore NOT updated by this plan. `docs/Codebase.md` IS updated, in the same commit as each module it documents.
- **Every probe is individually wrapped.** Building a `MachineProfile` must never raise. A failed probe appends to `probe_errors` and leaves its field `None`.
- **Budget-independence is the equivalence guarantee.** Any reduction in `volumeio.py` must produce bit-identical results for *any* `budget_bytes`. Tests assert equality across budgets, NOT equality with `np.sum`.
- **Estimators read `.shape` and `.dtype` only.** Never read dataset contents in an estimator.
- Line length 100, ruff `E`/`F`/`I`, double quotes. `ruff format` runs automatically on Write/Edit via the repo hook.
- **The full suite cannot complete on this machine, and that is expected.**
  `python3 -m pytest -q` exits 139 (SIGSEGV) at
  `tests/test_gui_viewer3d.py:38 test_window_builds_scene_from_source` →
  `gui/widgets/viewer3d_window.py:426 rebuild` → `gui/widgets/pv_canvas.py:39 ensure`,
  where `QtInteractor(self)` creates a Qt GL context. This box has software
  OpenGL (Mesa llvmpipe, no GPU) and the pre-existing failure is documented in
  the project's memory. It is unrelated to this plan.
  **So "run the full suite" here means:**
  `python3 -m pytest -q --deselect tests/test_gui_viewer3d.py`
  which deselects 17 tests. **Measured baseline as of Task 1 complete
  (commit 2bc39a7): 1035 passed, 13 skipped, 17 deselected** — verified by
  running it, not derived. (1035 + 13 + 17 = 1065 collected = the original 1064
  plus Task 1's one new test.) Each later task adds its own tests on top of
  1035. Always state the deselection in your report — it is a disclosed
  environment workaround, not suite-narrowing, and you do not need to
  investigate or fix it.
- This repo has **no git remote** — never pull, push, or open a PR.

---

## File Structure

| File | Responsibility |
|---|---|
| Create `dfxm/common/_glprobe.py` | Leaf module, run as `python -m dfxm.common._glprobe`. Builds an off-screen plotter, prints one JSON line describing the GL stack, exits 0. Imports pyvista only inside `main()`. |
| Create `dfxm/common/machine.py` | `GLInfo`, `MachineProfile`, the cheap probes (CPU/RAM/disk/ffmpeg), the GL child-process call with timeout, and the on-disk GL cache. |
| Create `dfxm/common/advice.py` | `RunPlan`, `Advice`, `plan_run`, `advise_3d`. Pure functions, no IO. |
| Create `dfxm/common/volumeio.py` | `volume_bytes`, `iter_blocks`, `load_or_stream`, `BlockReader`, `block_reduce`, `two_pass`, `scratch_array`. |
| Modify `dfxm/config/models.py` | Add `CostEstimate` dataclass and the `StageSpec.estimate` field. |
| Modify `dfxm/stages/{strain,mosaicity,slices,rocking,matched,paraview,visualize}.py` | Add one `estimate(params) -> CostEstimate` function each; wire it into that module's `STAGE` via `estimate="dfxm.stages.X:estimate"`. |
| Create `tests/equivalence.py` | Reusable harness asserting a callable gives bit-identical results across memory budgets. Not a pytest file (no `test_` prefix). |
| Create `tests/test_common_machine.py`, `tests/test_common_advice.py`, `tests/test_common_volumeio.py`, `tests/test_stage_estimates.py` | Unit tests. |
| Create `tests/machine_fixtures.py` | Synthetic `MachineProfile` fixtures: `laptop_hw_gl`, `workstation_sw_gl`, `windows_no_vtk`, `tiny_ram`. |
| Modify `pyproject.toml` | Add `psutil>=5.9` to dependencies. |
| Modify `docs/Codebase.md` | Document each new module, in the same commit. |

---

## Task 1: GL probe child process

**Files:**
- Create: `dfxm/common/_glprobe.py`
- Test: `tests/test_common_machine.py`
- Modify: `docs/Codebase.md`

**Interfaces:**
- Consumes: nothing.
- Produces: a module runnable as `python -m dfxm.common._glprobe` that prints exactly one JSON object on stdout and exits 0. Keys: `status` (`"ok"` | `"no-gl"` | `"no-vtk"`), `renderer` (str), `vendor` (str), `version` (str), `max_3d_texture` (int or null), `error` (str or null).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_common_machine.py
"""Hardware profile probing (dfxm/common/machine.py, dfxm/common/_glprobe.py)."""

from __future__ import annotations

import json
import subprocess
import sys


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_common_machine.py -v`
Expected: FAIL — `No module named dfxm.common._glprobe`.

- [ ] **Step 3: Write the implementation**

```python
# dfxm/common/_glprobe.py
"""Out-of-process OpenGL capability probe. Run as ``python -m dfxm.common._glprobe``.

Creating a GL context is the one operation in this codebase that can take the
whole process down: a broken or missing driver does not raise, it segfaults.
So the probe lives here, in a child process, and the parent
(:mod:`dfxm.common.machine`) treats "child died" as a normal answer meaning
"3-D unusable".

This module MUST stay a leaf: under the ``spawn`` start method a child
re-imports its module, and importing anything that reaches the GUI would spawn
windows recursively. It imports ``pyvista`` inside :func:`probe` only.

This module reports the raw renderer string and does NOT classify it —
``machine.is_software_renderer`` is the single owner of that rule, so the
classification lives in one place rather than being duplicated into the child.

Contract: print exactly one JSON object on stdout, exit 0, always.
"""

from __future__ import annotations

import json
import sys


def _parse_capabilities(caps: str) -> dict:
    """Pull vendor/renderer/version out of vtkRenderWindow.ReportCapabilities().

    The report is many lines of ``OpenGL <field> string:  <value>`` (note the
    doubled space); anything we cannot find comes back as "".
    """
    out = {"vendor": "", "renderer": "", "version": ""}
    for line in caps.splitlines():
        for field in out:
            marker = f"OpenGL {field} string:"
            if marker in line:
                out[field] = line.split(":", 1)[1].strip()
    return out


def probe() -> dict:
    """Build the result dict. Never raises."""
    result = {
        "status": "no-gl",
        "renderer": "",
        "vendor": "",
        "version": "",
        "max_3d_texture": None,
        "error": None,
    }
    try:
        import pyvista as pv
    except Exception as exc:  # noqa: BLE001 - no pyvista/vtk installed
        result["status"] = "no-vtk"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    plotter = None
    prev_off_screen = pv.OFF_SCREEN
    try:
        pv.OFF_SCREEN = True
        plotter = pv.Plotter(off_screen=True, window_size=[16, 16])
        plotter.show(auto_close=False)  # the context must exist before querying
        window = plotter.render_window
        result.update(_parse_capabilities(window.ReportCapabilities()))
        try:
            from vtkmodules.vtkRenderingOpenGL2 import vtkTextureObject

            limit = int(vtkTextureObject.GetMaximumTextureSize3D(window))
            result["max_3d_texture"] = limit if limit > 0 else None
        except Exception:  # noqa: BLE001 - old vtk: unknown limit, not a failure
            result["max_3d_texture"] = None
        result["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - any GL/driver failure
        result["status"] = "no-gl"
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        pv.OFF_SCREEN = prev_off_screen
        if plotter is not None:
            try:
                plotter.close()
            except Exception:  # noqa: BLE001 - closing a broken plotter
                pass
    return result


def main() -> int:
    sys.stdout.write(json.dumps(probe()) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_common_machine.py -v`
Expected: PASS. On this development box `status` will be `"ok"` with renderer `llvmpipe (...)` and `max_3d_texture` 2048.

- [ ] **Step 5: Document in Codebase.md**

Add an entry for `_glprobe.py` to the `dfxm/common/` section of
`docs/Codebase.md`. **Read the neighbouring entries first and match their
format exactly** — do not paste the wording below as-is. In that section each
entry opens with a `#### \`file.py\`` heading followed directly by prose; it
does NOT repeat the filename as a bold bullet under a heading that already
names it. Content to convey:

> Out-of-process OpenGL capability probe, run as
> `python -m dfxm.common._glprobe`. Prints one JSON object
> (`status`, `renderer`, `vendor`, `version`, `max_3d_texture`, `error`) and
> always exits 0, so callers distinguish outcomes by `status` rather than exit
> code. Kept a leaf module with pyvista imported inside `probe()`, because a
> broken GL driver segfaults rather than raising and because `spawn` re-imports
> the child's module. Reports the renderer string only; classifying it as
> software rendering is `machine.is_software_renderer`'s job.

- [ ] **Step 6: Commit**

```bash
git add dfxm/common/_glprobe.py tests/test_common_machine.py docs/Codebase.md
git commit -m "feat: out-of-process GL capability probe"
```

---

## Task 2: MachineProfile and the cheap probes

**Files:**
- Create: `dfxm/common/machine.py`
- Modify: `pyproject.toml`, `docs/Codebase.md`
- Test: `tests/test_common_machine.py`

**Interfaces:**
- Consumes: nothing from Task 1 yet (GL is wired in Task 3).
- Produces:
  - `GLInfo(renderer: str, vendor: str, version: str, max_3d_texture: int | None, software: bool)` — frozen dataclass.
  - `MachineProfile(os_name, cpu_logical, cpu_physical, ram_total, ram_available, disk_free, gl, gl_status, ffmpeg, probe_errors)` — frozen dataclass, field order as listed.
  - `probe_cpu() -> tuple[int, int | None]`
  - `probe_memory() -> tuple[int, int]` — `(total, available)` bytes; `(0, 0)` when unknown.
  - `probe_disk(path: str) -> int`
  - `probe_ffmpeg() -> str | None`
  - `is_software_renderer(renderer: str) -> bool`
  - `profile(*, output_dir: str | None = None) -> MachineProfile` — GL fields are `None` / `"unprobed"` until Task 3.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_common_machine.py`:

```python
import pytest

from dfxm.common import machine


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
    boom = lambda *a, **k: (_ for _ in ()).throw(OSError("simulated probe failure"))
    monkeypatch.setattr(machine, "probe_cpu", boom)
    monkeypatch.setattr(machine, "probe_memory", boom)
    monkeypatch.setattr(machine, "probe_disk", boom)
    monkeypatch.setattr(machine, "probe_ffmpeg", boom)
    prof = machine.profile(output_dir=str(tmp_path))
    assert prof.ram_total == 0
    assert prof.cpu_logical == 1  # documented floor
    assert len(prof.probe_errors) == 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_common_machine.py -v`
Expected: FAIL — `cannot import name 'machine'`.

- [ ] **Step 3: Write the implementation**

```python
# dfxm/common/machine.py
"""What kind of machine are we on? (Qt-free.)

Builds a frozen :class:`MachineProfile` describing CPU, RAM, disk, the OpenGL
stack and ffmpeg. Consumers (:mod:`dfxm.common.advice`, the GUI's System check)
read it to decide how much work fits in memory and whether 3-D is usable.

**Every probe is individually wrapped and :func:`profile` never raises.** A
machine we cannot measure is one we describe as unmeasured — never a crash,
because the machines hardest to measure are exactly the ones this project
exists to keep running.

``psutil`` is used when importable and falls back to stdlib
(``os.sysconf`` on POSIX, ``GlobalMemoryStatusEx`` via ctypes on Windows), so a
missing optional dependency degrades one field rather than breaking launch.
"""

from __future__ import annotations

import os
import platform
import shutil
from dataclasses import dataclass, field

_SOFTWARE_MARKERS = (
    "llvmpipe",
    "swrast",
    "softpipe",
    "software rasterizer",
    "microsoft basic render",
    "gdi generic",
)


@dataclass(frozen=True)
class GLInfo:
    """What the OpenGL stack is and what it can do."""

    renderer: str
    vendor: str
    version: str
    max_3d_texture: int | None
    software: bool


@dataclass(frozen=True)
class MachineProfile:
    """A measured description of the current machine. Never partially valid:
    unmeasured fields are None/0 and the reason is in :attr:`probe_errors`."""

    os_name: str
    cpu_logical: int
    cpu_physical: int | None
    ram_total: int  # bytes
    ram_available: int  # bytes
    disk_free: int  # bytes, measured against the output directory
    gl: GLInfo | None
    gl_status: str  # "ok" | "no-gl" | "crashed" | "no-vtk" | "unprobed"
    ffmpeg: str | None
    probe_errors: tuple[str, ...] = field(default_factory=tuple)


def is_software_renderer(renderer: str) -> bool:
    """True when *renderer* names a known CPU rasteriser (no GPU acceleration)."""
    low = renderer.lower()
    return any(marker in low for marker in _SOFTWARE_MARKERS)


def probe_cpu() -> tuple[int, int | None]:
    """(logical, physical) core counts; physical is None when unknowable."""
    logical = os.cpu_count() or 1
    physical: int | None = None
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
    except Exception:  # noqa: BLE001 - no psutil -> physical stays unknown
        physical = None
    return logical, physical


def _memory_stdlib() -> tuple[int, int]:
    """(total, available) bytes without psutil. Returns (0, 0) when unknown."""
    if os.name == "nt":
        import ctypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = _MemoryStatusEx()
        stat.dwLength = ctypes.sizeof(_MemoryStatusEx)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return int(stat.ullTotalPhys), int(stat.ullAvailPhys)
    page = os.sysconf("SC_PAGE_SIZE")
    total = page * os.sysconf("SC_PHYS_PAGES")
    # SC_AVPHYS_PAGES excludes reclaimable page cache, so it badly understates
    # what is actually obtainable; it is the only stdlib number available.
    available = page * os.sysconf("SC_AVPHYS_PAGES")
    return int(total), int(available)


def probe_memory() -> tuple[int, int]:
    """(total, available) bytes. psutil when present, else stdlib."""
    try:
        import psutil

        vm = psutil.virtual_memory()
        return int(vm.total), int(vm.available)
    except Exception:  # noqa: BLE001 - no psutil -> stdlib fallback
        return _memory_stdlib()


def probe_disk(path: str) -> int:
    """Free bytes on the filesystem holding *path* (walks up to an existing dir)."""
    probe_at = os.path.abspath(path or ".")
    while probe_at and not os.path.isdir(probe_at):
        parent = os.path.dirname(probe_at)
        if parent == probe_at:
            break
        probe_at = parent
    return int(shutil.disk_usage(probe_at).free)


def probe_ffmpeg() -> str | None:
    """Resolved ffmpeg path, or None. Probed once here rather than per call site."""
    return shutil.which("ffmpeg")


def profile(*, output_dir: str | None = None) -> MachineProfile:
    """Measure this machine. Never raises; failures land in ``probe_errors``."""
    errors: list[str] = []

    def _try(label, fn, fallback):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - probes must never be fatal
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
            return fallback

    cpu_logical, cpu_physical = _try("cpu", probe_cpu, (1, None))
    ram_total, ram_available = _try("memory", probe_memory, (0, 0))
    disk_free = _try("disk", lambda: probe_disk(output_dir or os.getcwd()), 0)
    ffmpeg = _try("ffmpeg", probe_ffmpeg, None)

    return MachineProfile(
        os_name=platform.system() or "unknown",
        cpu_logical=cpu_logical,
        cpu_physical=cpu_physical,
        ram_total=ram_total,
        ram_available=ram_available,
        disk_free=disk_free,
        gl=None,
        gl_status="unprobed",
        ffmpeg=ffmpeg,
        probe_errors=tuple(errors),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_common_machine.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Declare the psutil dependency**

In `pyproject.toml`, add to the `dependencies` list after `"pyyaml>=6.0",`:

```toml
    "psutil>=5.9",
```

Also add a note to `README.md`'s `pip install --user ...` line so the run-in-place workflow stays accurate: append ` psutil` to the package list.

- [ ] **Step 6: Document in Codebase.md**

```markdown
- **`machine.py`** — `MachineProfile` / `GLInfo` plus the probes behind them
  (`probe_cpu`, `probe_memory`, `probe_disk`, `probe_ffmpeg`,
  `is_software_renderer`, `profile`). Every probe is individually wrapped:
  `profile()` never raises, and unmeasured fields report `None`/`0` with the
  reason in `probe_errors`. `psutil` is used when importable, with `os.sysconf`
  / `GlobalMemoryStatusEx` fallbacks.
```

- [ ] **Step 7: Commit**

```bash
git add dfxm/common/machine.py tests/test_common_machine.py pyproject.toml README.md docs/Codebase.md
git commit -m "feat: MachineProfile with crash-proof CPU/RAM/disk/ffmpeg probes"
```

---

## Task 3: Wire the GL probe into the profile, with caching

**Files:**
- Modify: `dfxm/common/machine.py`, `docs/Codebase.md`
- Test: `tests/test_common_machine.py`

**Interfaces:**
- Consumes: `dfxm/common/_glprobe.py` (Task 1); `MachineProfile`, `GLInfo`, `is_software_renderer` (Task 2).
- Produces:
  - `gl_cache_path() -> str`
  - `probe_gl(*, timeout: float = 120.0, use_cache: bool = True) -> tuple[GLInfo | None, str]` — returns `(info, status)`.
  - `profile(*, output_dir=None, probe_gl_now: bool = False, gl_timeout: float = 120.0)` — GL is probed only when `probe_gl_now=True`, so the default stays instant.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_common_machine.py`:

```python
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
    first, _ = machine.probe_gl(use_cache=True)
    machine._GL_MEMO.clear()  # drop the in-process memo; disk cache must carry it
    second, _ = machine.probe_gl(use_cache=True)
    assert len(calls) == 1
    assert first == second
    assert first.software is False


def test_profile_does_not_probe_gl_unless_asked(monkeypatch, tmp_path):
    """Default profile() must stay instant — no child process at startup."""
    called = []
    monkeypatch.setattr(machine, "probe_gl", lambda **k: called.append(1) or (None, "ok"))
    machine.profile(output_dir=str(tmp_path))
    assert called == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_common_machine.py -v`
Expected: FAIL — `module 'dfxm.common.machine' has no attribute 'probe_gl'`.

- [ ] **Step 3: Write the implementation**

Add these imports at the top of `dfxm/common/machine.py` (keep them sorted; ruff `I` enforces isort):

```python
import json
import subprocess
import sys
```

Then append to `dfxm/common/machine.py`:

```python
# The GL answer costs a process spawn plus a context creation, and cannot change
# while the app runs — memoised in-process and cached on disk between runs.
_GL_MEMO: dict = {}
_GL_CACHE_VERSION = 1


def _cache_dir() -> str:
    """Per-user cache directory, without adding a dependency on platformdirs."""
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        return os.path.join(base, "dfxm", "cache")
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "dfxm")


def gl_cache_path() -> str:
    return os.path.join(_cache_dir(), "gl_probe.json")


def _cache_key() -> str:
    """Identity of the GL stack: re-probe when any of these change."""
    try:
        # Import the version symbol ONLY — `vtkmodules.all` would pull in the
        # entire VTK surface (slow, and it defeats the lazy-import discipline).
        from vtkmodules.vtkCommonCore import vtkVersion

        vtk_version = vtkVersion.GetVTKVersion()
    except Exception:  # noqa: BLE001 - no vtk -> still a valid key
        vtk_version = "none"
    return "|".join(
        [
            str(_GL_CACHE_VERSION),
            platform.system(),
            platform.node(),
            platform.python_version(),
            vtk_version,
        ]
    )


def _read_gl_cache(key: str) -> dict | None:
    try:
        with open(gl_cache_path(), encoding="utf-8") as fh:
            blob = json.load(fh)
    except Exception:  # noqa: BLE001 - absent/corrupt cache -> re-probe
        return None
    return blob.get("result") if blob.get("key") == key else None


def _write_gl_cache(key: str, result: dict) -> None:
    try:
        os.makedirs(_cache_dir(), exist_ok=True)
        with open(gl_cache_path(), "w", encoding="utf-8") as fh:
            json.dump({"key": key, "result": result}, fh)
    except Exception:  # noqa: BLE001 - an unwritable cache is not an error
        pass


def _run_gl_child(timeout: float) -> dict | None:
    """Run the probe child. None means it died, hung or spoke nonsense."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "dfxm.common._glprobe"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception:  # noqa: BLE001 - TimeoutExpired, OSError, ...
        return None
    if proc.returncode != 0:  # includes negative codes: killed by a signal
        return None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except ValueError:
            return None
    return None


def probe_gl(*, timeout: float = 120.0, use_cache: bool = True) -> tuple[GLInfo | None, str]:
    """Ask the child what the GL stack can do. Returns ``(info, status)``.

    ``status`` is "ok", "no-gl", "no-vtk" or "crashed"; ``info`` is non-None
    only for "ok". A crashed or hanging child is a *result*, not an exception —
    that is the entire reason this runs out of process.
    """
    key = _cache_key()
    if use_cache and "result" in _GL_MEMO:
        raw = _GL_MEMO["result"]
    else:
        raw = _read_gl_cache(key) if use_cache else None
        if raw is None:
            raw = _run_gl_child(timeout)
            if use_cache and raw is not None:
                _write_gl_cache(key, raw)
        if use_cache:
            _GL_MEMO["result"] = raw

    if raw is None:
        return None, "crashed"
    status = raw.get("status", "no-gl")
    if status != "ok":
        return None, status if status in ("no-gl", "no-vtk") else "crashed"
    renderer = str(raw.get("renderer", ""))
    limit = raw.get("max_3d_texture")
    return (
        GLInfo(
            renderer=renderer,
            vendor=str(raw.get("vendor", "")),
            version=str(raw.get("version", "")),
            max_3d_texture=int(limit) if limit else None,
            software=is_software_renderer(renderer),
        ),
        "ok",
    )
```

Now change `profile()`'s signature and GL fields. Replace its `def profile(...)` line and the two GL arguments in the returned `MachineProfile`:

```python
def profile(
    *,
    output_dir: str | None = None,
    probe_gl_now: bool = False,
    gl_timeout: float = 120.0,
) -> MachineProfile:
    """Measure this machine. Never raises; failures land in ``probe_errors``.

    GL is probed only when *probe_gl_now* is set: it costs a child process, so
    callers that just need CPU/RAM (the status bar, a cost estimate) stay
    instant and a broken driver can never delay startup.
    """
```

and inside it, before the `return`:

```python
    gl_info, gl_status = (None, "unprobed")
    if probe_gl_now:
        gl_info, gl_status = _try("gl", lambda: probe_gl(timeout=gl_timeout), (None, "crashed"))
```

with the returned dataclass using `gl=gl_info, gl_status=gl_status`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_common_machine.py -v`
Expected: PASS (13 tests).

- [ ] **Step 5: Verify against the real GL stack on this box**

Run:

```bash
python3 -c "
from dfxm.common import machine
info, status = machine.probe_gl(use_cache=False)
print(status, info)
"
```

Expected on this development machine: `ok GLInfo(renderer='llvmpipe (LLVM 20.1.2, 256 bits)', ..., max_3d_texture=2048, software=True)`. If `software` is not `True` here, `is_software_renderer` is wrong — fix it before continuing.

- [ ] **Step 6: Document in Codebase.md** — extend the `machine.py` entry:

```markdown
  GL is probed out of process (`_glprobe.py`) via `probe_gl()`, memoised and
  cached to `~/.cache/dfxm/gl_probe.json` (`%LOCALAPPDATA%\dfxm\cache` on
  Windows) keyed on OS/host/python/vtk. A child that segfaults, hangs or emits
  garbage yields `gl_status="crashed"` rather than an exception. `profile()`
  skips GL unless `probe_gl_now=True`.
```

- [ ] **Step 7: Commit**

```bash
git add dfxm/common/machine.py tests/test_common_machine.py docs/Codebase.md
git commit -m "feat: crash-tolerant out-of-process GL probing with on-disk cache"
```

---

## Task 4: Synthetic machine fixtures

**Files:**
- Create: `tests/machine_fixtures.py`
- Test: used by Tasks 5–7 (no standalone test file)

**Interfaces:**
- Consumes: `GLInfo`, `MachineProfile` (Task 2).
- Produces: `laptop_hw_gl()`, `workstation_sw_gl()`, `windows_no_vtk()`, `tiny_ram()` — each returns a `MachineProfile`. Also `ALL_PROFILES: tuple[tuple[str, MachineProfile], ...]` for parametrising.

These exist so every policy decision is tested against machines we do not own — the only way the Windows paths get real coverage before a Windows box is available.

- [ ] **Step 1: Write the fixtures module**

```python
# tests/machine_fixtures.py
"""Synthetic MachineProfiles for policy tests (not a pytest file).

Named after real target machines so a failing test says which kind of computer
would break. Deliberately NOT parametrised fixtures: policy tests want to name
the machine they are asserting about.
"""

from __future__ import annotations

from dfxm.common.machine import GLInfo, MachineProfile

GB = 1024**3


def laptop_hw_gl() -> MachineProfile:
    """16 GB laptop with a real GPU — small RAM, generous texture limit."""
    return MachineProfile(
        os_name="Linux",
        cpu_logical=8,
        cpu_physical=4,
        ram_total=16 * GB,
        ram_available=9 * GB,
        disk_free=200 * GB,
        gl=GLInfo("Intel Iris Xe Graphics", "Intel", "4.6", 16384, False),
        gl_status="ok",
        ffmpeg="/usr/bin/ffmpeg",
        probe_errors=(),
    )


def workstation_sw_gl() -> MachineProfile:
    """The development box: huge RAM, no GPU, 2048 px texture ceiling."""
    return MachineProfile(
        os_name="Linux",
        cpu_logical=36,
        cpu_physical=18,
        ram_total=502 * GB,
        ram_available=460 * GB,
        disk_free=2000 * GB,
        gl=GLInfo("llvmpipe (LLVM 20.1.2, 256 bits)", "Mesa", "4.5", 2048, True),
        gl_status="ok",
        ffmpeg="/usr/bin/ffmpeg",
        probe_errors=(),
    )


def windows_no_vtk() -> MachineProfile:
    """A Windows box where vtk failed to import: 3-D off, everything else fine."""
    return MachineProfile(
        os_name="Windows",
        cpu_logical=12,
        cpu_physical=6,
        ram_total=32 * GB,
        ram_available=20 * GB,
        disk_free=500 * GB,
        gl=None,
        gl_status="no-vtk",
        ffmpeg=None,
        probe_errors=(),
    )


def tiny_ram() -> MachineProfile:
    """8 GB with almost nothing free — the machine that must still finish."""
    return MachineProfile(
        os_name="Linux",
        cpu_logical=4,
        cpu_physical=2,
        ram_total=8 * GB,
        ram_available=1 * GB,
        disk_free=40 * GB,
        gl=GLInfo("llvmpipe (LLVM 15.0)", "Mesa", "4.5", 2048, True),
        gl_status="crashed",
        ffmpeg=None,
        probe_errors=("gl: child exited with -11",),
    )


ALL_PROFILES = (
    ("laptop_hw_gl", laptop_hw_gl()),
    ("workstation_sw_gl", workstation_sw_gl()),
    ("windows_no_vtk", windows_no_vtk()),
    ("tiny_ram", tiny_ram()),
)
```

- [ ] **Step 2: Verify the module imports cleanly**

Run: `python3 -c "from tests.machine_fixtures import ALL_PROFILES; print([n for n, _ in ALL_PROFILES])"`
Expected: `['laptop_hw_gl', 'workstation_sw_gl', 'windows_no_vtk', 'tiny_ram']`

- [ ] **Step 3: Commit**

```bash
git add tests/machine_fixtures.py
git commit -m "test: synthetic MachineProfile fixtures for policy tests"
```

---

## Task 5: `CostEstimate` and the `StageSpec.estimate` field

**Files:**
- Modify: `dfxm/config/models.py:96-118` (the `StageSpec` dataclass), `docs/Codebase.md`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `CostEstimate(peak_bytes: int, input_bytes: int, shape: tuple[int, ...] | None, chunkable: bool, note: str | None)` — frozen dataclass in `dfxm/config/models.py`.
  - `StageSpec.estimate: str | None = None` — a `"module:function"` target string.
  - `StageSpec.estimator() -> Callable | None` — resolves the target lazily via `dfxm.stages.registry.resolve`.

A **string**, not a callable, so `dfxm/stages/registry.py` keeps its property that importing the registry never drags in h5py or matplotlib.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py`:

```python
def test_stage_spec_estimate_defaults_to_none():
    from dfxm.config.models import Param, ParamType, StageSpec

    spec = StageSpec(name="x", label="X", description="d", params=(Param("a", ParamType.INT, "A"),))
    assert spec.estimate is None
    assert spec.estimator() is None


def test_stage_spec_estimator_resolves_lazily():
    from dfxm.config.models import StageSpec

    spec = StageSpec(
        name="x",
        label="X",
        description="d",
        params=(),
        estimate="dfxm.common.machine:probe_ffmpeg",
    )
    fn = spec.estimator()
    assert callable(fn)
    assert fn.__name__ == "probe_ffmpeg"


def test_cost_estimate_is_frozen_and_carries_chunkability():
    import dataclasses

    from dfxm.config.models import CostEstimate

    est = CostEstimate(
        peak_bytes=1000, input_bytes=500, shape=(10, 10, 5), chunkable=True, note=None
    )
    assert est.peak_bytes == 1000
    with pytest.raises(dataclasses.FrozenInstanceError):
        est.peak_bytes = 1
```

Ensure `import pytest` is present at the top of `tests/test_config.py`; add it if missing.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_config.py -v -k "estimate or cost_estimate"`
Expected: FAIL — `cannot import name 'CostEstimate'`.

- [ ] **Step 3: Write the implementation**

In `dfxm/config/models.py`, add after the `Param` class:

```python
@dataclass(frozen=True)
class CostEstimate:
    """What a stage run will cost, computed from HDF5 shapes alone.

    Produced by a stage's ``estimate(params)`` function, which opens the input
    and reads ``.shape``/``.dtype`` **only** — never data — so it is cheap
    enough to recompute on every form change.

    ``peak_bytes`` is the in-core high-water mark of the whole-volume strategy,
    including transient copies: a ``[:].astype(np.float64)`` on a float32 source
    holds both arrays at once and costs 3x the on-disk size, not 1x.
    ``chunkable`` is False for work that is irreducibly whole-array and must run
    disk-backed instead.
    """

    peak_bytes: int
    input_bytes: int
    shape: tuple[int, ...] | None
    chunkable: bool
    note: str | None = None
```

Then add the field to `StageSpec` (after `params`) and the resolver method:

```python
    estimate: str | None = None  # "module:function" target, resolved lazily

    def estimator(self):
        """Resolve :attr:`estimate` to a callable, or None when unset.

        Kept as a string on the spec so importing the stage registry never
        drags in h5py/matplotlib; resolution happens only when a caller
        actually wants a prediction.
        """
        if self.estimate is None:
            return None
        from dfxm.stages.registry import resolve

        return resolve(self.estimate)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_config.py -v`
Expected: PASS.

- [ ] **Step 5: Confirm the registry stayed light**

Run:

```bash
python3 -c "
import sys
import dfxm.stages.registry
heavy = [m for m in ('h5py', 'matplotlib', 'pyvista', 'PySide6') if m in sys.modules]
print('heavy modules pulled in:', heavy)
assert not heavy, heavy
print('registry still light')
"
```

Expected: `registry still light`.

- [ ] **Step 6: Document in Codebase.md** — extend the `dfxm/config/models.py` entry with `CostEstimate` and `StageSpec.estimate` / `StageSpec.estimator()`.

- [ ] **Step 7: Commit**

```bash
git add dfxm/config/models.py tests/test_config.py docs/Codebase.md
git commit -m "feat: CostEstimate and lazy StageSpec.estimate target"
```

---

## Task 6: `advice.py` — RunPlan, Advice and the policy

**Files:**
- Create: `dfxm/common/advice.py`
- Create: `tests/test_common_advice.py`
- Modify: `docs/Codebase.md`

**Interfaces:**
- Consumes: `MachineProfile` (Task 2), `CostEstimate` (Task 5), `tests/machine_fixtures.py` (Task 4).
- Produces:
  - `RunPlan(strategy, budget_bytes, chunk_layers, downsample, scratch_dir, reasons, blocked)` — frozen dataclass.
  - `Advice(downsample, render_mode, reasons)` — frozen dataclass.
  - `headroom_bytes(profile) -> int`
  - `plan_run(profile, estimate, *, allow_downsample=False, scratch_dir=None) -> RunPlan`
  - `advise_3d(profile, shape, mode) -> Advice`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_common_advice.py
"""Machine-aware policy (dfxm/common/advice.py). Pure functions, no IO."""

from __future__ import annotations

from dfxm.common.advice import advise_3d, headroom_bytes, plan_run
from dfxm.config.models import CostEstimate
from tests.machine_fixtures import laptop_hw_gl, tiny_ram, windows_no_vtk, workstation_sw_gl

GB = 1024**3


def _est(peak_gb, *, chunkable=True, shape=(100, 700, 2891)):
    return CostEstimate(
        peak_bytes=int(peak_gb * GB),
        input_bytes=int(peak_gb * GB / 3),
        shape=shape,
        chunkable=chunkable,
        note=None,
    )


def test_headroom_is_the_tighter_of_the_two_limits():
    # workstation: 0.6*460 = 276 GB vs 0.5*502 = 251 GB -> total-based wins
    assert headroom_bytes(workstation_sw_gl()) == int(0.5 * 502 * GB)
    # tiny_ram: 0.6*1 = 0.6 GB vs 0.5*8 = 4 GB -> available-based wins
    assert headroom_bytes(tiny_ram()) == int(0.6 * 1 * GB)


def test_small_job_on_a_big_machine_stays_in_core():
    """The fast path must not pay for the slow path's safety."""
    plan = plan_run(workstation_sw_gl(), _est(4))
    assert plan.strategy == "in-core"
    assert plan.chunk_layers == 0
    assert plan.downsample == 1
    assert plan.blocked is None


def test_big_job_on_a_small_machine_chunks():
    plan = plan_run(tiny_ram(), _est(20))
    assert plan.strategy == "chunked"
    assert plan.chunk_layers >= 1
    assert plan.blocked is None
    assert any("chunk" in r.lower() for r in plan.reasons)


def test_unchunkable_big_job_goes_disk_backed_not_blocked(tmp_path):
    """Nothing refuses for lack of RAM — the escalation ends at disk-backed."""
    plan = plan_run(tiny_ram(), _est(20, chunkable=False), scratch_dir=str(tmp_path))
    assert plan.strategy == "disk-backed"
    assert plan.scratch_dir == str(tmp_path)
    assert plan.blocked is None


def test_disk_backed_blocks_only_when_disk_is_short(tmp_path):
    """The one genuine blocker, and it is stated in advance."""
    profile = tiny_ram()
    huge = _est(100, chunkable=False)  # 100 GB needed, 40 GB free
    plan = plan_run(profile, huge, scratch_dir=str(tmp_path))
    assert plan.blocked is not None
    assert "disk" in plan.blocked.lower()


def test_downsample_stays_off_unless_opted_in():
    assert plan_run(tiny_ram(), _est(20)).downsample == 1
    opted = plan_run(tiny_ram(), _est(20), allow_downsample=True)
    assert opted.downsample >= 1


def test_every_plan_explains_itself():
    for profile in (laptop_hw_gl(), workstation_sw_gl(), windows_no_vtk(), tiny_ram()):
        plan = plan_run(profile, _est(20))
        assert plan.reasons, f"{profile.os_name} plan gave no reason"
        assert all(isinstance(r, str) and r for r in plan.reasons)


def test_advise_3d_downsamples_to_fit_the_texture_limit():
    """STO2 is 2891 px wide; llvmpipe caps 3-D textures at 2048."""
    advice = advise_3d(workstation_sw_gl(), (100, 700, 2891), "volume")
    assert advice.downsample >= 2
    assert advice.render_mode in ("surface", "isosurface")
    assert any("2048" in r for r in advice.reasons)


def test_advise_3d_leaves_a_fitting_volume_alone():
    advice = advise_3d(laptop_hw_gl(), (100, 700, 2891), "volume")
    assert advice.downsample == 1
    assert advice.render_mode is None


def test_advise_3d_is_silent_for_geometry_modes():
    """surface/isosurface upload geometry, not one big 3-D texture."""
    advice = advise_3d(workstation_sw_gl(), (100, 700, 2891), "surface")
    assert advice.downsample == 1


def test_advise_3d_without_gl_recommends_nothing():
    advice = advise_3d(windows_no_vtk(), (100, 700, 2891), "volume")
    assert advice.downsample == 1
    assert advice.render_mode is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_common_advice.py -v`
Expected: FAIL — `No module named 'dfxm.common.advice'`.

- [ ] **Step 3: Write the implementation**

```python
# dfxm/common/advice.py
"""Machine-aware policy: what should this run actually do? (Qt-free, no IO.)

Pure functions over a :class:`~dfxm.common.machine.MachineProfile` and a
:class:`~dfxm.config.models.CostEstimate`. Everything here is deterministic and
side-effect-free, so every decision is testable against synthetic machines we
do not own (see ``tests/machine_fixtures.py``).

The governing rule: **nothing refuses to run for lack of RAM.** The escalation
is in-core -> chunked -> disk-backed, and the only genuine blocker is running
out of *disk*, which is measured and reported before work starts.

Every decision carries a plain-language reason. Those strings are what the log,
the GUI banner and the stage result notes all display, so each explanation is
written exactly once — here.
"""

from __future__ import annotations

from dataclasses import dataclass

# Headroom: never plan to use more than this share of memory, leaving room for
# Qt, matplotlib, h5py buffers and the OS. Two limits, the tighter one wins:
# available RAM guards against other processes, total RAM guards against a
# misleadingly large "available" on a machine with a huge page cache.
AVAILABLE_FRACTION = 0.6
TOTAL_FRACTION = 0.5

# Below this there is no point chunking — the bookkeeping costs more than the
# memory it saves.
MIN_BUDGET_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class RunPlan:
    """How a stage should execute on this machine."""

    strategy: str  # "in-core" | "chunked" | "disk-backed"
    budget_bytes: int
    chunk_layers: int  # 0 when in-core
    downsample: int  # 1 unless allow_downsample was opted into
    scratch_dir: str | None  # set only for "disk-backed"
    reasons: tuple[str, ...]
    blocked: str | None  # set only for insufficient disk


@dataclass(frozen=True)
class Advice:
    """Recommended 3-D settings for this machine and volume."""

    downsample: int
    render_mode: str | None  # None = keep whatever the user chose
    reasons: tuple[str, ...]


def _human(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024 or unit == "TB":
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def headroom_bytes(profile) -> int:
    """How much memory a run may plan to use on *profile*."""
    if profile.ram_total <= 0:
        return MIN_BUDGET_BYTES  # unmeasurable machine: assume the worst
    return int(
        min(AVAILABLE_FRACTION * profile.ram_available, TOTAL_FRACTION * profile.ram_total)
    )


def plan_run(profile, estimate, *, allow_downsample: bool = False, scratch_dir=None) -> RunPlan:
    """Decide the execution strategy for *estimate* on *profile*."""
    budget = headroom_bytes(profile)
    reasons: list[str] = []

    if estimate.peak_bytes <= budget:
        reasons.append(
            f"needs {_human(estimate.peak_bytes)}, {_human(budget)} available — running in memory"
        )
        return RunPlan("in-core", budget, 0, 1, None, tuple(reasons), None)

    reasons.append(
        f"needs {_human(estimate.peak_bytes)} but only {_human(budget)} is safely available"
    )

    downsample = 1
    if allow_downsample:
        # Each doubling of the factor quarters the in-plane element count.
        while downsample < 8 and estimate.peak_bytes / (downsample**2) > budget:
            downsample *= 2
        if downsample > 1:
            reasons.append(
                f"'allow downsample' is on — coarsening by {downsample}x, recorded in the output"
            )

    effective_peak = estimate.peak_bytes / (downsample**2)

    if estimate.chunkable:
        n_layers = estimate.shape[0] if estimate.shape else 1
        per_layer = max(1, int(effective_peak / max(1, n_layers)))
        chunk_layers = max(1, min(n_layers, int(max(budget, MIN_BUDGET_BYTES) / per_layer)))
        reasons.append(
            f"chunking into groups of {chunk_layers} of {n_layers} layers — slower, same result"
        )
        return RunPlan("chunked", budget, chunk_layers, downsample, None, tuple(reasons), None)

    needed = int(effective_peak)
    reasons.append("this step needs the whole array addressable — running disk-backed")
    blocked = None
    if profile.disk_free and needed > profile.disk_free:
        blocked = (
            f"needs {_human(needed)} of scratch disk but only {_human(profile.disk_free)} is free"
        )
    return RunPlan("disk-backed", budget, 0, downsample, scratch_dir, tuple(reasons), blocked)


def advise_3d(profile, shape, mode: str) -> Advice:
    """Recommended downsample and render mode for a volume of *shape*.

    Volume mode uploads the grid as ONE 3-D texture; exceeding
    ``GL_MAX_3D_TEXTURE_SIZE`` makes VTK render nothing at all — a silently
    blank product. Geometry modes (surface/isosurface) upload geometry instead,
    so they are unaffected and get no advice.
    """
    reasons: list[str] = []
    if mode != "volume" or profile.gl is None or not profile.gl.max_3d_texture:
        return Advice(1, None, ())
    limit = int(profile.gl.max_3d_texture)
    # The texture is sized in POINTS — one more than the voxel count per axis.
    longest = max(int(d) + 1 for d in shape)
    if longest <= limit:
        return Advice(1, None, ())

    downsample = 1
    while downsample < 16 and (longest // downsample) + 1 > limit:
        downsample *= 2
    reasons.append(
        f"volume is {longest - 1} px on its longest axis but this GL stack caps 3-D "
        f"textures at {limit} px — downsample {downsample}x, or volume mode renders blank"
    )
    if profile.gl.software:
        reasons.append(
            f"software renderer ({profile.gl.renderer}) — surface mode uploads geometry "
            "instead of one large texture and will be far faster here"
        )
    return Advice(downsample, "surface", tuple(reasons))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_common_advice.py -v`
Expected: PASS (11 tests).

- [ ] **Step 5: Document in Codebase.md**

```markdown
- **`advice.py`** — machine-aware policy. `RunPlan` / `Advice` dataclasses,
  `headroom_bytes`, `plan_run` (in-core -> chunked -> disk-backed escalation;
  never blocks for RAM, only for scratch disk) and `advise_3d` (downsample to
  fit `GL_MAX_3D_TEXTURE_SIZE`, prefer geometry modes on software GL). Pure
  functions with no IO; every decision carries a plain-language reason string
  that the log, banner and result notes all reuse.
```

- [ ] **Step 6: Commit**

```bash
git add dfxm/common/advice.py tests/test_common_advice.py docs/Codebase.md
git commit -m "feat: machine-aware run planning and 3-D advice"
```

---

## Task 7: Stage estimators — strain and mosaicity

**Files:**
- Modify: `dfxm/stages/strain.py`, `dfxm/stages/mosaicity.py`, `docs/Codebase.md`
- Create: `tests/test_stage_estimates.py`

**Interfaces:**
- Consumes: `CostEstimate`, `StageSpec.estimate` (Task 5).
- Produces:
  - `dfxm.stages.strain:estimate(params: dict) -> CostEstimate`
  - `dfxm.stages.mosaicity:estimate(params: dict) -> CostEstimate`
  - `volumeio` is NOT available yet (Task 9). Keep the arithmetic local and explicit; do not refactor these later.

An estimator must **never raise**: an unreadable or missing input returns `CostEstimate(0, 0, None, True, note="...")` so the GUI shows "unknown", not an error.

**Both stages are folder-based, not file-based.** They resolve a work list of
layer folders (`mode` = `single` → `input_folder`; `batch` → `root_folder` +
`folder_pattern`), then open `<folder>/<maps_filename>` in each. There is no
`input_h5` parameter. The estimator mirrors `run()`'s resolution exactly, using
the same `find_matching_folders` helper, then sizes **one** layer and multiplies
by the layer count — it must not open every file.

**The peak models, read off the actual `run()` bodies:**

| Stage | What `run()` holds | Peak |
|---|---|---|
| `strain` | `slices: list[np.ndarray]` accumulates one float64 strain map per layer (`strain.py:836`), then `np.stack` builds a contiguous copy — both live at once | `2 * n_layers * H * W * 8` |
| `mosaicity` | `collected: dict[str, list[np.ndarray]]` accumulates every layer of **all four** datasets (chi/mu × com/fwhm) at once, then `np.stack` adds one more volume | `(n_present + 1) * n_layers * H * W * itemsize` |

Note that mosaicity holds four volumes simultaneously. The "memory-light, peak =
one layer" comment at `mosaicity.py:223` describes `_volume_stats`, a plotting
helper — it does **not** describe the stage, which is among the heaviest here.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_stage_estimates.py
"""Shape-only cost estimators for stages (never read data, never raise)."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from dfxm.config.models import CostEstimate

H, W = 8, 16
CCMTH_PATH = "/entry/ccmth/Center of mass/Center of mass"
MOSA_PATHS = {
    "chi_com_path": "/entry/chi/Center of mass/Center of mass",
    "chi_fwhm_path": "/entry/chi/FWHM/FWHM",
    "mu_com_path": "/entry/mu/Center of mass/Center of mass",
    "mu_fwhm_path": "/entry/mu/FWHM/FWHM",
}


def _make_layers(tmp_path, n_layers=3, dtype="float32", *, mosa=False):
    """A root with *n_layers* ``layer__N`` folders, each holding a maps.h5."""
    root = tmp_path / "root"
    root.mkdir()
    for i in range(n_layers):
        folder = root / f"layer__{i + 1}"
        folder.mkdir()
        with h5py.File(folder / "maps.h5", "w") as f:
            layer = np.zeros((H, W), dtype=dtype)
            paths = MOSA_PATHS.values() if mosa else (CCMTH_PATH,)
            for path in paths:
                f.create_dataset(path, data=layer)
    return str(root)


def _strain_params(root, **over):
    params = {
        "mode": "batch",
        "root_folder": root,
        "folder_pattern": "layer__*",
        "maps_filename": "maps.h5",
        "ccmth_com_path": CCMTH_PATH,
    }
    params.update(over)
    return params


def test_strain_estimate_reports_shape_and_peak(tmp_path):
    from dfxm.stages.strain import estimate

    root = _make_layers(tmp_path, n_layers=3, dtype="float32")
    est = estimate(_strain_params(root))
    assert isinstance(est, CostEstimate)
    assert est.shape == (3, H, W)
    assert est.input_bytes == 3 * H * W * 4
    # run() holds a float64 map per layer AND the np.stack copy simultaneously
    assert est.peak_bytes == 2 * 3 * H * W * 8


def test_strain_estimate_sizes_one_layer_not_all_of_them(tmp_path, monkeypatch):
    """It must open the first maps.h5 only — this runs on every form change."""
    from dfxm.stages import strain

    root = _make_layers(tmp_path, n_layers=5)
    opened = []
    real_open = h5py.File

    def counting_open(name, *a, **k):
        opened.append(str(name))
        return real_open(name, *a, **k)

    monkeypatch.setattr(strain.h5py, "File", counting_open)
    strain.estimate(_strain_params(root))
    assert len(opened) == 1, f"opened {len(opened)} files: {opened}"


def test_mosaicity_estimate_accounts_for_all_four_datasets(tmp_path):
    """run() holds chi/mu x com/fwhm at once, then np.stack adds one more."""
    from dfxm.stages.mosaicity import estimate

    root = _make_layers(tmp_path, n_layers=3, dtype="float32", mosa=True)
    params = {
        "mode": "batch",
        "root_folder": root,
        "folder_pattern": "layer__*",
        "maps_filename": "maps.h5",
        **MOSA_PATHS,
    }
    est = estimate(params)
    per_volume = 3 * H * W * 4
    assert est.input_bytes == 4 * per_volume
    assert est.peak_bytes == 5 * per_volume  # four collected + one stacked


def test_estimators_never_raise_on_a_missing_root(tmp_path):
    from dfxm.stages.mosaicity import estimate as mosa_estimate
    from dfxm.stages.strain import estimate as strain_estimate

    missing = str(tmp_path / "nope")
    for fn in (strain_estimate, mosa_estimate):
        est = fn({"mode": "batch", "root_folder": missing, "folder_pattern": "*"})
        assert est.peak_bytes == 0
        assert est.shape is None
        assert est.note  # says why it is unknown


def test_estimators_never_read_data(tmp_path, monkeypatch):
    """Guard the cheapness contract: shapes only, so it can run on every keystroke."""
    from dfxm.stages.strain import estimate

    root = _make_layers(tmp_path, n_layers=2)

    def explode(*a, **k):
        raise AssertionError("estimator read dataset contents")

    monkeypatch.setattr(h5py.Dataset, "__getitem__", explode)
    est = estimate(_strain_params(root))
    assert est.shape == (2, H, W)


def test_specs_declare_their_estimators():
    from dfxm.stages import mosaicity, strain

    for module in (strain, mosaicity):
        assert module.STAGE.estimate is not None
        assert callable(module.STAGE.estimator())


@pytest.mark.parametrize("dtype,itemsize", [("float32", 4), ("float64", 8), ("uint16", 2)])
def test_strain_input_bytes_follow_the_source_dtype(tmp_path, dtype, itemsize):
    """input_bytes tracks the file; peak is always float64 because run() converts."""
    from dfxm.stages.strain import estimate

    root = _make_layers(tmp_path, n_layers=2, dtype=dtype)
    est = estimate(_strain_params(root))
    assert est.input_bytes == 2 * H * W * itemsize
    assert est.peak_bytes == 2 * 2 * H * W * 8
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_stage_estimates.py -v`
Expected: FAIL — `cannot import name 'estimate' from 'dfxm.stages.strain'`.

- [ ] **Step 3: Add a shared work-list helper**

Both estimators need the same folder resolution `run()` uses. Add to
`dfxm/common/sort.py`, next to `find_matching_folders`:

```python
def resolve_layer_work(params: dict, *, maps_filename: str) -> list[str]:
    """The ``<folder>/<maps_filename>`` paths a folder-based stage would process.

    Mirrors the work-list resolution in ``strain.run`` / ``mosaicity.run``
    (single -> ``input_folder``; batch -> ``root_folder`` + ``folder_pattern``)
    so an estimator predicts the same job the run will do. Returns [] rather
    than raising when nothing resolves — estimators are advisory.
    """
    if params.get("mode") == "single":
        folder = str(params.get("input_folder") or "")
        folders = [folder] if folder else []
    else:
        root = str(params.get("root_folder") or "").rstrip("/")
        folders = find_matching_folders(root, params.get("folder_pattern") or "*") if root else []
    return [os.path.join(f, maps_filename) for f in folders if f]
```

`dfxm/common/sort.py` already imports `os` and `glob`; check before adding.

- [ ] **Step 4: Implement the estimator in `dfxm/stages/strain.py`**

Add near `load_map` (`dfxm/stages/strain.py:364`):

```python
def estimate(params: dict) -> CostEstimate:
    """Peak memory for a strain run, from HDF5 shapes only.

    Reads ``.shape``/``.dtype`` of ONE layer and multiplies by the layer count,
    never touching data, so the GUI can call this on every form change. Never
    raises: an unreadable input reports an unknown cost with the reason in
    ``note``.

    The peak is ``2 * n_layers * H * W * 8``: ``run()`` accumulates a float64
    strain map per layer in ``slices`` and then ``np.stack`` builds a contiguous
    copy, so both are resident at the high-water mark.
    """
    p = {**STAGE.defaults(), **params}
    try:
        work = resolve_layer_work(p, maps_filename=str(p["maps_filename"] or "maps.h5"))
        if not work:
            return CostEstimate(0, 0, None, True, "no layer folders resolved yet")
        ds_path = str(p["ccmth_com_path"])
        with h5py.File(work[0], "r") as f:
            if ds_path not in f:
                return CostEstimate(0, 0, None, True, f"{ds_path!r} not in {work[0]!r}")
            ds = f[ds_path]
            layer_shape = tuple(int(d) for d in ds.shape)
            itemsize = int(ds.dtype.itemsize)
    except Exception as exc:  # noqa: BLE001 - an estimate is advisory, never fatal
        return CostEstimate(0, 0, None, True, f"cannot size input: {type(exc).__name__}")

    layer_elems = 1
    for dim in layer_shape:
        layer_elems *= dim
    n_layers = len(work)
    input_bytes = n_layers * layer_elems * itemsize
    peak_bytes = 2 * n_layers * layer_elems * 8
    return CostEstimate(peak_bytes, input_bytes, (n_layers, *layer_shape), True, None)
```

Add `resolve_layer_work` to `strain.py`'s existing `..common.sort` import (it
already imports `find_matching_folders` from there — read the exact line before
editing), and `CostEstimate` to its existing `..config.models` import. Extend
the existing import lines; do not add duplicates, and do not reconstruct them
from memory.

Then add `estimate="dfxm.stages.strain:estimate",` to the `STAGE = StageSpec(...)`
call at `dfxm/stages/strain.py:61`, after the `params=(...)` tuple.

- [ ] **Step 5: Implement the estimator in `dfxm/stages/mosaicity.py`**

`mosaicity.run()` collects **all four** datasets' layers before stacking
(`collected: dict[str, list[np.ndarray]]`), so its peak is roughly five volumes,
not one layer:

```python
def estimate(params: dict) -> CostEstimate:
    """Peak memory for a mosaicity run, from HDF5 shapes only.

    ``run()`` holds every layer of all present datasets (chi/mu x com/fwhm) in
    ``collected`` at once, then ``np.stack`` builds one more contiguous volume
    per dataset. Peak is therefore ``(n_present + 1)`` volumes, which makes this
    one of the heaviest stages — not the layer-streaming one the comment at
    ``_volume_stats`` might suggest.
    """
    p = {**STAGE.defaults(), **params}
    try:
        work = resolve_layer_work(p, maps_filename=str(p["maps_filename"] or "maps.h5"))
        if not work:
            return CostEstimate(0, 0, None, True, "no layer folders resolved yet")
        present = 0
        layer_shape: tuple[int, ...] = ()
        itemsize = 8
        with h5py.File(work[0], "r") as f:
            for key, _default, _group, _name in _DATASETS:
                ds_path = str(p.get(key) or "")
                if ds_path and ds_path in f:
                    ds = f[ds_path]
                    layer_shape = tuple(int(d) for d in ds.shape)
                    itemsize = int(ds.dtype.itemsize)
                    present += 1
        if not present:
            return CostEstimate(0, 0, None, True, "none of the mosaicity datasets found")
    except Exception as exc:  # noqa: BLE001 - an estimate is advisory, never fatal
        return CostEstimate(0, 0, None, True, f"cannot size input: {type(exc).__name__}")

    layer_elems = 1
    for dim in layer_shape:
        layer_elems *= dim
    n_layers = len(work)
    per_volume = n_layers * layer_elems * itemsize
    input_bytes = present * per_volume
    peak_bytes = (present + 1) * per_volume
    return CostEstimate(
        peak_bytes,
        input_bytes,
        (n_layers, *layer_shape),
        True,
        f"{present} datasets stacked together",
    )
```

Add the same two imports and `estimate="dfxm.stages.mosaicity:estimate",` to
that module's `STAGE`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_stage_estimates.py -v`
Expected: PASS (8 tests).

- [ ] **Step 7: Run the full suite — this task modified shipped stage modules**

Run: `python3 -m pytest -q`
Expected: 1064 + new tests passing, no regressions.

- [ ] **Step 8: Document in Codebase.md** — note under both stages that they
expose `estimate(params) -> CostEstimate` wired into `STAGE.estimate`, and add
`resolve_layer_work` to the `dfxm/common/sort.py` entry.

- [ ] **Step 9: Commit**

```bash
git add dfxm/common/sort.py dfxm/stages/strain.py dfxm/stages/mosaicity.py \
        tests/test_stage_estimates.py docs/Codebase.md
git commit -m "feat: shape-only cost estimators for strain and mosaicity"
```

---

## Task 8: Stage estimators — slices, rocking, matched, paraview, visualize

**Files:**
- Modify: `dfxm/stages/slices.py`, `dfxm/stages/rocking.py`, `dfxm/stages/matched.py`, `dfxm/stages/paraview.py`, `dfxm/stages/visualize.py`, `docs/Codebase.md`
- Test: `tests/test_stage_estimates.py`

**Interfaces:**
- Consumes: `CostEstimate` (Task 5); the estimator contract established in Task 7.
- Produces:
  - `dfxm.common.h5io:sum_dataset_bytes(path) -> tuple[int, tuple[int, ...] | None, int]` — `(total_bytes, largest_shape, largest_itemsize)`, shapes only.
  - `estimate(params) -> CostEstimate` in each of the five modules, each wired into its `STAGE`.

Same contract as Task 7: shapes only, never raises, `peak_bytes` accounts for live conversions.

**These five split into two input styles**, neither of which is `input_h5`:

- **File-based** — `slices`, `paraview`, `visualize` read whole stacked volumes
  named by `mosa_volume_file` / `strain_volume_file` (and, for `slices`, also
  `aligned_rocking_file` / `aligned_mosa_file`).
- **Raw-scan-based** — `rocking` and `matched` glob `raw_root` with
  `rocking_pattern`, then read a detector stack at `detector_path` /
  `pco_ff_path` (both default `"1.1/measurement/pco_ff"`).

Per-stage peak arithmetic, read off the actual load sites:

> **CORRECTED by the fix wave (2026-08-20, commit b31d789 — the defective
> models shipped in e48e69a..1a262c9 as this plan then specified them):** the
> table below as originally written mismodelled five of these
> five sites — `rocking.py:985` cited by the plan is actually
> `_replot_default_clim` (the cold-replot helper), not `run()`'s real path;
> `slices`, `paraview` and `visualize` all summed-across-files instead of
> reading each `run()`'s actual sequencing (`paraview`/`slices` process one
> file/volume at a time and free the previous one — `max`, not `sum`;
> `visualize` keeps its mosaicity `datasets` dict alive through the strain
> section too — genuinely `sum`); and `matched`'s peak scaled with the folder
> count when `run()` only ever holds one scan at a time. The rows now show the
> arithmetic actually implemented (verified against `run()`/the named helpers
> and pinned by `tests/test_stage_estimates.py`); the struck-through original
> readings are kept for history.

| Stage | Site | Peak model | `chunkable` |
|---|---|---|---|
| `slices` | `slices.py:737` (stacked), `slices.py:753` (aligned), `slices.py:1327` (per-volume loop) | `run()` calls `prepare_volume` one selected dataset at a time; `prep` is rebound each iteration so the previous volume stays alive while the next is built. Per dataset `v`: stacked source (`mosa_volume_file`/`strain_volume_file`) = native read + 3 float64 copies (raw read, samy-shifted canvas, `interpolate_to_uniform_z` output) → `load_peak_v = elems_v*itemsize_v + 3*elems_v*8`; aligned source (`aligned_rocking_file`/`aligned_mosa_file`, already co-registered) = native read + 1 float64 copy → `load_peak_v = elems_v*itemsize_v + 1*elems_v*8`. `peak = max_v(load_peak_v + max_{w≠v}(elems_w*8))` — the max **pair**, not the sum across all selected volumes. ~~`input + 2 * n * 8` summed over every selected volume~~ | `False` — alignment is whole-volume |
| `paraview` | `paraview.py:663` `_process_mosaicity`, `paraview.py:721` `_process_strain`, `paraview.py:496` `save_volumes_as_pvti` | `run()` calls the two helpers **sequentially** — each one's locals (including the raw `datasets` dict) die on return, so the files' peaks don't add. Per file: raw datasets alive (`file_total`) + aligned float64 copies of every field (`file_elems*8`) + `save_volumes_as_pvti`'s `np.where`-cleaned float64 copies + `valid_mask` (`file_elems*8 + largest_elems*8`) → `file_peak = file_total + 2*file_elems*8 + largest_elems*8`. `peak = max` over the (≤2) files processed. ~~every dataset in the file, loaded together → `sum(n_i * itemsize_i)`~~ | `True` |
| `visualize` | `visualize.py:641` (mosaicity `datasets` load), `visualize.py:684` (strain load), `visualize.py:517` `_align` | Unlike `paraview`, the mosaicity and strain sections run **inline in the same function scope** — the mosaicity `datasets` dict is never deleted or reassigned, so it stays alive through the strain section too and the two files' input bytes **add**. `peak = total_input + 3 * largest_elems * 8` — one field's `_align` chain (ROI → samy-shift → `interpolate_to_uniform_z`, upcasting to float64) leaves up to three float64-sized temporaries of the largest field alive at once. ~~same whole-file sum as `paraview`~~ | `True` |
| `rocking` | `rocking.py:438` `process_raw_scan`, `rocking.py:521` `build_raw_volumes`'s `np.stack` | ~~`rocking.py:985` bare `dataset[:]`~~ — that line is `_replot_default_clim` (cold replot), not `run()`. The real path streams **one scan at a time**: uint16 read + its `.astype(np.float32)` copy coexist briefly, `del frames` before the next scan — nothing scales with folder count there. Only the running 2-D accumulators and the two final float32 volumes (doubled while `np.stack` builds each) persist across the loop: `peak = max(scan_elems*(itemsize+4) + 2*n*layer_elems*4, 20*n*layer_elems)`. Folder count is an upper bound on `n` (samz-union masking / `source_scan="mosaicity"` are not evaluated here). | `True` |
| `matched` | `matched.py:263` `load_pco_ff_frame`, `matched.py:467` matched-layer loop | `run()` loads **one scan at a time** (locals die on return each layer) — peak does not scale with folder count. `ds[:].astype(np.float64)` (native + float64 copy) + `np.nanmedian`'s own internal float64-sized sort copy → `scan_elems*(itemsize+16)`, plus ~10 pooled clim arrays + a couple of frame-sized working copies → `12*frame_elems*8`. `peak = scan_elems*(itemsize+16) + 12*frame_elems*8`. ~~`input + n * 8 + frame_bytes`~~ | `False` — an exact median needs the whole stack |

**Reference figures from the real STO2 dataset** (76 layers, verified 2026-08-20)
— use these to sanity-check your implementation:

| File | Contents | Total |
|---|---|---|
| `stacked_volumes.h5` | 4 × `(76, 1266, 1832)` float64 | 5.25 GB |
| `stacked_strain_volumes.h5` | 1 × `(76, 1266, 1832)` float64 | 1.31 GB |
| `aligned_raw_mosa_volumes.h5` | 2 × `(76, 700, 2891)` float32 | 1.15 GB |

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_stage_estimates.py`:

```python
ALL_ESTIMATOR_STAGES = ("strain", "mosaicity", "slices", "rocking", "matched", "paraview", "visualize")


@pytest.mark.parametrize("stage_name", ALL_ESTIMATOR_STAGES)
def test_every_volume_stage_declares_an_estimator(stage_name):
    import importlib

    module = importlib.import_module(f"dfxm.stages.{stage_name}")
    assert module.STAGE.estimate == f"dfxm.stages.{stage_name}:estimate"
    assert callable(module.STAGE.estimator())


@pytest.mark.parametrize("stage_name", ALL_ESTIMATOR_STAGES)
def test_every_estimator_survives_junk_params(stage_name):
    """Called on every form change, including while the user is mid-typing."""
    import importlib

    module = importlib.import_module(f"dfxm.stages.{stage_name}")
    junk = (
        {},
        {"raw_root": "", "mosa_volume_file": ""},
        {"raw_root": "/nonexistent", "mosa_volume_file": "/nonexistent/x.h5"},
    )
    for params in junk:
        est = module.estimate(params)
        assert isinstance(est, CostEstimate)
        assert est.peak_bytes >= 0


def test_sum_dataset_bytes_walks_nested_groups(tmp_path):
    from dfxm.common.h5io import sum_dataset_bytes

    path = tmp_path / "v.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("chi/Center of mass", data=np.zeros((3, 4, 5), dtype="float64"))
        f.create_dataset("mu/FWHM", data=np.zeros((3, 4, 5), dtype="float32"))
    total, largest, itemsize = sum_dataset_bytes(str(path))
    assert total == 3 * 4 * 5 * 8 + 3 * 4 * 5 * 4
    assert largest == (3, 4, 5)
    assert itemsize == 8  # the largest dataset's itemsize


def test_sum_dataset_bytes_on_a_missing_file_is_zero(tmp_path):
    from dfxm.common.h5io import sum_dataset_bytes

    assert sum_dataset_bytes(str(tmp_path / "nope.h5")) == (0, None, 0)


def test_slices_is_not_chunkable_and_peaks_at_four_arrays_worth(tmp_path):
    """astype(float64) + shifted canvas + interpolated output — four arrays'
    worth at the peak for a stacked-source volume."""
    from dfxm.stages.slices import estimate

    path = tmp_path / "mosa.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("chi/Center of mass", data=np.zeros((4, 8, 16), dtype="float32"))
    params = {"mosa_volume_file": str(path), "include_mosa_com_chi": True,
              "include_mosa_fwhm_chi": False, "include_mosa_com_mu": False,
              "include_mosa_fwhm_mu": False, "include_strain": False,
              "include_raw_sum": False, "include_raw_specific": False,
              "include_mosa_sum": False, "include_mosa_specific": False}
    est = estimate(params)
    n = 4 * 8 * 16
    assert est.chunkable is False
    assert est.input_bytes == n * 4
    assert est.peak_bytes == n * 4 + 3 * n * 8


def test_slices_peak_across_two_volumes_is_the_max_pair_not_the_sum(tmp_path):
    """run() holds at most the current + previous volume, never every volume."""
    # (two toggled files: mosa chi/CoM (stacked, 3 copies) + aligned rocking
    # sum_intensity (aligned, 1 copy); assert peak == max-pair, strictly less
    # than the sum of both volumes' individual load peaks)


def test_matched_is_not_chunkable(tmp_path):
    """An exact median needs the whole stack — bucket 3, disk-backed. Peak is
    the per-scan astype(float64) + nanmedian's internal copy + pooled/frame
    working set — independent of how many scan folders match."""
    from dfxm.stages.matched import estimate

    root = tmp_path / "raw"
    scan = root / "rock__1"
    scan.mkdir(parents=True)
    with h5py.File(scan / "rock__1.h5", "w") as f:
        f.create_dataset("1.1/measurement/pco_ff", data=np.zeros((6, 8, 16), dtype="uint16"))
    est = estimate({"raw_root": str(root), "rocking_pattern": "rock__*"})
    scan_elems, frame_elems = 6 * 8 * 16, 8 * 16
    assert est.chunkable is False
    assert est.peak_bytes == scan_elems * (2 + 16) + 12 * frame_elems * 8
    assert est.peak_bytes == 26112


def test_matched_peak_does_not_grow_with_folder_count(tmp_path):
    """A second scan folder must double input_bytes but leave peak_bytes put."""
    # (two rock__N folders, same (6, 8, 16) uint16 shape; assert peak_bytes ==
    # 26112 unchanged while input_bytes doubles to 2 * 6 * 8 * 16 * 2)


def test_rocking_peak_models_streaming_per_scan(tmp_path):
    """run() streams one scan at a time (uint16 + float32 coexist briefly,
    `del frames` before the next scan) — it does not hold every scan's stack
    at once. `rocking.py:985`, the site the plan originally cited, is
    `_replot_default_clim` (cold replot), not this path."""
    from dfxm.stages.rocking import estimate

    root = tmp_path / "raw"
    for i in range(2):
        scan = root / f"rock__{i}"
        scan.mkdir(parents=True)
        with h5py.File(scan / f"rock__{i}.h5", "w") as f:
            f.create_dataset("1.1/measurement/pco_ff", data=np.zeros((3, 8, 16), dtype="uint16"))
    est = estimate({"raw_root": str(root), "rocking_pattern": "rock__*"})
    assert est.input_bytes == 2 * 3 * 8 * 16 * 2  # 2 scans x 3 frames x uint16
    assert est.input_bytes == 1536
    assert est.peak_bytes == 5120


def test_paraview_peak_is_the_max_over_files_not_the_sum(tmp_path):
    """_process_mosaicity/_process_strain are separate calls — their locals
    (including the raw datasets dict) die on return, so the two files' peaks
    don't add."""
    # (4-field mosa file + 1-field strain file, both (4, 8, 16) float64;
    # assert peak == max(mosa_peak, strain_peak), strictly less than their sum)


def test_visualize_peak_sums_inputs_because_datasets_dict_outlives_the_loop(tmp_path):
    """Unlike paraview, mosaicity + strain share run()'s scope — the
    mosaicity `datasets` dict is never freed before the strain section runs,
    so the two files' input bytes DO add (unlike paraview's max-over-files)."""
    # (same two files as the paraview test; assert peak == total_input +
    # 3 * largest_field_elems * 8)
```

**CORRECTED by the fix wave (2026-08-20):** the four snippets above
(`test_slices_...`, `test_matched_is_not_chunkable`,
`test_rocking_peak_has_no_conversion_overhead` -> `..._models_streaming_per_scan`)
are shown with their corrected bodies; the new tests (`test_slices_peak_..._not_the_sum`,
`test_matched_peak_does_not_grow_with_folder_count`, `test_paraview_peak_is_the_max_...`,
`test_visualize_peak_sums_inputs_...`) are sketched rather than spelled out —
see `tests/test_stage_estimates.py` for the exact fixtures and assertions;
duplicating full fixture bodies here would just be a second place for them to
drift out of sync.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_stage_estimates.py -v`
Expected: FAIL — `cannot import name 'sum_dataset_bytes'`.

- [ ] **Step 3: Add the shared sizing helper to `dfxm/common/h5io.py`**

```python
def sum_dataset_bytes(path: str) -> tuple[int, tuple[int, ...] | None, int]:
    """Total in-memory size of every dataset in *path*, from shapes alone.

    Returns ``(total_bytes, largest_shape, largest_itemsize)``. Walks nested
    groups, reads no data, and returns ``(0, None, 0)`` for anything it cannot
    open — sizing is advisory and must never raise.
    """
    total = 0
    largest_elems = 0
    largest_shape: tuple[int, ...] | None = None
    largest_itemsize = 0

    def visit(_name, obj):
        nonlocal total, largest_elems, largest_shape, largest_itemsize
        if not isinstance(obj, h5py.Dataset):
            return
        n = 1
        for dim in obj.shape:
            n *= int(dim)
        itemsize = int(obj.dtype.itemsize)
        total += n * itemsize
        if n > largest_elems:
            largest_elems = n
            largest_shape = tuple(int(d) for d in obj.shape)
            largest_itemsize = itemsize

    try:
        with h5py.File(path, "r") as f:
            f.visititems(visit)
    except Exception:  # noqa: BLE001 - unreadable input -> unknown size
        return 0, None, 0
    return total, largest_shape, largest_itemsize
```

- [ ] **Step 4: Implement the three file-based estimators**

`slices`, `paraview` and `visualize` all sum the volume files their params name.
Write this in `dfxm/stages/slices.py`:

```python
_SLICES_VOLUME_PARAMS = (
    "mosa_volume_file",
    "strain_volume_file",
    "aligned_rocking_file",
    "aligned_mosa_file",
)


def estimate(params: dict) -> CostEstimate:
    """Peak memory for a slices run, from HDF5 shapes only.

    ``prepare_volume`` reads each selected volume with ``[:].astype(np.float64)``
    — source and float64 copy live together — then alignment
    (``apply_samy_shifts_to_volume`` / ``interpolate_to_uniform_z``) produces a
    further float64 copy. Peak is therefore ``input + 2 * n * 8`` summed over
    the selected volumes. Not chunkable: alignment is a whole-volume operation.
    """
    p = {**STAGE.defaults(), **params}
    total_input = 0
    total_elems = 0
    largest: tuple[int, ...] | None = None
    for name in _SLICES_VOLUME_PARAMS:
        path = str(p.get(name) or "")
        if not path:
            continue
        nbytes, shape, itemsize = sum_dataset_bytes(path)
        if not nbytes:
            continue
        total_input += nbytes
        total_elems += nbytes // max(1, itemsize)
        if shape is not None and (largest is None or len(shape) > len(largest)):
            largest = shape
    if not total_input:
        return CostEstimate(0, 0, None, False, "no readable volume files selected yet")
    return CostEstimate(total_input + 2 * total_elems * 8, total_input, largest, False, None)
```

`paraview` and `visualize` load every dataset in their volume files without a
conversion, so their peak is the plain sum. Write this in **both**
`dfxm/stages/paraview.py` and `dfxm/stages/visualize.py`:

```python
def estimate(params: dict) -> CostEstimate:
    """Peak memory for this run, from HDF5 shapes only.

    Loads every dataset in the selected volume files together (see the
    group-walk in ``_load_volumes``), with no dtype conversion, so the peak is
    simply their combined size.
    """
    p = {**STAGE.defaults(), **params}
    total = 0
    largest: tuple[int, ...] | None = None
    for name in ("mosa_volume_file", "strain_volume_file"):
        path = str(p.get(name) or "")
        if not path:
            continue
        nbytes, shape, _itemsize = sum_dataset_bytes(path)
        total += nbytes
        if shape is not None and (largest is None or len(shape) > len(largest)):
            largest = shape
    if not total:
        return CostEstimate(0, 0, None, True, "no readable volume files selected yet")
    return CostEstimate(total, total, largest, True, None)
```

- [ ] **Step 5: Implement the two raw-scan estimators**

`rocking` and `matched` glob `raw_root` with `rocking_pattern`, then read a
detector stack per scan folder. Size **one** folder and multiply. Write this in
`dfxm/stages/rocking.py`:

```python
def estimate(params: dict) -> CostEstimate:
    """Peak memory for a rocking run, from HDF5 shapes only.

    ``rocking.py:985`` is a bare ``dataset[:]`` with no dtype conversion, so the
    peak is just the combined detector stacks.
    """
    p = {**STAGE.defaults(), **params}
    try:
        root = str(p.get("raw_root") or "").rstrip("/")
        folders = find_matching_folders(root, p.get("rocking_pattern") or "*") if root else []
        if not folders:
            return CostEstimate(0, 0, None, True, "no scan folders resolved yet")
        first = resolve_input_file(folders[0])
        ds_path = str(p.get("detector_path") or "1.1/measurement/pco_ff")
        with h5py.File(first, "r") as f:
            if ds_path not in f:
                return CostEstimate(0, 0, None, True, f"{ds_path!r} not in {first!r}")
            ds = f[ds_path]
            scan_shape = tuple(int(d) for d in ds.shape)
            itemsize = int(ds.dtype.itemsize)
    except Exception as exc:  # noqa: BLE001 - an estimate is advisory, never fatal
        return CostEstimate(0, 0, None, True, f"cannot size input: {type(exc).__name__}")

    elems = 1
    for dim in scan_shape:
        elems *= dim
    total = len(folders) * elems * itemsize
    return CostEstimate(total, total, (len(folders), *scan_shape), True, None)
```

For `dfxm/stages/matched.py`, use the same body with three changes: read
`pco_ff_path` instead of `detector_path`; set `chunkable=False`; and compute the
peak as the `astype(np.float64)` conversion plus the median frame —

```python
    frame_elems = elems // scan_shape[0] if scan_shape and scan_shape[0] else elems
    input_bytes = len(folders) * elems * itemsize
    peak = input_bytes + len(folders) * elems * 8 + frame_elems * 8
    return CostEstimate(
        peak,
        input_bytes,
        (len(folders), *scan_shape),
        False,
        "exact median needs the whole stack",
    )
```

**CORRECTED by the fix wave (2026-08-20):** the `estimate()` bodies in Steps
4-5 above are the ORIGINAL (defective) implementations, kept verbatim for
history. All five were replaced with the corrected models from the
peak-arithmetic table above — read `dfxm/common/h5io.py:iter_dataset_sizes`
(new helper, added alongside `sum_dataset_bytes`) and the five stages'
`estimate()` functions directly for the arithmetic actually shipped; it is
not worth re-transcribing ~150 lines of corrected Python into the plan a
second time when the module docstrings (also corrected) carry the same
explanation next to the code they describe.

Each of the five modules needs `CostEstimate` added to its existing
`..config.models` import and `sum_dataset_bytes` / `resolve_input_file` /
`find_matching_folders` added to its existing `..common.*` imports. **Read the
exact existing import lines before editing** — several of these modules already
import from both, and reconstructing the line from memory is the known hazard
here. Then add `estimate="dfxm.stages.<name>:estimate",` to each `STAGE`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_stage_estimates.py -v`
Expected: PASS.

- [ ] **Step 7: Run the full suite**

Run: `python3 -m pytest -q`
Expected: no regressions.

- [ ] **Step 8: Print the real STO2 numbers — this is what phase 5 gets decided on**

The dataset drive was mounted and verified on 2026-08-20; if these paths have
moved, adjust the root and say so in the commit message.

```bash
python3 - <<'PY'
from dfxm.common.advice import headroom_bytes, plan_run
from dfxm.common.machine import profile
from dfxm.stages import mosaicity, paraview, slices, strain, visualize

ROOT = "/media/albert/DIC_SSD_3/ESRF/ma6778/id03/20251029/PROCESSED_DATA/STO2_overnight"
p = profile()
print(f"headroom here: {headroom_bytes(p) / 1024**3:.2f} GB")

cases = {
    "strain": (strain, {"mode": "batch", "root_folder": ROOT,
                        "folder_pattern": "STO2_overnight_layer_2x_energy_strain__*"}),
    "mosaicity": (mosaicity, {"mode": "batch", "root_folder": ROOT,
                              "folder_pattern": "STO2_overnight_layer_2x_mosa__*"}),
    "slices": (slices, {"aligned_mosa_file": f"{ROOT}/aligned_raw_mosa_volumes.h5",
                        "strain_volume_file": f"{ROOT}/stacked_strain_volumes.h5"}),
    "paraview": (paraview, {"mosa_volume_file": f"{ROOT}/stacked_volumes.h5",
                            "strain_volume_file": f"{ROOT}/stacked_strain_volumes.h5"}),
    "visualize": (visualize, {"mosa_volume_file": f"{ROOT}/stacked_volumes.h5",
                              "strain_volume_file": f"{ROOT}/stacked_strain_volumes.h5"}),
}
for name, (module, params) in cases.items():
    est = module.estimate(params)
    print(f"\n{name}: peak {est.peak_bytes / 1024**3:.2f} GB  "
          f"input {est.input_bytes / 1024**3:.2f} GB  chunkable={est.chunkable}")
    print("   here     :", plan_run(p, est).strategy)
    from tests.machine_fixtures import laptop_hw_gl, tiny_ram
    print("   16 GB    :", plan_run(laptop_hw_gl(), est).strategy)
    print("   8 GB busy:", plan_run(tiny_ram(), est).strategy)
PY
```

> **CORRECTED by the fix wave (2026-08-20):** the "expected shape of the
> answer" paragraph below was wrong on two counts, both adjudicated during
> Task 8's review and fixed in the peak-model table above. (1) `slices` ≈
> "3.0 GB" was an authoring arithmetic error — the plan's own formula applied
> to its own cited files gives ≈9.67 GB, which is what the as-shipped
> (uncorrected) estimator actually printed. (2) `paraview` ≈ "6.6 GB" assumed
> `peak == input`, which was simply wrong (`paraview` never modelled a
> conversion at all). The actual, re-run figures — with the corrected
> estimators and `visualize` added — are:
>
> | Stage | peak | input | chunkable | here (this box) | 16 GB laptop | 8 GB busy |
> |---|---|---|---|---|---|---|
> | strain | 2.63 GB | 1.31 GB | True | in-core | in-core | chunked |
> | mosaicity | 6.57 GB | 5.25 GB | True | in-core | chunked | chunked |
> | slices | 6.40 GB | 2.46 GB | False | in-core | disk-backed | disk-backed |
> | paraview | 17.07 GB | 6.57 GB | True | in-core | chunked | chunked |
> | visualize | 10.51 GB | 6.57 GB | True | in-core | chunked | chunked |
>
> (headroom here: 251.21 GB — this is the 502 GB workstation.) `strain` and
> `mosaicity` are unchanged from the pre-fix-wave figures (2.63 / 6.57 GB) —
> those two estimators (Task 7) were correct. `slices` **dropped** from the
> as-shipped 9.67 GB to 6.40 GB (the max-pair model, not the file-level sum;
> note the script's original slices case also slotted
> `aligned_raw_mosa_volumes.h5` into `mosa_volume_file`, where its datasets
> match no `include_*` toggle — the corrected `aligned_mosa_file` slot above
> is what a real run uses, and adds the two aligned f32 datasets to the pair).
> `paraview` **rose** ~2.6× over its `peak == input` figure (6.57 GB) to
> 17.07 GB, now that it accounts for the aligned-copy + cleaned-copy +
> valid-mask overhead `save_volumes_as_pvti` actually allocates. `visualize`
> is a new figure (not estimated pre-fix-wave): 10.51 GB, between `mosaicity`
> and `paraview` — it shares `mosaicity`'s alignment overhead per field but,
> unlike `paraview`, its two input files' bytes add rather than max (the
> `datasets` dict outlives the loop — see the peak-model table).
>
> On the 16 GB laptop profile (5.4 GB headroom), `mosaicity`, `paraview` and
> `visualize` now tip to `chunked`, and `slices` (6.40 GB, not chunkable)
> goes `disk-backed`. On the 8 GB busy profile (0.6 GB headroom) everything
> chunkable tips to `chunked` and `slices` goes `disk-backed`.

Record the printed table in the commit message. These figures decide which of
the twelve sites phase 5 converts, and in what order.

- [ ] **Step 9: Document in Codebase.md** — add `sum_dataset_bytes` (and, per
the fix wave, `iter_dataset_sizes`) to the `h5io.py` entry, and note the
`estimate` function under each of the five stages.

- [ ] **Step 10: Commit**

```bash
git add dfxm/common/h5io.py dfxm/stages/ tests/test_stage_estimates.py docs/Codebase.md
git commit -m "feat: shape-only cost estimators for the remaining volume stages"
```

---

## Task 9: `volumeio.py` — sizing, block iteration and bounded loading

**Files:**
- Create: `dfxm/common/volumeio.py`
- Create: `tests/test_common_volumeio.py`
- Modify: `docs/Codebase.md`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `volume_bytes(dset) -> int`
  - `iter_blocks(dset, *, budget_bytes, axis=0) -> Iterator[tuple[slice, np.ndarray]]`
  - `BlockReader(dset, budget_bytes, axis)` — has `.shape`, `.dtype`, and `__iter__` yielding `(slice, array)`.
  - `load_or_stream(dset, *, budget_bytes) -> np.ndarray | BlockReader`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_common_volumeio.py
"""Bounded-memory volume IO (dfxm/common/volumeio.py)."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from dfxm.common import volumeio


@pytest.fixture
def volume(tmp_path):
    """A (7, 5, 3) float32 volume with distinct, exactly-representable values."""
    data = np.arange(7 * 5 * 3, dtype=np.float32).reshape(7, 5, 3)
    path = tmp_path / "vol.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("vol", data=data)
    return str(path), data


def test_volume_bytes_is_shape_times_itemsize(volume):
    path, data = volume
    with h5py.File(path, "r") as f:
        assert volumeio.volume_bytes(f["vol"]) == data.nbytes


def test_iter_blocks_covers_every_element_exactly_once(volume):
    path, data = volume
    with h5py.File(path, "r") as f:
        seen = np.zeros(data.shape[0], dtype=int)
        for sl, block in volumeio.iter_blocks(f["vol"], budget_bytes=data.nbytes // 3):
            seen[sl] += 1
            assert np.array_equal(block, data[sl])
    assert (seen == 1).all()


@pytest.mark.parametrize("divisor", [1, 2, 3, 5, 100])
def test_iter_blocks_reassembles_the_original_for_any_budget(volume, divisor):
    path, data = volume
    with h5py.File(path, "r") as f:
        blocks = [b for _, b in volumeio.iter_blocks(f["vol"], budget_bytes=data.nbytes // divisor)]
    assert np.array_equal(np.concatenate(blocks, axis=0), data)


def test_iter_blocks_always_yields_at_least_one_layer(volume):
    """A budget smaller than a single layer must still make progress, not hang."""
    path, data = volume
    with h5py.File(path, "r") as f:
        blocks = list(volumeio.iter_blocks(f["vol"], budget_bytes=1))
    assert len(blocks) == data.shape[0]
    assert all(b.shape[0] == 1 for _, b in blocks)


def test_load_or_stream_returns_an_array_when_it_fits(volume):
    path, data = volume
    with h5py.File(path, "r") as f:
        result = volumeio.load_or_stream(f["vol"], budget_bytes=data.nbytes * 10)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, data)


def test_load_or_stream_returns_a_reader_when_it_does_not_fit(volume):
    path, data = volume
    with h5py.File(path, "r") as f:
        result = volumeio.load_or_stream(f["vol"], budget_bytes=data.nbytes // 4)
        assert isinstance(result, volumeio.BlockReader)
        assert result.shape == data.shape
        assert np.array_equal(np.concatenate([b for _, b in result], axis=0), data)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_common_volumeio.py -v`
Expected: FAIL — `No module named 'dfxm.common.volumeio'`.

- [ ] **Step 3: Write the implementation**

```python
# dfxm/common/volumeio.py
"""Bounded-memory volume reading (Qt-free).

One shared implementation so every stage streams the same way instead of eight
divergent schemes. ``mosaicity._volume_stats`` already streamed layer-by-layer
for plotting; this generalises that pattern and adds a memory budget. (Note that
``mosaicity.run`` itself does NOT stream — it collects four whole volumes.)

The governing guarantee is **budget-independence**: for any ``budget_bytes``,
these helpers produce bit-identical results. That is what makes a laptop and a
workstation emit the same publishable data product. See :func:`block_reduce`
for why ordinary summation would break it.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def volume_bytes(dset) -> int:
    """In-memory size of *dset* if fully loaded, in bytes."""
    n = 1
    for dim in dset.shape:
        n *= int(dim)
    return n * int(dset.dtype.itemsize)


def _layers_per_block(dset, budget_bytes: int, axis: int) -> int:
    """How many slices along *axis* fit in the budget. Always at least 1."""
    n_layers = int(dset.shape[axis])
    if n_layers <= 0:
        return 1
    per_layer = max(1, volume_bytes(dset) // n_layers)
    return max(1, min(n_layers, int(max(1, budget_bytes) // per_layer)))


def iter_blocks(dset, *, budget_bytes: int, axis: int = 0) -> Iterator[tuple[slice, np.ndarray]]:
    """Yield ``(slice, array)`` blocks along *axis*, each within the budget.

    Blocks are yielded in ascending order and together cover the dataset exactly
    once, so concatenating them along *axis* reproduces the whole volume. A
    budget smaller than one layer still yields single layers rather than
    stalling — progress always beats precision here.
    """
    if axis != 0:
        raise ValueError("only axis=0 blocking is supported")
    n_layers = int(dset.shape[0])
    step = _layers_per_block(dset, budget_bytes, axis)
    for start in range(0, n_layers, step):
        stop = min(start + step, n_layers)
        sl = slice(start, stop)
        yield sl, dset[sl]


class BlockReader:
    """A dataset too large for the budget, presented as a stream of blocks.

    Deliberately *not* an ndarray look-alike: code receiving one must handle
    blocks explicitly, so an accidental whole-volume materialisation is a
    visible change rather than a silent one.
    """

    def __init__(self, dset, budget_bytes: int, axis: int = 0) -> None:
        self._dset = dset
        self._budget = budget_bytes
        self._axis = axis
        self.shape = tuple(int(d) for d in dset.shape)
        self.dtype = dset.dtype

    def __iter__(self) -> Iterator[tuple[slice, np.ndarray]]:
        return iter_blocks(self._dset, budget_bytes=self._budget, axis=self._axis)

    @property
    def nbytes(self) -> int:
        return volume_bytes(self._dset)


def load_or_stream(dset, *, budget_bytes: int):
    """The whole array when it fits the budget, else a :class:`BlockReader`."""
    if volume_bytes(dset) <= budget_bytes:
        return dset[:]
    return BlockReader(dset, budget_bytes)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_common_volumeio.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add dfxm/common/volumeio.py tests/test_common_volumeio.py
git commit -m "feat: bounded-memory block iteration over HDF5 volumes"
```

---

## Task 10: Budget-independent reductions

**Files:**
- Modify: `dfxm/common/volumeio.py`, `docs/Codebase.md`
- Test: `tests/test_common_volumeio.py`

**Interfaces:**
- Consumes: `iter_blocks`, `volume_bytes` (Task 9).
- Produces:
  - `neumaier_sum(values, *, state=None) -> tuple[float, float]` — returns `(total, compensation)`; pass the previous tuple back in as `state` to continue across blocks.
  - `block_reduce(dset, fn, *, budget_bytes, init)` — general accumulator; `fn(acc, block) -> acc`.
  - `block_nansum(dset, *, budget_bytes) -> float`
  - `two_pass(dset, stat_fn, apply_fn, *, budget_bytes, init)` — pass 1 folds blocks into a statistic starting from `init`, pass 2 applies it block-wise and yields `(slice, result)` pairs.

**Why this is not `np.sum`.** NumPy's `add.reduce` uses pairwise summation with a 128-element base case, so summing an array whole and summing it in blocks give *different* bits. Neumaier compensated summation carrying `(total, compensation)` across block boundaries is order-preserving and budget-independent, which is the property the tests assert. It costs roughly four flops per element — negligible beside the IO. **Consequence to accept knowingly:** `block_nansum` differs from `np.nansum` by up to ~1 ulp. The tests assert cross-budget identity, never equality with `np.nansum`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_common_volumeio.py`:

```python
@pytest.fixture
def wide_volume(tmp_path):
    """Values spanning many magnitudes — where naive summation loses bits."""
    rng = np.random.default_rng(20260820)
    data = (rng.standard_normal((13, 9, 7)) * 10 ** rng.integers(-6, 7, (13, 9, 7))).astype(
        np.float64
    )
    path = tmp_path / "wide.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("vol", data=data)
    return str(path), data


def test_neumaier_sum_continues_across_calls():
    values = np.array([1e16, 1.0, -1e16, 1.0])
    whole = volumeio.neumaier_sum(values)
    state = volumeio.neumaier_sum(values[:2])
    part = volumeio.neumaier_sum(values[2:], state=state)
    assert whole == part
    assert whole[0] + whole[1] == 2.0  # the naive result would be 0.0


@pytest.mark.parametrize("divisor", [1, 2, 3, 7, 13, 1000])
def test_block_nansum_is_bit_identical_across_budgets(wide_volume, divisor):
    """The core guarantee: the memory budget must not change the answer."""
    path, data = wide_volume
    with h5py.File(path, "r") as f:
        reference = volumeio.block_nansum(f["vol"], budget_bytes=data.nbytes * 10)
        result = volumeio.block_nansum(f["vol"], budget_bytes=max(1, data.nbytes // divisor))
    assert result == reference  # exact equality, not approx


def test_block_nansum_ignores_nan(tmp_path):
    data = np.array([[[1.0, np.nan]], [[2.0, 4.0]]])
    path = tmp_path / "n.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("vol", data=data)
    with h5py.File(path, "r") as f:
        assert volumeio.block_nansum(f["vol"], budget_bytes=1) == 7.0


@pytest.mark.parametrize("divisor", [1, 4, 100])
def test_block_reduce_is_bit_identical_across_budgets(wide_volume, divisor):
    path, data = wide_volume
    with h5py.File(path, "r") as f:
        fn = lambda acc, block: max(acc, float(np.nanmax(block)))
        reference = volumeio.block_reduce(
            f["vol"], fn, budget_bytes=data.nbytes * 10, init=-np.inf
        )
        result = volumeio.block_reduce(
            f["vol"], fn, budget_bytes=max(1, data.nbytes // divisor), init=-np.inf
        )
    assert result == reference


@pytest.mark.parametrize("divisor", [1, 3, 50])
def test_two_pass_mean_subtraction_is_bit_identical_across_budgets(wide_volume, divisor):
    """The bucket-2 pattern: a global statistic, then a block-wise application."""
    path, data = wide_volume
    with h5py.File(path, "r") as f:

        def stat(acc, block):
            total, comp = volumeio.neumaier_sum(block.ravel(), state=acc[:2])
            return (total, comp, acc[2] + block.size)

        def apply(stat_value, block):
            total, comp, count = stat_value
            return block - ((total + comp) / count)

        budgets = (data.nbytes * 10, max(1, data.nbytes // divisor))
        outs = []
        for budget in budgets:
            blocks = list(
                volumeio.two_pass(f["vol"], stat, apply, budget_bytes=budget, init=(0.0, 0.0, 0))
            )
            outs.append(np.concatenate([b for _, b in blocks], axis=0))
    assert np.array_equal(outs[0], outs[1])  # bitwise, including NaN placement
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_common_volumeio.py -v -k "neumaier or nansum or reduce or two_pass"`
Expected: FAIL — `module 'dfxm.common.volumeio' has no attribute 'neumaier_sum'`.

- [ ] **Step 3: Write the implementation**

Append to `dfxm/common/volumeio.py`:

```python
def neumaier_sum(values, *, state: tuple[float, float] | None = None) -> tuple[float, float]:
    """Compensated sum of *values*, continuable across blocks.

    Returns ``(total, compensation)``; the true sum is ``total + compensation``.
    Pass a previous return value back as *state* to continue an accumulation —
    that is what makes the result independent of how the data was blocked.

    Why not ``np.sum``: numpy reduces pairwise with a 128-element base case, so
    summing an array whole and summing it in blocks give different bits. This
    walks elements in a fixed order with an explicit compensation term, so the
    answer depends only on the data — never on the memory budget.
    """
    total, comp = state if state is not None else (0.0, 0.0)
    for value in np.asarray(values, dtype=np.float64).ravel():
        item = float(value)
        tentative = total + item
        if abs(total) >= abs(item):
            comp += (total - tentative) + item
        else:
            comp += (item - tentative) + total
        total = tentative
    return total, comp


def block_reduce(dset, fn, *, budget_bytes: int, init):
    """Fold *dset* block-by-block with ``fn(acc, block) -> acc``.

    *fn* must be associative in the order blocks are produced (they always
    arrive in ascending order), so the result is budget-independent.
    """
    acc = init
    for _sl, block in iter_blocks(dset, budget_bytes=budget_bytes):
        acc = fn(acc, block)
    return acc


def block_nansum(dset, *, budget_bytes: int) -> float:
    """NaN-ignoring sum of *dset*, bit-identical for any budget.

    Differs from ``np.nansum`` by up to ~1 ulp — deliberately. Budget-
    independence is the property worth having; matching numpy's pairwise
    ordering is not.
    """

    def fold(acc, block):
        finite = block[np.isfinite(block)]
        return neumaier_sum(finite, state=acc)

    total, comp = block_reduce(dset, fold, budget_bytes=budget_bytes, init=(0.0, 0.0))
    return total + comp


def two_pass(dset, stat_fn, apply_fn, *, budget_bytes: int, init):
    """Global statistic first, then apply it block-wise.

    Pass 1 folds every block through ``stat_fn(acc, block) -> acc``. Pass 2
    yields ``(slice, apply_fn(stat, block))`` for each block. Costs one extra
    read of the dataset — the price of a lossless global operation in bounded
    memory.
    """
    stat = block_reduce(dset, stat_fn, budget_bytes=budget_bytes, init=init)
    for sl, block in iter_blocks(dset, budget_bytes=budget_bytes):
        yield sl, apply_fn(stat, block)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_common_volumeio.py -v`
Expected: PASS (all, including the parametrised budget sweeps).

- [ ] **Step 5: Confirm the guarantee is real, not an artefact of small test data**

```bash
python3 -c "
import numpy as np
from dfxm.common.volumeio import neumaier_sum
rng = np.random.default_rng(1)
a = (rng.standard_normal(100000) * 10.0 ** rng.integers(-8, 9, 100000))
whole = neumaier_sum(a)
state = (0.0, 0.0)
for i in range(0, a.size, 997):   # deliberately awkward block size
    state = neumaier_sum(a[i:i+997], state=state)
print('identical across blocking:', whole == state)
print('vs np.sum delta:', (whole[0]+whole[1]) - a.sum())
assert whole == state
"
```

Expected: `identical across blocking: True`, and a small non-zero delta versus `np.sum` — which is the documented, accepted difference.

- [ ] **Step 6: Document in Codebase.md**

```markdown
- **`volumeio.py`** — bounded-memory volume IO: `volume_bytes`, `iter_blocks`,
  `BlockReader`, `load_or_stream`, plus budget-independent reductions
  (`neumaier_sum`, `block_reduce`, `block_nansum`, `two_pass`). The guarantee is
  that any `budget_bytes` yields bit-identical results, so a laptop and the
  workstation produce the same data product. `block_nansum` therefore differs
  from `np.nansum` by ~1 ulp: numpy's pairwise ordering is budget-dependent and
  cannot provide that guarantee.
```

- [ ] **Step 7: Commit**

```bash
git add dfxm/common/volumeio.py tests/test_common_volumeio.py docs/Codebase.md
git commit -m "feat: budget-independent compensated reductions for chunked IO"
```

---

## Task 11: Disk-backed scratch arrays

**Files:**
- Modify: `dfxm/common/volumeio.py`, `docs/Codebase.md`
- Test: `tests/test_common_volumeio.py`

**Interfaces:**
- Consumes: `volume_bytes` (Task 9).
- Produces: `scratch_array(shape, dtype, *, dirpath, prefix="dfxm_scratch") -> ContextManager[np.memmap]`

This is what makes "nothing refuses for lack of RAM" true: bucket-3 work runs against a memmapped file instead of RAM.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_common_volumeio.py`:

```python
def test_scratch_array_is_writable_and_persists_within_the_block(tmp_path):
    with volumeio.scratch_array((4, 3), np.float64, dirpath=str(tmp_path)) as arr:
        arr[:] = 2.5
        arr[0, 0] = 7.0
        assert arr.shape == (4, 3)
        assert arr[0, 0] == 7.0
        assert arr[3, 2] == 2.5


def test_scratch_array_deletes_its_file_on_exit(tmp_path):
    with volumeio.scratch_array((4, 3), np.float64, dirpath=str(tmp_path)) as arr:
        path = arr.filename
        assert path is not None
        assert list(tmp_path.iterdir())
    assert not list(tmp_path.iterdir()), "scratch file outlived the context"


def test_scratch_array_deletes_its_file_on_exception(tmp_path):
    """A crash mid-run must not leave gigabytes of scratch behind."""
    with pytest.raises(RuntimeError):
        with volumeio.scratch_array((4, 3), np.float64, dirpath=str(tmp_path)) as arr:
            arr[:] = 1.0
            raise RuntimeError("simulated stage failure")
    assert not list(tmp_path.iterdir())


def test_scratch_array_behaves_like_an_ndarray(tmp_path):
    """Numerical code must not need to know it is disk-backed."""
    with volumeio.scratch_array((5,), np.float64, dirpath=str(tmp_path)) as arr:
        arr[:] = np.arange(5)
        np.multiply(arr, 2.0, out=arr)  # in-place: the bucket-3 discipline
        assert np.array_equal(np.asarray(arr), np.arange(5) * 2.0)
        assert float(np.nanmean(arr)) == 4.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_common_volumeio.py -v -k scratch`
Expected: FAIL — `module 'dfxm.common.volumeio' has no attribute 'scratch_array'`.

- [ ] **Step 3: Write the implementation**

Add to the imports at the top of `dfxm/common/volumeio.py` (keep isort order):

```python
import contextlib
import os
import tempfile
```

Append:

```python
@contextlib.contextmanager
def scratch_array(shape, dtype, *, dirpath: str, prefix: str = "dfxm_scratch"):
    """A disk-backed working array, deleted on exit even if the caller raises.

    Yields a ``np.memmap`` — an ``ndarray`` subclass, so numerical code needs no
    changes; its pages spill to disk instead of occupying RAM. This is what lets
    irreducibly whole-array work (an exact median, a global fit) run on a
    machine that cannot hold the volume: slower, but it finishes.

    Caveat the caller must respect: **temporaries still allocate in RAM**.
    ``out = a * b + c`` on a memmapped ``a`` materialises a full-size temporary
    and defeats the purpose. Bucket-3 code must operate in slabs with in-place
    operations (``np.multiply(a, b, out=a)``).
    """
    os.makedirs(dirpath, exist_ok=True)
    handle, path = tempfile.mkstemp(prefix=prefix, suffix=".dat", dir=dirpath)
    os.close(handle)
    memmap = None
    try:
        memmap = np.memmap(path, dtype=dtype, mode="w+", shape=tuple(shape))
        yield memmap
    finally:
        # Windows refuses to unlink a file while it is still mapped, so drop the
        # mapping before deleting. flush() then del is the documented way; the
        # gc call is what actually releases the handle on CPython/Windows.
        if memmap is not None:
            try:
                memmap.flush()
            except Exception:  # noqa: BLE001 - already-broken mapping
                pass
            if hasattr(memmap, "_mmap") and memmap._mmap is not None:
                try:
                    memmap._mmap.close()
                except Exception:  # noqa: BLE001
                    pass
            del memmap
            import gc

            gc.collect()
        try:
            os.unlink(path)
        except OSError:
            pass  # never mask the caller's exception with a cleanup failure
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_common_volumeio.py -v -k scratch`
Expected: PASS (4 tests).

- [ ] **Step 5: Run the full suite**

Run: `python3 -m pytest -q`
Expected: no regressions.

- [ ] **Step 6: Document in Codebase.md** — extend the `volumeio.py` entry with `scratch_array`, noting the Windows unmap-before-delete requirement and the RAM-temporaries caveat.

- [ ] **Step 7: Commit**

```bash
git add dfxm/common/volumeio.py tests/test_common_volumeio.py docs/Codebase.md
git commit -m "feat: disk-backed scratch arrays for irreducibly in-core work"
```

---

## Task 12: The equivalence harness

**Files:**
- Create: `tests/equivalence.py`
- Test: `tests/test_common_volumeio.py`
- Modify: `docs/Codebase.md`

**Interfaces:**
- Consumes: `volumeio` (Tasks 9–11).
- Produces: `assert_budget_independent(fn, dset, *, budgets=None)` — runs `fn(dset, budget_bytes=b)` for several budgets and asserts every result is bit-identical to the first. Handles ndarray, scalar and generator-of-blocks return types.

This is the harness every phase-5 conversion will use. Building it now — while there is nothing to convert — means phase 5 tasks are small and uniform.

- [ ] **Step 1: Write the harness**

```python
# tests/equivalence.py
"""Assert a chunked implementation is budget-independent (not a pytest file).

Every phase-5 conversion uses this: run the same callable at several memory
budgets and require bit-identical results. It is the executable form of the
guarantee that a laptop and the workstation produce the same data product.
"""

from __future__ import annotations

import numpy as np

DEFAULT_DIVISORS = (1, 2, 3, 7, 1000)


def _materialise(result):
    """Normalise a result to something comparable with array_equal."""
    if isinstance(result, np.ndarray):
        return result
    if isinstance(result, (int, float, np.floating, np.integer)):
        return np.asarray(result)
    # A generator/iterable of (slice, block) pairs, as two_pass yields.
    blocks = [block for _sl, block in result]
    return np.concatenate(blocks, axis=0) if blocks else np.asarray([])


def assert_budget_independent(fn, dset, *, budgets=None, nbytes=None) -> None:
    """Run *fn(dset, budget_bytes=b)* at several budgets; require identical bits.

    Compares with ``np.array_equal(..., equal_nan=True)`` so NaN placement is
    part of the guarantee — a chunked path that moves a NaN has changed the
    product just as surely as one that changes a number.
    """
    total = nbytes if nbytes is not None else dset.nbytes
    budgets = budgets or [max(1, int(total // d)) for d in DEFAULT_DIVISORS]
    reference = _materialise(fn(dset, budget_bytes=budgets[0]))
    for budget in budgets[1:]:
        candidate = _materialise(fn(dset, budget_bytes=budget))
        assert candidate.shape == reference.shape, (
            f"budget {budget}: shape {candidate.shape} != {reference.shape}"
        )
        assert np.array_equal(candidate, reference, equal_nan=True), (
            f"budget {budget} produced different bits than budget {budgets[0]}"
        )
```

- [ ] **Step 2: Write a test that exercises the harness both ways**

Append to `tests/test_common_volumeio.py`:

```python
def test_harness_passes_a_budget_independent_function(wide_volume):
    from tests.equivalence import assert_budget_independent

    path, data = wide_volume
    with h5py.File(path, "r") as f:
        assert_budget_independent(
            lambda d, budget_bytes: volumeio.block_nansum(d, budget_bytes=budget_bytes),
            f["vol"],
            nbytes=data.nbytes,
        )


def test_harness_catches_a_budget_dependent_function(wide_volume):
    """The harness is only worth having if it actually fails on a bad impl."""
    from tests.equivalence import assert_budget_independent

    path, data = wide_volume

    def naive_sum(dset, *, budget_bytes):
        # np.sum per block then adding up: pairwise ordering varies with budget
        return float(sum(float(np.nansum(b)) for _sl, b in volumeio.iter_blocks(
            dset, budget_bytes=budget_bytes
        )))

    with h5py.File(path, "r") as f:
        with pytest.raises(AssertionError, match="different bits"):
            assert_budget_independent(naive_sum, f["vol"], nbytes=data.nbytes)
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_common_volumeio.py -v -k harness`
Expected: PASS (2 tests). If `test_harness_catches_a_budget_dependent_function` fails, the test data does not span enough magnitudes for naive summation to differ — widen the exponent range in the `wide_volume` fixture until it does. A harness that cannot detect a bad implementation is worthless.

- [ ] **Step 4: Run the full suite**

Run: `python3 -m pytest -q`
Expected: all passing.

- [ ] **Step 5: Verify the Qt-free invariant held throughout**

```bash
python3 -c "
import sys
import dfxm.common.advice, dfxm.common.machine, dfxm.common.volumeio
bad = [m for m in ('PySide6', 'pyvista', 'vtk', 'vtkmodules') if m in sys.modules]
print('Qt/3-D leaked into core:', bad)
assert not bad, bad
print('dfxm/ is still Qt-free')
"
```

Expected: `dfxm/ is still Qt-free`.

- [ ] **Step 6: Document in Codebase.md** — add `tests/equivalence.py` beside the existing note about `tests/gui_smoke.py` not being a pytest file.

- [ ] **Step 7: Commit**

```bash
git add tests/equivalence.py tests/test_common_volumeio.py docs/Codebase.md
git commit -m "test: budget-independence harness for chunked implementations"
```

---

## Done criteria for phases 1–4

- [ ] `python3 -m pytest -q` passes with no regressions against the 1064-test baseline.
- [ ] `ruff check . && ruff format --check .` clean.
- [ ] `dfxm/` imports no Qt and no pyvista at module level (Task 12, Step 5).
- [ ] `python3 -m gui.app` still launches and runs a stage — phases 1–4 must be invisible.
- [ ] The STO2 numbers from Task 8, Step 7 are recorded, so phase 5 can be scoped on measurements rather than guesses.
