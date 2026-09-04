"""What kind of machine are we on? (Qt-free.)

Builds a frozen :class:`MachineProfile` describing CPU, RAM, disk, the OpenGL
stack and ffmpeg. Consumers (:mod:`darq_xray.common.advice`, the GUI's System check)
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

import json
import os
import platform
import shutil
import subprocess
import sys
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


def _read_meminfo() -> str:
    """``/proc/meminfo``'s text, or "" where there is no such file."""
    try:
        with open("/proc/meminfo", encoding="ascii", errors="replace") as fh:
            return fh.read()
    except OSError:  # not Linux, or /proc is not mounted
        return ""


def _parse_meminfo(text: str) -> tuple[int, int]:
    """(total, available) bytes from ``/proc/meminfo`` *text*; (0, 0) if unusable.

    ``MemAvailable`` is the kernel's own estimate of what a new allocation can
    obtain without swapping — it counts the reclaimable page cache, which
    ``MemFree`` does not. It is the number ``free``, ``btop`` and
    ``psutil.virtual_memory().available`` all report, so reading it here is what
    makes the psutil-free path agree with the psutil one.

    Absent (kernels before 3.14) means "this file cannot answer", reported as
    (0, 0) so the caller falls back rather than substituting ``MemFree`` — the
    substitution is the whole bug this replaces.
    """
    fields: dict[str, int] = {}
    for line in text.splitlines():
        key, sep, rest = line.partition(":")
        if not sep:
            continue
        parts = rest.split()
        if not parts:
            continue
        try:
            value = int(parts[0])
        except ValueError:
            continue
        # Every size line is "kB"; the handful of count lines (HugePages_*)
        # carry no unit and are not ones we read.
        fields[key.strip()] = value * 1024 if len(parts) > 1 else value
    total, available = fields.get("MemTotal", 0), fields.get("MemAvailable", 0)
    return (total, available) if total > 0 and available > 0 else (0, 0)


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
    from_meminfo = _parse_meminfo(_read_meminfo())
    if from_meminfo != (0, 0):
        return from_meminfo
    page = os.sysconf("SC_PAGE_SIZE")
    total = page * os.sysconf("SC_PHYS_PAGES")
    # Last resort only. SC_AVPHYS_PAGES is MemFree: it excludes the reclaimable
    # page cache, so on a machine that has been reading large volumes it
    # understates what is obtainable by orders of magnitude (8.4 GB against a
    # real 467.7 GB, measured). Reached only where /proc/meminfo cannot answer,
    # where a pessimistic number still beats no number.
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
        os.makedirs(os.path.dirname(gl_cache_path()), exist_ok=True)
        with open(gl_cache_path(), "w", encoding="utf-8") as fh:
            json.dump({"key": key, "result": result}, fh)
    except Exception:  # noqa: BLE001 - an unwritable cache is not an error
        pass


def _run_gl_child(timeout: float) -> dict | None:
    """Run the probe child. None means it died, hung or spoke nonsense."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "darq_xray.common._glprobe"],
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


def invalidate_gl_cache() -> None:
    """Discard every remembered GL answer so the next probe reaches a fresh
    child process.

    Clears the in-process memo (``_GL_MEMO``) and removes the on-disk cache
    file (``gl_cache_path()``, if it exists). Without this, ``probe_gl(...,
    use_cache=True)`` — which :func:`profile` always uses internally — keeps
    returning whatever this process (or a previous run) already measured, no
    matter how many times it is called again. This is the one function a
    caller that genuinely wants a fresh measurement (the System check
    dialog's Re-probe button) should reach for; a lone ``probe_gl(...,
    use_cache=False)`` call is not enough on its own, since its result is
    never written back for the *next* ``probe_gl`` call to see.
    """
    _GL_MEMO.clear()
    try:
        os.remove(gl_cache_path())
    except OSError:
        pass


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

    gl_info, gl_status = (None, "unprobed")
    if probe_gl_now:
        gl_info, gl_status = _try("gl", lambda: probe_gl(timeout=gl_timeout), (None, "crashed"))

    return MachineProfile(
        os_name=platform.system() or "unknown",
        cpu_logical=cpu_logical,
        cpu_physical=cpu_physical,
        ram_total=ram_total,
        ram_available=ram_available,
        disk_free=disk_free,
        gl=gl_info,
        gl_status=gl_status,
        ffmpeg=ffmpeg,
        probe_errors=tuple(errors),
    )
