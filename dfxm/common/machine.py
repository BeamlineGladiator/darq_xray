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
