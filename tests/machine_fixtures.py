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
