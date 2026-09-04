"""GUI-side plumbing for the cost advisory: a cached profile and a worker.

Computes no policy. `darq_xray/common/advisory.py` decides what the user is told;
this module only decides *when* and *on which thread*.
"""

from __future__ import annotations

import time

from PySide6.QtCore import QObject, QThread, QTimer, Signal

from darq_xray.common import machine
from darq_xray.common.advisory import Advisory, advise_stage, disk_probe_dir
from darq_xray.config.models import StageSpec

from .widgets.busy import keep_alive

# Short enough that the status bar tracks a filling disk, long enough that four
# surfaces plus a 5 s timer cost one probe rather than five.
_PROFILE_TTL_S = 5.0

# `machine.probe_gl`'s own default (120 s) exists so a slow-but-alive probe on
# unusual hardware still finishes; that is too long to sit on a GUI teardown
# or shutdown path. `_GlProbeWorker` is pinned via `keep_alive` and joined (no
# timeout) by `MainWindow.closeEvent`'s `wait_for_workers()`, so a hanging
# driver would otherwise make closing the window look frozen for up to two
# minutes. 15 s is generous for a real probe (context creation + one child
# process spawn) while keeping that worst case bounded to something a user
# will wait out rather than force-kill. `SystemCheckDialog._measure` reuses
# the same figure for the same reason on its own (modal) probe.
GL_PROBE_TIMEOUT_S = 15.0

_cache: dict[str, tuple[float, machine.MachineProfile]] = {}


def cached_profile(output_dir: str) -> machine.MachineProfile:
    """A recent :class:`MachineProfile` for *output_dir*'s filesystem.

    **Never probes GL** on its own — ``probe_gl_now=_gl_ready`` is False until
    the one-shot background probe (or the System check dialog) has already
    succeeded, so this is always a cache lookup here, never a spawn.
    """
    now = time.monotonic()
    hit = _cache.get(output_dir)
    if hit is not None and now - hit[0] < _PROFILE_TTL_S:
        return hit[1]
    prof = machine.profile(output_dir=output_dir, probe_gl_now=_gl_ready)
    _cache[output_dir] = (now, prof)
    return prof


def clear_profile_cache() -> None:
    """Drop every cached profile (tests, and a forced re-probe)."""
    _cache.clear()


_gl_ready = False


def gl_ready() -> bool:
    """True once a GL probe has succeeded in this session."""
    return _gl_ready


def _set_gl_ready(value: bool) -> None:
    """Test seam; production code sets this through `probe_gl_async`."""
    global _gl_ready
    _gl_ready = value


class _GlProbeWorker(QThread):
    """The one GL probe this session pays for. Result is cached on disk by
    `machine.probe_gl`, so every later launch is a file read."""

    finished_ok = Signal(bool)

    def run(self) -> None:  # worker thread
        try:
            _info, status = machine.probe_gl(timeout=GL_PROBE_TIMEOUT_S)
        except Exception:  # noqa: BLE001 — a dead probe is a result, not a crash
            status = "crashed"
        self.finished_ok.emit(status == "ok")


_gl_worker: _GlProbeWorker | None = None


def probe_gl_async() -> None:
    """Probe GL once per session, off the GUI thread.

    Costs a child process, so it is never triggered by the cost path. Once it
    lands, `cached_profile` starts including GL — `machine.probe_gl` memoises
    in-process and caches on disk, so that is a lookup, not another child.
    """
    global _gl_worker
    if _gl_ready or _gl_worker is not None:
        return

    def _done(ok: bool) -> None:
        global _gl_worker
        _gl_worker = None
        if ok:
            _set_gl_ready(True)
            clear_profile_cache()  # so the next profile carries GL

    worker = _GlProbeWorker()
    worker.finished_ok.connect(_done)
    _gl_worker = worker
    keep_alive(worker)
    worker.start()


class _AdvisoryWorker(QThread):
    """One `advise_stage` call off the GUI thread. Emits `done(Advisory|None)`."""

    done = Signal(object)

    def __init__(self, spec: StageSpec, params: dict) -> None:
        super().__init__()
        self._spec = spec
        self._params = params

    def run(self) -> None:  # worker thread — no Qt widgets in here
        try:
            probe_dir = disk_probe_dir(self._spec, self._params)
            result = advise_stage(self._spec, self._params, profile=cached_profile(probe_dir))
        except Exception:  # noqa: BLE001 — advise_stage promises not to, but a
            result = None  # dead worker must not take the window with it
        self.done.emit(result)


class StageAdvisor(QObject):
    """Debounced, latest-wins advisories for one stage form.

    Off the GUI thread because the estimators do real IO: `sum_dataset_bytes`
    opens every candidate HDF5 file and is not memoised (only the motor read
    is), so a synchronous call per keystroke stutters the form on network or
    external storage.
    """

    advisoryReady = Signal(object)  # Advisory

    def __init__(self, spec: StageSpec, values_fn, parent=None, debounce_ms: int = 400) -> None:
        super().__init__(parent)
        self._spec = spec
        self._values_fn = values_fn
        self._worker: _AdvisoryWorker | None = None
        self._pending = False
        self.latest: Advisory | None = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(debounce_ms)
        self._timer.timeout.connect(self._start)

    def request(self) -> None:
        """Ask for a fresh advisory once the form stops changing."""
        self._timer.start()

    def compute_blocking(self) -> Advisory:
        """Compute one now, on the calling thread. For the Run click only."""
        params = self._values_fn()
        result = advise_stage(
            self._spec, params, profile=cached_profile(disk_probe_dir(self._spec, params))
        )
        self.latest = result
        return result

    def _start(self) -> None:
        if self._worker is not None:
            self._pending = True  # latest-wins: one re-run after this one lands
            return
        worker = _AdvisoryWorker(self._spec, self._values_fn())
        worker.done.connect(self._on_done)
        self._worker = worker
        keep_alive(worker)
        worker.start()

    def _on_done(self, result) -> None:
        self._worker = None
        if result is not None:
            self.latest = result
            self.advisoryReady.emit(result)
        if self._pending:
            self._pending = False
            self._start()
