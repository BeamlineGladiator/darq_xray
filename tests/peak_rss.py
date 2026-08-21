"""Measure a stage's peak resident memory (not a pytest file).

`budget_bytes` proves the streaming code path runs. It does not prove the peak
dropped: a stage can stream its read and then materialise a float64 copy
anyway, which is exactly the class of mistake the phase-1-4 estimators made
(they under-predicted real peak RSS by ~1.66x on the real dataset). This
samples the real child process, so it measures the thing rather than the model
of it. `dfxm/runner.py` already spawns every stage in a child, so there is
nothing to build here — only to watch.

Two things this module is **not**, both of which the phase-5 conversions must
keep straight:

* **`tracemalloc` is not RSS.** `dfxm/common/alignment.py`'s working-set model
  is expressed in Python-level allocations; every figure here is resident set
  size, which additionally carries the interpreter, the extension modules,
  h5py's chunk cache, a memmap's resident pages and allocator fragmentation.
  The two quantities do not convert into one another, and a budget expressed
  in one must not be compared against a measurement in the other.
* **A sampled peak is a lower bound.** RSS is read once per `interval`, so an
  allocation that lives for less than one interval can be missed entirely and
  the reported peak can understate the truth. That direction is the safe one
  for a pass/fail assertion — a stage that passes may be better than measured,
  never worse — but a *failing* figure is real and a passing one is a floor,
  not a certificate.

Scope: it watches the one child `StageRunner` starts, not that child's own
descendants (nothing in this pipeline forks below the stage except `ffmpeg`
during a video export).
"""

from __future__ import annotations

import time

import psutil

from dfxm.runner import StageRunner

# A CPython child under `spawn` re-imports the interpreter and the stage's
# modules before it touches any data; its RSS is tens of MiB by the time the
# first sample lands. A figure below this floor therefore does not mean "a
# very frugal stage" — it means the sampler read something other than the
# child (a stale handle, a stubbed psutil, an accumulator left at zero), which
# is the silent-no-op failure this harness exists to make loud.
MIN_PLAUSIBLE_RSS = 4 << 20  # 4 MiB

# How long to keep draining the queue after the child exits, waiting for the
# Done/Failed message it flushed on its way out.
_DRAIN_GRACE = 5.0


# -----------------------------------------------------------------------------
# Targets for testing the harness itself (referenced as "tests.peak_rss:name")
# -----------------------------------------------------------------------------
def _sleepy_target(params: dict, progress=None):
    """A do-nothing stage, for testing the harness itself."""
    time.sleep(float(params.get("seconds", 0.1)))
    return {"ok": True}


def _hungry_target(params: dict, progress=None):
    """Allocate roughly `mib` MiB and hold it, for testing the harness itself.

    `np.ones` rather than `np.empty` on purpose: RSS counts *resident* pages,
    and only writing to a page makes it resident. `np.empty` would allocate
    address space the sampler could not see, which would make this target a
    poor control for a harness whose whole job is to see allocations.
    """
    import numpy as np

    block = np.ones(int(float(params.get("mib", 64)) * (1 << 20) // 8), dtype=np.float64)
    time.sleep(float(params.get("hold", 0.3)))  # >> one sampling interval
    return {"sum": float(block[0])}


def _boom_target(params: dict, progress=None):
    """A stage that raises, for testing that failures surface as RuntimeError."""
    raise RuntimeError("boom")


# -----------------------------------------------------------------------------
# Sampling
# -----------------------------------------------------------------------------
def _attach(pid: int | None) -> psutil.Process | None:
    """A psutil handle on *pid*, or None if it is gone already/never existed."""
    if pid is None:
        return None
    try:
        return psutil.Process(pid)
    except psutil.Error:
        return None


def _sample_rss(proc: psutil.Process | None) -> int | None:
    """One RSS reading in bytes, or None when the process cannot be read.

    The seam the harness's own tests stub to simulate a dead measurement.
    """
    if proc is None:
        return None
    try:
        return int(proc.memory_info().rss)
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return None


def _drain_to_finish(runner: StageRunner, grace: float = _DRAIN_GRACE) -> None:
    """Poll for the terminal message the child flushed as it exited."""
    deadline = time.monotonic() + grace
    while not runner.finished and time.monotonic() < deadline:
        runner.poll()
        if runner.finished:
            return
        time.sleep(0.005)


def measure_peak_rss(
    target: str, params: dict, *, interval: float = 0.02, timeout: float = 300.0
) -> tuple[object, int]:
    """Run *target* in a child and return ``(result, peak_rss_bytes)``.

    *interval* is the sampling period in seconds (see the module docstring: the
    peak is a lower bound, bounded by this). *timeout* is a wall-clock ceiling
    on the whole run.

    Raises :class:`RuntimeError` if the stage fails, times out, exits without a
    result, or — the case that matters most — if the measurement itself was
    dead: no sample was taken, or the peak came back below
    :data:`MIN_PLAUSIBLE_RSS`. A harness that quietly returns 0 would pass
    every memory assertion built on it, so 0 is an error, never an answer.
    """
    runner = StageRunner(target, params)
    runner.start()
    proc = _attach(runner.pid)
    peak = 0
    samples = 0
    deadline = time.monotonic() + timeout
    try:
        while True:
            rss = _sample_rss(proc)
            if rss is not None:
                samples += 1
                peak = max(peak, rss)
            runner.poll()
            if runner.finished or not runner.is_alive():
                break
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"{target} did not finish within {timeout}s "
                    f"(peak so far {peak / (1 << 20):.1f} MiB)"
                )
            time.sleep(interval)
        _drain_to_finish(runner)
    finally:
        if runner.is_alive():
            runner.cancel()
        runner.join(timeout=5.0)
    runner.poll()

    if runner.failure is not None:
        raise RuntimeError(f"{target} failed: {runner.failure.error}\n{runner.failure.traceback}")
    if not runner.finished:
        raise RuntimeError(f"{target} exited without producing a result")
    if samples == 0:
        raise RuntimeError(
            f"{target}: not one RSS sample was read (child pid {runner.pid}) — "
            "the measurement is dead, not zero"
        )
    if peak < MIN_PLAUSIBLE_RSS:
        raise RuntimeError(
            f"{target}: peak RSS {peak} B over {samples} samples is below the "
            f"{MIN_PLAUSIBLE_RSS} B floor no CPython child can be under — the "
            "sampler is not reading the child"
        )
    return runner.result, peak


def assert_peak_under(target: str, params: dict, limit_bytes: int, **kwargs) -> object:
    """Run *target* and require its peak RSS stayed under *limit_bytes*.

    Returns the stage result, so a caller can assert on the product as well as
    on the peak. Raises (never passes) when the measurement was dead — see
    :func:`measure_peak_rss`.
    """
    if limit_bytes < MIN_PLAUSIBLE_RSS:
        # Assert the precondition instead of letting the comparison decide: a
        # limit under the floor cannot be met by any child, so a "failure"
        # against it would say nothing about the stage.
        raise ValueError(
            f"limit_bytes={limit_bytes} is below the {MIN_PLAUSIBLE_RSS} B floor "
            "of a bare CPython child; no stage can meet it"
        )
    result, peak = measure_peak_rss(target, params, **kwargs)
    assert peak <= limit_bytes, (
        f"{target} peaked at {peak / (1 << 20):.1f} MiB, over the "
        f"{limit_bytes / (1 << 20):.1f} MiB limit"
    )
    return result
