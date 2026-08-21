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
    peak is a lower bound, bounded by this).

    *timeout* bounds the **run**, not the call. It is the ceiling on how long
    the child is allowed to be working; on top of it this function may spend up
    to `_DRAIN_GRACE` (5 s) waiting for the terminal message and a further 5 s
    in `join()`, so the worst-case wall clock is roughly ``timeout + 10 s``.
    A caller sizing a CI budget must budget for that: ``timeout=600`` can take
    610 s to return.

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
        # Sound only because the loop above never breaks out on a live child:
        # `StageRunner.cancel()` sets `finished` itself, so the `finally`'s
        # cancel would make this guard vacuous for any exit path that leaves
        # the child running. Anyone adding such a path must check
        # `runner.failure is None and runner.result is None` instead.
        raise RuntimeError(f"{target} exited without producing a result")
    if samples == 0:
        raise RuntimeError(
            f"{target}: not one RSS sample was read (child pid {runner.pid}) — "
            "the measurement is dead, not zero"
        )
    if peak < MIN_PLAUSIBLE_RSS:
        raise RuntimeError(
            f"{target}: peak RSS {peak} B over {samples} samples is below the "
            f"{MIN_PLAUSIBLE_RSS} B floor no CPython child settles under. Either "
            "the sampler is not reading the child, or the child finished so fast "
            f"that every sample caught spawn's pre-import RSS (interval={interval}s, "
            f"{samples} sample(s)) — in which case lower `interval`. Refusing to "
            "report the figure either way: it is not this stage's peak."
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


# -----------------------------------------------------------------------------
# Pinning a stage's RSS floor
# -----------------------------------------------------------------------------
# A stage that streams hands `advice.working_set_budget_bytes` its own
# `rss_floor_bytes` — the resident cost of its process image before it touches
# data, which `tracemalloc` cannot see. That constant is measured once and then
# written down, so nothing keeps it honest as VTK, numpy or the interpreter
# move underneath it. Declaring it too LOW is the dangerous direction: it
# over-states what the machine can afford and the budget comes out too large.
#
# `assert_floor_covers` is the check, and it is meant to be **copied verbatim**
# by every stage that declares a floor. One call, in that stage's test module:
#
#     def test_rss_floor_covers_the_measured_process_image(tmp_path):
#         from tests.peak_rss import assert_floor_covers
#         params = _trivial_params(tmp_path)          # smallest real export
#         assert_floor_covers(
#             MY_STAGE.RSS_FLOOR_BYTES,
#             "dfxm.stages.my_stage:run",
#             params,
#             data_bytes=<bytes of that input>,
#         )
#
# Do not copy another stage's *number*. The floor is per stage by construction —
# a VTK-importing stage sits hundreds of MB above one that only needs matplotlib
# — so each stage measures its own and this test is what proves it did.

# How far a declared floor may sit ABOVE the measured one before the slack is
# more likely to be a stale copy from another stage than deliberate headroom.
_FLOOR_SLACK_LIMIT = 2.5

# The trivial input must be small enough that the data is noise against the
# process image, or the "floor" measured is really a floor plus a volume.
_FLOOR_DATA_FRACTION = 0.05


def measure_process_floor(target: str, params: dict, *, samples: int = 3, **kwargs) -> int:
    """Peak RSS of *target* on input small enough for the data to be noise.

    The largest of *samples* runs, not the mean: this feeds a
    ``floor >= measured`` assertion, so the pessimistic reading is the one that
    keeps the assertion meaningful. Each run is a fresh `spawn` child, so the
    spread across samples is allocator and page-cache noise (measured ±0.1 MiB
    for `paraview`), not a trend.
    """
    peaks = [measure_peak_rss(target, params, **kwargs)[1] for _ in range(max(1, int(samples)))]
    return max(peaks)


def assert_floor_covers(
    floor_bytes: int, target: str, params: dict, *, data_bytes: int, **kwargs
) -> int:
    """A stage's declared RSS floor must not sit below its measured process image.

    *data_bytes* is how much input the trivial run actually reads; it is
    required so this cannot silently become a comparison against a floor that
    includes a volume. Returns the measured figure, so a caller can print or
    further assert on it.

    Fails in **both** directions, deliberately. Too low is unsafe (the budget
    comes out too large). Too high by more than
    :data:`_FLOOR_SLACK_LIMIT`x is the signature of a number copied from another
    stage rather than measured for this one — the failure mode that makes a
    per-stage constant quietly universal again.
    """
    measured = measure_process_floor(target, params, **kwargs)
    if data_bytes > _FLOOR_DATA_FRACTION * measured:
        # Assert the precondition rather than letting the comparison decide: on
        # a large input this function measures a whole export and the assertion
        # below would pass while meaning nothing.
        raise ValueError(
            f"{target}: the floor probe read {data_bytes / (1 << 20):.1f} MiB of input "
            f"against a {measured / (1 << 20):.1f} MiB peak — that is not a process "
            "image, it is an export. Shrink the fixture."
        )
    assert floor_bytes >= measured, (
        f"{target}: declared RSS floor {floor_bytes / (1 << 20):.1f} MiB is BELOW the "
        f"measured process image {measured / (1 << 20):.1f} MiB. The budget derived from "
        "it will be too large. Re-measure and raise the constant."
    )
    assert floor_bytes <= _FLOOR_SLACK_LIMIT * measured, (
        f"{target}: declared RSS floor {floor_bytes / (1 << 20):.1f} MiB is more than "
        f"{_FLOOR_SLACK_LIMIT}x the measured {measured / (1 << 20):.1f} MiB. Deliberate "
        "headroom is expected; this much suggests a number copied from another stage "
        "instead of measured for this one."
    )
    return measured
