"""Qt-free ETA estimation for progress readouts.

Shared by the GUI's busy overlay/progress text (darq_xray/gui/widgets/busy.py,
darq_xray/gui/stage_view.py) but importable and testable without Qt.
"""

from __future__ import annotations

import time

_MIN_FRAC = 0.05  # below this, any extrapolation is noise
_MIN_ELAPSED_S = 2.0
_EMA_KEEP = 0.7  # weight of the previous smoothed estimate

# The rate is measured over a trailing window rather than the whole run, so a
# stage that changes pace late is re-estimated at its CURRENT pace. This matters
# here because pipeline stages are built of phases with wildly different
# per-unit costs — a fast per-layer loop followed by one slow whole-volume
# alignment — and the whole-run average keeps quoting the fast phase long after
# it ended. Long enough to span several samples from a slow reporter; short
# enough that a phase change is reflected within a few of them.
_RATE_WINDOW_S = 30.0


def _format_remaining(remaining_s: float) -> str:
    if remaining_s >= 90.0:
        return f"~{max(1, int(round(remaining_s / 60.0)))} min left"
    return f"~{max(1, int(round(remaining_s)))} s left"


def format_eta(elapsed_s: float, frac: float) -> str:
    """Human remaining-time estimate; "" when too early/noisy to say."""
    if frac < _MIN_FRAC or elapsed_s < _MIN_ELAPSED_S or frac >= 1.0:
        return ""
    return _format_remaining(elapsed_s * (1.0 - frac) / frac)


class EtaEstimator:
    """Smoothed remaining-time estimate from monotonic (t, frac) samples.

    ``update(frac)`` records a progress sample against the injected *clock*
    (``time.monotonic`` by default; tests inject a fake). Fractions are
    clamped to [0, 1] and a regressing fraction is ignored, so a jittery
    reporter can never produce a negative or exploding estimate. Estimates
    are EMA-smoothed; ``eta_text()`` stays "" until >= 5 % done and >= 2 s
    elapsed (mirroring :func:`format_eta`).

    Three properties the first version did not have, each met on a real run and
    each pinned by a test in ``tests/test_common_eta.py``:

    * **A repeated fraction is not progress.** Stages re-emit the same *frac*
      with a new message (``visualize`` reports ``progress(0.6, ...)`` again
      for a clim note while the entire strain half runs). Recomputing
      ``elapsed * (1 - frac) / frac`` against a growing *elapsed* made the ETA
      count UP. Only a fraction that actually advanced updates the estimate.
    * **The estimate counts down between samples.** ``eta_text()`` subtracts
      the wall-clock since the estimate was made, so a stage that reports
      nothing for four minutes shows a shrinking number rather than a frozen
      one — and once it has overrun its own estimate it returns "" rather than
      standing at "~1 s left", because at that point it does not know.
    * **The rate is local.** See :data:`_RATE_WINDOW_S`.

    What this module cannot fix is *frac* itself, and it used to have to live
    with a bad one: stages assigned it by milestone (0.05 loading, 0.6 halfway,
    1.0 done) rather than in proportion to work, and a signal that sits at 0.6
    for 40 % of the run cannot be extrapolated accurately by anything. That was
    always a stage-side defect and it was fixed stage by stage — every stage now
    reports per layer, volume, piece, plane, frame or job through
    :mod:`darq_xray.common.progress`, with `tests/progress_trace.py` holding the
    largest step any of them may take. The clamping in ``update`` stays because
    the contract, not the current callers, is what it defends.
    """

    def __init__(self, clock=time.monotonic) -> None:
        self._clock = clock
        self.reset()

    def reset(self) -> None:
        now = self._clock()
        self._t0 = now
        self._frac = 0.0
        # (t, frac) samples, oldest first, pruned to the trailing rate window.
        self._samples: list[tuple[float, float]] = [(now, 0.0)]
        self._smoothed: float | None = None
        self._smoothed_at: float | None = None

    def update(self, frac: float) -> None:
        frac = min(1.0, max(0.0, float(frac)))
        # `<=`, not `<`: a repeated fraction carries no new information about
        # the rate, and treating it as a sample is what made the ETA grow.
        if frac <= self._frac:
            return
        self._frac = frac
        now = self._clock()
        self._samples.append((now, frac))
        # Keep the newest sample at or before the cutoff as the baseline, so
        # the window always spans at least `_RATE_WINDOW_S` once it can.
        cutoff = now - _RATE_WINDOW_S
        while len(self._samples) > 2 and self._samples[1][0] <= cutoff:
            self._samples.pop(0)

        if frac < _MIN_FRAC or now - self._t0 < _MIN_ELAPSED_S or frac >= 1.0:
            return
        t_base, frac_base = self._samples[0]
        d_t, d_frac = now - t_base, frac - frac_base
        if d_t <= 0 or d_frac <= 0:
            return
        raw = (1.0 - frac) * d_t / d_frac
        self._smoothed = (
            raw if self._smoothed is None else _EMA_KEEP * self._smoothed + (1.0 - _EMA_KEEP) * raw
        )
        self._smoothed_at = now

    def eta_text(self) -> str:
        if self._smoothed is None or self._smoothed_at is None or self._frac >= 1.0:
            return ""
        remaining = self._smoothed - (self._clock() - self._smoothed_at)
        if remaining <= 0.0:
            return ""  # overrun: no number is honest, a floored one is not
        return _format_remaining(remaining)
