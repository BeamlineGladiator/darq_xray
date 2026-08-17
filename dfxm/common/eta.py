"""Qt-free ETA estimation for progress readouts.

Shared by the GUI's busy overlay/progress text (gui/widgets/busy.py,
gui/stage_view.py) but importable and testable without Qt.
"""

from __future__ import annotations

import time

_MIN_FRAC = 0.05  # below this, any extrapolation is noise
_MIN_ELAPSED_S = 2.0
_EMA_KEEP = 0.7  # weight of the previous smoothed estimate


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
    """

    def __init__(self, clock=time.monotonic) -> None:
        self._clock = clock
        self.reset()

    def reset(self) -> None:
        self._t0 = self._clock()
        self._frac = 0.0
        self._smoothed: float | None = None

    def update(self, frac: float) -> None:
        frac = min(1.0, max(0.0, float(frac)))
        if frac < self._frac:
            return
        self._frac = frac
        elapsed = self._clock() - self._t0
        if frac < _MIN_FRAC or elapsed < _MIN_ELAPSED_S or frac >= 1.0:
            return
        raw = elapsed * (1.0 - frac) / frac
        self._smoothed = (
            raw if self._smoothed is None else _EMA_KEEP * self._smoothed + (1.0 - _EMA_KEEP) * raw
        )

    def eta_text(self) -> str:
        if self._smoothed is None or self._frac >= 1.0:
            return ""
        return _format_remaining(self._smoothed)
