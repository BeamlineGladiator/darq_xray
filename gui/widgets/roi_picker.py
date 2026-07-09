"""Interactive ROI picker: drag a rectangle on a preview plane, get pixel bounds.

Source-agnostic and dumb — it imports nothing from ``dfxm`` beyond numpy-level
types. Each call site supplies previews as ``(label, thunk)`` pairs where
``thunk() -> (array2d, sx, sy)`` (lazy). The plane is drawn exactly like the
map exports (``origin="lower"``, physical aspect via ``set_aspect(sy/sx)``), so
the crop you draw is the crop you get. On accept, :attr:`result` is the
half-open ``(r0, r1, c0, c1)`` pixel-index tuple; otherwise ``None``.
"""

from __future__ import annotations

import math
from typing import Callable  # noqa: F401  — used by ROIPickerDialog (Task 4)

import numpy as np  # noqa: F401  — used by ROIPickerDialog (Task 4)


def rect_to_indices(xmin, xmax, ymin, ymax, w, h) -> tuple[int, int, int, int]:
    """Map a selector rectangle (data coords on pixel-edge extents) to half-open
    ``(r0, r1, c0, c1)`` pixel indices, clamped to ``[0, w]`` / ``[0, h]``.

    ``x`` is columns (X), ``y`` is rows (Y). floor(min)/ceil(max) on pixel-edge
    extents gives inclusive-of-touched-pixels behaviour with no ±0.5 fencepost.
    """
    c0 = max(0, min(int(math.floor(min(xmin, xmax))), w))
    c1 = max(0, min(int(math.ceil(max(xmin, xmax))), w))
    r0 = max(0, min(int(math.floor(min(ymin, ymax))), h))
    r1 = max(0, min(int(math.ceil(max(ymin, ymax))), h))
    return r0, r1, c0, c1
