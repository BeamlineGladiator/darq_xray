"""Effective detector pixel size from a raw (pre-darfix) scan's motors.

Qt-free. Reads the far-field geometry motors from the first BLISS entry of a
raw scan and turns them into the physical detector pixel size (micrometres per
pixel) that the strain/mosaicity maps are scaled by.

Geometry (lens-maker):
    M   = mainx / obx - 1                 (CRL magnification)
    E_x = base / M                        (horizontal pixel size, um)
    2th = atan2(ffz, mainx)               (detector angle)
    E_y = E_x / sin(2th)  if condenser in, else E_x

``base`` is the far-field camera pixel (6.5 um) divided by the objective
magnification, selected from the ``ffsel`` motor:
    ffsel = -60 -> 2x  (base 3.25)
    ffsel =   0 -> 10x (base 0.65)
The condenser is detected from ``lenssel`` (0 -> in).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import h5py
import numpy as np

from .errors import StageUserError
from .h5io import get_filtered_entries, read_positioners

DEFAULT_POSITIONERS_PATH = "instrument/positioners"

# Positioner motor names (ID03 far-field geometry).
MAINX = "mainx"
OBX = "obx"
FFSEL = "ffsel"
FFZ = "ffz"
LENSSEL = "lenssel"

# ffsel value -> (objective label, base detector pixel in um).
_OBJECTIVES: dict[float, tuple[str, float]] = {
    -60.0: ("2x", 3.25),
    0.0: ("10x", 0.65),
}
_FFSEL_TOL = 1.0  # ffsel motor tolerance when matching an objective
_LENSSEL_TOL = 0.5  # |lenssel| below this = condenser in


@dataclass(frozen=True)
class PixelSizeResult:
    """Effective pixel sizes plus the derived geometry that produced them."""

    pixel_size_x_um: float
    pixel_size_y_um: float
    magnification: float
    two_theta_deg: float
    objective: str
    condenser_in: bool
    mainx: float
    obx: float
    ffsel: float
    ffz: float
    lenssel: float


def _scalar(pos: dict, key: str, where: str) -> float:
    if key not in pos:
        raise StageUserError(
            f"motor {key!r} not found in {where}",
            hint=f"This scan's positioners have no {key!r}; set the pixel size manually.",
        )
    return float(np.asarray(pos[key]).reshape(-1)[0])


def _match_objective(ffsel: float) -> tuple[str, float]:
    for value, (label, base) in _OBJECTIVES.items():
        if abs(ffsel - value) <= _FFSEL_TOL:
            return label, base
    raise StageUserError(
        f"unrecognized ffsel={ffsel:g}; cannot pick the far-field objective",
        hint="Expected ffsel -60 (2x) or 0 (10x). Set the pixel size manually.",
    )


def compute_pixel_size(
    scan_h5: str,
    positioners_path: str = DEFAULT_POSITIONERS_PATH,
    entry_suffix: str = ".1",
) -> PixelSizeResult:
    """Effective pixel size for the first ``entry_suffix`` entry of *scan_h5*."""
    with h5py.File(scan_h5, "r") as f:
        entries = get_filtered_entries(f, entry_suffix)
        if not entries:
            raise StageUserError(
                f"no scan entry ending in {entry_suffix!r} in {scan_h5}",
                hint="Point at a raw BLISS scan .h5 (entries like 1.1, 2.1, ...).",
            )
        group = f"{entries[0]}/{positioners_path}"
        try:
            pos = read_positioners(f, group)
        except KeyError as exc:
            raise StageUserError(
                str(exc),
                hint="Check the experiment's 'Positioners path' matches this file.",
            ) from exc

    mainx = _scalar(pos, MAINX, group)
    obx = _scalar(pos, OBX, group)
    ffsel = _scalar(pos, FFSEL, group)
    ffz = _scalar(pos, FFZ, group)
    lenssel = _scalar(pos, LENSSEL, group)

    if obx == 0.0:
        raise StageUserError(
            "obx = 0, cannot compute the magnification (mainx/obx - 1)",
            hint="Check obx; set the pixel size manually.",
        )
    magnification = mainx / obx - 1.0
    if magnification <= 0.0:
        raise StageUserError(
            f"non-physical magnification M={magnification:g} (mainx/obx - 1)",
            hint="Check mainx and obx (expected mainx > obx). Set the pixel size manually.",
        )

    objective, base = _match_objective(ffsel)
    px_x = base / magnification

    two_theta = math.atan2(ffz, mainx)
    condenser_in = abs(lenssel) <= _LENSSEL_TOL
    if condenser_in:
        sin2t = math.sin(two_theta)
        if abs(sin2t) < 1e-9:
            raise StageUserError(
                "2theta ~ 0, cannot divide the Y pixel size by sin(2theta)",
                hint="Check ffz/mainx; set the pixel size manually.",
            )
        px_y = px_x / sin2t
    else:
        px_y = px_x

    return PixelSizeResult(
        pixel_size_x_um=px_x,
        pixel_size_y_um=px_y,
        magnification=magnification,
        two_theta_deg=math.degrees(two_theta),
        objective=objective,
        condenser_in=condenser_in,
        mainx=mainx,
        obx=obx,
        ffsel=ffsel,
        ffz=ffz,
        lenssel=lenssel,
    )
