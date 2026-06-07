"""Sample-stage rastering: read samy/samz per layer and match positions.

The aligned-volume stages need each layer's physical ``samy``/``samz`` motor
position, read from the raw BLISS ``.h5`` in each folder. ``nearest_index``
finds the layer closest to a target ``(samy, samz)`` — used to match rocking
layers to mosaicity/strain layers.
"""

from __future__ import annotations

import glob
import os

import h5py
import numpy as np

DEFAULT_SAMY_PATH = "1.1/instrument/positioners/samy"
DEFAULT_SAMZ_PATH = "1.1/instrument/positioners/samz"


def find_h5_file(folder: str) -> str | None:
    """The ``.h5`` in *folder*: prefer ``<folder>.h5``, else the smallest ``*.h5``."""
    name = os.path.basename(folder)
    expected = os.path.join(folder, name + ".h5")
    if os.path.exists(expected):
        return expected
    h5s = sorted(glob.glob(os.path.join(folder, "*.h5")), key=os.path.getsize)
    return h5s[0] if h5s else None


def extract_motor_positions(
    folders: list[str],
    samy_path: str = DEFAULT_SAMY_PATH,
    samz_path: str = DEFAULT_SAMZ_PATH,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Read scalar ``(samy, samz)`` (mm) from each folder's raw ``.h5``.

    Returns ``(samy_array, samz_array, folder_names)``; folders without a file
    or without the motor datasets are silently skipped (kept aligned across the
    three returned sequences).
    """
    samy_vals: list[float] = []
    samz_vals: list[float] = []
    names: list[str] = []
    for folder in folders:
        h5_path = find_h5_file(folder)
        if h5_path is None:
            continue
        try:
            with h5py.File(h5_path, "r") as f:
                if samy_path not in f or samz_path not in f:
                    continue
                samy = float(np.array(f[samy_path]))
                samz = float(np.array(f[samz_path]))
        except OSError:
            continue
        samy_vals.append(samy)
        samz_vals.append(samz)
        names.append(os.path.basename(folder))
    return np.array(samy_vals), np.array(samz_vals), names


def nearest_index(
    samy_arr: np.ndarray, samz_arr: np.ndarray, target_y: float, target_z: float
) -> int:
    """Index of the layer whose ``(samy, samz)`` is closest to the target."""
    if len(samy_arr) == 0:
        raise ValueError("no motor positions to match against")
    d2 = (np.asarray(samy_arr) - target_y) ** 2 + (np.asarray(samz_arr) - target_z) ** 2
    return int(np.argmin(d2))
