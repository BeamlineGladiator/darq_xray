"""Shared fixtures: tiny synthetic BLISS-style HDF5 files.

The fixtures build files with the same structure the real ID03 data has —
``<entry>/instrument/pco_ff/image`` detector stacks and an
``instrument/positioners`` group with array motors (``mu``, ``samy``), a
varying scalar (``ccmth``), and a constant scalar (``obpitch``) — so the stage
ports can be exercised without the SSD.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

FRAME_H, FRAME_W = 2, 3


def write_bliss_file(path, specs, *, ccmth_varies: bool = True, with_const_motor: bool = True):
    """Write a BLISS-style file at *path*.

    *specs* is an iterable of ``(entry_name, n_frames)``. Detector frames are
    filled deterministically so equality checks are meaningful.
    """
    with h5py.File(path, "w") as f:
        for i, (entry, nfr) in enumerate(specs):
            g = f.create_group(entry)
            pos = g.create_group("instrument/positioners")
            pos.create_dataset("mu", data=np.linspace(11.0, 11.2, nfr))
            pos.create_dataset("ccmth", data=7.1 + (0.1 * i if ccmth_varies else 0.0))
            if with_const_motor:
                pos.create_dataset("obpitch", data=0.5)
            pos.create_dataset("samy", data=np.arange(nfr, dtype=float))
            pos.create_dataset("samz", data=np.full(nfr, 5.0))
            frames = np.arange(nfr * FRAME_H * FRAME_W).reshape(nfr, FRAME_H, FRAME_W) + i * 1000
            g.create_dataset("instrument/pco_ff/image", data=frames.astype("uint16"))
    return str(path)


@pytest.fixture
def bliss_factory(tmp_path):
    """Factory: create a folder ``<name>/<name>.h5`` and return the folder path."""

    def _make(name="scan", specs=(("1.1", 3), ("2.1", 2), ("3.1", 4)), **kwargs):
        folder = tmp_path / name
        folder.mkdir()
        write_bliss_file(folder / f"{name}.h5", specs, **kwargs)
        return str(folder)

    return _make


@pytest.fixture
def batch_root(tmp_path):
    """A root containing three ``layer__*`` folders (natural-sort order matters)."""
    names = ["layer__1", "layer__2", "layer__10"]
    specs = {
        "layer__1": (("1.1", 2), ("2.1", 3)),
        "layer__2": (("1.1", 1),),
        "layer__10": (("1.1", 4),),
    }
    root = tmp_path / "root"
    root.mkdir()
    for name in names:
        folder = root / name
        folder.mkdir()
        write_bliss_file(folder / f"{name}.h5", specs[name])
    return str(root)
