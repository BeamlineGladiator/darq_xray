#!/usr/bin/env python3
"""Emit a pinned single-plane ``slices_json`` from a swept ``oblique_slices.h5``.

After running the ``slices`` stage with a sweep (``sweep_step_um`` on an
``extent: "auto"`` plane) you get many parallel planes. Once you have decided
which offset is interesting (it is in each PNG filename, e.g.
``…__p012_+024.00um.png``, and in the ``offsets_um`` dataset), this helper prints
a one-element ``slices_json`` that reproduces *only* that plane — copy it into the
slices stage's "Slices (JSON)" field and re-run.

It reads the slice geometry (``normal``/``origin``/``up``/``half_u``/``half_v``/
``du``/``dv``) straight off the stored slice-group attributes, so the pinned plane
is byte-identical to the one the sweep produced, and drops ``extent: "auto"``
(which would otherwise overwrite the sweep window). The offset is snapped to the
nearest stored plane — the same rule the ``profiles`` stage uses.

Usage::

    python3 tools/pin_slice.py oblique_slices.h5 oblique_full --offset 24
"""

from __future__ import annotations

import argparse
import json
import sys

import h5py
import numpy as np


def _find_slice_group(f, slice_name, volume=None):
    """Return (volume_id, slice_group) for the first volume holding *slice_name*.

    Slice geometry is identical across volumes, so any volume that carries the
    slice works; ``--volume`` forces a specific one.
    """
    vids = [volume] if volume else list(f.keys())
    for vid in vids:
        if vid not in f:
            continue
        g = f[vid]
        if slice_name in g and "offsets_um" in g[slice_name]:
            return vid, g[slice_name]
    raise SystemExit(
        f"slice {slice_name!r} not found in any volume group of the file "
        f"(volumes present: {', '.join(f.keys())})"
    )


def pin_spec(h5_path, slice_name, offset_um, *, name=None, volume=None):
    """Build the pinned single-plane spec dict for *slice_name* at *offset_um*."""
    with h5py.File(h5_path, "r") as f:
        vid, sg = _find_slice_group(f, slice_name, volume)
        offsets = sg["offsets_um"][:].astype(np.float64)
        idx = int(np.argmin(np.abs(offsets - float(offset_um))))
        matched = float(offsets[idx])
        a = dict(sg.attrs)

    spec = {
        "name": name or f"{slice_name}_pin_{matched:+.2f}um",
        "normal": np.asarray(a["normal"], np.float64).tolist(),
        "origin": np.asarray(a["origin"], np.float64).tolist(),
        "up": np.asarray(a["up"], np.float64).tolist(),
        "half_u": float(a["half_u"]),
        "half_v": float(a["half_v"]),
        "du": float(a["du"]),
        "dv": float(a["dv"]),
        "sweep_step_um": float(a.get("sweep_step_um") or 1.0) or 1.0,
        "sweep_start_um": matched,
        "sweep_stop_um": matched,
    }
    return spec, vid, matched, offsets


def _main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("h5_path", help="path to a swept oblique_slices.h5")
    ap.add_argument("slice_name", help="slice name (the sweep's 'name', e.g. oblique_full)")
    ap.add_argument("--offset", type=float, required=True, help="desired offset along normal, µm")
    ap.add_argument("--name", default=None, help="name for the pinned plane (default: auto)")
    ap.add_argument("--volume", default=None, help="read geometry from this volume group only")
    args = ap.parse_args(argv)

    spec, vid, matched, offsets = pin_spec(
        args.h5_path, args.slice_name, args.offset, name=args.name, volume=args.volume
    )
    # Info to stderr so stdout stays a clean, pasteable JSON list.
    print(
        f"geometry from volume {vid!r}; requested {args.offset:+.2f} µm -> "
        f"nearest stored plane {matched:+.2f} µm "
        f"(of {len(offsets)} planes: {offsets.min():+.2f}..{offsets.max():+.2f} µm)",
        file=sys.stderr,
    )
    print(json.dumps([spec], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
