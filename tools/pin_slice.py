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
import os
import sys

# Running this file directly (not via `-m`) puts tools/ on sys.path, not the
# repo root, so the darq_xray package would not be importable without this.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from darq_xray.common.errors import StageUserError  # noqa: E402
from darq_xray.stages.slices import build_pinned_spec  # noqa: E402


def _main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("h5_path", help="path to a swept oblique_slices.h5")
    ap.add_argument("slice_name", help="slice name (the sweep's 'name', e.g. oblique_full)")
    ap.add_argument("--offset", type=float, required=True, help="desired offset along normal, µm")
    ap.add_argument("--name", default=None, help="name for the pinned plane (default: auto)")
    ap.add_argument("--volume", default=None, help="read geometry from this volume group only")
    args = ap.parse_args(argv)
    try:
        spec = build_pinned_spec(args.h5_path, args.slice_name, [args.offset], volume=args.volume)[
            0
        ]
    except StageUserError as exc:
        raise SystemExit(f"{exc} ({exc.hint})") from exc
    if args.name:
        spec["name"] = args.name
    print(
        f"requested {args.offset:+.2f} µm -> nearest stored plane {spec['sweep_start_um']:+.2f} µm",
        file=sys.stderr,
    )
    print(json.dumps([spec], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
