"""Headless recipe renderer: python3 -m dfxm.compose render recipe.json -o outdir."""

from __future__ import annotations

import argparse
import os
import sys


def _main(argv: list[str] | None = None) -> int:
    from ..common.errors import StageUserError
    from .recipe import recipe_from_json
    from .render import export_recipe

    ap = argparse.ArgumentParser(prog="python3 -m dfxm.compose")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render", help="render a figure recipe to PNG/PDF/SVG")
    r.add_argument("recipe", help="path to a recipe .json")
    r.add_argument("-o", "--out", required=True, help="output directory")
    r.add_argument("--formats", default="", help="comma list, e.g. png,pdf,svg (default: style)")
    r.add_argument("--dpi", type=int, default=None)
    args = ap.parse_args(argv)

    try:
        with open(args.recipe, encoding="utf-8") as fh:
            recipe = recipe_from_json(
                fh.read(), base_dir=os.path.dirname(os.path.abspath(args.recipe))
            )
        fmts = tuple(f for f in args.formats.split(",") if f) or None
        paths, res = export_recipe(recipe, args.out, formats=fmts, dpi=args.dpi)
    except StageUserError as exc:
        print(f"error: {exc}", file=sys.stderr)
        if exc.hint:
            print(f"hint: {exc.hint}", file=sys.stderr)
        return 2
    for note in res.notes:
        print(f"note: {note}")
    for path in paths:
        print(f"wrote {path}")
    if res.n_rendered == 0:
        print("error: no panel rendered (all placeholders)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
