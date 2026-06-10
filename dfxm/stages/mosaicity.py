"""Mosaicity stage — stack per-layer darfix maps into 3-D volumes.

Faithful port of ``stack_h5_darfix_volumes.py``: for each ``maps.h5`` it reads
the chi/mu Center-of-mass and FWHM 2-D maps and stacks them (along a new layer
axis) into ``stacked_volumes.h5`` with the structure::

    /chi/Center of mass   (L, H, W)
    /chi/FWHM             (L, H, W)
    /mu/Center of mass    (L, H, W)
    /mu/FWHM              (L, H, W)

A folder is included if at least one of its four maps is present (matching the
legacy behaviour); each output volume stacks only the layers where its map
existed.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field

import h5py
import numpy as np

from ..common.sort import find_matching_folders
from ..config.models import Param, ParamType, StageSpec

ProgressFn = Callable[[float, str], None]


def _noop(_frac: float, _msg: str) -> None:
    pass


# (param name, source-path default, output group, output dataset name)
_DATASETS = (
    ("chi_com_path", "/entry/chi/Center of mass/Center of mass", "chi", "Center of mass"),
    ("chi_fwhm_path", "/entry/chi/FWHM/FWHM", "chi", "FWHM"),
    ("mu_com_path", "/entry/mu/Center of mass/Center of mass", "mu", "Center of mass"),
    ("mu_fwhm_path", "/entry/mu/FWHM/FWHM", "mu", "FWHM"),
)


STAGE = StageSpec(
    name="mosaicity",
    label="Mosaicity volume",
    description=(
        "Stack per-layer darfix chi/mu Center-of-mass and FWHM maps into a 3D "
        "mosaicity volume (stacked_volumes.h5)."
    ),
    params=(
        Param("mode", ParamType.ENUM, "Mode", default="batch", choices=("single", "batch")),
        Param(
            "input_folder",
            ParamType.DIR,
            "Input folder",
            help="folder holding maps.h5 (single mode)",
        ),
        Param(
            "root_folder",
            ParamType.DIR,
            "Root folder",
            help="parent of mosaicity layer folders (batch)",
        ),
        Param("folder_pattern", ParamType.STR, "Folder pattern", default="*"),
        Param("maps_filename", ParamType.STR, "maps filename", default="maps.h5"),
        Param("chi_com_path", ParamType.STR, "chi COM path", default=_DATASETS[0][1]),
        Param("chi_fwhm_path", ParamType.STR, "chi FWHM path", default=_DATASETS[1][1]),
        Param("mu_com_path", ParamType.STR, "mu COM path", default=_DATASETS[2][1]),
        Param("mu_fwhm_path", ParamType.STR, "mu FWHM path", default=_DATASETS[3][1]),
        Param(
            "output_dir", ParamType.DIR, "Output dir", help="where stacked_volumes.h5 is written"
        ),
        Param("stacked_filename", ParamType.STR, "Stacked filename", default="stacked_volumes.h5"),
        Param(
            "compression",
            ParamType.ENUM,
            "Compression",
            default="gzip",
            choices=("gzip", "lzf", "none"),
        ),
    ),
)


@dataclass
class MosaicityResult:
    stacked_path: str | None = None
    datasets: dict[str, tuple[int, ...]] = field(default_factory=dict)
    layers: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def n_layers(self) -> int:
        return len(self.layers)


def _read_dataset(h5f: h5py.File, path: str) -> np.ndarray | None:
    obj = h5f.get(path)
    return obj[:] if isinstance(obj, h5py.Dataset) else None


def run(params: dict, progress: ProgressFn | None = None) -> MosaicityResult:
    progress = progress or _noop
    p = {**STAGE.defaults(), **params}
    maps_filename = p["maps_filename"]

    # source path per output key, plus the (group, name) routing
    config = {key: p[key] for key, _default, _g, _n in _DATASETS}
    routing = {key: (g, n) for key, _default, g, n in _DATASETS}

    if p["mode"] == "single":
        folder = p["input_folder"]
        if not folder:
            raise ValueError("single mode requires 'input_folder'")
        work = [(os.path.basename(folder.rstrip("/")), os.path.join(folder, maps_filename))]
        default_out_root = folder
    else:
        root = (p["root_folder"] or "").rstrip("/")
        if not root:
            raise ValueError("batch mode requires 'root_folder'")
        folders = find_matching_folders(root, p["folder_pattern"])
        if not folders:
            raise ValueError(f"no folders matching {p['folder_pattern']!r} in {root}")
        work = [(os.path.basename(f), os.path.join(f, maps_filename)) for f in folders]
        default_out_root = root

    collected: dict[str, list[np.ndarray]] = {key: [] for key in config}
    result = MosaicityResult()

    for i, (name, maps_path) in enumerate(work):
        progress(i / len(work), f"mosaicity: {name}")
        if not os.path.exists(maps_path):
            result.skipped.append(f"{name}: {maps_filename} not found")
            continue
        try:
            with h5py.File(maps_path, "r") as f:
                data = {key: _read_dataset(f, path) for key, path in config.items()}
        except OSError as exc:
            result.skipped.append(f"{name}: {exc}")
            continue
        if all(v is None for v in data.values()):
            result.skipped.append(f"{name}: no datasets")
            continue
        for key, arr in data.items():
            if arr is not None:
                collected[key].append(arr)
        result.layers.append(name)

    if not result.layers:
        progress(1.0, "no mosaicity layers produced")
        return result

    compression = None if p["compression"] == "none" else p["compression"]
    stacked_path = os.path.join(default_out_root, p["stacked_filename"])
    with h5py.File(stacked_path, "w") as f:
        groups = {g: f.require_group(g) for _k, _d, g, _n in _DATASETS}
        for key, slices in collected.items():
            if not slices:
                continue
            volume = np.stack(slices, axis=0)
            group_name, ds_name = routing[key]
            kw = {}
            if compression:
                kw["compression"] = compression
                if compression == "gzip":
                    kw["compression_opts"] = 4
            groups[group_name].create_dataset(ds_name, data=volume, **kw)
            result.datasets[f"/{group_name}/{ds_name}"] = tuple(volume.shape)
        f.attrs["num_layers"] = len(result.layers)
        f.attrs["source_folders"] = "\n".join(result.layers)
        f.attrs["description"] = "Stacked 3D volumes from 2D darfix maps"

    result.stacked_path = stacked_path
    progress(1.0, f"stacked {len(result.layers)} layers -> {os.path.basename(stacked_path)}")
    return result


def _main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Stack darfix mosaicity volumes.")
    ap.add_argument("--mode", choices=("single", "batch"), default="batch")
    ap.add_argument("--input-folder", default="")
    ap.add_argument("--root-folder", default="")
    ap.add_argument("--folder-pattern", default="*")
    args = ap.parse_args(argv)
    res = run(
        dict(
            mode=args.mode,
            input_folder=args.input_folder,
            root_folder=args.root_folder,
            folder_pattern=args.folder_pattern,
        ),
        progress=lambda f, m: print(f"  [{f * 100:5.1f}%] {m}"),
    )
    print(f"\n{res.n_layers} layers; stacked -> {res.stacked_path}; datasets {list(res.datasets)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
