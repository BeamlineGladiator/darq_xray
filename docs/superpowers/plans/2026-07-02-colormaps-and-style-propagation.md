# Per-group Colormaps + Publication-Style Propagation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Register ParaView's real *Fast* colormap, add four per-quantity colormap dropdowns to the publication-style controls, make every stage run render with the current publication style, and persist the style across sessions.

**Architecture:** A new Qt-free `dfxm/common/cmaps.py` bakes ParaView's *Fast* (Lab-interpolated) and registers it with matplotlib at `dfxm.common.plotting` import. `PlotStyle` gains four `cmap_*` fields resolved by figure builders via `resolve_cmap(style, group)`. The GUI snapshots the session style into a reserved `plot_style` params key at Run click; stages rebuild it with `style_from_params` and thread it through all run-time renders. `MainWindow` persists the style as JSON in QSettings.

**Tech Stack:** numpy, matplotlib (`Figure` API only — never pyplot), h5py, PySide6 (GUI layer only), pytest.

**Spec:** `docs/superpowers/specs/2026-07-02-colormaps-and-style-propagation-design.md`

## Global Constraints

- `dfxm/` stays Qt-free — never import PySide6/pyvista there.
- Figures via explicit `matplotlib.figure.Figure`; never `pyplot` / `matplotlib.use(...)`.
- Colormap groups: `mosa_com`, `mosa_fwhm`, `strain`, `raw`. Defaults: `fast`, `magma`, `RdBu_r`, `gray`.
- Curated dropdown list (exact order): `fast, magma, viridis, plasma, inferno, cividis, gray, bone, RdBu_r, coolwarm, seismic, turbo`.
- Reserved params key: `plot_style` (a plain dict from `dataclasses.asdict(PlotStyle)`); absent ⇒ legacy behavior (headless CLI unchanged).
- QSettings: org `dfxm`, app `pipeline` (default `QSettings()` — already configured in `gui/app.py`), key `plot_style`.
- Every code change to `dfxm/stages/` or `gui/` requires the matching `docs/Usage.md` + `docs/Codebase.md` update in the same change (Task 12 collects them; commit them with the final task).
- Verify per task: the named tests pass; final task runs `python3 -m pytest -q`, `ruff check . && ruff format .`, `python3 tests/gui_smoke.py`.

---

### Task 1: Real ParaView *Fast* colormap (`dfxm/common/cmaps.py`)

**Files:**
- Create: `dfxm/common/cmaps.py`
- Modify: `dfxm/common/plotting.py` (import-time registration; remove the `fast → coolwarm` fallback in `get_cmap`, lines 120-131)
- Test: `tests/test_cmaps.py` (new)

**Interfaces:**
- Produces: `dfxm.common.cmaps.fast_colormap(n: int = 256) -> matplotlib.colors.ListedColormap` (named `"fast"`); `dfxm.common.cmaps.register() -> None` (idempotent registration with `matplotlib.colormaps`). After `import dfxm.common.plotting`, `matplotlib.colormaps["fast"]` resolves.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cmaps.py
"""ParaView 'Fast' colormap: registration + fidelity to the official control points."""

from __future__ import annotations

import matplotlib
import numpy as np

import dfxm.common.plotting  # noqa: F401 — importing registers "fast"
from dfxm.common.cmaps import _FAST_POINTS, fast_colormap, register


def test_fast_is_registered_with_matplotlib():
    assert "fast" in matplotlib.colormaps
    # and resolvable through the shared lookup helper
    from dfxm.common.plotting import get_cmap

    assert get_cmap("fast").name == "fast"


def test_register_is_idempotent():
    register()
    register()  # second call must not raise "already registered"
    assert "fast" in matplotlib.colormaps


def test_fast_endpoints_match_paraview_control_points():
    cmap = fast_colormap()
    # x=0 -> first control point, x=1 -> last (Lab round-trip ~exact at nodes)
    np.testing.assert_allclose(cmap(0.0)[:3], _FAST_POINTS[0, 1:], atol=2e-3)
    np.testing.assert_allclose(cmap(1.0)[:3], _FAST_POINTS[-1, 1:], atol=2e-3)
    # the 0.5 node is a control point too
    np.testing.assert_allclose(cmap(0.5)[:3], [0.89950, 0.94465, 0.76866], atol=5e-3)


def test_fast_is_not_coolwarm():
    """The old silent coolwarm fallback must be gone."""
    fast = fast_colormap()
    coolwarm = matplotlib.colormaps["coolwarm"]
    # coolwarm(0) is a blue ~(0.23, 0.30, 0.75); fast(0) is a much darker navy
    assert not np.allclose(fast(0.0)[:3], coolwarm(0.0)[:3], atol=0.05)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_cmaps.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'dfxm.common.cmaps'`

- [ ] **Step 3: Write the module**

```python
# dfxm/common/cmaps.py
"""ParaView colormaps not shipped with matplotlib.

Currently one map: ParaView's default *Fast* (Francesca Samsel & Alan W.
Scott). The authoritative control points below are copied verbatim from
ParaView master ``Remoting/Views/ColorMaps.json`` (fetched 2026-07-02);
ParaView interpolates this map in CIELAB, so we do the same: convert the
sRGB control points to Lab, interpolate linearly on the normalized
positions, convert back, and bake a 256-entry ListedColormap.
"""

from __future__ import annotations

import matplotlib
import numpy as np
from matplotlib.colors import ListedColormap

# Columns: position (0..1), R, G, B  — ParaView "Fast", ColorSpace "Lab".
_FAST_POINTS = np.array(
    [
        [0.0, 0.05639999999999999, 0.05639999999999999, 0.47],
        [0.17159223942480895, 0.24300000000000013, 0.4603500000000004, 0.81],
        [0.2984914818394138, 0.3568143826543521, 0.7450246485363142, 0.954367702893722],
        [0.4321287371255907, 0.6882, 0.93, 0.9179099999999999],
        [0.5, 0.8994959551205902, 0.944646394975174, 0.7686567142818399],
        [0.5882260353170073, 0.957107977357604, 0.8338185108985666, 0.5089156299842102],
        [0.7061412605695164, 0.9275207599610714, 0.6214389091739178, 0.31535705838676426],
        [0.8476395308725272, 0.8, 0.3520000000000001, 0.15999999999999998],
        [1.0, 0.59, 0.07670000000000013, 0.11947499999999994],
    ]
)

# sRGB <-> CIELAB (D65), vectorised over trailing RGB axis.
_M_RGB2XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
_M_XYZ2RGB = np.linalg.inv(_M_RGB2XYZ)
_WHITE_D65 = np.array([0.95047, 1.0, 1.08883])
_DELTA = 6.0 / 29.0


def _srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    c = np.clip(c, 0.0, None)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1.0 / 2.4) - 0.055)


def _f(t):
    return np.where(t > _DELTA**3, np.cbrt(t), t / (3.0 * _DELTA**2) + 4.0 / 29.0)


def _f_inv(t):
    return np.where(t > _DELTA, t**3, 3.0 * _DELTA**2 * (t - 4.0 / 29.0))


def _rgb_to_lab(rgb):
    xyz = _srgb_to_linear(np.asarray(rgb, dtype=np.float64)) @ _M_RGB2XYZ.T / _WHITE_D65
    fx, fy, fz = _f(xyz[..., 0]), _f(xyz[..., 1]), _f(xyz[..., 2])
    return np.stack([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], axis=-1)


def _lab_to_rgb(lab):
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = fy + lab[..., 1] / 500.0
    fz = fy - lab[..., 2] / 200.0
    xyz = np.stack([_f_inv(fx), _f_inv(fy), _f_inv(fz)], axis=-1) * _WHITE_D65
    return np.clip(_linear_to_srgb(xyz @ _M_XYZ2RGB.T), 0.0, 1.0)


def fast_colormap(n: int = 256) -> ListedColormap:
    """Bake ParaView's *Fast* as an n-entry ListedColormap named ``"fast"``."""
    pos = _FAST_POINTS[:, 0]
    lab = _rgb_to_lab(_FAST_POINTS[:, 1:])
    x = np.linspace(0.0, 1.0, n)
    interp = np.stack([np.interp(x, pos, lab[:, i]) for i in range(3)], axis=-1)
    return ListedColormap(_lab_to_rgb(interp), name="fast")


def register() -> None:
    """Register ``"fast"`` with matplotlib once; safe to call repeatedly."""
    if "fast" not in matplotlib.colormaps:
        matplotlib.colormaps.register(fast_colormap(), name="fast")
```

- [ ] **Step 4: Hook registration into `dfxm/common/plotting.py` and drop the fallback**

In `dfxm/common/plotting.py`, after the existing imports add:

```python
from .cmaps import register as _register_fast_cmap

_register_fast_cmap()
```

and replace the body of `get_cmap` (currently lines 120-131):

```python
def get_cmap(name: str):
    """Look up a colormap by name (ParaView's ``"fast"`` is registered at import)."""
    registry = matplotlib.colormaps
    if name in registry:
        return registry[name]
    raise KeyError(f"unknown colormap {name!r}")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_cmaps.py tests/test_plot_style.py -q`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add dfxm/common/cmaps.py dfxm/common/plotting.py tests/test_cmaps.py
git commit -m "feat(cmaps): register ParaView's real Fast colormap, drop coolwarm fallback"
```

---

### Task 2: `PlotStyle` colormap groups + params/JSON helpers

**Files:**
- Modify: `dfxm/common/plotting.py` (PlotStyle dataclass ~line 20-53; imports)
- Test: `tests/test_plot_style.py` (append)

**Interfaces:**
- Produces (all in `dfxm.common.plotting`):
  - `CMAP_GROUPS: tuple[str, ...] = ("mosa_com", "mosa_fwhm", "strain", "raw")`
  - `CMAP_CHOICES: tuple[str, ...]` (curated list from Global Constraints)
  - `PlotStyle` fields `cmap_mosa_com="fast"`, `cmap_mosa_fwhm="magma"`, `cmap_strain="RdBu_r"`, `cmap_raw="gray"`; method `cmap_for(group: str) -> str` (KeyError on unknown group)
  - `resolve_cmap(style: PlotStyle | None, group: str | None, fallback: str = "magma") -> str` — `group=None` ⇒ `fallback`; `style=None` ⇒ `PlotStyle()` defaults
  - `style_from_params(params: dict) -> PlotStyle | None` — reads the reserved `plot_style` key; `None` when absent/empty; unknown keys dropped, missing keys defaulted, `formats` list→tuple
  - `style_to_json(style: PlotStyle) -> str` and `style_from_json(text: str) -> PlotStyle | None` (None on any parse/shape failure)

- [ ] **Step 1: Write the failing tests** (append to `tests/test_plot_style.py`)

```python
def test_cmap_groups_defaults_and_lookup():
    from dfxm.common.plotting import CMAP_CHOICES, CMAP_GROUPS, resolve_cmap

    s = PlotStyle()
    assert s.cmap_for("mosa_com") == "fast"
    assert s.cmap_for("mosa_fwhm") == "magma"
    assert s.cmap_for("strain") == "RdBu_r"
    assert s.cmap_for("raw") == "gray"
    assert CMAP_GROUPS == ("mosa_com", "mosa_fwhm", "strain", "raw")
    for g in CMAP_GROUPS:
        assert s.cmap_for(g) in CMAP_CHOICES
    import pytest

    with pytest.raises(KeyError):
        s.cmap_for("nope")
    # resolve_cmap: None style -> defaults; None group -> fallback
    assert resolve_cmap(None, "raw") == "gray"
    assert resolve_cmap(replace(s, cmap_raw="viridis"), "raw") == "viridis"
    assert resolve_cmap(None, None, fallback="bone") == "bone"


def test_style_from_params_roundtrip_and_tolerance():
    from dataclasses import asdict

    from dfxm.common.plotting import style_from_params

    src = replace(PUBLICATION_STYLE, cmap_strain="seismic", font_scale=3.0)
    p = {"plot_style": asdict(src)}
    got = style_from_params(p)
    assert got == src
    assert style_from_params({}) is None
    # unknown keys dropped, missing keys defaulted, formats list -> tuple
    got = style_from_params({"plot_style": {"font_scale": 2.0, "formats": ["png"], "bogus": 1}})
    assert got.font_scale == 2.0 and got.formats == ("png",) and got.cmap_mosa_com == "fast"


def test_style_json_roundtrip_and_bad_blob():
    from dfxm.common.plotting import style_from_json, style_to_json

    src = replace(PUBLICATION_STYLE, cmap_raw="turbo")
    assert style_from_json(style_to_json(src)) == src
    assert style_from_json("{not json") is None
    assert style_from_json("[1,2]") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_plot_style.py -q`
Expected: FAIL — `ImportError: cannot import name 'CMAP_CHOICES'`

- [ ] **Step 3: Implement in `dfxm/common/plotting.py`**

Change the dataclasses import and add json:

```python
import json
import math
from dataclasses import dataclass, fields
```

Below the imports (near the top, before `PlotStyle`):

```python
# Colormap quantity groups + the curated dropdown list (shared by the GUI).
CMAP_GROUPS: tuple[str, ...] = ("mosa_com", "mosa_fwhm", "strain", "raw")
CMAP_CHOICES: tuple[str, ...] = (
    "fast",
    "magma",
    "viridis",
    "plasma",
    "inferno",
    "cividis",
    "gray",
    "bone",
    "RdBu_r",
    "coolwarm",
    "seismic",
    "turbo",
)
```

Inside `PlotStyle`, after the `dpi: int = 300` field:

```python
    # per-quantity colormaps (see CMAP_GROUPS)
    cmap_mosa_com: str = "fast"
    cmap_mosa_fwhm: str = "magma"
    cmap_strain: str = "RdBu_r"
    cmap_raw: str = "gray"

    def cmap_for(self, group: str) -> str:
        """Colormap name for a quantity group (KeyError on unknown group)."""
        if group not in CMAP_GROUPS:
            raise KeyError(f"unknown colormap group {group!r}")
        return getattr(self, f"cmap_{group}")
```

After the `PUBLICATION_STYLE` constant:

```python
def resolve_cmap(style: PlotStyle | None, group: str | None, fallback: str = "magma") -> str:
    """Colormap for *group* from *style* (or the PlotStyle defaults when None).

    ``group=None`` means "not one of the four quantity groups" and returns
    *fallback* unchanged.
    """
    if group is None:
        return fallback
    return (style if style is not None else PlotStyle()).cmap_for(group)


def _style_from_dict(data: dict) -> PlotStyle:
    names = {f.name for f in fields(PlotStyle)}
    kwargs = {k: v for k, v in dict(data).items() if k in names}
    if isinstance(kwargs.get("formats"), list):
        kwargs["formats"] = tuple(kwargs["formats"])
    return PlotStyle(**kwargs)


def style_from_params(params: dict) -> PlotStyle | None:
    """Rebuild the GUI-injected style from the reserved ``plot_style`` params key.

    Returns ``None`` when the key is absent/empty (headless CLI ⇒ legacy look).
    Unknown keys are dropped and missing keys defaulted so an older or newer
    GUI snapshot never crashes a stage.
    """
    raw = params.get("plot_style")
    if not raw:
        return None
    return _style_from_dict(raw)


def style_to_json(style: PlotStyle) -> str:
    """Serialize a style for QSettings persistence."""
    from dataclasses import asdict

    return json.dumps(asdict(style))


def style_from_json(text: str) -> PlotStyle | None:
    """Inverse of :func:`style_to_json`; ``None`` on any parse/shape failure."""
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    try:
        return _style_from_dict(data)
    except (TypeError, ValueError):
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_plot_style.py tests/test_cmaps.py -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add dfxm/common/plotting.py tests/test_plot_style.py
git commit -m "feat(plotting): PlotStyle colormap groups + plot_style params/JSON helpers"
```

---

### Task 3: Style threading in shared renderers (`render.py`, `figures.py`)

**Files:**
- Modify: `dfxm/common/render.py` (`save_layer_pngs` line 70, `save_layer_animation` line 88)
- Modify: `dfxm/common/figures.py` (`volume_layer_specs` line 90)
- Test: `tests/test_figures_catalog.py` (append)

**Interfaces:**
- Consumes: `resolve_cmap` from Task 2.
- Produces: `save_layer_pngs(..., sx, sy, *, style: PlotStyle | None = None)`; `save_layer_animation(..., fmt, sx, sy, *, style: PlotStyle | None = None)`; `volume_layer_specs(*, ..., cmap: str, cmap_group: str | None = None, ...)` whose `build(style)` resolves `resolve_cmap(style, cmap_group, fallback=cmap)`.

- [ ] **Step 1: Write the failing test** (append to `tests/test_figures_catalog.py`; reuse the file's existing imports/fixtures for a small HDF5 volume — if none exists, create one with the code below)

```python
def test_volume_layer_specs_cmap_group_resolves_from_style(tmp_path):
    import h5py
    import numpy as np

    from dfxm.common.figures import volume_layer_specs
    from dfxm.common.plotting import PlotStyle

    p = tmp_path / "v.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("vol", data=np.random.default_rng(0).random((2, 4, 5)))
    specs = volume_layer_specs(
        h5_path=str(p),
        dataset="vol",
        id_prefix="t",
        title="T",
        cbar_label="c",
        cmap="magma",
        cmap_group="raw",
        sx=1.0,
        sy=1.0,
        vmin=0.0,
        vmax=1.0,
    )
    fig = specs[0].build(PlotStyle(cmap_raw="viridis"))
    assert fig.axes[0].images[0].cmap.name == "viridis"
    # no group -> fixed cmap wins regardless of style
    specs2 = volume_layer_specs(
        h5_path=str(p),
        dataset="vol",
        id_prefix="t2",
        title="T",
        cbar_label="c",
        cmap="bone",
        sx=1.0,
        sy=1.0,
        vmin=0.0,
        vmax=1.0,
    )
    fig2 = specs2[0].build(PlotStyle(cmap_raw="viridis"))
    assert fig2.axes[0].images[0].cmap.name == "bone"


def test_save_layer_pngs_accepts_style(tmp_path):
    import numpy as np

    from dfxm.common import render
    from dfxm.common.plotting import PlotStyle

    vol = np.zeros((1, 4, 5))
    d = render.save_layer_pngs(
        vol, [0.0], str(tmp_path), "x", 0, 1, "gray", "t", "c", 1.0, 1.0,
        style=PlotStyle(font_scale=3.0),
    )
    import os

    assert os.path.exists(os.path.join(d, "layer_0000.png"))
```

Note: `cmap_nan_transparent` copies the colormap but keeps its `.name`, so asserting on `images[0].cmap.name` is stable.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_figures_catalog.py -q`
Expected: FAIL — `TypeError: volume_layer_specs() got an unexpected keyword argument 'cmap_group'`

- [ ] **Step 3: Implement**

`dfxm/common/render.py` — thread `style` through the two savers (signatures + the `layer_figure` calls inside):

```python
def save_layer_pngs(volume, z_um, out_dir, name, vmin, vmax, cmap, title, cbar, sx, sy, *, style=None):
    ...
        fig, _, _ = layer_figure(
            volume[z], vmin, vmax, cmap, ext_x, ext_y, full_title, cbar, style=style
        )
```

```python
def save_layer_animation(
    volume, z_um, base_path, name, vmin, vmax, cmap, title, cbar, fmt, sx, sy, *, style=None
):
    ...
    fig, ax, im = layer_figure(volume[0], vmin, vmax, cmap, ext_x, ext_y, title, cbar, style=style)
```

`dfxm/common/figures.py` — `volume_layer_specs` gains `cmap_group: str | None = None` (keyword-only, after `cmap`), and its inner `build`:

```python
    def make(z):
        def build(style):
            from .plotting import resolve_cmap

            layer = _load_layer(h5_path, dataset, z)
            zlabel = f"\nZ = {z_um[z]:.2f} µm" if z_um is not None else ""
            fig, _, _ = render.layer_figure(
                layer,
                vmin,
                vmax,
                resolve_cmap(style, cmap_group, fallback=cmap),
                ext_x,
                ...
```

(keep the rest of the call unchanged; only the cmap argument changes).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_figures_catalog.py -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add dfxm/common/render.py dfxm/common/figures.py tests/test_figures_catalog.py
git commit -m "feat(render): style + cmap-group threading in shared layer renderers"
```

---

### Task 4: Visualize stage — groups + run style

**Files:**
- Modify: `dfxm/stages/visualize.py` (`_display_info` line 344, `_process_dataset` line 417, `run` calls lines 485/510, `aligned_field` lines 552-595, `_make_build` line 599, `figures` line 648)
- Test: `tests/test_stage_visualize.py` (append)

**Interfaces:**
- Consumes: `resolve_cmap`, `style_from_params` (Task 2); `save_layer_pngs/animation(style=)` (Task 3).
- Produces: `_display_info(dataset_name, is_strain=False) -> (title, cbar_label, cmap_group)` where `cmap_group ∈ {"mosa_com","mosa_fwhm","strain",None}` (**third element is now a group, not a cmap name**). `_process_dataset(..., p, out_dir, style=None)`. `_make_build(loader, z, vn, vx, cmap_group, ex, ey, t, cb)`.

- [ ] **Step 1: Write the failing test** (append to `tests/test_stage_visualize.py`)

```python
def test_display_info_returns_groups():
    from dfxm.stages.visualize import _display_info

    assert _display_info("chi_Center_of_mass")[2] == "mosa_com"
    assert _display_info("mu_FWHM")[2] == "mosa_fwhm"
    assert _display_info("strain", is_strain=True)[2] == "strain"
    assert _display_info("something_else")[2] is None


def test_figures_resolve_cmap_from_style(tmp_path):
    """A visualize FigureSpec built with a custom style uses the style's cmap."""
    from dfxm.common.plotting import PlotStyle
    # Reuse this file's existing fixture that creates stacked volumes + raw
    # folders and runs the stage (see existing end-to-end test); then:
    #   specs = V.figures(result, params)
    #   pick the first spec whose figure_id contains "Center_of_mass"
    #   fig = spec.build(PlotStyle(cmap_mosa_com="viridis"))
    #   assert fig.axes[0].images[0].cmap.name == "viridis"
```

(Adapt the body to the file's existing end-to-end fixture — `test_stage_visualize.py` already builds a full run; add the build/assert lines after it. The default-style check: `spec.build(None)` must yield cmap name `"fast"` for CoM.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_stage_visualize.py -q`
Expected: FAIL — `_display_info(...)[2] == "mosa_com"` is `"magma"`

- [ ] **Step 3: Implement**

`_display_info` (line 344) — return groups:

```python
def _display_info(dataset_name, is_strain=False):
    """(title, cbar_label, cmap_group) for a dataset; group None = not a std quantity."""
    if is_strain:
        return ("Strain (cot method)", "Strain (ε)", "strain")
    axis = (
        "χ"
        if dataset_name.startswith("chi_")
        else "μ"
        if dataset_name.startswith("mu_")
        else dataset_name
    )
    if "Center_of_mass" in dataset_name:
        return (f"{axis} Misorientation", "Misorientation (°)", "mosa_com")
    if "FWHM" in dataset_name:
        return (f"{axis} Peak Broadening", "Peak broadening (°)", "mosa_fwhm")
    return (dataset_name.replace("_", " "), "(°)", None)
```

`run()`: after `p = {**STAGE.defaults(), **params}` add `style = style_from_params(p)` (import `resolve_cmap, style_from_params` from `..common.plotting`). At the two call sites (lines ~485, ~510) the unpack becomes `title, cbar, group = _display_info(...)` and the cmap is `cmap = resolve_cmap(style, group)`; pass `style=style` into `_process_dataset`.

`_process_dataset(data, z_pos, scale_z, name, vmin, vmax, cmap, title, cbar, p, out_dir, style=None)`: pass `style=style` to `Rnd.save_layer_pngs(...)` and `Rnd.save_layer_animation(...)` (the 3-D top view keeps just the resolved cmap name).

`aligned_field` (3-D viewer glue): line 573 `cmap = "RdBu_r"` → `cmap = resolve_cmap(None, "strain")`; line 594 `_, _, cmap = _display_info(name)` → `_, _, group = _display_info(name)` then `cmap = resolve_cmap(None, group)`.

`_make_build` + `figures()`: parameter `cm`/`cmap` becomes `cmap_group`; inside `build`:

```python
def _make_build(loader, z, vn, vx, cmap_group, ex, ey, t, cb):
    def build(style, _loader=loader, _z=z, _vn=vn, _vx=vx, _grp=cmap_group, _ex=ex, _ey=ey, _t=t, _cb=cb):
        layer = _loader()[_z]
        fig, _, _ = Rnd.layer_figure(
            layer, _vn, _vx, resolve_cmap(style, _grp), _ex, _ey, f"{_t} (layer {_z})", _cb, style=style
        )
        return fig

    return build
```

and in `figures()` the unpack (line 648) becomes `title, cbar_label, group = _display_info(...)`, passing `group` to `_make_build`.

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_stage_visualize.py tests/test_gui_viewers.py -q`
Expected: all PASS (test_gui_viewers exercises `aligned_field`)

- [ ] **Step 5: Commit**

```bash
git add dfxm/stages/visualize.py tests/test_stage_visualize.py
git commit -m "feat(visualize): cmap groups + publication style in run renders"
```

---

### Task 5: Rocking stage — raw group + run style

**Files:**
- Modify: `dfxm/stages/rocking.py` (`_render` line 542, its two call sites lines 720/732, figure specs line 828)
- Test: `tests/test_stage_rocking.py` (append)

**Interfaces:**
- Consumes: `resolve_cmap`, `style_from_params`, threaded renderers.
- Produces: `_render(result, vol, z_um, scale_z, name, p, out_dir, cmap, title, cbar, style=None)`.

- [ ] **Step 1: Write the failing test** (append; reuse the file's existing run fixture)

```python
def test_rocking_figures_use_raw_group(...existing fixture args...):
    # after the existing end-to-end run that produces `result` and `params`:
    from dfxm.common.plotting import PlotStyle
    from dfxm.stages import rocking as R

    specs = R.figures(result, params)
    fig = specs[0].build(PlotStyle(cmap_raw="viridis"))
    assert fig.axes[0].images[0].cmap.name == "viridis"
    fig = specs[0].build(None)  # default -> gray (was magma)
    assert fig.axes[0].images[0].cmap.name == "gray"
```

- [ ] **Step 2: Run to verify failure** — `python3 -m pytest tests/test_stage_rocking.py -q`; expected FAIL (cmap is `magma`).

- [ ] **Step 3: Implement**

- `run()`: add `style = style_from_params(p)` near the top; the two `_render(...)` calls change `"magma"` → `resolve_cmap(style, "raw")` and append `style=style`.
- `_render(...)`: add trailing `style=None` param; pass `style=style` into `Rnd.save_layer_pngs` and `Rnd.save_layer_animation`.
- figure specs (line 828): `cmap="magma"` → `cmap="gray", cmap_group="raw"`.
- Import `resolve_cmap, style_from_params` from `..common.plotting`.

- [ ] **Step 4: Run tests** — `python3 -m pytest tests/test_stage_rocking.py -q`; expected PASS.

- [ ] **Step 5: Commit**

```bash
git add dfxm/stages/rocking.py tests/test_stage_rocking.py
git commit -m "feat(rocking): raw cmap group (default gray) + style in run renders"
```

---

### Task 6: Mosaicity stage — groups in the figure catalog

**Files:**
- Modify: `dfxm/stages/mosaicity.py` (`_KEY_DISPLAY` line 196, its use line 245, `volume_layer_specs` call line ~260)
- Test: `tests/test_stage_mosaicity.py` (append)

**Interfaces:**
- Consumes: `volume_layer_specs(cmap_group=)`.
- Produces: `_KEY_DISPLAY` values become `(cmap_group, cbar_label, title)`.

- [ ] **Step 1: Failing test** (append; reuse the existing figures fixture)

```python
def test_mosaicity_figures_use_groups(...existing fixture args...):
    from dfxm.common.plotting import PlotStyle

    specs = M.figures(result, params)
    com_spec = next(s for s in specs if "Center_of_mass" in s.figure_id or "com" in s.figure_id.lower())
    fig = com_spec.build(PlotStyle(cmap_mosa_com="viridis"))
    assert fig.axes[0].images[0].cmap.name == "viridis"
    assert com_spec.build(None).axes[0].images[0].cmap.name == "fast"
```

- [ ] **Step 2: Run to verify failure** — `python3 -m pytest tests/test_stage_mosaicity.py -q`; FAIL (magma).

- [ ] **Step 3: Implement**

```python
_KEY_DISPLAY: dict[str, tuple[str | None, str, str]] = {
    "/chi/Center of mass": ("mosa_com", "Misorientation (°)", "χ Misorientation"),
    "/chi/FWHM": ("mosa_fwhm", "Peak broadening (°)", "χ Peak Broadening"),
    "/mu/Center of mass": ("mosa_com", "Misorientation (°)", "μ Misorientation"),
    "/mu/FWHM": ("mosa_fwhm", "Peak broadening (°)", "μ Peak Broadening"),
}
```

Use site (line 245): `group, cbar_label, title = _KEY_DISPLAY.get(key, (None, "(°)", ...))`; the `volume_layer_specs(...)` call passes `cmap="magma", cmap_group=group`.

- [ ] **Step 4: Run tests** — `python3 -m pytest tests/test_stage_mosaicity.py -q`; PASS.

- [ ] **Step 5: Commit**

```bash
git add dfxm/stages/mosaicity.py tests/test_stage_mosaicity.py
git commit -m "feat(mosaicity): cmap groups in figure catalog"
```

---

### Task 7: Strain stage — strain group + run style

**Files:**
- Modify: `dfxm/stages/strain.py` (`build_strain_map` line 348, `build_detrend_diag` line 423, `process_maps_file` line 618, `run` line 706, `figures` line 532 — verify it already forwards style; if so leave it)
- Test: `tests/test_stage_strain.py` (append)

**Interfaces:**
- Consumes: `resolve_cmap`, `style_from_params`.
- Produces: `process_maps_file(..., style: PlotStyle | None = None)`; `build_strain_map`/`build_detrend_diag` resolve their imshow cmap via `resolve_cmap(style, "strain")` (identical `RdBu_r` when style is None).

- [ ] **Step 1: Failing test** (append)

```python
def test_strain_map_cmap_follows_style():
    import numpy as np

    from dfxm.common.plotting import PlotStyle
    from dfxm.stages.strain import build_strain_map

    strain = np.random.default_rng(0).standard_normal((6, 8)) * 1e-4
    fig = build_strain_map(strain, 0.152, 0.385, None, (None, None))
    assert fig.axes[0].images[0].cmap.name == "RdBu_r"  # legacy default preserved
    fig = build_strain_map(
        strain, 0.152, 0.385, None, (None, None), style=PlotStyle(cmap_strain="seismic")
    )
    assert fig.axes[0].images[0].cmap.name == "seismic"
```

- [ ] **Step 2: Run to verify failure** — `python3 -m pytest tests/test_stage_strain.py -q`; FAIL on the seismic assert.

- [ ] **Step 3: Implement**

- `build_strain_map`: `cmap="RdBu_r"` (line ~385) → `cmap=resolve_cmap(style, "strain")`.
- `build_detrend_diag`: the `imshow(..., cmap="RdBu_r", ...)` (line ~446) → `cmap=resolve_cmap(style, "strain")`.
- `process_maps_file(...)`: add `style: PlotStyle | None = None` kwarg; the three run-time save calls (lines 636-655) pass `style=style` into `build_strain_map(..., style=style)`, `build_strain_histogram(..., style=style)`, `build_detrend_diag(..., style=style)`.
- `run()`: `style = style_from_params(p)` and pass `style=style` into `process_maps_file`.
- `figures()`: confirm the existing builders already take the `style` argument from `build(style)` (they do — plot-export project); no change unless the cmap was captured, in which case route through `resolve_cmap(style, "strain")` the same way.
- Import `resolve_cmap, style_from_params`.

- [ ] **Step 4: Run tests** — `python3 -m pytest tests/test_stage_strain.py tests/test_export_fidelity.py -q`; PASS.

- [ ] **Step 5: Commit**

```bash
git add dfxm/stages/strain.py tests/test_stage_strain.py
git commit -m "feat(strain): strain cmap group + style threading in run diagnostics"
```

---

### Task 8: Slices stage — style-resolved cmaps end to end

**Files:**
- Modify: `dfxm/stages/slices.py` (`_STD_VOLUMES` line 70, `_standard_volumes` line 800, `prepare_volume` line 623, `save_slice_png` line 762, `run` line 827, `figures` line 958)
- Test: `tests/test_stage_slices.py` (append)

**Interfaces:**
- Consumes: `resolve_cmap`, `style_from_params`.
- Produces: `_GROUP_BY_KIND: dict[str, str]` mapping kind→group (`raw_sum`/`raw_specific` → `"raw"`); `prepare_volume(cfg, p, scale_x, scale_y, samy_dir, style=None)`; `save_slice_png(..., *, offset_um, dpi=150, style=None)`. `oblique_slices.h5` volume groups carry the **resolved** cmap in their `cmap` attr (profiles + line picker inherit unchanged).

- [ ] **Step 1: Failing test** (append; reuse `_setup(tmp_path)`-style fixture already in the file that runs the stage end to end)

```python
def test_run_resolves_cmaps_from_injected_style(tmp_path):
    # reuse the existing end-to-end fixture params, then:
    params["plot_style"] = {"cmap_mosa_com": "viridis", "cmap_raw": "bone", "font_scale": 1.0}
    res = S.run(params)
    with h5py.File(res.output_h5, "r") as f:
        assert f["mosa_com_chi"].attrs["cmap"] == "viridis"
        assert f["raw_sum"].attrs["cmap"] == "bone"
        assert f["strain"].attrs["cmap"] == "RdBu_r"  # default from PlotStyle


def test_run_without_style_uses_group_defaults(tmp_path):
    # same fixture, no plot_style key:
    res = S.run(params)
    with h5py.File(res.output_h5, "r") as f:
        assert f["mosa_com_chi"].attrs["cmap"] == "fast"  # real fast, no coolwarm fallback
        assert f["raw_sum"].attrs["cmap"] == "gray"


def test_figures_re_resolve_cmap_by_kind(tmp_path):
    # after a default run producing res:
    from dfxm.common.plotting import PlotStyle

    specs = S.figures(res, {})
    spec = next(s for s in specs if "mosa_com_chi" in s.figure_id)
    fig = spec.build(PlotStyle(cmap_mosa_com="plasma"))
    assert fig.axes[0].images[0].cmap.name == "plasma"
```

- [ ] **Step 2: Run to verify failure** — `python3 -m pytest tests/test_stage_slices.py -q`; FAIL (`cmap` attr is the old hard-coded value / `TypeError` on style kwarg).

- [ ] **Step 3: Implement**

- `_STD_VOLUMES` entries drop the trailing cmap element → 5-tuples `(toggle, source, file_param, dataset, kind)`; fix the unpack in `_standard_volumes` (`for toggle, source, file_param, dataset, kind in _STD_VOLUMES:`) and drop `"cmap": cmap` from the cfg dict.
- Add near `_CENTERED_KINDS`:

```python
# Quantity group per volume kind (for PlotStyle.cmap_for).
_GROUP_BY_KIND: dict[str, str] = {
    "mosa_com": "mosa_com",
    "mosa_fwhm": "mosa_fwhm",
    "strain": "strain",
    "raw_sum": "raw",
    "raw_specific": "raw",
}
```

- `prepare_volume(cfg, p, scale_x, scale_y, samy_dir, style=None)`: replace `"cmap_name": cfg.get("cmap") or "magma"` with `"cmap_name": resolve_cmap(style, _GROUP_BY_KIND.get(kind))`.
- `save_slice_png(prep, sl, slice2d, u_um, v_um, out_png, *, offset_um, dpi=150, style=None)`: forward `style=style` to `build_slice_figure`.
- `run()`: `style = style_from_params(p)` after building `p`; pass `style=style` to `prepare_volume(...)` and to both `save_slice_png(...)` calls.
- `figures()`: in the inner `build(...)`, after reading the h5 attrs, re-resolve:

```python
                    def build(style, vid=vid, sname=sname, k=k, prep=dict(prep), kind=kind):
                        prep = dict(prep)
                        prep["cmap_name"] = resolve_cmap(
                            style, _GROUP_BY_KIND.get(kind), fallback=prep["cmap_name"]
                        )
                        ...
```

(`kind` is already read from the group attrs at line 975; old files without a known kind keep their stored cmap via the fallback.)
- Import `resolve_cmap, style_from_params` from `..common.plotting` (extend the existing `from ..common.plotting import ...` line).

- [ ] **Step 4: Run tests** — `python3 -m pytest tests/test_stage_slices.py tests/test_stage_profiles.py -q`; PASS (profiles reads the `cmap` attr, so it inherits automatically).

- [ ] **Step 5: Commit**

```bash
git add dfxm/stages/slices.py tests/test_stage_slices.py
git commit -m "feat(slices): style-resolved per-group cmaps written to oblique_slices.h5"
```

---

### Task 9: Profiles stage — run-time style threading

**Files:**
- Modify: `dfxm/stages/profiles.py` (`save_companion_figure` line 496, `render_single` line 503, their call sites lines 603/671/688, `run` line 611)
- Test: `tests/test_stage_profiles.py` (append)

**Interfaces:**
- Consumes: `style_from_params`; `build_companion_figure(..., style=)` already exists.
- Produces: `save_companion_figure(ref, fields, geom, line_color, out_png, dpi, style=None)`; `render_single(ref, geom, line_color, out_png, header, dpi, style=None)`.

- [ ] **Step 1: Failing test** (append)

```python
def test_run_time_savers_accept_style(tmp_path, ...existing fixture...):
    """A profiles run with an injected style must not crash and must produce PNGs."""
    params["plot_style"] = {"font_scale": 2.0}
    res = P.run(params)
    assert res.pngs  # (use the result field the existing tests assert on)
```

(Adapt to the existing end-to-end fixture in the file; the point is exercising the styled path through `run`.)

- [ ] **Step 2: Run to verify failure** — `python3 -m pytest tests/test_stage_profiles.py -q`; this may PASS trivially if `plot_style` is ignored — in that case assert on the styled path directly: call `save_companion_figure(..., style=PlotStyle(font_scale=2.0))` and expect `TypeError` before the fix. Confirm at least one assertion fails pre-implementation.

- [ ] **Step 3: Implement**

```python
def save_companion_figure(ref, fields, geom, line_color, out_png, dpi, style=None):
    """Build a companion figure (legacy when style is None) and save it."""
    build_companion_figure(ref, fields, geom, line_color, style=style).savefig(
        out_png, dpi=dpi, facecolor="white", edgecolor="none"
    )


def render_single(ref, geom, line_color, out_png, header, dpi, style=None):
    plane, u_um, v_um, attrs, label = ref
    fig = Figure(figsize=(11, 9), facecolor="white")
    ax = fig.add_subplot(111)
    im = _draw_reference_image(
        ax, plane, u_um, v_um, attrs, line_color, geom=geom, title=f"{header}\nreference: {label}"
    )
    if style is None:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(attrs["cbar_label"])
    else:
        if style.colorbar:
            add_colorbar(fig, im, ax, attrs["cbar_label"], style)
        apply_text_scale(ax, style)
    fig.savefig(out_png, dpi=dpi, facecolor="white", edgecolor="none", bbox_inches="tight")
```

(`add_colorbar`/`apply_text_scale` are already imported in this module for `build_companion_figure`; if not, extend the existing `from ..common.plotting import ...` line.)

- `run()` (line 611): `style = style_from_params(p)`; thread `style=style` into every `save_companion_figure(...)` and `render_single(...)` call in the drivers it invokes (lines ~603, ~671, ~688 — pass `style` down through the driver functions' signatures as a plain trailing `style=None` parameter).

- [ ] **Step 4: Run tests** — `python3 -m pytest tests/test_stage_profiles.py -q`; PASS.

- [ ] **Step 5: Commit**

```bash
git add dfxm/stages/profiles.py tests/test_stage_profiles.py
git commit -m "feat(profiles): publication style threads into run-time figures"
```

---

### Task 10: Matched stage — ENUM colormap param + run style

**Files:**
- Modify: `dfxm/stages/matched.py` (`colormap` Param lines 160-168, run-time `layer_figure` call line 424, `run` for `style_from_params`)
- Test: `tests/test_stage_matched.py` (append)

**Interfaces:**
- Consumes: `CMAP_CHOICES`, `style_from_params`.
- Produces: `colormap` becomes `ParamType.ENUM` with `choices=CMAP_CHOICES`, default `"gray"` (a real dropdown in the auto-built form).

- [ ] **Step 1: Failing test** (append)

```python
def test_colormap_param_is_enum_dropdown():
    from dfxm.common.plotting import CMAP_CHOICES
    from dfxm.config.models import ParamType
    from dfxm.stages.matched import STAGE

    p = next(q for q in STAGE.params if q.name == "colormap")
    assert p.type is ParamType.ENUM
    assert p.choices == CMAP_CHOICES
    assert p.default == "gray"
```

- [ ] **Step 2: Run to verify failure** — `python3 -m pytest tests/test_stage_matched.py -q`; FAIL (type is STR).

- [ ] **Step 3: Implement**

```python
        Param(
            "colormap",
            ParamType.ENUM,
            "Colormap",
            default="gray",
            choices=CMAP_CHOICES,
            advanced=True,
            group="Appearance",
            help="Colormap for the saved PNGs (default gray).",
        ),
```

with `from ..common.plotting import CMAP_CHOICES, style_from_params` added to the imports. In `run()`: `style = style_from_params(p)`; the run-time `Rnd.layer_figure(...)` call (line 424) gains `style=style`. The `figures()` build already passes `style=style` (line 513) — matching behavior.

- [ ] **Step 4: Run tests** — `python3 -m pytest tests/test_stage_matched.py tests/test_param_metadata.py -q`; PASS.

- [ ] **Step 5: Commit**

```bash
git add dfxm/stages/matched.py tests/test_stage_matched.py
git commit -m "feat(matched): colormap param becomes dropdown; style in run renders"
```

---

### Task 11: GUI — Colormaps section, run injection, QSettings persistence

**Files:**
- Modify: `gui/widgets/export_dialog.py` (`_CMAPS` constant; `_build_controls` line 154; `sync_from_style` line 62; `_all_widgets` line 122)
- Modify: `gui/stage_view.py` (`_on_run` line 237)
- Modify: `gui/main_window.py` (init line 58, `_on_pub_style` line 156, `closeEvent` line 217)
- Test: `tests/gui_smoke.py` (append steps [23]-[25])

**Interfaces:**
- Consumes: `CMAP_CHOICES`, `style_to_json`, `style_from_json` (Task 2).
- Produces: `StyleControls` widgets `_w_cmap_mosa_com`, `_w_cmap_mosa_fwhm`, `_w_cmap_strain`, `_w_cmap_raw`; `MainWindow._save_plot_style()`; runs receive `params["plot_style"]`.

- [ ] **Step 1: `export_dialog.py` — Colormaps section**

Add to the constants block (line ~34): `_CMAPS = list(CMAP_CHOICES)` with `CMAP_CHOICES` added to the existing `from dfxm.common.plotting import PlotStyle` import. In `_build_controls`, insert a **Colormaps** section as the FIRST section (before "Scale bar"):

```python
        # --- Colormaps section (one dropdown per quantity group) ---
        form.addRow(QLabel("<b>Colormaps</b>"))
        cmap_rows = (
            ("_w_cmap_mosa_com", "cmap_mosa_com", "Mosa misorientation"),
            ("_w_cmap_mosa_fwhm", "cmap_mosa_fwhm", "Mosa FWHM"),
            ("_w_cmap_strain", "cmap_strain", "Strain"),
            ("_w_cmap_raw", "cmap_raw", "Raw intensity"),
        )
        for attr, field_name, label in cmap_rows:
            combo = QComboBox()
            combo.addItems(_CMAPS)
            current = getattr(s, field_name)
            combo.setCurrentText(current if current in _CMAPS else _CMAPS[0])
            combo.currentTextChanged.connect(
                lambda v, f=field_name: (setattr(self._style, f, v), self._emit())
            )
            setattr(self, attr, combo)
            form.addRow(label, combo)
```

`sync_from_style` additions (with the other widget syncs):

```python
        for combo, field_name in (
            (self._w_cmap_mosa_com, "cmap_mosa_com"),
            (self._w_cmap_mosa_fwhm, "cmap_mosa_fwhm"),
            (self._w_cmap_strain, "cmap_strain"),
            (self._w_cmap_raw, "cmap_raw"),
        ):
            val = getattr(s, field_name)
            combo.setCurrentText(val if val in _CMAPS else _CMAPS[0])
```

`_all_widgets`: append `self._w_cmap_mosa_com, self._w_cmap_mosa_fwhm, self._w_cmap_strain, self._w_cmap_raw` to the returned list.

- [ ] **Step 2: `stage_view.py` — inject the style at Run**

In `_on_run` (line 237), replace the runner construction block:

```python
        self._last_params = dict(params)
        run_params = dict(params)
        window = self.window()
        if hasattr(window, "global_plot_style"):
            from dataclasses import asdict

            # Snapshot the CURRENT session publication style so every new run
            # renders with whatever the style dialog says right now.
            run_params["plot_style"] = asdict(window.global_plot_style())
        target = STAGE_TARGETS[self._stage_name]
        ...
        self._runner = StageRunner(target, run_params, start_method="spawn")
```

(`self._last_params` stays the clean form values — the export/figures path gets its style separately.)

- [ ] **Step 3: `main_window.py` — persistence**

Init (line 58) becomes:

```python
        # Session-wide publication style — restored from QSettings when a
        # previous session saved one, else seeded from the module constant.
        self._plot_style: PlotStyle = self._load_plot_style()
```

New methods next to `global_plot_style`:

```python
    @staticmethod
    def _load_plot_style() -> PlotStyle:
        from PySide6.QtCore import QSettings

        from dfxm.common.plotting import style_from_json

        raw = QSettings().value("plot_style", "")
        loaded = style_from_json(raw) if raw else None
        return loaded if loaded is not None else replace(PUBLICATION_STYLE)

    def _save_plot_style(self) -> None:
        from PySide6.QtCore import QSettings

        from dfxm.common.plotting import style_to_json

        QSettings().setValue("plot_style", style_to_json(self._plot_style))
```

`_on_pub_style`: after `dlg.exec()` add `self._save_plot_style()`.
`closeEvent` (line 217): add `self._save_plot_style()` before `super().closeEvent(event)`.

- [ ] **Step 4: gui_smoke steps** (append, following the file's existing `[N]` step pattern)

```python
# [23] publication-style dialog exposes the four colormap dropdowns
from gui.widgets.export_dialog import StyleControls

style = window.global_plot_style()
controls = StyleControls(style)
assert controls._w_cmap_mosa_com.currentText() == style.cmap_mosa_com
controls._w_cmap_strain.setCurrentText("seismic")
assert style.cmap_strain == "seismic"
step(23, "StyleControls colormap dropdowns mutate the session style")

# [24] _on_run injects the current style into the worker params
import gui.stage_view as SV

captured = {}
real_runner = SV.StageRunner


class _RecordingRunner(real_runner):
    def __init__(self, target, params, **kw):
        captured.update(params)
        super().__init__(target, params, **kw)


SV.StageRunner = _RecordingRunner
try:
    # re-trigger the concat run used earlier in this script (same params)
    concat_view._on_run()
    wait_for_run(concat_view)
finally:
    SV.StageRunner = real_runner
assert captured.get("plot_style", {}).get("cmap_strain") == "seismic"
step(24, "runs receive the live publication style (plot_style params key)")

# [25] style persists via QSettings
window._save_plot_style()
from dfxm.common.plotting import style_from_json
from PySide6.QtCore import QSettings

restored = style_from_json(QSettings().value("plot_style", ""))
assert restored is not None and restored.cmap_strain == "seismic"
step(25, "publication style round-trips through QSettings")
```

(Adapt names — `window`, `concat_view`, `step`, `wait_for_run` — to the smoke script's existing helpers; it already isolates QSettings.)

- [ ] **Step 5: Run** — `python3 tests/gui_smoke.py` (expect all steps incl. [23]-[25] green) and `python3 -m pytest -q` (all green).

- [ ] **Step 6: Commit**

```bash
git add gui/widgets/export_dialog.py gui/stage_view.py gui/main_window.py tests/gui_smoke.py
git commit -m "feat(gui): colormap dropdowns, style injection into runs, QSettings persistence"
```

---

### Task 12: Docs + full verification

**Files:**
- Modify: `docs/Usage.md` (Export/style section + stage sections whose defaults changed)
- Modify: `docs/Codebase.md` (`dfxm/common` table: new `cmaps.py` row; `plotting.py` row: new fields/helpers; `render.py`/`figures.py` signatures; per-stage rows; `gui` rows: StyleControls/`_on_run`/persistence)

- [ ] **Step 1: `docs/Usage.md`** — add a "Colormaps & publication style" subsection covering: the four group dropdowns (where: Publication style… dialog + every Export dialog), group defaults (`fast`/`magma`/`RdBu_r`/`gray`), *runs always use the style as it is when you press Run* (edit → re-run now takes effect), persistence across restarts, and the visible default changes (visualize mosa CoM magma→fast; rocking raw layers magma→gray; slices mosa CoM now the real ParaView Fast).

- [ ] **Step 2: `docs/Codebase.md`** — add `cmaps.py` module row (Fast control points, Lab interpolation, `register()`); update `plotting.py` (CMAP_GROUPS/CMAP_CHOICES, PlotStyle cmap fields + `cmap_for`, `resolve_cmap`, `style_from_params`, `style_to_json`/`style_from_json`); `render.py`/`figures.py` style/`cmap_group` params; each touched stage row (group resolution + `plot_style` params key); `export_dialog.py` (Colormaps section), `stage_view.py` (`_on_run` injection), `main_window.py` (persistence).

- [ ] **Step 3: Full verification**

```bash
python3 -m pytest -q          # expect: all pass (≥ previous 262 + new)
ruff check . && ruff format . # expect: clean / already formatted
python3 tests/gui_smoke.py    # expect: all steps green
```

- [ ] **Step 4: Commit**

```bash
git add docs/Usage.md docs/Codebase.md
git commit -m "docs: colormap groups + publication-style propagation"
```
