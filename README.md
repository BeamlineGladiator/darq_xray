# DARQ

**A desktop app that turns X-ray microscopy measurements of a crystal into
maps of how that crystal is stretched and bent inside.**

The technique is **Dark-Field X-ray Microscopy (DFXM)**: you image a single
grain deep inside a sample through its diffracted beam, rocking the crystal
through its diffraction condition so that every detector pixel records a small
intensity-versus-angle curve. Fitting those curves tells you, pixel by pixel,
how the lattice is spaced and which way it is tilted — **strain** and
**mosaicity**.

DARQ takes that measurement from the raw beamline files all the way to finished
figures, in one window: nine stages, forms instead of scripts, and your
calibration saved as a reusable preset. It is built around data from the
**ESRF ID03** beamline.

![The DARQ main window: the pipeline rail on the left, the active experiment
preset above it, and one sentence per stage on the Overview
page](docs/images/main-window.png)

## Contents

- [Is this for you?](#is-this-for-you)
- [What you need first](#what-you-need-first)
- [Install](#install)
- [Launch it](#launch-it)
- [Your first run](#your-first-run)
- [What the nine stages do](#what-the-nine-stages-do)
- [Before you trust a strain map](#before-you-trust-a-strain-map)
- [Using DARQ after darfix](#using-darq-after-darfix)
- [Documentation](#documentation)
- [Development](#development)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Is this for you?

Probably yes, if:

- you have **DFXM scans from ESRF ID03** — BLISS-written `.h5` files, one
  folder per sample layer;
- you use **darfix** (the ESRF tool that fits the rocking curve at every pixel)
  and end up with a `maps.h5`;
- and what you want out is **axial strain maps**, **mosaicity maps**, aligned
  3-D volumes, **ParaView** exports, oblique slices through those volumes, line
  profiles across them, or publication figures composed from any of the above.

Probably not, if your data comes from a different beamline or instrument. The
motor names, the file layout and the calibration are ID03-specific. They are
all overridable settings rather than hard-coded constants, so adapting is
possible — but it is adapting, not configuring.

> [!NOTE]
> DARQ is an **analysis program for the output of
> [darfix](https://gitlab.esrf.fr/XRD/darfix)** — it starts where darfix stops.
> It does not replace darfix and does not run it. It **brackets** it: DARQ
> prepares darfix's input, you run darfix yourself, and DARQ picks up again
> from the `maps.h5` that darfix writes. See
> [Using DARQ after darfix](#using-darq-after-darfix).

## What you need first

| | |
|---|---|
| **Python 3.10 or newer** | Check with `python3 --version`. |
| **A desktop with a display** | This is a windowed application. (Every stage *also* has a command-line version — see [Running without the GUI](docs/Usage.md#running-without-the-gui-cli) — but the app itself needs a screen.) |
| **[darfix](https://gitlab.esrf.fr/XRD/darfix)**, installed separately | Not bundled here. You run it between two of DARQ's stages. |
| **Your data** | The raw scan folders, and/or the `maps.h5` darfix produced from them. |
| `ffmpeg` *(optional)* | On your `PATH`, it lets DARQ export animations as MP4. Without it they come out as GIF instead — nothing else changes. |

Disk and memory depend entirely on your dataset. DARQ measures your machine and
tells you what a run will cost *before* you start it, and falls back to
streaming from disk when a volume does not fit in RAM.

## Install

Copy these five lines into a terminal:

```bash
git clone https://github.com/BeamlineGladiator/darq_xray.git
cd darq_xray
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -e .
```

That is it — the last line installs everything DARQ needs.

<details>
<summary><b>What those lines do, and what to try if one fails</b></summary>

**`python3 -m venv venv` — what is a venv, and why?**
A *virtual environment* is a private folder of Python packages belonging to
this project alone. DARQ needs a fairly heavy graphics stack (Qt, VTK); a venv
keeps all of it out of your system Python, where it could collide with
something else you have installed. The folder is called `venv` and appears
right inside the clone. Delete it and re-run the two commands to start over —
nothing else is affected.

You must run `source venv/bin/activate` **once per terminal** before launching
DARQ. Your prompt shows `(venv)` when it is active.

**`error: externally-managed-environment`**
You skipped the venv. On Debian, Ubuntu and their derivatives the system Python
refuses to install anything into itself and suggests `--break-system-packages`.
Do not do that — make the venv instead. That is exactly what it is for.

**`pip install -e .` — why the `-e`?**
`-e` means *editable*: DARQ runs from this folder rather than being copied
elsewhere. **This is not optional here.** The shipped experiment presets live
in `experiments/`, beside the package rather than inside it, so they are not
installed as package data. After a plain `pip install .` the app starts with an
empty preset dropdown and no calibration.

**`pip` is not found, or installs into the wrong Python**
Use `python3 -m pip install -e .` instead — that guarantees the pip belonging
to the Python you just made the venv with.

</details>

## Launch it

```bash
source venv/bin/activate    # once per terminal
darq_xray
```

## Your first run

1. **The app opens on the Overview page** — the pipeline drawn left to right,
   one sentence per stage. Clicking a stage there, or in the rail down the left
   edge, opens its form.

2. **Pick an experiment preset** from the dropdown at the top. One ships with
   the app, `STO2_overnight`. An *experiment* holds everything the stages
   share: where your data lives, the calibration angles, the pixel scales, the
   regions of interest.

   Its `raw_root` and `processed_root` are written as `/path/to/data/ESRF/…`
   placeholders — they show the expected layout, not a real location. Open
   **Edit…** and point them at your own copy. Better still, use **Initialize
   from data…**, which reads your files and proposes the values for you.

3. **Run `Concatenate`** if your scans are not already joined into single
   darfix-ready files. It is marked *(optional)* in the rail, and it is: skip
   it if that is already done.

4. **Leave the app and run darfix yourself** on those files. It writes a
   `maps.h5`. DARQ shows darfix as a greyed-out row in the pipeline for exactly
   this reason — it is the one step DARQ does not perform.

5. **Come back and run `Axial strain`**, pointing it at the folder holding your
   `maps.h5` files. Click into any field and the help panel under the form
   explains it. Below the **Run** button, a cost line tells you what the run
   will need in memory, disk and time before you commit to it.

6. **You should now have** `stacked_strain_volumes.h5` and a folder of PNG maps
   under your output directory, and a green banner across the top of the panel.
   If the banner is red instead, it names what went wrong and, usually, how to
   fix it.

![A stage running: the parameter form in the middle, progress and the estimated
time remaining under the Run button, and the live log on the
right](docs/images/stage-panel.png)

From there, `Mosaicity`, `Visualize`, `ParaView export` and the rest all read
what you have just produced — each stage pre-fills its inputs from the one
before it.

The full walkthrough, field by field, is in
**[the user guide](docs/Usage.md)**.

## What the nine stages do

| Stage | What it does |
|---|---|
| [**Concatenate**](docs/Usage.md#1-concatenate) | Joins the per-scan `.h5` files of one sample layer into the single file darfix expects. *Optional* — skip it if your data is already in that form. |
| *darfix* | **External — you run this yourself.** Fits the rocking curve at every pixel and writes `maps.h5`. Not part of DARQ. |
| [**Axial strain**](docs/Usage.md#2-axial-strain) | Turns darfix's fitted angles into maps of relative lattice-spacing change, then stacks the layers into a 3-D volume. |
| [**Mosaicity volume**](docs/Usage.md#3-mosaicity-volume) | Stacks darfix's orientation maps into a 3-D volume showing how the lattice tilts from place to place. |
| [**Aligned rocking volumes**](docs/Usage.md#4-aligned-rocking-volumes) | Rebuilds the raw measured intensity as a 3-D volume on the same grid as the maps above, so raw and fitted data can be compared directly. |
| [**Visualize volumes**](docs/Usage.md#5-visualize-volumes) | Figures and animations of any of those volumes — slice views, top views, rotating 3-D. |
| [**ParaView export**](docs/Usage.md#6-paraview-export) | Writes the volumes as `.pvti` files, all in one shared coordinate frame, for viewing in ParaView. |
| [**Oblique slices**](docs/Usage.md#7-oblique-slices) | Cuts flat planes at any angle through the volumes, so you can look along a feature that does not lie on X, Y or Z. |
| [**Line profiles**](docs/Usage.md#8-line-profiles) | Measures values along a line drawn across those slices, as a figure and as CSV. |
| [**Rocking-matched layers**](docs/Usage.md#9-rocking-matched-layers) | Plots raw rocking data beside the strain result for the same layer, matched up. |

On top of these, a **figure builder** composes finished maps, traces and images
into multi-panel publication figures at a fixed, exact µm-per-cm scale.

![The figure builder: a multi-panel figure assembled from maps and line
profiles, with the panel outline on the left and the style controls on the
right](docs/images/figure-builder.png)

## Before you trust a strain map

> [!WARNING]
> **The calibration is physical, and it is yours.** The reference angle
> (`ccmth reference`) and the pixel scales in µm/px describe *your* beamtime and
> *your* optics. Get them wrong and DARQ will happily produce a strain map that
> looks entirely plausible and means nothing.
>
> The app marks these fields in red with **⚠ calibration** so they are hard to
> set carelessly, and its **Compute pixel size from scan…** button derives the
> pixel scales from your own scan's motor positions. Confirm them against the
> beamline calibration before you draw any conclusion from a map.

Two more things worth knowing before interpreting output, both explained in the
user guide: a DFXM dataset carries **two different regions of interest in two
different coordinate frames**
([which is which](docs/Usage.md#regions-of-interest-two-windows-two-frames)),
and every volume stage co-registers through **one shared alignment**, so
coordinates stay comparable between them.

## Using DARQ after darfix

DARQ is **post-darfix analysis**. Fitting the rocking curve at every pixel is
darfix's job, not DARQ's. What DARQ does is take the per-pixel results darfix
has already fitted and turn them into physics you can publish: calibrated
strain, mosaicity, co-registered 3-D volumes, ParaView exports, slices,
profiles and figures. Two stages touch raw scan files — `concat`, which
prepares darfix's input, and `rocking`, which builds an aligned raw volume
alongside the fitted maps. Everything else works from `maps.h5`.

The division of labour, in one line:

```
raw scans → [ DARQ: concat ] → [ darfix: fit every pixel → maps.h5 ] → [ DARQ: everything downstream ]
```

darfix is a separate program by a separate group, and DARQ neither bundles nor
launches it. Install it from its own repository and run it yourself:

- **Source:** <https://gitlab.esrf.fr/XRD/darfix>
- **Documentation:** <http://www.edna-site.org/pub/doc/darfix/latest>

If darfix contributed to a result you publish, cite the darfix paper:

> Garriga Ferrer, J., Rodríguez-Lamas, R., Payno, H., De Nolf, W., Cook, P.,
> Solé, V. A., Yildirim, C. & Detlefs, C. (2023). *darfix* — data analysis for
> dark-field X-ray microscopy. *Journal of Synchrotron Radiation* **30**,
> 527–537. <https://doi.org/10.1107/S1600577523001674>

## Documentation

| | For | |
|---|---|---|
| **[docs/Usage.md](docs/Usage.md)** | Anyone using the app | Every stage, every viewer, the figure builder — and a glossary of the terms the rest of it assumes. |
| **[docs/Codebase.md](docs/Codebase.md)** | Anyone changing the code | Module by module, class by class, function by function. |
| **[CLAUDE.md](CLAUDE.md)** | Contributors | The conventions and the invariants the test suite enforces. |

Both docs are written to render identically on GitHub and in Obsidian —
standard Markdown links, `<details>` blocks, and only the five alert types
(`> [!NOTE]`, `> [!TIP]`, `> [!IMPORTANT]`, `> [!WARNING]`, `> [!CAUTION]`)
that both tools understand. `tests/test_docs_links.py` enforces it.

## Development

```bash
pip install -e ".[test]"      # the test extra: pytest + pytest-timeout
python3 -m pytest -q          # the suite
ruff check . && ruff format . # lint + format
```

CI runs both on every push and pull request.

The code is two layers with a one-way dependency. **`darq_xray/`** is the
Qt-free core: one module per stage under `darq_xray/stages/`, each exposing
`run(params, progress)` and its own `__main__` CLI, plus shared helpers in
`darq_xray/common/`. **`darq_xray/gui/`** is the PySide6 app on top, building
each stage's form from that stage's declared parameter schema. Nothing outside
`darq_xray/gui/` imports Qt, which is what keeps every stage runnable and
testable headless. `docs/Codebase.md` maps the whole thing.

Two rules are worth knowing before your first change, because tests enforce
them: `pyproject.toml` is the single source of truth for dependencies and is
checked against the lists in `README.md` and `docs/Usage.md`; and a change to a
stage's parameters or behaviour must update `docs/Usage.md` and
`docs/Codebase.md` in the *same* change.

### Dependencies

Installed for you by the install command; listed here only so you can see what
you are pulling in.

<!-- deps:start -->
`numpy`, `h5py`, `scipy`, `matplotlib`, `pyyaml`, `psutil`, `PySide6`,
`pyvista`, `pyvistaqt`, `vtk`
<!-- deps:end -->

<details>
<summary><b>How the supported dependency range is tested</b></summary>

Only lower bounds are declared, plus one exclusion: **PySide6 6.12.0**, which
crashes the interpreter when DARQ quits after the Add panels dialog has been
opened (a PySide6 regression; 6.11 is unaffected).
`tests/test_gui_app_quit.py` reproduces it, so the exclusion can be lifted once
that test passes on a later 6.12 release. The suite is kept green across the
span they allow. Both ends are exercised **by CI on every push**: one matrix leg
**pins the declared floor** — **numpy 1.26.4 / scipy 1.11.4 / matplotlib
3.6.3** — and the other takes the newest compatible release, currently
**numpy 2.5.2 / scipy 1.18.1 / matplotlib 3.11.1**.

The pin is deliberate. An older Python alone does **not** select old
dependencies — pip resolves the newest compatible wheel — so without it CI
silently tested an untried middle set and neither end.

The two numpy generations genuinely differ where it matters here:
`np.percentile` on a float32 array returns float64 on 1.x and float32 on 2.x.
Tests therefore derive their oracle from the installed numpy rather than
hardcoding one generation's value.

Two constants that price memory are pinned to the versions they were *measured*
against and fail loudly on an unverified one, rather than drifting quietly:
`tests/test_common_alignment.py::VERIFIED_SCIPY_VERSIONS` and
`slices.RSS_FLOOR_BYTES`. A third, `volumeio.centring_scaffold_bytes`, is a
*bound* rather than a pin and carries deliberate headroom, because its measured
cost varies with the machine and not only the numpy version — see its
docstring.

Tests also run against an isolated `QSettings` store (`tests/conftest.py`), so
a result never depends on how the person running them last used the app.

</details>

## Acknowledgements

DARQ was developed at the **Karlsruhe Institute of Technology (KIT)** by
**Albert Zelenika**, at the **Institute for Applied Materials – Mechanics of
Materials and Interfaces (IAM-MMI)**, under the supervision of
**Dr. habil. Xufei Fang** and with the help of **Dr. Oliver Preuss**.

## License

MIT — see [LICENSE](LICENSE).

The dependencies keep their own licenses; note in particular that **PySide6 is
LGPLv3**. Running and modifying this source is unaffected, but if you
redistribute a *bundled binary* of the app, LGPL relinking obligations apply to
the Qt part.
