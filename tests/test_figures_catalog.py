import subprocess
import sys

import numpy as np

from dfxm.common import figures, render
from dfxm.common.plotting import PlotStyle
from dfxm.stages.registry import STAGE_TARGETS


def _layer():
    return np.random.default_rng(0).normal(size=(20, 40))


def test_layer_figure_legacy_has_scale_bar_and_equal_aspect():
    fig, ax, im = render.layer_figure(_layer(), -1, 1, "viridis", 40.0, 20.0, "t", "cb")
    assert ax.get_aspect() == 1.0  # physical equal aspect
    assert len(ax.patches) >= 1  # legacy scale bar present


def test_layer_figure_style_off_drops_scale_bar():
    fig, ax, im = render.layer_figure(
        _layer(),
        -1,
        1,
        "viridis",
        40.0,
        20.0,
        "t",
        "cb",
        style=PlotStyle(scale_bar=False),
    )
    assert len(ax.patches) == 0


def test_every_stage_has_a_catalog_entry():
    assert set(figures._FIGURE_CATALOGS) == set(STAGE_TARGETS)


def test_figures_for_concat_is_empty():
    assert figures.figures_for("concat", object(), {}) == []


def test_figures_for_unknown_stage_returns_empty():
    assert figures.figures_for("__no_such_stage__", object(), {}) == []


def test_importing_figures_does_not_eager_import_stage_modules():
    code = (
        "import sys, dfxm.common.figures as F; "
        "assert 'dfxm.stages.matched' not in sys.modules; "
        "assert len(F._FIGURE_CATALOGS) >= 9; "
        "print('ok')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout
