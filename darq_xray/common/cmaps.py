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

# sRGB <-> CIELAB (D65), vectorised over the trailing RGB axis.
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
