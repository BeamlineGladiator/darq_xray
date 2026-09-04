"""FitFigureHost — WYSIWYG preview host that scales the figure's *dpi* to fit the
available widget area while keeping its physical size in inches (so fonts and
line widths in points stay proportional, exactly as in the export)."""

import pytest
from matplotlib.figure import Figure
from PySide6.QtWidgets import QApplication

from darq_xray.gui.widgets.fit_canvas import FitFigureHost, fit_dpi

_ = QApplication.instance() or QApplication([])


def test_fit_dpi_limits_by_the_tighter_axis():
    assert fit_dpi((20.0, 10.0), (400, 400)) == pytest.approx(20.0)  # width-bound
    assert fit_dpi((20.0, 10.0), (800, 100)) == pytest.approx(10.0)  # height-bound
    assert fit_dpi((20.0, 10.0), (0, 0)) >= 1.0  # never zero/negative


def test_host_keeps_figure_inches_and_scales_dpi():
    fig = Figure(figsize=(20.0, 10.0), dpi=600)
    host = FitFigureHost(fig)
    host.resize(400, 400)
    host.show()
    QApplication.processEvents()
    assert host.canvas.figure is fig
    assert tuple(fig.get_size_inches()) == pytest.approx((20.0, 10.0))
    ratio = host.canvas.device_pixel_ratio
    assert fig.dpi == pytest.approx(20.0 * ratio, rel=0.05)
    assert host.canvas.width() == pytest.approx(400, abs=2)
    assert host.canvas.height() == pytest.approx(200, abs=2)
    # shrink -> smaller dpi, same inches
    host.resize(200, 400)
    QApplication.processEvents()
    assert fig.dpi == pytest.approx(10.0 * ratio, rel=0.05)
    assert tuple(fig.get_size_inches()) == pytest.approx((20.0, 10.0))
    host.close()
