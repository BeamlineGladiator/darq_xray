"""Offscreen tests: wheel over an unfocused spin/combo field must not edit it."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QWheelEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QComboBox, QDoubleSpinBox, QSpinBox  # noqa: E402

from dfxm.config.models import Param, ParamType  # noqa: E402
from gui.widgets.param_form import ParamForm  # noqa: E402


def _wheel(widget, delta=120):
    ev = QWheelEvent(
        QPointF(5, 5),
        widget.mapToGlobal(QPointF(5, 5)),
        QPoint(0, 0),
        QPoint(0, delta),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(widget, ev)


def _form():
    params = [
        Param("count", ParamType.INT, "Count", default=3),
        Param("ratio", ParamType.FLOAT, "Ratio", default=1.5),
        Param("mode", ParamType.ENUM, "Mode", default="a", choices=("a", "b", "c")),
    ]
    return ParamForm(params)


def test_unfocused_fields_ignore_wheel():
    _app = QApplication.instance() or QApplication([])
    form = _form()
    spin = form._editors["count"]
    dspin = form._editors["ratio"]
    combo = form._editors["mode"]
    assert isinstance(spin, QSpinBox)
    assert isinstance(dspin, QDoubleSpinBox)
    assert isinstance(combo, QComboBox)
    for w in (spin, dspin, combo):
        assert w.focusPolicy() == Qt.FocusPolicy.StrongFocus
        assert not w.hasFocus()
    _wheel(spin)
    _wheel(dspin)
    _wheel(combo)
    assert form.values() == {"count": 3, "ratio": 1.5, "mode": "a"}


def test_focused_spinbox_still_wheels():
    _app = QApplication.instance() or QApplication([])
    form = _form()
    form.show()
    QApplication.processEvents()
    spin = form._editors["count"]
    spin.setFocus()
    QApplication.processEvents()
    if not spin.hasFocus():
        pytest.skip("offscreen platform denied focus")
    _wheel(spin)
    assert form.values()["count"] == 4
