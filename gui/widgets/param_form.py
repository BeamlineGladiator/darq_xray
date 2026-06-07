"""Auto-build a Qt form from a list of :class:`~dfxm.config.models.Param`.

Maps each parameter type to an editor widget:

* ``ENUM``  -> ``QComboBox`` (dropdown)
* ``BOOL``  -> ``QCheckBox``
* ``INT``   -> ``QSpinBox``
* ``FLOAT`` -> ``QDoubleSpinBox`` (6 decimals)
* ``PATH`` / ``DIR`` / ``SAVE_PATH`` -> ``QLineEdit`` + a "Browse…" button
* ``STR``   -> ``QLineEdit``

Calibration parameters (``param.calibration``) get a highlighted label and a
"⚠ calibration" suffix, because their values are physically meaningful.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QWidget,
)

from dfxm.config.models import Param, ParamType

_FLOAT_RANGE = (-1.0e12, 1.0e12)
_INT_RANGE = (-(2**31) + 1, 2**31 - 1)


class ParamForm(QWidget):
    """A form whose rows are generated from a parameter schema.

    Use :meth:`values` to read coerced values and :meth:`set_values` to load a
    dict (e.g. experiment-derived defaults) back into the widgets.
    """

    changed = Signal()

    def __init__(
        self,
        params: Sequence[Param],
        values: dict[str, Any] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._params = list(params)
        self._getters: dict[str, Callable[[], Any]] = {}
        self._setters: dict[str, Callable[[Any], None]] = {}

        layout = QFormLayout(self)
        layout.setLabelAlignment(layout.labelAlignment())
        initial = values or {}
        for p in self._params:
            editor = self._build_editor(p, initial.get(p.name, p.default))
            layout.addRow(self._label_for(p), editor)

    # -- public API -------------------------------------------------------
    def values(self) -> dict[str, Any]:
        """Current values, coerced to each parameter's declared type."""
        out: dict[str, Any] = {}
        for p in self._params:
            raw = self._getters[p.name]()
            out[p.name] = p.coerce(raw) if raw is not None else None
        return out

    def set_values(self, values: dict[str, Any]) -> None:
        """Load *values* into the widgets (unknown keys ignored)."""
        for name, val in values.items():
            if name in self._setters and val is not None:
                self._setters[name](val)

    # -- label ------------------------------------------------------------
    def _label_for(self, p: Param) -> QLabel:
        text = p.label
        if p.unit:
            text += f" ({p.unit})"
        if p.calibration:
            text += "  ⚠ calibration"
        lbl = QLabel(text)
        if p.calibration:
            lbl.setStyleSheet("color: #b00020; font-weight: bold;")
        if p.help:
            lbl.setToolTip(p.help)
        return lbl

    # -- editors ----------------------------------------------------------
    def _build_editor(self, p: Param, value: Any) -> QWidget:
        if p.type is ParamType.ENUM:
            return self._enum_editor(p, value)
        if p.type is ParamType.BOOL:
            return self._bool_editor(p, value)
        if p.type is ParamType.INT:
            return self._int_editor(p, value)
        if p.type is ParamType.FLOAT:
            return self._float_editor(p, value)
        if p.type in (ParamType.PATH, ParamType.DIR, ParamType.SAVE_PATH):
            return self._path_editor(p, value)
        if p.type is ParamType.TEXT:
            return self._text_editor(p, value)
        return self._str_editor(p, value)

    def _register(self, name, getter, setter, signal=None) -> None:
        self._getters[name] = getter
        self._setters[name] = setter
        if signal is not None:
            signal.connect(self.changed)

    def _enum_editor(self, p: Param, value: Any) -> QWidget:
        box = QComboBox()
        choices = [str(c) for c in (p.choices or ())]
        box.addItems(choices)
        if value is not None and str(value) in choices:
            box.setCurrentText(str(value))
        if p.help:
            box.setToolTip(p.help)
        self._register(
            p.name, box.currentText, lambda v: box.setCurrentText(str(v)), box.currentTextChanged
        )
        return box

    def _bool_editor(self, p: Param, value: Any) -> QWidget:
        cb = QCheckBox()
        cb.setChecked(bool(value))
        if p.help:
            cb.setToolTip(p.help)
        self._register(p.name, cb.isChecked, lambda v: cb.setChecked(bool(v)), cb.toggled)
        return cb

    def _int_editor(self, p: Param, value: Any) -> QWidget:
        sb = QSpinBox()
        sb.setRange(*_INT_RANGE)
        if value is not None:
            sb.setValue(int(value))
        if p.help:
            sb.setToolTip(p.help)
        self._register(p.name, sb.value, lambda v: sb.setValue(int(v)), sb.valueChanged)
        return sb

    def _float_editor(self, p: Param, value: Any) -> QWidget:
        sb = QDoubleSpinBox()
        sb.setDecimals(6)
        sb.setRange(*_FLOAT_RANGE)
        sb.setSingleStep(0.001)
        if value is not None:
            sb.setValue(float(value))
        if p.help:
            sb.setToolTip(p.help)
        self._register(p.name, sb.value, lambda v: sb.setValue(float(v)), sb.valueChanged)
        return sb

    def _str_editor(self, p: Param, value: Any) -> QWidget:
        le = QLineEdit()
        if value is not None:
            le.setText(str(value))
        if p.help:
            le.setToolTip(p.help)
        self._register(p.name, le.text, lambda v: le.setText(str(v)), le.textChanged)
        return le

    def _text_editor(self, p: Param, value: Any) -> QWidget:
        te = QPlainTextEdit()
        te.setMinimumHeight(120)
        if value is not None:
            te.setPlainText(str(value))
        if p.help:
            te.setToolTip(p.help)
        self._register(p.name, te.toPlainText, lambda v: te.setPlainText(str(v)), te.textChanged)
        return te

    def _path_editor(self, p: Param, value: Any) -> QWidget:
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        le = QLineEdit()
        if value is not None:
            le.setText(str(value))
        if p.help:
            le.setToolTip(p.help)
        browse = QPushButton("Browse…")

        def pick() -> None:
            start = le.text() or ""
            if p.type is ParamType.DIR:
                chosen = QFileDialog.getExistingDirectory(self, p.label, start)
            elif p.type is ParamType.SAVE_PATH:
                chosen, _ = QFileDialog.getSaveFileName(self, p.label, start)
            else:
                chosen, _ = QFileDialog.getOpenFileName(self, p.label, start)
            if chosen:
                le.setText(chosen)

        browse.clicked.connect(pick)
        row.addWidget(le, 1)
        row.addWidget(browse)
        self._register(p.name, le.text, lambda v: le.setText(str(v)), le.textChanged)
        return container
