"""Auto-build a Qt form from a list of :class:`~darq_xray.config.models.Param`.

Maps each parameter type to an editor widget:

* ``ENUM``  -> ``QComboBox`` (dropdown)
* ``BOOL``  -> ``QCheckBox``
* ``INT``   -> ``QSpinBox``
* ``FLOAT`` -> ``QDoubleSpinBox`` (6 decimals)
* ``PATH`` / ``DIR`` / ``SAVE_PATH`` -> ``QLineEdit`` + a "Browse…" button
* ``STR``   -> ``QLineEdit``

A param may override that mapping with a ``Param.editor`` render hint:
``"summary_json"`` renders a ``TEXT`` param as a
:class:`~darq_xray.gui.widgets.jobs_summary.JobsSummaryEditor` (one-line summary + an
"Edit raw JSON…" dialog); ``"clim_table"`` renders one as a
:class:`~darq_xray.gui.widgets.clim_table.ClimTableEditor` (one-line summary + a labelled
per-volume vmin/vmax dialog). An unknown hint falls back to the type's editor.

An ``ENUM`` param carrying ``choice_labels`` shows those labels and stores the
matching ``choices`` entry (``currentData``, not ``currentText``), so a
self-describing row like "geom — high intensities" never leaks into the value.

Calibration parameters (``param.calibration``) get a highlighted label and a
"⚠ calibration" suffix, because their values are physically meaningful.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable

from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
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
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from darq_xray.config.models import Param, ParamType, SeeAlso

from .help_panel import param_help_html
from .wheel_guard import install_wheel_guard

_FLOAT_RANGE = (-1.0e12, 1.0e12)
_INT_RANGE = (-(2**31) + 1, 2**31 - 1)


class ParamForm(QWidget):
    """A form whose rows are generated from a parameter schema.

    Essentials (``advanced=False``) render first, in spec order; advanced
    params collapse into one "Advanced (N settings)" expander, grouped under
    their ``group`` headers. Use :meth:`values` to read coerced values and
    :meth:`set_values` to load a dict back into the widgets.
    """

    changed = Signal()
    focusedParamChanged = Signal(object)  # the focused Param
    focusCleared = Signal()  # focus left every field in this form

    def __init__(
        self,
        params: Sequence[Param],
        values: dict[str, Any] | None = None,
        parent: QWidget | None = None,
        see_also: Sequence[SeeAlso] = (),
    ) -> None:
        super().__init__(parent)
        self._params = list(params)
        self._getters: dict[str, Callable[[], Any]] = {}
        self._setters: dict[str, Callable[[Any], None]] = {}
        self._editors: dict[str, QWidget] = {}
        self._param_for_widget: dict[QObject, Param] = {}
        self._param_by_name: dict[str, Param] = {p.name: p for p in self._params}
        self._adv_toggle: QToolButton | None = None
        self._adv_box: QWidget | None = None
        self._labels: dict[str, QLabel] = {}
        self._base_label: dict[str, str] = {}
        self._notes: dict[str, QLabel] = {}
        self._errors: dict[str, QLabel] = {}
        self._see_also_labels: dict[str, QLabel] = {}
        self._see_also_by_param = {s.param_name: s for s in see_also if s.param_name}
        self._stage_see_also = tuple(s for s in see_also if not s.param_name)

        initial = values or {}
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        for entry in self._stage_see_also:
            outer.addWidget(self._see_also_label("", entry.text))

        essentials = [p for p in self._params if not p.advanced]
        advanced = [p for p in self._params if p.advanced]

        ess_form = QFormLayout()
        for p in essentials:
            ess_form.addRow(self._label_for(p), self._make_editor(p, initial))
            self._add_note_row(ess_form, p)
            self._add_see_also_row(ess_form, p)
        outer.addLayout(ess_form)

        if advanced:
            self._adv_toggle = QToolButton()
            self._adv_toggle.setText(f"Advanced ({len(advanced)} settings)")
            self._adv_toggle.setCheckable(True)
            self._adv_toggle.setArrowType(Qt.ArrowType.RightArrow)
            self._adv_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            self._adv_box = QWidget()
            adv_layout = QVBoxLayout(self._adv_box)
            adv_layout.setContentsMargins(12, 0, 0, 0)
            group_forms: dict[str, QFormLayout] = {}
            for p in advanced:
                form = group_forms.get(p.group)
                if form is None:
                    header = QLabel(p.group)
                    header.setProperty("role", "group-header")
                    adv_layout.addWidget(header)
                    form = QFormLayout()
                    adv_layout.addLayout(form)
                    group_forms[p.group] = form
                form.addRow(self._label_for(p), self._make_editor(p, initial))
                self._add_note_row(form, p)
                self._add_see_also_row(form, p)
            self._adv_box.setVisible(False)
            self._adv_toggle.toggled.connect(self._on_adv_toggled)
            outer.addWidget(self._adv_toggle)
            outer.addWidget(self._adv_box)

        app = QApplication.instance()
        if app is not None:
            app.focusChanged.connect(self._on_focus_changed)

    def _make_editor(self, p: Param, initial: dict[str, Any]) -> QWidget:
        editor = self._build_editor(p, initial.get(p.name, p.default))
        self._editors[p.name] = editor
        entry = self._see_also_by_param.get(p.name)
        tip = param_help_html(p, see_also=entry.text if entry else "")
        for w in (editor, *editor.findChildren(QWidget)):
            w.installEventFilter(self)
            self._param_for_widget[w] = p
            w.setToolTip(tip)
        return editor

    def _add_note_row(self, form: QFormLayout, p: Param) -> None:
        """Hidden, full-width note rows under *p*'s editor.

        Up to two, and only for params that ask for one — a hidden row per
        field would be dead weight on every form. An ``advice_key`` param gets
        the advisory row (`apply_hints`), a ``roi_axis`` param the error row
        (`apply_roi_problems`), and a param declaring both gets both. The
        editor itself is NOT wrapped: `self._editors[name]` must stay the real
        widget, which `gui_smoke` and the wheel-guard tests reach into directly.
        """
        if p.roi_axis:
            # A SECOND hidden row, not a restyled `_notes` one. The two carry
            # different severities (advice vs "this run would compute
            # nothing"), and a QLabel that swaps its `role` property mid-life
            # needs an unpolish/polish that this project has already been
            # bitten by — the pre-flight banner's cached geometry survived one
            # (see `StageView._show_banner`). Two labels, two fixed roles, no
            # restyling.
            err = QLabel("")
            err.setWordWrap(True)
            err.setProperty("role", "error")
            err.setVisible(False)
            form.addRow(err)
            self._errors[p.name] = err
        if not p.advice_key:
            return
        note = QLabel("")
        note.setWordWrap(True)
        note.setProperty("role", "warning")
        note.setVisible(False)
        form.addRow(note)
        self._notes[p.name] = note

    def _see_also_label(self, key: str, text: str) -> QLabel:
        """A quiet, always-visible pointer row (role="hint", never hidden).

        Distinct from `_add_note_row`'s role="warning" advisory rows, which
        start hidden and carry cost warnings: a pointer is static text whose
        whole purpose is being visible without being sought.
        """
        label = QLabel(text)
        label.setWordWrap(True)
        label.setProperty("role", "hint")
        self._see_also_labels[key] = label
        return label

    def _add_see_also_row(self, form: QFormLayout, p: Param) -> None:
        """A pointer row under *p*'s editor, when the spec declares one.

        The editor itself is NOT wrapped: `self._editors[name]` must stay the
        real widget, which `gui_smoke` and the wheel-guard tests reach into
        directly — the same constraint `_add_note_row` documents.

        An anchor naming a parameter this form does not have simply never
        matches and renders nothing; `StageSpec.see_also_problems()` is what
        turns that into a test failure, so it must not also crash the form.
        """
        entry = self._see_also_by_param.get(p.name)
        if entry is None:
            return
        form.addRow(self._see_also_label(p.name, entry.text))

    def _on_adv_toggled(self, checked: bool) -> None:
        assert self._adv_toggle is not None and self._adv_box is not None
        self._adv_box.setVisible(checked)
        self._adv_toggle.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )

    def eventFilter(self, obj: QObject, event) -> bool:  # noqa: N802 - Qt API
        if event.type() == QEvent.Type.FocusIn and obj in self._param_for_widget:
            self.focusedParamChanged.emit(self._param_for_widget[obj])
        return super().eventFilter(obj, event)

    def _on_focus_changed(self, old: QObject, new: QObject) -> None:  # Qt slot
        # Only react when focus leaves one of *our* fields for something that
        # is not one of our fields -> revert the help panel to the stage text.
        if old in self._param_for_widget and new not in self._param_for_widget:
            self.focusCleared.emit()

    def focus_param(self, name: str) -> None:
        """Reveal (if advanced) and focus the editor for *name*."""
        editor = self._editors.get(name)
        param = self._param_by_name.get(name)
        if editor is None or param is None:
            return
        if param.advanced and self._adv_toggle is not None:
            self._adv_toggle.setChecked(True)
        target = editor.findChild(QLineEdit) or editor
        target.setFocus()

    # -- public API -------------------------------------------------------
    def values(self) -> dict[str, Any]:
        """Current values, coerced to each parameter's declared type."""
        out: dict[str, Any] = {}
        for p in self._params:
            raw = self._getters[p.name]()
            out[p.name] = p.coerce(raw) if raw is not None else None
        return out

    def set_values(self, values: dict[str, Any]) -> None:
        """Load *values* into the widgets (unknown keys ignored; ``None`` skipped)."""
        for name, val in values.items():
            if name in self._setters and val is not None:
                self._setters[name](val)

    def reset_values(self, values: dict[str, Any]) -> None:
        """Set *every* param to ``values[name]``, clearing to a type-appropriate
        empty when the value is missing or ``None``.

        Unlike :meth:`set_values` (which skips ``None``), this leaves no field at
        its previous value — used to fully reset the form to a new baseline (e.g.
        an experiment switch) so a ``None``-default field can't retain a stale
        entry from the prior context.
        """
        for p in self._params:
            setter = self._setters.get(p.name)
            if setter is None:
                continue
            val = values.get(p.name)
            setter(self._empty_value(p) if val is None else val)

    def set_field_note(self, name: str, text: str) -> None:
        """Show *text* under *name*'s editor; empty text hides the row."""
        note = self._notes.get(name)
        if note is None:
            return
        note.setText(text)
        note.setVisible(bool(text))

    def apply_hints(self, hints: dict) -> None:
        """Route an advisory's hints to their fields, clearing every other note.

        Clearing matters: a hint that no longer applies (the user picked a
        lighter render mode) must disappear rather than linger as advice about
        a setting they already changed.
        """
        for p in self._params:
            if p.advice_key:
                self.set_field_note(p.name, hints.get(p.advice_key, ""))

    def apply_roi_problems(self, problems) -> None:
        """Show each :class:`~darq_xray.common.roi.RoiProblem` under its own field.

        Clearing every other row matters as much as setting these: the check
        re-runs on each keystroke, and a message about an ROI the user has
        already corrected is worse than none — it would keep pointing at a
        field that is now right.

        Several problems on one field (a four-int ``roi`` can have a bad row
        range AND a bad column range) stack into one label, worst first, since
        the caller hands them sorted.
        """
        texts: dict[str, list[str]] = {}
        for problem in problems:
            if problem.param in self._errors:
                texts.setdefault(problem.param, []).append(problem.message)
        for name, label in self._errors.items():
            lines = texts.get(name, ())
            label.setText("  ".join(f"⚠ {line}" for line in lines))
            label.setVisible(bool(lines))

    def set_field_marker(self, name: str, marked: bool, tooltip: str = "") -> None:
        """Toggle a '⚠' suffix on *name*'s row label (deviates-from-experiment)."""
        lbl = self._labels.get(name)
        p = self._param_by_name.get(name)
        if lbl is None or p is None:
            return
        base = self._base_label[name]
        lbl.setText(f"{base}  ⚠" if marked else base)
        lbl.setToolTip(tooltip if (marked and tooltip) else param_help_html(p))

    @staticmethod
    def _empty_value(p: Param) -> Any:
        """The cleared value for *p*, matching how its editor renders a ``None``."""
        if p.type is ParamType.BOOL:
            return False
        if p.type is ParamType.INT:
            return 0
        if p.type is ParamType.FLOAT:
            return 0.0
        if p.type is ParamType.ENUM:
            return str(p.choices[0]) if p.choices else ""
        return ""

    # -- label ------------------------------------------------------------
    def _label_for(self, p: Param) -> QLabel:
        text = p.label
        if p.unit:
            text += f" ({p.unit})"
        if p.calibration:
            text += "  ⚠ calibration"
        lbl = QLabel(text)
        if p.calibration:
            lbl.setProperty("role", "calib")
        lbl.setToolTip(param_help_html(p))
        self._labels[p.name] = lbl
        self._base_label[p.name] = text
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
        if p.type is ParamType.TEXT and p.editor == "summary_json":
            return self._summary_json_editor(p, value)
        if p.type is ParamType.TEXT and p.editor == "clim_table":
            return self._clim_table_editor(p, value)
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
        install_wheel_guard(box)
        choices = [str(c) for c in (p.choices or ())]
        if not p.choice_labels:
            box.addItems(choices)
            if value is not None and str(value) in choices:
                box.setCurrentText(str(value))
            self._register(
                p.name,
                box.currentText,
                lambda v: box.setCurrentText(str(v)),
                box.currentTextChanged,
            )
            return box
        # Labelled: the row SHOWS the description and STORES the bare choice.
        # Only params that opt in take this path — every other ENUM keeps the
        # `currentText` getter above, so the change cannot reach a value that
        # already round-trips through saved form state.
        for choice, label in zip(choices, p.choice_labels):
            box.addItem(str(label), choice)
        if value is not None:
            index = box.findData(str(value))
            if index >= 0:
                box.setCurrentIndex(index)

        def _set(v, _box=box) -> None:
            index = _box.findData(str(v))
            if index >= 0:
                _box.setCurrentIndex(index)

        self._register(p.name, box.currentData, _set, box.currentTextChanged)
        return box

    def _bool_editor(self, p: Param, value: Any) -> QWidget:
        cb = QCheckBox()
        cb.setChecked(bool(value))
        self._register(p.name, cb.isChecked, lambda v: cb.setChecked(bool(v)), cb.toggled)
        return cb

    def _int_editor(self, p: Param, value: Any) -> QWidget:
        sb = QSpinBox()
        install_wheel_guard(sb)
        sb.setRange(*_INT_RANGE)
        if value is not None:
            sb.setValue(int(value))
        self._register(p.name, sb.value, lambda v: sb.setValue(int(v)), sb.valueChanged)
        return sb

    def _float_editor(self, p: Param, value: Any) -> QWidget:
        sb = QDoubleSpinBox()
        install_wheel_guard(sb)
        sb.setDecimals(6)
        sb.setRange(*_FLOAT_RANGE)
        sb.setSingleStep(0.001)
        if value is not None:
            sb.setValue(float(value))
        self._register(p.name, sb.value, lambda v: sb.setValue(float(v)), sb.valueChanged)
        return sb

    def _str_editor(self, p: Param, value: Any) -> QWidget:
        le = QLineEdit()
        if value is not None:
            le.setText(str(value))
        self._register(p.name, le.text, lambda v: le.setText(str(v)), le.textChanged)
        return le

    def _text_editor(self, p: Param, value: Any) -> QWidget:
        te = QPlainTextEdit()
        te.setMinimumHeight(120)
        if value is not None:
            te.setPlainText(str(value))
        self._register(p.name, te.toPlainText, lambda v: te.setPlainText(str(v)), te.textChanged)
        return te

    def _summary_json_editor(self, p: Param, value: Any) -> QWidget:
        from .jobs_summary import JobsSummaryEditor

        ed = JobsSummaryEditor("" if value is None else str(value), p.label)
        self._register(p.name, ed.text, ed.setText, ed.textChanged)
        return ed

    def _clim_table_editor(self, p: Param, value: Any) -> QWidget:
        from .clim_table import ClimTableEditor

        ed = ClimTableEditor("" if value is None else str(value), p.label)
        self._register(p.name, ed.text, ed.setText, ed.textChanged)
        return ed

    def _path_editor(self, p: Param, value: Any) -> QWidget:
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        le = QLineEdit()
        if value is not None:
            le.setText(str(value))
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
