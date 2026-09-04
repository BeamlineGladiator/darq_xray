"""Review table for data-detected experiment values.

One row per :class:`~darq_xray.config.detect.Detection` — current vs detected,
with a per-row Apply checkbox. Pre-check rules: checked when the current
value is blank or still the schema default; unchecked (and marked "differs
from current") when applying would overwrite something the user set; when
the detected value already equals the current value the row renders as a
greyed, uncheckable "✓ matches current" info row instead. Skipped and
info-only detections render as greyed, uncheckable rows, so a pre-darfix
pass already shows what a later re-run will add.

The darfix-ROI row is special: detection recovers only the crop size, so
its Detected cell is editable (``?,?,w,h``) and its checkbox stays disabled
until the text parses as a full origin+size ROI — then it auto-checks.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from darq_xray.config.detect import Detection
from darq_xray.config.models import EXPERIMENT_SCHEMA

_LABELS = {p.name: p.label for p in EXPERIMENT_SCHEMA}
_COLS = ("Field", "Current", "Detected", "Note", "Apply")
_CHECKABLE = Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled


def _fmt(value: Any) -> str:
    if value in ("", None):
        return "—"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _is_unset(field: str, current: Any, defaults: dict[str, Any]) -> bool:
    """True when *current* is blank or still the schema default (safe to fill)."""
    return current in ("", None) or current == defaults.get(field)


class DetectReviewDialog(QDialog):
    """Apply-what-you-check review of :func:`~darq_xray.config.detect.detect_experiment`."""

    def __init__(
        self,
        detections: list[Detection],
        current: dict[str, Any],
        defaults: dict[str, Any],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Initialize from data — review")
        self.resize(780, 420)
        self._detections = list(detections)

        self._table = QTableWidget(len(self._detections), len(_COLS))
        self._table.setHorizontalHeaderLabels(_COLS)
        self._table.verticalHeader().setVisible(False)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)

        for row, d in enumerate(self._detections):
            cur = current.get(d.field, "")
            self._set_text(row, 0, _LABELS.get(d.field, d.field))
            self._set_text(row, 1, _fmt(cur))
            if d.error is not None or d.value is None:
                # skip-with-reason or info-only: greyed, nothing to apply
                self._set_text(row, 2, "—")
                self._set_text(row, 3, d.error if d.error is not None else d.note)
                for col in range(len(_COLS) - 1):
                    self._table.item(row, col).setFlags(Qt.ItemFlag.ItemIsSelectable)
                check = QTableWidgetItem()
                check.setFlags(Qt.ItemFlag.ItemIsSelectable)
                self._table.setItem(row, 4, check)
                continue
            partial = isinstance(d.value, str) and d.value.startswith("?,?")
            equal_to_current = not partial and _fmt(cur) == _fmt(d.value)
            cell = QTableWidgetItem(_fmt(d.value))
            if not partial:
                cell.setFlags(cell.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row, 2, cell)
            check = QTableWidgetItem()
            note = d.note
            if partial:
                check.setFlags(Qt.ItemFlag.ItemIsUserCheckable)  # gated: greyed until valid
                check.setCheckState(Qt.CheckState.Unchecked)
            elif equal_to_current:
                # detected matches current already: info row, nothing to apply
                check.setFlags(Qt.ItemFlag.ItemIsSelectable)
                check.setCheckState(Qt.CheckState.Unchecked)
                note = f"✓ matches current — {note}" if note else "✓ matches current"
            elif _is_unset(d.field, cur, defaults):
                check.setFlags(_CHECKABLE)
                check.setCheckState(Qt.CheckState.Checked)
            else:
                check.setFlags(_CHECKABLE)
                check.setCheckState(Qt.CheckState.Unchecked)
                note = f"{note} · differs from current" if note else "differs from current"
            self._set_text(row, 3, note)
            self._table.setItem(row, 4, check)
            if equal_to_current:
                for col in range(len(_COLS) - 1):
                    self._table.item(row, col).setFlags(Qt.ItemFlag.ItemIsSelectable)
        self._table.itemChanged.connect(self._on_item_changed)

        hint = QLabel(
            "Checked rows are written into the experiment form. Greyed rows show "
            "what a re-run will add once the data exists."
        )
        hint.setProperty("role", "muted")
        hint.setWordWrap(True)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Apply checked")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self._table, 1)
        layout.addWidget(hint)
        layout.addWidget(buttons)

    # -- helpers ----------------------------------------------------------
    def _set_text(self, row: int, col: int, text: str) -> None:
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self._table.setItem(row, col, item)

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        """Gate the partial darfix-ROI row on its edited text parsing cleanly."""
        if item.column() != 2:
            return
        row = item.row()
        d = self._detections[row]
        if not (isinstance(d.value, str) and d.value.startswith("?,?")):
            return
        from darq_xray.common.roi import parse_darfix_roi

        try:
            valid = parse_darfix_roi(item.text()) is not None
        except ValueError:
            valid = False
        check = self._table.item(row, 4)
        if check is None:
            return
        if valid:
            check.setFlags(_CHECKABLE)
            check.setCheckState(Qt.CheckState.Checked)
        else:
            check.setCheckState(Qt.CheckState.Unchecked)
            check.setFlags(Qt.ItemFlag.ItemIsUserCheckable)

    # -- result -----------------------------------------------------------
    def applied_values(self) -> dict[str, Any]:
        """Checked rows as {field: value} — edited cell text for string values."""
        out: dict[str, Any] = {}
        for row, d in enumerate(self._detections):
            check = self._table.item(row, 4)
            if check is None or check.checkState() != Qt.CheckState.Checked:
                continue
            if not (check.flags() & Qt.ItemFlag.ItemIsEnabled):
                continue
            cell = self._table.item(row, 2)
            out[d.field] = cell.text() if isinstance(d.value, str) else d.value
        return out
