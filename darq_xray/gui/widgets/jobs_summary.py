"""Summary editor for a TEXT param whose value is a JSON list.

The raw JSON stays the value — `text()` returns the exact string the form
stores — but the form shows a one-line summary instead of a wall of JSON, with
the raw text one click away. Nothing here knows about profile jobs
specifically: it reads `name`/`offset_um` when a list of objects happens to
carry them and falls back to a plain count otherwise.

`summarize_jobs` never raises **for a `str`** — the narrow claim is deliberate.
Its whole job is to describe text a user typed, so every way that text can be
wrong (malformed JSON, a top level that is not a list, objects missing the key
it reads) is a return value rather than an exception. A non-`str` argument is
not defended against: no call site can produce one (`JobsSummaryEditor` stores
`str(value)`), and an untested `except` would be worse than the honest bound.
"""

from __future__ import annotations

import json

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


def summarize_jobs(raw: str) -> str:
    """A one-line description of the JSON list in *raw*. Never raises for a `str`."""
    try:
        items = json.loads(raw) if raw.strip() else []
    except (ValueError, TypeError):
        return "unreadable JSON — open the editor to fix it"
    if not isinstance(items, list):
        return "unreadable JSON — expected a list"
    if not items:
        return "no jobs"
    n = len(items)
    described = []
    for item in items:
        # The `"name" not in item` half is what keeps the promise above: without
        # it, `[{"offset_um": 1}]` — a plausible thing to type into the raw
        # dialog — would KeyError two lines down instead of being described.
        if not isinstance(item, dict) or "name" not in item:
            return f"{n} {'entry' if n == 1 else 'entries'}"
        offset = item.get("offset_um")
        described.append(
            f"{item['name']} @ {offset:+g} µm"
            if isinstance(offset, (int, float))
            else str(item["name"])
        )
    return f"{n} {'job' if n == 1 else 'jobs'}: " + ", ".join(described)


class JobsSummaryEditor(QWidget):
    """A read-only summary plus an "Edit raw JSON…" dialog.

    Exposes `text()` / `setText()` / `textChanged` so `ParamForm._register`
    can treat it exactly like a line edit — which is what keeps the profiles
    line-picker call sites working unchanged.
    """

    textChanged = Signal(str)  # noqa: N815 - mirrors QLineEdit's signal name

    def __init__(self, value: str, label: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._value = value or ""
        self._label = label
        self._summary = QLabel(summarize_jobs(self._value))
        self._summary.setWordWrap(True)
        # PlainText, not the QLabel default AutoText: a job name carrying angle
        # brackets is rich text as far as `Qt.mightBeRichText` is concerned, and
        # would silently lose characters on screen.
        self._summary.setTextFormat(Qt.TextFormat.PlainText)
        self._summary.setProperty("role", "muted")
        self._edit_btn = QPushButton("Edit raw JSON…")
        self._edit_btn.clicked.connect(self._on_edit)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self._summary, 1)
        row.addWidget(self._edit_btn)

    def text(self) -> str:
        return self._value

    def setText(self, value) -> None:  # noqa: N802 - mirrors QLineEdit's API
        self._value = str(value)
        self._summary.setText(summarize_jobs(self._value))
        self.textChanged.emit(self._value)

    def _on_edit(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle(self._label)
        dlg.resize(600, 460)
        text = QPlainTextEdit()
        text.setPlainText(self._value)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout = QVBoxLayout(dlg)
        layout.addWidget(text, 1)
        layout.addWidget(buttons)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.setText(text.toPlainText())
