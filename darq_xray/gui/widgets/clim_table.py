"""Per-volume colour-limit editor for a stage form.

``visualize`` can render nine volumes, and a single ``vmin``/``vmax`` pair — the
idiom ``strain`` and ``matched`` use — cannot serve them: strain sits near
1e-4 while a raw intensity sum sits near 1e3. Eighteen more form fields would
have doubled an Advanced section that already carries twenty-three, so the
limits live in ONE ``TEXT`` param and are edited through a dialog.

:class:`ClimTableEditor` is the param's widget: a muted one-line summary plus an
"Edit colour limits…" button. It deliberately mirrors
:class:`~darq_xray.gui.widgets.jobs_summary.JobsSummaryEditor`'s contract — ``text()`` /
``setText()`` / ``textChanged`` — so ``ParamForm._register`` treats it exactly
like a line edit and no other part of the form machinery has to know about it.

The dialog itself hosts the **existing** :class:`~darq_xray.gui.widgets.clim_section.ClimGroupSection`,
the same nine labelled rows the replot dialogs show, so a limit is typed in the
same place and under the same name wherever you meet it.
"""

from __future__ import annotations

import json

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .clim_section import ClimGroupSection, volume_label


def _parsed(value) -> dict:
    """*value* as a ``{key: pair}`` dict, or ``{}`` for anything unusable.

    Never raises. The stage's ``_clim_overrides`` is what rejects bad input,
    with a hint and a banner; a widget that threw while painting its summary
    would hide that message behind a traceback.
    """
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def summarize_clim(value) -> str:
    """One line describing how many volumes carry an explicit limit."""
    data = _parsed(value)
    total = len(_volume_rows())
    n = sum(
        1
        for pair in data.values()
        if isinstance(pair, (list, tuple))
        and any(b is not None and str(b).strip() != "" for b in pair)
    )
    if not n:
        return "all automatic"
    return f"limits set for {n} of {total} volumes"


def _volume_rows() -> list[tuple[str, str]]:
    """``[(key, label), ...]`` for every volume the visualize stage can render.

    Read from the stage's own ``_VOLUME_KEYS`` rather than restated here, so a
    volume added or renamed there cannot leave this dialog offering a row the
    run will ignore. The import is local: ``darq_xray.gui`` may depend on ``darq_xray``, and
    keeping it out of module scope keeps this widget cheap to import.
    """
    from darq_xray.stages.visualize import _VOLUME_KEYS

    return [(key, volume_label(key)) for key, _ds, _label in _VOLUME_KEYS]


class ClimTableEditor(QWidget):
    """A read-only summary plus an "Edit colour limits…" dialog.

    Exposes ``text()`` / ``setText()`` / ``textChanged`` so ``ParamForm._register``
    can treat it exactly like a line edit.
    """

    textChanged = Signal(str)  # noqa: N815 - mirrors QLineEdit's signal name

    def __init__(self, value: str, label: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._value = value or "{}"
        self._label = label
        self._summary = QLabel(summarize_clim(self._value))
        self._summary.setTextFormat(Qt.TextFormat.PlainText)
        self._summary.setProperty("role", "muted")
        self._edit_btn = QPushButton("Edit colour limits…")
        self._edit_btn.clicked.connect(self._on_edit)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self._summary, 1)
        row.addWidget(self._edit_btn)

    @staticmethod
    def groups() -> list[tuple[str, str]]:
        """The dialog's rows — exposed so a test can pin them against the stage."""
        return _volume_rows()

    def text(self) -> str:
        return self._value

    def setText(self, value) -> None:  # noqa: N802 - mirrors QLineEdit's API
        self._value = str(value)
        self._summary.setText(summarize_clim(self._value))
        self.textChanged.emit(self._value)

    def _on_edit(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle(self._label)
        dlg.resize(520, 420)
        section = ClimGroupSection()
        section.set_groups(_volume_rows())
        section.set_clim_by_group(_parsed(self._value))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(section)
        note = QLabel(
            "Blank = automatic for that limit. A limit you set here is used exactly "
            "as given — 'Round colour limits' does not touch it."
        )
        note.setWordWrap(True)
        note.setProperty("role", "muted")
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.rejected.connect(dlg.reject)

        def _accept() -> None:
            # The section's own check, surfaced here rather than at Run: a
            # dialog that closes on an unparseable number would send the user
            # back to the form to find out.
            problem = section.validate()
            if problem:
                QMessageBox.warning(dlg, self._label, problem)
                return
            dlg.accept()

        buttons.accepted.connect(_accept)
        layout = QVBoxLayout(dlg)
        layout.addWidget(scroll, 1)
        layout.addWidget(note)
        layout.addWidget(buttons)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            mapping = section.clim_by_group() or {}
            self.setText(json.dumps({k: list(v) for k, v in mapping.items()}))
