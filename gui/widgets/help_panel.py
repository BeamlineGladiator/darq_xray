"""Focus-following help panel for parameter forms.

Shows the focused :class:`~dfxm.config.models.Param`'s label, unit,
calibration warning and full help text; idles on the stage description when
nothing is focused. Connect :attr:`ParamForm.focusedParamChanged` to
:meth:`show_param`.
"""

from __future__ import annotations

import html

from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from dfxm.config.models import Param

_CAL_WARNING = (
    '<span style="color:#b00020;">⚠ calibration — physically meaningful; '
    "confirm against the beamline calibration for your experiment.</span>"
)


class HelpPanel(QFrame):
    """Styled read-only box explaining the focused parameter."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("HelpPanel { background: #eef2fb; border-left: 3px solid #4a6fd0; }")
        self._label = QLabel("")
        self._label.setWordWrap(True)
        self._idle_html = ""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.addWidget(self._label)

    def set_idle(self, title: str, description: str) -> None:
        """Set (and show) the text used when no field is focused."""
        self._idle_html = f"<b>{html.escape(title)}</b> — {html.escape(description)}"
        self._label.setText(self._idle_html)

    def show_idle(self) -> None:
        self._label.setText(self._idle_html)

    def show_param(self, p: Param) -> None:
        head = f"<b>{html.escape(p.label)}</b>"
        if p.unit:
            head += f" ({html.escape(p.unit)})"
        parts = [head]
        if p.calibration:
            parts.append(_CAL_WARNING)
        if p.help:
            parts.append(html.escape(p.help))
        self._label.setText("<br>".join(parts))
