"""Focus-following help panel for parameter forms.

Shows the focused :class:`~darq_xray.config.models.Param`'s label, unit,
calibration warning and full help text; idles on the stage description when
nothing is focused. Connect :attr:`ParamForm.focusedParamChanged` to
:meth:`show_param`.
"""

from __future__ import annotations

import html

from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from darq_xray.config.models import Param

from ..theme import ThemeController


def param_help_html(p: Param, error_color: str | None = None, see_also: str = "") -> str:
    """Rich-text help for *p*: label (+unit), calibration note, help text.

    The calibration note is coloured with *error_color* when given (the help
    panel), otherwise rendered plain (tooltips, which do not restyle on theme
    change). *see_also* — a :class:`~darq_xray.config.models.SeeAlso` pointer's text
    for this parameter — is appended as a "See also:" line when non-empty.
    """
    head = f"<b>{html.escape(p.label)}</b>"
    if p.unit:
        head += f" ({html.escape(p.unit)})"
    parts = [head]
    if p.calibration:
        warn = (
            "⚠ calibration — physically meaningful; confirm against the beamline "
            "calibration for your experiment."
        )
        if error_color:
            parts.append(f'<span style="color:{error_color};">{warn}</span>')
        else:
            parts.append(warn)
    if p.help:
        parts.append(html.escape(p.help))
    if see_also:
        parts.append(f"<i>See also:</i> {html.escape(see_also)}")
    return "<br>".join(parts)


class HelpPanel(QFrame):
    """Styled read-only box explaining the focused parameter."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        # Background/border come from the global QSS (HelpPanel selector).
        self._label = QLabel("")
        self._label.setWordWrap(True)
        self._idle_html = ""
        self._current: Param | None = None
        self._see_also: dict[str, str] = {}
        self._error_color = ThemeController.instance().palette.error
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.addWidget(self._label)
        ThemeController.instance().themeChanged.connect(self._on_theme_changed)

    def _on_theme_changed(self, palette) -> None:
        self._error_color = palette.error
        self._render()

    def set_idle(self, title: str, description: str, see_also: str = "") -> None:
        """Set (and show) the text used when no field is focused.

        *see_also* is the stage-level pointer text, appended as its own line so
        a newcomer idling on the stage description sees where a feature they
        expected on the form actually lives.
        """
        self._idle_html = f"<b>{html.escape(title)}</b> — {html.escape(description)}"
        if see_also:
            self._idle_html += f"<br><i>See also:</i> {html.escape(see_also)}"
        self._current = None
        self._render()

    def set_see_also(self, mapping: dict[str, str]) -> None:
        """Pointer text per parameter name, appended when that param is shown."""
        self._see_also = dict(mapping)
        self._render()

    def show_idle(self) -> None:
        self._current = None
        self._render()

    def show_param(self, p: Param) -> None:
        self._current = p
        self._render()

    def _render(self) -> None:
        if self._current is None:
            self._label.setText(self._idle_html)
            return
        self._label.setText(
            param_help_html(
                self._current, self._error_color, self._see_also.get(self._current.name, "")
            )
        )
