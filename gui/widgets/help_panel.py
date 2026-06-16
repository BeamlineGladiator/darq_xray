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

from ..theme import ThemeController


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
        self._error_color = ThemeController.instance().palette.error
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.addWidget(self._label)
        ThemeController.instance().themeChanged.connect(self._on_theme_changed)

    def _on_theme_changed(self, palette) -> None:
        self._error_color = palette.error
        self._render()

    def _cal_warning(self) -> str:
        return (
            f'<span style="color:{self._error_color};">⚠ calibration — physically '
            "meaningful; confirm against the beamline calibration for your "
            "experiment.</span>"
        )

    def set_idle(self, title: str, description: str) -> None:
        """Set (and show) the text used when no field is focused."""
        self._idle_html = f"<b>{html.escape(title)}</b> — {html.escape(description)}"
        self._current = None
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
        p = self._current
        head = f"<b>{html.escape(p.label)}</b>"
        if p.unit:
            head += f" ({html.escape(p.unit)})"
        parts = [head]
        if p.calibration:
            parts.append(self._cal_warning())
        if p.help:
            parts.append(html.escape(p.help))
        self._label.setText("<br>".join(parts))
