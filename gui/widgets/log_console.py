"""A progress bar over a scrolling, read-only log view.

Fed by :class:`dfxm.runner` messages: :class:`~dfxm.runner.Progress` updates
the bar + status line, :class:`~dfxm.runner.Log` appends a line.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)


class LogConsole(QWidget):
    """Progress bar + status label + append-only log text area."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)

        self._status = QLabel("")
        self._status.setWordWrap(True)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(5000)  # cap memory on chatty stages

        layout.addWidget(self._bar)
        layout.addWidget(self._status)
        layout.addWidget(self._log, 1)

    # -- updates ----------------------------------------------------------
    def set_progress(self, frac: float, text: str = "") -> None:
        self._bar.setValue(max(0, min(100, int(round(frac * 100)))))
        if text:
            self._status.setText(text)

    def append(self, line: str) -> None:
        self._log.appendPlainText(line)

    def clear(self) -> None:
        self._bar.setValue(0)
        self._status.setText("")
        self._log.clear()

    # -- styling helpers --------------------------------------------------
    def set_status(self, text: str, *, error: bool = False) -> None:
        self._status.setText(text)
        self._status.setStyleSheet("color: #b00020;" if error else "")
