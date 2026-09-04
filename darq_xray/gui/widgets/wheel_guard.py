"""Stop unfocused arrow-fields from eating scroll-wheel events.

Spin boxes and combo boxes change value on wheel by default, so scrolling a
form inside a QScrollArea silently edits fields under the cursor — and form
persistence then saves the stray edit. The guard sets StrongFocus (wheel can
no longer *give* focus) and swallows wheel events while the widget is
unfocused; because the event is left unaccepted, Qt propagates it up to the
scroll area, which scrolls the page as expected.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt


class _WheelGuard(QObject):
    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt API
        if event.type() == QEvent.Type.Wheel and not obj.hasFocus():
            event.ignore()
            return True
        return False


_guard: _WheelGuard | None = None


def install_wheel_guard(widget) -> None:
    """Make *widget* ignore wheel events unless it has keyboard focus."""
    global _guard
    if _guard is None:
        _guard = _WheelGuard()
    widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    widget.installEventFilter(_guard)
