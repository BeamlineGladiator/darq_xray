"""Application entry point: ``python3 -m gui.app``.

Heavy imports (Qt, the main window) are deferred into :func:`main` on purpose:
the stage worker runs under the ``spawn`` start method, which re-imports this
module in the child. Keeping the module top-level light means the worker child
does not drag in Qt just to run a headless stage.
"""

from __future__ import annotations

import os
import sys


def main(argv: list[str] | None = None) -> int:
    # Bind matplotlib's Qt backend to PySide6 (must precede backend import).
    os.environ.setdefault("QT_API", "pyside6")

    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from .main_window import MainWindow
    from .theme import ThemeController

    QApplication.setOrganizationName("dfxm")
    QApplication.setApplicationName("pipeline")

    app = QApplication.instance() or QApplication(argv if argv is not None else sys.argv)

    mode = QSettings().value("theme", "light")
    if mode not in ("light", "dark"):
        mode = "light"
    ThemeController.instance().set_mode(mode)  # applies Fusion + palette + QSS

    window = MainWindow()
    window.show()
    ret = app.exec()
    # Defensive: MainWindow.closeEvent already joins every pinned worker
    # QThread, but app.exec() can also return via a path that skips
    # closeEvent (e.g. the last top-level window other than MainWindow
    # closing, or a quit triggered some other way) — a still-running QThread
    # left pinned in gui.widgets.busy._LIVE_WORKERS at interpreter teardown
    # aborts the process, so join here too; a no-op when closeEvent already
    # emptied the registry. Bounded (unlike closeEvent's own call): this is
    # the last-resort backstop right before the process exits, not a
    # user-facing wait, so it must not hang the interpreter forever on a
    # worker that never finishes — 60 s is generous for any batch/render/
    # export this app runs, then main() returns regardless.
    from .widgets.busy import wait_for_workers

    wait_for_workers(timeout_ms=60_000)
    return ret


if __name__ == "__main__":
    raise SystemExit(main())
