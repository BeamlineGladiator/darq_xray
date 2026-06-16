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
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
