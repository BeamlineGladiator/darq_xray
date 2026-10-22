"""Quitting the app exits cleanly (offscreen Qt, real `main()` in a subprocess).

PySide6 6.12.0 crashes the interpreter *after* `main()` has returned, while Qt
tears the QApplication down, when a widget is still alive whose own button
holds a ``lambda`` capturing ``self`` — the Add panels dialog's Back button is
one. 6.11.2 exits cleanly, so on older PySide6 this test cannot fail; on
6.12.0 it fails, which is why pyproject.toml excludes that release. Run it on a
later 6.12 before lifting the exclusion. It needs a subprocess because the
crash happens at interpreter exit, which no in-process assertion outlives.

Not fixed in `main()` by deleting the windows after ``app.exec()``: that cures
6.12.0 but segfaults 6.11 inside the delete, every time.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

REPO = Path(__file__).resolve().parents[1]

# Opens the figure builder and its Add panels dialog, cancels the dialog, then
# closes the main window — the session that crashed on quit under PySide6 6.12.
DRIVER = textwrap.dedent(
    """
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    app = QApplication([])
    from darq_xray.gui import app as A

    def cancel_dialog():
        # Poll, so a slow machine that has not shown the dialog yet just waits
        dlg = app.activeModalWidget()
        if dlg is None:
            QTimer.singleShot(100, cancel_dialog)
        else:
            dlg.reject()

    def drive():
        w = next(x for x in app.topLevelWidgets() if type(x).__name__ == "MainWindow")
        w._on_figure_builder()
        fb = w._figure_builder
        QTimer.singleShot(400, cancel_dialog)
        fb._on_add_panels()  # modal; cancelled by cancel_dialog
        fb.new_recipe()
        fb.hide()
        QTimer.singleShot(200, w.close)
        QTimer.singleShot(1500, app.quit)

    QTimer.singleShot(800, drive)
    print("main returned", A.main([]), flush=True)
    """
)


def test_quitting_after_the_add_panels_dialog_exits_cleanly(tmp_path):
    driver = tmp_path / "drive_app.py"
    driver.write_text(DRIVER)
    env = {
        **os.environ,
        "QT_QPA_PLATFORM": "offscreen",
        # QSettings on Linux follows XDG_CONFIG_HOME: never the user's own state
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "PYTHONPATH": os.pathsep.join(filter(None, [str(REPO), os.environ.get("PYTHONPATH")])),
    }
    # The crash is not certain on any one run (3 of 5 to 10 of 10 measured), so
    # one clean exit proves little; three in a row is the bar.
    for attempt in range(3):
        proc = subprocess.run(
            [sys.executable, str(driver)],
            cwd=str(tmp_path),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert "main returned 0" in proc.stdout, (  # precondition: the session ran
            f"attempt {attempt}: the driver never got through main()\n{proc.stderr[-2000:]}"
        )
        assert proc.returncode == 0, (
            f"attempt {attempt}: exited {proc.returncode} after main() returned 0 — "
            f"a crash in Qt's teardown\n{proc.stderr[-2000:]}"
        )
