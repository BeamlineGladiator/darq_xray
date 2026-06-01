"""Headless GUI smoke test (run manually, not collected by pytest).

Exercises the whole vertical slice through the real widgets:

1. build + show the main window (offscreen Qt),
2. confirm the STO2 preset auto-loaded and its calibration caveat is shown,
3. run the concat stage end-to-end through the StageView and check the result,
4. confirm cancelling a long run actually kills the worker process.

Run it with::

    python3 tests/gui_smoke.py

It is named ``gui_smoke.py`` (not ``test_*``) so pytest does not collect it —
it spawns processes and needs a Qt platform, which is awkward under pytest.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

# Make the package importable when run as a plain script, and force a
# windowless Qt platform so this works without a display.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_API", "pyside6")

import h5py  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


def _make_input() -> str:
    root = tempfile.mkdtemp()
    folder = os.path.join(root, "layer__3")
    os.makedirs(folder)
    with h5py.File(os.path.join(folder, "layer__3.h5"), "w") as f:
        for e, nfr in [("1.1", 3), ("2.1", 2), ("3.1", 4)]:
            g = f.create_group(e)
            pos = g.create_group("instrument/positioners")
            pos.create_dataset("mu", data=np.linspace(11, 11.1, nfr))
            pos.create_dataset("ccmth", data=7.1 + 0.01 * nfr)
            g.create_dataset("instrument/pco_ff/image", data=np.zeros((nfr, 2, 2), dtype="uint16"))
    return folder


def _sleeper(_params, progress=None):
    for i in range(400):
        if progress:
            progress(i / 400, f"step {i}")
        time.sleep(0.05)
    return "should-not-finish"


def main() -> int:
    from dfxm.runner import StageRunner
    from gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    win.show()
    app.processEvents()
    print("[1] main window built + shown")

    exp = win._experiment_panel.current_experiment()
    assert exp.name == "STO2_overnight", exp.name
    assert exp.mu_ref_deg == 11.5015 and exp.ccmth_ref_deg == 7.144
    assert win._experiment_panel._notes.isVisible()
    assert "11.2491" in win._experiment_panel._notes.text()
    print("[2] STO2 preset loaded; calibration caveat surfaced (mu_ref 11.5015 vs 11.2491)")

    view = win._views["concat"]
    folder = _make_input()
    view._form.set_values({"mode": "single", "input_folder": folder})
    done: list[tuple[str, bool]] = []
    view.runFinished.connect(lambda name, ok: done.append((name, ok)))
    view._on_run()
    t0 = time.time()
    while not done and time.time() - t0 < 60:
        app.processEvents()
        time.sleep(0.02)
    assert done == [("concat", True)], done
    app.processEvents()
    results = view._results.toPlainText()
    assert "1 ok" in results and "9 frames" in results, results
    assert win._status_items["concat"].text().startswith("✓")
    assert os.path.exists(os.path.join(folder, "layer__3_concat.h5"))
    print(f"[3] concat ran through the UI: {results.splitlines()[0]}; status ✓; output written")

    # Cancel kills the worker.
    runner = StageRunner(_sleeper, {}, start_method="fork")
    runner.start()
    time.sleep(0.4)
    assert runner.is_alive()
    runner.cancel(timeout=2.0)
    assert not runner.is_alive()
    print("[4] cancel terminated a long-running worker")

    print("\nGUI SMOKE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
