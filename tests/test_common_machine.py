"""Hardware profile probing (dfxm/common/machine.py, dfxm/common/_glprobe.py)."""

from __future__ import annotations

import json
import subprocess
import sys


def test_glprobe_child_prints_one_json_line_and_exits_zero():
    """The probe child must ALWAYS exit 0 with parseable JSON — never a traceback.

    It runs on machines with no GPU, no driver and no vtk; the parent
    distinguishes outcomes by the 'status' field, not by exit code.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "dfxm.common._glprobe"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    assert len(lines) == 1, f"expected exactly one line, got {lines!r}"
    data = json.loads(lines[0])
    assert data["status"] in ("ok", "no-gl", "no-vtk")
    assert set(data) == {"status", "renderer", "vendor", "version", "max_3d_texture", "error"}
    if data["status"] == "ok":
        assert isinstance(data["renderer"], str) and data["renderer"]
        assert data["max_3d_texture"] is None or data["max_3d_texture"] > 0
