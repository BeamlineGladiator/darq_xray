"""PostToolUse(Write|Edit) hook: ruff-format the file, and tell the model when
the bytes changed so its cached Read of the file is known-stale."""

import json
import pathlib
import shutil
import subprocess
import sys

try:
    d = json.load(sys.stdin)
    f = d.get("tool_input", {}).get("file_path", "") or d.get("tool_response", {}).get(
        "filePath", ""
    )
    ruff = shutil.which("ruff")
    if ruff and f.endswith(".py") and pathlib.Path(f).exists():
        before = pathlib.Path(f).read_bytes()
        subprocess.run(
            [ruff, "format", f],
            capture_output=True,
            timeout=25,
        )
        if pathlib.Path(f).read_bytes() != before:
            print(
                json.dumps(
                    {
                        "hookSpecificOutput": {
                            "hookEventName": "PostToolUse",
                            "additionalContext": (
                                f"ruff reformatted {f} — disk content no longer matches "
                                "your last Read. Read the region before any further Edit "
                                "to this file."
                            ),
                        }
                    }
                )
            )
except Exception:
    pass
