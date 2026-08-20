"""Out-of-process OpenGL capability probe. Run as ``python -m dfxm.common._glprobe``.

Creating a GL context is the one operation in this codebase that can take the
whole process down: a broken or missing driver does not raise, it segfaults.
So the probe lives here, in a child process, and the parent
(:mod:`dfxm.common.machine`) treats "child died" as a normal answer meaning
"3-D unusable".

This module MUST stay a leaf: under the ``spawn`` start method a child
re-imports its module, and importing anything that reaches the GUI would spawn
windows recursively. It imports ``pyvista`` inside :func:`main` only.

Contract: print exactly one JSON object on stdout, exit 0, always.
"""

from __future__ import annotations

import json
import sys

_SOFTWARE_MARKERS = (
    "llvmpipe",
    "swrast",
    "softpipe",
    "software rasterizer",
    "microsoft basic render",
    "gdi generic",
)


def _parse_capabilities(caps: str) -> dict:
    """Pull vendor/renderer/version out of vtkRenderWindow.ReportCapabilities().

    The report is many lines of ``OpenGL <field> string:  <value>`` (note the
    doubled space); anything we cannot find comes back as "".
    """
    out = {"vendor": "", "renderer": "", "version": ""}
    for line in caps.splitlines():
        for field in out:
            marker = f"OpenGL {field} string:"
            if marker in line:
                out[field] = line.split(":", 1)[1].strip()
    return out


def probe() -> dict:
    """Build the result dict. Never raises."""
    result = {
        "status": "no-gl",
        "renderer": "",
        "vendor": "",
        "version": "",
        "max_3d_texture": None,
        "error": None,
    }
    try:
        import pyvista as pv
    except Exception as exc:  # noqa: BLE001 - no pyvista/vtk installed
        result["status"] = "no-vtk"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    plotter = None
    prev_off_screen = pv.OFF_SCREEN
    try:
        pv.OFF_SCREEN = True
        plotter = pv.Plotter(off_screen=True, window_size=[16, 16])
        plotter.show(auto_close=False)  # the context must exist before querying
        window = plotter.render_window
        result.update(_parse_capabilities(window.ReportCapabilities()))
        try:
            from vtkmodules.vtkRenderingOpenGL2 import vtkTextureObject

            limit = int(vtkTextureObject.GetMaximumTextureSize3D(window))
            result["max_3d_texture"] = limit if limit > 0 else None
        except Exception:  # noqa: BLE001 - old vtk: unknown limit, not a failure
            result["max_3d_texture"] = None
        result["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - any GL/driver failure
        result["status"] = "no-gl"
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        pv.OFF_SCREEN = prev_off_screen
        if plotter is not None:
            try:
                plotter.close()
            except Exception:  # noqa: BLE001 - closing a broken plotter
                pass
    return result


def main() -> int:
    sys.stdout.write(json.dumps(probe()) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
