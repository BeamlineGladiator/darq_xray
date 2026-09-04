"""DARQ — DFXM (Dark-Field X-ray Microscopy) analysis pipeline.

Pure-Python implementations of the DFXM analysis stages plus their shared
helpers, usable from the GUI, the command line, and tests alike.

Invariant: **nothing outside :mod:`darq_xray.gui` may import Qt.** Keeping the
core Qt-free is what lets every stage run headless and be unit-tested.
"""

__version__ = "0.1.0"
