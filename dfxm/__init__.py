"""DFXM pipeline core library (Qt-free).

Pure-Python implementations of the DFXM analysis stages plus their shared
helpers, usable from the GUI, the command line, and tests alike.

Invariant: **nothing in this package may import Qt.** Keeping the core
Qt-free is what lets every stage run headless and be unit-tested.
"""

__version__ = "0.1.0"
