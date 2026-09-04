"""Persist and share window/splitter geometry (Qt-side; core stays Qt-free).

Uses the app-wide QSettings (org ``darq_xray``, app ``pipeline``):
  - ``geometry``      : QMainWindow.saveGeometry() (size, position, maximized)
  - ``mainSplitter``  : the top-level (left rail | stack) splitter state
  - ``stageSplitter`` : the shared middle|right split applied to every stage

Every stage's inner splitter is registered here and kept in lock-step: dragging
any one broadcasts its sizes to the others and writes them back, so all stages
show the same middle|right width across a session and across restarts.
"""

from __future__ import annotations

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QMainWindow, QSplitter

_KEY_GEOMETRY = "geometry"
_KEY_MAIN_SPLIT = "mainSplitter"
_KEY_STAGE_SPLIT = "stageSplitter"

#: First-run middle|right split — favours the middle (parameter) column.
DEFAULT_STAGE_SIZES: list[int] = [560, 460]


class WindowState:
    """Save/restore window geometry and keep the stage splitters in sync."""

    def __init__(self, settings: QSettings | None = None) -> None:
        self._settings = settings or QSettings()
        self._stage_splitters: list[QSplitter] = []
        self._applying = False

    # -- shared stage splitter -------------------------------------------
    def register_stage_splitter(self, splitter: QSplitter) -> None:
        """Track *splitter*, apply the shared sizes, and mirror future drags."""
        self._stage_splitters.append(splitter)
        splitter.setSizes(self._saved_stage_sizes())
        splitter.splitterMoved.connect(lambda _pos, _i, s=splitter: self._on_stage_moved(s))

    def _on_stage_moved(self, source: QSplitter) -> None:
        if self._applying:
            return
        sizes = source.sizes()
        if not sizes or sum(sizes) == 0:
            return
        self._settings.setValue(_KEY_STAGE_SPLIT, sizes)
        self._applying = True
        try:
            for s in self._stage_splitters:
                if s is not source:
                    s.setSizes(sizes)
        finally:
            self._applying = False

    def _saved_stage_sizes(self) -> list[int]:
        raw = self._settings.value(_KEY_STAGE_SPLIT)
        if raw is None:
            return list(DEFAULT_STAGE_SIZES)
        if isinstance(raw, str):
            raw = raw.split(",")
        try:
            sizes = [int(float(x)) for x in raw]
        except (TypeError, ValueError):
            return list(DEFAULT_STAGE_SIZES)
        return sizes if len(sizes) >= 2 else list(DEFAULT_STAGE_SIZES)

    # -- geometry + main splitter ----------------------------------------
    def restore(self, window: QMainWindow, main_splitter: QSplitter) -> None:
        """Restore window geometry + the top-level splitter, if saved."""
        geo = self._settings.value(_KEY_GEOMETRY)
        if geo is not None:
            try:
                window.restoreGeometry(geo)
            except (TypeError, ValueError):  # corrupt/foreign state
                pass
        state = self._settings.value(_KEY_MAIN_SPLIT)
        if state is not None:
            try:
                main_splitter.restoreState(state)
            except (TypeError, ValueError):
                pass

    def save(self, window: QMainWindow, main_splitter: QSplitter) -> None:
        """Persist window geometry + the top-level splitter state."""
        self._settings.setValue(_KEY_GEOMETRY, window.saveGeometry())
        self._settings.setValue(_KEY_MAIN_SPLIT, main_splitter.saveState())
