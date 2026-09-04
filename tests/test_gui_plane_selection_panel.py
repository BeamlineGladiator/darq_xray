"""Offscreen test: PlaneSelectionPanel's ★-only filter survives a set_rows reload."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from darq_xray.gui.widgets.plane_selection import PlaneSelectionPanel  # noqa: E402
from darq_xray.gui.widgets.plane_selection_model import PlaneRow  # noqa: E402


def _rows():
    return [
        PlaneRow(key="a", section="s", number=0, offset=-1.0, label="p000", marked=True),
        PlaneRow(key="b", section="s", number=1, offset=0.0, label="p001", marked=False),
    ]


def test_marked_only_filter_survives_reload():
    _app = QApplication.instance() or QApplication([])
    panel = PlaneSelectionPanel(show_quantities=False)
    panel.show()

    panel.set_rows(_rows())
    panel._marked_only.setChecked(True)
    assert panel._items["b"].isHidden()
    assert not panel._items["a"].isHidden()

    # Rebuild with the same rows: the ★-only filter must re-apply automatically.
    panel.set_rows(_rows())
    assert panel._marked_only.isVisible()
    assert panel._marked_only.isChecked()
    assert panel._items["b"].isHidden()
    assert not panel._items["a"].isHidden()

    # Visibility never affects selection: both keys remain checked.
    assert sorted(panel.checked_plane_keys()) == ["a", "b"]
