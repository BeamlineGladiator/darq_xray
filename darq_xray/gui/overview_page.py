"""Landing page: the pipeline drawn as clickable chips + stage descriptions.

Default screen on launch. Shows the stage order left-to-right (darfix as a
dashed external step, concat marked optional), one newcomer-friendly
sentence per stage from each :class:`StageSpec.description`, and a
per-session status recap. Clicking a chip (or a row) jumps to that stage.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from darq_xray.config.models import StageSpec


class OverviewPage(QWidget):
    """Pipeline chips + per-stage description list + status recap."""

    stageSelected = Signal(str)

    def __init__(
        self,
        order: tuple[str, ...],
        specs: dict[str, StageSpec],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._status_labels: dict[str, QLabel] = {}

        title = QLabel("<h2>DFXM pipeline — overview</h2>")
        intro = QLabel(
            "Run the stages top to bottom. <b>Concat is optional</b> — skip it if "
            "your scans are already concatenated. <b>darfix runs outside this "
            "app</b>, between concat and the map stages: it fits the rocking "
            "curves into the maps.h5 files that strain and mosaicity consume."
        )
        intro.setWordWrap(True)

        chips = QHBoxLayout()
        for i, name in enumerate(order):
            if i:
                chips.addWidget(QLabel("→"))
            label = specs[name].label + (" (optional)" if name == "concat" else "")
            btn = QPushButton(label)
            btn.setProperty("role", "chip")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _=False, n=name: self.stageSelected.emit(n))
            chips.addWidget(btn)
            if name == "concat":
                chips.addWidget(QLabel("→"))
                ext = QLabel("darfix (external)")
                ext.setProperty("role", "external")
                ext.setToolTip(
                    "Run darfix outside this app: it turns the concatenated .h5 "
                    "into the maps.h5 files used by strain and mosaicity."
                )
                chips.addWidget(ext)
        chips.addStretch(1)
        chips_host = QWidget()
        chips_host.setLayout(chips)
        chips_scroll = QScrollArea()
        chips_scroll.setWidgetResizable(True)
        chips_scroll.setWidget(chips_host)
        chips_scroll.setFixedHeight(64)

        rows = QVBoxLayout()
        for name in order:
            status = QLabel("—")
            status.setFixedWidth(18)
            self._status_labels[name] = status
            text = QLabel(f"<b>{specs[name].label}</b> — {specs[name].description}")
            text.setWordWrap(True)
            row = QHBoxLayout()
            row.addWidget(status)
            row.addWidget(text, 1)
            rows.addLayout(row)
        rows.addStretch(1)
        rows_host = QWidget()
        rows_host.setLayout(rows)
        rows_scroll = QScrollArea()
        rows_scroll.setWidgetResizable(True)
        rows_scroll.setWidget(rows_host)

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(intro)
        layout.addWidget(chips_scroll)
        layout.addWidget(rows_scroll, 1)

    def set_status(self, name: str, glyph: str) -> None:
        label = self._status_labels.get(name)
        if label is not None:
            label.setText(glyph)
