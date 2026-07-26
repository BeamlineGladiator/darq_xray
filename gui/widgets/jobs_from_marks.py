"""Jobs from marks… checklist: pick which marked planes become profile jobs.

A thin selection dialog over ``dfxm.stages.slices.read_marks`` output; the
caller (profiles StageView) opens one LinePickerDialog per checked row and
appends complete jobs via ``gui.viewers.append_line_job``. Holds no h5 handle.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)


class JobsFromMarksDialog(QDialog):
    """Checklist of marked planes; OK -> ``selected = [(slice_name, offset_um), ...]``."""

    def __init__(self, marks: dict[str, list[float]], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Jobs from marks")
        self.selected: list[tuple[str, float]] = []

        self._list = QListWidget()
        for sname in sorted(marks):
            for off in marks[sname]:
                it = QListWidgetItem(f"{sname}  @ {off:+.2f} µm")
                it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                it.setCheckState(Qt.CheckState.Checked)
                it.setData(Qt.ItemDataRole.UserRole, (sname, float(off)))
                self._list.addItem(it)

        ok = QPushButton("OK")
        ok.setProperty("role", "primary")
        ok.clicked.connect(self._on_ok)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        brow = QHBoxLayout()
        brow.addStretch(1)
        brow.addWidget(ok)
        brow.addWidget(cancel)

        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("A line picker opens for each checked plane:"))
        lay.addWidget(self._list, 1)
        lay.addLayout(brow)

    def _on_ok(self) -> None:
        self.selected = [
            self._list.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self._list.count())
            if self._list.item(i).checkState() == Qt.CheckState.Checked
        ]
        self.accept()
