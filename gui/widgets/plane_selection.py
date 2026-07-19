"""Planes-first selection panel: planes listed once, filter narrows, quantities right.

Shared by the slices + generic replot dialogs and the Pin planes… dialog. All
selection logic lives in the Qt-free ``plane_selection_model``; this widget is
the checkbox/visibility shell around it.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .plane_selection_model import PlaneRow, filter_rows


class PlaneSelectionPanel(QWidget):
    """Left: plane/layer rows (listed once). Right: quantity checkboxes."""

    selectionChanged = Signal()

    def __init__(self, show_quantities: bool = True, parent=None) -> None:
        super().__init__(parent)
        self._rows: list[PlaneRow] = []
        self._items: dict[object, QTreeWidgetItem] = {}
        self._show_quantities = show_quantities

        self._filter = QLineEdit()
        self._filter.setPlaceholderText("filter: plane numbers (118) or offsets (-3.7), commas")
        self._filter.textChanged.connect(self._apply_filter)
        self._no_match = QLabel("no match")
        self._no_match.setVisible(False)

        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Plane"])
        self._tree.itemChanged.connect(lambda *_: self.selectionChanged.emit())

        check_all = QPushButton("Check all")
        check_all.clicked.connect(lambda: self.set_all_checked(True))
        uncheck_all = QPushButton("Uncheck all")
        uncheck_all.clicked.connect(lambda: self.set_all_checked(False))
        check_visible = QPushButton("Check all visible")
        check_visible.clicked.connect(self.check_all_visible)
        btns = QHBoxLayout()
        for b in (check_all, uncheck_all, check_visible):
            btns.addWidget(b)
        btns.addStretch(1)

        left = QVBoxLayout()
        frow = QHBoxLayout()
        frow.addWidget(QLabel("Filter:"))
        frow.addWidget(self._filter, 1)
        frow.addWidget(self._no_match)
        left.addLayout(frow)
        left.addWidget(self._tree, 1)
        left.addLayout(btns)

        root = QHBoxLayout(self)
        lw = QWidget()
        lw.setLayout(left)
        root.addWidget(lw, 2)

        self._qty = QListWidget()
        self._qty.itemChanged.connect(lambda *_: self.selectionChanged.emit())
        if show_quantities:
            right = QVBoxLayout()
            right.addWidget(QLabel("Quantities:"))
            right.addWidget(self._qty, 1)
            rw = QWidget()
            rw.setLayout(right)
            root.addWidget(rw, 1)

    # -- population -------------------------------------------------------
    def set_rows(self, rows: list[PlaneRow]) -> None:
        """Rebuild the plane list; everything checked (a plain Render remakes all)."""
        self._rows = list(rows)
        self._tree.blockSignals(True)
        self._tree.clear()
        self._items.clear()
        sections: dict[str, QTreeWidgetItem] = {}
        for r in self._rows:
            if r.section:
                parent = sections.get(r.section)
                if parent is None:
                    parent = QTreeWidgetItem(self._tree, [r.section])
                    parent.setFlags(parent.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
                    sections[r.section] = parent
                item = QTreeWidgetItem(parent, [r.label])
            else:
                item = QTreeWidgetItem(self._tree, [r.label])
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(0, Qt.CheckState.Checked)
            item.setData(0, Qt.ItemDataRole.UserRole, r.key)
            self._items[r.key] = item
        self._tree.expandAll()
        self._tree.blockSignals(False)
        self._apply_filter(self._filter.text())
        self.selectionChanged.emit()

    def set_quantities(self, quantities: list[tuple[object, str]]) -> None:
        self._qty.blockSignals(True)
        self._qty.clear()
        for key, label in quantities:
            it = QListWidgetItem(label)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked)
            it.setData(Qt.ItemDataRole.UserRole, key)
            self._qty.addItem(it)
        self._qty.blockSignals(False)
        self.selectionChanged.emit()

    # -- filtering (visibility only; never selects) -----------------------
    def _apply_filter(self, text: str) -> None:
        visible = {r.key for r in filter_rows(self._rows, text)}
        for key, item in self._items.items():
            item.setHidden(key not in visible)
        for i in range(self._tree.topLevelItemCount()):
            top = self._tree.topLevelItem(i)
            if top.childCount():  # section header: hide when all children hidden
                top.setHidden(all(top.child(j).isHidden() for j in range(top.childCount())))
        self._no_match.setVisible(bool(text.strip()) and not visible)

    # -- bulk actions ------------------------------------------------------
    def set_all_checked(self, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for item in self._items.values():
            item.setCheckState(0, state)

    def check_all_visible(self) -> None:
        for item in self._items.values():
            if not item.isHidden():
                item.setCheckState(0, Qt.CheckState.Checked)

    # -- selection ---------------------------------------------------------
    def checked_plane_keys(self) -> list:
        return [k for k, it in self._items.items() if it.checkState(0) == Qt.CheckState.Checked]

    def checked_quantity_keys(self) -> list:
        return [
            self._qty.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self._qty.count())
            if self._qty.item(i).checkState() == Qt.CheckState.Checked
        ]

    def has_selection(self) -> bool:
        if not self.checked_plane_keys():
            return False
        return (not self._show_quantities) or bool(self.checked_quantity_keys())
