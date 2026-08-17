"""Drag-grid layout arranger for the figure builder.

A horizontal strip of columns over a GridModel (``dfxm.compose.gridmap``):
each column is a drag-enabled tile list (internal move + drag between
columns), with ◀/▶ reorder, ✕ merge-remove, and an "+ Add column" button.
Pure view over the grid — no recipe knowledge lives here. The bar schematic
(quantity chips, united/group colorbar strips, scale-bar corner dot) is
schematic only; the real preview stays the source of truth."""

from __future__ import annotations

from PySide6.QtCore import QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QStyle,
    QStyledItemDelegate,
    QVBoxLayout,
    QWidget,
)

GROUP_COLORS: dict[str | None, str] = {
    "mosa_com": "#7b6ff0",
    "mosa_fwhm": "#2fa4a9",
    "strain": "#d1495b",
    "raw": "#8d99ae",
    "trace": "#e9a03b",
    None: "#c7c7c7",
}
_CORNER_PX = 10  # corner hot-zone size for scale-bar picking


def _corner_at(pos, rect) -> str | None:
    """Map *pos* inside *rect* to a scale-bar corner name, or None off-corner."""
    near_l = pos.x() - rect.left() <= _CORNER_PX
    near_r = rect.right() - pos.x() <= _CORNER_PX
    near_t = pos.y() - rect.top() <= _CORNER_PX
    near_b = rect.bottom() - pos.y() <= _CORNER_PX
    if near_t and near_l:
        return "upper left"
    if near_t and near_r:
        return "upper right"
    if near_b and near_l:
        return "lower left"
    if near_b and near_r:
        return "lower right"
    return None


class _TileDelegate(QStyledItemDelegate):
    """Paints a tile: group chip + title + optional scale-bar corner dot."""

    def __init__(self, arranger):
        super().__init__(arranger)
        self._arranger = arranger

    def sizeHint(self, option, index):  # noqa: N802 — Qt override
        return QSize(150, 34)

    def paint(self, painter: QPainter, option, index):  # noqa: N802 — Qt override
        pid = index.data(Qt.ItemDataRole.UserRole)
        info = self._arranger._tile_info.get(pid, {})
        r = option.rect.adjusted(2, 2, -2, -2)
        painter.save()
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        painter.setPen(QColor("#666666"))
        painter.setBrush(QColor("#dce8e4") if selected else QColor("#f4f4f4"))
        painter.drawRoundedRect(r, 4, 4)
        chip = QRect(r.left() + 4, r.top() + 4, 10, r.height() - 8)
        painter.fillRect(chip, QColor(GROUP_COLORS.get(info.get("group"), GROUP_COLORS[None])))
        painter.setPen(QColor("#222222"))
        text = info.get("title") or pid or ""
        painter.drawText(
            r.adjusted(20, 0, -4, 0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            painter.fontMetrics().elidedText(text, Qt.TextElideMode.ElideRight, r.width() - 26),
        )
        if pid and pid == self._arranger._scale_bar_panel:
            loc = self._arranger._scale_bar_loc
            cx = r.right() - 7 if "right" in loc else r.left() + 20
            cy = r.top() + 7 if "upper" in loc else r.bottom() - 7
            painter.setBrush(QColor("#009682"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QRect(cx - 3, cy - 3, 6, 6))
        painter.restore()


class _TileList(QListWidget):
    """One column's tile list; drags move tiles within and between columns."""

    def __init__(self, arranger):
        super().__init__()
        self._arranger = arranger
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragDrop)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setItemDelegate(_TileDelegate(arranger))
        self.model().rowsInserted.connect(arranger._emit_grid_changed)
        self.model().rowsRemoved.connect(arranger._emit_grid_changed)

    def mousePressEvent(self, event):  # noqa: N802 — Qt override
        item = self.itemAt(event.position().toPoint())
        if item is not None:
            loc = _corner_at(event.position().toPoint(), self.visualItemRect(item))
            if loc is not None:
                self._arranger._on_corner_clicked(item.data(Qt.ItemDataRole.UserRole), loc)
                return
        super().mousePressEvent(event)


class _ArrangerColumn(QWidget):
    """Header (◀ ▶ … ✕) + tile list + group-mode schematic strip."""

    def __init__(self, arranger):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        head = QHBoxLayout()
        self.left_btn = QPushButton("◀")
        self.right_btn = QPushButton("▶")
        self.close_btn = QPushButton("✕")
        for b in (self.left_btn, self.right_btn, self.close_btn):
            b.setFixedWidth(28)
        head.addWidget(self.left_btn)
        head.addWidget(self.right_btn)
        head.addStretch(1)
        head.addWidget(self.close_btn)
        lay.addLayout(head)
        self.list = _TileList(arranger)
        lay.addWidget(self.list, 1)
        self.flag_strip = QLabel("")
        self.flag_strip.setFixedHeight(6)
        self.flag_strip.setVisible(False)
        lay.addWidget(self.flag_strip)
        self.left_btn.clicked.connect(lambda: arranger._move_column(self, -1))
        self.right_btn.clicked.connect(lambda: arranger._move_column(self, +1))
        self.close_btn.clicked.connect(lambda: arranger._remove_column(self))


class LayoutArranger(QWidget):
    gridChanged = Signal()
    scaleBarPicked = Signal(str, str)  # (panel_id, corner loc)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tile_info: dict[str, dict] = {}
        self._columns: list[_ArrangerColumn] = []
        self._loading = False
        self._cbar_mode = "per-panel"
        self._cbar_pos = "right"
        self._flagged_sets: set[frozenset] = set()
        self._scale_bar_panel: str | None = None
        self._scale_bar_loc = "lower right"

        outer = QVBoxLayout(self)
        strip_row = QHBoxLayout()
        self._strip_host = QHBoxLayout()
        strip_row.addLayout(self._strip_host, 1)
        self._add_btn = QPushButton("+ Add column")
        self._add_btn.clicked.connect(self._on_add_column)
        strip_row.addWidget(self._add_btn)
        self._right_strip = QLabel("")
        self._right_strip.setFixedWidth(8)
        self._right_strip.setVisible(False)
        strip_row.addWidget(self._right_strip)
        outer.addLayout(strip_row, 1)
        self._bottom_strip = QLabel("")
        self._bottom_strip.setFixedHeight(8)
        self._bottom_strip.setVisible(False)
        outer.addWidget(self._bottom_strip)

    # -- grid API --------------------------------------------------------------
    def set_grid(self, grid, tile_info_by_id) -> None:
        self._loading = True
        try:
            for col in self._columns:
                col.setParent(None)
                col.deleteLater()
            self._columns = []
            self._tile_info = dict(tile_info_by_id)
            for column in grid:
                col = self._new_column()
                for pid in column:
                    col.list.addItem(self._make_item(pid))
        finally:
            self._loading = False
        self._refresh_schematic()

    def grid(self) -> list[list[str]]:
        return [ids for col in self._columns if (ids := self._column_ids(col))]

    def _make_item(self, pid: str) -> QListWidgetItem:
        item = QListWidgetItem(self._tile_info.get(pid, {}).get("title") or pid)
        item.setData(Qt.ItemDataRole.UserRole, pid)
        item.setFlags(
            Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsDragEnabled
        )
        return item

    def _new_column(self) -> _ArrangerColumn:
        col = _ArrangerColumn(self)
        self._columns.append(col)
        self._strip_host.addWidget(col)
        return col

    def _column_ids(self, col) -> list[str]:
        return [col.list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(col.list.count())]

    # -- column ops ------------------------------------------------------------
    def _on_add_column(self) -> None:
        self._new_column()
        self._emit_grid_changed()

    def _move_column(self, col, delta: int) -> None:
        i = self._columns.index(col)
        j = i + delta
        if j < 0 or j >= len(self._columns):
            return
        self._columns[i], self._columns[j] = self._columns[j], self._columns[i]
        for c in self._columns:
            self._strip_host.removeWidget(c)
        for c in self._columns:
            self._strip_host.addWidget(c)
        self._emit_grid_changed()

    def _remove_column(self, col) -> None:
        if len(self._columns) <= 1:
            return  # never remove the last column
        i = self._columns.index(col)
        target = self._columns[i - 1] if i > 0 else self._columns[1]
        self._loading = True
        try:
            while col.list.count():
                item = col.list.takeItem(0)
                target.list.addItem(self._make_item(item.data(Qt.ItemDataRole.UserRole)))
            self._columns.remove(col)
            col.setParent(None)
            col.deleteLater()
        finally:
            self._loading = False
        self._emit_grid_changed()

    def _emit_grid_changed(self, *_args) -> None:
        if self._loading:
            return
        self._refresh_schematic()
        self.gridChanged.emit()

    # -- bar schematic ---------------------------------------------------------
    def set_bar_schematic(
        self, colorbar_mode: str, colorbar_pos: str, flagged_member_sets=None
    ) -> None:
        self._cbar_mode = colorbar_mode
        self._cbar_pos = colorbar_pos
        self._flagged_sets = set(flagged_member_sets or ())
        self._refresh_schematic()

    def set_scale_bar(self, panel_id: str | None, loc: str) -> None:
        self._scale_bar_panel = panel_id
        self._scale_bar_loc = loc or "lower right"
        for col in self._columns:
            col.list.viewport().update()

    def _on_corner_clicked(self, pid: str, loc: str) -> None:
        self.set_scale_bar(pid, loc)
        self.scaleBarPicked.emit(pid, loc)

    def _strip_css(self, horizontal: bool) -> str:
        groups: list[str] = []
        for col in self._columns:
            for pid in self._column_ids(col):
                g = self._tile_info.get(pid, {}).get("group")
                if g is not None and g != "trace" and g not in groups:
                    groups.append(g)
        if not groups:
            return "background: #c7c7c7;"
        n = len(groups)
        stops = []
        for i, g in enumerate(groups):
            c = GROUP_COLORS[g]
            stops.append(f"stop:{i / n:.3f} {c}, stop:{(i + 1) / n - 0.001:.3f} {c}")
        coords = "x1:0, y1:0, x2:1, y2:0" if horizontal else "x1:0, y1:0, x2:0, y2:1"
        return f"background: qlineargradient({coords}, {', '.join(stops)});"

    def _refresh_schematic(self) -> None:
        united = self._cbar_mode == "united"
        self._right_strip.setVisible(united and self._cbar_pos == "right")
        self._bottom_strip.setVisible(united and self._cbar_pos == "bottom")
        if united:
            css = self._strip_css(horizontal=self._cbar_pos == "bottom")
            self._right_strip.setStyleSheet(css)
            self._bottom_strip.setStyleSheet(css)
        for col in self._columns:
            members = frozenset(self._column_ids(col))
            flagged = (not united) and bool(members) and members in self._flagged_sets
            col.flag_strip.setVisible(bool(flagged))
            if flagged:
                col.flag_strip.setStyleSheet("background: #009682;")
