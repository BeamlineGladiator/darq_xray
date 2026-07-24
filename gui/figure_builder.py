"""Publication figure-builder window (Phase B skeleton, Task 10).

A non-modal window with a left sources+outline pane (Add panels…, the recipe
outline tree, structural-edit buttons, and recipe file I/O), a center preview
host and a right style pane — both placeholders here, filled in by Tasks
11-12. Only the Qt-free recipe model (:mod:`dfxm.compose.recipe`) is imported
at module level; the render/compose machinery (matplotlib-heavy) is left for
the tasks that actually draw a preview.
"""

from __future__ import annotations

import os
from dataclasses import replace as dc_replace

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from dfxm.common.plotting import PlotStyle
from dfxm.compose.recipe import (
    Col,
    ComposeStyle,
    FigureRecipe,
    PanelDef,
    PanelRef,
    Row,
    Spacer,
    TextCell,
    recipe_from_json,
    recipe_to_json,
)

from .widgets.panel_picker import AddPanelDialog


class FigureBuilderWindow(QMainWindow):
    """Non-modal window for composing a multi-panel publication figure.

    ``defaults_provider`` is a zero-arg callable the main window supplies —
    it closes over the live experiment + stage forms and returns the
    ``{stage: {...}}`` dict :class:`~gui.widgets.panel_picker.AddPanelDialog`
    needs. ``style`` is the session's live :class:`~dfxm.common.plotting.PlotStyle`,
    kept for the preview/style panes built in Tasks 11-12.
    """

    def __init__(self, defaults_provider, style: PlotStyle, parent=None) -> None:
        super().__init__(parent)
        self._defaults_provider = defaults_provider
        self._style = style
        self._recipe = FigureRecipe("untitled", {}, ComposeStyle(), Row([]), [])
        self._dirty = False
        self._current_path: str | None = None

        self._build_ui()
        self._rebuild_tree()
        self._update_title()

    # -- UI construction --------------------------------------------------------
    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)

        splitter.addWidget(self._build_left_pane())
        splitter.addWidget(self._build_center_pane())
        splitter.addWidget(self._build_right_pane())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)

        self.setCentralWidget(splitter)
        self.resize(1100, 700)

    def _build_left_pane(self) -> QWidget:
        pane = QWidget()
        layout = QVBoxLayout(pane)

        file_row = QHBoxLayout()
        open_btn = QPushButton("Open…")
        open_btn.clicked.connect(self._on_open)
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._on_save)
        save_as_btn = QPushButton("Save as…")
        save_as_btn.clicked.connect(self._on_save_as)
        file_row.addWidget(open_btn)
        file_row.addWidget(save_btn)
        file_row.addWidget(save_as_btn)
        layout.addLayout(file_row)

        add_btn = QPushButton("Add panels…")
        add_btn.clicked.connect(self._on_add_panels)
        layout.addWidget(add_btn)

        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Outline"])
        layout.addWidget(self._tree, 1)

        edit_row1 = QHBoxLayout()
        row_btn = QPushButton("Row")
        row_btn.clicked.connect(self.add_row)
        col_btn = QPushButton("Col")
        col_btn.clicked.connect(self.add_col)
        spacer_btn = QPushButton("Spacer")
        spacer_btn.clicked.connect(self.add_spacer)
        text_btn = QPushButton("Text")
        text_btn.clicked.connect(self.add_text)
        edit_row1.addWidget(row_btn)
        edit_row1.addWidget(col_btn)
        edit_row1.addWidget(spacer_btn)
        edit_row1.addWidget(text_btn)
        layout.addLayout(edit_row1)

        edit_row2 = QHBoxLayout()
        up_btn = QPushButton("↑")
        up_btn.clicked.connect(lambda: self.move_selected(-1))
        down_btn = QPushButton("↓")
        down_btn.clicked.connect(lambda: self.move_selected(1))
        delete_btn = QPushButton("Delete")
        delete_btn.clicked.connect(self.delete_selected)
        group_btn = QPushButton("Group")
        group_btn.clicked.connect(self.toggle_group_selected)
        label_btn = QPushButton("Label…")
        label_btn.clicked.connect(self._on_label_selected)
        edit_row2.addWidget(up_btn)
        edit_row2.addWidget(down_btn)
        edit_row2.addWidget(delete_btn)
        edit_row2.addWidget(group_btn)
        edit_row2.addWidget(label_btn)
        layout.addLayout(edit_row2)

        return pane

    def _build_center_pane(self) -> QWidget:
        pane = QWidget()
        layout = QVBoxLayout(pane)
        label = QLabel("preview")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        return pane

    def _build_right_pane(self) -> QWidget:
        pane = QWidget()
        layout = QVBoxLayout(pane)
        label = QLabel("style")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        return pane

    # -- recipe access ------------------------------------------------------------
    def recipe(self) -> FigureRecipe:
        return self._recipe

    def is_dirty(self) -> bool:
        return self._dirty

    # -- outline tree ---------------------------------------------------------
    def _selected_node(self):
        item = self._tree.currentItem()
        if item is None:
            return None
        return item.data(0, Qt.ItemDataRole.UserRole)

    def _node_label(self, node) -> str:
        if isinstance(node, Row):
            return "Row" + (" [group]" if node.group_label else "")
        if isinstance(node, Col):
            return "Col" + (" [group]" if node.group_label else "")
        if isinstance(node, PanelRef):
            panel = self._recipe.panel_by_id().get(node.panel_id)
            suffix = f" ({panel.label})" if panel and panel.label else ""
            return f"Panel: {node.panel_id}{suffix}"
        if isinstance(node, Spacer):
            return f"Spacer {node.w_cm:g}×{node.h_cm:g} cm"
        if isinstance(node, TextCell):
            return f"Text: {node.text}"
        return str(node)

    def _build_item(self, node) -> QTreeWidgetItem:
        item = QTreeWidgetItem([self._node_label(node)])
        item.setData(0, Qt.ItemDataRole.UserRole, node)
        if isinstance(node, (Row, Col)):
            for child in node.children:
                item.addChild(self._build_item(child))
        return item

    def _rebuild_tree(self) -> None:
        self._tree.clear()
        root_item = self._build_item(self._recipe.layout)
        self._tree.addTopLevelItem(root_item)
        self._tree.expandAll()

    def _update_title(self) -> None:
        star = " *" if self._dirty else ""
        self.setWindowTitle(f"Figure builder — {self._recipe.name}{star}")

    def _after_mutation(self) -> None:
        self._dirty = True
        self._rebuild_tree()
        self._update_title()

    # -- outline structural helpers --------------------------------------------
    def _parent_and_index(self, target):
        """Return ``(container, index)`` of *target* by identity search, or ``None``."""

        def walk(container):
            if isinstance(container, (Row, Col)):
                for i, child in enumerate(container.children):
                    if child is target:
                        return container, i
                    found = walk(child)
                    if found is not None:
                        return found
            return None

        return walk(self._recipe.layout)

    def _current_container(self):
        """The Row/Col a new node/panel should append into: the selection if
        it is itself a container, else its parent container, else the root
        (wrapped in a Row first if the root somehow isn't a container)."""
        if not isinstance(self._recipe.layout, (Row, Col)):
            self._recipe.layout = Row([self._recipe.layout])
        node = self._selected_node()
        if isinstance(node, (Row, Col)):
            return node
        if node is not None:
            found = self._parent_and_index(node)
            if found is not None:
                return found[0]
        return self._recipe.layout

    # -- outline mutators (all testable, no exec()) ----------------------------
    def add_row(self) -> None:
        self._current_container().children.append(Row([]))
        self._after_mutation()

    def add_col(self) -> None:
        self._current_container().children.append(Col([]))
        self._after_mutation()

    def add_spacer(self) -> None:
        self._current_container().children.append(Spacer(2.0, 2.0))
        self._after_mutation()

    def add_text(self) -> None:
        self._current_container().children.append(TextCell("text"))
        self._after_mutation()

    def add_panels(self, panels: list[PanelDef]) -> None:
        container = self._current_container()
        existing_ids = {p.id for p in self._recipe.panels}
        for p in panels:
            pid = p.id
            if pid in existing_ids:
                n = 1
                while f"{pid}_{n}" in existing_ids:
                    n += 1
                p = dc_replace(p, id=f"{pid}_{n}")
            existing_ids.add(p.id)
            self._recipe.panels.append(p)
            container.children.append(PanelRef(p.id))
        self._after_mutation()

    def move_selected(self, delta: int) -> None:
        node = self._selected_node()
        if node is None or node is self._recipe.layout:
            return
        found = self._parent_and_index(node)
        if found is None:
            return
        container, idx = found
        new_idx = idx + delta
        if new_idx < 0 or new_idx >= len(container.children):
            return
        container.children[idx], container.children[new_idx] = (
            container.children[new_idx],
            container.children[idx],
        )
        self._after_mutation()

    def delete_selected(self) -> None:
        node = self._selected_node()
        if node is None or node is self._recipe.layout:
            return
        found = self._parent_and_index(node)
        if found is None:
            return
        container, idx = found
        del container.children[idx]
        self._after_mutation()

    def toggle_group_selected(self) -> None:
        node = self._selected_node()
        if not isinstance(node, (Row, Col)):
            return
        node.group_label = None if node.group_label is not None else "auto"
        self._after_mutation()

    def set_selected_label(self, text: str) -> None:
        node = self._selected_node()
        if node is None:
            return
        if isinstance(node, PanelRef):
            panel = self._recipe.panel_by_id().get(node.panel_id)
            if panel is not None:
                panel.label = text
        elif isinstance(node, (Row, Col)):
            node.group_label = text or None
        elif isinstance(node, TextCell):
            node.text = text
        self._after_mutation()

    # -- slots ------------------------------------------------------------------
    def _on_add_panels(self) -> None:
        defaults = self._defaults_provider()
        dlg = AddPanelDialog(defaults, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.add_panels(dlg.selected_panels)

    def _on_label_selected(self) -> None:
        node = self._selected_node()
        if node is None:
            return
        current = ""
        if isinstance(node, TextCell):
            current = node.text
        elif isinstance(node, PanelRef):
            panel = self._recipe.panel_by_id().get(node.panel_id)
            current = (panel.label if panel and panel.label else "") or ""
        elif isinstance(node, (Row, Col)):
            current = node.group_label or ""
        text, ok = QInputDialog.getText(self, "Label", "Label text:", text=current)
        if ok:
            self.set_selected_label(text)

    def _on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open figure recipe", "", "Figure recipe (*.json)"
        )
        if not path:
            return
        try:
            self.load_recipe_file(path)
        except Exception as exc:  # noqa: BLE001 — surface load errors, never crash
            QMessageBox.warning(self, "Open failed", str(exc))

    def _on_save(self) -> None:
        if self._current_path:
            self.save_recipe_file(self._current_path)
        else:
            self._on_save_as()

    def _on_save_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save figure recipe", self._current_path or "", "Figure recipe (*.json)"
        )
        if not path:
            return
        try:
            self.save_recipe_file(path)
        except Exception as exc:  # noqa: BLE001 — surface save errors, never crash
            QMessageBox.warning(self, "Save failed", str(exc))

    # -- recipe file I/O --------------------------------------------------------
    def load_recipe_file(self, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        base_dir = os.path.dirname(os.path.abspath(path)) or None
        recipe = recipe_from_json(text, base_dir=base_dir)
        if not recipe.name:
            recipe.name = os.path.splitext(os.path.basename(path))[0]
        self._recipe = recipe
        self._current_path = path
        self._dirty = False
        self._rebuild_tree()
        self._update_title()

    def save_recipe_file(self, path: str) -> None:
        base_dir = os.path.dirname(os.path.abspath(path)) or None
        text = recipe_to_json(self._recipe, base_dir=base_dir)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        self._current_path = path
        self._dirty = False
        self._update_title()

    # -- lifecycle ----------------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802 — Qt override signature
        if not self._dirty:
            event.accept()
            return
        ret = QMessageBox.question(
            self,
            "Unsaved changes",
            "Save changes before closing?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if ret == QMessageBox.StandardButton.Cancel:
            event.ignore()
            return
        if ret == QMessageBox.StandardButton.Save:
            self._on_save()
            if self._dirty:  # Save As was cancelled — don't close with unsaved work
                event.ignore()
                return
        event.accept()
