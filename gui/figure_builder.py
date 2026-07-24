"""Publication figure-builder window (Phase B, Tasks 10-11).

A non-modal window with a left sources+outline pane (Add panels…, the recipe
outline tree, structural-edit buttons, and recipe file I/O), a center preview
pane with a cached, debounced live render + notes bar (Task 11), and a right
style pane — still a placeholder here, filled in by Task 12. Only the
Qt-free recipe model (:mod:`dfxm.compose.recipe`) is imported at module
level; the render/compose machinery (matplotlib-heavy) is imported lazily
inside :meth:`FigureBuilderWindow.render_now`/`_show_figure` so importing
this module stays light.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from dataclasses import replace as dc_replace

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from dfxm.common.plotting import CMAP_CHOICES, PlotStyle, style_from_params
from dfxm.compose.recipe import (
    SCALE_BAR_MODES,
    Col,
    ComposeStyle,
    FigureRecipe,
    PanelDef,
    PanelRef,
    Row,
    Spacer,
    TextCell,
    iter_leaves,
    recipe_from_json,
    recipe_to_json,
)

from .widgets.export_dialog import StyleControls
from .widgets.panel_picker import AddPanelDialog

# Tri-state (Follow/On/Off) combo choices shared by show_title/colorbar override
# rows — display label -> stored value (None = follow the composed default).
_TRI_STATE = (("Follow", None), ("On", True), ("Off", False))


class FigureBuilderWindow(QMainWindow):
    """Non-modal window for composing a multi-panel publication figure.

    ``defaults_provider`` is a zero-arg callable the main window supplies —
    it closes over the live experiment + stage forms and returns the
    ``{stage: {...}}`` dict :class:`~gui.widgets.panel_picker.AddPanelDialog`
    needs. ``style`` is the session's live :class:`~dfxm.common.plotting.PlotStyle`;
    it seeds the new recipe's ``style`` override dict (so a fresh figure
    already renders in the app's current look) and is kept around for the
    style pane built in Task 12.
    """

    def __init__(self, defaults_provider, style: PlotStyle, parent=None) -> None:
        super().__init__(parent)
        self._defaults_provider = defaults_provider
        # Independent working copy — builder edits (via the Style pane below)
        # must never mutate the app-wide session style the caller passed in.
        self._style = dc_replace(style)
        self._override_panel: PanelDef | None = None
        # Seed the new recipe's style overrides from the session's live style
        # so a freshly opened builder (and any recipe built from here without
        # ever touching Task 12's style pane) renders with the same look the
        # rest of the GUI is using — recipe.style stays a plain JSON-safe dict
        # (see FigureRecipe.style) so it round-trips through save/open intact.
        self._recipe = FigureRecipe("untitled", asdict(style), ComposeStyle(), Row([]), [])
        self._dirty = False
        self._current_path: str | None = None

        self._cache: dict = {}
        self._canvas = None
        self._result = None
        self._preview_host = QWidget()
        self._preview_layout = QVBoxLayout(self._preview_host)
        self._preview_layout.setContentsMargins(0, 0, 0, 0)
        self._notes_label = QLabel("")
        self._notes_label.setWordWrap(True)
        self._refresh_btn = QPushButton("Refresh data")
        self._refresh_btn.clicked.connect(self.refresh_data)
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(300)
        self._debounce.timeout.connect(self.render_now)

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
        self._tree.currentItemChanged.connect(self._on_tree_selection_changed)
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
        layout.addWidget(self._refresh_btn)
        layout.addWidget(self._preview_host, 1)
        layout.addWidget(self._notes_label)
        return pane

    def _build_right_pane(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        layout = QVBoxLayout(container)

        layout.addWidget(QLabel("<b>Style</b>"))
        self._controls = StyleControls(self._style)
        self._controls.changed.connect(self._sync_style_to_recipe)
        layout.addWidget(self._controls)

        layout.addWidget(QLabel("<b>Compose</b>"))
        layout.addLayout(self._build_compose_form())

        layout.addWidget(QLabel("<b>Selected panel overrides</b>"))
        layout.addWidget(self._build_override_editor())

        export_btn = QPushButton("Export…")
        export_btn.clicked.connect(self.export_now)
        layout.addWidget(export_btn)

        layout.addStretch(1)
        scroll.setWidget(container)
        return scroll

    def _build_compose_form(self) -> QFormLayout:
        """Widgets bound to ``recipe.compose``; every edit calls ``_on_compose_edited``."""
        c = self._recipe.compose
        form = QFormLayout()

        self._compose_template = QLineEdit(c.label_template)
        self._compose_template.textChanged.connect(self._on_compose_edited)
        form.addRow("Label template", self._compose_template)

        self._compose_font_scale = QDoubleSpinBox()
        self._compose_font_scale.setRange(0.1, 5.0)
        self._compose_font_scale.setDecimals(2)
        self._compose_font_scale.setSingleStep(0.1)
        self._compose_font_scale.setValue(c.label_font_scale)
        self._compose_font_scale.valueChanged.connect(self._on_compose_edited)
        form.addRow("Label font scale", self._compose_font_scale)

        self._compose_gutter = QDoubleSpinBox()
        self._compose_gutter.setRange(0.01, 20.0)
        self._compose_gutter.setDecimals(2)
        self._compose_gutter.setSuffix(" cm")
        self._compose_gutter.setValue(c.gutter_cm)
        self._compose_gutter.valueChanged.connect(self._on_compose_edited)
        form.addRow("Gutter", self._compose_gutter)

        self._compose_padding = QDoubleSpinBox()
        self._compose_padding.setRange(0.01, 20.0)
        self._compose_padding.setDecimals(2)
        self._compose_padding.setSuffix(" cm")
        self._compose_padding.setValue(c.padding_cm)
        self._compose_padding.valueChanged.connect(self._on_compose_edited)
        form.addRow("Padding", self._compose_padding)

        self._compose_scale_bar_mode = QComboBox()
        self._compose_scale_bar_mode.addItems(list(SCALE_BAR_MODES))
        self._compose_scale_bar_mode.setCurrentText(c.scale_bar_mode)
        self._compose_scale_bar_mode.currentTextChanged.connect(self._on_compose_edited)
        form.addRow("Scale-bar mode", self._compose_scale_bar_mode)

        self._compose_scale_bar_panel = QComboBox()
        self._compose_scale_bar_panel.currentTextChanged.connect(self._on_compose_edited)
        form.addRow("Scale-bar panel (one-panel mode)", self._compose_scale_bar_panel)
        self._refresh_compose_panel_combo()

        self._compose_pinned_width = QDoubleSpinBox()
        self._compose_pinned_width.setRange(0.0, 1000.0)
        self._compose_pinned_width.setDecimals(2)
        self._compose_pinned_width.setSuffix(" cm")
        self._compose_pinned_width.setSpecialValueText("auto")
        self._compose_pinned_width.setValue(c.pinned_width_cm or 0.0)
        self._compose_pinned_width.valueChanged.connect(self._on_compose_edited)
        form.addRow("Pinned total width (0 = auto)", self._compose_pinned_width)

        return form

    def _refresh_compose_panel_combo(self) -> None:
        """Repopulate the scale-bar-panel combo with the recipe's current panel ids."""
        combo = self._compose_scale_bar_panel
        target = self._recipe.compose.scale_bar_panel or ""
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("")  # "" = no single panel designated
        for p in self._recipe.panels:
            combo.addItem(p.id)
        idx = combo.findText(target)
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        combo.blockSignals(False)

    def _load_compose_into_widgets(self) -> None:
        """Refresh every compose widget from ``self._recipe.compose`` (e.g. after a load)."""
        c = self._recipe.compose
        widgets = (
            self._compose_template,
            self._compose_font_scale,
            self._compose_gutter,
            self._compose_padding,
            self._compose_scale_bar_mode,
            self._compose_pinned_width,
        )
        for w in widgets:
            w.blockSignals(True)
        self._compose_template.setText(c.label_template)
        self._compose_font_scale.setValue(c.label_font_scale)
        self._compose_gutter.setValue(c.gutter_cm)
        self._compose_padding.setValue(c.padding_cm)
        self._compose_scale_bar_mode.setCurrentText(c.scale_bar_mode)
        self._compose_pinned_width.setValue(c.pinned_width_cm or 0.0)
        for w in widgets:
            w.blockSignals(False)
        self._refresh_compose_panel_combo()

    def _on_compose_edited(self, *_args) -> None:
        c = self._recipe.compose
        c.label_template = self._compose_template.text()
        c.label_font_scale = self._compose_font_scale.value()
        c.gutter_cm = self._compose_gutter.value()
        c.padding_cm = self._compose_padding.value()
        c.scale_bar_mode = self._compose_scale_bar_mode.currentText()
        c.scale_bar_panel = self._compose_scale_bar_panel.currentText() or None
        pinned = self._compose_pinned_width.value()
        c.pinned_width_cm = pinned if pinned > 0 else None
        self._dirty = True
        self._update_title()
        self.schedule_preview()

    # -- style pane -------------------------------------------------------------
    def _sync_style_to_recipe(self) -> None:
        self._recipe.style = asdict(self._style)
        self._dirty = True
        self._update_title()
        self.schedule_preview()

    # -- per-node override editor -------------------------------------------------
    def _build_override_editor(self) -> QWidget:
        self._override_group = QWidget()
        form = QFormLayout(self._override_group)

        self._ov_roi = QLineEdit()
        self._ov_roi.setPlaceholderText("r0,r1,c0,c1 (blank = full)")
        self._ov_roi.textChanged.connect(lambda _t: self._on_override_field_edited("roi"))
        form.addRow("ROI crop (px)", self._ov_roi)

        self._ov_clim = QLineEdit()
        self._ov_clim.setPlaceholderText("lo,hi (blank half ok; blank both = stored)")
        self._ov_clim.textChanged.connect(lambda _t: self._on_override_field_edited("clim"))
        form.addRow("Colour limits", self._ov_clim)

        self._ov_cmap = QComboBox()
        self._ov_cmap.addItem("")  # follow style
        self._ov_cmap.addItems(list(CMAP_CHOICES))
        self._ov_cmap.currentTextChanged.connect(lambda _t: self._on_override_field_edited("cmap"))
        form.addRow("Colormap", self._ov_cmap)

        self._ov_label = QLineEdit()
        self._ov_label.setPlaceholderText("(blank = auto sequence letter)")
        self._ov_label.textChanged.connect(lambda _t: self._on_override_field_edited("label"))
        form.addRow("Label", self._ov_label)

        self._ov_show_title = QComboBox()
        for text, value in _TRI_STATE:
            self._ov_show_title.addItem(text, value)
        self._ov_show_title.currentIndexChanged.connect(
            lambda _i: self._on_override_field_edited("show_title")
        )
        form.addRow("Show title", self._ov_show_title)

        self._ov_scale = QDoubleSpinBox()
        self._ov_scale.setRange(0.0, 100_000.0)
        self._ov_scale.setDecimals(3)
        self._ov_scale.setSuffix(" µm/cm")
        self._ov_scale.setSpecialValueText("follow style")
        self._ov_scale.valueChanged.connect(
            lambda _v: self._on_override_field_edited("scale_um_per_cm")
        )
        form.addRow("Panel scale", self._ov_scale)

        self._ov_colorbar = QComboBox()
        for text, value in _TRI_STATE:
            self._ov_colorbar.addItem(text, value)
        self._ov_colorbar.currentIndexChanged.connect(
            lambda _i: self._on_override_field_edited("colorbar")
        )
        form.addRow("Colourbar", self._ov_colorbar)

        self._override_group.setEnabled(False)
        return self._override_group

    def _on_tree_selection_changed(self, *_args) -> None:
        """Show/enable the override editor for a selected panel; disable otherwise."""
        node = self._selected_node()
        panel = None
        if isinstance(node, PanelRef):
            panel = self._recipe.panel_by_id().get(node.panel_id)
        self._override_panel = panel
        if panel is None:
            self._override_group.setEnabled(False)
            return
        widgets = (
            self._ov_roi,
            self._ov_clim,
            self._ov_cmap,
            self._ov_label,
            self._ov_show_title,
            self._ov_scale,
            self._ov_colorbar,
        )
        for w in widgets:
            w.blockSignals(True)
        self._ov_roi.setText(",".join(str(v) for v in panel.roi) if panel.roi else "")
        if panel.clim is not None:
            lo, hi = panel.clim
            lo_s = "" if lo is None else f"{lo:g}"
            hi_s = "" if hi is None else f"{hi:g}"
            self._ov_clim.setText(f"{lo_s},{hi_s}")
        else:
            self._ov_clim.setText("")
        self._ov_cmap.setCurrentText(panel.cmap or "")
        self._ov_label.setText(panel.label or "")
        self._ov_show_title.setCurrentIndex(
            next(i for i, (_t, v) in enumerate(_TRI_STATE) if v is panel.show_title)
        )
        self._ov_scale.setValue(panel.scale_um_per_cm or 0.0)
        self._ov_colorbar.setCurrentIndex(
            next(i for i, (_t, v) in enumerate(_TRI_STATE) if v is panel.colorbar)
        )
        for w in widgets:
            w.blockSignals(False)
        self._override_group.setEnabled(True)

    def _on_override_field_edited(self, key: str) -> None:
        """Submit only the ONE widget the user actually changed.

        Submitting the whole widget set on every edit (the original Task 12
        shape) meant every other field got silently re-derived from its
        current (possibly lossy, e.g. clim's %g display) widget text on each
        unrelated edit — in particular it collapsed an explicitly-suppressed
        ``PanelDef.label == ""`` back to ``None`` (auto-lettering) the moment
        any other override field was touched, since "" and None display
        identically as a blank label box. Submitting one key at a time means
        :meth:`_apply_panel_overrides` only ever looks at (and only ever
        mutates) the field that changed.
        """
        if self._override_panel is None:
            return
        getters = {
            "roi": self._ov_roi.text,
            "clim": self._ov_clim.text,
            "cmap": self._ov_cmap.currentText,
            "label": self._ov_label.text,
            "show_title": self._ov_show_title.currentData,
            "scale_um_per_cm": self._ov_scale.value,
            "colorbar": self._ov_colorbar.currentData,
        }
        self._apply_panel_overrides(self._override_panel, {key: getters[key]()})

    @staticmethod
    def _parse_int(text: str) -> int | None:
        try:
            return int(text)
        except ValueError:
            return None

    def _apply_panel_overrides(self, panel: PanelDef, values: dict) -> None:
        """Parse and apply only the override keys PRESENT in *values* onto *panel*.

        This is a partial update by design: the override editor's widgets each
        submit only their own key via :meth:`_on_override_field_edited` (a test
        may still hand in all seven keys at once, e.g. to seed every field from
        a fresh selection). A key absent from *values* is left completely
        untouched on *panel* — this is what stops an edit to one field (say
        ROI) from re-deriving and clobbering another (say a ``label`` explicitly
        set to ``""``, or a ``clim`` more precise than its %g display). Malformed
        ROI/clim text reports to the notes bar and mutates nothing (including
        any other key also present in this same call); every other key is
        already structurally valid (combo/spinbox/tri-state values), so it is
        assigned as-is.
        """
        new_roi = panel.roi
        if "roi" in values:
            roi_text = (values.get("roi") or "").strip()
            if roi_text:
                parts = [p.strip() for p in roi_text.split(",")]
                ints = [self._parse_int(p) for p in parts]
                if len(ints) != 4 or any(v is None for v in ints):
                    self._notes_label.setText(
                        f"invalid ROI text {roi_text!r} — expected 'r0,r1,c0,c1' (four integers)"
                    )
                    return
                new_roi = tuple(ints)
            else:
                new_roi = None

        new_clim = panel.clim
        if "clim" in values:
            clim_text = (values.get("clim") or "").strip()
            if clim_text:
                parts = clim_text.split(",")
                if len(parts) != 2:
                    self._notes_label.setText(
                        f"invalid clim text {clim_text!r} — expected 'lo,hi' "
                        "(either half may be blank)"
                    )
                    return
                lo_s, hi_s = (p.strip() for p in parts)
                try:
                    lo = float(lo_s) if lo_s else None
                    hi = float(hi_s) if hi_s else None
                except ValueError:
                    self._notes_label.setText(
                        f"invalid clim text {clim_text!r} — expected 'lo,hi' "
                        "(either half may be blank)"
                    )
                    return
                new_clim = (lo, hi)
            else:
                new_clim = None

        if "roi" in values:
            panel.roi = new_roi
        if "clim" in values:
            panel.clim = new_clim
        if "cmap" in values:
            panel.cmap = values["cmap"] or None
        if "label" in values:
            panel.label = values["label"] or None
        if "show_title" in values:
            panel.show_title = values["show_title"]
        if "scale_um_per_cm" in values:
            scale = values["scale_um_per_cm"]
            panel.scale_um_per_cm = float(scale) if scale else None
        if "colorbar" in values:
            panel.colorbar = values["colorbar"]

        self._dirty = True
        self._rebuild_tree()
        self._update_title()
        self.schedule_preview()
        self._select_outline_panel(panel.id)

    # -- export -------------------------------------------------------------------
    def export_now(self) -> None:
        from dfxm.common.errors import StageUserError
        from dfxm.compose.render import export_recipe

        out = QFileDialog.getExistingDirectory(self, "Export directory")
        if not out:
            return
        try:
            paths, res = export_recipe(self._recipe, out, loader_cache=self._cache)
        except StageUserError as exc:
            hint = f"  Hint: {exc.hint}" if exc.hint else ""
            self._notes_label.setText(f"export failed: {exc}{hint}")
            return
        except Exception as exc:  # noqa: BLE001 — export must never crash the window
            self._notes_label.setText(f"export failed: {exc}")
            return
        notes = f"; {'; '.join(res.notes)}" if res.notes else ""
        self._notes_label.setText(f"wrote {len(paths)} file(s) → {out}{notes}")

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
        self._refresh_compose_panel_combo()
        self._update_title()
        self.schedule_preview()

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
        self._purge_orphaned_panels()
        self._after_mutation()

    def _purge_orphaned_panels(self) -> None:
        """Drop any PanelDef no longer referenced by a layout leaf.

        Deleting a Row/Col removes every PanelRef nested under it in one go
        (they were only reachable through that container's ``children``), but
        their backing PanelDefs live in the flat ``recipe.panels`` list and
        survive unless removed here. An orphaned PanelDef crashes a
        subsequent gutter-mode render (``render_recipe`` loads data for every
        panel in ``recipe.panels``, but its post-placement pid bookkeeping
        only covers layout leaves) — see ``dfxm.compose.render.render_recipe``.
        Also scrubs ``compose.scale_bar_panel`` if it named a purged id.
        """
        live_ids = {
            leaf.panel_id for leaf in iter_leaves(self._recipe.layout) if isinstance(leaf, PanelRef)
        }
        self._recipe.panels = [p for p in self._recipe.panels if p.id in live_ids]
        if self._recipe.compose.scale_bar_panel not in live_ids:
            self._recipe.compose.scale_bar_panel = None

    def toggle_group_selected(self) -> None:
        node = self._selected_node()
        if not isinstance(node, (Row, Col)):
            return
        node.group_label = None if node.group_label is not None else "auto"
        self._after_mutation()

    def set_selected_label(self, text: str) -> None:
        node = self._selected_node()
        if isinstance(node, PanelRef):
            panel = self._recipe.panel_by_id().get(node.panel_id)
            if panel is None:  # orphaned ref (shouldn't happen, but never mutate on it)
                return
            panel.label = text
        elif isinstance(node, (Row, Col)):
            node.group_label = text or None
        elif isinstance(node, TextCell):
            node.text = text
        else:  # None selection, or a node type with no label concept (e.g. Spacer)
            return
        self._after_mutation()

    # -- preview: cached debounced render --------------------------------------
    def schedule_preview(self) -> None:
        """Restart the 300 ms debounce; the actual render runs on timeout."""
        self._debounce.start()

    def refresh_data(self) -> None:
        """Drop the loader cache (stale/deleted source files) and re-render now."""
        self._cache.clear()
        self.render_now()

    def render_now(self):
        """Render the current recipe from ``self._cache`` right away.

        Returns the :class:`~dfxm.compose.render.ComposeResult` on success, or
        ``None`` if there was nothing to render or the render was refused —
        either way the notes bar carries the explanation and the window never
        crashes.
        """
        from dfxm.common.errors import StageUserError
        from dfxm.compose.render import render_recipe

        if not self._recipe.panels:
            self._notes_label.setText("add panels to preview")
            return None
        try:
            result = render_recipe(self._recipe, loader_cache=self._cache)
        except StageUserError as exc:
            hint = f"  Hint: {exc.hint}" if exc.hint else ""
            self._notes_label.setText(f"cannot render: {exc}{hint}")
            return None
        except Exception as exc:  # noqa: BLE001 — preview must never crash the window
            self._notes_label.setText(f"render failed: {exc}")
            return None
        self._show_figure(result.figure)
        self._result = result
        self._notes_label.setText("; ".join(result.notes) if result.notes else "")
        return result

    def _show_figure(self, figure) -> None:
        """Swap in a fresh, undecorated ``FigureCanvasQTAgg`` for *figure*.

        The composed figure is placed absolutely by the compose layout
        solver, so it must never be re-laid-out by a display canvas — always
        a new plain canvas here, never the themed :class:`~gui.widgets.mpl_canvas.MplCanvas`
        (that would restyle/re-fit the white publication figure and break
        WYSIWYG parity with the export).
        """
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        if self._canvas is not None:
            self._preview_layout.removeWidget(self._canvas)
            self._canvas.deleteLater()
        self._canvas = FigureCanvasQTAgg(figure)
        self._canvas.mpl_connect("button_press_event", self._on_preview_click)
        self._preview_layout.addWidget(self._canvas, 1)
        self._canvas.draw_idle()

    def _on_preview_click(self, event) -> None:
        if event.inaxes is not None:
            self._on_preview_pick(event.inaxes)

    def _on_preview_pick(self, ax) -> None:
        """Select the outline node for the clicked axes (reverse ``axes_by_id``)."""
        if self._result is None:
            return
        for pid, panel_ax in self._result.axes_by_id.items():
            if panel_ax is ax:
                self._select_outline_panel(pid)
                return

    def _select_outline_panel(self, pid: str) -> None:
        """Find and select the tree item for panel id *pid*, if still present."""

        def walk(item):
            node = item.data(0, Qt.ItemDataRole.UserRole)
            if isinstance(node, PanelRef) and node.panel_id == pid:
                return item
            for i in range(item.childCount()):
                found = walk(item.child(i))
                if found is not None:
                    return found
            return None

        root = self._tree.topLevelItem(0)
        if root is None:
            return
        item = walk(root)
        if item is not None:
            self._tree.setCurrentItem(item)

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
        self._style = style_from_params({"plot_style": self._recipe.style}) or PlotStyle()
        self._controls.set_style(self._style)
        self._load_compose_into_widgets()
        self._rebuild_tree()
        self._update_title()
        self.schedule_preview()

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
            self._debounce.stop()
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
        # A pending 300 ms debounce (schedule_preview) must not fire after
        # close — Qt keeps a QTimer parented to this window alive until the
        # event loop next turns, so an in-flight timer can call render_now()
        # against a window that is closing/closed (harmless here since Qt
        # widgets tolerate it, but wasteful and a source of surprising
        # behaviour — e.g. it re-populating the notes label of a hidden
        # window, or firing during an unrelated later event-loop turn).
        self._debounce.stop()
        event.accept()
