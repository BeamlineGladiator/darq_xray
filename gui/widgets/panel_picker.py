"""Add-panels dialog: pick items from a stage's replot catalog and turn the
checked leaves into :class:`~dfxm.compose.recipe.PanelDef` instances for the
figure builder.

Consumes the same Qt-free catalog functions as the existing replot dialogs
(``strain``/``mosaicity``/``rocking``/``slices``/``profiles``), so this is a
thin shell — no figure-composition logic lives here.
"""

from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from dfxm.compose.recipe import PanelDef, PanelSource
from dfxm.stages import mosaicity, profiles, rocking, slices, strain

_MAP_STAGES = {"strain": strain, "mosaicity": mosaicity, "rocking": rocking}
_STAGES = ("strain", "mosaicity", "rocking", "slices", "profiles")


class AddPanelDialog(QDialog):
    """Pick panels from one stage's replot catalog.

    ``defaults`` is ``{stage: {"h5": str, "sx": float, "sy": float,
    "jobs": list[dict]}}`` — pre-fill data the main window builds from the
    live experiment + stage forms. On accept, :attr:`selected_panels` holds
    the built :class:`~dfxm.compose.recipe.PanelDef` list.
    """

    def __init__(self, defaults: dict[str, dict], parent=None) -> None:
        super().__init__(parent)
        self._defaults = defaults or {}
        self.selected_panels: list[PanelDef] = []
        self._counter = 0
        self._catalog: list = []
        self._loaded_h5 = ""  # the path self._catalog was actually built from
        self.setWindowTitle("Add panels")

        self._stage = QComboBox()
        self._stage.addItems(list(_STAGES))
        self._stage.currentTextChanged.connect(self._on_stage_changed)

        self._h5_edit = QLineEdit()
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._on_browse)
        load_btn = QPushButton("Load")
        load_btn.clicked.connect(self._reload)

        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Stage:"))
        top_row.addWidget(self._stage)
        top_row.addWidget(QLabel("File:"))
        top_row.addWidget(self._h5_edit, 1)
        top_row.addWidget(browse_btn)
        top_row.addWidget(load_btn)

        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Item"])

        check_row = QHBoxLayout()
        check_all_btn = QPushButton("Check all")
        check_all_btn.clicked.connect(self._check_all)
        uncheck_all_btn = QPushButton("Uncheck all")
        uncheck_all_btn.clicked.connect(self._uncheck_all)
        check_row.addWidget(check_all_btn)
        check_row.addWidget(uncheck_all_btn)
        check_row.addStretch(1)

        self._status = QLabel("")
        self._status.setWordWrap(True)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(top_row)
        layout.addLayout(check_row)
        layout.addWidget(self._tree, 1)
        layout.addWidget(self._status)
        layout.addWidget(buttons)

        # pre-fill the h5 field for the initially-selected stage and load it.
        self._on_stage_changed(self._stage.currentText())

    # -- population -----------------------------------------------------------
    def _on_stage_changed(self, stage: str) -> None:
        d = self._defaults.get(stage, {})
        self._h5_edit.setText(d.get("h5", "") or "")
        self._reload()

    def _on_browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open h5", "", "HDF5 (*.h5)")
        if path:
            self._h5_edit.setText(path)
            self._reload()

    def _reload(self) -> None:
        stage = self._stage.currentText()
        h5 = self._h5_edit.text().strip()
        self._loaded_h5 = h5  # pin the path the (about-to-be-rebuilt) tree reflects
        self._tree.clear()
        self._catalog = []
        if not h5 or not os.path.exists(h5):
            self._status.setText("no such file")
            return
        try:
            if stage in _MAP_STAGES:
                self._catalog = _MAP_STAGES[stage].replot_catalog(h5)
                self._build_map_tree(stage, self._catalog)
            elif stage == "slices":
                self._catalog = slices.replot_catalog(h5)
                self._build_slice_tree(self._catalog)
            elif stage == "profiles":
                jobs = self._defaults.get("profiles", {}).get("jobs") or []
                self._catalog = profiles.replot_catalog(h5, jobs)
                self._build_profiles_tree(self._catalog, jobs)
        except Exception as exc:  # noqa: BLE001 — catalog reload: show status, never crash
            self._catalog = []
            self._tree.clear()
            self._status.setText(f"cannot read: {exc}")
            return
        self._status.setText(f"{len(self._catalog)} item(s)")

    def _build_map_tree(self, stage: str, catalog) -> None:
        d = self._defaults.get(stage, {})
        sx, sy = d.get("sx"), d.get("sy")
        for grp in catalog:
            top = QTreeWidgetItem([grp.label])
            top.setFlags(
                top.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsAutoTristate
            )
            top.setCheckState(0, Qt.CheckState.Unchecked)
            for z, label in enumerate(grp.item_labels):
                child = QTreeWidgetItem([label])
                child.setFlags(child.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                child.setCheckState(0, Qt.CheckState.Unchecked)
                selector: dict = {"stage": stage, "z": z}
                if stage != "strain":
                    selector["dataset"] = grp.key
                if sx is not None:
                    selector["sx"] = sx
                if sy is not None:
                    selector["sy"] = sy
                child.setData(
                    0, Qt.ItemDataRole.UserRole, {"kind": "map_layer", "selector": selector}
                )
                top.addChild(child)
            self._tree.addTopLevelItem(top)
            top.setExpanded(True)

    def _build_slice_tree(self, catalog) -> None:
        for e in catalog:
            top = QTreeWidgetItem([f"{e.volume_id} / {e.slice_name}"])
            top.setFlags(
                top.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsAutoTristate
            )
            top.setCheckState(0, Qt.CheckState.Unchecked)
            offsets = list(e.offsets_um)
            for k in range(e.n_planes):
                off = offsets[k] if k < len(offsets) else None
                label = f"plane {k}" + (f"  @ {off:+.2f} µm" if off is not None else "")
                child = QTreeWidgetItem([label])
                child.setFlags(child.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                child.setCheckState(0, Qt.CheckState.Unchecked)
                child.setData(
                    0,
                    Qt.ItemDataRole.UserRole,
                    {
                        "kind": "slice_plane",
                        "selector": {
                            "volume_id": e.volume_id,
                            "slice_name": e.slice_name,
                            "plane": k,
                        },
                    },
                )
                top.addChild(child)
            self._tree.addTopLevelItem(top)
            top.setExpanded(True)

    def _build_profiles_tree(self, catalog, jobs: list[dict]) -> None:
        for e in catalog:
            top = QTreeWidgetItem([e.label + ("   · pinned" if e.note else "")])
            top.setFlags(
                top.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsAutoTristate
            )
            top.setCheckState(0, Qt.CheckState.Unchecked)
            job = jobs[e.job_index] if e.job_index < len(jobs) else {"name": e.name}

            ref_child = QTreeWidgetItem(["reference"])
            ref_child.setFlags(ref_child.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            ref_child.setCheckState(0, Qt.CheckState.Unchecked)
            ref_child.setData(
                0,
                Qt.ItemDataRole.UserRole,
                {"kind": "profiles_ref", "selector": {"job": job, "field": None}},
            )
            top.addChild(ref_child)

            for vid in e.fields:
                child = QTreeWidgetItem([vid])
                child.setFlags(child.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                child.setCheckState(0, Qt.CheckState.Unchecked)
                child.setData(
                    0,
                    Qt.ItemDataRole.UserRole,
                    {"kind": "profiles_trace", "selector": {"job": job, "field": vid}},
                )
                top.addChild(child)
            self._tree.addTopLevelItem(top)
            top.setExpanded(True)

    # -- selection --------------------------------------------------------------
    def _check_all(self) -> None:
        self._set_all_leaves(Qt.CheckState.Checked)

    def _uncheck_all(self) -> None:
        self._set_all_leaves(Qt.CheckState.Unchecked)

    def _set_all_leaves(self, state) -> None:
        def walk(item):
            for i in range(item.childCount()):
                child = item.child(i)
                if child.childCount() == 0:
                    child.setCheckState(0, state)
                walk(child)

        for i in range(self._tree.topLevelItemCount()):
            walk(self._tree.topLevelItem(i))

    def _build_panels(self) -> list[PanelDef]:
        """Translate every checked leaf into a :class:`PanelDef` (testable without ``exec()``).

        Uses the h5 path the tree was actually loaded from (``self._loaded_h5``), not
        whatever the file field currently shows — an edit to the field after Load but
        before OK must not silently retarget already-picked panels.
        """
        h5 = self._loaded_h5
        stage = self._stage.currentText()
        panels: list[PanelDef] = []

        def walk(item):
            for i in range(item.childCount()):
                child = item.child(i)
                if child.childCount() == 0 and child.checkState(0) == Qt.CheckState.Checked:
                    data = child.data(0, Qt.ItemDataRole.UserRole)
                    if data:
                        pid = f"{stage}_{self._counter}"
                        self._counter += 1
                        src = PanelSource(h5_path=h5, kind=data["kind"], selector=data["selector"])
                        panels.append(PanelDef(id=pid, source=src))
                walk(child)

        for i in range(self._tree.topLevelItemCount()):
            walk(self._tree.topLevelItem(i))
        return panels

    def accept(self) -> None:
        self.selected_panels = self._build_panels()
        super().accept()
