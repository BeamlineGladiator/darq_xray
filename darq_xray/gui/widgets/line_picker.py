"""Interactive line picker for the profiles stage (built lazily on demand).

Scroll the planes of one slice on the shared PlaneBrowser canvas, switch the
background field group, click two endpoints, and read back
``(start_uv, end_uv, offset_um, fields, reference)`` in the slice's (u, v)
frame. Opened only when the user clicks "Pick line…" (or once per mark in the
"Jobs from marks…" flow), so the consolidated file is read and the canvas
built on demand, never at stage-view construction.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from darq_xray.stages import profiles as _pr
from darq_xray.stages import slices as _sl

from .plane_browser import PlaneBrowser


class LinePickerDialog(QDialog):
    """Modal picker over one slice of an oblique_slices.h5 file.

    On accept, :attr:`result` is ``(start_uv, end_uv, offset_um, fields,
    reference)``; otherwise None. *reference* is the background group the line
    was drawn against — it becomes the job's per-job ``"reference"``.
    """

    def __init__(self, h5_path, slice_name, init_offset=0.0, ref_pref="", parent=None) -> None:
        super().__init__(parent)
        self.result = None
        self._pts: list[tuple[float, float]] = []

        self._browser = PlaneBrowser(h5_path)
        try:
            self._browser.open_slice(slice_name, ref_pref=ref_pref, init_offset=init_offset)
        except Exception:
            self._browser.close_file()
            raise
        self._browser.post_draw = self._post_draw
        self._browser.viewChanged.connect(self._update_info)

        moffs = _sl.read_marks(self._browser.file).get(slice_name, [])
        self._marked_idx = {_pr.resolve_plane_index(self._browser.offsets, o)[0] for o in moffs}

        self.setWindowTitle(f"Pick line — {slice_name} ({self._browser.group_id})")
        self._info = QLabel()

        self._bg = QComboBox()
        self._bg.addItems(self._browser.present)
        self._bg.setCurrentText(self._browser.group_id)
        self._bg.currentTextChanged.connect(self._on_bg_changed)
        bg_row = QHBoxLayout()
        bg_row.addWidget(QLabel("Background:"))
        bg_row.addWidget(self._bg)
        bg_row.addStretch(1)

        self._prev = QPushButton("◀ plane")
        self._next = QPushButton("plane ▶")
        self._use = QPushButton("Use line")
        self._cancel = QPushButton("Cancel")
        self._use.setEnabled(False)
        self._prev.clicked.connect(lambda: self._browser.step(-1))
        self._next.clicked.connect(lambda: self._browser.step(+1))
        self._use.clicked.connect(self._accept)
        self._cancel.clicked.connect(self.reject)

        self._field_boxes: dict[str, QCheckBox] = {}
        fields_row = QHBoxLayout()
        fields_row.addWidget(QLabel("Fields:"))
        for vid in self._browser.present:
            box = QCheckBox(vid)
            box.setChecked(True)
            self._field_boxes[vid] = box
            fields_row.addWidget(box)
        fields_row.addStretch(1)

        nav = QHBoxLayout()
        nav.addWidget(self._prev)
        nav.addWidget(self._next)
        nav.addStretch(1)
        nav.addWidget(self._use)
        nav.addWidget(self._cancel)

        lay = QVBoxLayout(self)
        lay.addWidget(self._browser, 1)
        lay.addWidget(self._info)
        lay.addLayout(bg_row)
        lay.addLayout(fields_row)
        lay.addLayout(nav)

        for box in self._field_boxes.values():
            box.toggled.connect(self._refresh_use_button)

        self._browser.canvas.mpl_connect("button_press_event", self._on_click)
        self._update_info()

    # -- plane display ----------------------------------------------------
    def _post_draw(self, ax) -> None:
        if len(self._pts) == 2:
            (u0, v0), (u1, v1) = self._pts
            ax.plot([u0, u1], [v0, v1], "-o", color="cyan", lw=2, ms=6, zorder=6)

    def _on_bg_changed(self, vid: str) -> None:
        self._browser.set_group(vid)
        self.setWindowTitle(f"Pick line — {self._browser.slice_name} ({vid})")

    def _update_info(self) -> None:
        off = self._browser.current_offset()
        n = len(self._browser.offsets)
        pts = " -> ".join(f"({u:.2f}, {v:.2f})" for u, v in self._pts) or "click two points"
        star = "  ★" if self._browser.plane_index in self._marked_idx else ""
        self._info.setText(
            f"plane {self._browser.plane_index + 1}/{n}  offset {off:+.3f} µm{star}   |   "
            f"line: {pts}"
        )

    # -- interaction ------------------------------------------------------
    def _refresh_use_button(self) -> None:
        """Enable the Use button only when 2 points are picked AND ≥1 field is checked."""
        self._use.setEnabled(len(self._pts) == 2 and bool(self.selected_fields()))

    def _on_click(self, event) -> None:
        if event.inaxes is not self._browser.ax or event.xdata is None or event.ydata is None:
            return
        if len(self._pts) >= 2:
            self._pts = []
        self._pts.append((float(event.xdata), float(event.ydata)))
        self._refresh_use_button()
        self._browser.redraw()

    def _accept(self) -> None:
        if len(self._pts) != 2:
            return
        self.result = (
            self._pts[0],
            self._pts[1],
            self._browser.current_offset(),
            self.field_restriction(),
            self._browser.group_id,
        )
        self.accept()

    def selected_fields(self) -> list[str]:
        """Ticked field ids, in catalog order (all present when none unticked)."""
        return [vid for vid in self._browser.present if self._field_boxes[vid].isChecked()]

    def field_restriction(self) -> list[str] | None:
        """The per-job ``fields`` restriction, or None when the user left ALL
        present fields checked (= no restriction; job auto-adapts to new fields)."""
        sel = self.selected_fields()
        return None if set(sel) == set(self._browser.present) else sel

    # -- cleanup ----------------------------------------------------------
    def done(self, code) -> None:  # noqa: D401 - Qt override
        self._browser.close_file()
        super().done(code)
