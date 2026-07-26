"""Mark planes… dialog: browse slice planes visually and star the interesting ones.

Marks are offsets stored under ``/marks/<slice_name>`` inside the
oblique_slices.h5 itself (``dfxm.stages.slices.read_marks/write_marks``), so
every plane list in the app can show them. Save briefly closes the browser's
read handle, writes, and reopens (HDF5 file locking forbids r+ while a read
handle is open in the same process).
"""

from __future__ import annotations

import os

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from dfxm.common.errors import StageUserError
from dfxm.stages import slices as _sl

from .plane_browser import PlaneBrowser


class MarkPlanesDialog(QDialog):
    """Star planes of an oblique_slices.h5; Save persists to ``/marks``."""

    def __init__(self, h5_path, parent=None) -> None:
        super().__init__(parent)
        self._path = str(h5_path)
        self.saved = False
        self._browser = PlaneBrowser(h5_path)
        names = self._browser.slice_names()
        if not names:
            self._browser.close_file()
            raise KeyError(f"no slice groups in {h5_path}")

        # marks state: slice_name -> set of plane indices (offsets resolve on save)
        self._offsets: dict[str, list[float]] = {}
        for e in _sl.replot_catalog(self._path):
            self._offsets.setdefault(e.slice_name, list(e.offsets_um))
        self._marks: dict[str, set[int]] = {}
        for sname, offs in _sl.read_marks(self._browser.file).items():
            stored = self._offsets.get(sname)
            if not stored:
                continue
            self._marks[sname] = {
                min(range(len(stored)), key=lambda i: abs(stored[i] - o)) for o in offs
            }
        self._baseline = {k: set(v) for k, v in self._marks.items()}

        self.setWindowTitle(f"Mark planes — {os.path.basename(self._path)}")
        self._info = QLabel()
        self._slice_box = QComboBox()
        self._slice_box.addItems(names)
        self._group_box = QComboBox()
        self._mark_btn = QPushButton("★ Mark")
        self._mark_btn.setCheckable(True)
        self._prev = QPushButton("◀ plane")
        self._next = QPushButton("plane ▶")
        save = QPushButton("Save")
        save.setProperty("role", "primary")
        close = QPushButton("Close")

        top = QHBoxLayout()
        top.addWidget(QLabel("Slice:"))
        top.addWidget(self._slice_box)
        top.addWidget(QLabel("Background:"))
        top.addWidget(self._group_box)
        top.addStretch(1)

        nav = QHBoxLayout()
        nav.addWidget(self._prev)
        nav.addWidget(self._next)
        nav.addWidget(self._mark_btn)
        nav.addStretch(1)
        nav.addWidget(save)
        nav.addWidget(close)

        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self._browser, 1)
        lay.addWidget(self._info)
        lay.addLayout(nav)

        self._prev.clicked.connect(lambda: self._browser.step(-1))
        self._next.clicked.connect(lambda: self._browser.step(+1))
        self._mark_btn.toggled.connect(self._on_mark_toggled)
        self._slice_box.currentTextChanged.connect(self._on_slice_changed)
        self._group_box.currentTextChanged.connect(self._on_group_changed)
        self._browser.viewChanged.connect(self._sync_controls)
        save.clicked.connect(self._on_save)
        close.clicked.connect(self.reject)

        self._on_slice_changed(names[0])

    # -- state --------------------------------------------------------------
    def _dirty(self) -> bool:
        a = {k: v for k, v in self._marks.items() if v}
        b = {k: v for k, v in self._baseline.items() if v}
        return a != b

    def _current_marked(self) -> bool:
        s = self._browser.slice_name
        return s is not None and self._browser.plane_index in self._marks.get(s, ())

    # -- slots ----------------------------------------------------------------
    def _on_slice_changed(self, sname: str) -> None:
        self._browser.open_slice(sname)
        self._group_box.blockSignals(True)
        self._group_box.clear()
        self._group_box.addItems(self._browser.present)
        self._group_box.setCurrentText(self._browser.group_id)
        self._group_box.blockSignals(False)
        self._sync_controls()

    def _on_group_changed(self, vid: str) -> None:
        if vid:
            self._browser.set_group(vid)

    def _on_mark_toggled(self, checked: bool) -> None:
        s = self._browser.slice_name
        if s is None:
            return
        idxs = self._marks.setdefault(s, set())
        if checked:
            idxs.add(self._browser.plane_index)
        else:
            idxs.discard(self._browser.plane_index)
        self._sync_controls()

    def _sync_controls(self) -> None:
        s = self._browser.slice_name
        n_marked = len(self._marks.get(s, ()))
        off = self._browser.current_offset()
        n = len(self._browser.offsets)
        star = "  ★" if self._current_marked() else ""
        self._info.setText(
            f"plane {self._browser.plane_index + 1}/{n}  offset {off:+.3f} µm{star}"
            f"   |   ★ {n_marked} marked in this slice"
            + ("   |   unsaved changes" if self._dirty() else "")
        )
        self._mark_btn.blockSignals(True)
        self._mark_btn.setChecked(self._current_marked())
        self._mark_btn.blockSignals(False)

    def _on_save(self) -> None:
        self._browser.close_file()
        save_exc: StageUserError | None = None
        try:
            for sname in sorted(set(self._baseline) | set(self._marks)):
                offs = [self._offsets[sname][i] for i in sorted(self._marks.get(sname, set()))]
                _sl.write_marks(self._path, sname, offs)
        except StageUserError as exc:
            save_exc = exc
        finally:
            try:
                self._browser.reopen()
            except Exception:
                QMessageBox.warning(
                    self,
                    "Save marks",
                    "The file could not be re-opened after saving marks (it may be "
                    "locked or was rewritten elsewhere). Close this dialog and reopen "
                    "it to continue.",
                )
                for w in (
                    self._prev,
                    self._next,
                    self._mark_btn,
                    self._slice_box,
                    self._group_box,
                ):
                    w.setEnabled(False)
                return
        if save_exc is not None:
            QMessageBox.warning(
                self,
                "Save marks",
                f"{save_exc}\n\n{save_exc.hint}" if save_exc.hint else str(save_exc),
            )
            return
        self._baseline = {k: set(v) for k, v in self._marks.items()}
        self.saved = True
        self._sync_controls()

    # -- close ------------------------------------------------------------
    def reject(self) -> None:  # noqa: D401 - Qt override
        if self._dirty():
            ans = QMessageBox.question(
                self, "Discard mark changes?", "You have unsaved mark changes. Discard them?"
            )
            if ans != QMessageBox.StandardButton.Yes:
                return
        super().reject()

    def done(self, code) -> None:  # noqa: D401 - Qt override
        self._browser.close_file()
        super().done(code)
