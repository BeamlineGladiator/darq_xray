"""Per-kind colour-limit rows for the replot dialogs.

A single ``vmin/vmax`` pair makes no sense when a replot mixes several plot kinds
(a slices file holds mosa_com / mosa_fwhm / strain / raw; a mosaicity stack holds
mosa_com + mosa_fwhm). ``ClimGroupSection`` builds one labelled row per group and
collects a ``{group_key: (vmin, vmax)}`` mapping that the Qt-free
``render_replot`` cores consume via ``dfxm.common.figures.resolve_clim``.

The widget is rebuilt on every file reload (``set_groups``); values already typed
are preserved for groups that survive the reload.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)


class ClimGroupSection(QWidget):
    """One vmin/vmax row per replot group; yields a per-group clim mapping."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._edits: dict[str, tuple[QLineEdit, QLineEdit]] = {}

    def set_groups(self, groups: list[tuple[str, str]]) -> None:
        """Rebuild the rows for ``groups`` (``[(key, label), ...]``).

        Text already entered for a key that is still present is carried over, so
        reloading the same file (or a file with the same kinds) does not wipe the
        user's limits.
        """
        prev = {k: (vm.text(), vx.text()) for k, (vm, vx) in self._edits.items()}
        while self._layout.count():
            item = self._layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._edits.clear()
        for key, label in groups:
            vmin = QLineEdit()
            vmin.setPlaceholderText("vmin (blank = stored)")
            vmax = QLineEdit()
            vmax.setPlaceholderText("vmax (blank = stored)")
            if key in prev:
                vmin.setText(prev[key][0])
                vmax.setText(prev[key][1])
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            lab = QLabel(f"{label}:")
            lab.setMinimumWidth(120)
            row.addWidget(lab)
            row.addWidget(vmin)
            row.addWidget(vmax)
            container = QWidget()
            container.setLayout(row)
            self._layout.addWidget(container)
            self._edits[key] = (vmin, vmax)

    def clim_by_group(self):
        """Return ``{key: (vmin, vmax)}`` for groups with at least one box filled.

        Blank boxes become ``None`` (that limit keeps its stored value). Returns
        ``None`` when no group has any override, so the cores fall through to
        their stored/default limits exactly as before.
        """
        out: dict[str, tuple[float | None, float | None]] = {}
        for key, (vm, vx) in self._edits.items():
            tvm, tvx = vm.text().strip(), vx.text().strip()
            if not tvm and not tvx:
                continue
            out[key] = (float(tvm) if tvm else None, float(tvx) if tvx else None)
        return out or None

    def validate(self) -> str | None:
        """Return an error message for the first unparseable non-empty box, else None."""
        for key, (vm, vx) in self._edits.items():
            for lbl, edit in (("vmin", vm), ("vmax", vx)):
                t = edit.text().strip()
                if t:
                    try:
                        float(t)
                    except ValueError:
                        return f"invalid {lbl} for {key}: {t!r}"
        return None
