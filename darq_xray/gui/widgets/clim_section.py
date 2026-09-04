"""Per-kind colour-limit rows for the replot dialogs.

A single ``vmin/vmax`` pair makes no sense when a replot mixes several plot kinds
(a slices file holds mosa_com / mosa_fwhm / strain / raw; a mosaicity stack holds
mosa_com + mosa_fwhm). ``ClimGroupSection`` builds one labelled row per group and
collects a ``{group_key: (vmin, vmax)}`` mapping that the Qt-free
``render_replot`` cores consume via ``darq_xray.common.figures.resolve_clim``.

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

# Friendly labels for the per-quantity colour-limit rows, keyed by volume_id.
# volume_id is f"{kind}{suffix}" where suffix is ""/"_chi"/"_mu" (slices.py:_axis_suffix).
KIND_LABELS = {
    "mosa_com": "Mosaicity COM",
    "mosa_fwhm": "Mosaicity FWHM",
    "strain": "Strain",
    "raw_sum": "Raw sum intensity",
    "raw_specific": "Raw frame",
    "raw_mosa_sum": "Raw mosa-sum intensity",
    "raw_mosa_specific": "Raw mosa frame",
}


def volume_label(volume_id: str) -> str:
    """Human label for a clim row, e.g. 'mosa_com_chi' -> 'Mosaicity COM (χ)'."""
    for comp, sym in (("_chi", "χ"), ("_mu", "μ")):
        if volume_id.endswith(comp):
            base = volume_id[: -len(comp)]
            return f"{KIND_LABELS.get(base, base)} ({sym})"
    return KIND_LABELS.get(volume_id, volume_id)


class ClimGroupSection(QWidget):
    """One vmin/vmax row per replot group; yields a per-group clim mapping."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._edits: dict[str, tuple[QLineEdit, QLineEdit]] = {}
        # Persistent text cache keyed by group — survives rebuilds (including an
        # empty intermediate ``set_groups([])``) so typed limits are never wiped.
        self._values: dict[str, tuple[str, str]] = {}

    def set_groups(self, groups: list[tuple[str, str]]) -> None:
        """Rebuild the rows for ``groups`` (``[(key, label), ...]``).

        Text already entered for a key is carried over via a persistent cache, so
        reloading the file (or reordering rows) does not wipe the user's limits —
        even across an empty intermediate rebuild.
        """
        # Fold whatever is currently displayed into the persistent cache first.
        for k, (vm, vx) in self._edits.items():
            self._values[k] = (vm.text(), vx.text())
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
            if key in self._values:
                vmin.setText(self._values[key][0])
                vmax.setText(self._values[key][1])
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

    def set_clim_by_group(self, mapping) -> None:
        """Fill the rows from a ``{key: (vmin, vmax)}`` mapping, blanking the rest.

        The inverse of :meth:`clim_by_group`, for a caller that stores its limits
        rather than collecting them fresh each time (the visualize stage's
        ``volume_clim_json`` param). ``None`` or a blank bound leaves that box
        empty, i.e. automatic. A key with no row is ignored: the value outlives
        any one set of groups, exactly as ``_values`` does.

        Blanking the rest is the half that makes this a *set* rather than an
        update, and it only shows on a **reused** section: the one caller today
        (``ClimTableEditor._on_edit``) builds a fresh one per dialog, whose rows
        are empty already. On a section seeded twice, leaving unlisted rows alone
        would show — and then collect — limits the mapping does not contain,
        which is exactly what the persistent ``_values`` cache makes possible.
        An entry that is not a two-element pair is not a limit and blanks its row
        with the others.
        """
        pairs = dict(mapping or {})
        for key, edits in self._edits.items():
            pair = pairs.get(key)
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                pair = (None, None)
            for edit, bound in zip(edits, pair):
                edit.setText("" if bound is None else str(bound))

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
