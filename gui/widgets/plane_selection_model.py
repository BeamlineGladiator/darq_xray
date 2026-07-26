"""Qt-free selection model behind the planes-first replot/pin dialogs.

Planes/layers are listed ONCE (union across volumes/quantity groups); the
filter box only narrows visibility; render selections are the cartesian
product of checked planes × checked quantities, with missing combinations
reported as skip reasons (never errors). No PySide6 imports — unit-testable
without a QApplication.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PlaneRow:
    """One selectable plane/layer row in the left panel."""

    key: object  # (slice_name, plane_idx) for slices; layer int for generic
    section: str  # section header ("" = flat list)
    number: int  # plane/layer number (integer-token filtering)
    offset: float | None  # µm offset / Z (decimal-token filtering); None = unknown
    label: str  # display text, e.g. "p118  -3.72 µm"
    marked: bool = False  # starred in /marks — cosmetic; filtering-only


def build_slice_rows(entries, marks=None) -> list[PlaneRow]:
    """Rows from slices ReplotEntry list — one per (slice_name, plane_idx).

    *marks* is the ``read_marks`` mapping (slice_name -> [offset_um, ...]);
    each marked offset stars its nearest stored plane's row.
    """
    marked_idx: dict[str, set[int]] = {}
    if marks:
        stored_by_slice: dict[str, list[float]] = {}
        for e in entries:
            stored_by_slice.setdefault(e.slice_name, list(e.offsets_um))
        for sname, offs in marks.items():
            stored = stored_by_slice.get(sname)
            if not stored:
                continue
            marked_idx[sname] = {
                min(range(len(stored)), key=lambda i: abs(stored[i] - o)) for o in offs
            }
    seen: dict[tuple[str, int], PlaneRow] = {}
    for e in entries:
        for k, off in enumerate(e.offsets_um):
            key = (e.slice_name, k)
            if key not in seen:
                is_marked = k in marked_idx.get(e.slice_name, ())
                seen[key] = PlaneRow(
                    key=key,
                    section=e.slice_name,
                    number=k,
                    offset=float(off),
                    label=("★ " if is_marked else "") + f"p{k:03d}  {off:+.2f} µm",
                    marked=is_marked,
                )
    return list(seen.values())


_Z_IN_LABEL = re.compile(r"Z=([-+]?\d+(?:\.\d+)?)")


def build_layer_rows(groups) -> list[PlaneRow]:
    """Rows from generic ReplotGroup list — one per layer index, union across groups."""
    rows: dict[int, PlaneRow] = {}
    for g in groups:
        for z, lab in enumerate(g.item_labels):
            if z in rows:
                continue
            m = _Z_IN_LABEL.search(lab)
            rows[z] = PlaneRow(
                key=z,
                section="",
                number=z,
                offset=float(m.group(1)) if m else None,
                label=lab,
            )
    return [rows[z] for z in sorted(rows)]


def parse_tokens(text: str) -> list[tuple[str, float]]:
    """Classify comma tokens: bare unsigned int -> number; signed/decimal -> offset.

    Unparseable tokens become ("invalid", 0.0) — they match nothing (so a
    nonsense filter shows an empty list, not the full one).
    """
    out: list[tuple[str, float]] = []
    for tok in (t.strip() for t in (text or "").split(",")):
        if not tok:
            continue
        if re.fullmatch(r"\d+", tok):
            out.append(("number", float(tok)))
        else:
            try:
                out.append(("offset", float(tok)))
            except ValueError:
                out.append(("invalid", 0.0))
    return out


def _half_step(offsets: list[float]) -> float:
    if len(offsets) < 2:
        return float("inf")  # single plane: nearest matches unconditionally
    s = sorted(offsets)
    diffs = sorted(b - a for a, b in zip(s, s[1:]) if b > a)
    return diffs[len(diffs) // 2] / 2.0 if diffs else float("inf")


def filter_rows(rows: list[PlaneRow], text: str, *, marked_only: bool = False) -> list[PlaneRow]:
    """Rows visible under *text*. Blank -> all. Narrows only — never selects."""
    if marked_only:
        rows = [r for r in rows if r.marked]
    toks = parse_tokens(text)
    if not toks:
        return list(rows)
    by_section: dict[str, list[PlaneRow]] = {}
    for r in rows:
        by_section.setdefault(r.section, []).append(r)
    matched: set = set()
    for kind, v in toks:
        if kind == "number":
            matched.update(r.key for r in rows if r.number == int(v))
        elif kind == "offset":
            for sec_rows in by_section.values():
                with_off = [r for r in sec_rows if r.offset is not None]
                if not with_off:
                    continue
                tol = _half_step([r.offset for r in with_off])
                best = min(with_off, key=lambda r: abs(r.offset - v))
                if abs(best.offset - v) <= tol:
                    matched.add(best.key)
    return [r for r in rows if r.key in matched]


def slice_selections(entries, checked_plane_keys, checked_vids):
    """Checked planes × checked volume ids -> (selections, skip reasons).

    Selections match ``dfxm.stages.slices.render_replot``:
    ``[(volume_id, slice_name, [plane_idx, ...]), ...]``.
    """
    sels, skipped = [], []
    by_vid_slice = {(e.volume_id, e.slice_name): e for e in entries}
    wanted: dict[str, list[int]] = {}
    for sname, idx in checked_plane_keys:
        wanted.setdefault(sname, []).append(int(idx))
    for vid in checked_vids:
        for sname, idxs in wanted.items():
            e = by_vid_slice.get((vid, sname))
            if e is None:
                skipped.append(f"{vid}/{sname}: volume has no such slice group")
                continue
            ok = sorted(i for i in idxs if 0 <= i < e.n_planes)
            missing = sorted(set(idxs) - set(ok))
            if missing:
                skipped.append(f"{vid}/{sname}: no plane(s) {missing}")
            if ok:
                sels.append((vid, sname, ok))
    return sels, skipped


def layer_selections(groups, checked_layers, checked_keys):
    """Checked layers × checked quantity groups -> (selections, skip reasons).

    Selections match the generic ``render_replot``: ``[(group_key, [z, ...]), ...]``.
    """
    sels, skipped = [], []
    n_by_key = {g.key: len(g.item_labels) for g in groups}
    layers = sorted(int(z) for z in checked_layers)
    for key in checked_keys:
        n = n_by_key.get(key, 0)
        ok = [z for z in layers if 0 <= z < n]
        missing = [z for z in layers if z >= n]
        if missing:
            skipped.append(f"{key}: no layer(s) {missing}")
        if ok:
            sels.append((key, ok))
    return sels, skipped
