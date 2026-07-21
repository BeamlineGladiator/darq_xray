"""Darfix-window / map-frame ROI conversions and validation.

Two regions of interest describe every DFXM dataset:

* The **darfix window** — the detector crop darfix used when fitting the maps,
  displayed by darfix as *origin + size* ``x,y,w,h``. A fact about how the maps
  were made: map pixel (0, 0) sits at detector pixel ``(x, y)``.
* The **analysis window** — the sub-region chosen for study, expressed in
  *map-frame* start,end pairs (columns ``c0,c1``, rows ``r0,r1``).

Stages consume these in different frames: rocking crops raw detector frames
(absolute pixels), the map stages crop darfix maps (map pixels). The
converters here are the single place the frames meet:
``detector = darfix_origin + map``. Pure functions — no Qt, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DarfixWindow:
    """The darfix detector crop, as origin + size (what darfix displays)."""

    origin_x: int
    origin_y: int
    width: int
    height: int

    @property
    def x0(self) -> int:
        return self.origin_x

    @property
    def x1(self) -> int:
        return self.origin_x + self.width

    @property
    def y0(self) -> int:
        return self.origin_y

    @property
    def y1(self) -> int:
        return self.origin_y + self.height


def parse_pair(text: str | None) -> tuple[int, int] | None:
    """'start,end' -> (start, end); blank/None -> None; malformed -> ValueError."""
    if text is None or not str(text).strip():
        return None
    parts = [s.strip() for s in str(text).split(",")]
    if len(parts) != 2:
        raise ValueError(f"expected 'start,end' (two integers), got {text!r}")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError(f"expected 'start,end' (two integers), got {text!r}") from exc


def parse_darfix_roi(text: str | None) -> DarfixWindow | None:
    """'x,y,w,h' (origin+size, darfix's display) -> DarfixWindow; blank -> None."""
    if text is None or not str(text).strip():
        return None
    parts = [s.strip() for s in str(text).split(",")]
    if len(parts) != 4:
        raise ValueError(f"expected 'x,y,w,h' (four integers, origin+size), got {text!r}")
    try:
        x, y, w, h = (int(p) for p in parts)
    except ValueError as exc:
        raise ValueError(f"expected 'x,y,w,h' (four integers, origin+size), got {text!r}") from exc
    return DarfixWindow(x, y, w, h)


def map_to_detector(pair: tuple[int, int], origin: int) -> tuple[int, int]:
    """Map-frame start,end -> absolute detector pixels along one axis."""
    return pair[0] + origin, pair[1] + origin


def detector_to_map(pair: tuple[int, int], origin: int) -> tuple[int, int]:
    """Absolute detector start,end -> map-frame pixels along one axis."""
    return pair[0] - origin, pair[1] - origin


def format_pair(pair: tuple[int, int]) -> str:
    return f"{pair[0]},{pair[1]}"


def analysis_detector_window(
    darfix_roi: str, analysis_roi_x: str, analysis_roi_y: str
) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    """The analysis window in absolute detector pixels (what rocking crops).

    A blank analysis axis falls back to the full darfix window; without a
    darfix window nothing is derivable -> (None, None), and in that case the
    analysis strings are never parsed (so a malformed analysis axis alongside
    a blank darfix window does not raise). Once a darfix window is present,
    malformed analysis input raises ValueError (use :func:`validate_rois` for
    user-facing messages).
    """
    win = parse_darfix_roi(darfix_roi)
    if win is None:
        return None, None
    ax = parse_pair(analysis_roi_x)
    ay = parse_pair(analysis_roi_y)
    det_x = map_to_detector(ax, win.origin_x) if ax else (win.x0, win.x1)
    det_y = map_to_detector(ay, win.origin_y) if ay else (win.y0, win.y1)
    return det_x, det_y


def validate_rois(darfix_roi: str, analysis_roi_x: str, analysis_roi_y: str) -> list[str]:
    """Human-readable problems with the experiment ROI fields ([] = all fine)."""
    problems: list[str] = []
    win = None
    try:
        win = parse_darfix_roi(darfix_roi)
    except ValueError as exc:
        problems.append(f"Darfix ROI: {exc}")
    else:
        if win is not None and (win.width <= 0 or win.height <= 0):
            problems.append("Darfix ROI: width and height must be positive (it is origin+size)")
            win = None
    for label, text, size in (
        ("Analysis window X", analysis_roi_x, win.width if win else None),
        ("Analysis window Y", analysis_roi_y, win.height if win else None),
    ):
        try:
            pair = parse_pair(text)
        except ValueError as exc:
            problems.append(f"{label}: {exc}")
            continue
        if pair is None:
            continue
        start, end = pair
        if start < 0 or end <= start:
            problems.append(f"{label}: need 0 <= start < end, got {start},{end}")
        elif size is not None and end > size:
            problems.append(
                f"{label}: end {end} exceeds the darfix window size {size} "
                "(analysis windows are map-frame, relative to the darfix window)"
            )
    return problems
