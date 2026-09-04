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


@dataclass(frozen=True)
class RoiProblem:
    """One thing wrong with one ROI parameter of a stage form.

    ``blocking`` is the distinction every consumer acts on:

    * **blocking** — the crop yields no pixels at all (an inverted pair, a
      start past the data, unparseable text). The product would be blank and
      the run is pure waste, so the GUI refuses to start it and ``run()``
      raises :class:`StageUserError` rather than writing an empty volume.
    * **advisory** — the crop yields fewer pixels than were asked for (an end
      past the data, which numpy silently clamps). That is a real surprise
      worth saying out loud, but the run still produces a usable product, so
      it must never stand between the user and their data.
    """

    param: str
    message: str
    blocking: bool


def _blocking_message(start: int, end: int, limit: int | None, label: str) -> str:
    """Why this start,end range yields no pixels at all, or ``""`` if it does.

    *limit* is the extent of the axis the range crops (``None`` when no data has
    been read yet). *label* prefixes the message on the four-int ``both`` axis,
    where one field carries two ranges and the reader needs to know which one is
    wrong; a single-axis field is named by the form row it sits on and takes an
    empty prefix.
    """
    if end <= start:
        return f"{label}need 0 <= start < end, got {start},{end} — this crops to nothing"
    if start < 0:
        # Not clamped: `apply_roi_3d` hands the pair straight to numpy, where a
        # negative start counts back from the far edge — a silently different
        # region, not an error.
        return f"{label}need 0 <= start < end, got {start},{end} — a negative start wraps around"
    if limit is not None and start >= limit:
        return f"{label}start {start} is past this data's {limit} px extent — nothing is left"
    return ""


def validate_roi_params(
    spec, params: dict, *, extent: tuple[int, int] | None = None, strict_end: bool = False
) -> tuple[RoiProblem, ...]:
    """Everything wrong with *params*' ROI fields, worst first. ``()`` = fine.

    Schema-driven: every :class:`~darq_xray.config.models.Param` declaring a
    ``roi_axis`` is a start,end range and is checked, which is why the axis is
    declared independently of ``roi_group`` (see that field). Params outside
    that marker — the ``*_darfix_origin_xy`` origins, for instance — are pairs
    but not ranges, and are left alone.

    *extent* is ``(height, width)`` of the data the ROI crops, or ``None`` when
    no estimate has resolved yet. Without it only the parse-level rules apply,
    which is the right degradation: they need no data and can never be wrong,
    so a form being filled in still gets the check that matters most.

    *strict_end* says what the stage does with an end past that extent. Four of
    the five ROI stages crop through :func:`~darq_xray.common.alignment.apply_roi_3d`
    and let numpy clamp it, so the run succeeds on a narrower region and the
    problem is advisory. ``strain`` does not: `strain.apply_roi` raises on
    ``r1 > rows`` — deliberately, because a stale window from a differently-sized
    dataset is exactly the misregistration it exists to catch — so for that stage
    the same input is **blocking**, and saying "the run crops at N" there would
    promise something the run then refuses.

    **Never raises.** It runs on every keystroke; a half-typed ``"105,"`` is the
    ordinary state of a form, and it is reported, not thrown.
    """
    height, width = extent if extent else (None, None)
    problems: list[RoiProblem] = []
    for p in getattr(spec, "params", ()):
        axis = getattr(p, "roi_axis", "")
        if not axis:
            continue
        text = params.get(p.name)
        if text is None or not str(text).strip():
            continue
        try:
            values = [int(s.strip()) for s in str(text).split(",")]
        except (ValueError, TypeError, AttributeError):
            problems.append(RoiProblem(p.name, f"cannot read {text!r} as whole numbers", True))
            continue
        wanted = 4 if axis == "both" else 2
        if len(values) != wanted:
            shape = "'r0,r1,c0,c1' (four integers)" if wanted == 4 else "'start,end' (two integers)"
            problems.append(RoiProblem(p.name, f"expected {shape}, got {text!r}", True))
            continue
        if axis == "both":
            ranges = [
                (values[0], values[1], height, "rows: "),
                (values[2], values[3], width, "columns: "),
            ]
        else:
            # One axis, one extent: `roi_axis` says which half of (height,
            # width) bounds it. Getting this pairing backwards would pass an
            # ROI that crops to nothing on any non-square data.
            ranges = [(values[0], values[1], height if axis == "y" else width, "")]
        for start, end, limit, label in ranges:
            message = _blocking_message(start, end, limit, label)
            if message:
                problems.append(RoiProblem(p.name, message, True))
            elif limit is not None and end > limit:
                # Reached only when the range is otherwise sound, so `start` is
                # inside the data and real pixels survive. Whether that is a
                # blocker depends on what the stage does next — see *strict_end*.
                outcome = (
                    "this stage refuses an ROI that overruns its map"
                    if strict_end
                    else f"the run crops at {limit}"
                )
                problems.append(
                    RoiProblem(
                        p.name,
                        f"{label}end {end} is past this data's {limit} px extent — {outcome}",
                        strict_end,
                    )
                )
    problems.sort(key=lambda q: not q.blocking)
    return tuple(problems)
