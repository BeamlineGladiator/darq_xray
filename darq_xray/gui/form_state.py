"""Persist per-experiment stage-form values across restarts (Qt-side).

Sibling to :mod:`darq_xray.gui.window_state`; both use the app-wide ``QSettings`` (org
``darq_xray``, app ``pipeline``). Each stage's :meth:`~darq_xray.gui.widgets.param_form.ParamForm.values`
dict is stored under ``formState/<slug(experiment)>/<stage>`` as a JSON string —
JSON keeps value types stable across QSettings backends (an INI backend would
otherwise stringify ints/bools), and ``ParamForm.set_values`` + ``Param.coerce``
re-hydrate on load.

Calibration-flagged params are never handed to this store (the caller strips
them): those are physically tied to the experiment and always re-pulled from it,
so a stale saved value can't silently override an updated experiment.
"""

from __future__ import annotations

import json
import re
from typing import Any

from PySide6.QtCore import QSettings

_PREFIX = "formState"


def _slug(name: str) -> str:
    """Make an experiment name safe as a QSettings key path segment.

    ``QSettings`` treats ``/`` as a group separator; collapse anything that
    isn't alphanumeric/dash/underscore to ``_`` so odd names can't create stray
    nested groups. Empty names fall back to ``default``. Names that differ only
    in the collapsed characters (``a/b`` vs ``a_b``) share a slug — harmless for
    the simple preset names this sees, but a theoretical collision.
    """
    slug = re.sub(r"[^0-9A-Za-z_-]+", "_", (name or "").strip())
    return slug or "default"


class FormStateStore:
    """Save/restore per-experiment, per-stage form values via QSettings."""

    def __init__(self, settings: QSettings | None = None) -> None:
        self._settings = settings or QSettings()

    def _key(self, experiment: str, stage: str) -> str:
        return f"{_PREFIX}/{_slug(experiment)}/{stage}"

    def load(self, experiment: str, stage: str) -> dict[str, Any] | None:
        """Return the saved values for (*experiment*, *stage*), or ``None``.

        ``None`` on a missing entry or unreadable/foreign payload — the caller
        then falls back to experiment defaults, mirroring ``WindowState``'s
        defensive restore.
        """
        raw = self._settings.value(self._key(experiment, stage))
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except (TypeError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def save(self, experiment: str, stage: str, values: dict[str, Any]) -> None:
        """Persist *values* (a JSON-serialisable dict) for (*experiment*, *stage*)."""
        self._settings.setValue(self._key(experiment, stage), json.dumps(values))
