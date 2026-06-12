"""User-facing stage errors (Qt-free).

:class:`StageUserError` marks a failure caused by the stage's *inputs*
rather than a bug: the message says what is wrong, ``hint`` says what the
user should do about it. It subclasses :class:`ValueError` so existing
callers (and tests) that treat input validation as ValueError keep working.
The runner forwards ``hint`` to the GUI via ``Failed.hint``.
"""

from __future__ import annotations


class StageUserError(ValueError):
    """An input problem the user can fix, with an actionable hint."""

    def __init__(self, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.hint = hint
