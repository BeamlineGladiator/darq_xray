"""Run a stage in a child process, streaming progress/log over a queue.

Stages are pure ``run(params, progress=None) -> result`` functions. The GUI
must not call them in-thread: matplotlib (and the heavier 3-D stages) are not
thread-safe, and a long stage would freeze the event loop with no way to
cancel. :class:`StageRunner` runs the stage in a separate process and surfaces
four message kinds back to the parent:

* :class:`Progress` — ``(fraction, text)`` from the stage's progress callback,
* :class:`Log` — a line the stage printed to stdout/stderr,
* :class:`Done` — the (picklable) return value, on success,
* :class:`Failed` — the error message + traceback, on exception.

This module is UI-agnostic: the GUI polls :meth:`StageRunner.poll` from a
timer, while CLI/tests use :meth:`StageRunner.run_blocking`.
"""

from __future__ import annotations

import multiprocessing as mp
import queue as _queue
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .stages.registry import resolve


# -----------------------------------------------------------------------------
# Messages (all picklable, defined at module level so spawn can ship them)
# -----------------------------------------------------------------------------
@dataclass
class Progress:
    frac: float
    text: str


@dataclass
class Log:
    text: str


@dataclass
class Done:
    result: Any


@dataclass
class Failed:
    error: str
    traceback: str
    hint: str = ""  # actionable advice from StageUserError, "" otherwise


Message = Progress | Log | Done | Failed


# -----------------------------------------------------------------------------
# Child side
# -----------------------------------------------------------------------------
class _QueueWriter:
    """Minimal stdout/stderr shim: emit one :class:`Log` per completed line."""

    def __init__(self, q: "mp.Queue") -> None:
        self._q = q
        self._buf = ""

    def write(self, s: str) -> int:
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._q.put(Log(line))
        return len(s)

    def flush(self) -> None:
        if self._buf:
            self._q.put(Log(self._buf))
            self._buf = ""


def _worker(q: "mp.Queue", target: str, params: dict) -> None:
    """Child entry point: run *target* and report results over *q*."""
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = _QueueWriter(q)  # capture stage prints as Log lines
    try:
        fn = resolve(target)

        def progress(frac: float, msg: str = "") -> None:
            q.put(Progress(float(frac), str(msg)))

        result = fn(params, progress=progress)
        sys.stdout.flush()
        q.put(Done(result))
    except Exception as exc:  # noqa: BLE001 - surfaced to the parent as Failed
        try:
            sys.stdout.flush()
        except Exception:  # noqa: BLE001
            pass
        q.put(Failed(str(exc), traceback.format_exc(), str(getattr(exc, "hint", "") or "")))
    finally:
        sys.stdout, sys.stderr = old_out, old_err


# -----------------------------------------------------------------------------
# Parent side
# -----------------------------------------------------------------------------
class StageRunner:
    """Drive a stage in a child process and stream its messages back.

    *target* is a ``"module:function"`` string (resolved in the child, so the
    function need not be picklable) or a callable. Use a registry target string
    when running under the ``spawn`` start method.
    """

    def __init__(
        self,
        target: str | Callable,
        params: dict,
        *,
        start_method: str = "spawn",
    ) -> None:
        if callable(target) and start_method == "spawn":
            # spawn re-imports in the child and cannot pickle arbitrary
            # callables/closures reliably; require a dotted target instead.
            raise TypeError("with start_method='spawn', pass a 'module:function' target string")
        self._target = target
        self._params = params
        self._ctx = mp.get_context(start_method)
        self._q: mp.Queue | None = None
        self._proc: mp.Process | None = None
        self._result: Any = None
        self._failure: Failed | None = None
        self._finished = False

    # -- lifecycle --------------------------------------------------------
    def start(self) -> None:
        self._q = self._ctx.Queue()
        self._proc = self._ctx.Process(
            target=_worker, args=(self._q, self._target, self._params), daemon=True
        )
        self._proc.start()

    def poll(self) -> list[Message]:
        """Drain all currently-queued messages, updating result/failure state."""
        msgs: list[Message] = []
        if self._q is None:
            return msgs
        while True:
            try:
                m = self._q.get_nowait()
            except _queue.Empty:
                break
            if isinstance(m, Done):
                self._result = m.result
                self._finished = True
            elif isinstance(m, Failed):
                self._failure = m
                self._finished = True
            msgs.append(m)
        return msgs

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def cancel(self, timeout: float = 2.0) -> None:
        """Terminate the child; escalate to kill if it ignores SIGTERM.

        Sets ``finished`` even though no Done/Failed arrived, so ``finished``
        means "this run is over", not "this run produced something". A caller
        that cancels and then asks whether the stage delivered must look at
        ``result``/``failure``, not at ``finished``.
        """
        if self._proc and self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout)
            if self._proc.is_alive():
                self._proc.kill()
                self._proc.join(timeout)
        self._finished = True

    def join(self, timeout: float | None = None) -> None:
        if self._proc:
            self._proc.join(timeout)

    # -- state ------------------------------------------------------------
    @property
    def finished(self) -> bool:
        """True once a Done/Failed was received (or the run was cancelled)."""
        return self._finished

    @property
    def pid(self) -> int | None:
        """The child's PID once started, else None. Lets a caller watch its memory.

        Stays set after the child exits (that is ``mp.Process.pid``'s own
        behaviour), so a sampler can drain the queue after the process is gone
        without losing the identity of what it was watching.
        """
        return self._proc.pid if self._proc is not None else None

    @property
    def result(self) -> Any:
        return self._result

    @property
    def failure(self) -> Failed | None:
        return self._failure

    # -- convenience ------------------------------------------------------
    def run_blocking(
        self, on_message: Callable[[Message], None] | None = None, poll_interval: float = 0.02
    ) -> Any:
        """Start, stream messages to *on_message*, and block until completion.

        Returns the stage result, or raises :class:`RuntimeError` if the child
        failed or exited without reporting a result.
        """
        self.start()
        while True:
            for m in self.poll():
                if on_message:
                    on_message(m)
            if not self.is_alive():
                # process ended — drain anything still queued
                for m in self.poll():
                    if on_message:
                        on_message(m)
                break
            time.sleep(poll_interval)

        if self._failure is not None:
            raise RuntimeError(f"stage failed: {self._failure.error}\n{self._failure.traceback}")
        if not self._finished:
            raise RuntimeError("worker exited without producing a result")
        return self._result
