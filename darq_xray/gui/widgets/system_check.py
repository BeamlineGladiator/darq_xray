"""What this machine is, and what it implies for settings.

The only surface that pays for a GL probe on demand. Renders a
:class:`~darq_xray.common.machine.MachineProfile`; decides nothing.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from darq_xray.common import advice, machine
from darq_xray.common.advice import human_bytes

from ..advisor import GL_PROBE_TIMEOUT_S, clear_profile_cache
from .busy import busy_cursor

_UNKNOWN = "unknown"


class SystemCheckDialog(QDialog):
    """A probe table: measured value and what it means, one row per probe."""

    def __init__(self, parent=None, profile=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("System check")
        self.resize(760, 420)
        self._profile = profile if profile is not None else self._measure()

        self._table = QTableWidget(0, 3, self)
        self._table.setHorizontalHeaderLabels(["Probe", "Measured", "What it means"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        self._errors = QLabel("")
        self._errors.setWordWrap(True)
        self._errors.setProperty("role", "warning")

        self._reprobe_btn = QPushButton("Re-probe")
        self._reprobe_btn.clicked.connect(self._on_reprobe)
        self._copy_btn = QPushButton("Copy as text")
        self._copy_btn.clicked.connect(self._on_copy)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self._reprobe_btn)
        btn_row.addWidget(self._copy_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(buttons)

        layout = QVBoxLayout(self)
        layout.addWidget(self._table, 1)
        layout.addWidget(self._errors)
        layout.addLayout(btn_row)
        self._rebuild()

    # -- probing ----------------------------------------------------------
    @staticmethod
    def _measure(*, use_cache: bool = True):
        """Measure this machine, GL included. The one place that pays for it."""
        with busy_cursor("Probing this machine…"):
            if not use_cache:
                # A lone `probe_gl(use_cache=False)` call is not enough here:
                # its fresh result is never written back, so the very next
                # `probe_gl(use_cache=True)` call below (the one `profile`
                # makes) would just hit the still-stale memo/disk cache and
                # redisplay the old answer. Invalidate both first.
                machine.invalidate_gl_cache()
                clear_profile_cache()
            return machine.profile(probe_gl_now=True, gl_timeout=GL_PROBE_TIMEOUT_S)

    def _on_reprobe(self) -> None:
        self._profile = self._measure(use_cache=False)
        self._rebuild()

    def _on_copy(self) -> None:
        QApplication.clipboard().setText(self.as_text())

    # -- content ----------------------------------------------------------
    def rows(self) -> list[tuple[str, str, str]]:
        """(label, measured value, what it means) for every probe."""
        p = self._profile
        out: list[tuple[str, str, str]] = [
            (
                "CPU",
                f"{p.cpu_logical} logical / {p.cpu_physical or _UNKNOWN} physical",
                "Measured only — the pipeline does not parallelise yet.",
            ),
            (
                "RAM",
                f"{human_bytes(p.ram_available)} available of {human_bytes(p.ram_total)}"
                if p.ram_total
                else _UNKNOWN,
                "Stages stream when a run does not fit; slower, same result.",
            ),
            (
                "Headroom",
                human_bytes(advice.headroom_bytes(p)),
                "The most a run will plan to use, leaving room for Qt and the OS.",
            ),
            (
                "Disk",
                f"{human_bytes(p.disk_free)} free" if p.disk_free else _UNKNOWN,
                "Free space on the filesystem the app was launched from, not a stage's "
                "output directory — this dialog measures no output_dir. See a stage's "
                "own cost line for the figure that actually gates that stage's run.",
            ),
        ]
        # `gl_status` is the source of truth for whether the GL probe actually
        # succeeded THIS time — `gl` alone is not enough: a synthetic or stale
        # profile can carry a `GLInfo` object even when the status says the
        # probe crashed, and a failed probe must read as unknown, never as the
        # leftover data from a different measurement.
        if p.gl_status == "ok" and p.gl is not None:
            cap = p.gl.max_3d_texture or _UNKNOWN
            out.append(
                (
                    "OpenGL",
                    f"{p.gl.renderer} · 3-D texture cap {cap} px",
                    "Software renderer — prefer surface mode; volume mode renders "
                    "blank past the cap."
                    if p.gl.software
                    else "Hardware accelerated.",
                )
            )
        else:
            out.append(
                (
                    "OpenGL",
                    f"{_UNKNOWN} ({p.gl_status})",
                    "3-D products may render blank; re-probe, or use surface mode.",
                )
            )
        out.append(
            (
                "ffmpeg",
                p.ffmpeg or "not found",
                "MP4 export needs it; without it exports fall back to GIF.",
            )
        )
        return out

    def as_text(self) -> str:
        """Plain text for the clipboard — no markup, safe to paste anywhere."""
        lines = [f"{label}: {value} — {implication}" for label, value, implication in self.rows()]
        lines.extend(self._profile.probe_errors)
        return "\n".join(lines)

    def _rebuild(self) -> None:
        rows = self.rows()
        self._table.setRowCount(len(rows))
        for r, (label, value, implication) in enumerate(rows):
            for c, text in enumerate((label, value, implication)):
                self._table.setItem(r, c, QTableWidgetItem(text))
        self._table.resizeColumnsToContents()
        errors = self._profile.probe_errors
        self._errors.setText("\n".join(errors))
        self._errors.setVisible(bool(errors))
