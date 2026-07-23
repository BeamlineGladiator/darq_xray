"""Per-figure publication export dialog: live preview + style controls."""

from __future__ import annotations

import math
import os
import re
from dataclasses import replace

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from dfxm.common.figures import FigureSpec
from dfxm.common.plotting import AXES_MODES, CMAP_CHOICES, PlotStyle, fixed_scale

from .mpl_canvas import MplCanvas


def _own_trace_scale(s: PlotStyle) -> float | None:
    """The trace-scale FIELD's own validated value, for display.

    Deliberately NOT ``plotting.trace_fixed_scale`` — that falls back to the
    map scale, which must render as a blank (inheriting) trace field here.
    """
    v = getattr(s, "trace_scale_um_per_cm", None)
    if v is None or v == "":
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if (v > 0 and math.isfinite(v)) else None


# Module-level constants for option lists — shared by StyleControls so the
# two places can never drift apart.
_COLORS = ["black", "white", "red", "green", "blue", "yellow", "grey"]
_CMAPS = list(CMAP_CHOICES)
_WIDTHS = ["auto", "single", "double"]
_TICK_FMTS = ["auto", "scientific", "arb", "0", "1", "2", "3"]
_TICK_FMT_LABELS = {
    "auto": "auto (matplotlib default)",
    "scientific": "scientific (×10ⁿ offset)",
    "arb": "arbitrary units (no ticks)",
    "0": "0 decimals (plain numbers)",
    "1": "1 decimal (plain numbers)",
    "2": "2 decimals (plain numbers)",
    "3": "3 decimals (plain numbers)",
}
_OFFSET_POS = ["top", "bottom"]
# Display labels for plotting.AXES_MODES (values stay canonical in the core).
_AXES_MODE_LABELS = {"full": "Full", "no_frame": "No frame", "none": "None"}
# (group field-suffix, friendly label) — drives the per-group colourbar rows.
_CBAR_GROUPS = (
    ("mosa_com", "Mosa misorientation"),
    ("mosa_fwhm", "Mosa FWHM"),
    ("strain", "Strain"),
    ("raw", "Raw intensity"),
)
_LOCS = ["lower right", "lower left", "upper right", "upper left"]


class StyleControls(QWidget):
    """Widget encapsulating all ~21 PlotStyle controls.

    Holds a reference to the supplied *style* object and mutates it in place
    whenever a control changes.  Emits :attr:`changed` after each mutation so
    callers can connect a debounced re-render or any other side-effect.

    Call :meth:`sync_from_style` to push the style's current values back into
    all widgets (e.g. after a "Reset" that replaces the style's fields).
    """

    changed = Signal()

    def __init__(self, style: PlotStyle, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._style = style
        self._build_controls()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sync_from_style(self) -> None:
        """Re-push all style values into the widgets (used after reset).

        Signals are blocked during the sync so no spurious ``changed`` emissions
        occur while we programmatically set widget values.
        """
        s = self._style
        # Block signals on every widget to prevent feedback loops.
        for w in self._all_widgets():
            w.blockSignals(True)

        for combo, field_name in (
            (self._w_cmap_mosa_com, "cmap_mosa_com"),
            (self._w_cmap_mosa_fwhm, "cmap_mosa_fwhm"),
            (self._w_cmap_strain, "cmap_strain"),
            (self._w_cmap_raw, "cmap_raw"),
        ):
            val = getattr(s, field_name)
            combo.setCurrentText(val if val in _CMAPS else _CMAPS[0])
        self._w_scale_bar.setChecked(s.scale_bar)
        self._w_bar_auto.setChecked(s.scale_bar_length_um is None)
        self._w_bar_len.setValue(
            s.scale_bar_length_um if s.scale_bar_length_um is not None else 10.0
        )
        self._w_bar_len.setEnabled(s.scale_bar_length_um is not None)
        self._w_bar_thick.setValue(s.scale_bar_thickness_pt)
        self._w_bar_label_scale.setValue(s.scale_bar_label_scale)
        self._w_bar_color.setCurrentText(
            s.scale_bar_color if s.scale_bar_color in _COLORS else _COLORS[0]
        )
        self._w_bar_loc.setCurrentText(s.scale_bar_loc)
        self._w_bar_inset.setValue(s.scale_bar_inset_pt)
        self._w_bar_box.setChecked(s.scale_bar_box)
        self._w_box_color.setCurrentText(
            s.scale_bar_box_color if s.scale_bar_box_color in _COLORS else _COLORS[0]
        )
        self._w_box_alpha.setValue(s.scale_bar_box_alpha)
        self._w_box_margin.setValue(s.scale_bar_box_margin_pt)
        self._w_font_scale.setValue(s.font_scale)
        self._w_title_scale.setValue(s.title_scale)
        self._w_show_title.setChecked(s.show_title)
        self._w_center_labels.setChecked(s.center_axis_labels)
        self._w_axes_mode.setCurrentIndex(
            self._w_axes_mode.findData(s.axes_mode if s.axes_mode in AXES_MODES else "full")
        )
        self._w_colorbar.setChecked(s.colorbar)
        self._w_cbar_label.setText(s.colorbar_label or "")
        self._w_cbar_frac.setValue(s.colorbar_fraction)
        self._w_cbar_ticks.setValue(s.colorbar_ticks)
        for grp, _label in _CBAR_GROUPS:
            cur = getattr(s, f"tickfmt_{grp}")
            self._w_tickfmt[grp].setCurrentIndex(
                _TICK_FMTS.index(cur if cur in _TICK_FMTS else "auto")
            )
            self._w_offscale[grp].setValue(getattr(s, f"offset_scale_{grp}"))
            self._w_offpos[grp].setCurrentText(getattr(s, f"offset_pos_{grp}"))
        self._w_round_clim.setChecked(s.round_clim)
        self._w_fig_width.setCurrentText(
            s.figure_width
            if isinstance(s.figure_width, str) and s.figure_width in _WIDTHS
            else "auto"
        )
        _sv = fixed_scale(s)
        self._w_scale_umcm.setText(f"{_sv:g}" if _sv is not None else "")
        _tsv = _own_trace_scale(s)
        self._w_trace_scale_umcm.setText(f"{_tsv:g}" if _tsv is not None else "")
        _thv = getattr(s, "trace_height_cm", None)
        self._w_trace_height_cm.setText(f"{_thv:g}" if _thv is not None else "")
        for cb, name in zip(self._format_checkboxes, self._format_names):
            cb.setChecked(name in s.formats)
        self._w_dpi.setValue(s.dpi)

        for w in self._all_widgets():
            w.blockSignals(False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def set_style(self, style: PlotStyle) -> None:
        """Rebind to a new PlotStyle object and refresh all widgets from it."""
        self._style = style
        self.sync_from_style()

    def _all_widgets(self) -> list[QWidget]:
        """Return a flat list of all leaf widgets (for blockSignals)."""
        widgets = [
            self._w_cmap_mosa_com,
            self._w_cmap_mosa_fwhm,
            self._w_cmap_strain,
            self._w_cmap_raw,
            self._w_scale_bar,
            self._w_bar_auto,
            self._w_bar_len,
            self._w_bar_thick,
            self._w_bar_label_scale,
            self._w_bar_color,
            self._w_bar_loc,
            self._w_bar_inset,
            self._w_bar_box,
            self._w_box_color,
            self._w_box_alpha,
            self._w_box_margin,
            self._w_font_scale,
            self._w_title_scale,
            self._w_show_title,
            self._w_center_labels,
            self._w_axes_mode,
            self._w_colorbar,
            self._w_cbar_label,
            self._w_cbar_frac,
            self._w_cbar_ticks,
            self._w_round_clim,
            self._w_fig_width,
            self._w_scale_umcm,
            self._w_trace_scale_umcm,
            self._w_trace_height_cm,
            self._w_fmt_png,
            self._w_fmt_pdf,
            self._w_fmt_svg,
            self._w_dpi,
        ]
        for grp, _label in _CBAR_GROUPS:
            widgets += [self._w_tickfmt[grp], self._w_offscale[grp], self._w_offpos[grp]]
        return widgets

    def _emit(self) -> None:
        self.changed.emit()

    def _build_controls(self) -> None:
        """Build the full control set; each widget mutates self._style + emits changed."""
        form = QFormLayout(self)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        s = self._style  # alias (widgets capture self, so mutations always go to self._style)

        # --- Colormaps section (one dropdown per quantity group) ---
        form.addRow(QLabel("<b>Colormaps</b>"))
        cmap_rows = (
            ("_w_cmap_mosa_com", "cmap_mosa_com", "Mosa misorientation"),
            ("_w_cmap_mosa_fwhm", "cmap_mosa_fwhm", "Mosa FWHM"),
            ("_w_cmap_strain", "cmap_strain", "Strain"),
            ("_w_cmap_raw", "cmap_raw", "Raw intensity"),
        )
        for attr, field_name, label in cmap_rows:
            combo = QComboBox()
            combo.addItems(_CMAPS)
            current = getattr(s, field_name)
            combo.setCurrentText(current if current in _CMAPS else _CMAPS[0])
            combo.currentTextChanged.connect(
                lambda v, f=field_name: (setattr(self._style, f, v), self._emit())
            )
            setattr(self, attr, combo)
            form.addRow(label, combo)

        # --- Scale bar section ---
        form.addRow(QLabel("<b>Scale bar</b>"))

        self._w_scale_bar = QCheckBox()
        self._w_scale_bar.setChecked(s.scale_bar)
        self._w_scale_bar.toggled.connect(
            lambda v: (setattr(self._style, "scale_bar", v), self._emit())
        )
        form.addRow("Show scale bar", self._w_scale_bar)

        # Auto-length: checked means None (auto); unchecked enables the spin.
        self._w_bar_auto = QCheckBox("Auto length")
        self._w_bar_auto.setChecked(s.scale_bar_length_um is None)
        self._w_bar_len = QDoubleSpinBox()
        self._w_bar_len.setRange(0.1, 10_000.0)
        self._w_bar_len.setDecimals(1)
        self._w_bar_len.setSuffix(" µm")
        self._w_bar_len.setValue(
            s.scale_bar_length_um if s.scale_bar_length_um is not None else 10.0
        )
        self._w_bar_len.setEnabled(s.scale_bar_length_um is not None)
        self._w_bar_auto.toggled.connect(self._on_bar_auto_toggled)
        self._w_bar_len.valueChanged.connect(
            lambda v: (
                setattr(self._style, "scale_bar_length_um", v),
                self._emit(),
            )
        )
        bar_len_layout = QHBoxLayout()
        bar_len_layout.addWidget(self._w_bar_auto)
        bar_len_layout.addWidget(self._w_bar_len)
        form.addRow("Bar length", bar_len_layout)

        self._w_bar_thick = QDoubleSpinBox()
        self._w_bar_thick.setRange(0.5, 20.0)
        self._w_bar_thick.setDecimals(1)
        self._w_bar_thick.setSuffix(" pt")
        self._w_bar_thick.setValue(s.scale_bar_thickness_pt)
        self._w_bar_thick.valueChanged.connect(
            lambda v: (setattr(self._style, "scale_bar_thickness_pt", v), self._emit())
        )
        form.addRow("Bar thickness", self._w_bar_thick)

        self._w_bar_label_scale = QDoubleSpinBox()
        self._w_bar_label_scale.setRange(0.5, 3.0)
        self._w_bar_label_scale.setDecimals(2)
        self._w_bar_label_scale.setSingleStep(0.1)
        self._w_bar_label_scale.setValue(s.scale_bar_label_scale)
        self._w_bar_label_scale.valueChanged.connect(
            lambda v: (setattr(self._style, "scale_bar_label_scale", v), self._emit())
        )
        form.addRow("Label scale", self._w_bar_label_scale)

        self._w_bar_loc = QComboBox()
        self._w_bar_loc.addItems(_LOCS)
        self._w_bar_loc.setCurrentText(s.scale_bar_loc)
        self._w_bar_loc.currentTextChanged.connect(
            lambda v: (setattr(self._style, "scale_bar_loc", v), self._emit())
        )
        form.addRow("Bar location", self._w_bar_loc)

        self._w_bar_inset = QDoubleSpinBox()
        self._w_bar_inset.setRange(0.0, 100.0)
        self._w_bar_inset.setDecimals(1)
        self._w_bar_inset.setSuffix(" pt")
        self._w_bar_inset.setToolTip(
            "Distance of the scale bar from the axes corner, in printed points (0 = flush)"
        )
        self._w_bar_inset.setValue(s.scale_bar_inset_pt)
        self._w_bar_inset.valueChanged.connect(
            lambda v: (setattr(self._style, "scale_bar_inset_pt", v), self._emit())
        )
        form.addRow("Edge inset", self._w_bar_inset)

        self._w_bar_color = QComboBox()
        self._w_bar_color.addItems(_COLORS)
        self._w_bar_color.setCurrentText(
            s.scale_bar_color if s.scale_bar_color in _COLORS else _COLORS[0]
        )
        self._w_bar_color.currentTextChanged.connect(
            lambda v: (setattr(self._style, "scale_bar_color", v), self._emit())
        )
        form.addRow("Bar colour", self._w_bar_color)

        self._w_bar_box = QCheckBox()
        self._w_bar_box.setChecked(s.scale_bar_box)
        self._w_bar_box.toggled.connect(
            lambda v: (setattr(self._style, "scale_bar_box", v), self._emit())
        )
        form.addRow("Background box", self._w_bar_box)

        self._w_box_color = QComboBox()
        self._w_box_color.addItems(_COLORS)
        self._w_box_color.setCurrentText(
            s.scale_bar_box_color if s.scale_bar_box_color in _COLORS else _COLORS[0]
        )
        self._w_box_color.currentTextChanged.connect(
            lambda v: (setattr(self._style, "scale_bar_box_color", v), self._emit())
        )
        form.addRow("Box colour", self._w_box_color)

        self._w_box_alpha = QDoubleSpinBox()
        self._w_box_alpha.setRange(0.0, 1.0)
        self._w_box_alpha.setDecimals(2)
        self._w_box_alpha.setSingleStep(0.05)
        self._w_box_alpha.setValue(s.scale_bar_box_alpha)
        self._w_box_alpha.valueChanged.connect(
            lambda v: (setattr(self._style, "scale_bar_box_alpha", v), self._emit())
        )
        form.addRow("Box alpha", self._w_box_alpha)

        self._w_box_margin = QDoubleSpinBox()
        self._w_box_margin.setRange(0.0, 20.0)
        self._w_box_margin.setDecimals(1)
        self._w_box_margin.setSuffix(" pt")
        self._w_box_margin.setValue(s.scale_bar_box_margin_pt)
        self._w_box_margin.valueChanged.connect(
            lambda v: (setattr(self._style, "scale_bar_box_margin_pt", v), self._emit())
        )
        form.addRow("Box margin", self._w_box_margin)

        # --- Text section ---
        form.addRow(QLabel("<b>Text</b>"))

        self._w_font_scale = QDoubleSpinBox()
        self._w_font_scale.setRange(0.5, 5.0)
        self._w_font_scale.setDecimals(2)
        self._w_font_scale.setSingleStep(0.1)
        self._w_font_scale.setValue(s.font_scale)
        self._w_font_scale.valueChanged.connect(
            lambda v: (setattr(self._style, "font_scale", v), self._emit())
        )
        form.addRow("Font scale", self._w_font_scale)

        self._w_show_title = QCheckBox()
        self._w_show_title.setChecked(s.show_title)
        self._w_show_title.toggled.connect(
            lambda v: (setattr(self._style, "show_title", v), self._emit())
        )
        form.addRow("Show title", self._w_show_title)

        self._w_title_scale = QDoubleSpinBox()
        self._w_title_scale.setRange(0.1, 5.0)
        self._w_title_scale.setDecimals(2)
        self._w_title_scale.setSingleStep(0.1)
        self._w_title_scale.setValue(s.title_scale)
        self._w_title_scale.setToolTip(
            "Size of the title alone, independent of Font scale — set small if the "
            "title is only there to identify the plot."
        )
        self._w_title_scale.valueChanged.connect(
            lambda v: (setattr(self._style, "title_scale", v), self._emit())
        )
        form.addRow("Title scale", self._w_title_scale)

        self._w_center_labels = QCheckBox()
        self._w_center_labels.setChecked(s.center_axis_labels)
        self._w_center_labels.toggled.connect(
            lambda v: (setattr(self._style, "center_axis_labels", v), self._emit())
        )
        form.addRow("Centre axis labels", self._w_center_labels)

        self._w_axes_mode = QComboBox()
        for value in AXES_MODES:
            self._w_axes_mode.addItem(_AXES_MODE_LABELS.get(value, value), value)
        self._w_axes_mode.setCurrentIndex(
            self._w_axes_mode.findData(s.axes_mode if s.axes_mode in AXES_MODES else "full")
        )
        self._w_axes_mode.setToolTip(
            "Axis decoration on map figures: 'No frame' hides the box around the plot "
            "(ticks and numbers stay); 'None' removes ticks, numbers and axis labels too — "
            "the scale bar and colourbar then carry the physical context. Trace, companion "
            "and diagnostic figures always keep their axes."
        )
        self._w_axes_mode.currentIndexChanged.connect(
            lambda i: (
                setattr(self._style, "axes_mode", self._w_axes_mode.itemData(i)),
                self._emit(),
            )
        )
        form.addRow("Axes", self._w_axes_mode)

        # --- Colourbar section ---
        form.addRow(QLabel("<b>Colourbar</b>"))

        self._w_colorbar = QCheckBox()
        self._w_colorbar.setChecked(s.colorbar)
        self._w_colorbar.toggled.connect(
            lambda v: (setattr(self._style, "colorbar", v), self._emit())
        )
        form.addRow("Show colourbar", self._w_colorbar)

        self._w_cbar_label = QLineEdit()
        self._w_cbar_label.setPlaceholderText("(use figure's own label)")
        self._w_cbar_label.setText(s.colorbar_label or "")
        self._w_cbar_label.textChanged.connect(
            lambda v: (
                setattr(self._style, "colorbar_label", v if v else None),
                self._emit(),
            )
        )
        form.addRow("Colourbar label", self._w_cbar_label)

        self._w_cbar_frac = QDoubleSpinBox()
        self._w_cbar_frac.setRange(0.01, 0.3)
        self._w_cbar_frac.setDecimals(3)
        self._w_cbar_frac.setSingleStep(0.005)
        self._w_cbar_frac.setValue(s.colorbar_fraction)
        self._w_cbar_frac.valueChanged.connect(
            lambda v: (setattr(self._style, "colorbar_fraction", v), self._emit())
        )
        form.addRow("Colourbar fraction", self._w_cbar_frac)

        self._w_cbar_ticks = QSpinBox()
        self._w_cbar_ticks.setRange(0, 20)
        self._w_cbar_ticks.setSpecialValueText("auto")
        self._w_cbar_ticks.setValue(s.colorbar_ticks)
        self._w_cbar_ticks.valueChanged.connect(
            lambda v: (setattr(self._style, "colorbar_ticks", v), self._emit())
        )
        form.addRow("Colourbar ticks", self._w_cbar_ticks)

        form.addRow(QLabel("<b>Colourbar — per group</b>"))
        self._w_tickfmt: dict[str, QComboBox] = {}
        self._w_offscale: dict[str, QDoubleSpinBox] = {}
        self._w_offpos: dict[str, QComboBox] = {}
        for grp, label in _CBAR_GROUPS:
            fmt_combo = QComboBox()
            for fmt in _TICK_FMTS:
                fmt_combo.addItem(_TICK_FMT_LABELS[fmt], fmt)
            cur = getattr(s, f"tickfmt_{grp}")
            fmt_combo.setCurrentIndex(_TICK_FMTS.index(cur if cur in _TICK_FMTS else "auto"))
            fmt_combo.currentIndexChanged.connect(
                lambda _i, g=grp, c=fmt_combo: (
                    setattr(self._style, f"tickfmt_{g}", c.currentData()),
                    self._emit(),
                )
            )

            off_scale = QDoubleSpinBox()
            off_scale.setRange(0.2, 5.0)
            off_scale.setDecimals(2)
            off_scale.setSingleStep(0.1)
            off_scale.setValue(getattr(s, f"offset_scale_{grp}"))
            off_scale.setToolTip(
                "Size of the scientific ×10ⁿ exponent (only when format = scientific)."
            )
            off_scale.valueChanged.connect(
                lambda v, g=grp: (setattr(self._style, f"offset_scale_{g}", v), self._emit())
            )

            off_pos = QComboBox()
            off_pos.addItems(_OFFSET_POS)
            off_pos.setCurrentText(getattr(s, f"offset_pos_{grp}"))
            off_pos.setToolTip(
                "Where the scientific ×10ⁿ exponent sits (only when format = scientific)."
            )
            off_pos.currentTextChanged.connect(
                lambda v, g=grp: (setattr(self._style, f"offset_pos_{g}", v), self._emit())
            )

            row = QHBoxLayout()
            row.addWidget(fmt_combo, 2)
            row.addWidget(off_scale, 1)
            row.addWidget(off_pos, 1)
            form.addRow(label, row)
            self._w_tickfmt[grp] = fmt_combo
            self._w_offscale[grp] = off_scale
            self._w_offpos[grp] = off_pos

        self._w_round_clim = QCheckBox()
        self._w_round_clim.setChecked(s.round_clim)
        self._w_round_clim.setToolTip(
            "Round the automatic colour limits outward to nice values (e.g. ±0.0778 → "
            "±0.08) so evenly spaced ticks are round numbers. The run log and Results "
            "tab state exactly what was rounded."
        )
        self._w_round_clim.toggled.connect(
            lambda v: (setattr(self._style, "round_clim", v), self._emit())
        )
        form.addRow("Round colour limits", self._w_round_clim)

        # --- Figure section ---
        form.addRow(QLabel("<b>Figure</b>"))

        self._w_fig_width = QComboBox()
        self._w_fig_width.addItems(_WIDTHS)
        _cur_width = (
            s.figure_width
            if isinstance(s.figure_width, str) and s.figure_width in _WIDTHS
            else "auto"
        )
        self._w_fig_width.setCurrentText(_cur_width)
        self._w_fig_width.currentTextChanged.connect(
            lambda v: (setattr(self._style, "figure_width", v), self._emit())
        )
        form.addRow("Figure width", self._w_fig_width)

        self._w_scale_umcm = QLineEdit()
        self._w_scale_umcm.setPlaceholderText("(blank = off)")
        _sv = fixed_scale(s)
        if _sv is not None:
            self._w_scale_umcm.setText(f"{_sv:g}")
        self._w_scale_umcm.setToolTip(
            "Fixed physical scale for map figures: µm of data per cm of page. When set, every "
            "map's data box is fitted so the printed scale (and the scale bar) is identical "
            "across figures; Figure width is ignored for maps. Trace figures follow this too "
            "unless Trace scale below overrides it. Blank turns it off. For identical bars "
            "across different crops also set an explicit Bar length."
        )
        self._w_scale_umcm.textChanged.connect(self._on_scale_umcm)
        form.addRow("Scale (µm/cm)", self._w_scale_umcm)

        self._w_trace_scale_umcm = QLineEdit()
        self._w_trace_scale_umcm.setPlaceholderText("(blank = follow Scale)")
        _tsv = _own_trace_scale(s)
        if _tsv is not None:
            self._w_trace_scale_umcm.setText(f"{_tsv:g}")
        self._w_trace_scale_umcm.setToolTip(
            "Fixed physical scale for the profiles trace (line-profile) figures only: µm of "
            "distance per cm of page. Blank = traces follow Scale (µm/cm) above. Hint: traces "
            "usually need a smaller value than the maps — start at about half the map scale or "
            "less; at the map's own scale the trace box tends to come out too small."
        )
        self._w_trace_scale_umcm.textChanged.connect(self._on_trace_scale_umcm)
        form.addRow("Trace scale (µm/cm)", self._w_trace_scale_umcm)

        self._w_trace_height_cm = QLineEdit()
        self._w_trace_height_cm.setPlaceholderText("(blank = 3)")
        _thv = getattr(s, "trace_height_cm", None)
        if _thv is not None:
            self._w_trace_height_cm.setText(f"{_thv:g}")
        self._w_trace_height_cm.setToolTip(
            "Fixed height of every trace plot box in cm of page. All traces of a run share "
            "it, so they align side-by-side. Blank = 3 cm. Only takes effect when a fixed "
            "scale (Scale and/or Trace scale) is set."
        )
        self._w_trace_height_cm.textChanged.connect(self._on_trace_height_cm)
        form.addRow("Trace height (cm)", self._w_trace_height_cm)

        # --- Output section ---
        form.addRow(QLabel("<b>Output</b>"))

        # 3 checkboxes for format selection
        self._w_fmt_png = QCheckBox("PNG")
        self._w_fmt_pdf = QCheckBox("PDF")
        self._w_fmt_svg = QCheckBox("SVG")
        self._w_fmt_png.setChecked("png" in s.formats)
        self._w_fmt_pdf.setChecked("pdf" in s.formats)
        self._w_fmt_svg.setChecked("svg" in s.formats)
        for cb in (self._w_fmt_png, self._w_fmt_pdf, self._w_fmt_svg):
            cb.toggled.connect(self._on_formats_changed)
        fmt_layout = QHBoxLayout()
        fmt_layout.addWidget(self._w_fmt_png)
        fmt_layout.addWidget(self._w_fmt_pdf)
        fmt_layout.addWidget(self._w_fmt_svg)
        form.addRow("Formats", fmt_layout)

        self._w_dpi = QSpinBox()
        self._w_dpi.setRange(72, 1200)
        self._w_dpi.setSingleStep(50)
        self._w_dpi.setSuffix(" dpi")
        self._w_dpi.setValue(s.dpi)
        self._w_dpi.valueChanged.connect(lambda v: (setattr(self._style, "dpi", v), self._emit()))
        form.addRow("DPI", self._w_dpi)

        # Keep a reference so sync_from_style can reach them
        self._format_checkboxes = (self._w_fmt_png, self._w_fmt_pdf, self._w_fmt_svg)
        self._format_names = ("png", "pdf", "svg")

    def _on_bar_auto_toggled(self, checked: bool) -> None:
        self._w_bar_len.setEnabled(not checked)
        if checked:
            setattr(self._style, "scale_bar_length_um", None)
        else:
            setattr(self._style, "scale_bar_length_um", self._w_bar_len.value())
        self._emit()

    @staticmethod
    def _parse_positive_float(text: str) -> float | None:
        """Blank/unparsable/non-positive -> None; else the parsed float.

        Shared by the three fixed-scale line-edit handlers below (`_on_scale_umcm`,
        `_on_trace_scale_umcm`, `_on_trace_height_cm`) — they were byte-identical
        apart from which style attribute they wrote.
        """
        t = text.strip()
        if not t:
            return None
        try:
            val = float(t)
        except ValueError:
            return None
        return val if val > 0 else None

    def _on_scale_umcm(self, text: str) -> None:
        self._style.scale_um_per_cm = self._parse_positive_float(text)
        self._emit()

    def _on_trace_scale_umcm(self, text: str) -> None:
        self._style.trace_scale_um_per_cm = self._parse_positive_float(text)
        self._emit()

    def _on_trace_height_cm(self, text: str) -> None:
        self._style.trace_height_cm = self._parse_positive_float(text)
        self._emit()

    def _on_formats_changed(self) -> None:
        fmts = tuple(
            name for cb, name in zip(self._format_checkboxes, self._format_names) if cb.isChecked()
        )
        setattr(self._style, "formats", fmts)
        # No re-render needed for format changes (only affects export, not preview)
        # Still emit changed so callers can react if they need to.
        self._emit()


def sanitize_stem(name: str) -> str:
    """Replace path-unsafe characters with underscores (shared by ExportDialog and export_all)."""
    return re.sub(r"[^\w\-.]", "_", name)


def save_spec(spec, out_dir: str, style) -> list[str]:
    """Build *spec* at *style* (scale bar forced off for non-map kinds) and savefig
    one file per ``style.formats`` into *out_dir* (sanitised stem). Returns the list of
    written paths; a single format whose savefig fails is skipped (the others still write).
    ``spec.build`` may raise — the CALLER decides how to record that.
    """
    eff_style = style if spec.kind == "map" else replace(style, scale_bar=False)
    fig = spec.build(eff_style)
    try:
        stem = sanitize_stem(spec.filename)
        written: list[str] = []
        for fmt in style.formats:
            path = os.path.join(out_dir, f"{stem}.{fmt}")
            # Write to a temp file then atomically rename, so a savefig that
            # fails mid-write never leaves a truncated/corrupt file at the
            # target path (the user would otherwise mistake it for a good export).
            tmp = f"{path}.part"
            try:
                # Pass format explicitly: the ".part" temp suffix would otherwise
                # make matplotlib infer an unknown format from the extension.
                fig.savefig(tmp, format=fmt, dpi=style.dpi, bbox_inches="tight", facecolor="white")
                os.replace(tmp, path)
                written.append(path)
            except Exception:  # noqa: BLE001 — skip a failing format, keep the rest
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
                continue
        return written
    finally:
        # Release the (potentially large) image arrays promptly instead of
        # waiting for GC — an "Export all" builds one Figure per layer.
        fig.clear()


class ExportDialog(QDialog):
    """Dialog for previewing and exporting a publication-quality figure.

    Shows a live preview of the selected :class:`FigureSpec` rendered with
    the current :class:`PlotStyle`, and writes PNG/PDF/SVG via :meth:`export_to`.
    """

    def __init__(
        self,
        specs: list[FigureSpec],
        index: int,
        global_style: PlotStyle,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export figure")
        self.resize(900, 620)
        self._specs = specs
        self._index = index
        self._style = replace(global_style)  # working copy
        self._global = global_style
        # (index, preview-relevant style) last actually rendered — lets _render
        # skip rebuilding (and re-reading the HDF5) when only export-only fields
        # (formats/dpi) changed, which do not affect the preview.
        self._last_render_key: object = None

        # Preview canvas
        self._canvas = MplCanvas()

        # Figure selector (drop-down at the top of the controls pane)
        self._selector = QComboBox()
        self._selector.addItems([s.title for s in specs])
        self._selector.setCurrentIndex(index)
        self._selector.currentIndexChanged.connect(self._on_select)

        # Debounced re-render: only re-draws 150 ms after the last control change
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(150)
        self._debounce.timeout.connect(self._render)

        # StyleControls widget bound to the working-copy style.
        self._controls = StyleControls(self._style)
        self._controls.changed.connect(self._schedule)

        reset_btn = QPushButton("Reset to global style")
        reset_btn.clicked.connect(self._on_reset)
        export_btn = QPushButton("Export")
        export_btn.clicked.connect(self._on_export)

        btns = QHBoxLayout()
        btns.addStretch(1)
        btns.addWidget(reset_btn)
        btns.addWidget(export_btn)

        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setWidget(self._controls)

        right = QVBoxLayout()
        right.addWidget(self._selector)
        right.addWidget(controls_scroll, 1)
        right.addLayout(btns)

        rw = QWidget()
        rw.setLayout(right)

        root = QHBoxLayout(self)
        root.addWidget(self._canvas, 2)
        root.addWidget(rw, 1)

        # Render on first show
        self._render()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _spec(self) -> FigureSpec:
        return self._specs[self._index]

    def _on_select(self, i: int) -> None:
        self._index = i
        self._render()

    def _on_reset(self) -> None:
        self._style = replace(self._global)
        self._controls.set_style(self._style)
        self._render()

    def _schedule(self) -> None:
        self._debounce.start()

    def _render(self) -> None:
        spec = self._spec()
        style = self._style if spec.kind == "map" else replace(self._style, scale_bar=False)
        # Skip the rebuild (and its HDF5 read) when nothing preview-relevant
        # changed: formats/dpi only affect export, not the displayed figure.
        key = (self._index, replace(style, formats=(), dpi=0))
        if key == self._last_render_key:
            return
        self._last_render_key = key
        try:
            fig = spec.build(style)
        except Exception as e:  # noqa: BLE001 — surface any build failure in the preview
            from matplotlib.figure import Figure

            fig = Figure(figsize=(6, 4), facecolor="white")
            fig.text(0.5, 0.5, f"Cannot render figure:\n{e}", ha="center", va="center", wrap=True)
        # Swap the canvas's figure to the freshly built one.
        # MplCanvas stores its figure as self._canvas.figure (a matplotlib Figure)
        # and the Qt canvas as self._canvas.canvas (FigureCanvasQTAgg).
        # We wire the new figure into the existing canvas widget so Qt doesn't
        # need a new widget to be re-parented.
        self._canvas.canvas.figure = fig
        fig.set_canvas(self._canvas.canvas)
        self._canvas.figure = fig  # keep the public attribute in sync (smoke checks this)
        self._canvas.canvas.draw_idle()

    def export_to(self, out_dir: str) -> list[str]:
        """Build the current spec and write one file per format.

        Returns the list of absolute paths SUCCESSFULLY written.  If
        ``spec.build`` raises the exception propagates to the caller.
        A per-format ``savefig`` failure is caught and skipped so that one
        broken backend or permission error does not abort the remaining formats.
        """
        os.makedirs(out_dir, exist_ok=True)
        return save_spec(self._spec(), out_dir, self._style)

    def _on_export(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Export to folder")
        if not d:
            return
        try:
            written = self.export_to(d)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Export failed", str(e))
            return
        requested = len(self._style.formats)
        note = (
            "" if len(written) == requested else f"\n({requested - len(written)} format(s) failed)"
        )
        QMessageBox.information(
            self,
            "Export complete",
            "Written:\n" + "\n".join(written) + note,
        )
