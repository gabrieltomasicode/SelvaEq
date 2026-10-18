"""
Subpacote de Interface Gráfica e Controladores (`eq.ui`).
Depende de `eq.dsp` e `eq.io`, nunca o inverso (PB28).
"""

from eq.ui.decorators import validate_positive
from eq.ui.widgets import (
    SelvaSlider,
    apply_dark_ttk_style,
    resolve_font_family,
    get_ui_fonts,
    create_band_panel_widgets,
    create_audio_io_panel_widgets,
    create_test_mode_panel_widgets,
    create_eq_system_controls,
)
from eq.ui.curve_view import BandNode, InteractiveCurveController
from eq.ui.controls.band_controls import BandControls
from eq.ui.interface import EqInterface, PyGraphUI

__all__ = [
    "validate_positive",
    "SelvaSlider",
    "apply_dark_ttk_style",
    "resolve_font_family",
    "get_ui_fonts",
    "create_band_panel_widgets",
    "create_audio_io_panel_widgets",
    "create_test_mode_panel_widgets",
    "create_eq_system_controls",
    "BandNode",
    "InteractiveCurveController",
    "BandControls",
    "EqInterface",
    "PyGraphUI",
]
