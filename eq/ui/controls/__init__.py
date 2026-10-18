"""
Subpacote de controles da interface (`eq.ui.controls`).
"""

from eq.ui.controls.band_controls import (
    BandControls,
    update_band_type,
    update_band_freq,
    update_band_gain,
    update_band_q,
    update_band_mode,
    update_band_slope,
    update_band_bypass,
    update_band_solo,
)

__all__ = [
    "BandControls",
    "update_band_type",
    "update_band_freq",
    "update_band_gain",
    "update_band_q",
    "update_band_mode",
    "update_band_slope",
    "update_band_bypass",
    "update_band_solo",
]
