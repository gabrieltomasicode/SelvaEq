"""
Pacote de Processamento Digital de Sinais (`eq.dsp`).
Zero dependência de bibliotecas de interface gráfica (PB28).
"""

from eq.config import FilterType, ChannelMode, Slope, EqConfig
from eq.dsp.biquad_math import Coeffs, BiquadMath, PyBiquadMath
from eq.dsp.band import EqBand
from eq.dsp.engine import EqEngine, PyEqEngine
from eq.dsp.spectrum import SpectrumFFT, PySpectrumFFT
from eq.dsp.state_controller import StateController
from eq.dsp.utils import (
    ParameterSmoother,
    clamp,
    exp_curve,
    note_to_freq,
    get_time,
    db_to_linear,
    linear_to_db,
    validate_positive,
)

__all__ = [
    "FilterType",
    "ChannelMode",
    "Slope",
    "EqConfig",
    "Coeffs",
    "BiquadMath",
    "PyBiquadMath",
    "EqBand",
    "EqEngine",
    "PyEqEngine",
    "SpectrumFFT",
    "PySpectrumFFT",
    "StateController",
    "ParameterSmoother",
    "clamp",
    "exp_curve",
    "note_to_freq",
    "get_time",
    "db_to_linear",
    "linear_to_db",
    "validate_positive",
]
