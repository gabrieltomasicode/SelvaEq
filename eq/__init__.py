"""
Pacote Principal do Protótipo Python do Equalizador Paramétrico de 24 Bandas (`SelvaEQ`).
Derivado e refatorado a partir da arquitetura MVC/POO do projeto SelvaSonic.
"""

from eq.config import FilterType, ChannelMode, Slope, InputSource, EqConfig, SUPPORTED_SAMPLE_RATES
from eq.dsp import (
    Coeffs,
    BiquadMath,
    PyBiquadMath,
    EqBand,
    EqEngine,
    PyEqEngine,
    SpectrumFFT,
    PySpectrumFFT,
    StateController,
    ParameterSmoother,
)
from eq.io import AudioIO, PyAudioIO, RealtimeMediaStreamer, PresetIO, visual_queue

__all__ = [
    "FilterType",
    "ChannelMode",
    "Slope",
    "InputSource",
    "EqConfig",
    "SUPPORTED_SAMPLE_RATES",
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
    "AudioIO",
    "PyAudioIO",
    "RealtimeMediaStreamer",
    "PresetIO",
    "visual_queue",
]
