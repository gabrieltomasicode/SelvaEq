"""
Utilitários matemáticos de DSP e suavização de parâmetros (ParameterSmoother).

Estende `synth/utils.py` do SelvaSonic (`clamp`, `exp_curve`, `note_to_freq`, `get_time`)
e incorpora `validate_positive` (sem dependências de UI - PB28), conversões dB/linear
e a classe `ParameterSmoother` (filtro passa-baixa de 1 polo para evitar zipper noise - PB06/PB16).
"""

import time
from functools import wraps
from typing import Callable
import numpy as np


def validate_positive(func: Callable) -> Callable:
    """
    Decorador reutilizado de `ui/decorators.py` e `AudioSynth.py` que valida
    que argumentos numéricos posicionais ou nomeados não sejam negativos.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Se o primeiro argumento for 'self' ou 'cls' (objeto), ignora-o
        for arg in list(args) + list(kwargs.values()):
            if isinstance(arg, (int, float)) and not isinstance(arg, bool) and arg < 0:
                raise ValueError(f"Parâmetro negativo detectado: {arg}")
        return func(*args, **kwargs)
    return wrapper


def note_to_freq(note: int) -> float:
    """
    Converte número MIDI de nota (0-127) para frequência em Hz (A4 = 440 Hz).
    Reutilizado de `synth/utils.py`.
    """
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def exp_curve(x: float | np.ndarray, factor: float = 4.0) -> float | np.ndarray:
    """
    Calcula uma curva exponencial suave: 1 - exp(-factor * x).
    Reutilizado de `synth/utils.py`.
    """
    return 1.0 - np.exp(-factor * np.asarray(x))


def clamp(value: float, min_value: float, max_value: float) -> float:
    """
    Restringe `value` ao intervalo `[min_value, max_value]`.
    Reutilizado de `synth/utils.py`.
    """
    return max(min_value, min(value, max_value))


def get_time() -> float:
    """
    Retorna o timestamp atual em segundos.
    Reutilizado de `synth/utils.py`.
    """
    return time.time()


def db_to_linear(db: float | np.ndarray) -> float | np.ndarray:
    """Converte ganho em decibéis (dB) para amplitude linear: 10^(dB / 20)."""
    return np.power(10.0, np.asarray(db) / 20.0)


def linear_to_db(linear: float | np.ndarray, min_db: float = -120.0) -> float | np.ndarray:
    """Converte amplitude linear para decibéis (dBFS), protegendo contra log(0)."""
    eps = np.power(10.0, min_db / 20.0)
    mag = np.maximum(np.abs(linear), eps)
    return 20.0 * np.log10(mag)


class ParameterSmoother:
    """
    Filtro passa-baixa de 1 polo (One-pole lowpass) para interpolação de parâmetros
    (Frequência, Ganho, Q e transições de Bypass/Solo - PB06 / PB16).

    Espelha o contrato da classe `ParameterSmoother` prevista na camada C++/JUCE
    (`01_diagrama_classes.puml`).
    """

    def __init__(self, initial_value: float = 0.0, time_constant_ms: float = 10.0, fs: int = 44100):
        self.current: float = float(initial_value)
        self.target: float = float(initial_value)
        self.time_constant_ms: float = float(time_constant_ms)
        self.coeff: float = 1.0
        self.set_time_constant(time_constant_ms, fs)

    def set_time_constant(self, ms: float, fs: int) -> None:
        """Define a constante de tempo em milissegundos para a taxa de amostragem `fs`."""
        self.time_constant_ms = max(0.0, float(ms))
        if self.time_constant_ms <= 1e-6 or fs <= 0:
            self.coeff = 1.0
        else:
            tau_samples = (self.time_constant_ms * 0.001) * float(fs)
            self.coeff = float(1.0 - np.exp(-1.0 / max(tau_samples, 1.0)))

    def set_target(self, value: float) -> None:
        """Atualiza o valor alvo sem causar salto abrupto."""
        self.target = float(value)

    def reset(self, value: float) -> None:
        """Força o valor atual e o alvo imediatamente (sem rampa)."""
        self.current = float(value)
        self.target = float(value)

    def get_next_value(self, steps: int = 1) -> float:
        """Avança o filtro por `steps` amostras (ou um bloco) e retorna o valor suavizado."""
        if not self.is_smoothing():
            self.current = self.target
            return self.current
        if steps <= 1:
            self.current += self.coeff * (self.target - self.current)
        else:
            # Avanço analítico exato para `steps` amostras:
            retain = (1.0 - self.coeff) ** steps
            self.current = self.target - (self.target - self.current) * retain
        if abs(self.target - self.current) < 1e-5:
            self.current = self.target
        return self.current

    def is_smoothing(self) -> bool:
        """Indica se o parâmetro ainda está em transição em direção ao alvo."""
        return abs(self.target - self.current) >= 1e-5
