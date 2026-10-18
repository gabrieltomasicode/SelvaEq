"""
Analisador de Espectro via Transformada Rápida de Fourier (`SpectrumFFT` / `PySpectrumFFT`).

Refatorado a partir de `ui/visuals.py` do SelvaSonic (PB07 / PB08):
- Descarta `generate_static_wave()` (que dependia de `WaveType` e `wavetables`).
- Recebe blocos temporais vindos da `visual_queue`, acumula em um buffer circular (`ring`)
  in-place sem realocação de memória (T12), aplica janela de Hann (`_apply_window`),
  executa `scipy.fft.rfft` e converte para dBFS com suavização exponencial temporal (`set_smoothing`).
- Inclui suporte a `Freeze` e envelope `Peak Hold` com decaimento configurável (F14).
- Mantido em `eq/dsp/spectrum.py` sem nenhuma dependência de `tkinter` ou `matplotlib` (PB28).
"""

from typing import Optional, Tuple
import numpy as np
from scipy.fft import rfft, rfftfreq

from eq.dsp.utils import clamp, linear_to_db


class SpectrumFFT:
    """
    Calcula o espectro de magnitudes em dBFS a partir dos blocos de áudio processados (PB08 / T12 / F14).
    """

    def __init__(
        self,
        fft_size: int = 2048,
        sample_rate: int = 44100,
        smoothing_alpha: float = 0.65,
    ):
        if fft_size < 64 or (fft_size & (fft_size - 1)) != 0:
            raise ValueError(f"fft_size deve ser uma potência de 2 >= 64 (recebido: {fft_size})")
        self.fft_size: int = int(fft_size)
        self.sample_rate: int = int(sample_rate)
        self.window: np.ndarray = np.hanning(self.fft_size).astype(np.float64)
        # Ganho coerente da janela de Hann (~0.5) para calibração exata em dBFS
        self._window_sum: float = float(np.sum(self.window))
        self.ring: np.ndarray = np.zeros(self.fft_size, dtype=np.float64)
        self.smoothing_alpha: float = clamp(float(smoothing_alpha), 0.0, 0.99)
        self._smoothed_db: Optional[np.ndarray] = None
        self.frozen: bool = False
        self._peak_hold_db: Optional[np.ndarray] = None
        self.peak_decay_db_per_frame: float = 0.35

    def set_sample_rate(self, sample_rate: int) -> None:
        """Atualiza a taxa de amostragem de referência para cálculo dos bins de frequência (PB29)."""
        if sample_rate <= 0:
            raise ValueError("sample_rate deve ser > 0")
        self.sample_rate = int(sample_rate)

    def set_smoothing(self, alpha: float) -> None:
        """Define o coeficiente de suavização temporal (0.0 = sem suavização, 0.95 = muito suave)."""
        self.smoothing_alpha = clamp(float(alpha), 0.0, 0.99)

    def set_freeze(self, frozen: bool) -> None:
        """Congela ou descongela a atualização do espectro mantendo o áudio intacto (F14)."""
        self.frozen = bool(frozen)

    def _apply_window(self, frame: np.ndarray) -> np.ndarray:
        """Aplica a janela de Hann ao frame temporal de tamanho `fft_size`."""
        return frame * self.window

    def push_block(self, block: np.ndarray) -> None:
        """
        Insere um novo bloco de áudio (mono ou estéreo) no buffer circular (`ring`) in-place (T12).
        Se `self.frozen` for True, ignora novas amostras para manter o espectro congelado (F14).
        """
        if self.frozen or block is None or block.size == 0:
            return
        arr = np.asarray(block, dtype=np.float64)
        if arr.ndim == 2:
            mono = np.mean(arr[:, :2], axis=1) if arr.shape[1] >= 2 else arr[:, 0]
        else:
            mono = arr.ravel()

        n = len(mono)
        if n >= self.fft_size:
            self.ring[:] = mono[-self.fft_size:]
        else:
            # Deslocamento in-place sem realocar self.ring (T12)
            self.ring[:-n] = self.ring[n:]
            self.ring[-n:] = mono

    def magnitudes_db(self, min_db: float = -120.0) -> np.ndarray:
        """
        Calcula a FFT real (`scipy.fft.rfft`) sobre o buffer circular janelado e retorna
        o array de magnitudes em decibéis (dBFS) com suavização temporal e atualização de Peak Hold (PB08 / F14).
        """
        if self.frozen and self._smoothed_db is not None:
            if self._peak_hold_db is None or self._peak_hold_db.shape != self._smoothed_db.shape:
                self._peak_hold_db = self._smoothed_db.copy()
            return self._smoothed_db.copy()

        windowed = self._apply_window(self.ring)
        spectrum = rfft(windowed, n=self.fft_size)
        # Normalização de amplitude de pico para senoide em escala cheia (0 dBFS = amplitude 1.0)
        mag = (2.0 * np.abs(spectrum)) / max(self._window_sum, 1e-9)
        mag[0] *= 0.5  # Componente DC não tem espelho negativo
        if len(mag) > 1:
            mag[-1] *= 0.5  # Bin de Nyquist

        db = np.asarray(linear_to_db(mag, min_db=min_db), dtype=np.float64)
        if self._smoothed_db is None or self._smoothed_db.shape != db.shape:
            self._smoothed_db = db.copy()
        else:
            alpha = self.smoothing_alpha
            self._smoothed_db = alpha * self._smoothed_db + (1.0 - alpha) * db

        if self._peak_hold_db is None or self._peak_hold_db.shape != self._smoothed_db.shape:
            self._peak_hold_db = self._smoothed_db.copy()
        else:
            self._peak_hold_db = np.maximum(
                self._smoothed_db,
                self._peak_hold_db - self.peak_decay_db_per_frame,
            )
        return self._smoothed_db.copy()

    def frequencies_hz(self) -> np.ndarray:
        """Retorna o vetor de frequências em Hz correspondente aos bins da `rfft`."""
        return rfftfreq(self.fft_size, d=1.0 / float(self.sample_rate))

    def spectrum_curve(self, min_db: float = -120.0) -> Tuple[np.ndarray, np.ndarray]:
        """Retorna a tupla `(freqs_hz, magnitudes_db)` pronta para plotagem logarítmica."""
        return self.frequencies_hz(), self.magnitudes_db(min_db=min_db)

    def peak_hold_curve(self, min_db: float = -120.0) -> Tuple[np.ndarray, np.ndarray]:
        """Retorna a tupla `(freqs_hz, peak_hold_db)` com o envelope de pico (F14)."""
        if self._peak_hold_db is None:
            self.magnitudes_db(min_db=min_db)
        return self.frequencies_hz(), self._peak_hold_db.copy()

    def bin_to_hz(self, bin_index: int) -> float:
        """Converte o índice de um bin da FFT para sua frequência central em Hz."""
        if bin_index < 0 or bin_index > self.fft_size // 2:
            raise ValueError(f"bin_index fora do intervalo [0, {self.fft_size // 2}]: {bin_index}")
        return float(bin_index) * float(self.sample_rate) / float(self.fft_size)

    def reset(self) -> None:
        """Zera o buffer circular, o histórico de suavização e o envelope de pico (F14)."""
        self.ring.fill(0.0)
        self._smoothed_db = None
        self._peak_hold_db = None


# Alias explícito conforme seção 2.1 do documento de planejamento
PySpectrumFFT = SpectrumFFT
