"""
Matemática de Filtros Biquad (RBJ Audio EQ Cookbook) e Cascateamento Butterworth.

Refatora e estende `synth/filters.py::get_filter_coeffs` do SelvaSonic:
- Preserva o padrão `@lru_cache` para evitar recálculo desnecessário de coeficientes por bloco.
- Substitui `scipy.signal.butter(N=2)` de ordem fixa pelas equações analíticas completas
  do RBJ Audio EQ Cookbook (PB05) e pelo cascateamento Butterworth de 6 a 96 dB/oct.
- Garante independência total de Sample Rate (`fs` dinâmico - PB29).
- Implementa extração de resposta teórica vetorizada em frequência (PB09 / T05 / T07 / T11).
- Zero dependência de bibliotecas gráficas (PB28).
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple, Union, TYPE_CHECKING
import math
import numpy as np
from scipy.signal import freqz

from eq.config import FilterType, Slope
from eq.dsp.utils import clamp

if TYPE_CHECKING:
    from eq.dsp.band import EqBand


@dataclass(frozen=True)
class Coeffs:
    """
    Value Object imutável que representa os coeficientes normalizados (a0 = 1.0)
    de um estágio Biquad IIR:
        H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2)
    """
    b0: float = 1.0
    b1: float = 0.0
    b2: float = 0.0
    a1: float = 0.0
    a2: float = 0.0

    def as_ba(self) -> Tuple[np.ndarray, np.ndarray]:
        """Retorna a tupla `(b, a)` compatível com `scipy.signal.lfilter` e `freqz`."""
        b = np.array([self.b0, self.b1, self.b2], dtype=np.float64)
        a = np.array([1.0, self.a1, self.a2], dtype=np.float64)
        return b, a

    def as_sos(self) -> np.ndarray:
        """Retorna linha de matriz SOS `[b0, b1, b2, 1.0, a1, a2]`."""
        return np.array([self.b0, self.b1, self.b2, 1.0, self.a1, self.a2], dtype=np.float64)


class BiquadMath:
    """
    Módulo puramente matemático (RBJ Audio EQ Cookbook + Cascata Butterworth + Resposta Teórica).
    """

    @staticmethod
    def _sanitize_params(fs: int, f0: float, q: float) -> Tuple[float, float, float]:
        """
        Garante que frequência de corte e fator Q respeitem o limite de Nyquist
        e não causem divisão por zero, preservando a proteção original de `synth/filters.py`.
        """
        if fs <= 0:
            raise ValueError(f"Sample rate (fs) deve ser > 0, recebido: {fs}")
        nyquist = 0.5 * float(fs)
        safe_f0 = clamp(float(f0), 1.0, max(1.0, nyquist * 0.999))
        safe_q = max(float(q), 1e-4)
        return float(fs), safe_f0, safe_q

    @staticmethod
    @lru_cache(maxsize=512)
    def _cached_coeffs(
        ftype_name: str,
        fs: float,
        f0: float,
        gain_db: float,
        q_or_s: float,
    ) -> Coeffs:
        """
        Calcula e armazena em cache LRU os coeficientes RBJ normalizados por `a0`.
        Padrão herdado diretamente de `synth/filters.py::get_filter_coeffs`.
        """
        fs, f0, q = BiquadMath._sanitize_params(int(fs), f0, q_or_s)
        w0 = 2.0 * math.pi * f0 / fs
        cos_w0 = math.cos(w0)
        sin_w0 = math.sin(w0)
        alpha = sin_w0 / (2.0 * q)
        A = math.pow(10.0, float(gain_db) / 40.0)  # 10^(dBgain/40) para peaking e shelving

        if ftype_name == FilterType.PEAKING.value:
            b0 = 1.0 + alpha * A
            b1 = -2.0 * cos_w0
            b2 = 1.0 - alpha * A
            a0 = 1.0 + alpha / A
            a1 = -2.0 * cos_w0
            a2 = 1.0 - alpha / A

        elif ftype_name == FilterType.LOW_SHELF.value:
            # Usa parametrização direta por Q do RBJ Cookbook: alpha = sin(w0)/(2*Q) (T07)
            two_sqrt_a_alpha = 2.0 * math.sqrt(A) * alpha
            b0 = A * ((A + 1.0) - (A - 1.0) * cos_w0 + two_sqrt_a_alpha)
            b1 = 2.0 * A * ((A - 1.0) - (A + 1.0) * cos_w0)
            b2 = A * ((A + 1.0) - (A - 1.0) * cos_w0 - two_sqrt_a_alpha)
            a0 = (A + 1.0) + (A - 1.0) * cos_w0 + two_sqrt_a_alpha
            a1 = -2.0 * ((A - 1.0) + (A + 1.0) * cos_w0)
            a2 = (A + 1.0) + (A - 1.0) * cos_w0 - two_sqrt_a_alpha

        elif ftype_name == FilterType.HIGH_SHELF.value:
            # Usa parametrização direta por Q do RBJ Cookbook: alpha = sin(w0)/(2*Q) (T07)
            two_sqrt_a_alpha = 2.0 * math.sqrt(A) * alpha
            b0 = A * ((A + 1.0) + (A - 1.0) * cos_w0 + two_sqrt_a_alpha)
            b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cos_w0)
            b2 = A * ((A + 1.0) + (A - 1.0) * cos_w0 - two_sqrt_a_alpha)
            a0 = (A + 1.0) - (A - 1.0) * cos_w0 + two_sqrt_a_alpha
            a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cos_w0)
            a2 = (A + 1.0) - (A - 1.0) * cos_w0 - two_sqrt_a_alpha

        elif ftype_name == FilterType.LOW_CUT.value:
            # High-Pass Filter ( corta graves abaixo de f0 )
            b0 = (1.0 + cos_w0) / 2.0
            b1 = -(1.0 + cos_w0)
            b2 = (1.0 + cos_w0) / 2.0
            a0 = 1.0 + alpha
            a1 = -2.0 * cos_w0
            a2 = 1.0 - alpha

        elif ftype_name == FilterType.HIGH_CUT.value:
            # Low-Pass Filter ( corta agudos acima de f0 )
            b0 = (1.0 - cos_w0) / 2.0
            b1 = 1.0 - cos_w0
            b2 = (1.0 - cos_w0) / 2.0
            a0 = 1.0 + alpha
            a1 = -2.0 * cos_w0
            a2 = 1.0 - alpha

        elif ftype_name == FilterType.BAND_PASS.value:
            # Band-pass com ganho de pico constante 0 dB
            b0 = alpha
            b1 = 0.0
            b2 = -alpha
            a0 = 1.0 + alpha
            a1 = -2.0 * cos_w0
            a2 = 1.0 - alpha

        elif ftype_name == FilterType.NOTCH.value:
            b0 = 1.0
            b1 = -2.0 * cos_w0
            b2 = 1.0
            a0 = 1.0 + alpha
            a1 = -2.0 * cos_w0
            a2 = 1.0 - alpha

        elif ftype_name == FilterType.ALL_PASS.value:
            b0 = 1.0 - alpha
            b1 = -2.0 * cos_w0
            b2 = 1.0 + alpha
            a0 = 1.0 + alpha
            a1 = -2.0 * cos_w0
            a2 = 1.0 - alpha

        else:
            raise ValueError(f"Tipo de filtro desconhecido: {ftype_name}")

        inv_a0 = 1.0 / a0
        return Coeffs(
            b0=b0 * inv_a0,
            b1=b1 * inv_a0,
            b2=b2 * inv_a0,
            a1=a1 * inv_a0,
            a2=a2 * inv_a0,
        )

    @staticmethod
    def peaking(fs: int, f0: float, gain_db: float, q: float = 0.70710678) -> Coeffs:
        """Filtro Peaking / Bell EQ (RBJ Cookbook)."""
        return BiquadMath._cached_coeffs(
            FilterType.PEAKING.value, round(float(fs), 2), round(float(f0), 4), round(float(gain_db), 4), round(float(q), 6)
        )

    @staticmethod
    def low_shelf(fs: int, f0: float, gain_db: float, s: float = 1.0) -> Coeffs:
        """Filtro Low Shelf (RBJ Cookbook)."""
        return BiquadMath._cached_coeffs(
            FilterType.LOW_SHELF.value, round(float(fs), 2), round(float(f0), 4), round(float(gain_db), 4), round(float(s), 6)
        )

    @staticmethod
    def high_shelf(fs: int, f0: float, gain_db: float, s: float = 1.0) -> Coeffs:
        """Filtro High Shelf (RBJ Cookbook)."""
        return BiquadMath._cached_coeffs(
            FilterType.HIGH_SHELF.value, round(float(fs), 2), round(float(f0), 4), round(float(gain_db), 4), round(float(s), 6)
        )

    @staticmethod
    def low_cut(fs: int, f0: float, q: float = 0.70710678) -> Coeffs:
        """Filtro Low Cut / High-Pass de 2ª ordem (RBJ Cookbook)."""
        return BiquadMath._cached_coeffs(
            FilterType.LOW_CUT.value, round(float(fs), 2), round(float(f0), 4), 0.0, round(float(q), 6)
        )

    @staticmethod
    def high_cut(fs: int, f0: float, q: float = 0.70710678) -> Coeffs:
        """Filtro High Cut / Low-Pass de 2ª ordem (RBJ Cookbook)."""
        return BiquadMath._cached_coeffs(
            FilterType.HIGH_CUT.value, round(float(fs), 2), round(float(f0), 4), 0.0, round(float(q), 6)
        )

    @staticmethod
    def band_pass(fs: int, f0: float, q: float = 1.0) -> Coeffs:
        """Filtro Band-Pass (ganho de pico 0 dB - RBJ Cookbook)."""
        return BiquadMath._cached_coeffs(
            FilterType.BAND_PASS.value, round(float(fs), 2), round(float(f0), 4), 0.0, round(float(q), 6)
        )

    @staticmethod
    def notch(fs: int, f0: float, q: float = 1.0) -> Coeffs:
        """Filtro Notch / Rejeita-Faixa (RBJ Cookbook)."""
        return BiquadMath._cached_coeffs(
            FilterType.NOTCH.value, round(float(fs), 2), round(float(f0), 4), 0.0, round(float(q), 6)
        )

    @staticmethod
    def all_pass(fs: int, f0: float, q: float = 0.70710678) -> Coeffs:
        """Filtro All-Pass de 2ª ordem (RBJ Cookbook)."""
        return BiquadMath._cached_coeffs(
            FilterType.ALL_PASS.value, round(float(fs), 2), round(float(f0), 4), 0.0, round(float(q), 6)
        )

    @staticmethod
    def first_order_cut(fs: int, f0: float, ftype: FilterType) -> Coeffs:
        """
        Calcula estágio de 1ª ordem (6 dB/oct) via transformada bilinear exata,
        embutido na estrutura `Coeffs` (com b2=0, a2=0) para total interoperabilidade.
        """
        fs_f, safe_f0, _ = BiquadMath._sanitize_params(int(fs), f0, 0.707)
        k = math.tan(math.pi * safe_f0 / fs_f)
        norm = 1.0 / (1.0 + k)
        if ftype == FilterType.LOW_CUT:
            b0 = norm
            b1 = -norm
            a1 = (k - 1.0) * norm
        elif ftype == FilterType.HIGH_CUT:
            b0 = k * norm
            b1 = k * norm
            a1 = (k - 1.0) * norm
        else:
            raise ValueError(f"first_order_cut suporta apenas LOW_CUT ou HIGH_CUT, recebido: {ftype}")
        return Coeffs(b0=b0, b1=b1, b2=0.0, a1=a1, a2=0.0)

    @staticmethod
    def butterworth_q_table(order: int) -> List[float]:
        """
        Calcula a tabela de fatores Q de cada par de polos conjugados para um filtro
        Butterworth de ordem `order` (PB05).
        Para ordem par N = 2M, k = 0 .. M-1:
            Q_k = 1 / (2 * sin(pi * (2k + 1) / (2N)))
        """
        if order < 1:
            raise ValueError(f"Ordem Butterworth deve ser >= 1, recebido: {order}")
        if order == 1:
            return [1.0 / math.sqrt(2.0)]
        num_biquads = order // 2
        qs: List[float] = []
        for k in range(num_biquads):
            angle = math.pi * (2 * k + 1) / (2 * order)
            q_k = 1.0 / (2.0 * math.sin(angle))
            qs.append(q_k)
        return qs

    @staticmethod
    def cascade(
        fs: int,
        f0: float,
        slope: Slope = Slope.S12,
        ftype: FilterType = FilterType.LOW_CUT,
        q: float = 0.70710678,
        gain_db: float = 0.0,
    ) -> List[Coeffs]:
        """
        Gera a lista de estágios Biquad (`List[Coeffs]`) para o filtro configurado (PB05).
        - Para `LOW_CUT` e `HIGH_CUT`: gera de 1 a 8 biquads conforme `slope` (S6 a S96 dB/oct).
          Em S12, respeita o fator `q` ajustado pelo usuário; em ordens superiores (S24, S48, S96),
          distribui os fatores Q segundo os polos Butterworth escalonados pelo `q` relativo.
        - Para os demais tipos (`PEAKING`, `LOW_SHELF`, `HIGH_SHELF`, `BAND_PASS`, `NOTCH`, `ALL_PASS`):
          retorna uma lista contendo o estágio correspondente.
        """
        if ftype in (FilterType.LOW_CUT, FilterType.HIGH_CUT):
            order = slope.order
            if order == 1:
                return [BiquadMath.first_order_cut(fs, f0, ftype)]
            if order == 2:
                stage = (
                    BiquadMath.low_cut(fs, f0, q)
                    if ftype == FilterType.LOW_CUT
                    else BiquadMath.high_cut(fs, f0, q)
                )
                return [stage]

            # Ordens 4 (24 dB/oct), 8 (48 dB/oct), 16 (96 dB/oct -> 8 biquads)
            q_table = BiquadMath.butterworth_q_table(order)
            q_scale = max(float(q), 0.1) / (1.0 / math.sqrt(2.0))
            stages: List[Coeffs] = []
            for idx, q_base in enumerate(q_table):
                # Escala o Q do estágio mais ressonante (idx == 0, maior Q_k) (T07)
                stage_q = q_base * q_scale if idx == 0 else q_base
                stage = (
                    BiquadMath.low_cut(fs, f0, stage_q)
                    if ftype == FilterType.LOW_CUT
                    else BiquadMath.high_cut(fs, f0, stage_q)
                )
                stages.append(stage)
            return stages

        if ftype == FilterType.PEAKING:
            return [BiquadMath.peaking(fs, f0, gain_db, q)]
        if ftype == FilterType.LOW_SHELF:
            return [BiquadMath.low_shelf(fs, f0, gain_db, q)]
        if ftype == FilterType.HIGH_SHELF:
            return [BiquadMath.high_shelf(fs, f0, gain_db, q)]
        if ftype == FilterType.BAND_PASS:
            return [BiquadMath.band_pass(fs, f0, q)]
        if ftype == FilterType.NOTCH:
            return [BiquadMath.notch(fs, f0, q)]
        if ftype == FilterType.ALL_PASS:
            return [BiquadMath.all_pass(fs, f0, q)]

        raise ValueError(f"Tipo de filtro não suportado: {ftype}")

    @staticmethod
    def freq_response(
        coeffs: Union[Coeffs, List[Coeffs]],
        fs: int,
        n: int = 2048,
        freqs: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula a resposta em frequência complexa `(freqs_hz, H)` de um estágio ou cascata
        de estágios avaliando diretamente a função de transferência racional em z^-1 e z^-2 (PB09 / T11).
        """
        stages = [coeffs] if isinstance(coeffs, Coeffs) else list(coeffs)
        if freqs is None:
            freqs_hz = np.geomspace(20.0, min(20000.0, 0.499 * fs), num=n)
        else:
            freqs_hz = np.asarray(freqs, dtype=np.float64)

        w = 2.0 * np.pi * freqs_hz / float(fs)
        z1 = np.exp(-1j * w)
        z2 = z1 * z1

        h_total = np.ones_like(freqs_hz, dtype=np.complex128)
        for c in stages:
            num = c.b0 + c.b1 * z1 + c.b2 * z2
            den = 1.0 + c.a1 * z1 + c.a2 * z2
            h_total *= num / den
        return freqs_hz, h_total

    @staticmethod
    def combined_response(
        bands: List["EqBand"],
        fs: int,
        n: int = 2048,
        freqs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Calcula a curva teórica global em decibéis (dB) resultante da cascata de todas
        as bandas ativas na taxa de amostragem efetiva `fs` (PB09 / T02 / T05).

        Retorna:
            np.ndarray: Array 1D de tamanho `n` com a magnitude global em dB.
        """
        if freqs is None:
            freqs_hz = np.geomspace(20.0, min(20000.0, 0.499 * fs), num=n)
        else:
            freqs_hz = np.asarray(freqs, dtype=np.float64)

        # Verifica se há alguma banda em modo SOLO (PB12)
        solo_bands = [b for b in bands if b.solo and not b.bypass]
        active_bands = solo_bands if solo_bands else [b for b in bands if not b.bypass]

        h_global = np.ones_like(freqs_hz, dtype=np.complex128)
        for band in active_bands:
            if solo_bands:
                # Em modo Solo (PB12), a banda em foco opera como band-pass isolador
                bp_coeffs = BiquadMath.band_pass(fs, band.freq_hz, band.q)
                _, h_band = BiquadMath.freq_response(bp_coeffs, fs, n=len(freqs_hz), freqs=freqs_hz)
            else:
                # Avalia os estágios nos parâmetros alvos da banda na taxa efetiva `fs` (T02 / T05)
                target_stages = BiquadMath.cascade(
                    fs=fs,
                    f0=band.freq_hz,
                    slope=band.slope,
                    ftype=band.ftype,
                    q=band.q,
                    gain_db=band.gain_db,
                )
                _, h_band = BiquadMath.freq_response(target_stages, fs, n=len(freqs_hz), freqs=freqs_hz)
            h_global *= h_band

        mag = np.maximum(np.abs(h_global), 1e-9)
        return 20.0 * np.log10(mag)


# Alias explícito conforme seção 2.1 do documento de planejamento
PyBiquadMath = BiquadMath
