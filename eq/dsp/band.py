"""
Classe EqBand - Representa uma banda paramétrica independente do Equalizador.

Refatora e evolui o padrão de filtragem contínua de `synth/filters.py::apply_filter()`:
- Preserva o vetor de estado `zi`/`zf` entre blocos via `scipy.signal.sosfilt` e `lfilter_zi`
  para eliminar clicks e descontinuidades nas bordas dos buffers (T06).
- O estado vive na própria banda (`EqBand`), suportando múltiplos estágios em cascata
  (1 a 8 biquads conforme `Slope`) e 2 canais independentes (L/R ou Mid/Side),
  com estado isolado `_zi_solo` para escuta Solo e reinicialização suave pós-Bypass (T01 / T03).
- Inclui suavizadores `ParameterSmoother` por parâmetro (freq, gain, Q) processados
  em sub-blocos de 32 amostras (PB06 / PB16 / T02) e proteção contra NaN/Inf (T14).
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from scipy.signal import lfilter, lfilter_zi, sosfilt

from eq.config import FilterType, ChannelMode, Slope
from eq.dsp.biquad_math import BiquadMath, Coeffs
from eq.dsp.utils import ParameterSmoother, clamp


SUBBLOCK_SIZE = 32


class EqBand:
    """
    Encapsula os parâmetros, coeficientes em cascata e estado contínuo IIR (`zi`)
    de uma única banda do equalizador (PB04, PB05, PB12, PB16, PB22).
    """

    SUBBLOCK_SIZE = SUBBLOCK_SIZE

    def __init__(
        self,
        index: int = 0,
        freq_hz: float = 1000.0,
        gain_db: float = 0.0,
        q: float = 0.70710678,
        ftype: FilterType = FilterType.PEAKING,
        slope: Slope = Slope.S12,
        mode: ChannelMode = ChannelMode.STEREO,
        bypass: bool = False,
        solo: bool = False,
        sample_rate: int = 44100,
    ):
        self.index: int = int(index)
        self.freq_hz: float = clamp(float(freq_hz), 20.0, 20000.0)
        self.gain_db: float = clamp(float(gain_db), -24.0, 24.0)
        self.q: float = clamp(float(q), 0.1, 30.0)
        self.ftype: FilterType = ftype if isinstance(ftype, FilterType) else FilterType[str(ftype)]
        self.slope: Slope = slope if isinstance(slope, Slope) else (
            Slope(int(slope)) if isinstance(slope, int) else Slope[str(slope)]
        )
        self.mode: ChannelMode = mode if isinstance(mode, ChannelMode) else ChannelMode[str(mode)]
        self.bypass: bool = bool(bypass)
        self.solo: bool = bool(solo)

        self._fs: int = int(sample_rate)
        self.coeffs: List[Coeffs] = []
        self._sos: np.ndarray = np.zeros((0, 6), dtype=np.float64)
        self._filter_bundle: Tuple[Tuple[Coeffs, ...], np.ndarray] = ((), self._sos)
        # Estado zi para sosfilt: shape [num_stages, 2_delays, 2_canais] (T01 / T03 / T06)
        self._zi: Optional[np.ndarray] = None
        self._zi_solo: Optional[np.ndarray] = None
        self._was_bypassed: bool = bool(bypass)
        self._reset_requested: bool = False

        # Suavizadores de parâmetros (PB06 / PB16 / T02)
        self._freq_smoother = ParameterSmoother(self.freq_hz, time_constant_ms=5.0, fs=self._fs)
        self._gain_smoother = ParameterSmoother(self.gain_db, time_constant_ms=5.0, fs=self._fs)
        self._q_smoother = ParameterSmoother(self.q, time_constant_ms=5.0, fs=self._fs)
        self._mix_smoother = ParameterSmoother(0.0 if self.bypass else 1.0, time_constant_ms=5.0, fs=self._fs)

        self.recalculate(self._fs, use_smoothed=False)

    def is_smoothing(self) -> bool:
        """Indica se qualquer parâmetro acústico ou de bypass ainda está interpolando (T02 / T06)."""
        return (
            self._freq_smoother.is_smoothing()
            or self._gain_smoother.is_smoothing()
            or self._q_smoother.is_smoothing()
            or self._mix_smoother.is_smoothing()
        )

    def set_targets(
        self,
        freq_hz: Optional[float] = None,
        gain_db: Optional[float] = None,
        q: Optional[float] = None,
        bypass: Optional[bool] = None,
    ) -> None:
        """Define novos alvos para os suavizadores de parâmetros (PB06 / PB16 / T02 / T03)."""
        if freq_hz is not None:
            self.freq_hz = clamp(float(freq_hz), 20.0, max(20.0, 0.499 * self._fs))
            self._freq_smoother.set_target(self.freq_hz)
        if gain_db is not None:
            self.gain_db = clamp(float(gain_db), -24.0, 24.0)
            self._gain_smoother.set_target(self.gain_db)
        if q is not None:
            self.q = clamp(float(q), 0.1, 30.0)
            self._q_smoother.set_target(self.q)
        if bypass is not None:
            new_bypass = bool(bypass)
            if self.bypass and not new_bypass and self._mix_smoother.current <= 1e-4:
                self._was_bypassed = True
            self.bypass = new_bypass
            self._mix_smoother.set_target(0.0 if self.bypass else 1.0)

    def recalculate(self, fs: int, use_smoothed: bool = False) -> None:
        """
        Recalcula a cascata de coeficientes Biquad e a matriz SOS para a taxa `fs` (PB29 / T01 / T02 / T06).
        Atualiza `_filter_bundle = (tuple(new_coeffs), new_sos)` atomicamente sem invalidar `self._zi` (T01).
        """
        if fs <= 0:
            raise ValueError(f"Sample rate deve ser > 0 (recebido: {fs})")
        self._fs = int(fs)
        self._freq_smoother.set_time_constant(5.0, self._fs)
        self._gain_smoother.set_time_constant(5.0, self._fs)
        self._q_smoother.set_time_constant(5.0, self._fs)
        self._mix_smoother.set_time_constant(5.0, self._fs)

        if not use_smoothed:
            self._freq_smoother.reset(self.freq_hz)
            self._gain_smoother.reset(self.gain_db)
            self._q_smoother.reset(self.q)
            f0_val = self.freq_hz
            gain_val = self.gain_db
            q_val = self.q
        else:
            f0_val = clamp(self._freq_smoother.current, 20.0, max(20.0, 0.499 * self._fs))
            gain_val = clamp(self._gain_smoother.current, -24.0, 24.0)
            q_val = clamp(self._q_smoother.current, 0.1, 30.0)

        new_coeffs = BiquadMath.cascade(
            fs=self._fs,
            f0=f0_val,
            slope=self.slope,
            ftype=self.ftype,
            q=q_val,
            gain_db=gain_val,
        )
        new_sos = (
            np.vstack([c.as_sos() for c in new_coeffs])
            if new_coeffs
            else np.zeros((0, 6), dtype=np.float64)
        )
        # Troca atômica da tupla (coeffs, sos) para que stages e sos tenham sempre o mesmo tamanho (T01)
        self._filter_bundle = (tuple(new_coeffs), new_sos)
        self._sos = new_sos
        self.coeffs = list(new_coeffs)

    def reset_state(self) -> None:
        """Solicita limpeza segura da memória interna (`zi` e `_zi_solo`) no próximo ciclo (T01)."""
        self._reset_requested = True

    def _ensure_zi(
        self,
        stages: List[Coeffs],
        first_samples_lr: np.ndarray,
        solo_mode: bool = False,
    ) -> np.ndarray:
        """
        Inicializa ou adapta o estado `zi` usando `scipy.signal.lfilter_zi` ponderado pela primeira
        amostra de cada canal, operando exclusivamente dentro de `process()` (T01 / T03 / T06).
        """
        num_stages = len(stages)
        if solo_mode:
            zi_cur = self._zi_solo
            if (
                zi_cur is not None
                and not self._reset_requested
                and zi_cur.shape == (num_stages, 2, 2)
            ):
                return zi_cur
            new_zi_solo = np.zeros((num_stages, 2, 2), dtype=np.float64)
            for s_idx, c in enumerate(stages):
                b, a = c.as_ba()
                zi_base = lfilter_zi(b, a)
                for ch in range(2):
                    new_zi_solo[s_idx, :, ch] = zi_base * float(first_samples_lr[ch])
            self._zi_solo = new_zi_solo
            return new_zi_solo

        zi_cur = self._zi
        need_full_reinit = (
            self._reset_requested
            or self._was_bypassed
            or zi_cur is None
        )
        if not need_full_reinit and zi_cur.shape == (num_stages, 2, 2):
            return zi_cur

        new_zi = np.zeros((num_stages, 2, 2), dtype=np.float64)
        common = 0 if need_full_reinit or zi_cur is None else min(zi_cur.shape[0], num_stages)
        if common > 0:
            new_zi[:common] = zi_cur[:common]
        for s_idx in range(common, num_stages):
            b, a = stages[s_idx].as_ba()
            zi_base = lfilter_zi(b, a)
            for ch in range(2):
                new_zi[s_idx, :, ch] = zi_base * float(first_samples_lr[ch])

        self._zi = new_zi
        self._was_bypassed = False
        if self._reset_requested:
            self._zi_solo = None
            self._reset_requested = False
        return new_zi

    def _filter_chunk_sos(
        self,
        chunk: np.ndarray,
        sos: np.ndarray,
        zi_state: np.ndarray,
        active_channels: tuple,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Executa filtragem IIR em cascata em uma única chamada C via `sosfilt` (T06)."""
        if len(active_channels) == 2:
            out, zf = sosfilt(sos, chunk, axis=0, zi=zi_state)
            return out, zf
        out = chunk.copy()
        zf = zi_state.copy()
        for ch in active_channels:
            out[:, ch], zf[:, :, ch] = sosfilt(sos, chunk[:, ch], axis=0, zi=zi_state[:, :, ch])
        return out, zf

    def process(self, block: np.ndarray, solo_mode: bool = False) -> np.ndarray:
        """
        Processa um bloco de áudio `[frames]` (mono) ou `[frames, 2]` (estéreo/M-S)
        mantendo continuidade de fase/estado (`zi`/`zf`) entre blocos consecutivos (T01 / T02 / T03 / T06 / T14).
        """
        if block is None or block.size == 0:
            return block

        n_frames = block.shape[0]
        self._freq_smoother.set_target(self.freq_hz)
        self._gain_smoother.set_target(self.gain_db)
        self._q_smoother.set_target(self.q)
        self._mix_smoother.set_target(0.0 if self.bypass else 1.0)

        if self.bypass and not self._mix_smoother.is_smoothing() and not solo_mode:
            self._was_bypassed = True
            return block.copy()

        # Normaliza formato para 2D [frames, 2] internamente
        orig_ndim = block.ndim
        orig_dtype = block.dtype if np.issubdtype(block.dtype, np.floating) else np.float32
        if orig_ndim == 1:
            work = np.column_stack([block, block]).astype(np.float64)
        elif block.shape[1] == 1:
            work = np.column_stack([block[:, 0], block[:, 0]]).astype(np.float64)
        else:
            work = block[:, :2].astype(np.float64, copy=True)

        # Proteção preventiva caso o bloco de entrada já contenha NaN/Inf (T14)
        if not np.all(np.isfinite(work)):
            self._zi = None
            self._zi_solo = None
            self._was_bypassed = True
            clean = np.nan_to_num(work, nan=0.0, posinf=0.0, neginf=0.0)
            if orig_ndim == 1:
                return clean[:, 0].astype(orig_dtype)
            if block.shape[1] == 1:
                return clean[:, :1].astype(orig_dtype)
            return clean.astype(orig_dtype)

        dry = work.copy()

        # Determina quais canais processar conforme ChannelMode (PB22)
        if self.mode == ChannelMode.STEREO:
            active_channels = (0, 1)
        elif self.mode in (ChannelMode.LEFT, ChannelMode.MID):
            active_channels = (0,)
        elif self.mode in (ChannelMode.RIGHT, ChannelMode.SIDE):
            active_channels = (1,)
        else:
            active_channels = (0, 1)

        param_smoothing = (
            self._freq_smoother.is_smoothing()
            or self._gain_smoother.is_smoothing()
            or self._q_smoother.is_smoothing()
        )

        if solo_mode:
            f_curr = self._freq_smoother.get_next_value(n_frames)
            self._gain_smoother.get_next_value(n_frames)
            q_curr = self._q_smoother.get_next_value(n_frames)
            self._mix_smoother.get_next_value(n_frames)

            bp_coeff = BiquadMath.band_pass(self._fs, f_curr, q_curr)
            stages = [bp_coeff]
            sos = bp_coeff.as_sos().reshape(1, 6)
            zi_state = self._ensure_zi(stages, work[0], solo_mode=True)
            work, zf = self._filter_chunk_sos(work, sos, zi_state, active_channels)
            self._zi_solo = zf

        elif param_smoothing:
            # Interpolação em sub-blocos de 32 amostras (T02)
            stages_tup, sos_init = self._filter_bundle
            if not stages_tup or sos_init.shape[0] == 0:
                self.recalculate(self._fs, use_smoothed=True)
                stages_tup, sos_init = self._filter_bundle
            zi_state = self._ensure_zi(list(stages_tup), work[0], solo_mode=False)
            mix_start = self._mix_smoother.current
            mix_end = self._mix_smoother.get_next_value(n_frames)

            for start in range(0, n_frames, self.SUBBLOCK_SIZE):
                end = min(start + self.SUBBLOCK_SIZE, n_frames)
                sub_len = end - start
                self._freq_smoother.get_next_value(sub_len)
                self._gain_smoother.get_next_value(sub_len)
                self._q_smoother.get_next_value(sub_len)
                self.recalculate(self._fs, use_smoothed=True)
                stages_sub_tup, sos_sub = self._filter_bundle
                if zi_state.shape[0] != sos_sub.shape[0]:
                    zi_state = self._ensure_zi(list(stages_sub_tup), work[start], solo_mode=False)
                work[start:end], zi_state = self._filter_chunk_sos(
                    work[start:end], sos_sub, zi_state, active_channels
                )

            self._zi = zi_state
            if mix_start < 0.9999 or mix_end < 0.9999:
                mix_vec = np.linspace(mix_start, mix_end, n_frames, dtype=np.float64)[:, None]
                work = dry * (1.0 - mix_vec) + work * mix_vec
            if self.bypass and not self._mix_smoother.is_smoothing():
                self._was_bypassed = True

        else:
            # Caminho estacionário rápido com leitura atômica de _filter_bundle (T01 / T06)
            stages_tup, sos = self._filter_bundle
            if not stages_tup or sos.shape[0] == 0:
                self.recalculate(self._fs, use_smoothed=False)
                stages_tup, sos = self._filter_bundle
            zi_state = self._ensure_zi(list(stages_tup), work[0], solo_mode=False)
            mix_start = self._mix_smoother.current
            mix_end = self._mix_smoother.get_next_value(n_frames)

            work, zf = self._filter_chunk_sos(work, sos, zi_state, active_channels)
            self._zi = zf

            if mix_start < 0.9999 or mix_end < 0.9999:
                mix_vec = np.linspace(mix_start, mix_end, n_frames, dtype=np.float64)[:, None]
                work = dry * (1.0 - mix_vec) + work * mix_vec
            if self.bypass and not self._mix_smoother.is_smoothing():
                self._was_bypassed = True

        # Proteção pós-filtro contra NaN/Inf nos registradores IIR (T14)
        if not np.all(np.isfinite(work)):
            self._zi = None
            self._zi_solo = None
            self._was_bypassed = True
            work = np.nan_to_num(dry, nan=0.0, posinf=0.0, neginf=0.0)

        if orig_ndim == 1:
            return work[:, 0].astype(orig_dtype)
        if block.shape[1] == 1:
            return work[:, :1].astype(orig_dtype)
        return work.astype(orig_dtype)

    def to_dict(self) -> Dict[str, Any]:
        """Serializa os parâmetros da banda para dicionário compatível com JSON/XML (PB14)."""
        return {
            "index": self.index,
            "freq_hz": float(self.freq_hz),
            "gain_db": float(self.gain_db),
            "q": float(self.q),
            "ftype": self.ftype.name,
            "slope": self.slope.name,
            "mode": self.mode.name,
            "bypass": bool(self.bypass),
            "solo": bool(self.solo),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any], sample_rate: int = 44100) -> "EqBand":
        """Reconstrói uma instância `EqBand` a partir de um dicionário serializado."""
        ftype_val = d.get("ftype", "PEAKING")
        slope_val = d.get("slope", "S12")
        mode_val = d.get("mode", "STEREO")

        ftype = FilterType[ftype_val] if isinstance(ftype_val, str) else FilterType(ftype_val)
        slope = (
            Slope[slope_val]
            if isinstance(slope_val, str) and slope_val in Slope.__members__
            else Slope(int(slope_val))
            if isinstance(slope_val, (int, float))
            else Slope.S12
        )
        mode = ChannelMode[mode_val] if isinstance(mode_val, str) else ChannelMode(mode_val)

        return EqBand(
            index=int(d.get("index", 0)),
            freq_hz=float(d.get("freq_hz", 1000.0)),
            gain_db=float(d.get("gain_db", 0.0)),
            q=float(d.get("q", 0.70710678)),
            ftype=ftype,
            slope=slope,
            mode=mode,
            bypass=bool(d.get("bypass", False)),
            solo=bool(d.get("solo", False)),
            sample_rate=sample_rate,
        )
