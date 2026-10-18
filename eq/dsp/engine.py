"""
Orquestrador Central de DSP (`EqEngine` / `PyEqEngine`).

Implementa o fluxo completo de processamento de sinal (PB34):
Entrada L/R -> Oversampling UP opcional com estado FIR contínuo (PB11 / T04) ->
Codificação Mid/Side (PB22) -> Cascata de até 24 Bandas (PB04) com resolução de
Solo/Bypass (PB12 / T03), suavização em sub-blocos (T02) e fusão SOS (T06) ->
Decodificação Mid/Side (PB22) -> Oversampling DOWN opcional (PB11 / T04) -> Ganho de Saída (F13).
Mantém isolamento arquitetural estrito sem nenhuma dependência de UI (PB28)
e segurança concorrente lock-free para a thread de áudio (T01).
"""

import threading
from typing import List, Optional, Any, Tuple, Sequence
import numpy as np
from scipy.signal import resample_poly, firwin, lfilter, sosfilt

from eq.config import EqConfig, FilterType, ChannelMode, Slope
from eq.dsp.band import EqBand
from eq.dsp.biquad_math import BiquadMath
from eq.dsp.utils import db_to_linear, clamp
from eq.dsp.state_controller import StateController


class _StatefulOversampler:
    """
    Reamostrador polifásico/FIR com preservação de estado (`zi`) entre blocos (PB11 / T04).
    Elimina as descontinuidades de borda do `resample_poly` bloco a bloco.
    Latência total constante de 32 amostras na taxa base para 2x e 4x.
    """

    def __init__(self, factor: int = 2):
        self.factor: int = int(factor)
        if self.factor > 1:
            self.num_taps: int = 32 * self.factor + 1
            cutoff = 0.85 / float(self.factor)
            base_fir = firwin(self.num_taps, cutoff, window=("kaiser", 8.6)).astype(np.float64)
            # Normaliza ganho DC unitário no filtro de decimação e ganho = factor na interpolação
            base_fir /= float(np.sum(base_fir))
            self._h_up: np.ndarray = base_fir * float(self.factor)
            self._h_down: np.ndarray = base_fir
            self._zi_up: np.ndarray = np.zeros((self.num_taps - 1, 2), dtype=np.float64)
            self._zi_down: np.ndarray = np.zeros((self.num_taps - 1, 2), dtype=np.float64)
        else:
            self.num_taps = 1
            self._h_up = np.array([1.0], dtype=np.float64)
            self._h_down = np.array([1.0], dtype=np.float64)
            self._zi_up = np.zeros((0, 2), dtype=np.float64)
            self._zi_down = np.zeros((0, 2), dtype=np.float64)

    @property
    def latency_samples(self) -> int:
        return 32 if self.factor > 1 else 0

    def reset(self) -> None:
        if self.factor > 1:
            self._zi_up.fill(0.0)
            self._zi_down.fill(0.0)

    def upsample(self, stereo: np.ndarray) -> np.ndarray:
        if self.factor <= 1:
            return stereo
        n_frames = stereo.shape[0]
        up = np.zeros((n_frames * self.factor, 2), dtype=np.float64)
        up[:: self.factor, :] = stereo
        out, self._zi_up = lfilter(self._h_up, [1.0], up, axis=0, zi=self._zi_up)
        return out

    def downsample(self, work: np.ndarray) -> np.ndarray:
        if self.factor <= 1:
            return work
        filtered, self._zi_down = lfilter(self._h_down, [1.0], work, axis=0, zi=self._zi_down)
        return filtered[:: self.factor, :]


class EqEngine:
    """
    Motor central de equalização paramétrica de até 24 bandas (Modelo DSP).
    """

    MAX_BANDS = 24

    def __init__(self, config: Optional[EqConfig] = None, init_default_bands: bool = False):
        self.config: EqConfig = config if config is not None else EqConfig()
        self._edit_lock = threading.RLock()
        self.bands: List[EqBand] = []
        self._oversampler: _StatefulOversampler = _StatefulOversampler(self.config.oversampling)
        self.state_controller: StateController = StateController(self)

        if init_default_bands:
            self._create_default_bands()

    @property
    def latency_samples(self) -> int:
        """Retorna a latência total introduzida pelo oversampling na taxa base (T04)."""
        if self._oversampler.factor != self.config.oversampling:
            self._oversampler = _StatefulOversampler(self.config.oversampling)
        return self._oversampler.latency_samples

    def _create_default_bands(self) -> None:
        """Inicializa um conjunto padrão de bandas espaçadas logaritmicamente."""
        default_specs = [
            (30.0, 0.0, 0.707, FilterType.LOW_CUT, Slope.S12),
            (100.0, 0.0, 0.707, FilterType.LOW_SHELF, Slope.S12),
            (350.0, 0.0, 1.0, FilterType.PEAKING, Slope.S12),
            (1000.0, 0.0, 1.0, FilterType.PEAKING, Slope.S12),
            (3500.0, 0.0, 1.0, FilterType.PEAKING, Slope.S12),
            (10000.0, 0.0, 0.707, FilterType.HIGH_SHELF, Slope.S12),
            (18000.0, 0.0, 0.707, FilterType.HIGH_CUT, Slope.S12),
        ]
        for freq, gain, q, ftype, slope in default_specs:
            self.add_band(freq_hz=freq, gain_db=gain, q=q, ftype=ftype, slope=slope)

    def set_bands(self, bands: List[EqBand]) -> None:
        """Substitui atomicamente a lista de bandas do motor com reindexação segura (T01)."""
        with self._edit_lock:
            new_list = list(bands)
            for idx, b in enumerate(new_list):
                b.index = idx
            self.bands = new_list

    def flatten_bands(self) -> None:
        """Zera o ganho (0.0 dB) e desativa solo e bypass em todas as bandas (F07)."""
        effective_fs = self.config.sample_rate * max(1, self.config.oversampling)
        with self._edit_lock:
            for band in self.bands:
                band._fs = effective_fs
                band.solo = False
                band.set_targets(gain_db=0.0, bypass=False)

    def estimate_auto_gain_db(self) -> float:
        """
        Calcula a compensação automática de ganho ponderada pelas bandas ativas (F13).
        Limitada ao intervalo [-12.0, +12.0] dB.
        """
        bands_snapshot = tuple(self.bands)
        active = [b for b in bands_snapshot if not b.bypass and not b.solo]
        if not active:
            return 0.0
        weighted_sum = 0.0
        for b in active:
            if b.ftype in (FilterType.PEAKING, FilterType.LOW_SHELF, FilterType.HIGH_SHELF):
                weight = (
                    0.5
                    if b.ftype in (FilterType.LOW_SHELF, FilterType.HIGH_SHELF)
                    else clamp(1.0 / max(b.q, 0.25), 0.15, 1.0)
                )
                weighted_sum += b.gain_db * weight
        return float(clamp(-0.5 * weighted_sum, -12.0, 12.0))

    def prepare(self, fs: int, block_size: int) -> None:
        """
        Prepara o motor para uma nova taxa de amostragem `fs` e tamanho de bloco `block_size` (PB29).
        Recalcula os coeficientes de todas as bandas para o Sample Rate efetivo (considerando oversampling).
        """
        with self._edit_lock:
            self.config.sample_rate = int(fs)
            self.config.buffer_size = int(block_size)
            self.config.validate()
            self._oversampler = _StatefulOversampler(self.config.oversampling)
            effective_fs = self.config.sample_rate * self.config.oversampling
            for band in self.bands:
                band.recalculate(effective_fs, use_smoothed=False)
                band.reset_state()

    def add_band(
        self,
        freq_hz: float = 1000.0,
        gain_db: float = 0.0,
        q: float = 0.70710678,
        ftype: FilterType = FilterType.PEAKING,
        slope: Slope = Slope.S12,
        mode: ChannelMode = ChannelMode.STEREO,
        bypass: bool = False,
        solo: bool = False,
    ) -> EqBand:
        """
        Aloca e adiciona uma nova banda ao motor usando copy-on-write, respeitando o limite de 24 bandas (PB04 / T01).
        """
        with self._edit_lock:
            if len(self.bands) >= self.config.num_bands or len(self.bands) >= self.MAX_BANDS:
                raise ValueError(f"Limite máximo de {self.config.num_bands} bandas atingido (PB04).")

            effective_fs = self.config.sample_rate * self.config.oversampling
            band = EqBand(
                index=len(self.bands),
                freq_hz=freq_hz,
                gain_db=gain_db,
                q=q,
                ftype=ftype,
                slope=slope,
                mode=mode,
                bypass=bypass,
                solo=solo,
                sample_rate=effective_fs,
            )
            new_bands = list(self.bands)
            new_bands.append(band)
            self.bands = new_bands
            return band

    def remove_band(self, index: int) -> None:
        """Remove a banda no índice especificado usando copy-on-write e reindexa as restantes (T01)."""
        with self._edit_lock:
            if not (0 <= index < len(self.bands)):
                raise IndexError(f"Índice de banda inválido: {index}")
            new_bands = list(self.bands)
            new_bands.pop(index)
            for idx, b in enumerate(new_bands):
                b.index = idx
            self.bands = new_bands

    def set_param(self, index: int, name: str, value: Any) -> None:
        """
        Atualiza dinamicamente um parâmetro da banda `index` (PB03 / PB06 / T02).
        Para `freq_hz`, `gain_db` e `q`, define apenas os alvos nos suavizadores para que
        `EqBand.process()` interpole os coeficientes em sub-blocos de 32 amostras sem degrau (T02).
        """
        bands_ref = self.bands
        if not (0 <= index < len(bands_ref)):
            raise IndexError(f"Índice de banda fora do intervalo: {index}")
        band = bands_ref[index]
        effective_fs = self.config.sample_rate * self.config.oversampling

        if name == "freq_hz":
            if band._fs != effective_fs:
                band.recalculate(effective_fs, use_smoothed=True)
            band.set_targets(freq_hz=float(value))
        elif name == "gain_db":
            if band._fs != effective_fs:
                band.recalculate(effective_fs, use_smoothed=True)
            band.set_targets(gain_db=float(value))
        elif name == "q":
            if band._fs != effective_fs:
                band.recalculate(effective_fs, use_smoothed=True)
            band.set_targets(q=float(value))
        elif name == "ftype":
            band.ftype = value if isinstance(value, FilterType) else FilterType[str(value).upper()]
            band.recalculate(effective_fs, use_smoothed=False)
        elif name == "slope":
            if isinstance(value, Slope):
                band.slope = value
            elif isinstance(value, int):
                band.slope = Slope(value)
            else:
                s_str = str(value).upper()
                band.slope = Slope[s_str] if s_str.startswith("S") else Slope(int(s_str))
            band.recalculate(effective_fs, use_smoothed=False)
        elif name == "mode":
            band.mode = value if isinstance(value, ChannelMode) else ChannelMode[str(value).upper()]
        elif name == "bypass":
            band.set_targets(bypass=bool(value))
        elif name == "solo":
            band.solo = bool(value)
        else:
            raise AttributeError(f"Parâmetro desconhecido em EqBand: {name}")

    def _active_bands(self, bands_snapshot: Optional[Sequence[EqBand]] = None) -> List[EqBand]:
        """
        Resolve quais bandas devem processar o sinal no ciclo atual (PB12 / T01 / T03).
        Se houver qualquer banda em `solo` (e não em bypass), retorna apenas as bandas em solo.
        Caso contrário, retorna todas as bandas que não estão em bypass (ou que ainda estão suavizando),
        marcando `_was_bypassed = True` nas bandas inativas em bypass (T03).
        """
        source = bands_snapshot if bands_snapshot is not None else tuple(self.bands)
        solo_bands = [b for b in source if b.solo and not b.bypass]
        if solo_bands:
            return solo_bands
        active: List[EqBand] = []
        for b in source:
            if not b.bypass or b._mix_smoother.is_smoothing():
                active.append(b)
            else:
                b._was_bypassed = True
        return active

    @staticmethod
    def _encode_mid_side(lr: np.ndarray) -> np.ndarray:
        """
        Codifica matriz Estéreo [L, R] em Mid/Side [M, S] (PB22):
            M = (L + R) / 2
            S = (L - R) / 2
        """
        ms = np.empty_like(lr, dtype=np.float64)
        ms[:, 0] = 0.5 * (lr[:, 0] + lr[:, 1])
        ms[:, 1] = 0.5 * (lr[:, 0] - lr[:, 1])
        return ms

    @staticmethod
    def _decode_mid_side(ms: np.ndarray) -> np.ndarray:
        """
        Decodifica matriz Mid/Side [M, S] de volta para Estéreo [L, R] (PB22):
            L = M + S
            R = M - S
        """
        lr = np.empty_like(ms, dtype=np.float64)
        lr[:, 0] = ms[:, 0] + ms[:, 1]
        lr[:, 1] = ms[:, 0] - ms[:, 1]
        return lr

    @staticmethod
    def _try_get_fusable_state(
        band: EqBand, effective_fs: int, any_solo: bool
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Captura atomicamente `(sos, zi)` de uma banda se ela estiver em regime estacionário
        STEREO e com dimensões `sos` e `zi` consistentes (T01 / T06).
        """
        if any_solo or band.bypass or band.mode != ChannelMode.STEREO:
            return None
        if band._fs != effective_fs or band.is_smoothing():
            return None
        _, sos = band._filter_bundle
        n_sections = sos.shape[0]
        if n_sections == 0:
            return None
        zi = band._zi
        if (
            zi is None
            or band._reset_requested
            or band._was_bypassed
            or zi.shape != (n_sections, 2, 2)
        ):
            return None
        return sos, zi

    @classmethod
    def _process_band_sequence(
        cls,
        work: np.ndarray,
        active: List[EqBand],
        effective_fs: int,
        any_solo: bool,
        allow_ms_per_band: bool,
    ) -> np.ndarray:
        """
        Processa a sequência de bandas ativas fundindo lotes consecutivos de bandas
        estacionárias em uma única chamada C `sosfilt` com referências locais seguras (T01 / T06).
        """
        idx = 0
        n_active = len(active)
        while idx < n_active:
            band = active[idx]
            first_pair = cls._try_get_fusable_state(band, effective_fs, any_solo)
            if first_pair is not None:
                batch_items: List[Tuple[EqBand, np.ndarray, np.ndarray]] = [
                    (band, first_pair[0], first_pair[1])
                ]
                j = idx + 1
                while j < n_active:
                    nxt = active[j]
                    nxt_pair = cls._try_get_fusable_state(nxt, effective_fs, any_solo)
                    if nxt_pair is None:
                        break
                    batch_items.append((nxt, nxt_pair[0], nxt_pair[1]))
                    j += 1

                if len(batch_items) == 1:
                    b0, sos0, zi0 = batch_items[0]
                    work, zf = sosfilt(sos0, work, axis=0, zi=zi0)
                    b0._zi = zf
                else:
                    fused_sos = np.concatenate([item[1] for item in batch_items], axis=0)
                    fused_zi = np.concatenate([item[2] for item in batch_items], axis=0)
                    work, fused_zf = sosfilt(fused_sos, work, axis=0, zi=fused_zi)
                    offset = 0
                    for b_item, sos_item, _ in batch_items:
                        sec = sos_item.shape[0]
                        b_item._zi = fused_zf[offset : offset + sec]
                        offset += sec
                idx = j
                continue

            if band._fs != effective_fs:
                band.recalculate(effective_fs, use_smoothed=True)
            if allow_ms_per_band and band.mode in (ChannelMode.MID, ChannelMode.SIDE):
                ms_block = cls._encode_mid_side(work)
                ms_block = band.process(ms_block, solo_mode=any_solo)
                work = cls._decode_mid_side(ms_block)
            else:
                work = band.process(work, solo_mode=any_solo)
            idx += 1
        return work

    def process(self, block: np.ndarray) -> np.ndarray:
        """
        Executa um ciclo completo de processamento de áudio sobre `block` (PB34 / `07_atividade_processblock.puml`).

        Suporta entrada 1D `[frames]` (mono) ou 2D `[frames, 2]` (estéreo).
        Lock-free em relação à UI via captura de snapshot imutável de `self.bands` (T01).
        """
        if block is None or block.size == 0:
            return block

        if self.config.bypass:
            self.config.last_audio_buffer = block.copy()
            return block.copy()

        orig_ndim = block.ndim
        orig_dtype = block.dtype if np.issubdtype(block.dtype, np.floating) else np.float32
        if orig_ndim == 1:
            stereo = np.column_stack([block, block]).astype(np.float64)
        elif block.shape[1] == 1:
            stereo = np.column_stack([block[:, 0], block[:, 0]]).astype(np.float64)
        else:
            stereo = block[:, :2].astype(np.float64, copy=True)

        if not np.all(np.isfinite(stereo)):
            stereo = np.nan_to_num(stereo, nan=0.0, posinf=0.0, neginf=0.0)

        n_frames = stereo.shape[0]
        os_factor = int(self.config.oversampling)
        effective_fs = int(self.config.sample_rate * os_factor)

        if self._oversampler.factor != os_factor:
            self._oversampler = _StatefulOversampler(os_factor)

        # 1. Oversampling UP opcional com estado contínuo entre blocos (PB11 / T04)
        if os_factor > 1:
            work = self._oversampler.upsample(stereo)
        else:
            work = stereo

        # 2. Captura snapshot imutável de bandas para segurança concorrente lock-free (T01)
        bands_snapshot = tuple(self.bands)
        any_solo = any(b.solo and not b.bypass for b in bands_snapshot)
        active = self._active_bands(bands_snapshot)

        # 3. Processamento em cascata das bandas ativas com suporte a L/R, M/S e fusão SOS (PB22 / T06)
        if self.config.ms_enabled:
            work = self._encode_mid_side(work)
            work = self._process_band_sequence(
                work, active, effective_fs, any_solo, allow_ms_per_band=False
            )
            work = self._decode_mid_side(work)
        else:
            work = self._process_band_sequence(
                work, active, effective_fs, any_solo, allow_ms_per_band=True
            )

        # 4. Oversampling DOWN opcional com estado contínuo entre blocos (PB11 / T04)
        if os_factor > 1:
            work = self._oversampler.downsample(work)
            if work.shape[0] > n_frames:
                work = work[:n_frames, :]
            elif work.shape[0] < n_frames:
                pad = np.zeros((n_frames - work.shape[0], 2), dtype=np.float64)
                work = np.vstack([work, pad])

        # 5. Ganho de saída global + Auto-Gain opcional (F13)
        total_gain_db = float(self.config.output_gain_db)
        if getattr(self.config, "auto_gain", False):
            total_gain_db += self.estimate_auto_gain_db()
        if abs(total_gain_db) > 1e-6:
            work *= float(db_to_linear(total_gain_db))

        if not np.all(np.isfinite(work)):
            self.reset_all()
            work = np.zeros_like(stereo)

        out = work[:, 0].astype(orig_dtype) if orig_ndim == 1 else (
            work[:, :1].astype(orig_dtype) if block.shape[1] == 1 else work.astype(orig_dtype)
        )
        self.config.last_audio_buffer = out
        return out

    def global_response(
        self, n: int = 2048, freqs: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retorna `(freqs_hz, mag_db)` representando a curva teórica global do EQ
        avaliada na taxa de amostragem efetiva `sample_rate * oversampling` (PB09 / T05 / F13).
        """
        if freqs is None:
            freqs_hz = np.geomspace(20.0, min(20000.0, 0.499 * self.config.sample_rate), num=n)
        else:
            freqs_hz = np.asarray(freqs, dtype=np.float64)

        if self.config.bypass:
            return freqs_hz, np.zeros_like(freqs_hz, dtype=np.float64)

        effective_fs = int(self.config.sample_rate * max(1, self.config.oversampling))
        mag_db = BiquadMath.combined_response(
            list(self.bands), fs=effective_fs, n=len(freqs_hz), freqs=freqs_hz
        )
        total_gain_db = float(self.config.output_gain_db)
        if getattr(self.config, "auto_gain", False):
            total_gain_db += self.estimate_auto_gain_db()
        if abs(total_gain_db) > 1e-6:
            mag_db = mag_db + total_gain_db
        return freqs_hz, mag_db

    def reset_all(self) -> None:
        """Reseta o estado interno (`zi`) de todas as bandas e do oversampler (T01 / T04)."""
        self._oversampler.reset()
        for band in tuple(self.bands):
            band.reset_state()


# Alias explícito conforme seção 2.1 do documento de planejamento
PyEqEngine = EqEngine
PyBiquadMath = BiquadMath
