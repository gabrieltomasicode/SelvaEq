"""
Módulo de Injeção de Áudio em Tempo Real (`RealtimeMediaStreamer` - `eq/io/media_player.py`).

Módulo 100% independente e desacoplado de UI (respeitando PB28 e arquitetura POO/MVC),
responsável por decodificar arquivos de áudio (`.mp3`, `.wav`, `.flac`, `.ogg`) e
fornecer blocos `[frames, 2]` quadro a quadro dentro do callback de tempo real,
simulando exatamente uma fonte de entrada ao vivo para testar e refinar o equalizador.

Inclui também geradores de sinal de referência em tempo real (`SYNTH_DRONE`, `PINK_NOISE`,
`SINE_SWEEP`) com continuidade de fase entre blocos, reamostragem fora do lock e aquisição
não-bloqueante de lock no callback de áudio (T13).
"""

import os
import shutil
import subprocess
import threading
import wave
from typing import Optional, Tuple
import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter, lfilter_zi, resample_poly

from eq.config import InputSource
from eq.dsp.utils import clamp


class RealtimeMediaStreamer:
    """
    Reprodutor de mídia (MP3/WAV/FLAC/OGG) e gerador de sinais de referência em tempo real.
    Projetado para alimentar o `AudioIO.audio_callback` bloco a bloco (`read_media_block(frames)`)
    como se fosse um fluxo de entrada físico contínuo.
    """

    # Filtro IIR de Paul Kellet para Ruído Rosa (reaproveitado de synth/waveforms.py)
    _PINK_B = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786], dtype=np.float64)
    _PINK_A = np.array([1.0, -2.494956002, 2.017265875, -0.522189400], dtype=np.float64)

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate: int = int(sample_rate)
        self.buffer: Optional[np.ndarray] = None
        self.raw_buffer: Optional[np.ndarray] = None
        self.raw_sample_rate: int = int(sample_rate)
        self.filepath: Optional[str] = None
        self.playhead: int = 0
        self.playing: bool = True
        self.loop: bool = True
        self.gain: float = 0.85
        self._lock = threading.Lock()

        # Estados contínuos dos geradores sintéticos para transição sem clicks entre buffers
        self._drone_phase: float = 0.0
        self._sweep_phase: float = 0.0
        self._sweep_pos: float = 0.0
        self._pink_zi_l: np.ndarray = lfilter_zi(self._PINK_B, self._PINK_A) * 0.0
        self._pink_zi_r: np.ndarray = lfilter_zi(self._PINK_B, self._PINK_A) * 0.0

    @staticmethod
    def _normalize_to_stereo_float32(data: np.ndarray) -> np.ndarray:
        """Converte qualquer array PCM (int8/int16/int32/float) Mono ou Multicanal para `[N, 2] float32`."""
        if np.issubdtype(data.dtype, np.integer):
            info = np.iinfo(data.dtype)
            scale = float(max(abs(info.min), abs(info.max)))
            arr = data.astype(np.float32) / max(scale, 1.0)
        else:
            arr = data.astype(np.float32, copy=False)

        if arr.ndim == 1:
            return np.column_stack([arr, arr]).astype(np.float32)
        if arr.shape[1] == 1:
            return np.column_stack([arr[:, 0], arr[:, 0]]).astype(np.float32)
        return arr[:, :2].astype(np.float32, copy=True)

    @classmethod
    def decode_audio_file(cls, filepath: str, target_sr: int) -> Tuple[np.ndarray, int]:
        """
        Decodifica um arquivo `.mp3`, `.wav`, `.flac` ou `.ogg` para um array estéreo
        `float32` `[frames, 2]`, utilizando uma cadeia resiliente de decodificadores.
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Arquivo de áudio não encontrado: {filepath}")

        ext = os.path.splitext(filepath)[1].lower()
        errors = []

        # 1. Tenta soundfile (suporta WAV, FLAC, OGG e MP3)
        try:
            import soundfile as sf
            data, sr = sf.read(filepath, dtype="float32", always_2d=True)
            if data is not None and len(data) > 0:
                return cls._normalize_to_stereo_float32(data), int(sr)
        except Exception as e:
            errors.append(f"soundfile: {e}")

        # 2. Se for .wav, tenta scipy.io.wavfile e módulo nativo wave
        if ext == ".wav":
            try:
                sr, data = wavfile.read(filepath)
                return cls._normalize_to_stereo_float32(data), int(sr)
            except Exception as e:
                errors.append(f"scipy.io.wavfile: {e}")

            try:
                with wave.open(filepath, "rb") as wf:
                    sr = wf.getframerate()
                    n_ch = wf.getnchannels()
                    sw = wf.getsampwidth()
                    raw = wf.readframes(wf.getnframes())
                dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sw, np.int16)
                samples = np.frombuffer(raw, dtype=dtype)
                if n_ch > 1:
                    samples = samples.reshape(-1, n_ch)
                return cls._normalize_to_stereo_float32(samples), int(sr)
            except Exception as e:
                errors.append(f"wave: {e}")

        # 3. Tenta miniaudio ou pydub
        try:
            import miniaudio
            decoded = miniaudio.decode_file(
                filepath,
                output_format=miniaudio.SampleFormat.FLOAT32,
                nchannels=2,
                sample_rate=int(target_sr),
            )
            arr = np.frombuffer(decoded.samples, dtype=np.float32).reshape(-1, 2)
            return arr.copy(), int(target_sr)
        except Exception as e:
            errors.append(f"miniaudio: {e}")

        try:
            from pydub import AudioSegment
            seg = AudioSegment.from_file(filepath)
            sr = seg.frame_rate
            ch = seg.channels
            samples = np.array(seg.get_array_of_samples())
            if ch > 1:
                samples = samples.reshape(-1, ch)
            return cls._normalize_to_stereo_float32(samples), int(sr)
        except Exception as e:
            errors.append(f"pydub: {e}")

        # 4. Tenta FFmpeg via subprocess (decodifica MP3, WAV, M4A, OGG, FLAC direto para memória)
        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin:
            try:
                cmd = [
                    ffmpeg_bin,
                    "-v", "error",
                    "-i", filepath,
                    "-f", "f32le",
                    "-acodec", "pcm_f32le",
                    "-ac", "2",
                    "-ar", str(int(target_sr)),
                    "-",
                ]
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                arr = np.frombuffer(proc.stdout, dtype=np.float32)
                if arr.size >= 2:
                    stereo = arr[: (arr.size // 2) * 2].reshape(-1, 2).copy()
                    return stereo, int(target_sr)
            except Exception as e:
                errors.append(f"ffmpeg: {e}")

        raise RuntimeError(
            f"Não foi possível decodificar '{os.path.basename(filepath)}'. "
            f"Detalhes: {'; '.join(errors)}"
        )

    def load_file(self, filepath: str, target_sr: Optional[int] = None) -> int:
        """
        Carrega um arquivo `.mp3` ou `.wav` (ou `.flac`/`.ogg`), reamostra para `target_sr`
        fora do lock (T13) e prepara o buffer para streaming em tempo real no callback.
        """
        new_target_sr = int(target_sr) if (target_sr is not None and target_sr > 0) else self.sample_rate
        stereo, sr = self.decode_audio_file(filepath, new_target_sr)

        if int(sr) != int(new_target_sr):
            gcd = int(np.gcd(int(sr), int(new_target_sr)))
            up = int(new_target_sr) // gcd
            down = int(sr) // gcd
            resampled = resample_poly(stereo, up=up, down=down, axis=0).astype(np.float32)
        else:
            resampled = stereo.astype(np.float32, copy=True)

        with self._lock:
            self.sample_rate = new_target_sr
            self.raw_buffer = stereo
            self.raw_sample_rate = int(sr)
            self.buffer = resampled
            self.filepath = filepath
            self.playhead = 0
            self.playing = True

        return len(resampled)

    def set_sample_rate(self, new_sr: int) -> None:
        """
        Atualiza a taxa de amostragem do streamer (PB29), executando a reamostragem
        fora do lock para não bloquear a thread de áudio (T13).
        """
        if new_sr <= 0 or int(new_sr) == self.sample_rate:
            return
        target_sr = int(new_sr)
        cur_time_sec = self.get_position_seconds()
        raw_buf = self.raw_buffer
        raw_sr = self.raw_sample_rate

        resampled: Optional[np.ndarray] = None
        if raw_buf is not None:
            if raw_sr != target_sr:
                gcd = int(np.gcd(raw_sr, target_sr))
                up = target_sr // gcd
                down = raw_sr // gcd
                resampled = resample_poly(raw_buf, up=up, down=down, axis=0).astype(np.float32)
            else:
                resampled = raw_buf.astype(np.float32, copy=True)

        with self._lock:
            self.sample_rate = target_sr
            if resampled is not None:
                self.buffer = resampled
                self.playhead = min(int(cur_time_sec * self.sample_rate), max(0, len(resampled) - 1))

    def play(self) -> None:
        with self._lock:
            self.playing = True

    def pause(self) -> None:
        with self._lock:
            self.playing = False

    def toggle_play_pause(self) -> bool:
        with self._lock:
            self.playing = not self.playing
            return self.playing

    def stop(self) -> None:
        with self._lock:
            self.playing = False
            self.playhead = 0

    def seek(self, seconds: float) -> None:
        """Move a agulha de reprodução para o instante `seconds`."""
        with self._lock:
            if self.buffer is None or len(self.buffer) == 0:
                self.playhead = 0
                return
            frame_idx = int(round(float(seconds) * self.sample_rate))
            self.playhead = int(clamp(frame_idx, 0, len(self.buffer) - 1))

    def seek_normalized(self, ratio: float) -> None:
        """Move a agulha de reprodução para a fração `ratio` em `[0.0, 1.0]`."""
        with self._lock:
            if self.buffer is None or len(self.buffer) == 0:
                self.playhead = 0
                return
            r = clamp(float(ratio), 0.0, 1.0)
            self.playhead = min(int(round(r * len(self.buffer))), len(self.buffer) - 1)

    def set_loop(self, enabled: bool) -> None:
        with self._lock:
            self.loop = bool(enabled)

    def set_gain(self, gain_linear: float) -> None:
        self.gain = float(clamp(gain_linear, 0.0, 4.0))

    def has_file(self) -> bool:
        return self.buffer is not None and len(self.buffer) > 0

    def get_duration_seconds(self) -> float:
        if self.buffer is None or len(self.buffer) == 0 or self.sample_rate <= 0:
            return 0.0
        return float(len(self.buffer)) / float(self.sample_rate)

    def get_position_seconds(self) -> float:
        if self.buffer is None or len(self.buffer) == 0 or self.sample_rate <= 0:
            return 0.0
        return float(self.playhead) / float(self.sample_rate)

    def get_progress_ratio(self) -> float:
        if self.buffer is None or len(self.buffer) == 0:
            return 0.0
        return float(self.playhead) / float(len(self.buffer))

    def read_media_block(self, frames: int) -> np.ndarray:
        """
        Extrai `frames` amostras do arquivo MP3/WAV carregado em tempo real
        usando lock não-bloqueante para respeitar invariantes da thread de áudio (R6 / T13).
        """
        if not self._lock.acquire(blocking=False):
            return np.zeros((frames, 2), dtype=np.float32)
        try:
            if not self.playing or self.buffer is None or len(self.buffer) == 0:
                return np.zeros((frames, 2), dtype=np.float32)

            total = len(self.buffer)
            if self.loop:
                indices = (self.playhead + np.arange(frames)) % total
                self.playhead = int((self.playhead + frames) % total)
                return (self.buffer[indices] * self.gain).astype(np.float32)

            remaining = total - self.playhead
            if remaining <= 0:
                self.playing = False
                return np.zeros((frames, 2), dtype=np.float32)

            if remaining >= frames:
                chunk = self.buffer[self.playhead : self.playhead + frames] * self.gain
                self.playhead += frames
                if self.playhead >= total:
                    self.playing = False
                return chunk.astype(np.float32)

            out = np.zeros((frames, 2), dtype=np.float32)
            out[:remaining] = self.buffer[self.playhead : total] * self.gain
            self.playhead = total
            self.playing = False
            return out
        finally:
            self._lock.release()

    def generate_reference_block(self, frames: int, mode: InputSource) -> np.ndarray:
        """Gera ou extrai um bloco `[frames, 2]` para os modos de teste em tempo real."""
        if not self.playing:
            return np.zeros((frames, 2), dtype=np.float32)

        if mode in (InputSource.MEDIA_FILE, InputSource.WAV_FILE):
            if self.has_file():
                return self.read_media_block(frames)
            mode = InputSource.SYNTH_DRONE

        sr = float(self.sample_rate)

        if mode == InputSource.PINK_NOISE:
            white_l = np.random.uniform(-0.5, 0.5, frames)
            white_r = np.random.uniform(-0.5, 0.5, frames)
            pink_l, self._pink_zi_l = lfilter(self._PINK_B, self._PINK_A, white_l, zi=self._pink_zi_l)
            pink_r, self._pink_zi_r = lfilter(self._PINK_B, self._PINK_A, white_r, zi=self._pink_zi_r)
            scale = 0.35 * self.gain
            return np.column_stack([pink_l * scale, pink_r * scale]).astype(np.float32)

        if mode == InputSource.SINE_SWEEP:
            sweep_period_sec = 6.0
            t_step = 1.0 / sr
            pos = (self._sweep_pos + np.arange(frames) * t_step) % sweep_period_sec
            self._sweep_pos = float((self._sweep_pos + frames * t_step) % sweep_period_sec)
            inst_freqs = 20.0 * np.power(1000.0, pos / sweep_period_sec)
            phase_increments = 2.0 * np.pi * inst_freqs / sr
            cum_phase = self._sweep_phase + np.cumsum(phase_increments)
            self._sweep_phase = float(cum_phase[-1] % (2.0 * np.pi))
            wave_sig = (0.4 * self.gain * np.sin(cum_phase)).astype(np.float32)
            return np.column_stack([wave_sig, wave_sig])

        # Padrão: SYNTH_DRONE (Acorde Musical + Harmônicos de 55 Hz a 11 kHz)
        t_idx = np.arange(frames, dtype=np.float64)
        base_f = 110.0
        phase_inc = 2.0 * np.pi * base_f / sr
        phases = self._drone_phase + t_idx * phase_inc
        self._drone_phase = float((self._drone_phase + frames * phase_inc) % (2.0 * np.pi))

        sig_l = (
            0.22 * np.sin(0.5 * phases)
            + 0.25 * np.sin(1.0 * phases)
            + 0.18 * np.sin(1.5 * phases + 0.3)
            + 0.20 * np.sin(2.0 * phases)
            + 0.15 * np.sin(2.5 * phases + 0.7)
            + 0.16 * np.sin(3.0 * phases)
            + 0.15 * np.sin(4.0 * phases + 0.2)
            + 0.12 * np.sin(6.0 * phases)
            + 0.10 * np.sin(8.0 * phases + 0.5)
            + 0.08 * np.sin(12.0 * phases)
            + 0.07 * np.sin(16.0 * phases + 1.1)
            + 0.06 * np.sin(24.0 * phases)
            + 0.05 * np.sin(32.0 * phases + 0.4)
            + 0.04 * np.sin(48.0 * phases)
            + 0.03 * np.sin(72.0 * phases + 0.9)
            + 0.02 * np.sin(100.0 * phases)
        )
        sig_r = (
            0.22 * np.sin(0.5 * phases)
            + 0.25 * np.sin(1.0 * phases)
            + 0.18 * np.sin(1.5 * phases - 0.3)
            + 0.20 * np.sin(2.0 * phases + 0.15)
            + 0.15 * np.sin(2.5 * phases - 0.5)
            + 0.16 * np.sin(3.0 * phases + 0.25)
            + 0.15 * np.sin(4.0 * phases - 0.2)
            + 0.12 * np.sin(6.0 * phases + 0.4)
            + 0.10 * np.sin(8.0 * phases - 0.5)
            + 0.08 * np.sin(12.0 * phases + 0.3)
            + 0.07 * np.sin(16.0 * phases - 0.8)
            + 0.06 * np.sin(24.0 * phases + 0.6)
            + 0.05 * np.sin(32.0 * phases - 0.4)
            + 0.04 * np.sin(48.0 * phases + 0.5)
            + 0.03 * np.sin(72.0 * phases - 0.9)
            + 0.02 * np.sin(100.0 * phases + 0.7)
        )
        scale = 0.35 * self.gain
        return np.column_stack([sig_l * scale, sig_r * scale]).astype(np.float32)

    @classmethod
    def generate_demo_track_file(cls, filepath: str, duration_sec: float = 4.0, sample_rate: int = 44100) -> str:
        """Gera um arquivo de áudio de demonstração (`.wav` ou `.mp3` via ffmpeg)."""
        n_samples = int(duration_sec * sample_rate)
        t = np.arange(n_samples, dtype=np.float64) / float(sample_rate)
        chord1_mask = (t % duration_sec) < (duration_sec * 0.5)
        f_root = np.where(chord1_mask, 110.0, 146.83)

        harmonics = (
            0.25 * np.sin(2 * np.pi * f_root * t)
            + 0.20 * np.sin(2 * np.pi * (f_root * 1.5) * t)
            + 0.18 * np.sin(2 * np.pi * (f_root * 2.0) * t)
            + 0.12 * np.sin(2 * np.pi * (f_root * 3.0) * t)
            + 0.08 * np.sin(2 * np.pi * (f_root * 4.0) * t)
            + 0.05 * np.sin(2 * np.pi * (f_root * 8.0) * t)
        )
        beat_t = t % 0.5
        kick_env = np.exp(-beat_t * 18.0)
        kick_freq = 55.0 + 90.0 * np.exp(-beat_t * 35.0)
        kick = 0.45 * kick_env * np.sin(2 * np.pi * kick_freq * beat_t)

        hat_t = (t + 0.25) % 0.5
        hat_env = np.exp(-hat_t * 45.0)
        rng = np.random.default_rng(42)
        hat = 0.15 * hat_env * rng.uniform(-1.0, 1.0, size=n_samples)

        left = np.clip(harmonics + kick + hat, -0.95, 0.95).astype(np.float32)
        right = np.clip(np.roll(harmonics, 120) + kick + hat * 0.9, -0.95, 0.95).astype(np.float32)
        stereo = np.column_stack([left, right])

        if filepath.lower().endswith(".mp3") and shutil.which("ffmpeg"):
            tmp_wav = filepath + ".tmp.wav"
            wavfile.write(tmp_wav, sample_rate, stereo)
            try:
                subprocess.run(
                    [shutil.which("ffmpeg"), "-y", "-v", "error", "-i", tmp_wav, "-b:a", "192k", filepath],
                    check=True,
                )
            finally:
                if os.path.exists(tmp_wav):
                    os.remove(tmp_wav)
        else:
            wavfile.write(filepath, sample_rate, stereo)
        return filepath
