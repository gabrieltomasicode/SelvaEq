"""
Motor de Entrada/Saída de Áudio (`AudioIO` / `PyAudioIO` - `eq/io/audio_io.py`).

Refatorado a partir de `synth/audio.py` e `synth/synth.py` (MidiSynth) do SelvaSonic:
- Corrige definitivamente o bug de ausência de áudio em tempo real:
  1. Nos modos de teste (`SYNTH_DRONE`, `MEDIA_FILE` MP3/WAV, `PINK_NOISE`, `SINE_SWEEP`),
     utiliza `sd.OutputStream` diretamente (evitando falhas de permissão ou incompatibilidade
     de canais de microfone mono do Windows/macOS/Linux).
  2. No modo `MICROPHONE` (entrada física ao vivo), detecta dinamicamente a quantidade real
     de canais do dispositivo de entrada (`1` mono ou `2` estéreo) e de saída, convertendo
     automaticamente microfones mono para estéreo `[frames, 2]`.
  3. Caso a placa de áudio rejeite a taxa de amostragem inicial (ex: WASAPI travado em 48 kHz),
     realiza fallback automático para o `default_samplerate` nativo do dispositivo, recalcula
     todos os filtros (`EqEngine.prepare`) e o streamer (`RealtimeMediaStreamer.set_sample_rate`)
     e notifica observadores registrados (`add_sample_rate_listener` - T09).
- Delega a decodificação e streaming em tempo real de arquivos `.mp3`/`.wav` à classe modular
  `RealtimeMediaStreamer` (`eq/io/media_player.py`).
- Expande `visual_queue = Queue(maxsize=32)` para eliminar lacunas de espectro (T10)
  e implementa trava temporal de clipping (`clip_hold_seconds = 1.5`) + medição RMS (PB20 / T08).
"""

import time
from queue import Queue, Full, Empty
from typing import Callable, List, Optional, Tuple, Union
import numpy as np
from scipy.io import wavfile

from eq.config import EqConfig, InputSource
from eq.dsp.engine import EqEngine
from eq.dsp.utils import linear_to_db
from eq.io.media_player import RealtimeMediaStreamer

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except Exception:
    sd = None
    HAS_SOUNDDEVICE = False


# Fila global compartilhada com capacidade para 32 blocos sem descarte entre frames de UI (T10)
visual_queue: Queue = Queue(maxsize=32)


class _DummyStream:
    """Stream simulado para ambientes headless/CI sem placa de som física."""

    def __init__(self, samplerate: int, blocksize: int, channels, callback):
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.channels = channels
        self.callback = callback
        self.active = False

    def start(self) -> None:
        self.active = True

    def stop(self) -> None:
        self.active = False

    def close(self) -> None:
        self.active = False


class AudioIO:
    """
    Gerencia o ciclo de vida do stream de áudio, seleção de dispositivos de Entrada/Saída,
    integração com o `RealtimeMediaStreamer` (MP3/WAV) e callback em tempo real (PB01, PB02, PB20, T08, T09, T10).
    """

    def __init__(
        self,
        config: Optional[EqConfig] = None,
        engine: Optional[EqEngine] = None,
        queue: Optional[Queue] = None,
        media_streamer: Optional[RealtimeMediaStreamer] = None,
    ):
        self.config: EqConfig = config if config is not None else EqConfig()
        self.engine: EqEngine = engine if engine is not None else EqEngine(self.config)
        self.visual_queue: Queue = queue if queue is not None else visual_queue
        self.media_streamer: RealtimeMediaStreamer = (
            media_streamer
            if media_streamer is not None
            else RealtimeMediaStreamer(sample_rate=self.config.sample_rate)
        )
        self.running: bool = False
        self.muted: bool = False
        self.master_gain: float = 0.85
        self.clip_detected: bool = False
        self.clip_hold_seconds: float = 1.5
        self._clip_until_time: float = 0.0
        self.last_peak_dbfs: float = -120.0
        self.last_rms_dbfs: float = -120.0
        self.xrun_count: int = 0
        self.stream = None
        self.stream_mode: str = "dummy"
        self.last_stream_error: Optional[str] = None
        self._sr_listeners: List[Callable[[int], None]] = []

        if self.config.media_file_path:
            try:
                self.load_media_file(self.config.media_file_path)
            except Exception:
                pass

        self._init_stream()

    @property
    def wav_buffer(self) -> Optional[np.ndarray]:
        return self.media_streamer.buffer

    @wav_buffer.setter
    def wav_buffer(self, val: Optional[np.ndarray]) -> None:
        self.media_streamer.buffer = val

    @property
    def loaded_wav_path(self) -> Optional[str]:
        return self.media_streamer.filepath

    def add_sample_rate_listener(self, callback: Callable[[int], None]) -> None:
        """
        Registra um callback `fn(new_sr: int)` notificado quando o hardware altera o sample rate (PB28 / PB29 / T09).
        """
        if callable(callback) and callback not in self._sr_listeners:
            self._sr_listeners.append(callback)

    def reset_clip(self) -> None:
        """Reseta manualmente a trava temporal do indicador de clipping (V07 / T08)."""
        self._clip_until_time = 0.0
        self.clip_detected = False

    @staticmethod
    def list_output_devices() -> List[Tuple[Optional[int], str]]:
        """Retorna lista `[(device_index, label_formatado)]` dos dispositivos de saída disponíveis."""
        items: List[Tuple[Optional[int], str]] = [(None, "Padrão do Sistema (Saída)")]
        if not HAS_SOUNDDEVICE or sd is None:
            return items
        try:
            devices = sd.query_devices()
            for idx, dev in enumerate(devices):
                max_out = int(dev.get("max_output_channels", 0))
                if max_out > 0:
                    sr = int(dev.get("default_samplerate", 44100))
                    items.append((idx, f"[{idx}] {dev['name']} ({max_out}ch, {sr}Hz)"))
        except Exception:
            pass
        return items

    @staticmethod
    def list_input_devices() -> List[Tuple[Optional[int], str]]:
        """Retorna lista `[(device_index, label_formatado)]` dos dispositivos de entrada disponíveis."""
        items: List[Tuple[Optional[int], str]] = [(None, "Padrão do Sistema (Entrada)")]
        if not HAS_SOUNDDEVICE or sd is None:
            return items
        try:
            devices = sd.query_devices()
            for idx, dev in enumerate(devices):
                max_in = int(dev.get("max_input_channels", 0))
                if max_in > 0:
                    sr = int(dev.get("default_samplerate", 44100))
                    items.append((idx, f"[{idx}] {dev['name']} ({max_in}ch, {sr}Hz)"))
        except Exception:
            pass
        return items

    def set_input_device(self, device_id: Optional[Union[int, str]]) -> None:
        """Define o dispositivo de entrada (microfone/linha) e reinicia o stream se necessário."""
        self.config.input_device = device_id
        if self.config.input_source == InputSource.MICROPHONE:
            self.restart_stream()

    def set_output_device(self, device_id: Optional[Union[int, str]]) -> None:
        """Define o dispositivo de saída (alto-falantes/fones) e reinicia o stream imediatamente."""
        self.config.output_device = device_id
        self.restart_stream()

    def _query_device_caps(self, dev_id, kind: str) -> Tuple[int, int]:
        """Retorna `(max_channels, default_samplerate)` do dispositivo ou `(2, config.sample_rate)`."""
        if not HAS_SOUNDDEVICE or sd is None:
            return 2, self.config.sample_rate
        try:
            info = sd.query_devices(dev_id, kind=kind)
            ch_key = "max_input_channels" if kind == "input" else "max_output_channels"
            max_ch = int(info.get(ch_key, 2))
            def_sr = int(round(float(info.get("default_samplerate", self.config.sample_rate))))
            return max(1, min(max_ch, 2)), def_sr
        except Exception:
            return 2, self.config.sample_rate

    def _apply_fallback_samplerate(self, fallback_sr: int) -> None:
        """
        Sincroniza `config`, `engine`, `media_streamer` e observadores registrados
        caso o hardware exija outra taxa de amostragem (PB28 / PB29 / T09).
        """
        if fallback_sr > 0 and fallback_sr != self.config.sample_rate:
            self.config.sample_rate = int(fallback_sr)
            self.engine.prepare(self.config.sample_rate, self.config.buffer_size)
            self.media_streamer.set_sample_rate(self.config.sample_rate)
            for cb in list(self._sr_listeners):
                try:
                    cb(self.config.sample_rate)
                except Exception:
                    pass

    def _init_stream(self) -> None:
        """
        Inicializa o stream PortAudio (`sounddevice`) com proteção contra diferenças de canais
        (microfone mono vs saída estéreo) e fallback automático de `sample_rate` nativo da placa.
        """
        self.last_stream_error = None
        if HAS_SOUNDDEVICE and sd is not None:
            out_ch, out_def_sr = self._query_device_caps(self.config.output_device, "output")

            # 1. Modo Entrada ao Vivo (MICROPHONE) -> Tenta Stream Full-Duplex (in_ch -> out_ch)
            if self.config.input_source == InputSource.MICROPHONE:
                in_ch, in_def_sr = self._query_device_caps(self.config.input_device, "input")
                for candidate_sr in (self.config.sample_rate, out_def_sr, in_def_sr, 48000, 44100):
                    try:
                        self.stream = sd.Stream(
                            samplerate=candidate_sr,
                            blocksize=self.config.buffer_size,
                            device=(self.config.input_device, self.config.output_device),
                            channels=(in_ch, out_ch),
                            dtype="float32",
                            callback=self._duplex_callback,
                        )
                        self._apply_fallback_samplerate(candidate_sr)
                        self.stream_mode = f"Duplex ({in_ch}ch In -> {out_ch}ch Out @ {candidate_sr}Hz)"
                        print(f"🎙️ [SelvaEQ] {self.stream_mode}")
                        return
                    except Exception as e:
                        self.last_stream_error = str(e)

            # 2. Modo Teste MP3/WAV / Gerador Interno (ou fallback se não houver microfone) -> OutputStream
            for candidate_sr in (self.config.sample_rate, out_def_sr, 48000, 44100):
                try:
                    self.stream = sd.OutputStream(
                        samplerate=candidate_sr,
                        blocksize=self.config.buffer_size,
                        device=self.config.output_device,
                        channels=out_ch,
                        dtype="float32",
                        callback=self._output_only_callback,
                    )
                    self._apply_fallback_samplerate(candidate_sr)
                    self.stream_mode = f"OutputStream ({out_ch}ch Out @ {candidate_sr}Hz - {self.config.input_source.name})"
                    print(f"🔊 [SelvaEQ] {self.stream_mode}")
                    return
                except Exception as e:
                    self.last_stream_error = str(e)

        self.stream = _DummyStream(
            samplerate=self.config.sample_rate,
            blocksize=self.config.buffer_size,
            channels=2,
            callback=self.audio_callback,
        )
        self.stream_mode = "Simulado (Headless / Sem Dispositivo)"

    def _output_only_callback(self, outdata: np.ndarray, frames: int, time_info, status) -> None:
        """
        Callback do `sd.OutputStream`: obtém o bloco em tempo real do `RealtimeMediaStreamer`
        (arquivo MP3/WAV ou sinal de referência) e injeta no `audio_callback` exatamente como
        uma entrada em tempo real.
        """
        if self.muted or not self.media_streamer.playing:
            outdata.fill(0.0)
            return
        in_block = self.media_streamer.generate_reference_block(frames, self.config.input_source)
        in_block = (in_block * self.master_gain).astype(np.float32, copy=False)
        self.audio_callback(in_block, outdata, frames, time_info, status)

    def _duplex_callback(self, indata: np.ndarray, outdata: np.ndarray, frames: int, time_info, status) -> None:
        """Callback do `sd.Stream` full-duplex (converte entrada mono para estéreo se necessário)."""
        if self.muted:
            outdata.fill(0.0)
            return
        if indata is not None and indata.ndim == 2 and indata.shape[1] == 1:
            stereo_in = np.column_stack([indata[:, 0], indata[:, 0]]) * self.master_gain
        elif indata is not None:
            stereo_in = indata * self.master_gain
        else:
            stereo_in = np.zeros((frames, 2), dtype=np.float32)
        self.audio_callback(stereo_in.astype(np.float32, copy=False), outdata, frames, time_info, status)

    def audio_callback(
        self,
        indata: np.ndarray,
        outdata: np.ndarray,
        frames: int,
        time_info,
        status,
    ) -> None:
        """
        Callback central de processamento de áudio em tempo real (PB01, PB02, PB20, T08, T10).
        """
        if status:
            self.xrun_count += 1

        try:
            if indata is None or indata.size == 0:
                in_block = self.media_streamer.generate_reference_block(frames, self.config.input_source)
            else:
                in_block = indata

            processed = self.engine.process(in_block)
            if processed.ndim == 1:
                outdata[:frames, 0] = processed[:frames]
                if outdata.shape[1] > 1:
                    outdata[:frames, 1] = processed[:frames]
            elif outdata.shape[1] == 1:
                outdata[:frames, 0] = np.mean(processed[:frames, :2], axis=1)
            else:
                ch = min(processed.shape[1], outdata.shape[1])
                outdata[:frames, :ch] = processed[:frames, :ch]

            self.last_peak_dbfs = self.peak_dbfs(outdata)
            self.last_rms_dbfs = self.rms_dbfs(outdata)

            # Latch temporal de clipping por clip_hold_seconds (T08)
            now = time.monotonic()
            if self.is_clipping(outdata):
                self._clip_until_time = now + self.clip_hold_seconds
                self.clip_detected = True
            else:
                self.clip_detected = bool(now < self._clip_until_time)

            try:
                if self.visual_queue.full():
                    try:
                        self.visual_queue.get_nowait()
                    except Empty:
                        pass
                self.visual_queue.put_nowait(outdata.copy())
            except Full:
                pass

        except Exception as e:
            self.last_stream_error = str(e)
            outdata.fill(0.0)

    def load_media_file(self, filepath: str) -> int:
        """
        Carrega um arquivo `.mp3` ou `.wav` (ou `.flac`/`.ogg`) no `RealtimeMediaStreamer`
        e ativa automaticamente o modo `InputSource.MEDIA_FILE` para escuta imediata.
        """
        n_frames = self.media_streamer.load_file(filepath, target_sr=self.config.sample_rate)
        self.config.media_file_path = filepath
        self.set_input_source(InputSource.MEDIA_FILE)
        return n_frames

    def load_wav_source(self, filepath: str) -> int:
        """Alias de compatibilidade retroativa para `load_media_file`."""
        return self.load_media_file(filepath)

    def set_input_source(self, source: Union[InputSource, str]) -> None:
        """Altera a fonte de entrada de áudio e reinicia o stream se alternar de/para `MICROPHONE`."""
        if isinstance(source, InputSource):
            new_src = source
        else:
            key = str(source).upper()
            new_src = InputSource[key] if key in InputSource.__members__ else InputSource.SYNTH_DRONE
        old_src = self.config.input_source
        self.config.input_source = new_src
        self.media_streamer.play()
        if (old_src == InputSource.MICROPHONE) != (new_src == InputSource.MICROPHONE):
            self.restart_stream()

    def play_media(self) -> None:
        self.media_streamer.play()

    def pause_media(self) -> None:
        self.media_streamer.pause()

    def toggle_play_pause_media(self) -> bool:
        return self.media_streamer.toggle_play_pause()

    def stop_media(self) -> None:
        self.media_streamer.stop()

    def seek_media(self, seconds: float) -> None:
        self.media_streamer.seek(seconds)

    def set_media_loop(self, enabled: bool) -> None:
        self.media_streamer.set_loop(enabled)

    @staticmethod
    def peak_dbfs(block: np.ndarray) -> float:
        """Calcula o nível de pico do bloco em decibéis Full Scale (dBFS) (PB20)."""
        if block is None or block.size == 0:
            return -120.0
        peak_lin = float(np.max(np.abs(block)))
        return float(linear_to_db(peak_lin, min_db=-120.0))

    @staticmethod
    def rms_dbfs(block: np.ndarray) -> float:
        """Calcula o nível RMS do bloco em decibéis Full Scale (dBFS) (V07 / T08)."""
        if block is None or block.size == 0:
            return -120.0
        rms_lin = float(np.sqrt(np.mean(np.square(block, dtype=np.float64))))
        return float(linear_to_db(rms_lin, min_db=-120.0))

    @staticmethod
    def is_clipping(block: np.ndarray, threshold_dbfs: float = 0.0) -> bool:
        """Verifica se o bloco ultrapassa o limiar de clipping digital (> 0 dBFS / amplitude > 1.0) (PB20)."""
        if block is None or block.size == 0:
            return False
        limit_linear = float(np.power(10.0, threshold_dbfs / 20.0))
        return bool(np.max(np.abs(block)) > limit_linear)

    def start(self) -> None:
        """Inicia o stream de áudio."""
        if self.stream is not None:
            self.stream.start()
        self.running = True

    def stop(self) -> None:
        """Interrompe o stream de áudio."""
        if self.stream is not None:
            try:
                self.stream.stop()
            except Exception:
                pass
            if hasattr(self.stream, "close"):
                try:
                    self.stream.close()
                except Exception:
                    pass
        self.running = False

    def restart_stream(self) -> None:
        """Reinicia o stream de áudio após mudança de parâmetros de hardware ou fonte."""
        was_running = self.running
        if self.running:
            self.stop()
        self.engine.prepare(self.config.sample_rate, self.config.buffer_size)
        self.media_streamer.set_sample_rate(self.config.sample_rate)
        self._init_stream()
        if was_running:
            self.start()

    def process_wav_file(self, input_wav_path: str, output_wav_path: Optional[str] = None) -> np.ndarray:
        """Processa um arquivo de áudio (.wav ou .mp3) offline passando pelo `audio_callback`."""
        stereo, sr = RealtimeMediaStreamer.decode_audio_file(input_wav_path, self.config.sample_rate)
        if sr != self.config.sample_rate:
            self.config.sample_rate = int(sr)
            self.engine.prepare(self.config.sample_rate, self.config.buffer_size)

        total_frames = stereo.shape[0]
        output = np.zeros_like(stereo, dtype=np.float32)
        bsize = self.config.buffer_size

        for start_idx in range(0, total_frames, bsize):
            end_idx = min(start_idx + bsize, total_frames)
            chunk = stereo[start_idx:end_idx]
            out_chunk = np.zeros_like(chunk, dtype=np.float32)
            self.audio_callback(chunk, out_chunk, len(chunk), None, None)
            output[start_idx:end_idx] = out_chunk

        if output_wav_path:
            wavfile.write(output_wav_path, self.config.sample_rate, output)
        return output

    def __enter__(self) -> "AudioIO":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


# Alias explícito conforme seção 2.1 do documento de planejamento
PyAudioIO = AudioIO
