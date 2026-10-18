"""
Testes Unitários da Camada de Infraestrutura e I/O (`AudioIO`, `RealtimeMediaStreamer` e `PresetIO`).
Cobre PB01, PB02, PB10, PB13, PB14, PB15, PB20, PB29, reprodução em tempo real de MP3/WAV
e seleção de dispositivos de entrada/saída.
"""

import os
import shutil
import tempfile
import unittest
from queue import Queue
import numpy as np
from scipy.io import wavfile

from eq.config import EqConfig, FilterType, ChannelMode, Slope, InputSource
from eq.dsp.engine import EqEngine
from eq.io.audio_io import AudioIO
from eq.io.media_player import RealtimeMediaStreamer
from eq.io.preset_io import PresetIO


class TestAudioIOAndPresetIO(unittest.TestCase):
    """Valida stream de áudio, player MP3/WAV em tempo real, detecção de clipping e persistência JSON/XML."""

    def test_audio_callback_and_clipping_detection_pb20(self):
        """
        PB01 / PB02 / PB20: Verifica que `audio_callback` processa o buffer de entrada pelo
        `EqEngine`, alimenta a `visual_queue` sem bloquear e detecta picos > 0 dBFS.
        """
        cfg = EqConfig(sample_rate=48000, buffer_size=128)
        engine = EqEngine(cfg)
        engine.add_band(freq_hz=1000.0, gain_db=12.0, q=0.707, ftype=FilterType.PEAKING)
        q = Queue(maxsize=2)
        io = AudioIO(config=cfg, engine=engine, queue=q)

        t = np.arange(512) / 48000.0
        sine = (0.6 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)
        indata = np.column_stack([sine, sine])
        outdata = np.zeros_like(indata)

        with io:
            self.assertTrue(io.running)
            for i in range(0, 512, 128):
                io.audio_callback(indata[i : i + 128], outdata[i : i + 128], 128, None, None)

        self.assertFalse(io.running)
        self.assertTrue(io.clip_detected)
        self.assertGreater(io.last_peak_dbfs, 0.0)
        self.assertEqual(q.qsize(), 2)

    def test_output_only_callback_produces_audible_sound_by_default(self):
        """
        Garante que o modo padrão (`SYNTH_DRONE`), `PINK_NOISE` e `SINE_SWEEP` geram
        sinal de áudio audível (não-zero) diretamente em `_output_only_callback`.
        """
        cfg = EqConfig(sample_rate=44100, buffer_size=256)
        engine = EqEngine(cfg, init_default_bands=True)
        io = AudioIO(config=cfg, engine=engine)

        for src_mode in (InputSource.SYNTH_DRONE, InputSource.PINK_NOISE, InputSource.SINE_SWEEP):
            io.set_input_source(src_mode)
            outdata = np.zeros((256, 2), dtype=np.float32)
            io._output_only_callback(outdata, 256, None, None)
            rms = float(np.sqrt(np.mean(outdata ** 2)))
            self.assertGreater(rms, 0.01, f"Saída silenciosa para a fonte {src_mode}")

    def test_mono_microphone_upmix_in_duplex_callback(self):
        """
        Verifica que um microfone Mono (shape `[frames, 1]`) é convertido automaticamente
        para Estéreo `[frames, 2]` no `_duplex_callback` sem erro de canais.
        """
        cfg = EqConfig(sample_rate=44100, buffer_size=128, input_source=InputSource.MICROPHONE)
        engine = EqEngine(cfg, init_default_bands=True)
        io = AudioIO(config=cfg, engine=engine)

        t = np.arange(128, dtype=np.float32) / 44100.0
        mono_in = (0.4 * np.sin(2 * np.pi * 440.0 * t))[:, None].astype(np.float32)
        stereo_out = np.zeros((128, 2), dtype=np.float32)
        io._duplex_callback(mono_in, stereo_out, 128, None, None)
        self.assertGreater(np.max(np.abs(stereo_out[:, 0])), 0.05)
        np.testing.assert_allclose(stereo_out[:, 0], stereo_out[:, 1], atol=1e-6)

    def test_realtime_media_streamer_wav_and_mp3_playback(self):
        """
        Verifica que `RealtimeMediaStreamer` decodifica e injeta arquivos `.wav` e `.mp3`
        no fluxo de áudio em tempo real como se fossem uma entrada ao vivo, suportando
        Play, Pause, Stop, Seek e Loop.
        """
        cfg = EqConfig(sample_rate=44100, buffer_size=512)
        engine = EqEngine(cfg, init_default_bands=True)
        io = AudioIO(config=cfg, engine=engine)

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = os.path.join(tmpdir, "demo_test.wav")
            RealtimeMediaStreamer.generate_demo_track_file(wav_path, duration_sec=1.0, sample_rate=44100)
            n_frames = io.load_media_file(wav_path)
            self.assertEqual(n_frames, 44100)
            self.assertEqual(io.config.input_source, InputSource.MEDIA_FILE)

            # Simula o callback de saída lendo blocos em tempo real do arquivo carregado
            outdata = np.zeros((512, 2), dtype=np.float32)
            io._output_only_callback(outdata, 512, None, None)
            self.assertGreater(float(np.max(np.abs(outdata))), 0.05)
            self.assertEqual(io.media_streamer.playhead, 512)

            # Testa Seek, Pause e Play
            io.seek_media(0.5)
            self.assertAlmostEqual(io.media_streamer.get_position_seconds(), 0.5, delta=0.02)
            io.pause_media()
            out_paused = np.ones((512, 2), dtype=np.float32)
            io._output_only_callback(out_paused, 512, None, None)
            # Quando pausado, o bloco de entrada é zero
            self.assertLess(float(np.max(np.abs(out_paused))), 0.05)
            io.play_media()

            # Se ffmpeg estiver disponível, testa também decodificação e streaming real de .MP3
            if shutil.which("ffmpeg"):
                mp3_path = os.path.join(tmpdir, "demo_test.mp3")
                RealtimeMediaStreamer.generate_demo_track_file(mp3_path, duration_sec=1.0, sample_rate=44100)
                mp3_frames = io.load_media_file(mp3_path)
                self.assertGreater(mp3_frames, 40000)
                out_mp3 = np.zeros((512, 2), dtype=np.float32)
                io._output_only_callback(out_mp3, 512, None, None)
                self.assertGreater(float(np.max(np.abs(out_mp3))), 0.05)

    def test_input_and_output_device_selection(self):
        """Verifica listagem e seleção de dispositivos de entrada e saída no `AudioIO`."""
        io = AudioIO()
        in_devs = io.list_input_devices()
        out_devs = io.list_output_devices()
        self.assertGreaterEqual(len(in_devs), 1)
        self.assertGreaterEqual(len(out_devs), 1)
        io.set_input_device(None)
        io.set_output_device(None)
        self.assertIsNone(io.config.input_device)
        self.assertIsNone(io.config.output_device)

    def test_wav_file_processing_end_to_end(self):
        """Verifica o processamento offline de um arquivo WAV completo pelo `AudioIO`."""
        cfg = EqConfig(sample_rate=44100, buffer_size=256)
        engine = EqEngine(cfg)
        engine.add_band(freq_hz=1000.0, gain_db=-6.0, q=1.0, ftype=FilterType.PEAKING)
        io = AudioIO(config=cfg, engine=engine)

        with tempfile.TemporaryDirectory() as tmpdir:
            in_wav = os.path.join(tmpdir, "in.wav")
            out_wav = os.path.join(tmpdir, "out.wav")
            t = np.arange(4410) / 44100.0
            sig = (0.5 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)
            wavfile.write(in_wav, 44100, np.column_stack([sig, sig]))

            processed = io.process_wav_file(in_wav, out_wav)
            self.assertTrue(os.path.exists(out_wav))
            self.assertEqual(processed.shape, (4410, 2))
            self.assertLess(np.std(processed[1000:]), np.std(sig[1000:]))

    def test_preset_io_json_and_xml_serialization_pb14(self):
        """
        PB13 / PB14 / PB15: Verifica salvamento e carregamento de presets em JSON e XML.
        """
        cfg = EqConfig(sample_rate=96000, buffer_size=512, ms_enabled=True, oversampling=2)
        engine = EqEngine(cfg)
        engine.add_band(freq_hz=80.0, gain_db=0.0, q=0.707, ftype=FilterType.LOW_CUT, slope=Slope.S24)
        engine.add_band(freq_hz=2500.0, gain_db=4.5, q=2.2, ftype=FilterType.PEAKING, mode=ChannelMode.MID)
        engine.config.last_audio_buffer = np.ones((512, 2), dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "vocal_eq.json")
            xml_path = os.path.join(tmpdir, "vocal_eq.xml")

            PresetIO.save_preset(engine, json_path)
            PresetIO.save_preset(engine, xml_path)

            for path in (json_path, xml_path):
                restored_engine = EqEngine()
                PresetIO.load_preset(restored_engine, path)
                self.assertEqual(restored_engine.config.sample_rate, 96000)
                self.assertTrue(restored_engine.config.ms_enabled)
                self.assertEqual(restored_engine.config.oversampling, 2)
                self.assertEqual(len(restored_engine.bands), 2)
                self.assertEqual(restored_engine.bands[0].ftype, FilterType.LOW_CUT)
                self.assertEqual(restored_engine.bands[0].slope, Slope.S24)
                self.assertAlmostEqual(restored_engine.bands[1].gain_db, 4.5, places=4)
                self.assertEqual(restored_engine.bands[1].mode, ChannelMode.MID)


if __name__ == "__main__":
    unittest.main()
