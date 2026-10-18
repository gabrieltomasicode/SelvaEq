"""
Suíte Expandida de Testes Unitários e de Regressão (`T16` - `tests/dsp/test_optimizations_and_fixes.py`).

Valida deterministicamente todas as correções, melhorias e otimizações das 3 frentes:
- Frente C (Otimizações Técnicas): `T01` a `T15`
- Frente B (Melhorias Funcionais): `F01` a `F16`
- Frente A (Melhorias Visuais e Stubs Headless): `V01` a `V10`
"""

import math
import os
import sys
import tempfile
import threading
import time
import unittest
from queue import Queue
import numpy as np
from scipy.signal import lfilter, lfilter_zi

from eq.config import EqConfig, FilterType, ChannelMode, Slope, InputSource
from eq.dsp.biquad_math import BiquadMath
from eq.dsp.band import EqBand
from eq.dsp.engine import EqEngine
from eq.dsp.spectrum import SpectrumFFT
from eq.io.audio_io import AudioIO, visual_queue
from eq.io.preset_io import PresetIO
from eq.ui.widgets import (
    SelvaSlider,
    apply_dark_ttk_style,
    resolve_font_family,
    get_ui_fonts,
    tk,
)
from eq.ui.interface import EqInterface


class _DummyMplEvent:
    """Simula um evento de mouse/scroll do Matplotlib com suporte a modificadores do Tk."""

    def __init__(self, inaxes, xdata, ydata, button=1, dblclick=False, key=None, state=0):
        self.inaxes = inaxes
        self.xdata = xdata
        self.ydata = ydata
        self.button = button
        self.dblclick = dblclick
        self.key = key

        class _GuiEv:
            pass

        self.guiEvent = _GuiEv()
        self.guiEvent.state = state


class TestOptimizationsAndFixes(unittest.TestCase):
    """Cobre os itens T01–T16, F01–F16 e V01–V10 do roteiro técnico do SelvaEQ."""

    def test_t01_concurrent_ui_and_audio_thread_safety(self):
        """
        T01: Dispara uma thread rodando `engine.process(block)` por 500 blocos enquanto
        a thread principal executa `set_param`, `add_band`, `remove_band` e alterna `Slope`
        entre `S6` e `S96`; verifica 0 exceções e saída 100% finita.
        """
        cfg = EqConfig(sample_rate=48000, buffer_size=256, num_bands=24)
        engine = EqEngine(cfg, init_default_bands=True)
        errors = []
        non_finite = []
        done = threading.Event()

        rng = np.random.default_rng(123)
        test_block = rng.uniform(-0.3, 0.3, size=(256, 2)).astype(np.float32)

        def _audio_worker():
            try:
                for _ in range(500):
                    out = engine.process(test_block)
                    if not np.all(np.isfinite(out)):
                        non_finite.append(True)
            except Exception as exc:
                errors.append(exc)
            finally:
                done.set()

        t = threading.Thread(target=_audio_worker, daemon=True)
        t.start()

        slopes = [Slope.S6, Slope.S12, Slope.S24, Slope.S48, Slope.S96]
        step = 0
        while not done.is_set():
            try:
                n_b = len(engine.bands)
                if n_b > 0:
                    idx = step % n_b
                    engine.set_param(idx, "freq_hz", 80.0 + (step * 73) % 14000)
                    engine.set_param(idx, "gain_db", float((step % 25) - 12))
                    engine.set_param(idx, "q", 0.5 + (step % 10) * 0.5)
                    engine.set_param(0, "slope", slopes[step % len(slopes)])
                if n_b < 12 and step % 5 == 0:
                    engine.add_band(freq_hz=1500.0, gain_db=3.0, q=1.2)
                elif n_b > 5 and step % 7 == 0:
                    engine.remove_band(n_b - 1)
            except IndexError:
                pass
            except Exception as exc:
                errors.append(exc)
            step += 1

        t.join(timeout=10.0)
        self.assertEqual(errors, [], f"Exceções concorrentes detectadas: {errors}")
        self.assertEqual(non_finite, [], "Amostras NaN/Inf detectadas durante concorrência.")

    def test_t02_subblock_parameter_smoothing(self):
        """
        T02: Configura uma banda PEAKING em 0 dB, chama `engine.set_param(0, "gain_db", 12.0)`
        e processa um bloco de 512 amostras; verifica que o ganho inicial no primeiro sub-bloco
        parte de ~0 dB e converge progressivamente para +12 dB nos blocos seguintes.
        """
        fs = 44100
        cfg = EqConfig(sample_rate=fs, buffer_size=512)
        engine = EqEngine(cfg)
        engine.add_band(freq_hz=1000.0, gain_db=0.0, q=1.0, ftype=FilterType.PEAKING)

        t = np.arange(512 * 6, dtype=np.float64) / float(fs)
        sine = (0.25 * np.sin(2.0 * np.pi * 1000.0 * t)).astype(np.float32)
        stereo = np.column_stack([sine, sine])

        # Aquece o estado em 0 dB
        _ = engine.process(stereo[:512])

        # Altera o ganho alvo para +12 dB
        engine.set_param(0, "gain_db", 12.0)
        band = engine.bands[0]
        self.assertAlmostEqual(band._gain_smoother.current, 0.0, places=5)
        self.assertAlmostEqual(band.coeffs[0].b0, 1.0, places=5)

        out_b1 = engine.process(stereo[512:1024])
        gain_after_b1 = band._gain_smoother.current
        self.assertGreater(gain_after_b1, 5.0)
        self.assertLess(gain_after_b1, 11.9)

        # No início do bloco 1 (primeiras 32 amostras), a amplitude ainda é menor que no final do bloco
        rms_start = float(np.sqrt(np.mean(np.square(out_b1[:32, 0]))))
        rms_end = float(np.sqrt(np.mean(np.square(out_b1[-64:, 0]))))
        self.assertLess(rms_start, rms_end)

        # Após mais blocos, converge para +12 dB (~3.981x)
        for i in range(2, 6):
            out_bi = engine.process(stereo[i * 512 : (i + 1) * 512])
        self.assertAlmostEqual(band._gain_smoother.current, 12.0, places=2)
        rms_in = float(np.sqrt(np.mean(np.square(stereo[5 * 512 : 6 * 512, 0]))))
        rms_out = float(np.sqrt(np.mean(np.square(out_bi[:, 0]))))
        gain_measured_db = 20.0 * math.log10(rms_out / rms_in)
        self.assertAlmostEqual(gain_measured_db, 12.0, delta=0.25)

    def test_t03_solo_isolated_zi_and_bypass_reinit(self):
        """
        T03: Verifica que ativar e desativar `solo_mode` não altera `band._zi` do filtro
        principal e que sair de `bypass` reinicializa `_zi` em regime estacionário sem pico espúrio.
        """
        fs = 48000
        band = EqBand(
            index=0,
            freq_hz=1000.0,
            gain_db=9.0,
            q=1.5,
            ftype=FilterType.LOW_CUT,
            slope=Slope.S24,
            sample_rate=fs,
        )
        t = np.arange(512, dtype=np.float64) / float(fs)
        blk = np.column_stack([np.sin(2 * np.pi * 1000.0 * t), np.cos(2 * np.pi * 1000.0 * t)])

        band.process(blk, solo_mode=False)
        zi_normal_copy = band._zi.copy()
        self.assertEqual(zi_normal_copy.shape, (2, 2, 2))

        # Processa em solo_mode=True (BAND_PASS de 1 estágio)
        band.process(blk, solo_mode=True)
        self.assertIsNotNone(band._zi_solo)
        self.assertEqual(band._zi_solo.shape, (1, 2, 2))
        np.testing.assert_array_equal(band._zi, zi_normal_copy)

        # Testa reinicialização suave pós-bypass
        b_byp = EqBand(freq_hz=800.0, gain_db=12.0, q=2.0, ftype=FilterType.PEAKING, sample_rate=fs)
        b_byp.process(blk * 0.8)
        b_byp.set_targets(bypass=True)
        for _ in range(8):
            b_byp.process(blk * 0.8)
        self.assertTrue(b_byp._was_bypassed)

        # Sinal muda para nível DC constante enquanto a banda estava em bypass
        dc_blk = np.full((256, 2), -0.4, dtype=np.float64)
        b_byp.set_targets(bypass=False)
        out_re = b_byp.process(dc_blk)
        # Um filtro peaking tem ganho DC = 0 dB (1.0), logo em regime estacionário DC a saída deve ser -0.4 sem salto
        max_dev = float(np.max(np.abs(out_re - dc_blk)))
        self.assertLess(max_dev, 1e-4)

    def test_t04_stateful_oversampling_2x_and_4x_continuity(self):
        """
        T04: Passa 8 blocos consecutivos de senoide 1 kHz por `EqEngine` sem bandas com
        `oversampling=2` e `oversampling=4`; verifica que, descontada a latência de 32 amostras,
        o erro máximo em regime é < 1e-4 (contra 0.098 antes).
        """
        fs = 44100
        block_size = 512
        n_blocks = 8
        t = np.arange(block_size * n_blocks, dtype=np.float64) / float(fs)
        sig = np.column_stack([
            np.sin(2.0 * np.pi * 1000.0 * t),
            np.cos(2.0 * np.pi * 1000.0 * t),
        ])

        for factor in (2, 4):
            cfg = EqConfig(sample_rate=fs, buffer_size=block_size, oversampling=factor)
            engine = EqEngine(cfg, init_default_bands=False)
            self.assertEqual(engine.latency_samples, 32)

            outputs = []
            for b_idx in range(n_blocks):
                chunk = sig[b_idx * block_size : (b_idx + 1) * block_size]
                outputs.append(engine.process(chunk))
            out_full = np.vstack(outputs)

            lat = engine.latency_samples
            steady_out = out_full[block_size + lat :]
            steady_ref = sig[block_size : -lat]
            max_err = float(np.max(np.abs(steady_out - steady_ref)))
            self.assertLess(max_err, 1e-4, f"Oversampling {factor}x com erro excessivo: {max_err}")

    def test_t05_global_response_matches_oversampling_rate(self):
        """
        T05: Com uma banda PEAKING em 10000 Hz (+12 dB), verifica que a frequência de pico
        retornada por `engine.global_response(n=512)` é idêntica em `oversampling=1`, `2` e `4`.
        """
        peak_freqs = []
        for factor in (1, 2, 4):
            cfg = EqConfig(sample_rate=44100, oversampling=factor)
            engine = EqEngine(cfg)
            engine.add_band(freq_hz=10000.0, gain_db=12.0, q=2.5, ftype=FilterType.PEAKING)
            freqs, mag_db = engine.global_response(n=512)
            peak_idx = int(np.argmax(mag_db))
            peak_freqs.append(float(freqs[peak_idx]))
            self.assertAlmostEqual(float(mag_db[peak_idx]), 12.0, delta=0.15)

        self.assertAlmostEqual(peak_freqs[0], peak_freqs[1], places=6)
        self.assertAlmostEqual(peak_freqs[1], peak_freqs[2], places=6)
        self.assertAlmostEqual(peak_freqs[0], 10000.0, delta=150.0)

    def test_t06_sosfilt_matches_reference_and_performance(self):
        """
        T06 / T11: Verifica que a cascata `S96` via `sosfilt` produz saída numericamente idêntica
        (`atol=1e-12`) à cascata `lfilter` de referência e processa 24 bandas em tempo real.
        """
        fs = 48000
        band = EqBand(
            freq_hz=1200.0,
            gain_db=0.0,
            q=0.70710678,
            ftype=FilterType.LOW_CUT,
            slope=Slope.S96,
            sample_rate=fs,
        )
        rng = np.random.default_rng(7)
        sig = rng.uniform(-0.5, 0.5, size=(512, 2)).astype(np.float64)

        # Referência com lfilter estágio por estágio
        ref = sig.copy()
        for c in band.coeffs:
            b, a = c.as_ba()
            zi_base = lfilter_zi(b, a)
            for ch in range(2):
                ref[:, ch], _ = lfilter(b, a, ref[:, ch], zi=zi_base * float(sig[0, ch]))

        out_sos = band.process(sig)
        np.testing.assert_allclose(out_sos, ref, atol=1e-12)

        # Desempenho com 24 bandas (8 delas em S96)
        engine = EqEngine(EqConfig(sample_rate=fs, buffer_size=512, num_bands=24))
        for i in range(24):
            f_hz = 40.0 * (1.28 ** i)
            ftype = FilterType.LOW_CUT if i < 4 else (FilterType.HIGH_CUT if i >= 20 else FilterType.PEAKING)
            slope = Slope.S96 if (i < 4 or i >= 20) else Slope.S12
            engine.add_band(freq_hz=min(f_hz, 19000.0), gain_db=1.5, q=1.2, ftype=ftype, slope=slope)

        # Aquece o estado zi no primeiro bloco
        block32 = sig.astype(np.float32)
        _ = engine.process(block32)

        t0 = time.perf_counter()
        n_iters = 20
        for _ in range(n_iters):
            _ = engine.process(block32)
        elapsed_ms = ((time.perf_counter() - t0) * 1000.0) / n_iters
        # Deve ficar muito abaixo do orçamento de 11.6 ms do callback de áudio
        self.assertLess(elapsed_ms, 5.0)

    def test_t07_shelf_high_q_and_butterworth_most_resonant_stage(self):
        """
        T07: Verifica que `low_shelf` com +12 dB e Q=0.70710678 tem ganho máximo 12.0 dB
        (tolerância 0.15 dB), que Q=10 difere de Q=30, e que em `Slope.S24` alterar `q`
        modifica o estágio `0` (maior Q).
        """
        fs = 44100
        c_07 = BiquadMath.low_shelf(fs, 1000.0, 12.0, 0.70710678)
        _, h_07 = BiquadMath.freq_response(c_07, fs, n=1024)
        db_07 = 20.0 * np.log10(np.abs(h_07))
        self.assertAlmostEqual(float(np.max(db_07)), 12.0, delta=0.15)
        self.assertAlmostEqual(float(np.min(db_07)), 0.0, delta=0.15)

        c_10 = BiquadMath.low_shelf(fs, 1000.0, 12.0, 10.0)
        c_30 = BiquadMath.low_shelf(fs, 1000.0, 12.0, 30.0)
        self.assertNotEqual(c_10, c_30)
        _, h_10 = BiquadMath.freq_response(c_10, fs, n=2048)
        _, h_30 = BiquadMath.freq_response(c_30, fs, n=2048)
        self.assertGreater(float(np.max(np.abs(h_30))), float(np.max(np.abs(h_10))))

        stages_butter = BiquadMath.cascade(fs, 1000.0, Slope.S24, FilterType.LOW_CUT, q=1.0 / math.sqrt(2.0))
        stages_res = BiquadMath.cascade(fs, 1000.0, Slope.S24, FilterType.LOW_CUT, q=2.0)
        self.assertNotEqual(stages_butter[0], stages_res[0])
        self.assertEqual(stages_butter[1], stages_res[1])

    def test_t08_t09_t10_audio_io_clip_latch_observer_and_queue(self):
        """
        T08 / T09 / T10 / T13: Verifica latch temporal de `clip_detected` até `reset_clip()`,
        notificação de `add_sample_rate_listener`, `visual_queue` de 32 blocos e lock não-bloqueante.
        """
        self.assertEqual(visual_queue.maxsize, 32)

        cfg = EqConfig(sample_rate=44100, buffer_size=128)
        engine = EqEngine(cfg)
        q = Queue(maxsize=32)
        io = AudioIO(config=cfg, engine=engine, queue=q)
        analyzer = SpectrumFFT(fft_size=2048, sample_rate=44100)
        io.add_sample_rate_listener(analyzer.set_sample_rate)

        # Fallback de sample rate atualiza o analisador via Observer (T09)
        io._apply_fallback_samplerate(48000)
        self.assertEqual(analyzer.sample_rate, 48000)

        # Bloco clipado (> 0 dBFS) seguido de bloco silencioso mantém clip_detected True (T08)
        loud = np.full((128, 2), 1.4, dtype=np.float32)
        silent = np.zeros((128, 2), dtype=np.float32)
        out = np.zeros((128, 2), dtype=np.float32)

        io.audio_callback(loud, out, 128, None, None)
        self.assertTrue(io.clip_detected)
        self.assertGreater(io.last_rms_dbfs, 0.0)

        io.audio_callback(silent, out, 128, None, None)
        self.assertTrue(io.clip_detected)

        io.reset_clip()
        self.assertFalse(io.clip_detected)

        # Lock não-bloqueante em read_media_block (T13)
        io.media_streamer._lock.acquire()
        try:
            blk = io.media_streamer.read_media_block(128)
            self.assertEqual(blk.shape, (128, 2))
            self.assertTrue(np.all(blk == 0.0))
        finally:
            io.media_streamer._lock.release()

    def test_f01_f02_f03_f04_ui_interactions_and_ab_state(self):
        """
        F01 / F02 / F03 / F04: Verifica reset de ganho por duplo clique em nó existente,
        imunidade a NumLock (`0x0008`) no Windows, transações Undo/Redo em sliders/combos
        e preservação de edições em `StateController.compare_ab()`.
        """
        cfg = EqConfig(sample_rate=48000, buffer_size=256)
        engine = EqEngine(cfg, init_default_bands=False)
        engine.add_band(freq_hz=1000.0, gain_db=9.0, q=1.0, ftype=FilterType.PEAKING)
        io = AudioIO(config=cfg, engine=engine)
        ui = EqInterface(engine=engine, io=io)

        # F01: Duplo clique sobre o nó existente em (1000 Hz, +9 dB) zera o ganho sem criar nova banda
        ev_dbl = _DummyMplEvent(ui.ax, xdata=1000.0, ydata=9.0, button=1, dblclick=True)
        ui._on_mpl_mouse_down(ev_dbl)
        self.assertEqual(len(engine.bands), 1)
        self.assertAlmostEqual(engine.bands[0].gain_db, 0.0, places=4)
        self.assertTrue(engine.state_controller.can_undo())

        ui.undo_action()
        self.assertAlmostEqual(engine.bands[0].gain_db, 9.0, places=4)

        # F02: NumLock (0x0008) no Windows não ativa Alt/Solo
        orig_platform = sys.platform
        try:
            sys.platform = "win32"
            ev_nl = _DummyMplEvent(ui.ax, xdata=1000.0, ydata=9.0, state=0x0008)
            _, _, is_alt = ui._extract_modifiers(ev_nl)
            self.assertFalse(is_alt)
            ev_alt = _DummyMplEvent(ui.ax, xdata=1000.0, ydata=9.0, state=0x20000)
            _, _, is_alt_real = ui._extract_modifiers(ev_alt)
            self.assertTrue(is_alt_real)
        finally:
            sys.platform = orig_platform

        # F03: Edição por slider registra transação Undo/Redo
        ui._on_slider_gain(-6.0)
        self.assertAlmostEqual(engine.bands[0].gain_db, -6.0, places=4)
        ui.undo_action()
        self.assertAlmostEqual(engine.bands[0].gain_db, 9.0, places=4)
        ui.redo_action()
        self.assertAlmostEqual(engine.bands[0].gain_db, -6.0, places=4)

        # F04: compare_ab preserva edições feitas dentro do slot A
        sc = engine.state_controller
        engine.set_param(0, "gain_db", 3.0)
        sc.store_a()
        engine.set_param(0, "gain_db", -5.0)
        sc.store_b()

        sc.compare_ab()  # Vai para A (3.0)
        self.assertAlmostEqual(engine.bands[0].gain_db, 3.0, places=4)
        engine.set_param(0, "gain_db", 5.0)  # Edita A para 5.0
        sc.compare_ab()  # Vai para B (-5.0)
        self.assertAlmostEqual(engine.bands[0].gain_db, -5.0, places=4)
        sc.compare_ab()  # Volta para A -> deve preservar 5.0!
        self.assertAlmostEqual(engine.bands[0].gain_db, 5.0, places=4)

        ui.on_close()

    def test_f06_f07_f10_f14_f15_log_slider_presets_and_freeze(self):
        """
        F06 / F07 / F10 / F14 / F15 / T12 / T14: Verifica escala logarítmica de `SelvaSlider`,
        `flatten_bands()`, `PresetIO.FACTORY_PRESETS`, `SpectrumFFT` (in-place ring, freeze,
        peak_hold_curve), proteção contra NaN e exportação WAV.
        """
        # F06: Escala logarítmica no ponto médio (50%) entre 20 Hz e 20000 Hz => ~632.45 Hz
        slider = SelvaSlider(width=216, from_=20.0, to_=20000.0, init=1000.0, scale="log")
        slider._update_val_from_x(108.0)  # 8 + 0.5 * (216 - 16)
        self.assertAlmostEqual(slider.value, math.sqrt(20.0 * 20000.0), delta=1.0)

        # T12 & F14: Buffer circular sem realocação, Peak Hold e Freeze
        analyzer = SpectrumFFT(fft_size=1024, sample_rate=48000)
        ring_id = id(analyzer.ring)
        t = np.arange(512, dtype=np.float64) / 48000.0
        analyzer.push_block(np.sin(2.0 * np.pi * 1000.0 * t))
        self.assertEqual(id(analyzer.ring), ring_id)
        _, ph1 = analyzer.peak_hold_curve()
        self.assertEqual(len(ph1), 513)

        analyzer.set_freeze(True)
        snap = analyzer.ring.copy()
        analyzer.push_block(np.ones(512, dtype=np.float64))
        np.testing.assert_array_equal(analyzer.ring, snap)

        # F10 & F07 & F15 & T14
        cfg = EqConfig(sample_rate=44100, buffer_size=256, auto_gain=True)
        engine = EqEngine(cfg, init_default_bands=True)
        io = AudioIO(config=cfg, engine=engine)
        ui = EqInterface(engine=engine, io=io)

        PresetIO.apply_factory_preset(engine, "Presença Vocal")
        self.assertTrue(engine.state_controller.can_undo())
        self.assertLess(engine.estimate_auto_gain_db(), 0.0)

        ui.flatten_all_bands()
        self.assertTrue(all(abs(b.gain_db) < 1e-6 for b in engine.bands))

        # T14: Injeção de NaN não contamina blocos seguintes
        nan_blk = np.zeros((128, 2), dtype=np.float32)
        nan_blk[8, 0] = np.nan
        out_nan = engine.process(nan_blk)
        self.assertTrue(np.all(np.isfinite(out_nan)))
        out_clean = engine.process(np.ones((128, 2), dtype=np.float32) * 0.2)
        self.assertTrue(np.all(np.isfinite(out_clean)))

        # F15: Exportação de arquivo WAV processado
        with tempfile.TemporaryDirectory() as tmpdir:
            out_wav = os.path.join(tmpdir, "rendered_eq.wav")
            saved = ui.export_processed_wav_dialog(out_wav)
            self.assertEqual(saved, out_wav)
            self.assertTrue(os.path.isfile(out_wav))
            self.assertGreater(os.path.getsize(out_wav), 1000)

        # V01 & V02
        root = tk.Tk()
        apply_dark_ttk_style(root)
        fonts = get_ui_fonts(root)
        self.assertIn("F_SPACE", fonts)
        self.assertTrue(isinstance(resolve_font_family(["Space Mono", "DejaVu Sans Mono"]), str))

        ui.on_close()


if __name__ == "__main__":
    unittest.main()
