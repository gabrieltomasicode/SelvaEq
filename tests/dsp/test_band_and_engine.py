"""
Testes Unitários de `EqBand`, `EqEngine`, `ParameterSmoother` e `StateController`.
Cobre PB04, PB06, PB11, PB12, PB16, PB18, PB22 e continuidade de estado `zi` entre blocos.
"""

import unittest
import numpy as np

from eq.config import EqConfig, FilterType, ChannelMode, Slope
from eq.dsp.band import EqBand
from eq.dsp.engine import EqEngine
from eq.dsp.utils import ParameterSmoother


class TestEqBandAndEngine(unittest.TestCase):
    """Valida a operação isolada e conjunta de `EqBand` e `EqEngine`."""

    def test_config_rejects_non_finite_values_and_boolean_sizes(self):
        for sample_rate in (float("nan"), float("inf"), float("-inf"), True):
            with self.subTest(sample_rate=sample_rate), self.assertRaises(ValueError):
                EqConfig(sample_rate=sample_rate)

        for output_gain_db in (float("nan"), float("inf"), float("-inf"), True):
            with self.subTest(output_gain_db=output_gain_db), self.assertRaises(ValueError):
                EqConfig(output_gain_db=output_gain_db)

        for config_values in (
            {"buffer_size": True},
            {"num_bands": True},
            {"oversampling": True},
        ):
            with self.subTest(config_values=config_values), self.assertRaises(ValueError):
                EqConfig(**config_values)

        self.assertEqual(EqConfig(sample_rate=44100.5).sample_rate, 44100.5)

    def test_state_continuity_across_blocks_no_clicks(self):
        """
        Verifica que o estado contínuo `zi`/`zf` (herdado de `synth/filters.py::apply_filter`)
        produz exatamente o mesmo sinal quando processado em 8 blocos de 128 frames
        versus 1 bloco único de 1024 frames.
        """
        fs = 48000
        total_frames = 1024
        chunk_size = 128
        t = np.arange(total_frames) / float(fs)
        signal = np.column_stack([
            np.sin(2 * np.pi * 440.0 * t) + 0.5 * np.sin(2 * np.pi * 4000.0 * t),
            np.cos(2 * np.pi * 440.0 * t) + 0.5 * np.cos(2 * np.pi * 4000.0 * t),
        ]).astype(np.float64)

        band_single = EqBand(freq_hz=1200.0, gain_db=9.0, q=1.5, ftype=FilterType.PEAKING, sample_rate=fs)
        band_chunked = EqBand(freq_hz=1200.0, gain_db=9.0, q=1.5, ftype=FilterType.PEAKING, sample_rate=fs)

        out_single = band_single.process(signal)
        chunks = []
        for i in range(0, total_frames, chunk_size):
            chunks.append(band_chunked.process(signal[i : i + chunk_size]))
        out_chunked = np.vstack(chunks)

        np.testing.assert_allclose(out_chunked, out_single, atol=1e-6)

    def test_dynamic_24_bands_allocation_pb04(self):
        """PB04: Valida alocação de até 24 bandas simultâneas e bloqueio da 25ª banda."""
        engine = EqEngine(EqConfig(sample_rate=48000, buffer_size=256, num_bands=24))
        for i in range(24):
            freq = 30.0 * (1.3 ** i)
            b = engine.add_band(freq_hz=min(freq, 19000.0), gain_db=0.5, q=1.0)
            self.assertEqual(b.index, i)
        self.assertEqual(len(engine.bands), 24)

        with self.assertRaises(ValueError):
            engine.add_band(freq_hz=1000.0)

        # Processa um bloco pelas 24 bandas em cascata
        block = np.random.uniform(-0.2, 0.2, (256, 2)).astype(np.float32)
        out = engine.process(block)
        self.assertEqual(out.shape, (256, 2))
        self.assertFalse(np.any(np.isnan(out)))

    def test_mid_side_encoding_decoding_and_routing_pb22(self):
        """
        PB22: Valida a identidade da matriz M/S e roteamento independente MID vs SIDE.
        """
        stereo = np.random.uniform(-0.8, 0.8, (512, 2)).astype(np.float64)
        ms = EqEngine._encode_mid_side(stereo)
        reconstructed = EqEngine._decode_mid_side(ms)
        np.testing.assert_allclose(reconstructed, stereo, atol=1e-12)

        # Sinal puramente em fase (L == R => Side == 0)
        t = np.arange(1024) / 48000.0
        mono_sine = np.sin(2 * np.pi * 1000.0 * t)
        in_phase_stereo = np.column_stack([mono_sine, mono_sine])

        # Banda atuando apenas no canal SIDE não deve alterar um sinal puramente em fase (Mid)
        engine = EqEngine(EqConfig(sample_rate=48000, ms_enabled=True))
        engine.add_band(freq_hz=1000.0, gain_db=18.0, q=1.0, ftype=FilterType.PEAKING, mode=ChannelMode.SIDE)
        out = engine.process(in_phase_stereo)
        np.testing.assert_allclose(out, in_phase_stereo, atol=1e-5)

    def test_solo_and_bypass_resolution_pb12(self):
        """
        PB12: Verifica que ativar Solo em uma banda silencia as demais e aplica Band-Pass na banda focada.
        """
        fs = 48000
        engine = EqEngine(EqConfig(sample_rate=fs))
        engine.add_band(freq_hz=100.0, gain_db=12.0, q=1.0, ftype=FilterType.PEAKING)
        engine.add_band(freq_hz=2000.0, gain_db=-12.0, q=4.0, ftype=FilterType.PEAKING, solo=True)

        active = engine._active_bands()
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0].index, 1)

        # Na curva teórica em modo Solo, o ganho em 2000 Hz deve ser 0 dB (pico do band-pass)
        # e em 100 Hz deve estar fortemente atenuado (< -15 dB)
        _, mag_db = engine.global_response(freqs=np.array([100.0, 2000.0]))
        self.assertAlmostEqual(mag_db[1], 0.0, places=2)
        self.assertLess(mag_db[0], -15.0)

    def test_oversampling_decramping_pb11(self):
        """PB11: Verifica operação do motor com oversampling 2x e 4x."""
        for factor in (1, 2, 4):
            cfg = EqConfig(sample_rate=48000, buffer_size=256, oversampling=factor)
            engine = EqEngine(cfg)
            engine.add_band(freq_hz=18000.0, gain_db=6.0, q=1.2, ftype=FilterType.PEAKING)
            inp = np.random.uniform(-0.5, 0.5, (256, 2)).astype(np.float32)
            out = engine.process(inp)
            self.assertEqual(out.shape, inp.shape)
            self.assertFalse(np.any(np.isnan(out)))

    def test_parameter_smoother_pb06_pb16(self):
        """PB06 / PB16: Verifica interpolação monotônica do `ParameterSmoother`."""
        sm = ParameterSmoother(initial_value=100.0, time_constant_ms=10.0, fs=48000)
        sm.set_target(1000.0)
        self.assertTrue(sm.is_smoothing())
        v1 = sm.get_next_value(128)
        v2 = sm.get_next_value(128)
        self.assertGreater(v1, 100.0)
        self.assertGreater(v2, v1)
        self.assertLess(v2, 1000.0)
        # Após suficientes amostras, converge para o alvo
        sm.get_next_value(48000)
        self.assertAlmostEqual(sm.current, 1000.0, places=3)
        self.assertFalse(sm.is_smoothing())

    def test_state_controller_undo_redo_and_ab_pb18(self):
        """PB18: Verifica pilha de histórico Undo/Redo e comparação A/B no `StateController`."""
        engine = EqEngine(EqConfig(sample_rate=48000))
        engine.add_band(freq_hz=500.0, gain_db=0.0, q=1.0)
        ctrl = engine.state_controller

        ctrl.begin_transaction("Boost 500 Hz")
        engine.set_param(0, "gain_db", 8.0)
        ctrl.commit()

        self.assertTrue(ctrl.can_undo())
        self.assertEqual(engine.bands[0].gain_db, 8.0)

        ctrl.undo()
        self.assertEqual(engine.bands[0].gain_db, 0.0)
        self.assertTrue(ctrl.can_redo())

        ctrl.redo()
        self.assertEqual(engine.bands[0].gain_db, 8.0)

        # Teste A/B
        ctrl.store_a()
        engine.set_param(0, "gain_db", -6.0)
        ctrl.store_b()

        ctrl.compare_ab()  # Volta para A (+8 dB)
        self.assertEqual(engine.bands[0].gain_db, 8.0)
        ctrl.compare_ab()  # Volta para B (-6 dB)
        self.assertEqual(engine.bands[0].gain_db, -6.0)


if __name__ == "__main__":
    unittest.main()
