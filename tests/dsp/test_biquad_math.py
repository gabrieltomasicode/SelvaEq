"""
Testes Unitários da Camada Matemática Biquad (`BiquadMath`, `Coeffs`, `SpectrumFFT`).
Cobre PB05, PB08, PB09, PB29 e UC19 (validação contra `scipy.signal`).
"""

import math
import unittest
import numpy as np
from scipy.signal import butter, freqz

from eq.config import FilterType, Slope, SUPPORTED_SAMPLE_RATES
from eq.dsp.biquad_math import BiquadMath, Coeffs
from eq.dsp.band import EqBand
from eq.dsp.spectrum import SpectrumFFT


class TestBiquadMathRBJ(unittest.TestCase):
    """Valida todas as equações do RBJ Audio EQ Cookbook e cascateamento Butterworth."""

    def test_peaking_zero_gain_is_unity(self):
        """Um filtro Peaking com gain_db = 0 dB deve ter resposta plana (0 dB em todo o espectro)."""
        c = BiquadMath.peaking(fs=48000, f0=1000.0, gain_db=0.0, q=1.0)
        self.assertAlmostEqual(c.b0, 1.0, places=6)
        self.assertAlmostEqual(c.b1, c.a1, places=6)
        self.assertAlmostEqual(c.b2, c.a2, places=6)
        _, h = BiquadMath.freq_response(c, fs=48000, n=512)
        mag_db = 20.0 * np.log10(np.abs(h))
        np.testing.assert_allclose(mag_db, 0.0, atol=1e-5)

    def test_peaking_boost_and_cut_accuracy(self):
        """Valida ganho exato na frequência central f0 (+12 dB e -12 dB) em Peaking."""
        for target_db in (12.0, -12.0, 6.0, -6.0):
            c = BiquadMath.peaking(fs=48000, f0=1000.0, gain_db=target_db, q=2.0)
            _, h = BiquadMath.freq_response(c, fs=48000, freqs=np.array([20.0, 1000.0, 20000.0]))
            mag_db = 20.0 * np.log10(np.abs(h))
            self.assertAlmostEqual(mag_db[1], target_db, places=3)
            self.assertAlmostEqual(mag_db[0], 0.0, delta=0.15)

    def test_low_and_high_shelf_response(self):
        """Valida que Low Shelf afeta graves e High Shelf afeta agudos conforme esperado."""
        ls = BiquadMath.low_shelf(fs=48000, f0=200.0, gain_db=6.0, s=1.0)
        _, h_ls = BiquadMath.freq_response(ls, fs=48000, freqs=np.array([20.0, 10000.0]))
        db_ls = 20.0 * np.log10(np.abs(h_ls))
        self.assertAlmostEqual(db_ls[0], 6.0, delta=0.2)
        self.assertAlmostEqual(db_ls[1], 0.0, delta=0.1)

        hs = BiquadMath.high_shelf(fs=48000, f0=2000.0, gain_db=-8.0, s=1.0)
        _, h_hs = BiquadMath.freq_response(hs, fs=48000, freqs=np.array([50.0, 18000.0]))
        db_hs = 20.0 * np.log10(np.abs(h_hs))
        self.assertAlmostEqual(db_hs[0], 0.0, delta=0.1)
        self.assertAlmostEqual(db_hs[1], -8.0, delta=0.3)

    def test_rbj_cut_matches_scipy_butterworth_2nd_order(self):
        """UC19: Compara `low_cut` e `high_cut` (Q=1/sqrt(2)) com `scipy.signal.butter(N=2)`."""
        fs = 48000
        f0 = 1000.0
        q_butter = 1.0 / math.sqrt(2.0)

        # High-cut (Lowpass)
        c_lp = BiquadMath.high_cut(fs, f0, q=q_butter)
        b_ref, a_ref = butter(N=2, Wn=f0 / (0.5 * fs), btype="low")
        b_rbj, a_rbj = c_lp.as_ba()
        np.testing.assert_allclose(b_rbj, b_ref, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(a_rbj, a_ref, rtol=1e-3, atol=1e-3)

        # Low-cut (Highpass)
        c_hp = BiquadMath.low_cut(fs, f0, q=q_butter)
        b_ref_hp, a_ref_hp = butter(N=2, Wn=f0 / (0.5 * fs), btype="high")
        b_rbj_hp, a_rbj_hp = c_hp.as_ba()
        np.testing.assert_allclose(b_rbj_hp, b_ref_hp, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(a_rbj_hp, a_ref_hp, rtol=1e-3, atol=1e-3)

    def test_notch_and_bandpass_and_allpass(self):
        """Valida Notch (rejeição profunda em f0), Band-Pass (0 dB em f0) e All-Pass (0 dB global)."""
        fs = 48000
        f0 = 1500.0
        c_notch = BiquadMath.notch(fs, f0, q=5.0)
        _, h_n = BiquadMath.freq_response(c_notch, fs, freqs=np.array([100.0, f0, 10000.0]))
        self.assertLess(20.0 * np.log10(max(abs(h_n[1]), 1e-12)), -60.0)
        self.assertAlmostEqual(20.0 * np.log10(abs(h_n[0])), 0.0, delta=0.05)

        c_bp = BiquadMath.band_pass(fs, f0, q=4.0)
        _, h_bp = BiquadMath.freq_response(c_bp, fs, freqs=np.array([100.0, f0, 10000.0]))
        self.assertAlmostEqual(20.0 * np.log10(abs(h_bp[1])), 0.0, places=4)
        self.assertLess(20.0 * np.log10(abs(h_bp[0])), -20.0)

        c_ap = BiquadMath.all_pass(fs, f0, q=1.0)
        _, h_ap = BiquadMath.freq_response(c_ap, fs, n=256)
        np.testing.assert_allclose(np.abs(h_ap), 1.0, atol=1e-6)

    def test_butterworth_cascades_6_to_96_db_oct(self):
        """
        PB05: Valida cascateamento Butterworth para Slopes S6, S12, S24, S48 e S96:
        - Número de biquads gerados: 1, 1, 2, 4, 8.
        - Ponto de corte -3.01 dB em f0.
        - Atenuação na oitava seguinte (2*f0 para HIGH_CUT) igual ao slope nominal.
        """
        fs = 96000
        f0 = 1000.0
        expected_stages = {
            Slope.S6: 1,
            Slope.S12: 1,
            Slope.S24: 2,
            Slope.S48: 4,
            Slope.S96: 8,
        }
        for slope, n_stages in expected_stages.items():
            stages = BiquadMath.cascade(fs=fs, f0=f0, slope=slope, ftype=FilterType.HIGH_CUT)
            self.assertEqual(len(stages), n_stages)
            _, h = BiquadMath.freq_response(stages, fs=fs, freqs=np.array([f0, 2.0 * f0]))
            db = 20.0 * np.log10(np.abs(h))
            # Em um filtro Butterworth de qualquer ordem, a magnitude em f0 é sempre -3.01 dB
            self.assertAlmostEqual(db[0], -3.0103, delta=0.25)
            # Em 2*f0 (1 oitava acima), a atenuação assintótica aproxima-se de -slope.value dB
            self.assertAlmostEqual(db[1], -float(slope.value), delta=3.5)

    def test_sample_rate_independence_pb29(self):
        """PB29: Garante que os cálculos funcionam dinamicamente de 44.1 kHz a 192 kHz."""
        for sr in SUPPORTED_SAMPLE_RATES:
            c = BiquadMath.peaking(fs=sr, f0=2500.0, gain_db=9.0, q=1.5)
            _, h = BiquadMath.freq_response(c, fs=sr, freqs=np.array([2500.0]))
            self.assertAlmostEqual(20.0 * np.log10(abs(h[0])), 9.0, places=3)

    def test_combined_response_multi_band_pb09(self):
        """PB09: Verifica que `combined_response` soma em dB os ganhos de bandas distintas."""
        fs = 48000
        b1 = EqBand(index=0, freq_hz=200.0, gain_db=6.0, q=4.0, ftype=FilterType.PEAKING, sample_rate=fs)
        b2 = EqBand(index=1, freq_hz=4000.0, gain_db=-6.0, q=4.0, ftype=FilterType.PEAKING, sample_rate=fs)
        freqs = np.array([200.0, 1000.0, 4000.0])
        mag_db = BiquadMath.combined_response([b1, b2], fs=fs, freqs=freqs)
        self.assertAlmostEqual(mag_db[0], 6.0, delta=0.2)
        self.assertAlmostEqual(mag_db[1], 0.0, delta=0.2)
        self.assertAlmostEqual(mag_db[2], -6.0, delta=0.2)


class TestSpectrumFFT(unittest.TestCase):
    """Valida o analisador de espectro `SpectrumFFT` (PB08)."""

    def test_sine_peak_detection_and_bin_to_hz(self):
        """Verifica que uma senoide pura de 1 kHz a 0 dBFS gera pico no bin exato próximo a 0 dBFS."""
        sr = 48000
        fft_size = 4096
        analyzer = SpectrumFFT(fft_size=fft_size, sample_rate=sr, smoothing_alpha=0.0)

        f_test = 1000.0
        t = np.arange(fft_size) / float(sr)
        sine = np.sin(2.0 * np.pi * f_test * t)
        analyzer.push_block(sine)

        freqs, db = analyzer.spectrum_curve()
        peak_bin = int(np.argmax(db))
        peak_freq = analyzer.bin_to_hz(peak_bin)
        self.assertAlmostEqual(peak_freq, f_test, delta=sr / fft_size)
        self.assertAlmostEqual(db[peak_bin], 0.0, delta=1.5)


if __name__ == "__main__":
    unittest.main()
