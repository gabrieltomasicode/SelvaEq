"""
Testes Unitários de Isolamento Arquitetural (PB28), Interface Gráfica Responsiva / Tela Cheia,
Criação/Edição/Exclusão de Bandas no Gráfico de Espectro (`InteractiveCurveController`, `BandNode`),
Abas de I/O e Modo de Teste MP3/WAV, e Interoperabilidade Ponta-a-Ponta MVC.
"""

import ast
import glob
import os
import tempfile
import unittest
import numpy as np

from eq.config import EqConfig, FilterType, ChannelMode, InputSource
from eq.dsp.engine import EqEngine
from eq.dsp.spectrum import SpectrumFFT
from eq.io.audio_io import AudioIO
from eq.ui.decorators import validate_positive
from eq.ui.widgets import SelvaSlider
from eq.ui.curve_view import BandNode
from eq.ui.controls.band_controls import BandControls
from eq.ui.interface import EqInterface


class _DummyMplEvent:
    """Simula um evento de mouse/scroll do Matplotlib sobre o eixo do espectro."""
    def __init__(self, inaxes, xdata, ydata, button=1, dblclick=False, key=None):
        self.inaxes = inaxes
        self.xdata = xdata
        self.ydata = ydata
        self.button = button
        self.dblclick = dblclick
        self.key = key
        self.guiEvent = None


class TestArchitecturalIsolationPB28(unittest.TestCase):
    """
    PB28: Garante que NENHUM módulo de `eq/dsp/` ou `eq/io/` (incluindo `media_player.py`)
    importe `tkinter`, `matplotlib` ou submódulos de `eq.ui`.
    """

    def test_zero_gui_imports_in_dsp_and_io(self):
        forbidden = {"tkinter", "matplotlib", "eq.ui", "ui"}
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "eq"))
        target_files = (
            glob.glob(os.path.join(base_dir, "config.py"))
            + glob.glob(os.path.join(base_dir, "dsp", "*.py"))
            + glob.glob(os.path.join(base_dir, "io", "*.py"))
        )
        self.assertGreaterEqual(len(target_files), 9)

        for fpath in target_files:
            with open(fpath, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=fpath)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        root_mod = alias.name.split(".")[0]
                        self.assertNotIn(
                            root_mod,
                            forbidden,
                            f"Violação PB28 em {fpath}: import {alias.name}",
                        )
                elif isinstance(node, ast.ImportFrom) and node.module:
                    root_mod = node.module.split(".")[0]
                    self.assertNotIn(
                        root_mod,
                        forbidden,
                        f"Violação PB28 em {fpath}: from {node.module} import ...",
                    )
                    self.assertFalse(
                        node.module.startswith("eq.ui"),
                        f"Violação PB28 em {fpath}: dependência reversa para {node.module}",
                    )


class TestUIAndMVCInteroperability(unittest.TestCase):
    """
    Valida os componentes de UI/Controller, responsividade/tela cheia, criação/edição/exclusão
    de bandas no gráfico de espectro e interoperabilidade total com o motor DSP e Player MP3/WAV.
    """

    def test_validate_positive_decorator(self):
        @validate_positive
        def set_freq(f, q=1.0):
            return f * q

        self.assertEqual(set_freq(1000.0, q=2.0), 2000.0)
        with self.assertRaises(ValueError):
            set_freq(-50.0, q=1.0)
        with self.assertRaises(ValueError):
            set_freq(100.0, q=-0.5)

    def test_selva_slider_interaction_and_responsive_resize(self):
        received = []
        slider = SelvaSlider(width=200, from_=20.0, to_=20000.0, init=1000.0, command=lambda v: received.append(v))
        self.assertAlmostEqual(slider.value, 1000.0)
        slider.set(5000.0)
        self.assertAlmostEqual(slider.value, 5000.0)
        slider._update_val_from_x(100.0)
        self.assertEqual(len(received), 1)
        self.assertTrue(20.0 <= received[0] <= 20000.0)

        # Simula redimensionamento da tela (evento <Configure>)
        class _ResizeEv:
            width = 420
        slider._on_resize(_ResizeEv())
        self.assertEqual(slider.width, 420)

    def test_band_node_log_coordinates_drag_and_wheel_pb21(self):
        """PB21: Valida conversões logarítmicas e interação de arraste/roda do `BandNode` com `EqEngine`."""
        engine = EqEngine(EqConfig(sample_rate=48000))
        engine.add_band(freq_hz=1000.0, gain_db=0.0, q=1.0, ftype=FilterType.PEAKING)
        node = BandNode(band_index=0, engine=engine, canvas_width=1000.0, canvas_height=400.0)

        self.assertTrue(node.hit_test(node.x_px, node.y_px))
        self.assertFalse(node.hit_test(node.x_px + 50.0, node.y_px + 50.0))

        target_x = BandNode.freq_to_x(2000.0, 1000.0)
        target_y = BandNode.gain_to_y(12.0, 400.0)
        node.drag_to(target_x, target_y)

        self.assertAlmostEqual(engine.bands[0].freq_hz, 2000.0, delta=1.0)
        self.assertAlmostEqual(engine.bands[0].gain_db, 12.0, delta=0.05)

        node.on_wheel(delta=2.0)
        self.assertGreater(engine.bands[0].q, 1.0)

        node.toggle_solo()
        self.assertTrue(engine.bands[0].solo)
        node.toggle_bypass()
        self.assertTrue(engine.bands[0].bypass)

    def test_create_edit_and_delete_bands_directly_on_spectrum_graph(self):
        """
        Valida o ciclo completo de interação das bandas no gráfico de espectro:
        1. Duplo clique em ponto vazio cria uma nova banda (ex: 2500 Hz, +9 dB).
        2. Clique esquerdo e arraste edita frequência e ganho da banda em tempo real.
        3. Roda do mouse altera o Fator Q da banda no gráfico.
        4. Clique direito sobre o nó no gráfico exclui a banda e reindexa as demais.
        5. Undo restaura a banda excluída.
        """
        cfg = EqConfig(sample_rate=48000, buffer_size=256)
        engine = EqEngine(config=cfg, init_default_bands=False)
        engine.add_band(freq_hz=200.0, gain_db=0.0, q=1.0)
        io = AudioIO(config=cfg, engine=engine)
        ui = EqInterface(engine=engine, io=io)

        self.assertEqual(len(engine.bands), 1)

        # 1. Duplo clique em (2500 Hz, +9.0 dB) para CRIAR uma 2ª banda
        ev_dbl = _DummyMplEvent(ui.ax, xdata=2500.0, ydata=9.0, button=1, dblclick=True)
        ui._on_mpl_mouse_down(ev_dbl)
        self.assertEqual(len(engine.bands), 2)
        self.assertAlmostEqual(engine.bands[1].freq_hz, 2500.0, places=2)
        self.assertAlmostEqual(engine.bands[1].gain_db, 9.0, places=2)

        # 2. Clique esquerdo e arraste da nova banda para (4000 Hz, -6.0 dB)
        ev_down = _DummyMplEvent(ui.ax, xdata=2500.0, ydata=9.0, button=1, dblclick=False)
        ui._on_mpl_mouse_down(ev_down)
        ev_drag = _DummyMplEvent(ui.ax, xdata=4000.0, ydata=-6.0, button=1)
        ui._on_mpl_mouse_drag(ev_drag)
        ui._on_mpl_mouse_up(ev_drag)
        self.assertAlmostEqual(engine.bands[1].freq_hz, 4000.0, places=2)
        self.assertAlmostEqual(engine.bands[1].gain_db, -6.0, places=2)

        # 3. Scroll da roda do mouse sobre o nó em (4000 Hz, -6.0 dB) aumenta o Q
        old_q = engine.bands[1].q
        ev_scroll = _DummyMplEvent(ui.ax, xdata=4000.0, ydata=-6.0, button="up")
        ui._on_mpl_scroll(ev_scroll)
        self.assertGreater(engine.bands[1].q, old_q)

        # 4. Clique direito (button=3) sobre o nó em (4000 Hz, -6.0 dB) EXCLUI a banda!
        ev_right = _DummyMplEvent(ui.ax, xdata=4000.0, ydata=-6.0, button=3)
        ui._on_mpl_mouse_down(ev_right)
        self.assertEqual(len(engine.bands), 1)

        # 5. Undo restaura a banda excluída
        ui.undo_action()
        self.assertEqual(len(engine.bands), 2)
        ui.on_close()

    def test_tabs_fullscreen_and_test_mode_media_pipeline(self):
        """
        Valida a navegação entre as 3 abas ('bands', 'io', 'test'), o modo Tela Cheia (F11/Esc),
        a geração/carregamento de faixa de teste no fluxo em tempo real e a atualização visual.
        """
        cfg = EqConfig(sample_rate=48000, buffer_size=256)
        engine = EqEngine(config=cfg, init_default_bands=True)
        io = AudioIO(config=cfg, engine=engine)
        analyzer = SpectrumFFT(fft_size=1024, sample_rate=cfg.sample_rate)
        ui = EqInterface(engine=engine, io=io, analyzer=analyzer)

        # Navegação entre as 3 abas
        for tab in ("io", "test", "bands"):
            ui.select_tab(tab)
            self.assertEqual(ui.current_tab, tab)

        # Modo Tela Cheia (Fullscreen toggle e saída)
        self.assertFalse(ui.is_fullscreen)
        ui.toggle_fullscreen()
        self.assertTrue(ui.is_fullscreen)
        ui.set_fullscreen(False)
        self.assertFalse(ui.is_fullscreen)

        # Edita Banda 4 (1 kHz) para +10 dB
        BandControls.update_band_freq(engine, 3, 1000.0)
        BandControls.update_band_gain(engine, 3, 10.0)
        BandControls.update_band_q(engine, 3, 2.0)
        BandControls.update_band_mode(engine, 3, ChannelMode.STEREO)

        # Gera e carrega faixa demo no Modo de Teste em Tempo Real
        with tempfile.TemporaryDirectory() as tmpdir:
            demo_wav = os.path.join(tmpdir, "demo.wav")
            ui.generate_and_play_demo_track(demo_wav)
            self.assertEqual(ui.config.input_source, InputSource.MEDIA_FILE)
            self.assertTrue(io.media_streamer.has_file())

            # Roda callbacks de saída usando o arquivo carregado como entrada em tempo real
            out_stereo = np.zeros((256, 2), dtype=np.float32)
            for _ in range(4):
                io._output_only_callback(out_stereo, 256, None, None)

            ui.update_visuals()
            freqs, mag_db = ui.draw_theoretical_curve()
            idx_1k = int(np.argmin(np.abs(freqs - 1000.0)))
            self.assertAlmostEqual(mag_db[idx_1k], 10.0, delta=0.35)

            # Testa salvamento e carregamento de preset
            preset_file = os.path.join(tmpdir, "mvc_test.json")
            ui.save_preset_dialog(preset_file)
            BandControls.update_band_gain(engine, 3, -5.0)
            ui.load_preset_dialog(preset_file)
            self.assertAlmostEqual(engine.bands[3].gain_db, 10.0)

        ui.on_close()


if __name__ == "__main__":
    unittest.main()
