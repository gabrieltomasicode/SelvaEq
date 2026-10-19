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
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
from matplotlib.figure import Figure

from eq.config import EqConfig, FilterType, ChannelMode, InputSource
from eq.dsp.engine import EqEngine
from eq.dsp.spectrum import SpectrumFFT
from eq.io.audio_io import AudioIO
from eq.ui.decorators import validate_positive
from eq.ui.widgets import SelvaSlider, _DummyWidget, _styled_button
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

        draw_calls = []
        original_draw = slider.draw

        def counted_draw():
            draw_calls.append(1)
            original_draw()

        slider.draw = counted_draw
        slider.set(slider.value)
        self.assertEqual(draw_calls, [])
        slider.set(slider.value + 1.0)
        self.assertEqual(len(draw_calls), 1)

    def test_styled_button_setters_skip_unchanged_state(self):
        with patch(
            "eq.ui.widgets.tk",
            SimpleNamespace(Frame=_DummyWidget, Label=_DummyWidget),
        ):
            button = _styled_button(_DummyWidget(), "Teste", None)

        frame_config_calls = []
        label_config_calls = []
        button.config = lambda **kwargs: frame_config_calls.append(kwargs)
        button._label.config = lambda **kwargs: label_config_calls.append(kwargs)

        button.set_active(False)
        button.set_text("Teste")
        self.assertEqual(frame_config_calls, [])
        self.assertEqual(label_config_calls, [])

        button.set_active(True)
        button.set_active(True)
        button.set_text("Outro")
        button.set_text("Outro")
        self.assertEqual(len(frame_config_calls), 1)
        self.assertEqual(len(label_config_calls), 2)

    def test_periodic_status_refresh_only_updates_active_panel(self):
        ui = object.__new__(EqInterface)
        ui.current_tab = "bands"
        ui._tabs_initialized = False
        ui.panels = {tab: _DummyWidget() for tab in ("bands", "io", "test")}
        ui.tab_buttons = {
            tab: {"canvas": _DummyWidget(), "text": tab}
            for tab in ("bands", "io", "test")
        }
        tab_draws = {tab: 0 for tab in ui.tab_buttons}
        panel_actions = {tab: {"pack": 0, "forget": 0} for tab in ui.panels}
        for tab, data in ui.tab_buttons.items():
            data["canvas"].delete = lambda *args, _tab=tab: tab_draws.__setitem__(
                _tab, tab_draws[_tab] + 1
            )
        for tab, panel in ui.panels.items():
            panel.pack = lambda _tab=tab, **kwargs: panel_actions[_tab].__setitem__(
                "pack", panel_actions[_tab]["pack"] + 1
            )
            panel.pack_forget = lambda _tab=tab: panel_actions[_tab].__setitem__(
                "forget", panel_actions[_tab]["forget"] + 1
            )
        ui.clip_label = _DummyWidget()
        ui.meter_canvas = None
        ui.config = SimpleNamespace(input_source=SimpleNamespace(name="SYNTH_DRONE"))
        ui.io = SimpleNamespace(
            last_peak_dbfs=-12.0,
            last_rms_dbfs=-18.0,
            muted=False,
            last_stream_error="",
            stream_mode="simulated",
            media_streamer=SimpleNamespace(
                playing=False,
                loop=False,
                has_file=lambda: False,
            ),
        )
        io_status = _DummyWidget()
        media_status = _DummyWidget()
        ui.io_widgets = {"io_status_label": io_status}
        ui.test_widgets = {"media_info_label": media_status}

        ui.show_clip_indicator(False)
        self.assertEqual(io_status.cget("text"), "")
        self.assertEqual(media_status.cget("text"), "")

        ui.select_tab("io")
        self.assertNotEqual(io_status.cget("text"), "")
        self.assertEqual(media_status.cget("text"), "")
        first_tab_draws = dict(tab_draws)
        first_panel_actions = {
            tab: dict(actions) for tab, actions in panel_actions.items()
        }
        ui.select_tab("io")
        self.assertEqual(tab_draws, first_tab_draws)
        self.assertEqual(panel_actions, first_panel_actions)

        io_text = io_status.cget("text")
        ui.select_tab("bands")
        self.assertEqual(tab_draws, {"bands": 2, "io": 2, "test": 1})
        self.assertEqual(panel_actions["io"]["forget"], 1)
        self.assertEqual(panel_actions["bands"]["pack"], 1)
        ui.show_clip_indicator(False)
        self.assertEqual(io_status.cget("text"), io_text)
        self.assertEqual(media_status.cget("text"), "")
        ui.select_tab("test")
        self.assertEqual(tab_draws, {"bands": 3, "io": 2, "test": 2})
        self.assertNotEqual(media_status.cget("text"), "")

    def test_visual_refresh_logs_errors_and_keeps_scheduling(self):
        ui = object.__new__(EqInterface)
        ui.visual_paused = False
        ui.io = SimpleNamespace(
            visual_queue=SimpleNamespace(empty=lambda: True),
            clip_detected=False,
        )
        ui._update_window_title = Mock()
        ui.show_clip_indicator = Mock()
        ui.draw_theoretical_curve = Mock(side_effect=RuntimeError("curve failure"))
        ui.draw_spectrum = Mock()
        ui.canvas_mpl = None
        ui.master = SimpleNamespace(after=Mock(return_value="next-refresh"))

        with self.assertLogs("eq.ui.interface", level="ERROR") as captured:
            ui.update_visuals()

        self.assertIn("curve failure", captured.output[0])
        self.assertEqual(ui.master.after.call_args.args[0], 33)
        self.assertIs(ui.master.after.call_args.args[1].__self__, ui)
        self.assertEqual(ui.visual_update_id, "next-refresh")

    def test_close_cancels_pending_tkagg_draw(self):
        ui = object.__new__(EqInterface)
        ui._resize_job = None
        ui.visual_update_id = None
        ui.io = Mock()
        tk_canvas = SimpleNamespace(after_cancel=Mock())
        ui.canvas_mpl = SimpleNamespace(_idle_draw_id="draw-id", _tkcanvas=tk_canvas)
        ui.master = Mock()

        with patch("cleancache.remove_pycache_and_pyc") as clean_cache:
            ui.on_close()

        tk_canvas.after_cancel.assert_called_once_with("draw-id")
        self.assertIsNone(ui.canvas_mpl._idle_draw_id)
        ui.master.destroy.assert_called_once_with()
        clean_cache.assert_called_once()

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

    def test_curve_redraw_reuses_fill_and_band_annotation_artists(self):
        engine = EqEngine(EqConfig(sample_rate=48000))
        engine.add_band(freq_hz=1000.0, gain_db=3.0, q=1.0)
        ui = object.__new__(EqInterface)
        ui.engine = engine
        ui.config = engine.config
        ui.selected_band_index = 0
        ui.ax = Figure().add_subplot(111)
        ui.curve_line = None
        ui._curve_fill = None
        ui.selected_band_curve_line = None
        ui.legend = None
        ui.nodes_active_line = None
        ui.nodes_bypass_line = None
        ui.nodes_solo_line = None
        ui.selected_node_line = None
        ui._node_annotations = []

        ui.draw_theoretical_curve()
        fill_artist = ui._curve_fill
        annotation = ui._node_annotations[0]
        self.assertEqual(len(ui.ax.texts), 1)

        engine.set_param(0, "gain_db", 6.0)
        ui.draw_theoretical_curve()
        self.assertIs(ui._curve_fill, fill_artist)
        self.assertIs(ui._node_annotations[0], annotation)
        self.assertEqual(annotation.get_text(), "1")
        self.assertEqual(annotation.xy, (engine.bands[0].freq_hz, engine.bands[0].gain_db))

        engine.add_band(freq_hz=2000.0, gain_db=-3.0, q=1.0)
        ui.draw_theoretical_curve()
        self.assertEqual(len(ui._node_annotations), 2)
        engine.remove_band(1)
        ui.draw_theoretical_curve()
        self.assertEqual(len(ui._node_annotations), 1)
        self.assertEqual(len(ui.ax.texts), 1)

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
