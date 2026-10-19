"""
Interface Gráfica Principal do Equalizador (`EqInterface` / `PyGraphUI` - `eq/ui/interface.py`).

Refatorado a partir de `ui/interface.py` (`FullSynthInterface`) do SelvaSonic:
- Layout 100% responsivo a mudanças de tamanho de janela (`<Configure>`, `rowconfigure`, `columnconfigure`),
  redimensionamento real da `Figure` Matplotlib com debounce de `60 ms` sem faixa branca (D01),
  cabeçalho sem truncamento com badge `"● SALVO"` / `"● NÃO SALVO"` (D02), altura fixa unificada do
  painel inferior sem pulo entre abas (D04) e suporte nativo a Tela Cheia (`toggle_fullscreen` via
  botão ou tecla `F11` e saída com `Escape`).
- Sistema de 3 Abas Modulares (estilo SelvaSonic) com ícones e alto contraste (D07):
  1. "🎚 1. Equalizador & Bandas" (`bands`): edição paramétrica das 24 bandas + Flat + Navegação + Undo/Redo/A-B (D03).
  2. "🔌 2. Entrada & Saída de Áudio" (`io`): seleção independente de dispositivos de Entrada e Saída,
     Sample Rate, Buffer Size, Oversampling, Ganho Master, Mid/Side ON/OFF e LED de status (D09).
  3. "▶ 3. Modo de Teste (MP3 / WAV)" (`test`): injeção de arquivos `.mp3`/`.wav` em tempo real no
     fluxo de áudio via `RealtimeMediaStreamer`, controles de transporte agrupados, presets de fábrica,
     exportação WAV, Freeze FFT e gerador de faixa demo (D10).
- Gráfico de Espectro e Curva Interativo (`InteractiveCurveController` + `BandNode`) com eixos e grade
  de década de alto contraste, fill suave (`alpha=0.06`), legenda compacta (D06) e barra inferior
  contextual dinâmica (D05):
  - Duplo clique esquerdo sobre nó existente: reseta ganho da banda para `0.0 dB` (F01).
  - Duplo clique esquerdo (ou `Shift + Clique esquerdo`) em área livre: cria nova banda.
  - Arrastar com clique esquerdo: edita Frequência e Ganho com HUD flutuante em tempo real (V06).
  - Roda do mouse: ajusta Fator Q com suporte a Undo/Redo (F03).
  - `Alt + Clique` (imune a NumLock - F02): Solo Band-Pass; `Ctrl + Clique`: Bypass.
  - Clique direito (`button == 3`) sobre o nó ou tecla `Delete`: exclui banda.
"""

import os
import sys
import tempfile
from queue import Empty
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import matplotlib

from eq.config import EqConfig, FilterType, InputSource
from eq.dsp.biquad_math import BiquadMath
from eq.dsp.engine import EqEngine
from eq.dsp.spectrum import SpectrumFFT
from eq.io.audio_io import AudioIO
from eq.io.media_player import RealtimeMediaStreamer
from eq.io.preset_io import PresetIO
from eq.ui.curve_view import BandNode, InteractiveCurveController
from eq.ui.controls.band_controls import BandControls
from eq.ui.widgets import (
    HAS_TK,
    tk,
    C_AMARELO,
    C_VERDE,
    C_VERDE_CLARO,
    C_MARROM,
    C_AZUL,
    C_VERMELHO,
    C_PRETO,
    C_ROXO,
    C_BG,
    C_BG2,
    C_BG3,
    C_TEXTO,
    C_TEXTO2,
    C_EIXO,
    F_BEBAS,
    F_BARLOW,
    F_SPACE,
    apply_dark_ttk_style,
    _styled_button,
    create_band_panel_widgets,
    create_audio_io_panel_widgets,
    create_test_mode_panel_widgets,
)

if not HAS_TK:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

if HAS_TK:
    from tkinter import filedialog, messagebox
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
else:
    filedialog = None
    messagebox = None
    FigureCanvasTkAgg = None


class EqInterface:
    """
    Interface gráfica responsiva e controlador MVC em tempo quase-real do SelvaEQ.
    """

    def __init__(
        self,
        master=None,
        engine: Optional[EqEngine] = None,
        io: Optional[AudioIO] = None,
        analyzer: Optional[SpectrumFFT] = None,
    ):
        self.master = master if master is not None else tk.Tk()
        self.engine: EqEngine = engine if engine is not None else EqEngine(init_default_bands=True)
        self.config: EqConfig = self.engine.config
        self.io: AudioIO = io if io is not None else AudioIO(config=self.config, engine=self.engine)
        self.analyzer: SpectrumFFT = (
            analyzer if analyzer is not None else SpectrumFFT(fft_size=2048, sample_rate=self.config.sample_rate)
        )

        # Registra observer para sincronizar SpectrumFFT caso a placa force fallback de sample rate (T09)
        self.io.add_sample_rate_listener(self._on_hardware_sample_rate_changed)

        # Controlador MVC da curva interativa e nós (criação, edição e exclusão no gráfico)
        self.curve_controller = InteractiveCurveController(self.engine, canvas_width=900.0, canvas_height=340.0)
        self.selected_band_index: int = min(3, max(0, len(self.engine.bands) - 1))
        self.curve_controller.selected_index = self.selected_band_index

        self.visual_update_id = None
        self._resize_job = None
        self.dirty_badge_label = None
        self.footer_label = None
        self.legend = None
        self.visual_paused: bool = False
        self.clip_indicator_state: bool = False
        self.is_fullscreen: bool = False
        self.current_tab: str = "bands"
        self._tabs_initialized: bool = False
        self.tab_buttons: Dict[str, dict] = {}
        self.panels: Dict[str, Any] = {}
        self._node_annotations: List[Any] = []
        self.ax_fft = None

        self._configure_responsive_window()
        self._build_header()
        self._build_curve_area()
        self._build_tabs_and_panels()
        self._build_footer()
        self.setup_visuals()
        self._sync_nodes()
        self._sync_controls_from_selected_band()
        self._setup_keyboard_shortcuts()

        self.io.start()
        self.update_visuals()

    @property
    def nodes(self) -> List[BandNode]:
        return self.curve_controller.nodes

    @nodes.setter
    def nodes(self, val: List[BandNode]) -> None:
        self.curve_controller.nodes = val

    def _on_hardware_sample_rate_changed(self, new_sr: int) -> None:
        """Callback Observer acionado quando `AudioIO` aplica fallback de sample rate (T09)."""
        self.analyzer.set_sample_rate(int(new_sr))
        try:
            if hasattr(self, "io_widgets") and "sample_rate" in self.io_widgets:
                self.io_widgets["sample_rate"].set(int(new_sr))
        except Exception:
            pass

    def _update_window_title(self) -> None:
        """Atualiza o título da janela e o badge visual ('● SALVO' vs '● NÃO SALVO') (F11 / D02)."""
        is_dirty = bool(self.engine.state_controller.dirty)
        base_title = "SelvaEQ - 24-Band Parametric Equalizer"
        dirty_suffix = " *" if is_dirty else ""
        if hasattr(self.master, "title"):
            try:
                self.master.title(f"{base_title}{dirty_suffix}")
            except Exception:
                pass
        if getattr(self, "dirty_badge_label", None) is not None:
            try:
                if is_dirty:
                    self.dirty_badge_label.config(
                        text="● NÃO SALVO",
                        fg=C_AMARELO,
                        highlightbackground=C_AMARELO,
                    )
                else:
                    self.dirty_badge_label.config(
                        text="● SALVO",
                        fg=C_TEXTO2,
                        highlightbackground=C_MARROM,
                    )
            except Exception:
                pass

    def _transact(self, name: str, action: Callable) -> Any:
        """Executa `action()` dentro de uma transação atômica do `StateController` (F03)."""
        sc = self.engine.state_controller
        started = False
        if sc._pending_snapshot is None:
            sc.begin_transaction(name)
            started = True
        try:
            return action()
        finally:
            if started:
                sc.commit()
            self._update_window_title()

    def _sync_nodes(self) -> None:
        """Sincroniza os objetos `BandNode` com as bandas ativas no `EqEngine`."""
        self.curve_controller.sync_nodes()
        if len(self.engine.bands) > 0:
            self.selected_band_index = int(
                max(0, min(self.selected_band_index, len(self.engine.bands) - 1))
            )
            self.curve_controller.selected_index = self.selected_band_index
        else:
            self.selected_band_index = -1
            self.curve_controller.selected_index = -1

    # ─────────────────────────────────────────────────────────────────────────
    #  Configuração de Janela Responsiva, Tema Escuro e Modo Tela Cheia
    # ─────────────────────────────────────────────────────────────────────────
    def _configure_responsive_window(self) -> None:
        """Configura tema escuro ttk (V01), pesos de grid responsivos e painel inferior com altura fixa (D01 / D04)."""
        apply_dark_ttk_style(self.master)
        self._update_window_title()
        if hasattr(self.master, "configure"):
            self.master.configure(bg=C_BG)
        if hasattr(self.master, "geometry"):
            try:
                self.master.geometry("1180x780")
            except Exception:
                pass
        if hasattr(self.master, "minsize"):
            try:
                self.master.minsize(880, 600)
            except Exception:
                pass
        if hasattr(self.master, "columnconfigure"):
            self.master.columnconfigure(0, weight=1)
        if hasattr(self.master, "rowconfigure"):
            self.master.rowconfigure(0, weight=0)
            self.master.rowconfigure(1, weight=1)               # Gráfico expande em 100% da área livre (D01 / D04)
            self.master.rowconfigure(2, weight=0)               # Barra de abas fixa
            self.master.rowconfigure(3, weight=0, minsize=235)  # Painel inferior com altura fixa constante (D04)
            self.master.rowconfigure(4, weight=0)               # Barra inferior de dicas fixa
        if hasattr(self.master, "protocol"):
            self.master.protocol("WM_DELETE_WINDOW", self.on_close)

    def _is_typing_in_entry(self) -> bool:
        """Verifica se o foco do teclado está dentro de um `Entry` de edição numérica (F12)."""
        if not hasattr(self.master, "focus_get"):
            return False
        try:
            focused = self.master.focus_get()
            if focused is None:
                return False
            if HAS_TK and isinstance(focused, tk.Entry):
                return True
            wclass = str(focused.winfo_class()) if hasattr(focused, "winfo_class") else ""
            return wclass in ("Entry", "TEntry")
        except Exception:
            return False

    def _setup_keyboard_shortcuts(self) -> None:
        """Configura atalhos globais protegidos contra conflito de foco em campos de texto (F08 / F12)."""
        if not hasattr(self.master, "bind_all"):
            return
        self.master.bind_all("<F11>", lambda e: self.toggle_fullscreen())
        self.master.bind_all("<Escape>", lambda e: self.set_fullscreen(False))
        self.master.bind_all(
            "<Delete>",
            lambda e: None if self._is_typing_in_entry() else self.delete_selected_band(),
        )
        self.master.bind_all(
            "<BackSpace>",
            lambda e: None if self._is_typing_in_entry() else self.delete_selected_band(),
        )
        self.master.bind_all("<Control-z>", lambda e: self.undo_action())
        self.master.bind_all("<Control-y>", lambda e: self.redo_action())
        self.master.bind_all("<Control-b>", lambda e: self.toggle_global_bypass())
        self.master.bind_all("<Control-s>", lambda e: self.save_preset_dialog())
        self.master.bind_all("<Control-o>", lambda e: self.load_preset_dialog())
        self.master.bind_all(
            "<space>",
            lambda e: None if self._is_typing_in_entry() else self.toggle_media_play_pause(),
        )
        self.master.bind_all(
            "<Left>",
            lambda e: None if self._is_typing_in_entry() else self.select_prev_band(),
        )
        self.master.bind_all(
            "<Right>",
            lambda e: None if self._is_typing_in_entry() else self.select_next_band(),
        )

    def set_fullscreen(self, state: bool) -> bool:
        """Ativa ou desativa o modo de Tela Cheia (Fullscreen) atualizando o botão correspondente (V08)."""
        self.is_fullscreen = bool(state)
        try:
            if hasattr(self.master, "attributes"):
                self.master.attributes("-fullscreen", self.is_fullscreen)
        except Exception:
            try:
                if hasattr(self.master, "state"):
                    self.master.state("zoomed" if self.is_fullscreen else "normal")
            except Exception:
                pass
        if hasattr(self, "fullscreen_btn") and hasattr(self.fullscreen_btn, "set_active"):
            self.fullscreen_btn.set_active(self.is_fullscreen)
        return self.is_fullscreen

    def toggle_fullscreen(self, event=None) -> bool:
        """Alterna entre modo Janela Redimensionável e Tela Cheia (F11)."""
        return self.set_fullscreen(not self.is_fullscreen)

    # ─────────────────────────────────────────────────────────────────────────
    #  Construção do Cabeçalho e Gráfico Interativo de Espectro
    # ─────────────────────────────────────────────────────────────────────────
    def _build_header(self) -> None:
        """Constrói o cabeçalho sem truncamento com badge '● SALVO / ● NÃO SALVO', medidor Pico/RMS, Bypass e Tela Cheia (D02)."""
        self.header_frame = tk.Frame(self.master, bg=C_PRETO)
        self.header_frame.grid(row=0, column=0, sticky="ew")
        if hasattr(self.header_frame, "columnconfigure"):
            self.header_frame.columnconfigure(0, weight=1)

        # D02: Empacota right_hdr ANTES de left_hdr para nunca cortar o botão "⛶ Tela Cheia (F11)"
        right_hdr = tk.Frame(self.header_frame, bg=C_PRETO)
        right_hdr.pack(side="right", padx=12, pady=4)

        self.fullscreen_btn = _styled_button(
            right_hdr, "⛶ Tela Cheia (F11)", self.toggle_fullscreen, accent=C_AZUL
        )
        self.fullscreen_btn.pack(side="right", padx=(8, 0))

        self.global_bypass_btn = _styled_button(
            right_hdr, "⏻ Bypass Global", self.toggle_global_bypass, accent=C_AMARELO
        )
        self.global_bypass_btn.pack(side="right", padx=(8, 0))

        self.clip_label = tk.Label(
            right_hdr,
            text="● CLIP: OK (-∞ dBFS)",
            font=F_SPACE,
            bg=C_PRETO,
            fg=C_VERDE_CLARO,
            cursor="hand2",
        )
        self.clip_label.pack(side="right", padx=6)
        self.clip_label.bind("<Button-1>", lambda e: self.reset_clip_indicator())

        self.meter_canvas = tk.Canvas(
            right_hdr,
            width=110,
            height=14,
            bg=C_BG3,
            highlightthickness=1,
            highlightbackground=C_MARROM,
            cursor="hand2",
        )
        self.meter_canvas.pack(side="right", padx=(4, 4))
        self.meter_canvas.bind("<Button-1>", lambda e: self.reset_clip_indicator())

        left_hdr = tk.Frame(self.header_frame, bg=C_PRETO)
        left_hdr.pack(side="left", fill="x", expand=True, padx=12, pady=4)

        self.title_label = tk.Label(
            left_hdr,
            text="🎚 SELVAEQ — EQUALIZADOR PARAMÉTRICO 24 BANDAS",
            font=F_BEBAS,
            bg=C_PRETO,
            fg=C_AMARELO,
        )
        self.title_label.pack(side="left")

        self.dirty_badge_label = tk.Label(
            left_hdr,
            text="● SALVO",
            font=F_SPACE,
            bg=C_BG3,
            fg=C_TEXTO2,
            padx=8,
            pady=2,
            highlightthickness=1,
            highlightbackground=C_MARROM,
        )
        self.dirty_badge_label.pack(side="left", padx=(10, 0))
        self._update_window_title()

    def reset_clip_indicator(self) -> None:
        """Reseta a trava temporal de clipping no `AudioIO` e atualiza o cabeçalho (V07 / T08)."""
        self.io.reset_clip()
        self.show_clip_indicator(False)

    def _build_curve_area(self) -> None:
        """Constrói a área responsiva do gráfico espectral logarítmico sem faixa branca (D01 / D04)."""
        self.curve_container = tk.Frame(self.master, bg=C_PRETO, highlightthickness=1, highlightbackground=C_MARROM)
        self.curve_container.grid(row=1, column=0, sticky="nsew", padx=12, pady=(4, 4))
        if hasattr(self.curve_container, "columnconfigure"):
            self.curve_container.columnconfigure(0, weight=1)
            self.curve_container.rowconfigure(0, weight=1)
        self.curve_container.bind("<Configure>", self._on_curve_canvas_resize)

        self.fig, self.ax = plt.subplots(figsize=(9.5, 3.6), facecolor=C_PRETO)
        self.fig.patch.set_facecolor(C_PRETO)
        self.fig.subplots_adjust(left=0.06, right=0.93, top=0.93, bottom=0.14)
        self.draw_log_axes()

        if HAS_TK and FigureCanvasTkAgg is not None:
            self.canvas_mpl = FigureCanvasTkAgg(self.fig, master=self.curve_container)
            tk_wid = self.canvas_mpl.get_tk_widget()
            tk_wid.configure(bg=C_PRETO, highlightthickness=0)
            tk_wid.pack(fill="both", expand=True, padx=0, pady=0)
            self.fig.canvas.mpl_connect("button_press_event", self._on_mpl_mouse_down)
            self.fig.canvas.mpl_connect("motion_notify_event", self._on_mpl_mouse_drag)
            self.fig.canvas.mpl_connect("button_release_event", self._on_mpl_mouse_up)
            self.fig.canvas.mpl_connect("scroll_event", self._on_mpl_scroll)
        else:
            self.canvas_mpl = None

    def _on_curve_canvas_resize(self, event) -> None:
        """Agenda com debounce de 60 ms o redimensionamento real da `Figure` Matplotlib (D01)."""
        w = int(getattr(event, "width", 0) or 0)
        h = int(getattr(event, "height", 0) or 0)
        if w < 120 or h < 90:
            return
        self.curve_controller.set_canvas_size(float(w), float(h))
        if self._resize_job is not None and hasattr(self.master, "after_cancel"):
            try:
                self.master.after_cancel(self._resize_job)
            except Exception:
                pass
            self._resize_job = None
        if HAS_TK and hasattr(self.master, "after"):
            self._resize_job = self.master.after(60, lambda: self._apply_figure_resize(w, h))
        else:
            self._apply_figure_resize(w, h)

    def _apply_figure_resize(self, width_px: int, height_px: int) -> None:
        """Aplica as novas dimensões em polegadas e margens proporcionais à `Figure` Matplotlib (D01)."""
        self._resize_job = None
        if width_px < 120 or height_px < 90:
            return
        self.curve_controller.set_canvas_size(float(width_px), float(height_px))
        dpi = float(self.fig.get_dpi() or 100.0)
        self.fig.set_size_inches(max(4.0, width_px / dpi), max(1.8, height_px / dpi), forward=False)
        left_m = max(0.055, min(0.095, 64.0 / max(width_px, 400)))
        right_m = min(0.945, max(0.905, 1.0 - (64.0 / max(width_px, 400))))
        bottom_m = max(0.14, min(0.24, 42.0 / max(height_px, 180)))
        top_m = min(0.94, max(0.88, 1.0 - (18.0 / max(height_px, 180))))
        self.fig.subplots_adjust(left=left_m, right=right_m, top=top_m, bottom=bottom_m)
        if self.canvas_mpl is not None:
            self.canvas_mpl.draw_idle()

    # ─────────────────────────────────────────────────────────────────────────
    #  Sistema de Abas (Equalizador | Entrada/Saída | Modo de Teste MP3/WAV)
    # ─────────────────────────────────────────────────────────────────────────
    def _build_tabs_and_panels(self) -> None:
        """Constrói a barra de abas com ícones (D07) e `content_area` de altura fixa unificada (D04)."""
        self.tab_bar = tk.Frame(self.master, bg=C_PRETO)
        self.tab_bar.grid(row=2, column=0, sticky="ew", padx=12)

        tab_container = tk.Frame(self.tab_bar, bg=C_PRETO)
        tab_container.pack(side="left", pady=(2, 0))

        tabs_info = [
            ("bands", "🎚 1. Equalizador & Bandas"),
            ("io", "🔌 2. Entrada & Saída de Áudio"),
            ("test", "▶ 3. Modo de Teste (MP3 / WAV)"),
        ]

        for tab_id, text in tabs_info:
            btn = tk.Canvas(tab_container, width=235, height=30, bg=C_PRETO, highlightthickness=0, cursor="hand2")
            btn.pack(side="left", padx=2)
            btn.bind("<Button-1>", lambda e, tid=tab_id: self.select_tab(tid))
            self.tab_buttons[tab_id] = {"canvas": btn, "text": text}

        self.content_area = tk.Frame(
            self.master,
            bg=C_BG2,
            height=235,
            highlightthickness=1,
            highlightbackground=C_AMARELO,
        )
        self.content_area.grid(row=3, column=0, sticky="nsew", padx=12, pady=(0, 4))
        self.content_area.pack_propagate(False)
        self.content_area.grid_propagate(False)
        if hasattr(self.content_area, "columnconfigure"):
            self.content_area.columnconfigure(0, weight=1)
            self.content_area.rowconfigure(0, weight=1)

        self._build_band_panel()
        self._build_io_panel()
        self._build_test_panel()
        self.select_tab("bands")

    def select_tab(self, tab_id: str) -> None:
        """Alterna entre as abas 'bands', 'io' e 'test' com alto contraste nas abas inativas (D07)."""
        if tab_id not in self.panels:
            return
        previous_tab = self.current_tab
        if self._tabs_initialized and previous_tab == tab_id:
            self._sync_active_tab_status()
            return

        self.current_tab = tab_id
        tab_ids = self.tab_buttons if not self._tabs_initialized else (previous_tab, tab_id)
        for tid in tab_ids:
            data = self.tab_buttons[tid]
            c = data["canvas"]
            text = data["text"]
            c.delete("all")
            pts = [6, 0, 235, 0, 229, 30, 0, 30]
            if tid == tab_id:
                c.create_polygon(pts, fill=C_AMARELO, outline="")
                c.create_text(117, 15, text=text.upper(), font=F_BARLOW, fill=C_PRETO)
            else:
                c.create_polygon(pts, fill=C_BG3, outline=C_MARROM)
                c.create_text(117, 15, text=text.upper(), font=F_BARLOW, fill=C_TEXTO)

        if self._tabs_initialized:
            self.panels[previous_tab].pack_forget()
        self.panels[tab_id].pack(fill="both", expand=True)
        self._tabs_initialized = True
        self._sync_active_tab_status()

    def _build_band_panel(self) -> None:
        """Constrói a Aba 1: Controles das Bandas."""
        container = tk.Frame(self.content_area, bg=C_BG2)
        self.panels["bands"] = container

        self.band_widgets = create_band_panel_widgets(
            container,
            on_band_select=self._on_band_selected,
            on_type_change=self._on_type_changed,
            on_freq_change=lambda v: self._on_slider_freq(v),
            on_gain_change=lambda v: self._on_slider_gain(v),
            on_q_change=lambda v: self._on_slider_q(v),
            on_slope_change=self._on_slope_changed,
            on_mode_change=self._on_mode_changed,
            on_bypass_toggle=self._on_bypass_toggled,
            on_solo_toggle=self._on_solo_toggled,
            on_add_band=self._create_smart_band_from_button,
            on_delete_band=self.delete_selected_band,
            on_undo=self.undo_action,
            on_redo=self.redo_action,
            on_ab_compare=self.ab_compare_action,
            on_flatten=self.flatten_all_bands,
            on_prev_band=self.select_prev_band,
            on_next_band=self.select_next_band,
            on_freq_press=lambda: self.engine.state_controller.begin_transaction("Ajustar Frequência"),
            on_freq_release=self._commit_slider_transaction,
            on_gain_press=lambda: self.engine.state_controller.begin_transaction("Ajustar Ganho"),
            on_gain_release=self._commit_slider_transaction,
            on_q_press=lambda: self.engine.state_controller.begin_transaction("Ajustar Fator Q"),
            on_q_release=self._commit_slider_transaction,
        )

    def _commit_slider_transaction(self) -> None:
        self.engine.state_controller.commit()
        self._update_window_title()

    def _create_smart_band_from_button(self) -> Optional[int]:
        """Cria banda pelo botão da Aba 1 escolhendo uma frequência livre (F07)."""
        free_f = self.curve_controller.suggest_free_frequency(1000.0)
        return self.create_band_on_graph(free_f, 0.0)

    def _build_io_panel(self) -> None:
        """Constrói a Aba 2: Seleção de Entrada e Saída de Áudio."""
        container = tk.Frame(self.content_area, bg=C_BG2)
        self.panels["io"] = container

        in_devs = [label for _, label in self.io.list_input_devices()]
        out_devs = [label for _, label in self.io.list_output_devices()]

        self.io_widgets = create_audio_io_panel_widgets(
            container,
            self.config,
            input_devices=in_devs,
            output_devices=out_devs,
            on_input_device=self.update_input_device,
            on_output_device=self.update_output_device,
            on_refresh_devices=self.refresh_audio_devices,
            on_sample_rate=self.update_sample_rate,
            on_buffer_size=self.update_buffer_size,
            on_oversampling=self.update_oversampling,
            on_output_gain=self.update_output_gain,
            on_ms_toggle=self.toggle_ms_global,
            on_mute_toggle=self.toggle_mute,
            on_output_gain_press=lambda: self.engine.state_controller.begin_transaction("Ajustar Ganho de Saída"),
            on_output_gain_release=self._commit_slider_transaction,
        )
        # Mantém sys_widgets apontando para os controles de sistema para compatibilidade com testes
        self.sys_widgets = self.io_widgets

    def _build_test_panel(self) -> None:
        """Constrói a Aba 3: Modo de Teste em Tempo Real (MP3 / WAV & Geradores)."""
        container = tk.Frame(self.content_area, bg=C_BG2)
        self.panels["test"] = container

        self.test_widgets = create_test_mode_panel_widgets(
            container,
            self.config,
            on_source_change=self.update_input_source,
            on_load_media=self.load_media_dialog,
            on_generate_demo=self.generate_and_play_demo_track,
            on_play_pause=self.toggle_media_play_pause,
            on_stop=self.stop_media_playback,
            on_loop_toggle=self.toggle_media_loop,
            on_seek=self.seek_media_position,
            on_test_gain=self.update_test_source_gain,
            on_save_preset=self.save_preset_dialog,
            on_load_preset=self.load_preset_dialog,
            preset_names=PresetIO.list_factory_presets(),
            on_factory_preset=self.apply_selected_factory_preset,
            on_export_wav=self.export_processed_wav_dialog,
            on_freeze_toggle=self.toggle_spectrum_freeze,
        )

    def _build_footer(self) -> None:
        """Constrói o rodapé informativo com dicas contextuais dinâmicas de interação no gráfico (D05)."""
        self.footer = tk.Frame(self.master, bg=C_PRETO, highlightthickness=1, highlightbackground=C_VERDE)
        self.footer.grid(row=4, column=0, sticky="ew", padx=12, pady=(0, 8))
        self.footer_label = tk.Label(
            self.footer,
            text="",
            font=F_SPACE,
            bg=C_PRETO,
            fg=C_TEXTO2,
        )
        self.footer_label.pack(side="left", padx=10, pady=3)
        self._set_footer_hint("idle")

    def _set_footer_hint(self, mode: str = "idle", band_idx: Optional[int] = None) -> None:
        """Atualiza `self.footer_label` dinamicamente conforme o estado do mouse sobre o gráfico (D05)."""
        if getattr(self, "footer_label", None) is None:
            return
        if mode == "dragging" and band_idx is not None:
            text = (
                f"🎚 ARRASTANDO BANDA {int(band_idx) + 1:02d}: "
                "Mova para ajustar Frequência/Ganho | Solte o mouse para registrar no Undo/Redo"
            )
            fg = C_AMARELO
        elif mode == "hover" and band_idx is not None:
            text = (
                f"💡 NÓ BANDA {int(band_idx) + 1:02d}: "
                "Arrastar = Freq/Ganho | Duplo Clique = Resetar 0 dB | Roda do Mouse = Fator Q | "
                "Alt+Clique = Solo | Ctrl+Clique = Bypass | Botão Direito = Excluir"
            )
            fg = C_TEXTO
        else:
            text = (
                "💡 DICAS: Duplo Clique Vazio (ou Shift+Clique) = Criar Banda | "
                "Passe o mouse sobre um ponto (1–24) para ver os comandos de edição | Ctrl+Z/Y = Undo/Redo"
            )
            fg = C_TEXTO2
        try:
            self.footer_label.config(text=text, fg=fg)
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    #  Criação, Edição e Exclusão de Bandas via Gráfico de Espectro (PB04 / PB21)
    # ─────────────────────────────────────────────────────────────────────────
    def create_band_on_graph(
        self,
        freq_hz: float,
        gain_db: float = 0.0,
        q: float = 1.2,
        ftype: FilterType = FilterType.PEAKING,
    ) -> Optional[int]:
        """Cria uma nova banda diretamente nas coordenadas `(freq_hz, gain_db)` do gráfico."""
        new_band = self.curve_controller.create_band_at(freq_hz=freq_hz, gain_db=gain_db, q=q, ftype=ftype)
        if new_band is None:
            return None
        self.selected_band_index = new_band.index
        self._sync_controls_from_selected_band()
        return new_band.index

    def delete_band_on_graph(self, index: int) -> bool:
        """Exclui a banda `index` do gráfico de espectro e atualiza controles e curva."""
        ok = self.curve_controller.remove_band_at(index)
        if ok:
            self.selected_band_index = self.curve_controller.selected_index
            self._sync_controls_from_selected_band()
        return ok

    def delete_selected_band(self) -> bool:
        """Exclui a banda atualmente selecionada."""
        if 0 <= self.selected_band_index < len(self.engine.bands):
            return self.delete_band_on_graph(self.selected_band_index)
        return False

    def select_prev_band(self) -> Optional[int]:
        """Seleciona a banda anterior de forma cíclica (F16)."""
        if not self.engine.bands:
            return None
        self.selected_band_index = (self.selected_band_index - 1) % len(self.engine.bands)
        self.curve_controller.selected_index = self.selected_band_index
        self._sync_controls_from_selected_band()
        return self.selected_band_index

    def select_next_band(self) -> Optional[int]:
        """Seleciona a próxima banda de forma cíclica (F16)."""
        if not self.engine.bands:
            return None
        self.selected_band_index = (self.selected_band_index + 1) % len(self.engine.bands)
        self.curve_controller.selected_index = self.selected_band_index
        self._sync_controls_from_selected_band()
        return self.selected_band_index

    def flatten_all_bands(self) -> None:
        """Reseta todas as bandas para 0.0 dB e desativa Solo/Bypass com suporte a Undo (F07)."""
        self._transact("Resetar Ganhos (Flat)", self.engine.flatten_bands)
        self._sync_nodes()
        self._sync_controls_from_selected_band()

    def toggle_global_bypass(self) -> bool:
        """Alterna o Bypass Global do equalizador registrando em Undo/Redo (F08)."""
        def _do():
            self.config.bypass = not self.config.bypass
            return self.config.bypass

        state = self._transact("Bypass Global", _do)
        if hasattr(self, "global_bypass_btn") and hasattr(self.global_bypass_btn, "set_active"):
            self.global_bypass_btn.set_active(self.config.bypass)
        self._sync_controls_from_selected_band()
        return state

    @staticmethod
    def _extract_modifiers(event) -> Tuple[bool, bool, bool]:
        """Extrai `(is_shift, is_ctrl, is_alt)` sem falso-positivo de NumLock no Windows (F02)."""
        gui_ev = getattr(event, "guiEvent", None)
        state_flags = int(getattr(gui_ev, "state", 0)) if gui_ev is not None else 0
        key_str = str(getattr(event, "key", "") or "").lower()

        is_shift = ("shift" in key_str) or bool(state_flags & 0x0001)
        is_ctrl = ("control" in key_str) or ("ctrl" in key_str) or bool(state_flags & 0x0004)
        if sys.platform.startswith("win"):
            is_alt = ("alt" in key_str) or bool(state_flags & 0x20000)
        else:
            is_alt = ("alt" in key_str) or bool(state_flags & 0x0008) or bool(state_flags & 0x20000)
        return is_shift, is_ctrl, is_alt

    def _extract_event_coords(self, event) -> Optional[Tuple[float, float]]:
        """Extrai `(freq_hz, gain_db)` aceitando eventos tanto em `self.ax` quanto em `self.ax_fft` (V03)."""
        inaxes = getattr(event, "inaxes", None)
        if inaxes not in (self.ax, getattr(self, "ax_fft", None)):
            return None
        if event.xdata is None or event.ydata is None:
            return None
        freq = float(event.xdata)
        if inaxes == getattr(self, "ax_fft", None) and inaxes != self.ax:
            gain = (float(event.ydata) + 48.0) * 0.5
        else:
            gain = float(event.ydata)
        return freq, gain

    def _update_hover_hud(self, band_idx: Optional[int]) -> None:
        """Atualiza o card flutuante (HUD) de inspeção acústica sobre o nó em foco (V06)."""
        if getattr(self, "hover_hud", None) is None:
            return
        if band_idx is not None and 0 <= band_idx < len(self.engine.bands):
            b = self.engine.bands[band_idx]
            self.hover_hud.xy = (b.freq_hz, b.gain_db)
            self.hover_hud.set_text(
                f"B{band_idx + 1} [{b.ftype.name}]\n{b.freq_hz:.0f} Hz | {b.gain_db:+.1f} dB | Q {b.q:.2f}"
            )
            self.hover_hud.set_visible(True)
        else:
            self.hover_hud.set_visible(False)

    def _on_mpl_mouse_down(self, event) -> None:
        coords = self._extract_event_coords(event)
        if coords is None:
            return
        click_freq, click_gain = coords
        is_shift, is_ctrl, is_alt = self._extract_modifiers(event)

        # Verifica se o clique ocorreu sobre um nó existente
        hit_idx = self.curve_controller.find_closest_node(click_freq, click_gain, max_oct_dist=0.75)

        # 1. Botão Direito (button == 3) sobre um nó -> EXCLUIR BANDA no gráfico!
        if event.button == 3:
            if hit_idx is not None:
                self.delete_band_on_graph(hit_idx)
            return

        # 2. Duplo Clique Esquerdo sobre um nó existente -> RESETAR GANHO PARA 0 dB (F01)
        if getattr(event, "dblclick", False) and event.button == 1 and hit_idx is not None:
            self.selected_band_index = hit_idx
            self.curve_controller.selected_index = hit_idx
            self.engine.state_controller.begin_transaction(f"Resetar Ganho Banda {hit_idx + 1}")
            BandControls.update_band_gain(self.engine, hit_idx, 0.0)
            self.engine.state_controller.commit()
            self._sync_nodes()
            self._sync_controls_from_selected_band()
            return

        # 3. Duplo Clique Esquerdo ou Shift+Clique em área livre -> CRIAR NOVA BANDA no gráfico!
        if (getattr(event, "dblclick", False) or is_shift) and event.button == 1 and hit_idx is None:
            self.create_band_on_graph(click_freq, click_gain)
            return

        # 4. Clique sobre um nó existente -> Selecionar / Editar / Solo / Bypass
        if hit_idx is not None:
            self.selected_band_index = hit_idx
            self.curve_controller.selected_index = hit_idx
            if is_alt:
                self._transact(f"Solo Banda {hit_idx + 1}", self.nodes[hit_idx].toggle_solo)
            elif is_ctrl or event.button == 2:
                self._transact(f"Bypass Banda {hit_idx + 1}", self.nodes[hit_idx].toggle_bypass)
            else:
                self.engine.state_controller.begin_transaction(f"Editar Banda {hit_idx + 1}")
                self.curve_controller.dragging_index = hit_idx
                self.nodes[hit_idx].drag_to_acoustic(click_freq, click_gain)
                self._update_hover_hud(hit_idx)
                self._set_footer_hint("dragging", hit_idx)
            self._sync_controls_from_selected_band()

    def _on_mpl_mouse_drag(self, event) -> None:
        coords = self._extract_event_coords(event)
        if coords is None:
            self._update_hover_hud(None)
            self._set_footer_hint("idle")
            return
        freq, gain = coords
        drag_idx = self.curve_controller.dragging_index
        if drag_idx is not None:
            if 0 <= drag_idx < len(self.nodes):
                self.nodes[drag_idx].drag_to_acoustic(freq, gain)
                self._update_hover_hud(drag_idx)
                self._set_footer_hint("dragging", drag_idx)
                self._sync_controls_from_selected_band()
            return

        near_idx = self.curve_controller.find_closest_node(freq, gain, max_oct_dist=0.45)
        self._update_hover_hud(near_idx)
        if near_idx is not None:
            self._set_footer_hint("hover", near_idx)
        else:
            self._set_footer_hint("idle")

    def _on_mpl_mouse_up(self, event) -> None:
        if self.curve_controller.dragging_index is not None:
            self.engine.state_controller.commit()
            self.curve_controller.dragging_index = None
            self._update_window_title()
        self._set_footer_hint("idle")

    def _on_mpl_scroll(self, event) -> None:
        coords = self._extract_event_coords(event)
        if coords is None:
            return
        freq, gain = coords
        step = 1.0 if getattr(event, "button", "up") == "up" else -1.0
        target_idx = self.selected_band_index
        near_idx = self.curve_controller.find_closest_node(freq, gain, max_oct_dist=0.9)
        if near_idx is not None:
            target_idx = near_idx
            self.selected_band_index = near_idx
        if 0 <= target_idx < len(self.nodes):
            self._transact(
                f"Ajustar Q Banda {target_idx + 1}",
                lambda: self.nodes[target_idx].on_wheel(step),
            )
            self._update_hover_hud(target_idx)
            self._sync_controls_from_selected_band()

    # ─────────────────────────────────────────────────────────────────────────
    #  Renderização Visual (Curva Teórica + Espectro FFT + Nós Numerados)
    # ─────────────────────────────────────────────────────────────────────────
    def setup_visuals(self) -> None:
        self.curve_line = None
        self._curve_fill = None
        self.selected_band_curve_line = None
        self.spectrum_line = None
        self.peak_hold_line = None
        self.nodes_active_line = None
        self.nodes_bypass_line = None
        self.nodes_solo_line = None
        self.selected_node_line = None
        self.legend = None
        self._legend_has_spectrum = False
        self.hover_hud = self.ax.annotate(
            "",
            xy=(1000.0, 0.0),
            xytext=(12, 12),
            textcoords="offset points",
            color=C_TEXTO,
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc=C_PRETO, ec=C_AMARELO, alpha=0.9),
            zorder=10,
            visible=False,
        )

    def draw_log_axes(self) -> None:
        """
        Configura o eixo X logarítmico (`20..20k`) apenas com 'Frequência (Hz)' (D05),
        eixo Y em dB (`-24..+24 dB`), eixo secundário `ax_fft` em dBFS (`-96..0 dBFS`) e
        linhas verticais de década em 100 Hz, 1k e 10k com alto contraste (D06).
        """
        self.ax.set_facecolor(C_PRETO)
        self.ax.set_xscale("log")
        self.ax.set_xlim(20.0, 20000.0)
        self.ax.set_ylim(-24.0, 24.0)

        xticks = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        xlabels = ["20", "50", "100", "200", "500", "1k", "2k", "5k", "10k", "20k"]
        self.ax.xaxis.set_major_locator(FixedLocator(xticks))
        self.ax.xaxis.set_major_formatter(FixedFormatter(xlabels))
        self.ax.set_yticks([-24, -18, -12, -6, 0, 6, 12, 18, 24])

        self.ax.axhline(0.0, color="#444444", lw=0.9, ls="--")
        self.ax.grid(True, which="both", color="#222222", lw=0.6)
        for f_dec in (100.0, 1000.0, 10000.0):
            self.ax.axvline(f_dec, color="#2E2E2E", lw=0.95, ls="-", zorder=1)

        self.ax.set_xlabel("Frequência (Hz)", color=C_EIXO, fontsize=9.5)
        self.ax.set_ylabel("Ganho (dB)", color=C_EIXO, fontsize=9.5)
        self.ax.tick_params(colors=C_EIXO, labelsize=8.5)

        if self.ax_fft is None:
            self.ax_fft = self.ax.twinx()
        self.ax_fft.set_ylim(-96.0, 0.0)
        self.ax_fft.set_yticks([-96, -72, -48, -24, 0])
        self.ax_fft.set_ylabel("FFT (dBFS)", color=C_VERDE_CLARO, fontsize=9.5)
        self.ax_fft.tick_params(colors=C_VERDE_CLARO, labelsize=8.5)

    def _ensure_legend(self) -> None:
        """Cria ou atualiza a legenda compacta do gráfico (D06)."""
        self.legend = self.ax.legend(
            loc="upper right",
            facecolor=C_BG2,
            edgecolor=C_MARROM,
            labelcolor=C_TEXTO,
            fontsize=7.5,
            framealpha=0.85,
        )

    def draw_theoretical_curve(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula e plota a curva teórica global, preenchimento translúcido suave (`alpha=0.06` - D06),
        curva individual da banda selecionada (V05), legenda (D06) e nós numerados (PB09 / V04).
        """
        freqs, mag_db = self.engine.global_response(n=512)
        if self.curve_line is None:
            (self.curve_line,) = self.ax.plot(
                freqs, mag_db, color=C_AMARELO, lw=2.2, zorder=4, label="Curva EQ (Ativa)"
            )
        else:
            self.curve_line.set_xdata(freqs)
            self.curve_line.set_ydata(mag_db)

        if self._curve_fill is not None:
            try:
                self._curve_fill.remove()
            except Exception:
                pass
        self._curve_fill = self.ax.fill_between(freqs, 0.0, mag_db, color=C_AMARELO, alpha=0.06, zorder=3)

        # Curva individual da banda selecionada (V05 / D06)
        if (
            not self.config.bypass
            and 0 <= self.selected_band_index < len(self.engine.bands)
            and not self.engine.bands[self.selected_band_index].bypass
        ):
            sb = self.engine.bands[self.selected_band_index]
            effective_fs = int(self.config.sample_rate * max(1, self.config.oversampling))
            sb_db = BiquadMath.combined_response([sb], fs=effective_fs, n=len(freqs), freqs=freqs)
            if self.selected_band_curve_line is None:
                (self.selected_band_curve_line,) = self.ax.plot(
                    freqs,
                    sb_db,
                    color="#38BDF8",
                    lw=1.2,
                    ls="--",
                    alpha=0.75,
                    zorder=3.5,
                    label="Banda Selecionada (Ref.)",
                )
            else:
                self.selected_band_curve_line.set_xdata(freqs)
                self.selected_band_curve_line.set_ydata(sb_db)
        else:
            if self.selected_band_curve_line is None:
                (self.selected_band_curve_line,) = self.ax.plot(
                    [],
                    [],
                    color="#38BDF8",
                    lw=1.2,
                    ls="--",
                    alpha=0.75,
                    zorder=3.5,
                    label="Banda Selecionada (Ref.)",
                )
            else:
                self.selected_band_curve_line.set_xdata([])
                self.selected_band_curve_line.set_ydata([])

        if self.legend is None:
            self._ensure_legend()

        act_f = [b.freq_hz for b in self.engine.bands if not b.bypass]
        act_g = [b.gain_db for b in self.engine.bands if not b.bypass]
        byp_f = [b.freq_hz for b in self.engine.bands if b.bypass]
        byp_g = [b.gain_db for b in self.engine.bands if b.bypass]
        solo_f = [b.freq_hz for b in self.engine.bands if b.solo and not b.bypass]
        solo_g = [b.gain_db for b in self.engine.bands if b.solo and not b.bypass]

        if self.nodes_active_line is None:
            (self.nodes_active_line,) = self.ax.plot(
                act_f, act_g, "o", markerfacecolor=C_AMARELO, markeredgecolor=C_VERMELHO, markersize=8, zorder=5
            )
        else:
            self.nodes_active_line.set_xdata(act_f)
            self.nodes_active_line.set_ydata(act_g)

        if self.nodes_bypass_line is None:
            (self.nodes_bypass_line,) = self.ax.plot(
                byp_f, byp_g, "o", markerfacecolor=C_PRETO, markeredgecolor=C_TEXTO2, markersize=7, zorder=5
            )
        else:
            self.nodes_bypass_line.set_xdata(byp_f)
            self.nodes_bypass_line.set_ydata(byp_g)

        if self.nodes_solo_line is None:
            (self.nodes_solo_line,) = self.ax.plot(
                solo_f,
                solo_g,
                "o",
                markerfacecolor="none",
                markeredgecolor=C_VERMELHO,
                markersize=11,
                markeredgewidth=2.2,
                zorder=6,
            )
        else:
            self.nodes_solo_line.set_xdata(solo_f)
            self.nodes_solo_line.set_ydata(solo_g)

        if 0 <= self.selected_band_index < len(self.engine.bands):
            sb = self.engine.bands[self.selected_band_index]
            sel_f, sel_g = [sb.freq_hz], [sb.gain_db]
        else:
            sel_f, sel_g = [], []

        if self.selected_node_line is None:
            (self.selected_node_line,) = self.ax.plot(
                sel_f, sel_g, "o", markerfacecolor="none", markeredgecolor="#38BDF8", markersize=13, markeredgewidth=2.0, zorder=6
            )
        else:
            self.selected_node_line.set_xdata(sel_f)
            self.selected_node_line.set_ydata(sel_g)

        # Numeração dos nós de 1 a 24 (V04)
        for ann in self._node_annotations:
            try:
                ann.remove()
            except Exception:
                pass
        self._node_annotations.clear()

        for b in self.engine.bands:
            ann = self.ax.annotate(
                str(b.index + 1),
                xy=(b.freq_hz, b.gain_db),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                va="bottom",
                color=C_TEXTO if not b.bypass else C_TEXTO2,
                fontsize=7.5,
                fontweight="bold",
                zorder=7,
            )
            self._node_annotations.append(ann)

        return freqs, mag_db

    def draw_spectrum(self) -> Tuple[np.ndarray, np.ndarray]:
        """Obtém o espectro FFT e o envelope Peak Hold em dBFS do `SpectrumFFT` e plota no gráfico (PB07/PB08/F14/D06)."""
        freqs, spec_db = self.analyzer.spectrum_curve(min_db=-96.0)
        _, peak_db = self.analyzer.peak_hold_curve(min_db=-96.0)
        mapped_db = (spec_db + 48.0) * 0.5
        mapped_peak = (peak_db + 48.0) * 0.5
        valid = freqs >= 20.0
        if self.spectrum_line is None:
            (self.spectrum_line,) = self.ax.plot(
                freqs[valid], mapped_db[valid], color=C_VERDE_CLARO, lw=1.1, alpha=0.80, zorder=2, label="Espectro FFT"
            )
            self._ensure_legend()
            self._legend_has_spectrum = True
        else:
            self.spectrum_line.set_xdata(freqs[valid])
            self.spectrum_line.set_ydata(mapped_db[valid])
            if not getattr(self, "_legend_has_spectrum", False):
                self._ensure_legend()
                self._legend_has_spectrum = True

        if self.peak_hold_line is None:
            (self.peak_hold_line,) = self.ax.plot(
                freqs[valid], mapped_peak[valid], color=C_VERDE_CLARO, lw=0.7, ls=":", alpha=0.45, zorder=2
            )
        else:
            self.peak_hold_line.set_xdata(freqs[valid])
            self.peak_hold_line.set_ydata(mapped_peak[valid])
        return freqs, spec_db

    @staticmethod
    def _format_time_mmss(seconds: float) -> str:
        """Formata segundos em MM:SS.d para exibição no transporte da Aba 3 (F09)."""
        sec = max(0.0, float(seconds))
        mins = int(sec) // 60
        rem = sec - (mins * 60)
        return f"{mins:02d}:{rem:04.1f}"

    def _sync_active_tab_status(self) -> None:
        if self.current_tab == "io":
            self._sync_io_status()
        elif self.current_tab == "test":
            self._sync_test_media_status()

    def _sync_io_status(self) -> None:
        if not hasattr(self, "io_widgets") or "io_status_label" not in self.io_widgets:
            return
        if getattr(self.io, "last_stream_error", ""):
            text, fg = f"✖ Stream Erro: {self.io.stream_mode}", C_VERMELHO
        elif "Simulado" in str(self.io.stream_mode):
            text, fg = f"● Stream: {self.io.stream_mode}", C_AMARELO
        else:
            text, fg = f"● Stream Ativo: {self.io.stream_mode}", C_VERDE_CLARO
        label = self.io_widgets["io_status_label"]
        if label.cget("text") != text or label.cget("fg") != fg:
            label.config(text=text, fg=fg)

    def _sync_test_media_status(self) -> None:
        if not hasattr(self, "test_widgets") or "media_info_label" not in self.test_widgets:
            return

        ms = self.io.media_streamer
        if "play_btn" in self.test_widgets and hasattr(self.test_widgets["play_btn"], "set_active"):
            self.test_widgets["play_btn"].set_active(ms.playing)
        if "loop_btn" in self.test_widgets and hasattr(self.test_widgets["loop_btn"], "set_active"):
            self.test_widgets["loop_btn"].set_active(ms.loop)
            self.test_widgets["loop_btn"].set_text("🔁 Loop: ON" if ms.loop else "🔁 Loop: OFF")

        if ms.has_file() and ms.filepath:
            pos_s = ms.get_position_seconds()
            dur_s = ms.get_duration_seconds()
            fname = os.path.basename(ms.filepath)
            state_icon = "▶" if ms.playing else "⏸"
            pos_fmt = self._format_time_mmss(pos_s)
            dur_fmt = self._format_time_mmss(dur_s)
            media_text = f"{state_icon} {fname} | {pos_fmt} / {dur_fmt} (SR: {self.config.sample_rate}Hz)"
            media_label = self.test_widgets["media_info_label"]
            if media_label.cget("text") != media_text or media_label.cget("fg") != C_AMARELO:
                media_label.config(text=media_text, fg=C_AMARELO)
            seek_sl = self.test_widgets.get("seek_slider")
            if seek_sl is not None and not getattr(seek_sl, "is_dragging", False):
                new_to = max(1.0, dur_s)
                old_value = seek_sl.value
                range_changed = seek_sl.to_ != new_to
                seek_sl.to_ = new_to
                seek_sl.set(pos_s)
                if range_changed and seek_sl.value == old_value:
                    seek_sl.draw()
            if "seek_label" in self.test_widgets:
                seek_label = self.test_widgets["seek_label"]
                seek_text = f"{pos_s:.1f}s / {dur_s:.1f}s"
                if seek_label.cget("text") != seek_text:
                    seek_label.config(text=seek_text)
        else:
            media_text = "Arquivo: Nenhum MP3/WAV carregado (Usando Gerador em Tempo Real)"
            media_label = self.test_widgets["media_info_label"]
            if media_label.cget("text") != media_text or media_label.cget("fg") != C_TEXTO2:
                media_label.config(text=media_text, fg=C_TEXTO2)
            if "seek_label" in self.test_widgets:
                seek_label = self.test_widgets["seek_label"]
                if seek_label.cget("text") != "0.0s / 0.0s":
                    seek_label.config(text="0.0s / 0.0s")

    def show_clip_indicator(self, state: bool) -> None:
        """
        Atualiza o indicador visual de Clipping (> 0 dBFS), barra de nível Pico/RMS (V07),
        e sincroniza os estados de I/O/player somente quando a aba correspondente está ativa.
        """
        self.clip_indicator_state = bool(state)
        peak_db = float(self.io.last_peak_dbfs)
        rms_db = float(getattr(self.io, "last_rms_dbfs", -120.0))
        peak_str = f"P:{peak_db:+.1f} / R:{rms_db:+.1f} dBFS"
        src_str = "MUTE" if self.io.muted else self.config.input_source.name
        if self.clip_indicator_state:
            self.clip_label.config(text=f"⚠ CLIP! ({peak_str}) | {src_str}", fg=C_VERMELHO)
        else:
            self.clip_label.config(text=f"● {src_str} ({peak_str})", fg=C_VERDE_CLARO)

        if hasattr(self, "meter_canvas") and self.meter_canvas is not None:
            try:
                self.meter_canvas.delete("all")
                ratio = max(0.0, min(1.0, (peak_db + 60.0) / 60.0))
                bar_col = (
                    C_VERMELHO
                    if (self.clip_indicator_state or peak_db > 0.0)
                    else (C_AMARELO if peak_db >= -6.0 else C_VERDE_CLARO)
                )
                self.meter_canvas.create_rectangle(0, 0, int(110 * ratio), 14, fill=bar_col, outline="")
            except Exception:
                pass

        self._sync_active_tab_status()

    def update_visuals(self) -> None:
        """Consome todos os blocos pendentes da `visual_queue` em ordem FIFO (T10), alimenta o `SpectrumFFT` e redesenha a UI."""
        if not self.visual_paused:
            try:
                drained = 0
                while not self.io.visual_queue.empty() and drained < 16:
                    try:
                        blk = self.io.visual_queue.get_nowait()
                        if blk is not None:
                            self.analyzer.push_block(blk)
                        drained += 1
                    except Empty:
                        break

                self._update_window_title()
                self.show_clip_indicator(self.io.clip_detected)
                self.draw_theoretical_curve()
                self.draw_spectrum()
                if self.canvas_mpl is not None:
                    self.canvas_mpl.draw_idle()
            except Exception:
                pass

        if hasattr(self.master, "after"):
            self.visual_update_id = self.master.after(33, self.update_visuals)

    def start_visual_updates(self) -> None:
        if self.visual_update_id and hasattr(self.master, "after_cancel"):
            self.master.after_cancel(self.visual_update_id)
        if hasattr(self.master, "after"):
            self.visual_update_id = self.master.after(33, self.update_visuals)

    # ─────────────────────────────────────────────────────────────────────────
    #  Sincronização e Callbacks das 3 Abas
    # ─────────────────────────────────────────────────────────────────────────
    def _sync_controls_from_selected_band(self) -> None:
        """
        Atualiza os controles da Aba 1 para refletir a banda selecionada no gráfico,
        incluindo habilitação contextual de Ganho/Slope (V09), rótulo curto A/B (D03)
        e botão Mid/Side: ON/OFF (D09).
        """
        self._update_window_title()
        try:
            values = [f"Banda {i+1:02d} ({b.freq_hz:.0f}Hz)" for i, b in enumerate(self.engine.bands)]
            if hasattr(self.band_widgets["band_selector"], "config"):
                self.band_widgets["band_selector"].config(values=values or ["Sem Bandas"])
        except Exception:
            pass

        if hasattr(self, "global_bypass_btn") and hasattr(self.global_bypass_btn, "set_active"):
            self.global_bypass_btn.set_active(self.config.bypass)
        if hasattr(self, "io_widgets"):
            if "ms_btn" in self.io_widgets and hasattr(self.io_widgets["ms_btn"], "set_active"):
                self.io_widgets["ms_btn"].set_active(self.config.ms_enabled)
                self.io_widgets["ms_btn"].set_text(
                    "🔀 Mid/Side: ON" if self.config.ms_enabled else "🔀 Mid/Side: OFF"
                )
            if "mute_btn" in self.io_widgets and hasattr(self.io_widgets["mute_btn"], "set_active"):
                self.io_widgets["mute_btn"].set_active(self.io.muted)
        if self.band_widgets.get("ab_btn") is not None:
            slot = self.engine.state_controller.active_slot
            self.band_widgets["ab_btn"].set_text(f"⇄ A/B: Slot {slot}")
            self.band_widgets["ab_btn"].set_active(slot == "B")

        if not (0 <= self.selected_band_index < len(self.engine.bands)):
            return
        band = self.engine.bands[self.selected_band_index]
        try:
            self.band_widgets["band_selector"].set(f"Banda {self.selected_band_index + 1:02d}")
            self.band_widgets["ftype_combo"].set(band.ftype.name)
            self.band_widgets["freq_slider"].set(band.freq_hz)
            self.band_widgets["freq_label"].config(text=f"{band.freq_hz:.0f} Hz")
            self.band_widgets["gain_slider"].set(band.gain_db)
            self.band_widgets["gain_label"].config(text=f"{band.gain_db:+.1f} dB")
            self.band_widgets["q_slider"].set(band.q)
            self.band_widgets["q_label"].config(text=f"{band.q:.2f}")
            self.band_widgets["slope_combo"].set(band.slope.name)
            self.band_widgets["mode_combo"].set(band.mode.name)

            # Habilitação contextual de Ganho e Slope conforme FilterType (V09)
            uses_gain = band.ftype in (FilterType.PEAKING, FilterType.LOW_SHELF, FilterType.HIGH_SHELF)
            uses_slope = band.ftype in (FilterType.LOW_CUT, FilterType.HIGH_CUT)
            self.band_widgets["gain_slider"].set_enabled(uses_gain)
            self.band_widgets["slope_combo"].config(state="readonly" if uses_slope else "disabled")

            # Estados visuais dos botões Bypass e Solo da banda (V08)
            if hasattr(self.band_widgets["bypass_btn"], "set_active"):
                self.band_widgets["bypass_btn"].set_active(band.bypass)
            if hasattr(self.band_widgets["solo_btn"], "set_active"):
                self.band_widgets["solo_btn"].set_active(band.solo)
        except Exception:
            pass

    def _on_slider_freq(self, val: float) -> None:
        if 0 <= self.selected_band_index < len(self.engine.bands):
            idx = self.selected_band_index
            lbl = self.band_widgets["freq_label"]
            sl = self.band_widgets["freq_slider"]
            if getattr(sl, "is_dragging", False) or self.engine.state_controller._pending_snapshot is not None:
                BandControls.update_band_freq(self.engine, idx, val, lbl)
            else:
                self._transact(
                    f"Frequência Banda {idx + 1}",
                    lambda: BandControls.update_band_freq(self.engine, idx, val, lbl),
                )
            self._sync_nodes()

    def _on_slider_gain(self, val: float) -> None:
        if 0 <= self.selected_band_index < len(self.engine.bands):
            idx = self.selected_band_index
            lbl = self.band_widgets["gain_label"]
            sl = self.band_widgets["gain_slider"]
            if getattr(sl, "is_dragging", False) or self.engine.state_controller._pending_snapshot is not None:
                BandControls.update_band_gain(self.engine, idx, val, lbl)
            else:
                self._transact(
                    f"Ganho Banda {idx + 1}",
                    lambda: BandControls.update_band_gain(self.engine, idx, val, lbl),
                )
            self._sync_nodes()

    def _on_slider_q(self, val: float) -> None:
        if 0 <= self.selected_band_index < len(self.engine.bands):
            idx = self.selected_band_index
            lbl = self.band_widgets["q_label"]
            sl = self.band_widgets["q_slider"]
            if getattr(sl, "is_dragging", False) or self.engine.state_controller._pending_snapshot is not None:
                BandControls.update_band_q(self.engine, idx, val, lbl)
            else:
                self._transact(
                    f"Fator Q Banda {idx + 1}",
                    lambda: BandControls.update_band_q(self.engine, idx, val, lbl),
                )
            self._sync_nodes()

    def _on_band_selected(self, event=None) -> None:
        raw = str(self.band_widgets["band_selector"].get())
        try:
            tokens = raw.replace("Banda", "").strip().split()
            idx = int(tokens[0]) - 1
            if 0 <= idx < len(self.engine.bands):
                self.selected_band_index = idx
                self.curve_controller.selected_index = idx
                self._sync_controls_from_selected_band()
        except Exception:
            pass

    def _on_type_changed(self, event=None) -> None:
        if 0 <= self.selected_band_index < len(self.engine.bands):
            idx = self.selected_band_index
            val = self.band_widgets["ftype_combo"].get()
            self._transact(
                f"Tipo Filtro Banda {idx + 1}",
                lambda: BandControls.update_band_type(self.engine, idx, val),
            )
            self._sync_controls_from_selected_band()

    def _on_slope_changed(self, event=None) -> None:
        if 0 <= self.selected_band_index < len(self.engine.bands):
            idx = self.selected_band_index
            val = self.band_widgets["slope_combo"].get()
            self._transact(
                f"Slope Banda {idx + 1}",
                lambda: BandControls.update_band_slope(self.engine, idx, val),
            )
            self._sync_controls_from_selected_band()

    def _on_mode_changed(self, event=None) -> None:
        if 0 <= self.selected_band_index < len(self.engine.bands):
            idx = self.selected_band_index
            val = self.band_widgets["mode_combo"].get()
            self._transact(
                f"Canal Banda {idx + 1}",
                lambda: BandControls.update_band_mode(self.engine, idx, val),
            )
            self._sync_controls_from_selected_band()

    def _on_bypass_toggled(self) -> None:
        if 0 <= self.selected_band_index < len(self.engine.bands):
            idx = self.selected_band_index
            b = self.engine.bands[idx]
            self._transact(
                f"Bypass Banda {idx + 1}",
                lambda: BandControls.update_band_bypass(self.engine, idx, not b.bypass),
            )
            self._sync_controls_from_selected_band()

    def _on_solo_toggled(self) -> None:
        if 0 <= self.selected_band_index < len(self.engine.bands):
            idx = self.selected_band_index
            b = self.engine.bands[idx]
            self._transact(
                f"Solo Banda {idx + 1}",
                lambda: BandControls.update_band_solo(self.engine, idx, not b.solo),
            )
            self._sync_controls_from_selected_band()

    def undo_action(self) -> bool:
        ok = self.engine.state_controller.undo()
        if ok:
            self._sync_nodes()
            self._sync_controls_from_selected_band()
        return ok

    def redo_action(self) -> bool:
        ok = self.engine.state_controller.redo()
        if ok:
            self._sync_nodes()
            self._sync_controls_from_selected_band()
        return ok

    def ab_compare_action(self) -> str:
        slot = self.engine.state_controller.compare_ab()
        self._sync_nodes()
        self._sync_controls_from_selected_band()
        return slot

    # ─────────────────────────────────────────────────────────────────────────
    #  Callbacks da Aba 2: Seleção de Entrada e Saída de Áudio
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _parse_device_id(raw_label: str) -> Optional[int]:
        raw = str(raw_label).strip()
        if raw.startswith("[") and "]" in raw:
            try:
                return int(raw.split("]")[0][1:])
            except Exception:
                return None
        return None

    def update_input_device(self, event=None) -> None:
        """Seleciona o dispositivo de entrada de áudio escolhido na Aba 2."""
        dev_id = self._parse_device_id(self.io_widgets["input_dev_combo"].get())
        self.io.set_input_device(dev_id)

    def update_output_device(self, event=None) -> None:
        """Seleciona o dispositivo de saída de áudio escolhido na Aba 2."""
        dev_id = self._parse_device_id(self.io_widgets["output_dev_combo"].get())
        self.io.set_output_device(dev_id)

    def refresh_audio_devices(self) -> None:
        """Reconsulta os dispositivos de entrada e saída conectados ao computador."""
        in_devs = [label for _, label in self.io.list_input_devices()]
        out_devs = [label for _, label in self.io.list_output_devices()]
        try:
            self.io_widgets["input_dev_combo"].config(values=in_devs)
            self.io_widgets["output_dev_combo"].config(values=out_devs)
        except Exception:
            pass

    def update_output_gain(self, val: float) -> None:
        sl = self.io_widgets.get("out_gain_slider")
        if (sl is not None and getattr(sl, "is_dragging", False)) or self.engine.state_controller._pending_snapshot is not None:
            self.config.output_gain_db = float(val)
        else:
            def _do():
                self.config.output_gain_db = float(val)
            self._transact("Ganho de Saída", _do)

    def toggle_ms_global(self) -> bool:
        def _do():
            self.config.ms_enabled = not self.config.ms_enabled
            return self.config.ms_enabled

        res = self._transact("Matriz Mid/Side Global", _do)
        if "ms_btn" in self.io_widgets and hasattr(self.io_widgets["ms_btn"], "set_active"):
            self.io_widgets["ms_btn"].set_active(self.config.ms_enabled)
            self.io_widgets["ms_btn"].set_text(
                "🔀 Mid/Side: ON" if self.config.ms_enabled else "🔀 Mid/Side: OFF"
            )
        return res

    def toggle_mute(self) -> bool:
        self.io.muted = not self.io.muted
        if "mute_btn" in self.io_widgets and hasattr(self.io_widgets["mute_btn"], "set_active"):
            self.io_widgets["mute_btn"].set_active(self.io.muted)
            self.io_widgets["mute_btn"].set_text(
                "🔇 MUTE ATIVO" if self.io.muted else "🔊 Mute / Ativar Som"
            )
        return self.io.muted

    def update_sample_rate(self, event=None) -> None:
        new_sr = int(self.io_widgets["sample_rate"].get())
        self.config.sample_rate = new_sr
        self.analyzer.set_sample_rate(new_sr)
        self.io.restart_stream()

    def update_buffer_size(self, event=None) -> None:
        new_bs = int(self.io_widgets["buffer_size"].get())
        self.config.buffer_size = new_bs
        self.io.restart_stream()

    def update_oversampling(self, event=None) -> None:
        new_os = int(self.io_widgets["oversampling"].get())
        self.config.oversampling = new_os
        self.engine.prepare(self.config.sample_rate, self.config.buffer_size)

    # ─────────────────────────────────────────────────────────────────────────
    #  Callbacks da Aba 3: Modo de Teste (Injeção MP3 / WAV em Tempo Real)
    # ─────────────────────────────────────────────────────────────────────────
    def update_input_source(self, event=None) -> None:
        val = str(self.test_widgets["source_combo"].get())
        if val in (InputSource.MEDIA_FILE.name, "WAV_FILE") and not self.io.media_streamer.has_file():
            self.load_media_dialog()
            return
        self.io.set_input_source(val)

    def apply_selected_factory_preset(self, event=None) -> None:
        """Aplica o preset de fábrica selecionado no `preset_combo` da Aba 3 (F10)."""
        if "preset_combo" not in self.test_widgets:
            return
        preset_name = str(self.test_widgets["preset_combo"].get())
        if preset_name in PresetIO.FACTORY_PRESETS:
            PresetIO.apply_factory_preset(self.engine, preset_name)
            self._sync_nodes()
            self._sync_controls_from_selected_band()

    def toggle_spectrum_freeze(self) -> bool:
        """Congela ou descongela o analisador de espectro FFT na tela (F14)."""
        self.analyzer.set_freeze(not self.analyzer.frozen)
        if "freeze_btn" in self.test_widgets and hasattr(self.test_widgets["freeze_btn"], "set_active"):
            self.test_widgets["freeze_btn"].set_active(self.analyzer.frozen)
        return self.analyzer.frozen

    def export_processed_wav_dialog(self, output_path: Optional[str] = None) -> Optional[str]:
        """Exporta o áudio atual (ou faixa demo) processado pelo `EqEngine` para um arquivo `.wav` (F15)."""
        path = output_path
        if path is None and filedialog is not None:
            path = filedialog.asksaveasfilename(
                defaultextension=".wav",
                filetypes=[("WAV Audio", "*.wav")],
                title="Exportar Áudio Processado (.wav)",
            )
        if not path:
            return None

        temp_in: Optional[str] = None
        try:
            ms = self.io.media_streamer
            if ms.has_file() and ms.filepath and os.path.isfile(ms.filepath):
                src_path = ms.filepath
            else:
                fd, temp_in = tempfile.mkstemp(suffix=".wav", prefix="selvaeq_src_")
                os.close(fd)
                RealtimeMediaStreamer.generate_demo_track_file(
                    temp_in, duration_sec=2.0, sample_rate=self.config.sample_rate
                )
                src_path = temp_in

            self.io.process_wav_file(src_path, path)
            return path
        finally:
            if temp_in and os.path.exists(temp_in):
                try:
                    os.remove(temp_in)
                except Exception:
                    pass

    def load_media_dialog(self, filepath: Optional[str] = None) -> Optional[str]:
        """Abre diálogo para escolher um arquivo `.mp3`, `.wav`, `.flac` ou `.ogg` e injetá-lo em tempo real."""
        path = filepath
        if path is None and filedialog is not None:
            path = filedialog.askopenfilename(
                defaultextension=".mp3",
                filetypes=[
                    ("Arquivos de Áudio (MP3, WAV, FLAC, OGG)", "*.mp3 *.wav *.flac *.ogg"),
                    ("MP3 Audio", "*.mp3"),
                    ("WAV Audio", "*.wav"),
                    ("Todos os Arquivos", "*.*"),
                ],
                title="Inserir Arquivo MP3 ou WAV no Fluxo de Áudio em Tempo Real",
            )
        if path:
            try:
                self.io.load_media_file(path)
                dur_raw = self.io.media_streamer.get_duration_seconds()
                dur = max(1.0, dur_raw)
                seek_slider = self.test_widgets["seek_slider"]
                old_value = seek_slider.value
                range_changed = seek_slider.to_ != dur
                seek_slider.to_ = dur
                seek_slider.set(0.0)
                if range_changed and seek_slider.value == old_value:
                    seek_slider.draw()
                if "seek_label" in self.test_widgets:
                    self.test_widgets["seek_label"].config(text=f"0.0s / {dur_raw:.1f}s")
                self.test_widgets["source_combo"].set(InputSource.MEDIA_FILE.name)
                return path
            except Exception as e:
                if messagebox is not None:
                    messagebox.showerror("SelvaEQ", f"Erro ao carregar arquivo de áudio:\n{e}")
        return None

    def load_wav_dialog(self, filepath: Optional[str] = None) -> Optional[str]:
        """Alias mantido para compatibilidade retroativa."""
        return self.load_media_dialog(filepath)

    def generate_and_play_demo_track(self, filepath: Optional[str] = None) -> str:
        """Gera instantaneamente uma faixa demo musical e a coloca em reprodução no modo de teste."""
        target_path = filepath or os.path.join(tempfile.gettempdir(), "selvaeq_demo_track.wav")
        RealtimeMediaStreamer.generate_demo_track_file(
            target_path, duration_sec=4.0, sample_rate=self.config.sample_rate
        )
        self.load_media_dialog(target_path)
        return target_path

    def toggle_media_play_pause(self) -> bool:
        playing = self.io.toggle_play_pause_media()
        if "play_btn" in self.test_widgets and hasattr(self.test_widgets["play_btn"], "set_active"):
            self.test_widgets["play_btn"].set_active(playing)
        return playing

    def stop_media_playback(self) -> None:
        self.io.stop_media()
        self.test_widgets["seek_slider"].set(0.0)
        if "seek_label" in self.test_widgets:
            ms = self.io.media_streamer
            if ms.has_file():
                self.test_widgets["seek_label"].config(text=f"0.0s / {ms.get_duration_seconds():.1f}s")
            else:
                self.test_widgets["seek_label"].config(text="0.0s / 0.0s")
        if "play_btn" in self.test_widgets and hasattr(self.test_widgets["play_btn"], "set_active"):
            self.test_widgets["play_btn"].set_active(False)

    def toggle_media_loop(self) -> bool:
        new_loop = not self.io.media_streamer.loop
        self.io.set_media_loop(new_loop)
        if "loop_btn" in self.test_widgets and hasattr(self.test_widgets["loop_btn"], "set_active"):
            self.test_widgets["loop_btn"].set_active(new_loop)
            self.test_widgets["loop_btn"].set_text("🔁 Loop: ON" if new_loop else "🔁 Loop: OFF")
        return new_loop

    def seek_media_position(self, seconds: float) -> None:
        self.io.seek_media(float(seconds))
        if "seek_label" in self.test_widgets:
            ms = self.io.media_streamer
            dur_s = ms.get_duration_seconds() if ms.has_file() else 0.0
            self.test_widgets["seek_label"].config(text=f"{float(seconds):.1f}s / {dur_s:.1f}s")

    def update_test_source_gain(self, val: float) -> None:
        self.io.media_streamer.set_gain(float(val))

    def save_preset_dialog(self, filepath: Optional[str] = None) -> Optional[str]:
        path = filepath
        if path is None and filedialog is not None:
            path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("Preset SelvaEQ JSON", "*.json"), ("Preset SelvaEQ XML", "*.xml")],
                title="Salvar Preset SelvaEQ",
            )
        if path:
            PresetIO.save_preset(self.engine, path)
            self._update_window_title()
            return path
        return None

    def load_preset_dialog(self, filepath: Optional[str] = None) -> Optional[str]:
        path = filepath
        if path is None and filedialog is not None:
            path = filedialog.askopenfilename(
                defaultextension=".json",
                filetypes=[("Preset SelvaEQ", "*.json *.xml")],
                title="Carregar Preset SelvaEQ",
            )
        if path:
            old_sr = self.config.sample_rate
            PresetIO.load_preset(self.engine, path)
            if self.config.sample_rate != old_sr:
                self.analyzer.set_sample_rate(self.config.sample_rate)
                self.io.restart_stream()
            self._sync_nodes()
            self._sync_controls_from_selected_band()
            return path
        return None

    def on_close(self) -> None:
        """Encerra o stream, fecha a janela e executa a limpeza de cache (`cleancache.py`)."""
        if self._resize_job is not None and hasattr(self.master, "after_cancel"):
            try:
                self.master.after_cancel(self._resize_job)
            except Exception:
                pass
            self._resize_job = None
        if self.visual_update_id and hasattr(self.master, "after_cancel"):
            self.master.after_cancel(self.visual_update_id)
            self.visual_update_id = None
        if self.io:
            self.io.stop()
        plt.close(self.fig)
        if hasattr(self.master, "destroy"):
            self.master.destroy()
        try:
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if root_dir not in sys.path:
                sys.path.append(root_dir)
            from cleancache import remove_pycache_and_pyc
            remove_pycache_and_pyc(root_dir)
        except Exception:
            pass


# Alias explícito conforme seção 2.1 do documento de planejamento
PyGraphUI = EqInterface
