"""
Widgets Customizados e Construtores de Abas Responsivas do Equalizador (`eq/ui/widgets.py`).

Reaproveita `SelvaSlider` e os construtores visuais de `ui/widgets.py` do SelvaSonic,
adaptando-os para redimensionamento dinâmico (`<Configure>`) e organizando a interface em 3 abas:
1. Aba "Equalizador & Bandas" (edição paramétrica + criar/excluir/solo/bypass/flat/undo/redo/A-B).
2. Aba "Entrada & Saída de Áudio" (seleção independente de dispositivo de Entrada e de Saída,
   Sample Rate, Buffer Size, Oversampling, Ganho de Saída e Mute).
3. Aba "Modo de Teste (MP3 / WAV)" (injeção de arquivos `.mp3`/`.wav` em tempo real no fluxo
   de áudio, controles Play/Pause/Stop/Loop/Seek, presets rápidos, exportação WAV e gerador demo).

Inclui tema escuro `ttk.Style` (`clam` - V01), resolvedor multiplataforma de fontes (V02 / D11),
estado visual ativo/inativo em botões toggle (`set_active`/`set_text` - V08), habilitação contextual
de sliders (`set_enabled` - V09), trilho de 8 px com barra preenchida unipolar/bipolar (D08),
padronização de contraste de rótulos e botões sem truncamento (D03, D07, D09, D10), entrada numérica
inline por duplo clique (F05), escala logarítmica em sliders de frequência/Q (F06), callbacks
transacionais `on_press`/`on_release` (F03) e stubs headless completos (T15 / R4).
"""

import math
from typing import Callable, Dict, List, Optional
from eq.config import EqConfig, FilterType, ChannelMode, Slope, InputSource, SUPPORTED_SAMPLE_RATES
from eq.dsp.utils import clamp
from eq.io.preset_io import PresetIO


class _DummyWidget:
    """Fallback completo para execução de testes unitários em ambientes sem servidor gráfico/Tkinter (R4 / R5 / T15)."""
    def __init__(self, parent=None, **kwargs):
        self.parent = parent
        self.kwargs = kwargs
        self._children = []
        self._bindings = {}
        self._val = kwargs.get("init", "")
        self._title = "SelvaEQ - 24-Band Parametric Equalizer"
        self._attrs = {"-fullscreen": False}
        if parent is not None and hasattr(parent, "_children"):
            parent._children.append(self)

    def bind(self, seq, func):
        self._bindings[seq] = func

    def bind_all(self, seq, func):
        self._bindings[seq] = func

    def pack(self, **kwargs):
        pass

    def pack_forget(self):
        pass

    def pack_propagate(self, *args):
        pass

    def grid(self, **kwargs):
        pass

    def grid_forget(self):
        pass

    def grid_propagate(self, *args):
        pass

    def place(self, **kwargs):
        pass

    def place_forget(self):
        pass

    def config(self, **kwargs):
        self.kwargs.update(kwargs)

    def configure(self, **kwargs):
        self.kwargs.update(kwargs)

    def cget(self, key):
        return self.kwargs.get(key, "")

    def __getitem__(self, key):
        return self.kwargs.get(key, "")

    def columnconfigure(self, *args, **kwargs):
        pass

    def rowconfigure(self, *args, **kwargs):
        pass

    def protocol(self, *args, **kwargs):
        pass

    def geometry(self, *args):
        pass

    def minsize(self, *args):
        pass

    def title(self, text=None):
        if text is not None:
            self._title = str(text)
        return self._title

    def attributes(self, key, val=None):
        if val is not None:
            self._attrs[key] = val
        return self._attrs.get(key)

    def state(self, *args):
        pass

    def delete(self, *args):
        if args and args[0] == 0:
            self._val = ""

    def create_rectangle(self, *args, **kwargs):
        return 1

    def create_polygon(self, *args, **kwargs):
        return 2

    def create_text(self, *args, **kwargs):
        return 3

    def create_window(self, *args, **kwargs):
        return 4

    def create_line(self, *args, **kwargs):
        return 5

    def coords(self, *args):
        pass

    def itemconfig(self, *args, **kwargs):
        pass

    def bbox(self, *args):
        return (0, 0, 100, 100)

    def find_all(self):
        return [1]

    def winfo_exists(self):
        return 1

    def winfo_width(self):
        return 800

    def winfo_height(self):
        return 600

    def winfo_reqwidth(self):
        return 80

    def winfo_children(self):
        return self._children

    def winfo_toplevel(self):
        return self.parent.winfo_toplevel() if (self.parent and hasattr(self.parent, "winfo_toplevel")) else self

    def winfo_class(self):
        return "Dummy"

    def focus_set(self):
        pass

    def focus_get(self):
        return None

    def after(self, ms, func=None, *args):
        return 1

    def after_cancel(self, *args):
        pass

    def option_add(self, *args):
        pass

    def insert(self, index, text):
        self._val = str(text)

    def select_range(self, *args):
        pass

    def destroy(self):
        pass

    def get_tk_widget(self):
        return self

    def set(self, val):
        self._val = val

    def get(self):
        return self._val

    def yview(self, *args):
        pass


class _TtkStyleStub:
    """Stub headless para `ttk.Style` (V01 / T15)."""
    def __init__(self, master=None):
        self.master = master

    def theme_use(self, *args):
        pass

    def configure(self, *args, **kwargs):
        pass

    def map(self, *args, **kwargs):
        pass


class _TkStub:
    Canvas = _DummyWidget
    Frame = _DummyWidget
    Label = _DummyWidget
    Entry = _DummyWidget
    Scrollbar = _DummyWidget
    Tk = _DummyWidget


class _TtkStub:
    Combobox = _DummyWidget
    Scrollbar = _DummyWidget
    Treeview = _DummyWidget
    Style = _TtkStyleStub


try:
    import tkinter as tk
    from tkinter import ttk
    HAS_TK = True
    _CanvasBase = tk.Canvas
except Exception:
    HAS_TK = False
    tk = _TkStub()
    ttk = _TtkStub()
    _CanvasBase = _DummyWidget


# ─────────────────────────────────────────────────────────────────────────────
#  Paleta SelvaSonic & Extensões de Alto Contraste (D06 / D07)
# ─────────────────────────────────────────────────────────────────────────────
C_AMARELO     = "#F5C800"
C_VERDE       = "#1A7A2E"
C_VERDE_CLARO = "#4ADE80"
C_MARROM      = "#5C3317"
C_AZUL        = "#1B4FD8"
C_VERMELHO    = "#C8281E"
C_PRETO       = "#0A0A0A"
C_ROXO        = "#6B2D8B"
C_BG          = "#111111"
C_BG2         = "#1A1A1A"
C_BG3         = "#222222"
C_TEXTO       = "#E8E0C8"
C_TEXTO2      = "#A09880"
C_EIXO        = "#C8C2B0"

F_BEBAS  = ("Bebas Neue", 14)
F_BARLOW = ("Barlow Condensed", 10, "bold")
F_SPACE  = ("Space Mono", 9)

_AVAILABLE_FONT_FAMILIES: Optional[Dict[str, str]] = None


def resolve_font_family(preferred: List[str], default: str = "Sans") -> str:
    """Resolve a primeira família tipográfica disponível no sistema operacional com cache (V02)."""
    global _AVAILABLE_FONT_FAMILIES
    if HAS_TK:
        try:
            if _AVAILABLE_FONT_FAMILIES is None:
                import tkinter.font as tkfont
                _AVAILABLE_FONT_FAMILIES = {f.lower(): f for f in tkfont.families()}
            for candidate in preferred:
                if candidate.lower() in _AVAILABLE_FONT_FAMILIES:
                    return _AVAILABLE_FONT_FAMILIES[candidate.lower()]
        except Exception:
            pass
    return preferred[0] if preferred else default


def get_ui_fonts(root=None) -> dict:
    """Resolve e atualiza as fontes globais da interface com fallback seguro e nítido por SO (V02 / D11)."""
    global F_BEBAS, F_BARLOW, F_SPACE
    title_fam = resolve_font_family(
        ["Segoe UI Semibold", "DejaVu Sans", "Liberation Sans", "Arial", "Bebas Neue", "Helvetica"],
        default="Sans",
    )
    ui_fam = resolve_font_family(
        ["Barlow Condensed", "Segoe UI", "DejaVu Sans", "Liberation Sans", "Helvetica", "Arial"],
        default="Sans",
    )
    mono_fam = resolve_font_family(
        ["Space Mono", "Consolas", "DejaVu Sans Mono", "Liberation Mono", "Menlo", "Courier New", "Courier"],
        default="Monospace",
    )
    F_BEBAS = (title_fam, 14, "bold") if title_fam != "Bebas Neue" else ("Bebas Neue", 14)
    F_BARLOW = (ui_fam, 10, "bold")
    F_SPACE = (mono_fam, 9)
    return {"F_BEBAS": F_BEBAS, "F_BARLOW": F_BARLOW, "F_SPACE": F_SPACE}


def apply_dark_ttk_style(root=None) -> None:
    """Configura o tema escuro 'clam' para todos os widgets ttk (V01)."""
    try:
        get_ui_fonts(root)
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure(
            "TCombobox",
            fieldbackground=C_BG3,
            background=C_BG2,
            foreground=C_TEXTO,
            arrowcolor=C_AMARELO,
            bordercolor=C_MARROM,
            lightcolor=C_BG3,
            darkcolor=C_PRETO,
            selectbackground=C_BG3,
            selectforeground=C_AMARELO,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", C_BG3), ("disabled", C_PRETO)],
            foreground=[("readonly", C_TEXTO), ("disabled", C_TEXTO2)],
            selectbackground=[("readonly", C_BG3)],
            selectforeground=[("readonly", C_AMARELO)],
            background=[("active", C_MARROM), ("pressed", C_AMARELO)],
        )
        style.configure(
            "Vertical.TScrollbar",
            background=C_BG3,
            troughcolor=C_PRETO,
            arrowcolor=C_AMARELO,
            bordercolor=C_MARROM,
        )
        style.configure(
            "Treeview",
            background=C_BG2,
            fieldbackground=C_BG2,
            foreground=C_TEXTO,
            rowheight=22,
            font=F_SPACE,
        )
        style.configure(
            "Treeview.Heading",
            background=C_BG3,
            foreground=C_AMARELO,
            font=F_BARLOW,
        )
        style.map(
            "Treeview",
            background=[("selected", C_MARROM)],
            foreground=[("selected", C_AMARELO)],
        )
        if root is not None and hasattr(root, "option_add"):
            root.option_add("*TCombobox*Listbox.background", C_BG2)
            root.option_add("*TCombobox*Listbox.foreground", C_TEXTO)
            root.option_add("*TCombobox*Listbox.selectBackground", C_MARROM)
            root.option_add("*TCombobox*Listbox.selectForeground", C_AMARELO)
    except Exception:
        pass


class SelvaSlider(_CanvasBase):
    """
    Slider customizado reaproveitado de `ui/widgets.py` do SelvaSonic, agora com
    adaptação automática de largura (`<Configure>`), escala linear ou logarítmica (F06),
    estado habilitado/desabilitado (V09), flag de arraste (V10), trilho espesso de 8 px com
    barra preenchida unipolar/bipolar (D08) e callbacks de início/fim de transação para Undo/Redo (F03).
    """

    def __init__(
        self,
        parent=None,
        width: int = 150,
        height: int = 22,
        from_: float = 0.0,
        to_: float = 1.0,
        init: float = 0.5,
        command: Optional[Callable] = None,
        accent: str = C_AMARELO,
        scale: str = "linear",
        on_press: Optional[Callable] = None,
        on_release: Optional[Callable] = None,
    ):
        super().__init__(parent, width=width, height=height, bg=C_BG2, highlightthickness=0)
        self.width: int = max(60, int(width))
        self.height: int = int(height)
        self.from_: float = float(from_)
        self.to_: float = float(to_)
        self.value: float = clamp(float(init), min(self.from_, self.to_), max(self.from_, self.to_))
        self.command: Optional[Callable] = command
        self.accent: str = accent
        self.scale: str = str(scale).lower()
        self.enabled: bool = True
        self.is_dragging: bool = False
        self.on_press: Optional[Callable] = on_press
        self.on_release: Optional[Callable] = on_release

        self.bind("<Button-1>", self._on_click)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Configure>", self._on_resize)
        self.draw()

    def set_enabled(self, enabled: bool) -> None:
        """Habilita ou desabilita visualmente e funcionalmente o slider (V09)."""
        self.enabled = bool(enabled)
        self.draw()

    def _on_resize(self, event) -> None:
        new_w = getattr(event, "width", self.width)
        if isinstance(new_w, (int, float)) and new_w > 40 and int(new_w) != self.width:
            self.width = int(new_w)
            self.draw()

    def draw(self) -> None:
        self.delete("all")
        outline_col = C_MARROM if self.enabled else "#333333"
        # D08: Trilho espesso de 8 px (y=7..15)
        self.create_rectangle(4, 7, self.width - 4, 15, fill=C_PRETO, outline=outline_col, width=1)
        if self.scale == "log" and self.from_ > 0.0 and self.to_ > 0.0 and self.to_ != self.from_:
            lo, hi = min(self.from_, self.to_), max(self.from_, self.to_)
            safe_val = clamp(self.value, lo, hi)
            percent = math.log(safe_val / self.from_) / math.log(self.to_ / self.from_)
        else:
            val_range = self.to_ - self.from_
            percent = 0.0 if val_range == 0 else (self.value - self.from_) / val_range
        percent = clamp(percent, 0.0, 1.0)
        x_center = 8 + percent * (self.width - 16)

        # D08: Barra ativa preenchida até o cursor (bipolar a partir de 0.0 dB ou unipolar a partir da esquerda)
        fill_col = self.accent if self.enabled else "#444444"
        if self.scale == "linear" and self.from_ < 0.0 < self.to_:
            x_zero = 8 + ((-self.from_) / (self.to_ - self.from_)) * (self.width - 16)
            x_left, x_right = min(x_zero, x_center), max(x_zero, x_center)
            self.create_rectangle(x_left, 8, x_right, 14, fill=fill_col, outline="")
        else:
            self.create_rectangle(8, 8, max(8, x_center), 14, fill=fill_col, outline="")

        x0, y0, y1 = x_center - 8, 0, self.height
        points = [x0 + 3, y0, x0 + 16, y0, x0 + 13, y1, x0, y1]
        cursor_col = self.accent if self.enabled else C_TEXTO2
        self.create_polygon(points, fill=cursor_col, outline=C_PRETO)

    def _update_val_from_x(self, x: float) -> None:
        if not self.enabled:
            return
        percent = (float(x) - 8.0) / max(1.0, float(self.width - 16))
        percent = clamp(percent, 0.0, 1.0)
        if self.scale == "log" and self.from_ > 0.0 and self.to_ > 0.0:
            self.value = self.from_ * math.pow(self.to_ / self.from_, percent)
        else:
            self.value = self.from_ + percent * (self.to_ - self.from_)
        self.draw()
        if self.command:
            self.command(self.value)

    def _on_click(self, event) -> None:
        if not self.enabled:
            return
        self.is_dragging = True
        if self.on_press:
            self.on_press()
        self._update_val_from_x(event.x)

    def _on_drag(self, event) -> None:
        if not self.enabled:
            return
        self.is_dragging = True
        self._update_val_from_x(event.x)

    def _on_release(self, event=None) -> None:
        was_dragging = self.is_dragging
        self.is_dragging = False
        if was_dragging and self.on_release:
            self.on_release()

    def set(self, val: float) -> None:
        lo, hi = min(self.from_, self.to_), max(self.from_, self.to_)
        self.value = clamp(float(val), lo, hi)
        self.draw()


def _make_ss_label(parent, text: str, accent: str = C_AMARELO):
    """Cria rótulo angulado de alto contraste sem texto roxo escuro (D07)."""
    canvas = tk.Canvas(parent, width=115, height=22, bg=C_BG2, highlightthickness=0)
    pts = [0, 0, 109, 0, 115, 11, 109, 22, 0, 22]
    canvas.create_polygon(pts, fill=C_BG3, outline=C_MARROM)
    bar_col = C_AMARELO if accent == C_ROXO else accent
    canvas.create_rectangle(0, 0, 3, 22, fill=bar_col, outline="")
    txt_col = C_AMARELO if accent == C_AMARELO else C_TEXTO
    canvas.create_text(9, 11, text=text.upper(), fill=txt_col, font=F_BARLOW, anchor="w")
    return canvas


def _styled_button(parent, text: str, command: Optional[Callable], accent: str = C_AMARELO, danger: bool = False):
    """Cria botão estilizado SelvaEQ com suporte a estado ativo persistente (`set_active`/`set_text` - V08)."""
    btn_frame = tk.Frame(parent, bg=C_BG3, highlightthickness=1, highlightbackground=C_MARROM)
    accent_color = C_VERMELHO if danger else accent
    left_border = tk.Frame(btn_frame, width=3, bg=accent_color)
    left_border.pack(side="left", fill="y")
    lbl = tk.Label(btn_frame, text=text.upper(), font=F_BARLOW, bg=C_BG3, fg=C_TEXTO, padx=10, pady=3, cursor="hand2")
    lbl.pack(side="left")

    btn_frame._is_active = False
    btn_frame._label = lbl
    btn_frame._text = str(text).upper()
    btn_frame._accent = accent_color
    btn_frame._danger = danger

    def _apply_visual(hover: bool = False):
        if btn_frame._is_active or hover:
            btn_frame.config(highlightbackground=accent_color)
            lbl.config(bg=accent_color, fg=C_PRETO if not danger else "#FFFFFF")
        else:
            btn_frame.config(highlightbackground=C_MARROM)
            lbl.config(bg=C_BG3, fg=C_TEXTO)

    def set_active(active: bool) -> None:
        btn_frame._is_active = bool(active)
        _apply_visual(hover=False)

    def set_text(new_text: str) -> None:
        btn_frame._text = str(new_text).upper()
        lbl.config(text=btn_frame._text)

    btn_frame.set_active = set_active
    btn_frame.set_text = set_text

    def on_enter(e=None):
        _apply_visual(hover=True)

    def on_leave(e=None):
        _apply_visual(hover=False)

    def on_click(e=None):
        if command:
            command()

    for w in (btn_frame, left_border, lbl):
        w.bind("<Enter>", on_enter)
        w.bind("<Leave>", on_leave)
        w.bind("<Button-1>", on_click)
    return btn_frame


def _make_section_title(parent, text: str, accent: str = C_VERDE):
    """Cria título de seção com fonte nítida de 14 pt e espaçamento enxuto (D11)."""
    f = tk.Frame(parent, bg=C_BG2)
    f.pack(fill="x", pady=(4, 3))
    tk.Frame(f, width=4, bg=accent).pack(side="left", fill="y")
    tk.Label(f, text=text.upper(), font=F_BEBAS, bg=C_BG2, fg=C_AMARELO, padx=8).pack(side="left")
    return f


def _make_subsection(parent, text: str):
    f = tk.Frame(parent, bg=C_BG2)
    f.pack(fill="x", pady=(4, 2))
    tk.Label(f, text=text.upper(), font=F_BARLOW, bg=C_BG2, fg=C_TEXTO2).pack(side="left")
    tk.Frame(f, height=1, bg=C_MARROM).pack(side="left", fill="x", expand=True, padx=(8, 0))
    return f


def _make_row(parent, label_text: str, accent: str = C_AMARELO):
    """Cria linha padronizada com altura uniforme para alinhar combos e sliders lado a lado (D09)."""
    row = tk.Frame(parent, bg=C_BG2, height=26)
    row.pack(fill="x", padx=0, pady=2)
    lbl = _make_ss_label(row, label_text, accent)
    lbl.pack(side="left", padx=(0, 8))
    content = tk.Frame(row, bg=C_BG2)
    content.pack(side="left", fill="x", expand=True)
    return row, content


def _make_slider_row(
    parent,
    label_text: str,
    from_: float,
    to_: float,
    init: float,
    command=None,
    accent: str = C_AMARELO,
    fmt: str = "{:.2f}",
    scale: str = "linear",
    on_press: Optional[Callable] = None,
    on_release: Optional[Callable] = None,
):
    """Cria uma linha com `SelvaSlider` + `Label` e edição numérica inline por duplo clique (F03 / F05 / F06)."""
    row, content = _make_row(parent, label_text, accent)
    val_label = tk.Label(
        content,
        text=fmt.format(init),
        font=F_SPACE,
        bg=C_PRETO,
        fg=C_AMARELO,
        width=10,
        anchor="e",
        padx=4,
        pady=2,
        highlightthickness=1,
        highlightbackground=C_MARROM,
    )
    val_label.pack(side="right", padx=(8, 0))

    def _internal_cmd(val):
        val_label.config(text=fmt.format(float(val)))
        if command:
            command(val)

    slider = SelvaSlider(
        content,
        from_=from_,
        to_=to_,
        init=init,
        command=_internal_cmd,
        accent=accent,
        scale=scale,
        on_press=on_press,
        on_release=on_release,
    )
    slider.pack(side="left", fill="x", expand=True)

    def _open_inline_entry(event=None):
        """Abre campo `tk.Entry` sobreposto para digitação direta de valor numérico (F05)."""
        if not slider.enabled:
            return None
        entry = tk.Entry(
            content,
            font=F_SPACE,
            bg=C_BG3,
            fg=C_AMARELO,
            insertbackground=C_AMARELO,
            justify="right",
        )
        entry.place(in_=val_label, relx=0, rely=0, relwidth=1.0, relheight=1.0)
        entry.insert(0, f"{slider.value:.2f}".rstrip("0").rstrip("."))
        entry.select_range(0, "end")
        entry.focus_set()

        committed = {"done": False}

        def _commit_entry(e=None):
            if committed["done"]:
                return
            committed["done"] = True
            raw = str(entry.get()).strip().lower()
            for suffix in ("hz", "db", "s", "x"):
                raw = raw.replace(suffix, "")
            raw = raw.replace(",", ".").strip()
            try:
                new_val = float(raw)
                if slider.on_press:
                    slider.on_press()
                slider.set(new_val)
                _internal_cmd(slider.value)
                if slider.on_release:
                    slider.on_release()
            except Exception:
                pass
            try:
                entry.destroy()
            except Exception:
                pass

        def _cancel_entry(e=None):
            if committed["done"]:
                return
            committed["done"] = True
            try:
                entry.destroy()
            except Exception:
                pass

        entry.bind("<Return>", _commit_entry)
        entry.bind("<FocusOut>", _commit_entry)
        entry.bind("<Escape>", _cancel_entry)
        return entry

    val_label.bind("<Double-Button-1>", _open_inline_entry)
    slider.bind("<Double-Button-1>", _open_inline_entry)
    slider.open_inline_entry = _open_inline_entry
    return slider, val_label


def _make_combo_row(parent, label_text: str, values, init, command=None, accent: str = C_AMARELO):
    row, content = _make_row(parent, label_text, accent)
    combo_frame = tk.Frame(content, bg=C_BG3, highlightthickness=1, highlightbackground=C_MARROM)
    combo_frame.pack(side="left", fill="x", expand=True)
    tk.Frame(combo_frame, width=3, bg=C_AMARELO).pack(side="left", fill="y")
    combo = ttk.Combobox(combo_frame, values=values, state="readonly", font=F_BARLOW)
    combo.set(init)
    combo.pack(side="left", fill="x", expand=True, padx=2, pady=2)
    if command:
        combo.bind("<<ComboboxSelected>>", command)
    return combo


def _panel(parent, scrollable: bool = False):
    outer = tk.Frame(parent, bg=C_BG2)
    outer.pack(fill="both", expand=True, padx=12, pady=4)
    if hasattr(outer, "columnconfigure"):
        outer.columnconfigure(0, weight=1)
        outer.columnconfigure(1, weight=1)
    return outer


# ─────────────────────────────────────────────────────────────────────────────
#  Aba 1: Painel de Controles das Bandas e Edição Interativa
# ─────────────────────────────────────────────────────────────────────────────
def create_band_panel_widgets(
    parent,
    on_band_select=None,
    on_type_change=None,
    on_freq_change=None,
    on_gain_change=None,
    on_q_change=None,
    on_slope_change=None,
    on_mode_change=None,
    on_bypass_toggle=None,
    on_solo_toggle=None,
    on_add_band=None,
    on_delete_band=None,
    on_undo=None,
    on_redo=None,
    on_ab_compare=None,
    on_flatten=None,
    on_prev_band=None,
    on_next_band=None,
    on_freq_press=None,
    on_freq_release=None,
    on_gain_press=None,
    on_gain_release=None,
    on_q_press=None,
    on_q_release=None,
):
    """Constrói o painel responsivo de controle paramétrico das bandas (Aba 1 - D03 / D07)."""
    panel = _panel(parent)
    _make_section_title(panel, "Controles da Banda Selecionada & Edição", accent=C_VERDE)

    cols = tk.Frame(panel, bg=C_BG2)
    cols.pack(fill="both", expand=True)
    left_c = tk.Frame(cols, bg=C_BG2)
    left_c.pack(side="left", fill="both", expand=True, padx=(0, 8))
    right_c = tk.Frame(cols, bg=C_BG2)
    right_c.pack(side="right", fill="both", expand=True, padx=(8, 0))

    band_selector = _make_combo_row(
        left_c, "Banda Ativa", [f"Banda {i+1:02d}" for i in range(24)], "Banda 04", on_band_select, C_AMARELO
    )
    ftype_combo = _make_combo_row(
        left_c, "Tipo Filtro", [ft.name for ft in FilterType], FilterType.PEAKING.name, on_type_change, C_TEXTO
    )
    slope_combo = _make_combo_row(
        left_c, "Slope Corte", [s.name for s in Slope], Slope.S12.name, on_slope_change, C_TEXTO
    )
    mode_combo = _make_combo_row(
        left_c, "Canal M/S", [m.name for m in ChannelMode], ChannelMode.STEREO.name, on_mode_change, C_TEXTO
    )

    freq_slider, freq_label = _make_slider_row(
        right_c,
        "Frequência",
        20.0,
        20000.0,
        1000.0,
        on_freq_change,
        C_AMARELO,
        "{:.0f} Hz",
        scale="log",
        on_press=on_freq_press,
        on_release=on_freq_release,
    )
    gain_slider, gain_label = _make_slider_row(
        right_c,
        "Ganho (dB)",
        -24.0,
        24.0,
        0.0,
        on_gain_change,
        C_VERDE,
        "{:+.1f} dB",
        scale="linear",
        on_press=on_gain_press,
        on_release=on_gain_release,
    )
    q_slider, q_label = _make_slider_row(
        right_c,
        "Fator Q",
        0.1,
        30.0,
        1.0,
        on_q_change,
        C_AZUL,
        "{:.2f}",
        scale="log",
        on_press=on_q_press,
        on_release=on_q_release,
    )

    # D03: Duas linhas equilibradas de botões sem truncar "⇄ A/B: Slot A"
    btn_area = tk.Frame(panel, bg=C_BG2)
    btn_area.pack(fill="x", pady=(4, 2))

    btn_row1 = tk.Frame(btn_area, bg=C_BG2)
    btn_row1.pack(fill="x", pady=(0, 3))

    prev_btn = _styled_button(btn_row1, "◀ Ant", on_prev_band, accent=C_TEXTO2)
    prev_btn.pack(side="left", padx=(0, 6))
    next_btn = _styled_button(btn_row1, "Próx ▶", on_next_band, accent=C_TEXTO2)
    next_btn.pack(side="left", padx=(0, 8))

    add_btn = _styled_button(btn_row1, "+ Criar Banda", on_add_band, accent=C_VERDE)
    add_btn.pack(side="left", padx=(0, 6))
    del_btn = _styled_button(btn_row1, "🗑 Excluir Banda", on_delete_band, accent=C_VERMELHO, danger=True)
    del_btn.pack(side="left", padx=(0, 6))
    flat_btn = _styled_button(btn_row1, "⎯ Flat (0 dB)", on_flatten, accent=C_TEXTO2)
    flat_btn.pack(side="left", padx=(0, 6))

    btn_row2 = tk.Frame(btn_area, bg=C_BG2)
    btn_row2.pack(fill="x")

    bypass_btn = _styled_button(btn_row2, "Bypass Banda", on_bypass_toggle, accent=C_AMARELO)
    bypass_btn.pack(side="left", padx=(0, 6))
    solo_btn = _styled_button(btn_row2, "Solo (Band-Pass)", on_solo_toggle, accent=C_VERMELHO)
    solo_btn.pack(side="left", padx=(0, 10))

    undo_btn = None
    redo_btn = None
    ab_btn = None
    if on_undo:
        undo_btn = _styled_button(btn_row2, "↶ Undo", on_undo, accent=C_TEXTO2)
        undo_btn.pack(side="left", padx=(0, 6))
    if on_redo:
        redo_btn = _styled_button(btn_row2, "↷ Redo", on_redo, accent=C_TEXTO2)
        redo_btn.pack(side="left", padx=(0, 6))
    if on_ab_compare:
        ab_btn = _styled_button(btn_row2, "⇄ A/B: Slot A", on_ab_compare, accent=C_AMARELO)
        ab_btn.pack(side="left")

    return {
        "frame": panel,
        "band_selector": band_selector,
        "ftype_combo": ftype_combo,
        "freq_slider": freq_slider,
        "freq_label": freq_label,
        "gain_slider": gain_slider,
        "gain_label": gain_label,
        "q_slider": q_slider,
        "q_label": q_label,
        "slope_combo": slope_combo,
        "mode_combo": mode_combo,
        "prev_btn": prev_btn,
        "next_btn": next_btn,
        "bypass_btn": bypass_btn,
        "solo_btn": solo_btn,
        "add_btn": add_btn,
        "del_btn": del_btn,
        "flat_btn": flat_btn,
        "undo_btn": undo_btn,
        "redo_btn": redo_btn,
        "ab_btn": ab_btn,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Aba 2: Painel de Seleção de Entrada e Saída de Áudio & Hardware
# ─────────────────────────────────────────────────────────────────────────────
def create_audio_io_panel_widgets(
    parent,
    config: EqConfig,
    input_devices: Optional[List[str]] = None,
    output_devices: Optional[List[str]] = None,
    on_input_device=None,
    on_output_device=None,
    on_refresh_devices=None,
    on_sample_rate=None,
    on_buffer_size=None,
    on_oversampling=None,
    on_output_gain=None,
    on_ms_toggle=None,
    on_mute_toggle=None,
    on_output_gain_press=None,
    on_output_gain_release=None,
):
    """Constrói a aba dedicada para seleção de Entrada e Saída de Áudio e parâmetros de hardware (Aba 2 - D07 / D09)."""
    panel = _panel(parent)
    _make_section_title(panel, "Seleção de Dispositivos de Entrada e Saída de Áudio", accent=C_AZUL)

    cols = tk.Frame(panel, bg=C_BG2)
    cols.pack(fill="both", expand=True)
    left_c = tk.Frame(cols, bg=C_BG2)
    left_c.pack(side="left", fill="both", expand=True, padx=(0, 8))
    right_c = tk.Frame(cols, bg=C_BG2)
    right_c.pack(side="right", fill="both", expand=True, padx=(8, 0))

    in_vals = input_devices if input_devices else ["Padrão do Sistema (Entrada)"]
    out_vals = output_devices if output_devices else ["Padrão do Sistema (Saída)"]

    input_dev_combo = _make_combo_row(
        left_c, "Entrada (In)", in_vals, in_vals[0], on_input_device, C_VERDE
    )
    output_dev_combo = _make_combo_row(
        left_c, "Saída (Out)", out_vals, out_vals[0], on_output_device, C_AMARELO
    )
    out_gain_slider, out_gain_label = _make_slider_row(
        left_c,
        "Ganho Saída",
        -24.0,
        12.0,
        config.output_gain_db,
        on_output_gain,
        C_AMARELO,
        "{:+.1f} dB",
        scale="linear",
        on_press=on_output_gain_press,
        on_release=on_output_gain_release,
    )

    sample_rate = _make_combo_row(
        right_c, "Sample Rate", list(SUPPORTED_SAMPLE_RATES), config.sample_rate, on_sample_rate, C_AMARELO
    )
    buffer_size = _make_combo_row(
        right_c, "Buffer Size", [128, 256, 512, 1024, 2048], config.buffer_size, on_buffer_size, C_AMARELO
    )
    oversampling = _make_combo_row(
        right_c, "Oversampling", [1, 2, 4], config.oversampling, on_oversampling, C_AMARELO
    )

    btn_row = tk.Frame(panel, bg=C_BG2)
    btn_row.pack(fill="x", pady=(6, 2))
    refresh_btn = _styled_button(btn_row, "🔄 Atualizar Dispositivos", on_refresh_devices, accent=C_AZUL)
    refresh_btn.pack(side="left", padx=(0, 8))
    mute_btn = _styled_button(btn_row, "🔊 Mute / Ativar Som", on_mute_toggle, accent=C_VERDE)
    mute_btn.pack(side="left", padx=(0, 8))

    ms_init_text = "🔀 Mid/Side: ON" if config.ms_enabled else "🔀 Mid/Side: OFF"
    ms_btn = _styled_button(btn_row, ms_init_text, on_ms_toggle, accent=C_AMARELO)
    ms_btn.set_active(config.ms_enabled)
    ms_btn.pack(side="left", padx=(0, 12))

    io_status_label = tk.Label(
        btn_row,
        text="● Stream: Inicializando...",
        font=F_SPACE,
        bg=C_PRETO,
        fg=C_AMARELO,
        padx=8,
        pady=3,
        highlightthickness=1,
        highlightbackground=C_MARROM,
    )
    io_status_label.pack(side="right")

    return {
        "frame": panel,
        "input_dev_combo": input_dev_combo,
        "output_dev_combo": output_dev_combo,
        "sample_rate": sample_rate,
        "buffer_size": buffer_size,
        "oversampling": oversampling,
        "out_gain_slider": out_gain_slider,
        "out_gain_label": out_gain_label,
        "refresh_btn": refresh_btn,
        "mute_btn": mute_btn,
        "ms_btn": ms_btn,
        "io_status_label": io_status_label,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Aba 3: Modo de Teste em Tempo Real (Injeção de MP3 / WAV & Presets)
# ─────────────────────────────────────────────────────────────────────────────
def create_test_mode_panel_widgets(
    parent,
    config: EqConfig,
    on_source_change=None,
    on_load_media=None,
    on_generate_demo=None,
    on_play_pause=None,
    on_stop=None,
    on_loop_toggle=None,
    on_seek=None,
    on_test_gain=None,
    on_save_preset=None,
    on_load_preset=None,
    preset_names: Optional[List[str]] = None,
    on_factory_preset=None,
    on_export_wav=None,
    on_freeze_toggle=None,
):
    """Constrói a aba do Modo de Teste para injetar MP3/WAV no fluxo de áudio em tempo real (Aba 3 - D07 / D10)."""
    panel = _panel(parent)
    _make_section_title(panel, "Modo de Teste em Tempo Real (Player MP3 / WAV & Geradores)", accent=C_AMARELO)

    cols = tk.Frame(panel, bg=C_BG2)
    cols.pack(fill="both", expand=True)
    left_c = tk.Frame(cols, bg=C_BG2)
    left_c.pack(side="left", fill="both", expand=True, padx=(0, 8))
    right_c = tk.Frame(cols, bg=C_BG2)
    right_c.pack(side="right", fill="both", expand=True, padx=(8, 0))

    source_values = [
        InputSource.SYNTH_DRONE.name,
        InputSource.MEDIA_FILE.name,
        InputSource.PINK_NOISE.name,
        InputSource.SINE_SWEEP.name,
        InputSource.MICROPHONE.name,
    ]
    source_combo = _make_combo_row(
        left_c, "Fonte Fluxo", source_values, config.input_source.name, on_source_change, C_VERDE
    )
    seek_slider, seek_label = _make_slider_row(
        left_c, "Posição (s)", 0.0, 100.0, 0.0, on_seek, C_AMARELO, "{:.1f}s / 0.0s"
    )
    seek_label.config(width=13, text="0.0s / 0.0s")

    factory_list = preset_names if preset_names else PresetIO.list_factory_presets()
    preset_combo = _make_combo_row(
        left_c, "Preset Rápido", factory_list, factory_list[0], on_factory_preset, C_AMARELO
    )

    vol_slider, vol_label = _make_slider_row(
        right_c, "Nível Teste", 0.0, 1.5, 0.85, on_test_gain, C_VERDE, "{:.2f}x"
    )

    media_info_label = tk.Label(
        right_c,
        text="Arquivo: Nenhum MP3/WAV carregado (Usando Gerador em Tempo Real)",
        font=F_SPACE,
        bg=C_PRETO,
        fg=C_TEXTO2,
        anchor="w",
        padx=8,
        pady=4,
    )
    media_info_label.pack(fill="x", pady=4)

    # D10: Agrupamento dos 9 botões em 2 linhas e 3 grupos funcionais com Play/Pause destacado
    btn_area = tk.Frame(panel, bg=C_BG2)
    btn_area.pack(fill="x", pady=(4, 2))

    btn_row1 = tk.Frame(btn_area, bg=C_BG2)
    btn_row1.pack(fill="x", pady=(0, 3))

    load_media_btn = _styled_button(btn_row1, "📂 Inserir MP3/WAV", on_load_media, accent=C_VERDE)
    load_media_btn.pack(side="left", padx=(0, 6))
    demo_btn = _styled_button(btn_row1, "🎵 Gerar Faixa Demo", on_generate_demo, accent=C_AMARELO)
    demo_btn.pack(side="left", padx=(0, 4))

    tk.Frame(btn_row1, width=2, bg=C_MARROM).pack(side="left", fill="y", padx=8, pady=2)

    play_btn = _styled_button(btn_row1, "⏯ PLAY / PAUSE", on_play_pause, accent=C_AZUL)
    play_btn._label.config(padx=16)
    play_btn.pack(side="left", padx=(0, 6))
    stop_btn = _styled_button(btn_row1, "⏹ Stop", on_stop, accent=C_VERMELHO)
    stop_btn.pack(side="left", padx=(0, 6))
    loop_btn = _styled_button(btn_row1, "🔁 Loop: ON", on_loop_toggle, accent=C_AMARELO)
    loop_btn.set_active(True)
    loop_btn.pack(side="left", padx=(0, 8))

    btn_row2 = tk.Frame(btn_area, bg=C_BG2)
    btn_row2.pack(fill="x")

    export_wav_btn = _styled_button(btn_row2, "💾 Exportar WAV", on_export_wav, accent=C_TEXTO2)
    export_wav_btn.pack(side="left", padx=(0, 6))

    freeze_btn = _styled_button(btn_row2, "❄ Freeze FFT", on_freeze_toggle, accent=C_AZUL)
    freeze_btn.pack(side="left", padx=(0, 10))

    save_preset_btn = _styled_button(btn_row2, "↓ Salvar Preset", on_save_preset, accent=C_AMARELO)
    save_preset_btn.pack(side="left", padx=(0, 6))
    load_preset_btn = _styled_button(btn_row2, "↑ Carregar Preset", on_load_preset, accent=C_AMARELO)
    load_preset_btn.pack(side="left")

    return {
        "frame": panel,
        "source_combo": source_combo,
        "seek_slider": seek_slider,
        "seek_label": seek_label,
        "preset_combo": preset_combo,
        "vol_slider": vol_slider,
        "vol_label": vol_label,
        "media_info_label": media_info_label,
        "load_media_btn": load_media_btn,
        "demo_btn": demo_btn,
        "play_btn": play_btn,
        "stop_btn": stop_btn,
        "loop_btn": loop_btn,
        "export_wav_btn": export_wav_btn,
        "freeze_btn": freeze_btn,
        "save_preset_btn": save_preset_btn,
        "load_preset_btn": load_preset_btn,
    }


def create_eq_system_controls(
    parent,
    config: EqConfig,
    on_sample_rate=None,
    on_buffer_size=None,
    on_oversampling=None,
    on_save=None,
    on_load=None,
    on_source_change=None,
    on_load_wav=None,
    on_output_device=None,
    on_mute_toggle=None,
    output_devices: Optional[List[str]] = None,
):
    """Função mantida por compatibilidade retroativa com testes existentes."""
    return create_audio_io_panel_widgets(
        parent,
        config,
        output_devices=output_devices,
        on_output_device=on_output_device,
        on_sample_rate=on_sample_rate,
        on_buffer_size=on_buffer_size,
        on_oversampling=on_oversampling,
        on_mute_toggle=on_mute_toggle,
    )
