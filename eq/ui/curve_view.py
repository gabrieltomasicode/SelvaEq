"""
Nó Interativo de Banda (`BandNode`) e Controlador de Curva Interativa (`InteractiveCurveController` - `eq/ui/curve_view.py`).

Implementa os requisitos PB04, PB19 e PB21 (`01_diagrama_classes.puml` e `06_sequencia_edicao_banda.puml`):
- Conversão bidirecional entre coordenadas de pixel `(x_px, y_px)` e domínio acústico `(freq_hz, gain_db)`
  usando escala logarítmica para frequência (20 Hz a 20 kHz) e linear em dB para ganho (-24 dB a +24 dB).
- Criação de novas bandas diretamente por clique/duplo-clique no gráfico de espectro (`create_band_at`).
- Edição de bandas existentes por arraste (`drag_to`), roda do mouse (`on_wheel` para Fator Q),
  reset de ganho (`reset_gain` - F01), Solo (`toggle_solo`) e Bypass (`toggle_bypass`).
- Exclusão de bandas diretamente por clique direito sobre o nó no gráfico ou tecla Delete (`remove_band_at`).
"""

import math
from typing import List, Optional, Tuple
from eq.config import FilterType
from eq.dsp.band import EqBand
from eq.dsp.engine import EqEngine
from eq.dsp.utils import clamp
from eq.ui.controls.band_controls import BandControls


class BandNode:
    """
    Representa um nó gráfico interativo na curva de resposta do equalizador (PB21).
    """

    MIN_FREQ = 20.0
    MAX_FREQ = 20000.0
    MIN_GAIN = -24.0
    MAX_GAIN = 24.0

    def __init__(
        self,
        band_index: int,
        engine: Optional[EqEngine] = None,
        x_px: float = 0.0,
        y_px: float = 0.0,
        canvas_width: float = 800.0,
        canvas_height: float = 300.0,
        hit_radius: float = 12.0,
    ):
        self.band_index: int = int(band_index)
        self.engine: Optional[EqEngine] = engine
        self.canvas_width: float = max(1.0, float(canvas_width))
        self.canvas_height: float = max(1.0, float(canvas_height))
        self.hit_radius: float = float(hit_radius)
        self.x_px: float = float(x_px)
        self.y_px: float = float(y_px)

        if self.engine is not None and 0 <= self.band_index < len(self.engine.bands):
            self.sync_from_band()

    @classmethod
    def freq_to_x(cls, freq_hz: float, width: float) -> float:
        """Converte frequência em Hz (escala log 20..20000 Hz) para coordenada X em pixels."""
        f = clamp(float(freq_hz), cls.MIN_FREQ, cls.MAX_FREQ)
        norm = math.log10(f / cls.MIN_FREQ) / math.log10(cls.MAX_FREQ / cls.MIN_FREQ)
        return norm * float(width)

    @classmethod
    def x_to_freq(cls, x_px: float, width: float) -> float:
        """Converte coordenada X em pixels para frequência em Hz (escala logarítmica)."""
        w = max(1.0, float(width))
        norm = clamp(float(x_px) / w, 0.0, 1.0)
        return cls.MIN_FREQ * math.pow(cls.MAX_FREQ / cls.MIN_FREQ, norm)

    @classmethod
    def gain_to_y(cls, gain_db: float, height: float) -> float:
        """Converte ganho em dB (-24..+24 dB) para coordenada Y em pixels (topo = +24 dB)."""
        g = clamp(float(gain_db), cls.MIN_GAIN, cls.MAX_GAIN)
        norm = (cls.MAX_GAIN - g) / (cls.MAX_GAIN - cls.MIN_GAIN)
        return norm * float(height)

    @classmethod
    def y_to_gain(cls, y_px: float, height: float) -> float:
        """Converte coordenada Y em pixels para ganho em dB (-24..+24 dB)."""
        h = max(1.0, float(height))
        norm = clamp(float(y_px) / h, 0.0, 1.0)
        return cls.MAX_GAIN - norm * (cls.MAX_GAIN - cls.MIN_GAIN)

    def set_canvas_size(self, width: float, height: float) -> None:
        """Atualiza as dimensões de referência do canvas em redimensionamentos de tela/fullscreen."""
        self.canvas_width = max(1.0, float(width))
        self.canvas_height = max(1.0, float(height))
        self.sync_from_band()

    def sync_from_band(self) -> None:
        """Atualiza `x_px` e `y_px` a partir dos valores atuais de `freq_hz` e `gain_db` da banda."""
        if self.engine is not None and 0 <= self.band_index < len(self.engine.bands):
            band = self.engine.bands[self.band_index]
            self.x_px = self.freq_to_x(band.freq_hz, self.canvas_width)
            self.y_px = self.gain_to_y(band.gain_db, self.canvas_height)

    def hit_test(self, px: float, py: float) -> bool:
        """Retorna True se o ponto em pixels `(px, py)` estiver dentro do raio de clique do nó (PB21)."""
        dist = math.hypot(float(px) - self.x_px, float(py) - self.y_px)
        return dist <= self.hit_radius

    def hit_test_acoustic(self, freq_hz: float, gain_db: float, max_oct_dist: float = 0.85) -> Tuple[bool, float]:
        """
        Verifica se um clique em coordenadas do gráfico `(freq_hz, gain_db)` atinge este nó,
        retornando `(atingiu, distancia_normalizada)`.
        """
        if self.engine is None or not (0 <= self.band_index < len(self.engine.bands)):
            return False, 999.0
        band = self.engine.bands[self.band_index]
        df_oct = math.log2(max(float(freq_hz), self.MIN_FREQ) / max(band.freq_hz, self.MIN_FREQ))
        dg_norm = (float(gain_db) - band.gain_db) / 6.0
        dist = float(math.hypot(df_oct, dg_norm))
        return dist <= max_oct_dist, dist

    def drag_to(self, px: float, py: float) -> None:
        """
        Arrasta o nó para a coordenada em pixels `(px, py)`, convertendo X para frequência
        logarítmica e Y para ganho dB e aplicando ao `EqEngine` (PB21 / PB06).
        """
        self.x_px = clamp(float(px), 0.0, self.canvas_width)
        self.y_px = clamp(float(py), 0.0, self.canvas_height)
        new_freq = self.x_to_freq(self.x_px, self.canvas_width)
        new_gain = self.y_to_gain(self.y_px, self.canvas_height)
        self.drag_to_acoustic(new_freq, new_gain)

    def drag_to_acoustic(self, freq_hz: float, gain_db: float) -> None:
        """Move o nó diretamente para `(freq_hz, gain_db)` no domínio acústico."""
        f = clamp(float(freq_hz), self.MIN_FREQ, self.MAX_FREQ)
        g = clamp(float(gain_db), self.MIN_GAIN, self.MAX_GAIN)
        if self.engine is not None and 0 <= self.band_index < len(self.engine.bands):
            BandControls.update_band_freq(self.engine, self.band_index, f)
            BandControls.update_band_gain(self.engine, self.band_index, g)
            self.sync_from_band()

    def reset_gain(self) -> None:
        """Reseta o ganho da banda associada para 0.0 dB (F01)."""
        if self.engine is not None and 0 <= self.band_index < len(self.engine.bands):
            BandControls.update_band_gain(self.engine, self.band_index, 0.0)
            self.sync_from_band()

    def on_wheel(self, delta: float) -> None:
        """
        Ajusta o fator Q da banda de forma multiplicativa a partir da roda do mouse (PB21).
        """
        if self.engine is not None and 0 <= self.band_index < len(self.engine.bands):
            band = self.engine.bands[self.band_index]
            step_factor = math.pow(1.12, float(delta))
            new_q = clamp(band.q * step_factor, 0.1, 30.0)
            BandControls.update_band_q(self.engine, self.band_index, new_q)

    def toggle_solo(self) -> None:
        """Alterna o estado Solo da banda associada (PB12)."""
        if self.engine is not None and 0 <= self.band_index < len(self.engine.bands):
            band = self.engine.bands[self.band_index]
            BandControls.update_band_solo(self.engine, self.band_index, not band.solo)

    def toggle_bypass(self) -> None:
        """Alterna o estado Bypass da banda associada (PB12 / PB16)."""
        if self.engine is not None and 0 <= self.band_index < len(self.engine.bands):
            band = self.engine.bands[self.band_index]
            BandControls.update_band_bypass(self.engine, self.band_index, not band.bypass)


class InteractiveCurveController:
    """
    Controlador MVC que gerencia a coleção de `BandNode` sobre o gráfico de espectro,
    permitindo criar, selecionar, editar (arrastar/scroll/solo/bypass) e excluir bandas
    diretamente pela interação no gráfico (PB04, PB18, PB19, PB21).
    """

    def __init__(self, engine: EqEngine, canvas_width: float = 800.0, canvas_height: float = 300.0):
        self.engine: EqEngine = engine
        self.canvas_width: float = float(canvas_width)
        self.canvas_height: float = float(canvas_height)
        self.nodes: List[BandNode] = []
        self.selected_index: int = 0
        self.dragging_index: Optional[int] = None
        self.sync_nodes()

    def sync_nodes(self) -> List[BandNode]:
        """Reconstrói e sincroniza os objetos `BandNode` com a lista atual de bandas do `EqEngine`."""
        self.nodes = [
            BandNode(
                band_index=i,
                engine=self.engine,
                canvas_width=self.canvas_width,
                canvas_height=self.canvas_height,
            )
            for i in range(len(self.engine.bands))
        ]
        if len(self.engine.bands) == 0:
            self.selected_index = -1
        else:
            self.selected_index = int(clamp(self.selected_index, 0, len(self.engine.bands) - 1))
        return self.nodes

    def set_canvas_size(self, width: float, height: float) -> None:
        """Atualiza o tamanho do gráfico durante redimensionamento de janela ou tela cheia."""
        self.canvas_width = max(1.0, float(width))
        self.canvas_height = max(1.0, float(height))
        for node in self.nodes:
            node.set_canvas_size(self.canvas_width, self.canvas_height)

    def find_closest_node(self, freq_hz: float, gain_db: float, max_oct_dist: float = 0.75) -> Optional[int]:
        """Localiza o índice da banda mais próxima das coordenadas `(freq_hz, gain_db)` no gráfico."""
        self.sync_nodes()
        best_idx: Optional[int] = None
        best_dist: float = 999.0
        for node in self.nodes:
            hit, dist = node.hit_test_acoustic(freq_hz, gain_db, max_oct_dist=max_oct_dist)
            if hit and dist < best_dist:
                best_dist = dist
                best_idx = node.band_index
        return best_idx

    def suggest_free_frequency(self, default_hz: float = 1000.0) -> float:
        """Sugere uma frequência livre em torno de `default_hz` caso já exista banda próxima (F07)."""
        existing = [b.freq_hz for b in self.engine.bands]
        if not existing:
            return clamp(default_hz, BandNode.MIN_FREQ, BandNode.MAX_FREQ)
        candidates = [default_hz, 500.0, 2000.0, 125.0, 4000.0, 63.0, 8000.0, 250.0, 1600.0, 6300.0, 12500.0]
        for cand in candidates:
            if all(abs(math.log2(max(cand, 20.0) / max(ef, 20.0))) > 0.25 for ef in existing):
                return cand
        return clamp(default_hz, BandNode.MIN_FREQ, BandNode.MAX_FREQ)

    def create_band_at(
        self,
        freq_hz: float,
        gain_db: float = 0.0,
        q: float = 1.0,
        ftype: FilterType = FilterType.PEAKING,
    ) -> Optional[EqBand]:
        """
        Cria uma nova banda nas coordenadas `(freq_hz, gain_db)` do gráfico de espectro (PB04 / PB21),
        registrando a transação no `StateController` (Undo/Redo).
        """
        if len(self.engine.bands) >= self.engine.config.num_bands:
            return None
        f = clamp(float(freq_hz), BandNode.MIN_FREQ, BandNode.MAX_FREQ)
        g = clamp(float(gain_db), BandNode.MIN_GAIN, BandNode.MAX_GAIN)
        self.engine.state_controller.begin_transaction("Criar Banda no Gráfico")
        band = self.engine.add_band(freq_hz=f, gain_db=g, q=q, ftype=ftype)
        self.engine.state_controller.commit()
        self.sync_nodes()
        self.selected_index = band.index
        return band

    def remove_band_at(self, index: int) -> bool:
        """
        Exclui a banda de índice `index` do `EqEngine` via interação no gráfico (clique direito / Delete),
        reindexando os nós restantes e registrando no `StateController` (Undo/Redo).
        """
        if not (0 <= index < len(self.engine.bands)):
            return False
        self.engine.state_controller.begin_transaction(f"Excluir Banda {index + 1}")
        self.engine.remove_band(index)
        self.engine.state_controller.commit()
        if self.dragging_index == index:
            self.dragging_index = None
        self.sync_nodes()
        return True
