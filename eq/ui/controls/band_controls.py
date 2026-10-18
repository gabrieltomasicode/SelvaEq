"""
Controladores de Banda (`BandControls` - `eq/ui/controls/band_controls.py`).

Refatorado a partir de `ui/controls/filter_controls.py` do SelvaSonic:
- As funções `update_filter_type`, `update_filter_freq` e `update_filter_q` foram
  generalizadas para receber o `engine: EqEngine` e o índice da banda `i: int`,
  escrevendo na instância `EqBand` correspondente e adicionando suporte a Ganho,
  Modo de Canal (L/R/M/S), Slope (6..96 dB/oct), Bypass e Solo.
"""

from typing import Any, Optional
from eq.config import FilterType, ChannelMode, Slope
from eq.dsp.engine import EqEngine


class BandControls:
    """
    Controlador MVC para atualização dos parâmetros de cada uma das 24 bandas do `EqEngine`.
    """

    @staticmethod
    def update_band_type(engine: EqEngine, i: int, value: str | FilterType) -> None:
        """Atualiza o tipo de filtro (`FilterType`) da banda `i`."""
        ftype = value if isinstance(value, FilterType) else FilterType[str(value).upper()]
        engine.set_param(i, "ftype", ftype)

    @staticmethod
    def update_band_freq(engine: EqEngine, i: int, value: float, label: Optional[Any] = None) -> None:
        """Atualiza a frequência central/corte (`freq_hz`) da banda `i` e o label opcional."""
        val = float(value)
        engine.set_param(i, "freq_hz", val)
        if label is not None and hasattr(label, "config"):
            label.config(text=f"{engine.bands[i].freq_hz:.0f} Hz")

    @staticmethod
    def update_band_gain(engine: EqEngine, i: int, value: float, label: Optional[Any] = None) -> None:
        """Atualiza o ganho (`gain_db`) da banda `i` e o label opcional."""
        val = float(value)
        engine.set_param(i, "gain_db", val)
        if label is not None and hasattr(label, "config"):
            label.config(text=f"{engine.bands[i].gain_db:+.1f} dB")

    @staticmethod
    def update_band_q(engine: EqEngine, i: int, value: float, label: Optional[Any] = None) -> None:
        """Atualiza o fator de qualidade (`q`) da banda `i` e o label opcional."""
        val = float(value)
        engine.set_param(i, "q", val)
        if label is not None and hasattr(label, "config"):
            label.config(text=f"{engine.bands[i].q:.2f}")

    @staticmethod
    def update_band_mode(engine: EqEngine, i: int, value: str | ChannelMode) -> None:
        """Atualiza o modo de roteamento (`STEREO`, `LEFT`, `RIGHT`, `MID`, `SIDE`) da banda `i` (PB22)."""
        mode = value if isinstance(value, ChannelMode) else ChannelMode[str(value).upper()]
        engine.set_param(i, "mode", mode)

    @staticmethod
    def update_band_slope(engine: EqEngine, i: int, value: str | int | Slope) -> None:
        """Atualiza a inclinação Butterworth (`S6`..`S96`) da banda `i` (PB05)."""
        engine.set_param(i, "slope", value)

    @staticmethod
    def update_band_bypass(engine: EqEngine, i: int, value: bool) -> None:
        """Ativa ou desativa o bypass da banda `i` (PB12 / PB16)."""
        engine.set_param(i, "bypass", bool(value))

    @staticmethod
    def update_band_solo(engine: EqEngine, i: int, value: bool) -> None:
        """Ativa ou desativa o modo solo (escuta isolada band-pass) da banda `i` (PB12)."""
        engine.set_param(i, "solo", bool(value))


# Exporta funções de nível de módulo para compatibilidade direta
update_band_type = BandControls.update_band_type
update_band_freq = BandControls.update_band_freq
update_band_gain = BandControls.update_band_gain
update_band_q = BandControls.update_band_q
update_band_mode = BandControls.update_band_mode
update_band_slope = BandControls.update_band_slope
update_band_bypass = BandControls.update_band_bypass
update_band_solo = BandControls.update_band_solo
