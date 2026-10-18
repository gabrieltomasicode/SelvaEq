"""
Reexportação de EqConfig e enums no subpacote `eq.io.config` para compatibilidade
integral com os diagramas de componentes (PB32) e de pacotes (PB31).
"""

from eq.config import (
    FilterType,
    ChannelMode,
    Slope,
    InputSource,
    SUPPORTED_SAMPLE_RATES,
    EqConfig,
)

__all__ = [
    "FilterType",
    "ChannelMode",
    "Slope",
    "InputSource",
    "SUPPORTED_SAMPLE_RATES",
    "EqConfig",
]
