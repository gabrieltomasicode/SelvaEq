"""
Configurações Globais e Tipos Compartilhados do Equalizador (SelvaEQ).

Refatorado a partir de `synth/config.py` (SynthConfig) do projeto SelvaSonic,
atendendo aos requisitos PB04, PB11, PB22, PB28, PB29 e PB31.
Contrato de tipos compartilhado entre a Fase 1 (Python) e a Fase 2 (C++/JUCE).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union
import numpy as np


class FilterType(Enum):
    """
    Tipos de filtro suportados por cada banda do equalizador (PB05 / PB31).
    """
    PEAKING = "PEAKING"
    LOW_SHELF = "LOW_SHELF"
    HIGH_SHELF = "HIGH_SHELF"
    LOW_CUT = "LOW_CUT"
    HIGH_CUT = "HIGH_CUT"
    BAND_PASS = "BAND_PASS"
    NOTCH = "NOTCH"
    ALL_PASS = "ALL_PASS"


class ChannelMode(Enum):
    """
    Modos de roteamento de canal por banda (PB22 / PB31).
    """
    STEREO = "STEREO"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    MID = "MID"
    SIDE = "SIDE"


class Slope(Enum):
    """
    Inclinações (slopes) para filtros de corte Butterworth em cascata (PB05 / PB31).
    Cada valor representa a atenuação em dB por oitava (6 a 96 dB/oct).
    """
    S6 = 6
    S12 = 12
    S24 = 24
    S48 = 48
    S96 = 96

    @property
    def order(self) -> int:
        """Retorna a ordem do filtro Butterworth correspondente (1 a 16)."""
        return self.value // 6


class InputSource(Enum):
    """
    Fonte de sinal de entrada para o fluxo de áudio em tempo real do protótipo Python:
    - SYNTH_DRONE: Acorde rico em harmônicos gerado em tempo real (padrão audível imediato).
    - MEDIA_FILE: Arquivo de áudio (.MP3, .WAV, .FLAC, .OGG) injetado no fluxo como entrada em tempo real.
    - PINK_NOISE: Ruído rosa contínuo (ideal para calibrar e testar a curva no analisador FFT).
    - SINE_SWEEP: Varredura senoidal contínua (20 Hz a 20 kHz) para inspeção auditiva e espectral.
    - MICROPHONE: Captura em tempo real da entrada física selecionada (microfone / interface de linha).
    """
    SYNTH_DRONE = "SYNTH_DRONE"
    MEDIA_FILE = "MEDIA_FILE"
    WAV_FILE = "MEDIA_FILE"  # Alias de compatibilidade retroativa
    PINK_NOISE = "PINK_NOISE"
    SINE_SWEEP = "SINE_SWEEP"
    MICROPHONE = "MICROPHONE"


SUPPORTED_SAMPLE_RATES = (44100, 48000, 88200, 96000, 176400, 192000)


@dataclass
class EqConfig:
    """
    Configuração global do motor de equalização (Modelo).

    Reaproveita a estrutura de `SynthConfig` (`synth/config.py`), mantendo
    `sample_rate`, `buffer_size`, `validate()`, `__post_init__()` e `last_audio_buffer`,
    e removendo atributos exclusivos de síntese (WaveType, ADSR, LFO/HFO/FM,
    polifonia, wavetables e porta MIDI).
    Inclui `auto_gain` opcional para compensação automática de ganho de saída (F13).
    """
    sample_rate: int = 44100
    buffer_size: int = 512
    input_device: Optional[Union[int, str]] = None
    output_device: Optional[Union[int, str]] = None
    num_bands: int = 24
    ms_enabled: bool = False
    oversampling: int = 1
    output_gain_db: float = 0.0
    bypass: bool = False
    auto_gain: bool = False
    input_source: InputSource = InputSource.SYNTH_DRONE
    media_file_path: Optional[str] = None
    last_audio_buffer: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        """Valida automaticamente os parâmetros ao instanciar a configuração."""
        if isinstance(self.input_source, str):
            key = self.input_source.upper()
            self.input_source = InputSource[key] if key in InputSource.__members__ else InputSource.SYNTH_DRONE
        self.validate()

    def validate(self) -> None:
        """
        Valida os parâmetros de configuração do equalizador (PB29 / PB33).

        Raises:
            ValueError: Se algum parâmetro estiver fora dos limites operacionais.
        """
        if not isinstance(self.sample_rate, (int, float)) or self.sample_rate <= 0:
            raise ValueError(f"sample_rate deve ser maior que 0 (recebido: {self.sample_rate}).")
        if not isinstance(self.buffer_size, int) or self.buffer_size <= 0:
            raise ValueError(f"buffer_size deve ser maior que 0 (recebido: {self.buffer_size}).")
        if not isinstance(self.num_bands, int) or not (1 <= self.num_bands <= 24):
            raise ValueError(f"num_bands deve estar entre 1 e 24 (recebido: {self.num_bands}).")
        if self.oversampling not in (1, 2, 4):
            raise ValueError(f"oversampling deve ser 1, 2 ou 4 (recebido: {self.oversampling}).")
