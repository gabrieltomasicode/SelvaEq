"""
Ponto de entrada da aplicação SelvaEQ (`main.py`).

Refatorado a partir de `SelvaSonicScripts/main.py`, mantendo o padrão de
Injeção de Dependência (DI) na arquitetura MVC:
1. Instancia `EqConfig` (Configuração única).
2. Instancia `EqEngine` (Model DSP) e `SpectrumFFT` (Analisador).
3. Instancia `AudioIO` (Infraestrutura de Stream Duplex).
4. Injeta as dependências em `EqInterface` (View + Controllers).
"""

from eq.config import EqConfig
from eq.dsp.engine import EqEngine
from eq.dsp.spectrum import SpectrumFFT
from eq.io.audio_io import AudioIO
from eq.ui.interface import EqInterface
from eq.ui.widgets import HAS_TK, tk


def main():
    print("🎚️ Inicializando o Ecossistema SelvaEQ (24-Band Parametric Equalizer)...")

    # 1. Cria a configuração única global
    config = EqConfig(sample_rate=44100, buffer_size=512, num_bands=24)

    # 2. Inicializa o motor DSP e o analisador de espectro
    engine = EqEngine(config=config, init_default_bands=True)
    analyzer = SpectrumFFT(fft_size=2048, sample_rate=config.sample_rate)

    # 3. Inicializa o motor de I/O de áudio injetando config e engine
    audio_io = AudioIO(config=config, engine=engine)

    # 4. Inicia a interface gráfica com Injeção de Dependência
    root = tk.Tk()
    if HAS_TK and hasattr(root, "title"):
        root.title("SelvaEQ - 24-Band Parametric Equalizer")
    app = EqInterface(master=root, engine=engine, io=audio_io, analyzer=analyzer)

    if HAS_TK and hasattr(root, "mainloop"):
        root.mainloop()
    return app


if __name__ == "__main__":
    main()
