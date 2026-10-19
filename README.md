# SelvaEQ

SelvaEQ é um protótipo desktop de equalizador paramétrico estéreo feito em Python. A aplicação reúne processamento digital de áudio (DSP), reprodução/captura de áudio, visualização de espectro e uma interface gráfica interativa. O código está organizado para separar essas responsabilidades e permitir testar o motor sem depender da interface ou de uma placa de som.

## Arquitetura

O ponto de entrada é `main.py`. Ele cria uma configuração compartilhada (`EqConfig`), injeta essa configuração no motor DSP (`EqEngine`) e na camada de áudio (`AudioIO`) e, por fim, cria a interface (`EqInterface`) com o analisador (`SpectrumFFT`). Esse arranjo segue uma separação semelhante a MVC: o motor e a configuração formam o modelo, `AudioIO` cuida da infraestrutura de áudio, e a interface coordena a interação e a visualização.

```text
main.py
  ├── eq/config.py                 Configuração, fontes de áudio e tipos de filtro
  ├── eq/dsp/
  │   ├── engine.py                Cadeia de processamento e equalização
  │   ├── band.py                  Filtros e estado de cada banda
  │   ├── biquad_math.py           Cálculo de coeficientes
  │   ├── spectrum.py              FFT, suavização e Peak Hold
  │   ├── state_controller.py      Undo/Redo e comparação A/B
  │   └── utils.py                 Funções auxiliares de DSP
  ├── eq/io/
  │   ├── audio_io.py              Streams, callbacks e processamento de arquivos
  │   ├── media_player.py          Reprodução e geração de sinais de referência
  │   ├── preset_io.py             Presets de fábrica e persistência
  │   └── config.py                Reexportação de tipos de configuração
  └── eq/ui/
      ├── interface.py             Janela e coordenação da interface
      ├── curve_view.py            Nós e edição interativa da curva
      ├── widgets.py               Widgets, abas e controles
      └── controls/band_controls.py Controles das bandas
```

### Camadas principais

- **Configuração (`eq/config.py`)** — `EqConfig` compartilha taxa de amostragem, tamanho de bloco, número de bandas, oversampling, ganho, bypass, modo Mid/Side e fonte de entrada. A configuração limita o equalizador a 24 bandas.
- **DSP (`eq/dsp/`)** — `EqEngine` processa o sinal estéreo por uma cascata de bandas `EqBand`. Cada banda aplica um tipo de filtro selecionado, como peaking, shelf, corte, passa-banda, notch ou all-pass. Há suporte a modos estéreo, esquerdo, direito, Mid e Side, além de bypass, solo, suavização de parâmetros e oversampling opcional.
- **I/O (`eq/io/`)** — `AudioIO` controla os streams de entrada/saída e encaminha blocos ao motor. `RealtimeMediaStreamer` fornece áudio de arquivo ou sinais de demonstração. A fila visual desacopla a atualização do espectro do callback de áudio.
- **Interface (`eq/ui/`)** — `EqInterface` constrói a janela Tkinter, conecta ações aos componentes de DSP/I/O e atualiza a curva e o espectro Matplotlib. `InteractiveCurveController` mantém os nós da curva sincronizados com as bandas do motor.
- **Estado e presets** — `StateController` mantém histórico de Undo/Redo e slots A/B. `PresetIO` oferece presets de fábrica e funções para salvar/carregar configurações.

## Fluxo de áudio

### Reprodução/captura em tempo real

1. `AudioIO` abre um stream PortAudio via `sounddevice`, se disponível. Para fontes geradas ou arquivos usa saída; para `MICROPHONE`, tenta abrir um stream full-duplex.
2. A fonte de áudio produz blocos estéreo: microfone, arquivo ou sinal interno (drone sintetizado, ruído rosa ou varredura senoidal).
3. O callback chama `EqEngine.process()`. O motor aplica, conforme a configuração, oversampling, codificação Mid/Side, bandas ativas em sequência, decodificação Mid/Side e ganho de saída.
4. O bloco processado é copiado para a saída de áudio. Nível de pico/RMS, clipping e status do stream são registrados, e uma cópia do bloco é colocada na fila visual.
5. A interface consome os blocos da fila; `SpectrumFFT` calcula a FFT, aplica janela de Hann e suavização temporal, e o gráfico apresenta o espectro junto à curva de resposta do equalizador.

Se `sounddevice` ou um dispositivo de áudio não estiver disponível, a camada de I/O pode usar um stream simulado para execução headless e testes. Esse modo não fornece saída de áudio física.

### Processamento de arquivo

`AudioIO.process_wav_file()` permite processar um WAV completo fora do stream em tempo real, usando blocos e o mesmo motor DSP. O método pode retornar o áudio processado e gravar um WAV de saída quando um caminho de destino é informado.

## Funcionamento do protótipo

- A janela é iniciada com 24 posições disponíveis para bandas e sete bandas padrão carregadas.
- A aba **Equalizador & Bandas** permite alterar parâmetros e a curva: arraste os nós para ajustar frequência/ganho, use a roda do mouse para o Q, crie bandas com duplo clique em uma área livre ou `Shift`+clique, e remova-as com clique direito ou `Delete`. Também há operações de solo, bypass, Flat, Undo/Redo e comparação A/B.
- A aba **Entrada & Saída de Áudio** reúne seleção de dispositivos, taxa de amostragem, tamanho de bloco, oversampling, ganho, mute e configuração Mid/Side.
- A aba **Modo de Teste** permite carregar arquivos de áudio, controlar reprodução/pausa/parada/loop/posição, selecionar presets, congelar a visualização FFT e gerar uma faixa de demonstração.
- Os tipos de entrada declarados em `InputSource` incluem `SYNTH_DRONE`, `MEDIA_FILE`, `PINK_NOISE`, `SINE_SWEEP` e `MICROPHONE`.
- A lista de filtros, modos de canal e inclinações disponíveis está definida em `eq/config.py`. O número de bandas usadas pode ser menor que o limite de 24.

## Requisitos

- Python 3
- NumPy, SciPy e Matplotlib
- Tkinter para abrir a interface gráfica (normalmente fornecido com a instalação do Python)
- `sounddevice` para áudio em tempo real com dispositivos físicos
- `soundfile` e/ou `ffmpeg` para ampliar a decodificação de formatos de áudio. WAV também possui caminhos de leitura usando bibliotecas padrão/Científicas.

Este repositório não contém atualmente um arquivo de dependências. Em um ambiente Python com `pip`, as dependências principais podem ser instaladas com:

```bash
python -m pip install numpy scipy matplotlib
```

Para habilitar áudio com dispositivos, instale também:

```bash
python -m pip install sounddevice
```

Para suporte adicional de decodificação, instale `soundfile`; `ffmpeg` é um executável externo opcional usado como alternativa por alguns formatos.

## Executar

Na raiz do repositório:

```bash
python main.py
```

## Testes

A suíte usa `unittest` e pode ser executada com:

```bash
python test_runner.py
```

Os testes estão em `tests/` e cobrem DSP, I/O, presets, interface e interoperabilidade. Parte dos testes de reprodução MP3 depende da disponibilidade de `ffmpeg`.
