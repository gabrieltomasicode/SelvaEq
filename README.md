# 🎚️ SelvaEQ — Protótipo Python MVC/POO (24-Band Parametric Equalizer)

Protótipo funcional e validado matematicamente do **SelvaEQ**, refatorado a partir da arquitetura modular MVC/POO do sintetizador **SelvaSonic** conforme o documento `Planejamento EQ.docx` e os 10 diagramas UML (`PB01` a `PB34`).

## ✨ Novidades e Melhorias Implementadas

1. **Correção Definitiva da Saída de Áudio (`eq/io/audio_io.py`):**
   - Abre `sd.OutputStream` diretamente para todos os modos de teste (evitando bloqueios ou erro `Invalid number of channels` quando o microfone do PC é Mono ou inexistente).
   - No modo Microfone (`MICROPHONE`), detecta automaticamente os canais reais de entrada (`1` Mono ou `2` Estéreo) e faz *upmix* automático de microfone Mono para Estéreo `[frames, 2]`.
   - Realiza *fallback* automático para o `default_samplerate` nativo da placa de som caso o driver do SO rejeite a taxa inicial, recalculando todas as bandas (`EqEngine.prepare`) e o streamer.
2. **Módulo Dedicado de Streaming MP3/WAV em Tempo Real (`eq/io/media_player.py`):**
   - Classe POO `RealtimeMediaStreamer` (100% isolada de UI — `PB28`) que decodifica arquivos `.mp3`, `.wav`, `.flac` e `.ogg` (via `soundfile`, `scipy.io.wavfile`, `miniaudio`, `pydub` ou `ffmpeg`) e injeta blocos `[frames, 2]` quadro a quadro no callback de tempo real, simulando uma entrada física ao vivo.
   - Controles completos de transporte (`play`, `pause`, `stop`, `seek`, `set_loop`, `set_gain`), geradores em tempo real (`SYNTH_DRONE`, `PINK_NOISE`, `SINE_SWEEP`) e gerador instantâneo de faixa demo musical (`generate_demo_track_file`).
3. **Sistema de 3 Abas Modulares (`eq/ui/interface.py` & `eq/ui/widgets.py`):**
   - **Aba 1 — Equalizador & Bandas:** Edição paramétrica das 24 bandas + botões `+ Criar Banda`, `🗑 Excluir Banda`, `Bypass`, `Solo (Band-Pass)`, `↶ Undo`, `↷ Redo` e `⇄ A/B`.
   - **Aba 2 — Entrada & Saída de Áudio:** Seleção independente de Dispositivo de Entrada (`Input Device`) e Dispositivo de Saída (`Output Device`), botão `🔄 Atualizar Dispositivos`, `Sample Rate`, `Buffer Size`, `Oversampling`, `Ganho Saída (dB)`, `Matriz Mid/Side Global` e `🔊 Mute / Ativar Som`.
   - **Aba 3 — Modo de Teste (MP3 / WAV):** Seletor de Fonte no Fluxo (`SYNTH_DRONE`, `MEDIA_FILE`, `PINK_NOISE`, `SINE_SWEEP`, `MICROPHONE`), botão `📂 Inserir MP3 / WAV`, botão `🎵 Gerar Faixa Demo`, controles `⏯ Play/Pause`, `⏹ Stop`, `🔁 Loop`, slider de `Posição (s)` e `Nível de Teste`.
4. **UI Responsiva e Modo Tela Cheia (`F11` / `Esc`):**
   - Layout com pesos dinâmicos (`rowconfigure`/`columnconfigure`) e sliders adaptativos (`<Configure>`) que se ajustam a qualquer redimensionamento de janela, maximização ou **Tela Cheia (`F11` ou botão `⛶ Tela Cheia`)**.
5. **Bandas Criáveis, Editáveis e Excluíveis Diretamente no Gráfico de Espectro (`eq/ui/curve_view.py`):**
   - **Criar Banda:** Duplo clique esquerdo (ou `Shift + Clique esquerdo`) em qualquer ponto vazio do gráfico cria uma nova banda naquela frequência e ganho (até 24 bandas).
   - **Editar Banda:** Clique esquerdo + arrastar move Frequência (Hz) e Ganho (dB); Roda do mouse ajusta o Fator Q; `Alt + Clique` alterna Solo; `Ctrl + Clique` alterna Bypass.
   - **Excluir Banda:** Clique direito (`button == 3`) diretamente sobre o nó no gráfico (ou tecla `Delete` / `BackSpace`) exclui a banda (com suporte a Undo `Ctrl+Z`).

## 🧪 Como Executar os 29 Testes Unitários

```bash
python3 test_runner.py
```

## 🚀 Como Iniciar a Aplicação

```bash
python3 main.py
```
