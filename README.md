# 🎚️ SelvaEQ

<div align="center">
  <a href="https://git.io/typing-svg">
    <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=500&size=22&pause=1000&color=00C2A8&center=true&vCenter=true&width=600&lines=Equalizador+Param%C3%A9trico+em+Python;Audio+DSP+%26+Interface+gr%C3%A1fica;SelvaEQ" alt="SelvaEQ" />
  </a>
</div>

<p align="center">
  <strong>Protótipo de equalizador paramétrico estéreo desenvolvido em Python.</strong>
</p>

<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/DSP-Audio-FF8C00?style=for-the-badge&logo=audio-technica&logoColor=white" alt="DSP" />
  <img src="https://img.shields.io/badge/Tkinter-0088CC?style=for-the-badge&logo=tkinter&logoColor=white" alt="Tkinter" />
  <img src="https://img.shields.io/badge/Realtime-4B0082?style=for-the-badge&logo=audio&logoColor=white" alt="Realtime" />
</div>

<br>

O projeto **SelvaEQ** é um protótipo de equalizador digital pensado para processamento em tempo real de áudio, com interface gráfica interativa, controle de bandas paramétricas e arquitetura modular em Python.

A aplicação foi organizada para separar claramente as camadas de:

- configuração e parâmetros do sistema;
- processamento DSP do sinal;
- entrada/saída de áudio;
- interface gráfica e interação do usuário;
- controle de estado, presets e comparações A/B.

---

## 🎯 Objetivo

Criar uma ferramenta de equalização de áudio com:

- interface visual para ajuste de curva e ganhos;
- suporte a múltiplas bandas paramétricas;
- processamento em tempo real de sinal estéreo;
- controle de entrada/saída e fonte de áudio;
- possibilidade de carregar presets e manter histórico de ajustes.

---

## ✨ Funcionalidades principais

- Equalizador paramétrico com até 24 bandas;
- suporte a filtros como peaking, shelf, low/high-pass, notch e outros modos disponíveis na configuração;
- processamento estéreo com controle de ganho, bypass e opções Mid/Side;
- visualização do espectro e resposta do equalizador;
- reprodução/entrada via microfone, arquivo de áudio ou sinal sintético;
- controle de presets e comparação de estados A/B;
- arquitetura MVC/DI com separação entre model, interface e infraestrutura.

---

## 📂 Estrutura do repositório

```text
SelvaEq/
├── README.md
├── main.py                     # ponto de entrada da aplicação
├── test_runner.py              # execução da suíte de testes
├── cleancache.py               # utilitário de limpeza de cache
├── eq/
│   ├── __init__.py             # exports do pacote
│   ├── config.py               # configuração global do sistema
│   ├── dsp/
│   │   ├── band.py             # definição e estado de cada banda
│   │   ├── biquad_math.py      # cálculos dos coeficientes dos filtros
│   │   ├── engine.py           # motor DSP do equalizador
│   │   ├── spectrum.py         # FFT, peak hold e análise do espectro
│   │   ├── state_controller.py # Undo/Redo e presets de comparação A/B
│   │   └── utils.py            # utilitários de DSP
│   ├── io/
│   │   ├── audio_io.py         # streams de áudio e processamento em tempo real
│   │   ├── media_player.py     # reprodução e sinais de referência
│   │   ├── preset_io.py        # persistência de presets
│   │   └── config.py           # reexportação de tipos de configuração
│   └── ui/
│       ├── interface.py        # janela principal e coordenação da UI
│       ├── curve_view.py       # edição interativa da curva de resposta
│       ├── widgets.py          # widgets, abas e controles visuais
│       └── controls/
│           └── band_controls.py # controles específicos das bandas
├── tests/
│   ├── dsp/
│   ├── io/
│   └── ui/
└── Documentação/              # material complementar e documentação do projeto
```

---

## 🧩 Arquitetura do sistema

O ponto de entrada é `main.py`. Neste arquivo, a aplicação:

1. cria a configuração global (`EqConfig`);
2. inicializa o motor DSP (`EqEngine`);
3. monta o analisador de espectro (`SpectrumFFT`);
4. cria a camada de I/O de áudio (`AudioIO`);
5. injeta tudo na interface (`EqInterface`).

Essa organização segue um padrão modular e orientado à injeção de dependências, com separação entre:

- **Configuração**: taxas de amostragem, tamanho do buffer, número de bandas, modos de operação e fontes de entrada;
- **DSP**: processamento por banda, ganho, filtros e análise espectral;
- **I/O**: streams de entrada/saída, reprodução e processamento de arquivos;
- **UI**: janela principal, controles visuais, curva de equalização e feedback gráfico;
- **Estado**: histórico, presets e comparação de ajustes.

---

## 🔊 Fluxo de funcionamento

### 1) Captura e reprodução

A camada `eq/io/` gerencia a comunicação com dispositivos de áudio e a reprodução de origem do sinal. A aplicação pode receber áudio via:

- microfone;
- arquivo de mídia;
- sinais internos gerados programaticamente;
- fontes de referência para testes de equalização.

### 2) Processamento DSP

O `EqEngine` aplica a cadeia de processamento sobre o sinal recebido. Cada banda do equalizador ajusta frequência, ganho, Q e resposta do filtro conforme a configuração ativa.

### 3) Visualização

O `SpectrumFFT` analisa o sinal em frequência e a interface mostra a resposta do equalizador em paralelo com o espectro do áudio, permitindo ajuste visual imediato.

---

## 🛠️ Requisitos

- Python 3.x
- NumPy
- SciPy
- Matplotlib
- Tkinter
- `sounddevice` para entrada/saída de áudio em tempo real
- `soundfile` e/ou `ffmpeg` para suporte a formatos de áudio mais amplos

### Instalação das dependências principais

```bash
python -m pip install numpy scipy matplotlib
```

Para uso com áudio em dispositivos físicos:

```bash
python -m pip install sounddevice
```

---

## 🚀 Como executar

Na raiz do projeto:

```bash
python main.py
```

Isso inicia a interface gráfica do equalizador e o processamento do áudio conforme a configuração ativa.

---

## ✅ Testes

A suíte de testes do projeto pode ser executada com:

```bash
python test_runner.py
```

Os testes estão organizados em módulos de DSP, I/O e interface, e ajudam a validar comportamento do processamento e da camada visual.

---

## 📌 Observações

Este repositório reúne tanto a implementação do protótipo quanto a documentação estrutural do projeto. O objetivo central é demonstrar a construção de um equalizador paramétrico em Python com abordagem modular, sustentável e alinhada com boas práticas de engenharia de software.

---

## ✒️ Autor

<div align="center">
  <strong>Gabriel Tomasi de Melo</strong><br>
  <em>Estudante de Análise e Desenvolvimento de Sistemas</em>
  <br><br>
  <a href="https://github.com/gabrieltomasicode">
    <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub" />
  </a>
</div>
