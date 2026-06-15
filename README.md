# IA Local Benchmarks — Hardware para IA local

Este repositório contém os scripts, dados e gráficos utilizados no artigo **Hardware para IA local: NPUs, GPUs e o futuro dos computadores pessoais**.

O objetivo é gerar resultados rastreáveis para avaliar a execução local de cargas relacionadas à inteligência artificial em computador pessoal. Os testes consideram operações sintéticas, modelos reais de visão computacional e modelos de linguagem executados localmente via Ollama.

A comparação experimental foi feita principalmente entre CPU e GPU. A NPU é discutida no artigo de forma técnica e bibliográfica, pois não houve acesso direto a esse acelerador por meio dos frameworks utilizados no ambiente de teste.

## Estado atual dos testes

O projeto possui os seguintes grupos de benchmark:

| Grupo | Script | Objetivo |
|---|---|---|
| Coleta do sistema | `coletar_sistema.py` | Registrar sistema operacional, CPU, RAM, GPU, CUDA e PyTorch. |
| NumPy CPU | `benchmark_numpy_cpu.py` | Medir multiplicação de matrizes em CPU. |
| PyTorch CPU/GPU | `benchmark_torch_cpu_gpu.py` | Comparar CPU e CUDA em multiplicação de matrizes e MLP sintético. |
| Limite da GPU | `benchmark_limite_gpu.py` | Avaliar matrizes maiores em CUDA e pico de VRAM. |
| CNN sintético | `benchmark_cnn_sintetico.py` | Simular uma carga de visão computacional em CPU e CUDA. |
| Modelos reais | `benchmark_modelos_reais.py` | Testar MobileNetV2 e ResNet18 em CPU e CUDA. |
| LLMs locais | `benchmark_ollama_llms.py` | Testar qwen3.5 e gemma3:4b localmente via Ollama. |

## Estrutura

```text
ia-local-bench/
├── README.md
├── requirements.txt
├── scripts/
│   ├── coletar_sistema.py
│   ├── benchmark_numpy_cpu.py
│   ├── benchmark_torch_cpu_gpu.py
│   ├── benchmark_limite_gpu.py
│   ├── benchmark_cnn_sintetico.py
│   ├── benchmark_modelos_reais.py
│   ├── benchmark_ollama_llms.py
│   ├── gerar_graficos.py
│   ├── gerar_graficos_com_ollama_llms.py
│   └── rodar_tudo.py
├── dados/
│   ├── raw/
│   └── processed/
└── resultados/
    ├── graficos/
    └── tabelas/
```

## Instalação

Recomenda-se usar um ambiente virtual:

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/WSL/macOS
# .venv\Scripts\activate      # Windows PowerShell
pip install -r requirements.txt
```

No PowerShell do Windows, caso o ambiente virtual já exista com outro nome, por exemplo `venv_ia_local_bench`, use:

```powershell
.\venv_ia_local_bench\Scripts\activate
```

Para GPU NVIDIA com CUDA, instale a versão do PyTorch adequada ao seu ambiente conforme o seletor oficial do PyTorch. No ambiente utilizado no artigo, a execução com CUDA só funcionou após instalar o PyTorch com suporte CUDA.

Caso o teste de modelos reais apresente erro relacionado ao `torchvision`, instale também:

```bash
pip install torchvision
```

## Ollama e modelos de linguagem locais

Os testes com LLMs utilizam o Ollama rodando localmente na porta padrão `11434`.

Antes de executar o benchmark de LLMs, verifique se o Ollama está funcionando:

```bash
ollama list
```

Os modelos utilizados foram:

```bash
ollama pull qwen3.5
ollama pull gemma3:4b
```

O benchmark de LLMs usa três prompts de complexidade diferente e coleta tempo total, tempo de carregamento, tokens gerados e tokens por segundo.

## Como rodar

Para rodar a coleta básica original:

```bash
python scripts/rodar_tudo.py
```

Para reproduzir o conjunto completo de testes usados no artigo, execute as etapas manualmente:

```bash
python scripts/coletar_sistema.py
python scripts/benchmark_numpy_cpu.py
python scripts/benchmark_torch_cpu_gpu.py
python scripts/benchmark_limite_gpu.py
python scripts/benchmark_cnn_sintetico.py
python scripts/benchmark_modelos_reais.py
python scripts/benchmark_ollama_llms.py
python scripts/gerar_graficos.py
python scripts/gerar_graficos_com_ollama_llms.py
```

## Saídas esperadas

Os principais arquivos gerados ou utilizados pelo artigo são:

- `dados/raw/system_info.json`: informações do ambiente experimental.
- `dados/raw/benchmark_numpy_cpu.csv`: benchmark de multiplicação de matrizes com NumPy em CPU.
- `dados/raw/benchmark_torch_cpu_gpu.csv`: benchmark com PyTorch em CPU e CUDA.
- `dados/raw/benchmark_limite_gpu.csv`: teste de matrizes maiores em CUDA.
- `dados/raw/benchmark_cnn_sintetico.csv`: inferência em CNN sintético.
- `dados/raw/benchmark_modelos_reais.csv`: inferência em MobileNetV2 e ResNet18.
- `dados/raw/benchmark_ollama_llms.csv`: geração local com qwen3.5 e gemma3:4b via Ollama.
- `dados/processed/resultados_consolidados.csv`: dados consolidados.
- `resultados/tabelas/tabela_resultados_artigo.csv`: tabela final usada no artigo.
- `resultados/graficos/*.png`: gráficos utilizados na análise.

## Principais gráficos

Os gráficos gerados incluem:

- `tempo_matmul.png`: comparação de multiplicação de matrizes.
- `tempo_mlp_inferencia.png`: inferência em MLP sintético.
- `tempo_limite_gpu.png`: teste de limite da GPU.
- `tempo_cnn_sintetico.png`: inferência em CNN sintético.
- `tempo_modelo_real_mobilenet_v2.png`: inferência em MobileNetV2.
- `tempo_modelo_real_resnet18.png`: inferência em ResNet18.
- `tempo_ollama_llms_total.png`: tempo total médio dos LLMs locais.
- `tokens_ollama_llms.png`: velocidade média em tokens por segundo.

## Interpretação geral

Os resultados mostram que a GPU acelerou significativamente as cargas paralelizáveis, como multiplicação de matrizes, MLP, CNN sintético e modelos reais de visão computacional. Nos modelos MobileNetV2 e ResNet18, a vantagem da GPU ficou mais clara conforme o batch aumentou.

Os testes com Ollama aproximam o projeto de um cenário real de IA generativa local. O qwen3.5 conseguiu ser executado localmente, mas apresentou maior tempo total de resposta. O gemma3:4b, por ser menor, apresentou melhor velocidade prática no ambiente testado. Isso reforça a ideia central do artigo: a viabilidade da IA local depende da relação entre hardware disponível, tamanho do modelo, suporte de software e tempo aceitável de uso.

## Observação sobre NPU

Este conjunto de scripts não inventa medição de NPU. A presença de NPU pode ser discutida com base na literatura e nas características do hardware, mas o benchmark direto só deve ser incluído se houver suporte real por framework, driver e biblioteca no ambiente utilizado.
