# IA Local Benchmarks — Hardware para IA local

Este repositório contém scripts simples para coletar dados experimentais do artigo **Hardware para IA local: NPUs, GPUs e o futuro dos computadores pessoais**.

O objetivo é gerar resultados rastreáveis para comparar execução local em CPU e, quando disponível, GPU. A NPU fica registrada como análise condicionada ao suporte real do hardware, drivers e frameworks disponíveis.

## Estrutura

```text
ia-local-bench/
├── README.md
├── requirements.txt
├── scripts/
│   ├── coletar_sistema.py
│   ├── benchmark_numpy_cpu.py
│   ├── benchmark_torch_cpu_gpu.py
│   ├── gerar_graficos.py
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

Para GPU NVIDIA com CUDA, instale o PyTorch adequado ao seu ambiente conforme o seletor oficial do PyTorch.

## Como rodar

```bash
python scripts/rodar_tudo.py
```

Ou rode cada etapa manualmente:

```bash
python scripts/coletar_sistema.py
python scripts/benchmark_numpy_cpu.py
python scripts/benchmark_torch_cpu_gpu.py
python scripts/gerar_graficos.py
```

## Saídas esperadas

- `dados/raw/system_info.json`: informações do ambiente.
- `dados/raw/benchmark_numpy_cpu.csv`: benchmark CPU com NumPy.
- `dados/raw/benchmark_torch_cpu_gpu.csv`: benchmark CPU/GPU com PyTorch, se disponível.
- `dados/processed/resultados_consolidados.csv`: dados consolidados.
- `resultados/graficos/*.png`: gráficos para o artigo.

## Observação sobre NPU

Este conjunto de scripts não inventa medição de NPU. A presença de NPU será documentada na coleta do sistema quando possível, mas o benchmark direto só deve ser incluído se houver suporte real por framework/driver no ambiente usado.
