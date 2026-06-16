# IA Local Benchmarks

Este repositorio sustenta o estudo "Hardware para IA Local: NPUs, GPUs e o Futuro dos Computadores Pessoais" com foco em reproducibilidade experimental. O escopo experimental real cobre CPU e GPU CUDA quando disponiveis. NPU aparece apenas como discussao tecnica e bibliografica: ela nao e medida automaticamente e nao deve ser apresentada como comparacao experimental neste ambiente.

## Escopo experimental real

- CPU com NumPy e PyTorch.
- GPU CUDA com PyTorch, quando `torch.cuda.is_available()` for verdadeiro.
- CNN sintetica e inferencia com arquiteturas reais (`mobilenet_v2` e `resnet18`) usando entrada sintetica.
- LLMs locais via Ollama, avaliando tempo, responsividade e uso de CPU, RAM, GPU e VRAM quando essas metricas estiverem disponiveis.
- NPU nao e benchmarkada experimentalmente neste projeto.

## Estrutura

```text
ia-local-bench/
|-- README.md
|-- VALIDADE.md
|-- REPRODUCAO.md
|-- ARTIGO_MELHORIAS_SUGERIDAS.md
|-- requirements.txt
|-- scripts/
|   |-- benchmark_config.json
|   |-- common.py
|   |-- coletar_sistema.py
|   |-- benchmark_numpy_cpu.py
|   |-- benchmark_torch_cpu_gpu.py
|   |-- benchmark_limite_gpu.py
|   |-- benchmark_cnn_sintetico.py
|   |-- benchmark_modelos_reais.py
|   |-- benchmark_ollama_llms.py
|   |-- analisar_resultados.py
|   `-- gerar_graficos.py
|-- dados/
|   |-- raw/
|   `-- processed/
`-- resultados/
    |-- graficos/
    |-- tabelas/
    `-- logs/
```

## Instalacao

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

No Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`torch` e `torchvision` devem ser instalados conforme o seletor oficial do PyTorch, porque a variante CUDA depende do driver NVIDIA e da versao CUDA compativel no sistema. Se a instalacao local for apenas CPU, o pipeline continua funcionando e os testes CUDA ficam como `skip`.

## Configuracao CUDA

- Verifique `python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"`.
- Opcionalmente confirme a GPU com `nvidia-smi`.
- Se `torch.cuda.is_available()` for falso, o projeto nao falha: apenas registra ausencia de CUDA.

## Uso do Ollama

O benchmark de LLMs mede desempenho e uso de recursos, nao qualidade profunda das respostas.

- Verifique se o Ollama esta ativo em `http://localhost:11434`.
- Ajuste modelos, repeticoes, prompts e timeout em `scripts/benchmark_config.json`.
- O script nao baixa modelos automaticamente.
- Se um modelo nao estiver instalado, o resultado sera `skip` com sugestao `ollama pull <modelo>`.

Exemplos de download manual:

```bash
ollama pull gemma3:1b
ollama pull gemma3:4b
ollama pull qwen2.5:3b
ollama pull qwen2.5-coder
ollama pull llama3.2:1b
ollama pull llama3.2:3b
```

## Execucao

Pipeline completo:

```bash
python scripts/rodar_tudo.py
```

Etapas individuais:

```bash
python scripts/coletar_sistema.py
python scripts/benchmark_numpy_cpu.py
python scripts/benchmark_torch_cpu_gpu.py
python scripts/benchmark_limite_gpu.py
python scripts/benchmark_cnn_sintetico.py
python scripts/benchmark_modelos_reais.py
python scripts/benchmark_ollama_llms.py
python scripts/analisar_resultados.py
python scripts/gerar_graficos.py
```

Para usar pesos pretrained nos modelos de visao:

```bash
python scripts/benchmark_modelos_reais.py --pretrained
```

Isso pode exigir download de pesos na primeira execucao. Sem pesos pretrained, o benchmark mede apenas desempenho da arquitetura com entrada sintetica e nao deve sustentar conclusoes sobre acuracia ou qualidade do modelo.

## Interpretacao dos resultados

- `tempo_medio_ms`, `tempo_mediana_ms`, `tempo_p95_ms`: latencia observada por teste.
- `throughput_itens_s`: vazao estimada para casos com batch.
- `speedup_cuda_vs_cpu`: comparacao direta apenas quando existe par equivalente CPU/CUDA.
- `gpu_peak_mb`: pico de VRAM reportado pelo PyTorch, quando disponivel.
- `status`:
  - `ok`: execucao concluida.
  - `skip`: teste nao executado por indisponibilidade ou configuracao.
  - `erro`: houve falha, mas o pipeline seguiu.

Nos testes Ollama:

- `tokens_por_segundo` aproxima responsividade durante a geracao.
- `tempo_cliente_ms` mede o tempo observado pelo cliente.
- `cpu_media`, `gpu_media`, `ram_processo_mb_pico`, `vram_mb_pico` mostram custo de hardware durante a geracao.
- Um modelo executavel localmente pode continuar impratico para uso interativo.
- Modelos menores tendem a responder mais rapido, mas isso nao implica melhor qualidade semantica.
- Modelos maiores podem produzir respostas potencialmente melhores, mas isso nao foi avaliado profundamente neste repositorio.

## Arquivos gerados

- `dados/raw/system_info.json`
- `dados/raw/benchmark_*.csv`
- `dados/raw/ollama_benchmark_resumo.csv`
- `dados/raw/ollama_hardware_trace.csv`
- `dados/processed/resultados_consolidados.csv`
- `resultados/tabelas/*.csv`
- `resultados/graficos/*.png`
- `resultados/logs/system_info_<timestamp>.json`
- `resultados/logs/run_manifest.json`

## Limitacoes metodologicas

- O ambiente observado mede CPU e GPU CUDA; NPU nao e medida automaticamente.
- O projeto nao inventa metricas: quando algo nao estiver disponivel, o artefato registra `not_available`, `skip` ou `erro`.
- Temperatura, potencia e uso de GPU dependem de `nvidia-smi`.
- Sem pesos pretrained, os testes de `mobilenet_v2` e `resnet18` nao medem qualidade do modelo.
- Os benchmarks Ollama medem desempenho e uso de recursos, nao qualidade profunda de geracao.

## Checklist de reproducibilidade

- Mesmo Python e mesma pilha de dependencias instalados.
- `scripts/benchmark_config.json` versionado junto com os artefatos.
- `system_info.json` gerado no mesmo ambiente da coleta.
- Estado do Ollama e modelos instalados documentados.
- `run_manifest.json` salvo com status de cada etapa.
- Resultados `skip` ou `erro` preservados em vez de omitidos.
