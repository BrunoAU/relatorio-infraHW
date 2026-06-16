# Preflight Check

- Data/hora da preparacao: `2026-06-16 14:46:55`
- Repositorio: `C:\Users\user\Desktop\ia-local-bench\ia-local-bench`

## Status do ambiente Python

- Ambiente Python: `ok`
- `.venv` encontrada: `sim`
- Python da `.venv`: `3.14.3`
- Executavel: `C:\Users\user\Desktop\ia-local-bench\ia-local-bench\.venv\Scripts\python.exe`
- `pip`: `25.3`
- Upgrade de `pip`: `nao atualizado nesta etapa por bloqueio de rede/permissao; versao atual ja esta funcional`

## Status das dependencias

- `requirements.txt`: encontrado
- Instalacao de dependencias: `ok`
- Pacotes principais presentes:
  - `numpy 2.4.6`
  - `pandas 3.0.3`
  - `psutil 7.2.2`
  - `matplotlib 3.11.0`
  - `torch 2.12.0`
  - `torchvision 0.27.0`
  - `requests 2.34.2`

## Status do PyTorch

- PyTorch importado: `sim`
- Versao: `2.12.0+cpu`
- CUDA no PyTorch: `False`
- `torch.version.cuda`: `None`
- GPU visivel pelo PyTorch: `CUDA indisponivel`

## Status do CUDA

- CUDA disponivel para o PyTorch nesta instalacao: `nao`
- Observacao: a GPU NVIDIA existe no sistema, mas a variante CUDA do PyTorch nao esta instalada nesta `.venv`
- Acao recomendada: instalar a versao CUDA do PyTorch conforme o seletor oficial do PyTorch, sem escolher uma build aleatoria

## Status da GPU/NVIDIA

- `nvidia-smi`: `ok`
- GPU detectada: `NVIDIA GeForce MX570 A`
- Driver detectado: `610.47`
- VRAM detectada: `2048 MiB`
- Temperatura disponivel: `sim` (`54C`)
- Potencia disponivel: `sim` (`9W / 30W`)
- Uso de GPU visivel: `sim` (`0%` no momento da checagem)

## Status do Ollama

- Comando `ollama`: `indisponivel`
- Resultado: o Ollama precisa estar instalado e acessivel no PATH, e o runtime precisa estar rodando antes dos testes de LLM

## Modelos Ollama configurados

- `qwen3.5`
- `gemma3:4b`
- `gemma3:1b`
- `qwen2.5:3b`
- `qwen2.5-coder`
- `llama3.2:1b`
- `llama3.2:3b`

## Modelos Ollama disponiveis

- Nao foi possivel listar, porque `ollama` nao esta instalado ou nao esta acessivel neste ambiente

## Modelos Ollama ausentes ou nao verificados

- `qwen3.5`
- `gemma3:4b`
- `gemma3:1b`
- `qwen2.5:3b`
- `qwen2.5-coder`
- `llama3.2:1b`
- `llama3.2:3b`

## Sugestoes de comandos para modelos ausentes

```powershell
ollama pull qwen3.5
ollama pull gemma3:1b
ollama pull gemma3:4b
ollama pull qwen2.5:3b
ollama pull qwen2.5-coder
ollama pull llama3.2:1b
ollama pull llama3.2:3b
```

## Arquivos de configuracao encontrados

- `scripts/common.py`
- `scripts/benchmark_config.json`
- `requirements.txt`
- `README.md`
- `dados/`
- `resultados/`

## Diretorios de saida

- `dados/raw`: `ok`
- `dados/processed`: `ok`
- `resultados/graficos`: `ok`
- `resultados/tabelas`: `ok`
- `resultados/logs`: `ok`

## Validacao da configuracao dos benchmarks

- `repetitions`: `3`
- `warmups`: `1`
- `matrix_sizes`: `[512, 1024]`
- `batches`: `[1, 8]`
- `vision_models`: `["mobilenet_v2", "resnet18"]`
- `ollama.models`: `["qwen3.5", "gemma3:4b", "gemma3:1b", "qwen2.5:3b", "qwen2.5-coder", "llama3.2:1b", "llama3.2:3b"]`
- `ollama.timeout_s`: `180`
- `hardware_sampling_interval_s`: `1.0`
- `run_ollama`: `true`
- `run_gpu_limit`: `true`
- `pretrained_vision`: `false`
- Coerencia geral: `ok`
- Observacao: a configuracao atual esta leve e apropriada para preflight; nada foi aumentado para teste pesado

## Scripts compilados com sucesso

- `scripts/common.py`
- `scripts/coletar_sistema.py`
- `scripts/benchmark_numpy_cpu.py`
- `scripts/benchmark_torch_cpu_gpu.py`
- `scripts/benchmark_limite_gpu.py`
- `scripts/benchmark_cnn_sintetico.py`
- `scripts/benchmark_modelos_reais.py`
- `scripts/benchmark_ollama_llms.py`
- `scripts/analisar_resultados.py`
- `scripts/gerar_graficos.py`
- `scripts/rodar_tudo.py`

## Validacoes rapidas executadas

- `python scripts\coletar_sistema.py`: `ok`
- Arquivo gerado: `dados/raw/system_info.json`
- Arquivo gerado: `resultados/logs/system_info_20260616_144655.json`

## Problemas encontrados

- O upgrade online de `pip` nao conseguiu consultar o indice remoto por bloqueio de rede/permissao. O ambiente continua funcional com `pip 25.3`.
- O PyTorch instalado na `.venv` e `cpu-only`, portanto `torch.cuda.is_available()` retorna `False`.
- O `ollama` nao esta instalado ou nao esta no `PATH`, entao os testes de LLM ainda nao estao prontos para execucao final.

## Proximos passos para rodar os testes finais

1. Instalar a variante CUDA do PyTorch conforme o seletor oficial do PyTorch, se o objetivo for medir GPU pelo PyTorch.
2. Instalar/iniciar o Ollama e garantir que `ollama list` funcione.
3. Baixar apenas os modelos realmente desejados com `ollama pull ...`.
4. Ativar a `.venv`.
5. Rodar o pipeline final apenas quando CUDA/Ollama estiverem no estado desejado.

## Comandos exatos para depois

Ativar o ambiente:

```powershell
.\.venv\Scripts\Activate.ps1
```

Revalidar PyTorch/CUDA:

```powershell
python -c "import torch; print('torch:', torch.__version__); print('cuda_available:', torch.cuda.is_available()); print('cuda_version:', torch.version.cuda); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA indisponivel')"
```

Verificar NVIDIA:

```powershell
nvidia-smi
```

Verificar Ollama:

```powershell
ollama list
```

Rodar a coleta final de benchmarks:

```powershell
python scripts\rodar_tudo.py
```
