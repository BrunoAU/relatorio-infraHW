# Preflight To Final Status

- Data/hora: `2026-06-16 14:59:17`

## Ambiente

- Python: `3.14.3`
- Executavel: `C:\Users\user\Desktop\ia-local-bench\ia-local-bench\.venv\Scripts\python.exe`

## PyTorch final

- Versao final do PyTorch: `2.12.0+cu126`
- CUDA disponivel: `sim`
- CUDA version no PyTorch: `12.6`
- GPU detectada pelo PyTorch: `NVIDIA GeForce MX570 A`
- Teste curto CUDA: `ok` com `cuda_test_ok torch.Size([512, 512])`

## NVIDIA

- GPU detectada pelo `nvidia-smi`: `NVIDIA GeForce MX570 A`
- Driver: `610.47`
- VRAM: `2048 MiB`
- Temperatura observada: `57C`
- Potencia observada: `8W / 30W`
- Uso atual observado: `0%`

## Ollama

- Status do Ollama: `nao disponivel`
- Resultado de `ollama list`: comando nao encontrado no PATH
- Modelos Ollama disponiveis: `nao foi possivel listar`
- Modelos que serao testados: nenhum localmente confirmado; o pipeline deve registrar `skip` para os testes LLM se o runtime continuar indisponivel

## Configuracao final dos benchmarks

```json
{
  "seed": 42,
  "repetitions": 10,
  "warmups": 2,
  "matrix_sizes": [512, 1024, 1536, 2048],
  "gpu_limit_matrix_sizes": [2560, 3072, 4096],
  "batches": [1, 8, 16, 32],
  "vision_models": ["mobilenet_v2", "resnet18"],
  "vision_input": {
    "channels": 3,
    "height": 224,
    "width": 224
  },
  "ollama": {
    "url": "http://localhost:11434",
    "models": [
      "gemma3:1b",
      "gemma3:4b",
      "llama3.2:1b",
      "llama3.2:3b",
      "qwen2.5-coder",
      "qwen3.5"
    ],
    "temperature": 0,
    "num_predict": 128,
    "timeout_s": 180,
    "repetitions": 3,
    "warmups": 1,
    "prompt_categories": [
      "explicacao_conceitual_curta",
      "explicacao_tecnica_media",
      "resumo_texto",
      "geracao_codigo_simples",
      "analise_codigo",
      "raciocinio_logico_curto",
      "hardware_ia_local",
      "reescrita_objetiva",
      "tarefa_longa_uso_real"
    ]
  },
  "hardware_sampling_interval_s": 1.0,
  "heavy_tests_enabled": true,
  "run_ollama": true,
  "run_gpu_limit": true,
  "pretrained_vision": false
}
```

## Decisao

- Pronto para coleta final: `sim`

## Problemas pendentes

- `ollama` nao esta instalado ou nao esta acessivel no PATH neste ambiente.
- Os testes de LLM so serao validos como `skip` enquanto o runtime local do Ollama continuar indisponivel.
- NPU continua fora do escopo experimental direto e deve permanecer documentada como nao medida.
