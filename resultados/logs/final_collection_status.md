# Final Collection Status

Data: 2026-06-16

## Ambiente validado

- CUDA disponivel no PyTorch: `True`
- PyTorch: `2.12.0+cu126`
- CUDA do PyTorch: `12.6`
- GPU detectada: `NVIDIA GeForce MX570 A` com `2.0 GB` de VRAM
- Ollama local respondendo em `http://localhost:11434`

## Modelos Ollama validados

Modelos presentes e benchmarkados:

- `gemma3:1b`
- `llama3.2:1b`
- `llama3.2:3b`
- `gemma3:4b`
- `llama3:8b`
- `qwen3.5:latest`
- `qwen2.5-coder:14b`

Testes manuais confirmados antes da coleta longa:

- `gemma3:1b`
- `llama3.2:1b`
- `gemma3:4b`

## Resultado do benchmark Ollama

- Benchmark real concluido com sucesso em `dados/raw` e refletido em `resultados/tabelas`
- Nenhum modelo ficou com `skip`
- Nenhum modelo ficou com taxa de falha acima de `0.0`
- Classificacao de viabilidade:
  - `viavel com espera`: `gemma3:1b`, `llama3.2:1b`, `llama3.2:3b`
  - `pesado para o hardware testado`: `gemma3:4b`, `llama3:8b`, `qwen3.5:latest`, `qwen2.5-coder:14b`

## Pipeline executada

Executado com sucesso:

- `python scripts/benchmark_ollama_llms.py`
- `python scripts/analisar_resultados.py`
- `python scripts/gerar_graficos.py`
- `python scripts/rodar_tudo.py`

Observacao importante:

- O `rodar_tudo.py` foi ajustado para reutilizar os CSVs validos do Ollama ja existentes em `dados/raw`, evitando rerodar a coleta longa de LLMs e preservando os resultados confirmados.

## Arquivos finais validados

- `resultados/logs/run_manifest.json`
- `resultados/tabelas/ollama_modelos_resumo.csv`
- `resultados/tabelas/ollama_por_tarefa.csv`
- `resultados/tabelas/ollama_tokens_por_segundo.csv`
- `resultados/tabelas/ollama_uso_hardware.csv`
- `resultados/tabelas/ollama_viabilidade_pratica.csv`
- `resultados/tabelas/tabela_resultados_artigo.csv`
- `resultados/graficos/tempo_ollama_llms_total.png`
- `resultados/graficos/tokens_ollama_llms.png`

## Problemas e observacoes

- O bloqueio inicial do sandbox impediu acesso ao `localhost` e gerou um `skip` antigo do Ollama; isso foi corrigido com uma execucao real fora do sandbox.
- A etapa final registrada em `run_manifest.json` esta limpa, com todas as etapas em status `ok` e `erros_ou_skips` vazio.
- A limitacao experimental restante e apenas a ausencia de medicao de NPU, mantida explicitamente em `resultados/tabelas/limitacoes_experimentais.csv`.

## Prontidao

Status final: pronto para revisao do artigo.
