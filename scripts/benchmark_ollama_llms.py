from pathlib import Path
import csv
import json
import statistics
import time
import urllib.request
import urllib.error


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "dados" / "raw"

URL = "http://localhost:11434/api/generate"

MODELOS = [
    "qwen3.5",
    "gemma3:4b",
]

REPETICOES = 5
AQUECIMENTOS = 1

PROMPTS = [
    {
        "nome": "curto_conceitual",
        "prompt": "Explique em uma frase o que é IA local."
    },
    {
        "nome": "medio_explicativo",
        "prompt": (
            "Explique em um parágrafo curto a diferença entre executar IA localmente "
            "em um computador pessoal e executar IA em nuvem."
        )
    },
    {
        "nome": "tecnico_hardware",
        "prompt": (
            "Explique de forma objetiva como CPU, GPU e NPU podem participar da execução "
            "local de modelos de inteligência artificial."
        )
    }
]


def ns_para_ms(valor):
    if valor is None:
        return 0.0
    return valor / 1_000_000


def tokens_por_segundo(eval_count, eval_duration_ns):
    if not eval_count or not eval_duration_ns:
        return 0.0
    segundos = eval_duration_ns / 1_000_000_000
    if segundos == 0:
        return 0.0
    return eval_count / segundos


def chamar_ollama(modelo, prompt):
    payload = {
        "model": modelo,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 128
        }
    }

    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    inicio_local = time.perf_counter()

    with urllib.request.urlopen(req, timeout=600) as resposta:
        bruto = resposta.read().decode("utf-8")

    fim_local = time.perf_counter()

    resultado = json.loads(bruto)
    resultado["tempo_cliente_ms"] = (fim_local - inicio_local) * 1000

    return resultado


def executar_teste(modelo, nome_prompt, prompt):
    print(f"\nAquecendo modelo {modelo} para prompt: {nome_prompt}")

    for _ in range(AQUECIMENTOS):
        chamar_ollama(modelo, prompt)

    linhas_execucoes = []

    print(f"Rodando {REPETICOES} repetições | modelo={modelo} | prompt={nome_prompt}")

    for repeticao in range(1, REPETICOES + 1):
        resultado = chamar_ollama(modelo, prompt)

        eval_count = resultado.get("eval_count", 0)
        eval_duration = resultado.get("eval_duration", 0)

        linha = {
            "teste": "ollama_llm_generate",
            "modelo": modelo,
            "prompt_nome": nome_prompt,
            "repeticao": repeticao,
            "tempo_cliente_ms": resultado.get("tempo_cliente_ms", 0),
            "total_duration_ms": ns_para_ms(resultado.get("total_duration")),
            "load_duration_ms": ns_para_ms(resultado.get("load_duration")),
            "prompt_eval_count": resultado.get("prompt_eval_count", 0),
            "prompt_eval_duration_ms": ns_para_ms(resultado.get("prompt_eval_duration")),
            "eval_count": eval_count,
            "eval_duration_ms": ns_para_ms(eval_duration),
            "tokens_por_segundo": tokens_por_segundo(eval_count, eval_duration),
            "resposta_tamanho_chars": len(resultado.get("response", "")),
            "status": "ok",
            "observacao": "geracao local via Ollama API"
        }

        linhas_execucoes.append(linha)

        print(
            f"Rep {repeticao}: "
            f"total={linha['total_duration_ms']:.2f} ms | "
            f"tokens/s={linha['tokens_por_segundo']:.2f} | "
            f"tokens={linha['eval_count']}"
        )

    return linhas_execucoes


def resumir(linhas):
    grupos = {}

    for linha in linhas:
        chave = (linha["modelo"], linha["prompt_nome"])
        if chave not in grupos:
            grupos[chave] = []
        grupos[chave].append(linha)

    resumos = []

    for (modelo, prompt_nome), grupo in grupos.items():
        total = [x["total_duration_ms"] for x in grupo]
        cliente = [x["tempo_cliente_ms"] for x in grupo]
        load = [x["load_duration_ms"] for x in grupo]
        prompt_eval = [x["prompt_eval_duration_ms"] for x in grupo]
        eval_ms = [x["eval_duration_ms"] for x in grupo]
        tokens_s = [x["tokens_por_segundo"] for x in grupo]
        eval_count = [x["eval_count"] for x in grupo]

        resumos.append({
            "teste": "ollama_llm_generate",
            "modelo": modelo,
            "prompt_nome": prompt_nome,
            "repeticoes": len(grupo),
            "total_medio_ms": statistics.mean(total),
            "total_desvio_ms": statistics.stdev(total) if len(total) > 1 else 0.0,
            "cliente_medio_ms": statistics.mean(cliente),
            "load_medio_ms": statistics.mean(load),
            "prompt_eval_medio_ms": statistics.mean(prompt_eval),
            "eval_medio_ms": statistics.mean(eval_ms),
            "tokens_s_medio": statistics.mean(tokens_s),
            "tokens_s_desvio": statistics.stdev(tokens_s) if len(tokens_s) > 1 else 0.0,
            "eval_count_medio": statistics.mean(eval_count),
            "status": "ok",
            "observacao": "resumo de geracao local via Ollama API"
        })

    return resumos


def salvar_csv(path, linhas, campos):
    with open(path, "w", newline="", encoding="utf-8") as arquivo:
        writer = csv.DictWriter(arquivo, fieldnames=campos)
        writer.writeheader()
        writer.writerows(linhas)


def main():
    RAW.mkdir(parents=True, exist_ok=True)

    print("Benchmark Ollama com LLMs locais")
    print(f"Modelos: {', '.join(MODELOS)}")
    print(f"Endpoint: {URL}")

    todas_execucoes = []

    try:
        for modelo in MODELOS:
            for item in PROMPTS:
                linhas = executar_teste(modelo, item["nome"], item["prompt"])
                todas_execucoes.extend(linhas)

    except urllib.error.URLError as erro:
        print("\nErro ao acessar Ollama.")
        print("Verifique se o Ollama está instalado e rodando em http://localhost:11434")
        print(f"Detalhe: {erro}")
        return

    resumos = resumir(todas_execucoes)

    execucoes_path = RAW / "benchmark_ollama_llms_execucoes.csv"
    resumo_path = RAW / "benchmark_ollama_llms.csv"

    campos_execucoes = [
        "teste",
        "modelo",
        "prompt_nome",
        "repeticao",
        "tempo_cliente_ms",
        "total_duration_ms",
        "load_duration_ms",
        "prompt_eval_count",
        "prompt_eval_duration_ms",
        "eval_count",
        "eval_duration_ms",
        "tokens_por_segundo",
        "resposta_tamanho_chars",
        "status",
        "observacao"
    ]

    campos_resumo = [
        "teste",
        "modelo",
        "prompt_nome",
        "repeticoes",
        "total_medio_ms",
        "total_desvio_ms",
        "cliente_medio_ms",
        "load_medio_ms",
        "prompt_eval_medio_ms",
        "eval_medio_ms",
        "tokens_s_medio",
        "tokens_s_desvio",
        "eval_count_medio",
        "status",
        "observacao"
    ]

    salvar_csv(execucoes_path, todas_execucoes, campos_execucoes)
    salvar_csv(resumo_path, resumos, campos_resumo)

    print(f"\nResumo salvo em: {resumo_path}")
    print(f"Execuções individuais salvas em: {execucoes_path}")


if __name__ == "__main__":
    main()
