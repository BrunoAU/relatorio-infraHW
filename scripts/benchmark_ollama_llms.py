import json
import time

from bootstrap import ensure_local_deps

ensure_local_deps()

import requests
from common import (
    RAW,
    HardwareMonitor,
    config_get,
    ensure_dirs,
    infer_compute_path,
    load_config,
    now_iso,
    set_seeds,
    stats,
    summarize_hardware,
    write_csv,
)

PROMPTS = {
    "explicacao_conceitual_curta": "Explique em duas frases o que e IA local.",
    "explicacao_tecnica_media": "Explique de forma objetiva como CPU, GPU e NPU podem participar da execucao local de modelos de IA e por que isso depende do runtime.",
    "resumo_texto": "Resuma em quatro frases: executar modelos localmente aumenta privacidade e controle, mas o desempenho depende do hardware, da quantizacao e do suporte de aceleracao.",
    "geracao_codigo_simples": "Escreva uma funcao Python que retorne os primeiros n numeros pares.",
    "analise_codigo": "Analise este trecho: for i in range(len(xs)): total += xs[i] / len(xs). Aponte um risco e uma melhoria simples.",
    "raciocinio_logico_curto": "Se todo benchmark reproduzivel precisa de ambiente descrito e este benchmark descreve o ambiente, o que isso sugere sobre a reproducibilidade? Responda em duas frases.",
    "hardware_ia_local": "Quais fatores mais afetam a viabilidade pratica de um LLM local em um notebook com pouca VRAM?",
    "reescrita_objetiva": "Reescreva de forma mais objetiva: rodar localmente nao garante boa experiencia para o usuario.",
    "tarefa_longa_uso_real": "Imagine que um estudante quer comparar CPU, GPU CUDA e NPU em computadores pessoais para IA local. Estruture uma resposta curta com contexto, limites metodologicos e uma recomendacao pratica de como interpretar benchmarks.",
}


def ns_to_ms(value):
    if value is None:
        return 0.0
    return float(value) / 1_000_000


def tokens_per_second(eval_count, eval_duration_ns):
    try:
        if not eval_count or not eval_duration_ns:
            return 0.0
        seconds = float(eval_duration_ns) / 1_000_000_000
        return round(float(eval_count) / seconds, 4) if seconds > 0 else 0.0
    except Exception:
        return 0.0


def ollama_get(base_url, endpoint, timeout_s):
    return requests.get(f"{base_url}{endpoint}", timeout=timeout_s)


def ollama_post(base_url, endpoint, payload, timeout_s):
    return requests.post(f"{base_url}{endpoint}", json=payload, timeout=timeout_s)


def available_models(base_url, timeout_s):
    response = ollama_get(base_url, "/api/tags", timeout_s)
    response.raise_for_status()
    data = response.json()
    models = data.get("models", [])
    return {item.get("name"): item for item in models if item.get("name")}


def run_once(base_url, model, prompt, category, repetition, temperature, num_predict, timeout_s, sampling_interval_s):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    with HardwareMonitor(interval_s=sampling_interval_s, context={"modelo": model, "categoria_tarefa": category, "repeticao": repetition}) as monitor:
        start = time.perf_counter()
        response = ollama_post(base_url, "/api/generate", payload, timeout_s)
        elapsed = time.perf_counter() - start
    response.raise_for_status()
    result = response.json()
    hardware = summarize_hardware(monitor.samples)
    row = {
        "timestamp": now_iso(),
        "teste": "ollama_llm_generate",
        "modelo": model,
        "categoria_tarefa": category,
        "prompt": prompt,
        "repeticao": repetition,
        "status": "ok",
        "tempo_total_s": round(elapsed, 4),
        "tempo_cliente_ms": round(elapsed * 1000, 4),
        "total_duration_ms": round(ns_to_ms(result.get("total_duration")), 4),
        "load_duration_ms": round(ns_to_ms(result.get("load_duration")), 4),
        "prompt_eval_duration_ms": round(ns_to_ms(result.get("prompt_eval_duration")), 4),
        "eval_duration_ms": round(ns_to_ms(result.get("eval_duration")), 4),
        "tokens_entrada": result.get("prompt_eval_count", 0),
        "tokens_gerados": result.get("eval_count", 0),
        "tokens_por_segundo": tokens_per_second(result.get("eval_count"), result.get("eval_duration")),
        "resposta_tamanho_chars": len(result.get("response", "")),
        "erro_ou_skip": "",
        **hardware,
    }
    row["caminho_execucao_observado"] = infer_compute_path(hardware)
    traces = []
    for sample in monitor.samples:
        traces.append(
            {
                "tempo_relativo_s": sample.get("tempo_relativo_s"),
                "modelo": model,
                "categoria_tarefa": category,
                "repeticao": repetition,
                "cpu_percent": sample.get("cpu_percent", "not_available"),
                "ram_process_mb": sample.get("ram_process_mb", "not_available"),
                "ram_system_used_mb": sample.get("ram_system_used_mb", "not_available"),
                "gpu_percent": sample.get("gpu_percent", "not_available"),
                "vram_used_mb": sample.get("vram_used_mb", "not_available"),
                "gpu_temperature_c": sample.get("gpu_temperature_c", "not_available"),
                "gpu_power_w": sample.get("gpu_power_w", "not_available"),
            }
        )
    return row, traces


def viability(row):
    if row.get("status") != "ok":
        return "nao executado / skip"
    errors = row.get("taxa_falha", 0)
    tps = row.get("tokens_por_segundo_media", 0)
    tempo = row.get("tempo_total_ms_media", 0)
    ram = row.get("ram_processo_mb_pico_medio", 0)
    try:
        errors = float(errors)
        tps = float(tps)
        tempo = float(tempo)
        ram = float(ram)
    except Exception:
        return "nao executado / skip"
    if errors > 0.3:
        return "pesado para o hardware testado"
    if tps >= 15 and tempo <= 4000 and ram <= 6000:
        return "viavel para uso interativo"
    if tps >= 4 and tempo <= 15000:
        return "viavel com espera"
    return "pesado para o hardware testado"


def aggregate_execution(rows):
    groups = {}
    for row in rows:
        key = (row["modelo"], row["categoria_tarefa"])
        groups.setdefault(key, []).append(row)

    per_task = []
    model_summary = {}
    for (model, category), group in groups.items():
        ok_rows = [item for item in group if item["status"] == "ok"]
        fail_rate = 1 - (len(ok_rows) / len(group) if group else 0)
        total_ms = [item["tempo_cliente_ms"] for item in ok_rows]
        tps = [item["tokens_por_segundo"] for item in ok_rows]
        row = {
            "modelo": model,
            "categoria_tarefa": category,
            "execucoes": len(group),
            "status": "ok" if ok_rows else "skip",
            "tempo_total_ms_media": stats(total_ms)["media"] if ok_rows else "not_available",
            "tempo_total_ms_p95": stats(total_ms)["p95"] if ok_rows else "not_available",
            "tokens_por_segundo_media": stats(tps)["media"] if ok_rows else "not_available",
            "tokens_gerados_media": stats([item["tokens_gerados"] for item in ok_rows])["media"] if ok_rows else "not_available",
            "taxa_falha": round(fail_rate, 4),
            "ram_processo_mb_pico_medio": stats([item["ram_processo_mb_pico"] for item in ok_rows if isinstance(item.get("ram_processo_mb_pico"), (int, float))], 2)["media"] if ok_rows else "not_available",
            "gpu_media_media": stats([item["gpu_media"] for item in ok_rows if isinstance(item.get("gpu_media"), (int, float))], 2)["media"] if ok_rows else "not_available",
            "vram_mb_pico_medio": stats([item["vram_mb_pico"] for item in ok_rows if isinstance(item.get("vram_mb_pico"), (int, float))], 2)["media"] if ok_rows else "not_available",
        }
        per_task.append(row)
        model_summary.setdefault(model, []).append(row)

    models = []
    tokens = []
    hardware = []
    viability_rows = []
    for model, rows in model_summary.items():
        ok_rows = [item for item in rows if item["status"] == "ok"]
        summary = {
            "modelo": model,
            "tarefas": len(rows),
            "status": "ok" if ok_rows else "skip",
            "tempo_total_ms_media": stats([item["tempo_total_ms_media"] for item in ok_rows if isinstance(item.get("tempo_total_ms_media"), (int, float))])["media"] if ok_rows else "not_available",
            "tokens_por_segundo_media": stats([item["tokens_por_segundo_media"] for item in ok_rows if isinstance(item.get("tokens_por_segundo_media"), (int, float))])["media"] if ok_rows else "not_available",
            "taxa_falha": stats([item["taxa_falha"] for item in rows if isinstance(item.get("taxa_falha"), (int, float))], 4)["media"] if rows else "not_available",
            "ram_processo_mb_pico_medio": stats([item["ram_processo_mb_pico_medio"] for item in ok_rows if isinstance(item.get("ram_processo_mb_pico_medio"), (int, float))], 2)["media"] if ok_rows else "not_available",
            "gpu_media_media": stats([item["gpu_media_media"] for item in ok_rows if isinstance(item.get("gpu_media_media"), (int, float))], 2)["media"] if ok_rows else "not_available",
            "vram_mb_pico_medio": stats([item["vram_mb_pico_medio"] for item in ok_rows if isinstance(item.get("vram_mb_pico_medio"), (int, float))], 2)["media"] if ok_rows else "not_available",
        }
        summary["classificacao_viabilidade"] = viability(summary)
        models.append(summary)
        viability_rows.append({"modelo": model, "classificacao_viabilidade": summary["classificacao_viabilidade"], **summary})
        if ok_rows:
            tokens.append({"modelo": model, "tokens_por_segundo_media": summary["tokens_por_segundo_media"]})
            related_exec = [r for r in rows if r["modelo"] == model and r.get("status") == "ok"]
            hardware.append(
                {
                    "modelo": model,
                    "cpu_media": stats([r["cpu_media"] for r in related_exec if isinstance(r.get("cpu_media"), (int, float))], 2)["media"],
                    "ram_processo_mb_pico": summary["ram_processo_mb_pico_medio"],
                    "gpu_media": summary["gpu_media_media"],
                    "vram_mb_pico": summary["vram_mb_pico_medio"],
                }
            )
    return models, per_task, tokens, hardware, viability_rows


def main():
    ensure_dirs()
    config = load_config()
    set_seeds(int(config.get("seed", 42)))
    if not config.get("run_ollama", True):
        row = {
            "timestamp": now_iso(),
            "teste": "ollama_llm_generate",
            "modelo": "config_disabled",
            "categoria_tarefa": "config_disabled",
            "prompt": "",
            "repeticao": 0,
            "status": "skip",
            "tempo_total_s": "",
            "tempo_cliente_ms": "",
            "total_duration_ms": "",
            "load_duration_ms": "",
            "prompt_eval_duration_ms": "",
            "eval_duration_ms": "",
            "tokens_entrada": "",
            "tokens_gerados": "",
            "tokens_por_segundo": "",
            "resposta_tamanho_chars": "",
            "erro_ou_skip": "Benchmark Ollama desativado no benchmark_config.json.",
        }
        write_csv(RAW / "benchmark_ollama_llms_execucoes.csv", [row])
        write_csv(RAW / "benchmark_ollama_llms.csv", [row])
        write_csv(RAW / "ollama_benchmark_resumo.csv", [row])
        write_csv(RAW / "ollama_hardware_trace.csv", [])
        return

    base_url = config_get(config, "ollama.url", "http://localhost:11434").rstrip("/")
    timeout_s = int(config_get(config, "ollama.timeout_s", 180))
    repetitions = int(config_get(config, "ollama.repetitions", 1))
    warmups = int(config_get(config, "ollama.warmups", 0))
    models = config_get(config, "ollama.models", ["qwen3.5", "gemma3:4b"])
    categories = config_get(config, "ollama.prompt_categories", list(PROMPTS.keys()))
    temperature = float(config_get(config, "ollama.temperature", 0))
    num_predict = int(config_get(config, "ollama.num_predict", 128))
    sampling_interval_s = float(config.get("hardware_sampling_interval_s", 1.0))

    execution_rows = []
    trace_rows = []
    try:
        installed = available_models(base_url, timeout_s)
    except requests.RequestException as exc:
        skip_row = {
            "timestamp": now_iso(),
            "teste": "ollama_llm_generate",
            "modelo": "ollama_unavailable",
            "categoria_tarefa": "ollama_unavailable",
            "prompt": "",
            "repeticao": 0,
            "status": "skip",
            "tempo_total_s": "",
            "tempo_cliente_ms": "",
            "total_duration_ms": "",
            "load_duration_ms": "",
            "prompt_eval_duration_ms": "",
            "eval_duration_ms": "",
            "tokens_entrada": "",
            "tokens_gerados": "",
            "tokens_por_segundo": "",
            "resposta_tamanho_chars": "",
            "erro_ou_skip": f"Ollama indisponivel: {str(exc)[:200]}",
        }
        write_csv(RAW / "benchmark_ollama_llms_execucoes.csv", [skip_row])
        write_csv(RAW / "benchmark_ollama_llms.csv", [skip_row])
        write_csv(RAW / "ollama_benchmark_resumo.csv", [skip_row])
        write_csv(RAW / "ollama_hardware_trace.csv", [])
        write_csv(RAW / "ollama_modelos_resumo.csv", [skip_row])
        write_csv(RAW / "ollama_por_tarefa.csv", [skip_row])
        write_csv(RAW / "ollama_tokens_por_segundo.csv", [skip_row])
        write_csv(RAW / "ollama_uso_hardware.csv", [skip_row])
        write_csv(RAW / "ollama_viabilidade_pratica.csv", [skip_row])
        return

    for model in models:
        if model not in installed:
            for category in categories:
                execution_rows.append(
                    {
                        "timestamp": now_iso(),
                        "teste": "ollama_llm_generate",
                        "modelo": model,
                        "categoria_tarefa": category,
                        "prompt": PROMPTS.get(category, ""),
                        "repeticao": 0,
                        "status": "skip",
                        "tempo_total_s": "",
                        "tempo_cliente_ms": "",
                        "total_duration_ms": "",
                        "load_duration_ms": "",
                        "prompt_eval_duration_ms": "",
                        "eval_duration_ms": "",
                        "tokens_entrada": "",
                        "tokens_gerados": "",
                        "tokens_por_segundo": "",
                        "resposta_tamanho_chars": "",
                        "erro_ou_skip": f"Modelo nao instalado. Sugestao: ollama pull {model}",
                    }
                )
            continue
        for category in categories:
            prompt = PROMPTS.get(category)
            if not prompt:
                execution_rows.append(
                    {
                        "timestamp": now_iso(),
                        "teste": "ollama_llm_generate",
                        "modelo": model,
                        "categoria_tarefa": category,
                        "prompt": "",
                        "repeticao": 0,
                        "status": "skip",
                        "tempo_total_s": "",
                        "tempo_cliente_ms": "",
                        "total_duration_ms": "",
                        "load_duration_ms": "",
                        "prompt_eval_duration_ms": "",
                        "eval_duration_ms": "",
                        "tokens_entrada": "",
                        "tokens_gerados": "",
                        "tokens_por_segundo": "",
                        "resposta_tamanho_chars": "",
                        "erro_ou_skip": "Categoria de prompt nao mapeada no script.",
                    }
                )
                continue
            for _ in range(warmups):
                try:
                    ollama_post(
                        base_url,
                        "/api/generate",
                        {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": temperature, "num_predict": num_predict}},
                        timeout_s,
                    )
                except requests.RequestException:
                    break
            for repetition in range(1, repetitions + 1):
                print(f"Ollama {model} {category} repeticao {repetition}")
                try:
                    row, traces = run_once(base_url, model, prompt, category, repetition, temperature, num_predict, timeout_s, sampling_interval_s)
                    execution_rows.append(row)
                    trace_rows.extend(traces)
                except requests.Timeout:
                    execution_rows.append(
                        {
                            "timestamp": now_iso(),
                            "teste": "ollama_llm_generate",
                            "modelo": model,
                            "categoria_tarefa": category,
                            "prompt": prompt,
                            "repeticao": repetition,
                            "status": "erro",
                            "tempo_total_s": "",
                            "tempo_cliente_ms": "",
                            "total_duration_ms": "",
                            "load_duration_ms": "",
                            "prompt_eval_duration_ms": "",
                            "eval_duration_ms": "",
                            "tokens_entrada": "",
                            "tokens_gerados": "",
                            "tokens_por_segundo": "",
                            "resposta_tamanho_chars": "",
                            "erro_ou_skip": "timeout",
                        }
                    )
                except requests.RequestException as exc:
                    execution_rows.append(
                        {
                            "timestamp": now_iso(),
                            "teste": "ollama_llm_generate",
                            "modelo": model,
                            "categoria_tarefa": category,
                            "prompt": prompt,
                            "repeticao": repetition,
                            "status": "erro",
                            "tempo_total_s": "",
                            "tempo_cliente_ms": "",
                            "total_duration_ms": "",
                            "load_duration_ms": "",
                            "prompt_eval_duration_ms": "",
                            "eval_duration_ms": "",
                            "tokens_entrada": "",
                            "tokens_gerados": "",
                            "tokens_por_segundo": "",
                            "resposta_tamanho_chars": "",
                            "erro_ou_skip": str(exc)[:200],
                        }
                    )

    model_rows, task_rows, token_rows, hardware_rows, viability_rows = aggregate_execution(execution_rows)
    write_csv(RAW / "benchmark_ollama_llms_execucoes.csv", execution_rows)
    write_csv(RAW / "benchmark_ollama_llms.csv", task_rows)
    write_csv(RAW / "ollama_benchmark_resumo.csv", execution_rows)
    write_csv(RAW / "ollama_hardware_trace.csv", trace_rows)
    write_csv(RAW / "ollama_modelos_resumo.csv", model_rows)
    write_csv(RAW / "ollama_por_tarefa.csv", task_rows)
    write_csv(RAW / "ollama_tokens_por_segundo.csv", token_rows)
    write_csv(RAW / "ollama_uso_hardware.csv", hardware_rows)
    write_csv(RAW / "ollama_viabilidade_pratica.csv", viability_rows)


if __name__ == "__main__":
    main()
