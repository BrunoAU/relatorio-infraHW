from pathlib import Path

from bootstrap import ensure_local_deps

ensure_local_deps()

import pandas as pd
from pandas.errors import EmptyDataError
from common import PROCESSED, RAW, TABLES, ensure_dirs


FILES = [
    RAW / "benchmark_numpy_cpu.csv",
    RAW / "benchmark_torch_cpu_gpu.csv",
    RAW / "benchmark_limite_gpu.csv",
    RAW / "benchmark_cnn_sintetico.csv",
    RAW / "benchmark_modelos_reais.csv",
    RAW / "benchmark_ollama_llms.csv",
]


def load_frames():
    frames = []
    for path in FILES:
        if path.exists():
            frame = pd.read_csv(path)
            frame["arquivo_origem"] = path.name
            frames.append(frame)
    return frames


def main():
    ensure_dirs()
    frames = load_frames()
    if not frames:
        raise FileNotFoundError("Nenhum CSV de benchmark encontrado em dados/raw.")

    consolidated = pd.concat(frames, ignore_index=True, sort=False)
    consolidated.to_csv(PROCESSED / "resultados_consolidados.csv", index=False)

    article_columns = [
        "teste",
        "modelo",
        "pretrained",
        "hardware",
        "tamanho",
        "batch",
        "repeticoes",
        "tempo_medio_ms",
        "tempo_mediana_ms",
        "tempo_desvio_ms",
        "tempo_min_ms",
        "tempo_max_ms",
        "tempo_p95_ms",
        "throughput_itens_s",
        "speedup_cuda_vs_cpu",
        "gpu_peak_mb",
        "status",
        "observacao",
    ]
    existing = [column for column in article_columns if column in consolidated.columns]
    consolidated[existing].to_csv(TABLES / "tabela_resultados_artigo.csv", index=False)

    if {"teste", "hardware", "speedup_cuda_vs_cpu"}.issubset(consolidated.columns):
        speedups = consolidated[
            (consolidated["hardware"].astype(str).str.upper() == "CUDA")
            & (consolidated["speedup_cuda_vs_cpu"].notna())
            & (consolidated["speedup_cuda_vs_cpu"].astype(str) != "")
        ].copy()
    else:
        speedups = pd.DataFrame(columns=["teste", "hardware", "speedup_cuda_vs_cpu"])
    speedups.to_csv(TABLES / "speedups_cpu_cuda.csv", index=False)

    limitations = []
    for _, row in consolidated.iterrows():
        status = str(row.get("status", ""))
        if status in {"skip", "erro", "error"}:
            limitations.append(
                {
                    "teste": row.get("teste", ""),
                    "modelo": row.get("modelo", ""),
                    "hardware": row.get("hardware", ""),
                    "status": status,
                    "limitacao": row.get("observacao", row.get("erro_ou_skip", "")),
                }
            )
    limitations.append(
        {
            "teste": "escopo_experimental",
            "modelo": "",
            "hardware": "NPU",
            "status": "nao_medido",
            "limitacao": "NPU nao foi testada experimentalmente neste projeto; o escopo experimental cobre CPU e GPU CUDA quando disponiveis.",
        }
    )
    pd.DataFrame(limitations).to_csv(TABLES / "limitacoes_experimentais.csv", index=False)

    ollama_sources = [
        RAW / "ollama_modelos_resumo.csv",
        RAW / "ollama_por_tarefa.csv",
        RAW / "ollama_tokens_por_segundo.csv",
        RAW / "ollama_uso_hardware.csv",
        RAW / "ollama_viabilidade_pratica.csv",
        RAW / "ollama_hardware_trace.csv",
    ]
    for path in ollama_sources:
        if path.exists():
            target = TABLES / path.name
            try:
                pd.read_csv(path).to_csv(target, index=False)
            except EmptyDataError:
                continue


if __name__ == "__main__":
    main()
