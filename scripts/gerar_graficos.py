from pathlib import Path

from bootstrap import ensure_local_deps

ensure_local_deps()

import matplotlib.pyplot as plt
import pandas as pd
from pandas.errors import EmptyDataError
from common import CHARTS, PROCESSED, RAW, ensure_dirs


def load_csv(path: Path):
    if path.exists():
        try:
            return pd.read_csv(path)
        except EmptyDataError:
            return pd.DataFrame()
    return pd.DataFrame()


def save_plot(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def line_plot(df, x, y, hue, title, xlabel, ylabel, path):
    if df.empty or y not in df.columns:
        return
    plt.figure(figsize=(10, 5))
    for label, group in df.groupby(hue):
        group = group.sort_values(x)
        plt.plot(group[x].astype(str), group[y], marker="o", label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    save_plot(path)


def bar_plot(df, x, y, title, xlabel, ylabel, path):
    if df.empty or y not in df.columns:
        return
    plt.figure(figsize=(10, 5))
    plt.bar(df[x].astype(str), df[y])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20, ha="right")
    save_plot(path)


def scatter_plot(df, x, y, label, title, xlabel, ylabel, path):
    if df.empty or x not in df.columns or y not in df.columns:
        return
    plt.figure(figsize=(8, 5))
    plt.scatter(df[x], df[y])
    for _, row in df.iterrows():
        plt.annotate(str(row[label]), (row[x], row[y]), fontsize=8)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    save_plot(path)


def main():
    ensure_dirs()
    CHARTS.mkdir(parents=True, exist_ok=True)
    consolidated = load_csv(PROCESSED / "resultados_consolidados.csv")
    if consolidated.empty:
        consolidated = pd.concat(
            [frame for frame in [load_csv(RAW / "benchmark_numpy_cpu.csv"), load_csv(RAW / "benchmark_torch_cpu_gpu.csv"), load_csv(RAW / "benchmark_cnn_sintetico.csv"), load_csv(RAW / "benchmark_modelos_reais.csv"), load_csv(RAW / "benchmark_limite_gpu.csv")] if not frame.empty],
            ignore_index=True,
        )

    matmul = consolidated[consolidated["teste"].isin(["numpy_matmul_cpu", "torch_matmul"])].copy() if not consolidated.empty else pd.DataFrame()
    if not matmul.empty:
        matmul["serie"] = matmul["teste"].astype(str) + "_" + matmul["hardware"].astype(str)
        line_plot(matmul, "tamanho", "tempo_medio_ms", "serie", "Tempo de matmul por backend", "Tamanho da matriz", "Tempo medio (ms)", CHARTS / "tempo_matmul.png")

    mlp = consolidated[consolidated["teste"] == "torch_mlp_inference"].copy() if not consolidated.empty else pd.DataFrame()
    if not mlp.empty:
        line_plot(mlp, "batch", "tempo_medio_ms", "hardware", "MLP por batch", "Batch", "Tempo medio (ms)", CHARTS / "tempo_mlp_inferencia.png")

    cnn = consolidated[consolidated["teste"] == "torch_cnn_sintetico"].copy() if not consolidated.empty else pd.DataFrame()
    if not cnn.empty:
        line_plot(cnn, "batch", "tempo_medio_ms", "hardware", "CNN sintetica por batch", "Batch", "Tempo medio (ms)", CHARTS / "tempo_cnn_sintetico.png")

    gpu_limit = consolidated[consolidated["teste"] == "limite_matmul_cuda"].copy() if not consolidated.empty else pd.DataFrame()
    if not gpu_limit.empty:
        line_plot(gpu_limit[gpu_limit["status"] == "ok"], "tamanho", "tempo_medio_ms", "hardware", "Tempo limite GPU", "Tamanho da matriz", "Tempo medio (ms)", CHARTS / "tempo_limite_gpu.png")

    vision = consolidated[consolidated["teste"] == "torch_modelo_real_inferencia"].copy() if not consolidated.empty else pd.DataFrame()
    if not vision.empty:
        for model, group in vision.groupby("modelo"):
            line_plot(group[group["status"] == "ok"], "batch", "tempo_medio_ms", "hardware", f"{model} por batch", "Batch", "Tempo medio (ms)", CHARTS / f"tempo_modelo_real_{model}.png")

    speedups = load_csv(Path(RAW.parent.parent / "resultados" / "tabelas" / "speedups_cpu_cuda.csv"))
    if not speedups.empty and "speedup_cuda_vs_cpu" in speedups.columns:
        speedups = speedups.copy()
        label_cols = [col for col in ["teste", "modelo", "tamanho", "batch"] if col in speedups.columns]
        speedups["label"] = speedups[label_cols].apply(
            lambda row: " | ".join(str(value) for value in row.fillna("na").tolist()),
            axis=1,
        )
        bar_plot(speedups, "label", "speedup_cuda_vs_cpu", "Speedup CUDA vs CPU", "Configuracao", "Speedup", CHARTS / "speedup_cuda_vs_cpu.png")

    ollama_exec = load_csv(RAW / "ollama_benchmark_resumo.csv")
    ollama_models = load_csv(RAW / "ollama_modelos_resumo.csv")
    ollama_tasks = load_csv(RAW / "ollama_por_tarefa.csv")
    ollama_hw = load_csv(RAW / "ollama_uso_hardware.csv")
    ollama_viability = load_csv(RAW / "ollama_viabilidade_pratica.csv")
    ollama_trace = load_csv(RAW / "ollama_hardware_trace.csv")

    if not ollama_models.empty:
        bar_plot(ollama_models[ollama_models["status"] == "ok"], "modelo", "tempo_total_ms_media", "Tempo total dos LLMs", "Modelo", "Tempo medio (ms)", CHARTS / "tempo_ollama_llms_total.png")
        bar_plot(ollama_models[ollama_models["status"] == "ok"], "modelo", "tokens_por_segundo_media", "Tokens/s dos LLMs", "Modelo", "Tokens/s", CHARTS / "tokens_ollama_llms.png")
        bar_plot(ollama_models[ollama_models["status"] == "ok"], "modelo", "ram_processo_mb_pico_medio", "Pico de RAM por modelo", "Modelo", "RAM processo pico media (MB)", CHARTS / "ollama_pico_ram_por_modelo.png")
        bar_plot(ollama_models[ollama_models["status"] == "ok"], "modelo", "gpu_media_media", "Uso medio de GPU por modelo", "Modelo", "GPU media (%)", CHARTS / "ollama_gpu_media_por_modelo.png")
        bar_plot(ollama_models[ollama_models["status"] == "ok"], "modelo", "vram_mb_pico_medio", "Pico de VRAM por modelo", "Modelo", "VRAM pico media (MB)", CHARTS / "ollama_pico_vram_por_modelo.png")
        scatter_plot(ollama_models[ollama_models["status"] == "ok"], "tempo_total_ms_media", "ram_processo_mb_pico_medio", "modelo", "Tempo de resposta vs uso de RAM", "Tempo medio (ms)", "RAM processo pico media (MB)", CHARTS / "ollama_tempo_vs_hardware.png")

    if not ollama_tasks.empty:
        line_plot(ollama_tasks[ollama_tasks["status"] == "ok"], "categoria_tarefa", "tokens_por_segundo_media", "modelo", "Tokens/s por categoria de tarefa", "Categoria", "Tokens/s", CHARTS / "ollama_tokens_por_tarefa.png")

    if not ollama_hw.empty:
        bar_plot(ollama_hw, "modelo", "cpu_media", "Uso medio de CPU por modelo", "Modelo", "CPU media (%)", CHARTS / "ollama_cpu_medio_por_modelo.png")

    if not ollama_viability.empty and "classificacao_viabilidade" in ollama_viability.columns:
        counts = ollama_viability["classificacao_viabilidade"].value_counts().reset_index()
        counts.columns = ["classificacao_viabilidade", "quantidade"]
        bar_plot(counts, "classificacao_viabilidade", "quantidade", "Viabilidade pratica dos modelos Ollama", "Classe", "Quantidade", CHARTS / "ollama_viabilidade_pratica.png")
        if "status" in ollama_viability.columns:
            status_counts = ollama_viability["status"].value_counts().reset_index()
            status_counts.columns = ["status", "quantidade"]
            bar_plot(status_counts, "status", "quantidade", "Taxa de sucesso, skip ou erro por modelo", "Status", "Quantidade", CHARTS / "ollama_taxa_status.png")

    if not ollama_trace.empty:
        rep = ollama_trace.sort_values(["modelo", "categoria_tarefa", "repeticao", "tempo_relativo_s"]).head(50).copy()
        plt.figure(figsize=(10, 5))
        if "cpu_percent" in rep.columns:
            plt.plot(rep["tempo_relativo_s"], rep["cpu_percent"], label="CPU%")
        if "ram_process_mb" in rep.columns:
            plt.plot(rep["tempo_relativo_s"], rep["ram_process_mb"], label="RAM processo MB")
        if "gpu_percent" in rep.columns:
            numeric_gpu = pd.to_numeric(rep["gpu_percent"], errors="coerce")
            if numeric_gpu.notna().any():
                plt.plot(rep["tempo_relativo_s"], numeric_gpu, label="GPU%")
        if "vram_used_mb" in rep.columns:
            numeric_vram = pd.to_numeric(rep["vram_used_mb"], errors="coerce")
            if numeric_vram.notna().any():
                plt.plot(rep["tempo_relativo_s"], numeric_vram, label="VRAM MB")
        plt.title("Serie temporal representativa de uso de hardware no Ollama")
        plt.xlabel("Tempo relativo (s)")
        plt.ylabel("Uso")
        plt.legend()
        save_plot(CHARTS / "ollama_serie_temporal_representativa.png")


if __name__ == "__main__":
    main()
