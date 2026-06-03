from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "dados" / "raw"
PROCESSED = ROOT / "dados" / "processed"
TABLES = ROOT / "resultados" / "tabelas"
CHARTS = ROOT / "resultados" / "graficos"


def load_data():
    frames = []

    numpy_path = RAW / "benchmark_numpy_cpu.csv"
    torch_path = RAW / "benchmark_torch_cpu_gpu.csv"

    if numpy_path.exists():
        frames.append(pd.read_csv(numpy_path))

    if torch_path.exists():
        frames.append(pd.read_csv(torch_path))

    if len(frames) == 0:
        raise FileNotFoundError("Nenhum CSV de benchmark encontrado em dados/raw.")

    return pd.concat(frames, ignore_index=True)


def save_consolidated(df):
    PROCESSED.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    consolidated = PROCESSED / "resultados_consolidados.csv"
    df.to_csv(consolidated, index=False)

    summary = df[["teste", "hardware", "tamanho", "batch", "tempo_medio_ms", "memoria_inicio_mb", "memoria_fim_mb", "gpu_peak_mb", "observacao"]]
    summary.to_csv(TABLES / "tabela_resultados_artigo.csv", index=False)

    print(f"Dados consolidados salvos em: {consolidated}")


def plot_matmul(df):
    CHARTS.mkdir(parents=True, exist_ok=True)
    subset = df[df["teste"].isin(["numpy_matmul_cpu", "torch_matmul"])].copy()

    if subset.empty:
        return

    subset["label"] = subset["teste"] + " - " + subset["hardware"]

    plt.figure(figsize=(9, 5))

    for label, group in subset.groupby("label"):
        group = group.sort_values("tamanho")
        plt.plot(group["tamanho"].astype(str), group["tempo_medio_ms"], marker="o", label=label)

    plt.xlabel("Tamanho da matriz")
    plt.ylabel("Tempo médio (ms)")
    plt.title("Comparação de tempo em multiplicação de matrizes")
    plt.legend()
    plt.tight_layout()
    path = CHARTS / "tempo_matmul.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Gráfico salvo em: {path}")


def plot_mlp(df):
    subset = df[df["teste"] == "torch_mlp_inference"].copy()

    if subset.empty:
        return

    plt.figure(figsize=(9, 5))

    for hardware, group in subset.groupby("hardware"):
        group = group.sort_values("batch")
        plt.plot(group["batch"].astype(str), group["tempo_medio_ms"], marker="o", label=hardware)

    plt.xlabel("Batch")
    plt.ylabel("Tempo médio (ms)")
    plt.title("Inferência local em MLP sintético")
    plt.legend()
    plt.tight_layout()
    path = CHARTS / "tempo_mlp_inferencia.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Gráfico salvo em: {path}")


def main():
    df = load_data()
    save_consolidated(df)
    plot_matmul(df)
    plot_mlp(df)


if __name__ == "__main__":
    main()
