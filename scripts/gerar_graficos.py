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

    paths = [
        RAW / "benchmark_numpy_cpu.csv",
        RAW / "benchmark_torch_cpu_gpu.csv",
        RAW / "benchmark_limite_gpu.csv",
        RAW / "benchmark_cnn_sintetico.csv",
        RAW / "benchmark_modelos_reais.csv",
    ]

    for path in paths:
        if path.exists():
            frames.append(pd.read_csv(path))

    if len(frames) == 0:
        raise FileNotFoundError("Nenhum CSV de benchmark encontrado em dados/raw.")

    return pd.concat(frames, ignore_index=True)


def save_consolidated(df):
    PROCESSED.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    consolidated = PROCESSED / "resultados_consolidados.csv"
    df.to_csv(consolidated, index=False)

    preferred_columns = [
        "teste",
        "modelo",
        "pretrained",
        "hardware",
        "tamanho",
        "batch",
        "repeticoes",
        "tempo_medio_ms",
        "tempo_desvio_ms",
        "tempo_min_ms",
        "tempo_max_ms",
        "memoria_inicio_mb",
        "memoria_fim_mb",
        "gpu_peak_mb",
        "status",
        "observacao",
    ]

    existing_columns = []

    for column in preferred_columns:
        if column in df.columns:
            existing_columns.append(column)

    summary = df[existing_columns]
    summary.to_csv(TABLES / "tabela_resultados_artigo.csv", index=False)

    print(f"Dados consolidados salvos em: {consolidated}")
    print(f"Tabela do artigo salva em: {TABLES / 'tabela_resultados_artigo.csv'}")


def plot_matmul(df):
    CHARTS.mkdir(parents=True, exist_ok=True)

    subset = df[df["teste"].isin(["numpy_matmul_cpu", "torch_matmul"])].copy()

    if subset.empty:
        return

    subset["label"] = subset["teste"] + " - " + subset["hardware"]

    plt.figure(figsize=(9, 5))

    for label, group in subset.groupby("label"):
        group = group.sort_values("tamanho")
        yerr = group["tempo_desvio_ms"] if "tempo_desvio_ms" in group.columns else None

        plt.errorbar(
            group["tamanho"].astype(str),
            group["tempo_medio_ms"],
            yerr=yerr,
            marker="o",
            capsize=4,
            label=label,
        )

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

    subset["batch"] = pd.to_numeric(subset["batch"], errors="coerce")
    subset = subset.sort_values("batch")

    plt.figure(figsize=(9, 5))

    for hardware, group in subset.groupby("hardware"):
        group = group.sort_values("batch")
        yerr = group["tempo_desvio_ms"] if "tempo_desvio_ms" in group.columns else None

        plt.errorbar(
            group["batch"].astype(int).astype(str),
            group["tempo_medio_ms"],
            yerr=yerr,
            marker="o",
            capsize=4,
            label=hardware,
        )

    plt.xlabel("Batch")
    plt.ylabel("Tempo médio (ms)")
    plt.title("Inferência local em MLP sintético")
    plt.legend()
    plt.tight_layout()

    path = CHARTS / "tempo_mlp_inferencia.png"
    plt.savefig(path, dpi=150)
    plt.close()

    print(f"Gráfico salvo em: {path}")


def plot_cnn(df):
    subset = df[df["teste"] == "torch_cnn_sintetico"].copy()

    if subset.empty:
        return

    subset["batch"] = pd.to_numeric(subset["batch"], errors="coerce")
    subset = subset.sort_values("batch")

    plt.figure(figsize=(9, 5))

    for hardware, group in subset.groupby("hardware"):
        group = group.sort_values("batch")
        yerr = group["tempo_desvio_ms"] if "tempo_desvio_ms" in group.columns else None

        plt.errorbar(
            group["batch"].astype(int).astype(str),
            group["tempo_medio_ms"],
            yerr=yerr,
            marker="o",
            capsize=4,
            label=hardware,
        )

    plt.xlabel("Batch")
    plt.ylabel("Tempo médio (ms)")
    plt.title("Inferência local em CNN sintético")
    plt.legend()
    plt.tight_layout()

    path = CHARTS / "tempo_cnn_sintetico.png"
    plt.savefig(path, dpi=150)
    plt.close()

    print(f"Gráfico salvo em: {path}")


def plot_limite_gpu(df):
    subset = df[df["teste"] == "limite_matmul_cuda"].copy()

    if subset.empty:
        return

    subset = subset[subset["status"] == "ok"].copy()

    if subset.empty:
        return

    subset = subset.sort_values("tamanho")

    plt.figure(figsize=(9, 5))

    yerr = subset["tempo_desvio_ms"] if "tempo_desvio_ms" in subset.columns else None

    plt.errorbar(
        subset["tamanho"].astype(str),
        subset["tempo_medio_ms"],
        yerr=yerr,
        marker="o",
        capsize=4,
        label="CUDA",
    )

    plt.xlabel("Tamanho da matriz")
    plt.ylabel("Tempo médio (ms)")
    plt.title("Teste de limite da GPU em multiplicação de matrizes")
    plt.legend()
    plt.tight_layout()

    path = CHARTS / "tempo_limite_gpu.png"
    plt.savefig(path, dpi=150)
    plt.close()

    print(f"Gráfico salvo em: {path}")


def plot_modelos_reais(df):
    subset = df[df["teste"] == "torch_modelo_real_inferencia"].copy()

    if subset.empty:
        return

    subset = subset[subset["status"] == "ok"].copy()

    if subset.empty:
        return

    subset["batch"] = pd.to_numeric(subset["batch"], errors="coerce")
    subset = subset.sort_values("batch")

    for modelo, df_modelo in subset.groupby("modelo"):
        plt.figure(figsize=(9, 5))

        for hardware, group in df_modelo.groupby("hardware"):
            group = group.sort_values("batch")
            yerr = group["tempo_desvio_ms"] if "tempo_desvio_ms" in group.columns else None

            plt.errorbar(
                group["batch"].astype(int).astype(str),
                group["tempo_medio_ms"],
                yerr=yerr,
                marker="o",
                capsize=4,
                label=hardware,
            )

        plt.xlabel("Batch")
        plt.ylabel("Tempo médio (ms)")
        plt.title(f"Inferência local em modelo real: {modelo}")
        plt.legend()
        plt.tight_layout()

        path = CHARTS / f"tempo_modelo_real_{modelo}.png"
        plt.savefig(path, dpi=150)
        plt.close()

        print(f"Gráfico salvo em: {path}")


def main():
    df = load_data()

    save_consolidated(df)
    plot_matmul(df)
    plot_mlp(df)
    plot_cnn(df)
    plot_limite_gpu(df)
    plot_modelos_reais(df)


if __name__ == "__main__":
    main()
