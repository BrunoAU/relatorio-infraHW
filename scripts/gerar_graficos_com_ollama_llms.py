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
        RAW / "benchmark_ollama_llms.csv",
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
        "prompt_nome",
        "repeticoes",
        "tempo_medio_ms",
        "tempo_desvio_ms",
        "tempo_min_ms",
        "tempo_max_ms",
        "total_medio_ms",
        "total_desvio_ms",
        "cliente_medio_ms",
        "load_medio_ms",
        "prompt_eval_medio_ms",
        "eval_medio_ms",
        "tokens_s_medio",
        "tokens_s_desvio",
        "eval_count_medio",
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


def plot_ollama_total(df):
    subset = df[df["teste"] == "ollama_llm_generate"].copy()

    if subset.empty:
        return

    subset = subset[subset["status"] == "ok"].copy()

    if subset.empty:
        return

    CHARTS.mkdir(parents=True, exist_ok=True)

    prompts = list(subset["prompt_nome"].dropna().unique())
    modelos = list(subset["modelo"].dropna().unique())

    x = range(len(prompts))
    largura = 0.8 / max(len(modelos), 1)

    plt.figure(figsize=(10, 5))

    for i, modelo in enumerate(modelos):
        grupo = subset[subset["modelo"] == modelo].set_index("prompt_nome")
        valores = [grupo.loc[p, "total_medio_ms"] if p in grupo.index else 0 for p in prompts]
        posicoes = [pos + i * largura for pos in x]
        plt.bar(posicoes, valores, width=largura, label=modelo)

    centro = [pos + largura * (len(modelos) - 1) / 2 for pos in x]
    plt.xticks(centro, prompts, rotation=20)
    plt.xlabel("Prompt")
    plt.ylabel("Tempo total médio (ms)")
    plt.title("Geração local via Ollama: tempo total médio")
    plt.legend()
    plt.tight_layout()

    path = CHARTS / "tempo_ollama_llms_total.png"
    plt.savefig(path, dpi=150)
    plt.close()

    print(f"Gráfico salvo em: {path}")


def plot_ollama_tokens(df):
    subset = df[df["teste"] == "ollama_llm_generate"].copy()

    if subset.empty:
        return

    subset = subset[subset["status"] == "ok"].copy()

    if subset.empty:
        return

    CHARTS.mkdir(parents=True, exist_ok=True)

    prompts = list(subset["prompt_nome"].dropna().unique())
    modelos = list(subset["modelo"].dropna().unique())

    x = range(len(prompts))
    largura = 0.8 / max(len(modelos), 1)

    plt.figure(figsize=(10, 5))

    for i, modelo in enumerate(modelos):
        grupo = subset[subset["modelo"] == modelo].set_index("prompt_nome")
        valores = [grupo.loc[p, "tokens_s_medio"] if p in grupo.index else 0 for p in prompts]
        posicoes = [pos + i * largura for pos in x]
        plt.bar(posicoes, valores, width=largura, label=modelo)

    centro = [pos + largura * (len(modelos) - 1) / 2 for pos in x]
    plt.xticks(centro, prompts, rotation=20)
    plt.xlabel("Prompt")
    plt.ylabel("Tokens por segundo")
    plt.title("Geração local via Ollama: velocidade média")
    plt.legend()
    plt.tight_layout()

    path = CHARTS / "tokens_ollama_llms.png"
    plt.savefig(path, dpi=150)
    plt.close()

    print(f"Gráfico salvo em: {path}")


def main():
    df = load_data()
    save_consolidated(df)
    plot_ollama_total(df)
    plot_ollama_tokens(df)


if __name__ == "__main__":
    main()
