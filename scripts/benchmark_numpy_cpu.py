import csv
import gc
import statistics
import time
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dados" / "raw" / "benchmark_numpy_cpu.csv"
OUT_DETAILED = ROOT / "dados" / "raw" / "benchmark_numpy_cpu_execucoes.csv"

SIZES = [512, 1024, 1536, 2048]
REPETITIONS = 10
WARMUP = 1


def memory_mb():
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 2)


def calculate_stats(times):
    times_ms = [value * 1000 for value in times]

    return {
        "tempo_medio_ms": round(statistics.mean(times_ms), 4),
        "tempo_desvio_ms": round(statistics.stdev(times_ms), 4) if len(times_ms) > 1 else 0.0,
        "tempo_min_ms": round(min(times_ms), 4),
        "tempo_max_ms": round(max(times_ms), 4),
    }


def run_matmul(size):
    a = np.random.randn(size, size).astype(np.float32)
    b = np.random.randn(size, size).astype(np.float32)

    for _ in range(WARMUP):
        _ = a @ b

    times = []
    detailed_rows = []
    mem_before = memory_mb()

    for repetition in range(1, REPETITIONS + 1):
        start = time.perf_counter()
        c = a @ b
        end = time.perf_counter()

        elapsed = end - start
        times.append(elapsed)
        detailed_rows.append({
            "teste": "numpy_matmul_cpu",
            "hardware": "CPU",
            "tamanho": size,
            "batch": "na",
            "repeticao": repetition,
            "tempo_ms": round(elapsed * 1000, 4),
            "observacao": "Execução individual de multiplicação de matrizes float32 em CPU com NumPy",
        })

        _ = float(c[0, 0])

    mem_after = memory_mb()
    stats = calculate_stats(times)

    del a
    del b
    del c
    gc.collect()

    summary = {
        "teste": "numpy_matmul_cpu",
        "hardware": "CPU",
        "tamanho": size,
        "batch": "na",
        "repeticoes": REPETITIONS,
        **stats,
        "memoria_inicio_mb": round(mem_before, 2),
        "memoria_fim_mb": round(mem_after, 2),
        "gpu_peak_mb": "na",
        "observacao": "Multiplicação de matrizes float32 em CPU com NumPy",
    }

    return summary, detailed_rows


def write_csv(path, rows):
    if len(rows) == 0:
        return

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    detailed_rows = []

    for size in SIZES:
        print(f"Rodando NumPy CPU matmul tamanho {size}x{size} com {REPETITIONS} repetições...")
        summary, details = run_matmul(size)
        summary_rows.append(summary)
        detailed_rows.extend(details)

    write_csv(OUT, summary_rows)
    write_csv(OUT_DETAILED, detailed_rows)

    print(f"Benchmark NumPy CPU salvo em: {OUT}")
    print(f"Execuções individuais NumPy CPU salvas em: {OUT_DETAILED}")


if __name__ == "__main__":
    main()
