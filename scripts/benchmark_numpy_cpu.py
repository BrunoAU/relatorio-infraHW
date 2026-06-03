import csv
import gc
import time
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dados" / "raw" / "benchmark_numpy_cpu.csv"

SIZES = [512, 1024, 1536, 2048]
REPETITIONS = 5
WARMUP = 1


def memory_mb():
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 2)


def run_matmul(size):
    a = np.random.randn(size, size).astype(np.float32)
    b = np.random.randn(size, size).astype(np.float32)

    for _ in range(WARMUP):
        _ = a @ b

    times = []
    mem_before = memory_mb()

    for _ in range(REPETITIONS):
        start = time.perf_counter()
        c = a @ b
        end = time.perf_counter()
        times.append(end - start)
        _ = float(c[0, 0])

    mem_after = memory_mb()

    del a
    del b
    del c
    gc.collect()

    return {
        "teste": "numpy_matmul_cpu",
        "hardware": "CPU",
        "tamanho": size,
        "repeticoes": REPETITIONS,
        "tempo_medio_ms": round((sum(times) / len(times)) * 1000, 4),
        "tempo_min_ms": round(min(times) * 1000, 4),
        "tempo_max_ms": round(max(times) * 1000, 4),
        "memoria_inicio_mb": round(mem_before, 2),
        "memoria_fim_mb": round(mem_after, 2),
        "observacao": "Multiplicação de matrizes float32 em CPU com NumPy",
    }


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for size in SIZES:
        print(f"Rodando NumPy CPU matmul tamanho {size}x{size}...")
        rows.append(run_matmul(size))

    with OUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Benchmark NumPy CPU salvo em: {OUT}")


if __name__ == "__main__":
    main()
