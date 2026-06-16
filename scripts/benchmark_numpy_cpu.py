import gc
import time

from bootstrap import ensure_local_deps

ensure_local_deps()

import numpy as np
from common import RAW, ensure_dirs, load_config, process_memory_mb, set_seeds, timing_stats_ms, write_csv


def run_matmul(size: int, repetitions: int, warmups: int):
    a = np.random.randn(size, size).astype(np.float32)
    b = np.random.randn(size, size).astype(np.float32)
    for _ in range(warmups):
        _ = a @ b

    mem_before = process_memory_mb()
    times = []
    details = []
    status = "ok"
    observation = "Multiplicação de matrizes float32 em CPU com NumPy."
    try:
        for repetition in range(1, repetitions + 1):
            start = time.perf_counter()
            c = a @ b
            elapsed = time.perf_counter() - start
            _ = float(c[0, 0])
            times.append(elapsed)
            details.append(
                {
                    "teste": "numpy_matmul_cpu",
                    "hardware": "CPU",
                    "tamanho": size,
                    "batch": "na",
                    "repeticao": repetition,
                    "tempo_ms": round(elapsed * 1000, 4),
                    "memoria_ram_mb": round(process_memory_mb(), 2),
                    "gpu_peak_mb": "na",
                    "status": status,
                    "observacao": observation,
                }
            )
    except Exception as exc:
        status = "erro"
        observation = str(exc)[:250]

    mem_after = process_memory_mb()
    stats = timing_stats_ms(times)
    summary = {
        "teste": "numpy_matmul_cpu",
        "hardware": "CPU",
        "tamanho": size,
        "batch": "na",
        "repeticoes": len(times),
        **stats,
        "throughput_itens_s": "",
        "speedup_cuda_vs_cpu": "",
        "memoria_inicio_mb": round(mem_before, 2),
        "memoria_fim_mb": round(mem_after, 2),
        "gpu_peak_mb": "na",
        "status": status,
        "observacao": observation,
    }
    del a, b
    if "c" in locals():
        del c
    gc.collect()
    return summary, details


def main():
    ensure_dirs()
    config = load_config()
    set_seeds(int(config.get("seed", 42)))
    repetitions = int(config.get("repetitions", 3))
    warmups = int(config.get("warmups", 1))
    sizes = config.get("matrix_sizes", [512, 1024])

    summaries = []
    details = []
    for size in sizes:
        print(f"NumPy CPU matmul {size}x{size}")
        summary, rows = run_matmul(int(size), repetitions, warmups)
        summaries.append(summary)
        details.extend(rows)

    write_csv(RAW / "benchmark_numpy_cpu.csv", summaries)
    write_csv(RAW / "benchmark_numpy_cpu_execucoes.csv", details)


if __name__ == "__main__":
    main()
