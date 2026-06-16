import time

from bootstrap import ensure_local_deps

ensure_local_deps()

import torch
from common import (
    RAW,
    clear_cuda_cache,
    cuda_peak_mb,
    cuda_sync,
    ensure_dirs,
    load_config,
    process_memory_mb,
    reset_cuda_peak,
    set_seeds,
    timing_stats_ms,
    write_csv,
)


def run_size(size, repetitions, warmups):
    if not torch.cuda.is_available():
        return {
            "teste": "limite_matmul_cuda",
            "hardware": "CUDA",
            "tamanho": size,
            "batch": "na",
            "repeticoes": 0,
            "tempo_medio_ms": "",
            "tempo_mediana_ms": "",
            "tempo_desvio_ms": "",
            "tempo_min_ms": "",
            "tempo_max_ms": "",
            "tempo_p95_ms": "",
            "throughput_itens_s": "",
            "speedup_cuda_vs_cpu": "",
            "memoria_inicio_mb": "",
            "memoria_fim_mb": "",
            "gpu_peak_mb": "",
            "status": "skip",
            "observacao": "CUDA nao disponivel neste ambiente.",
        }, []

    device = torch.device("cuda")
    clear_cuda_cache()
    reset_cuda_peak(device)
    status = "ok"
    observation = "Teste de limite da GPU com multiplicacao de matrizes em CUDA."
    details = []
    times = []
    mem_before = process_memory_mb()
    try:
        a = torch.randn((size, size), device=device)
        b = torch.randn((size, size), device=device)
        for _ in range(warmups):
            _ = a @ b
            cuda_sync(device)
        for repetition in range(1, repetitions + 1):
            cuda_sync(device)
            start = time.perf_counter()
            c = a @ b
            cuda_sync(device)
            elapsed = time.perf_counter() - start
            _ = float(c[0, 0].detach().cpu())
            times.append(elapsed)
            details.append(
                {
                    "teste": "limite_matmul_cuda",
                    "hardware": "CUDA",
                    "tamanho": size,
                    "batch": "na",
                    "repeticao": repetition,
                    "tempo_ms": round(elapsed * 1000, 4),
                    "memoria_ram_mb": round(process_memory_mb(), 2),
                    "gpu_peak_mb": cuda_peak_mb(device),
                    "status": status,
                    "observacao": observation,
                }
            )
    except RuntimeError as exc:
        status = "erro"
        observation = str(exc).replace("\n", " ")[:250]

    st = timing_stats_ms(times)
    summary = {
        "teste": "limite_matmul_cuda",
        "hardware": "CUDA",
        "tamanho": size,
        "batch": "na",
        "repeticoes": len(times),
        **st,
        "throughput_itens_s": "",
        "speedup_cuda_vs_cpu": "",
        "memoria_inicio_mb": round(mem_before, 2),
        "memoria_fim_mb": round(process_memory_mb(), 2),
        "gpu_peak_mb": cuda_peak_mb(device),
        "status": status,
        "observacao": observation,
    }
    if "a" in locals():
        del a
    if "b" in locals():
        del b
    if "c" in locals():
        del c
    clear_cuda_cache()
    return summary, details


def main():
    ensure_dirs()
    config = load_config()
    set_seeds(int(config.get("seed", 42)))
    repetitions = int(config.get("repetitions", 3))
    warmups = int(config.get("warmups", 1))
    sizes = [int(x) for x in config.get("gpu_limit_matrix_sizes", [1536, 2048, 2560])]
    if not config.get("run_gpu_limit", True):
        sizes = []

    summaries = []
    details = []
    if not sizes:
        summaries.append(
            {
                "teste": "limite_matmul_cuda",
                "hardware": "CUDA",
                "tamanho": "config_disabled",
                "batch": "na",
                "repeticoes": 0,
                "tempo_medio_ms": "",
                "tempo_mediana_ms": "",
                "tempo_desvio_ms": "",
                "tempo_min_ms": "",
                "tempo_max_ms": "",
                "tempo_p95_ms": "",
                "throughput_itens_s": "",
                "speedup_cuda_vs_cpu": "",
                "memoria_inicio_mb": "",
                "memoria_fim_mb": "",
                "gpu_peak_mb": "",
                "status": "skip",
                "observacao": "Teste de limite da GPU desativado no benchmark_config.json.",
            }
        )
    for size in sizes:
        print(f"Limite GPU {size}x{size}")
        summary, rows = run_size(size, repetitions, warmups)
        summaries.append(summary)
        details.extend(rows)

    write_csv(RAW / "benchmark_limite_gpu.csv", summaries)
    write_csv(RAW / "benchmark_limite_gpu_execucoes.csv", details)


if __name__ == "__main__":
    main()
