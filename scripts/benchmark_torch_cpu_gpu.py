import gc
import time

from bootstrap import ensure_local_deps

ensure_local_deps()

import torch
from common import (
    RAW,
    calculate_speedups,
    clear_cuda_cache,
    cuda_peak_mb,
    cuda_sync,
    ensure_dirs,
    load_config,
    process_memory_mb,
    reset_cuda_peak,
    set_seeds,
    throughput_items_per_second,
    timing_stats_ms,
    write_csv,
)


def devices():
    result = [torch.device("cpu")]
    if torch.cuda.is_available():
        result.append(torch.device("cuda"))
    return result


def run_matmul(device, size, repetitions, warmups):
    hardware = device.type.upper()
    a = torch.randn((size, size), dtype=torch.float32, device=device)
    b = torch.randn((size, size), dtype=torch.float32, device=device)
    with torch.no_grad():
        for _ in range(warmups):
            _ = a @ b
            cuda_sync(device)
    reset_cuda_peak(device)

    times = []
    details = []
    mem_before = process_memory_mb()
    status = "ok"
    observation = "Multiplicação de matrizes float32 com PyTorch."
    try:
        with torch.no_grad():
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
                        "teste": "torch_matmul",
                        "hardware": hardware,
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

    summary = {
        "teste": "torch_matmul",
        "hardware": hardware,
        "tamanho": size,
        "batch": "na",
        "repeticoes": len(times),
        **timing_stats_ms(times),
        "throughput_itens_s": "",
        "speedup_cuda_vs_cpu": "",
        "memoria_inicio_mb": round(mem_before, 2),
        "memoria_fim_mb": round(process_memory_mb(), 2),
        "gpu_peak_mb": cuda_peak_mb(device),
        "status": status,
        "observacao": observation,
    }
    del a, b
    if "c" in locals():
        del c
    clear_cuda_cache()
    return summary, details


def run_mlp(device, batch, repetitions, warmups):
    hardware = device.type.upper()
    model = torch.nn.Sequential(
        torch.nn.Linear(1024, 2048),
        torch.nn.ReLU(),
        torch.nn.Linear(2048, 2048),
        torch.nn.ReLU(),
        torch.nn.Linear(2048, 512),
    ).to(device)
    model.eval()
    x = torch.randn((batch, 1024), dtype=torch.float32, device=device)
    with torch.no_grad():
        for _ in range(warmups):
            _ = model(x)
            cuda_sync(device)
    reset_cuda_peak(device)

    times = []
    details = []
    mem_before = process_memory_mb()
    status = "ok"
    observation = "Inferência sintética MLP com entrada aleatória; mede desempenho, não acurácia."
    try:
        with torch.no_grad():
            for repetition in range(1, repetitions + 1):
                cuda_sync(device)
                start = time.perf_counter()
                y = model(x)
                cuda_sync(device)
                elapsed = time.perf_counter() - start
                _ = float(y[0, 0].detach().cpu())
                times.append(elapsed)
                details.append(
                    {
                        "teste": "torch_mlp_inference",
                        "hardware": hardware,
                        "tamanho": "mlp_1024_2048x2_512",
                        "batch": batch,
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

    stats = timing_stats_ms(times)
    summary = {
        "teste": "torch_mlp_inference",
        "hardware": hardware,
        "tamanho": "mlp_1024_2048x2_512",
        "batch": batch,
        "repeticoes": len(times),
        **stats,
        "throughput_itens_s": throughput_items_per_second(batch, stats["tempo_medio_ms"]),
        "speedup_cuda_vs_cpu": "",
        "memoria_inicio_mb": round(mem_before, 2),
        "memoria_fim_mb": round(process_memory_mb(), 2),
        "gpu_peak_mb": cuda_peak_mb(device),
        "status": status,
        "observacao": observation,
    }
    del model, x
    if "y" in locals():
        del y
    clear_cuda_cache()
    return summary, details


def main():
    ensure_dirs()
    config = load_config()
    set_seeds(int(config.get("seed", 42)))
    repetitions = int(config.get("repetitions", 3))
    warmups = int(config.get("warmups", 1))
    sizes = config.get("matrix_sizes", [512, 1024])
    batches = config.get("batches", [1, 8])

    summaries = []
    details = []
    for device in devices():
        for size in sizes:
            print(f"PyTorch matmul {device} {size}x{size}")
            summary, rows = run_matmul(device, int(size), repetitions, warmups)
            summaries.append(summary)
            details.extend(rows)
        for batch in batches:
            print(f"PyTorch MLP {device} batch {batch}")
            summary, rows = run_mlp(device, int(batch), repetitions, warmups)
            summaries.append(summary)
            details.extend(rows)

    speedups = calculate_speedups(summaries, ["teste", "tamanho", "batch"])
    for summary in summaries:
        for item in speedups:
            if all(summary.get(k) == item.get(k) for k in ["teste", "tamanho", "batch"]) and summary.get("hardware") == "CUDA":
                summary["speedup_cuda_vs_cpu"] = round(float(item["speedup_cuda_vs_cpu"]), 4) if item["speedup_cuda_vs_cpu"] != "" else ""

    write_csv(RAW / "benchmark_torch_cpu_gpu.csv", summaries)
    write_csv(RAW / "benchmark_torch_cpu_gpu_execucoes.csv", details)


if __name__ == "__main__":
    main()
