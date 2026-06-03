import csv
import gc
import statistics
import time
from pathlib import Path

import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dados" / "raw" / "benchmark_torch_cpu_gpu.csv"
OUT_DETAILED = ROOT / "dados" / "raw" / "benchmark_torch_cpu_gpu_execucoes.csv"

SIZES = [512, 1024, 1536, 2048]
BATCHES = [1, 8, 32]
REPETITIONS = 10
WARMUP = 3


def memory_mb():
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 2)


def sync_if_needed(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def clear_cuda_if_needed(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()


def reset_cuda_peak_if_needed(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def get_cuda_peak_mb(device):
    if device.type != "cuda":
        return "na"

    return round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2)


def get_devices():
    devices = [torch.device("cpu")]

    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    return devices


def calculate_stats(times):
    times_ms = [value * 1000 for value in times]

    return {
        "tempo_medio_ms": round(statistics.mean(times_ms), 4),
        "tempo_desvio_ms": round(statistics.stdev(times_ms), 4) if len(times_ms) > 1 else 0.0,
        "tempo_min_ms": round(min(times_ms), 4),
        "tempo_max_ms": round(max(times_ms), 4),
    }


def run_matmul(device, size):
    a = torch.randn((size, size), dtype=torch.float32, device=device)
    b = torch.randn((size, size), dtype=torch.float32, device=device)

    for _ in range(WARMUP):
        _ = a @ b
    sync_if_needed(device)

    reset_cuda_peak_if_needed(device)

    mem_before = memory_mb()
    times = []
    detailed_rows = []

    for repetition in range(1, REPETITIONS + 1):
        sync_if_needed(device)
        start = time.perf_counter()
        c = a @ b
        sync_if_needed(device)
        end = time.perf_counter()

        elapsed = end - start
        times.append(elapsed)
        detailed_rows.append({
            "teste": "torch_matmul",
            "hardware": device.type.upper(),
            "tamanho": size,
            "batch": "na",
            "repeticao": repetition,
            "tempo_ms": round(elapsed * 1000, 4),
            "observacao": "Execução individual de multiplicação de matrizes float32 com PyTorch",
        })

        _ = float(c[0, 0].detach().cpu())

    mem_after = memory_mb()
    gpu_peak_mb = get_cuda_peak_mb(device)
    stats = calculate_stats(times)

    del a
    del b
    del c
    gc.collect()
    clear_cuda_if_needed(device)

    summary = {
        "teste": "torch_matmul",
        "hardware": device.type.upper(),
        "tamanho": size,
        "batch": "na",
        "repeticoes": REPETITIONS,
        **stats,
        "memoria_inicio_mb": round(mem_before, 2),
        "memoria_fim_mb": round(mem_after, 2),
        "gpu_peak_mb": gpu_peak_mb,
        "observacao": "Multiplicação de matrizes float32 com PyTorch",
    }

    return summary, detailed_rows


def run_mlp_inference(device, batch):
    input_size = 1024
    hidden_size = 2048
    output_size = 512
    layers = 4

    modules = []
    current_size = input_size

    for _ in range(layers):
        modules.append(torch.nn.Linear(current_size, hidden_size))
        modules.append(torch.nn.ReLU())
        current_size = hidden_size

    modules.append(torch.nn.Linear(current_size, output_size))
    model = torch.nn.Sequential(*modules).to(device)
    model.eval()

    x = torch.randn((batch, input_size), dtype=torch.float32, device=device)

    with torch.no_grad():
        for _ in range(WARMUP):
            _ = model(x)
    sync_if_needed(device)

    reset_cuda_peak_if_needed(device)

    mem_before = memory_mb()
    times = []
    detailed_rows = []

    with torch.no_grad():
        for repetition in range(1, REPETITIONS + 1):
            sync_if_needed(device)
            start = time.perf_counter()
            y = model(x)
            sync_if_needed(device)
            end = time.perf_counter()

            elapsed = end - start
            times.append(elapsed)
            detailed_rows.append({
                "teste": "torch_mlp_inference",
                "hardware": device.type.upper(),
                "tamanho": "mlp_1024_2048x4_512",
                "batch": batch,
                "repeticao": repetition,
                "tempo_ms": round(elapsed * 1000, 4),
                "observacao": "Execução individual de inferência local em MLP sintético com PyTorch",
            })

            _ = float(y[0, 0].detach().cpu())

    mem_after = memory_mb()
    gpu_peak_mb = get_cuda_peak_mb(device)
    stats = calculate_stats(times)

    del model
    del x
    del y
    gc.collect()
    clear_cuda_if_needed(device)

    summary = {
        "teste": "torch_mlp_inference",
        "hardware": device.type.upper(),
        "tamanho": "mlp_1024_2048x4_512",
        "batch": batch,
        "repeticoes": REPETITIONS,
        **stats,
        "memoria_inicio_mb": round(mem_before, 2),
        "memoria_fim_mb": round(mem_after, 2),
        "gpu_peak_mb": gpu_peak_mb,
        "observacao": "Inferência local em MLP sintético com PyTorch",
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
    devices = get_devices()

    for device in devices:
        print(f"Dispositivo PyTorch detectado: {device}")

        for size in SIZES:
            print(f"Rodando PyTorch matmul em {device}, tamanho {size}x{size}, com {REPETITIONS} repetições...")
            summary, details = run_matmul(device, size)
            summary_rows.append(summary)
            detailed_rows.extend(details)

        for batch in BATCHES:
            print(f"Rodando PyTorch MLP em {device}, batch {batch}, com {REPETITIONS} repetições...")
            summary, details = run_mlp_inference(device, batch)
            summary_rows.append(summary)
            detailed_rows.extend(details)

    write_csv(OUT, summary_rows)
    write_csv(OUT_DETAILED, detailed_rows)

    print(f"Benchmark PyTorch CPU/GPU salvo em: {OUT}")
    print(f"Execuções individuais PyTorch salvas em: {OUT_DETAILED}")


if __name__ == "__main__":
    main()
