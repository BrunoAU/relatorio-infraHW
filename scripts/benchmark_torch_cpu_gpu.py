import csv
import gc
import time
from pathlib import Path

import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dados" / "raw" / "benchmark_torch_cpu_gpu.csv"

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


def get_devices():
    devices = [torch.device("cpu")]

    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    return devices


def run_matmul(device, size):
    a = torch.randn((size, size), dtype=torch.float32, device=device)
    b = torch.randn((size, size), dtype=torch.float32, device=device)

    for _ in range(WARMUP):
        _ = a @ b
    sync_if_needed(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    mem_before = memory_mb()
    times = []

    for _ in range(REPETITIONS):
        sync_if_needed(device)
        start = time.perf_counter()
        c = a @ b
        sync_if_needed(device)
        end = time.perf_counter()
        times.append(end - start)
        _ = float(c[0, 0].detach().cpu())

    mem_after = memory_mb()
    gpu_peak_mb = None

    if device.type == "cuda":
        gpu_peak_mb = round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2)

    del a
    del b
    del c
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "teste": "torch_matmul",
        "hardware": device.type.upper(),
        "tamanho": size,
        "batch": "na",
        "repeticoes": REPETITIONS,
        "tempo_medio_ms": round((sum(times) / len(times)) * 1000, 4),
        "tempo_min_ms": round(min(times) * 1000, 4),
        "tempo_max_ms": round(max(times) * 1000, 4),
        "memoria_inicio_mb": round(mem_before, 2),
        "memoria_fim_mb": round(mem_after, 2),
        "gpu_peak_mb": gpu_peak_mb,
        "observacao": "Multiplicação de matrizes float32 com PyTorch",
    }


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

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    mem_before = memory_mb()
    times = []

    with torch.no_grad():
        for _ in range(REPETITIONS):
            sync_if_needed(device)
            start = time.perf_counter()
            y = model(x)
            sync_if_needed(device)
            end = time.perf_counter()
            times.append(end - start)
            _ = float(y[0, 0].detach().cpu())

    mem_after = memory_mb()
    gpu_peak_mb = None

    if device.type == "cuda":
        gpu_peak_mb = round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2)

    del model
    del x
    del y
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "teste": "torch_mlp_inference",
        "hardware": device.type.upper(),
        "tamanho": "mlp_1024_2048x4_512",
        "batch": batch,
        "repeticoes": REPETITIONS,
        "tempo_medio_ms": round((sum(times) / len(times)) * 1000, 4),
        "tempo_min_ms": round(min(times) * 1000, 4),
        "tempo_max_ms": round(max(times) * 1000, 4),
        "memoria_inicio_mb": round(mem_before, 2),
        "memoria_fim_mb": round(mem_after, 2),
        "gpu_peak_mb": gpu_peak_mb,
        "observacao": "Inferência local em MLP sintético com PyTorch",
    }


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    devices = get_devices()

    for device in devices:
        print(f"Dispositivo PyTorch detectado: {device}")

        for size in SIZES:
            print(f"Rodando PyTorch matmul em {device}, tamanho {size}x{size}...")
            rows.append(run_matmul(device, size))

        for batch in BATCHES:
            print(f"Rodando PyTorch MLP em {device}, batch {batch}...")
            rows.append(run_mlp_inference(device, batch))

    with OUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Benchmark PyTorch CPU/GPU salvo em: {OUT}")


if __name__ == "__main__":
    main()
