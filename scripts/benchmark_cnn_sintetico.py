import time

from bootstrap import ensure_local_deps

ensure_local_deps()

import torch
import torch.nn as nn
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


class CNNSintetica(nn.Module):
    def __init__(self):
        super().__init__()
        self.rede = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.rede(x)


def devices():
    result = [torch.device("cpu")]
    if torch.cuda.is_available():
        result.append(torch.device("cuda"))
    return result


def run(device, batch, repetitions, warmups, shape):
    c, h, w = shape
    hardware = device.type.upper()
    model = CNNSintetica().to(device)
    model.eval()
    x = torch.randn((batch, c, h, w), device=device)
    with torch.no_grad():
        for _ in range(warmups):
            _ = model(x)
            cuda_sync(device)
    reset_cuda_peak(device)

    times = []
    details = []
    mem_before = process_memory_mb()
    status = "ok"
    observation = "CNN sintética com entrada aleatória; mede desempenho da arquitetura, não acurácia."
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
                        "teste": "torch_cnn_sintetico",
                        "hardware": hardware,
                        "tamanho": f"{c}x{h}x{w}",
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

    st = timing_stats_ms(times)
    summary = {
        "teste": "torch_cnn_sintetico",
        "hardware": hardware,
        "tamanho": f"{c}x{h}x{w}",
        "batch": batch,
        "repeticoes": len(times),
        **st,
        "throughput_itens_s": throughput_items_per_second(batch, st["tempo_medio_ms"]),
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
    batches = config.get("batches", [1, 8])
    vision = config.get("vision_input", {})
    shape = (int(vision.get("channels", 3)), int(vision.get("height", 224)), int(vision.get("width", 224)))

    summaries = []
    details = []
    for device in devices():
        for batch in batches:
            print(f"CNN sintética {device} batch {batch}")
            summary, rows = run(device, int(batch), repetitions, warmups, shape)
            summaries.append(summary)
            details.extend(rows)

    speedups = calculate_speedups(summaries, ["teste", "tamanho", "batch"])
    for summary in summaries:
        for item in speedups:
            if all(summary.get(k) == item.get(k) for k in ["teste", "tamanho", "batch"]) and summary.get("hardware") == "CUDA":
                summary["speedup_cuda_vs_cpu"] = round(float(item["speedup_cuda_vs_cpu"]), 4) if item["speedup_cuda_vs_cpu"] != "" else ""

    write_csv(RAW / "benchmark_cnn_sintetico.csv", summaries)
    write_csv(RAW / "benchmark_cnn_sintetico_execucoes.csv", details)


if __name__ == "__main__":
    main()
