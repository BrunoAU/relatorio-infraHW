import argparse
import time

from bootstrap import ensure_local_deps

ensure_local_deps()

import torch
import torchvision.models as models
from common import (
    RAW,
    calculate_speedups,
    clear_cuda_cache,
    config_get,
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


def build_model(name: str, pretrained: bool):
    if name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        return models.mobilenet_v2(weights=weights)
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        return models.resnet18(weights=weights)
    raise ValueError(f"Modelo nao suportado: {name}")


def run_model(name, pretrained, device, batch, repetitions, warmups, shape):
    c, h, w = shape
    hardware = device.type.upper()
    model = build_model(name, pretrained).to(device)
    model.eval()
    x = torch.randn((batch, c, h, w), device=device)
    with torch.no_grad():
        for _ in range(warmups):
            _ = model(x)
            cuda_sync(device)
    reset_cuda_peak(device)

    status = "ok"
    if pretrained:
        observation = "Benchmark de inferencia com pesos pretrained; mede desempenho, nao qualidade."
    else:
        observation = "Arquitetura real com entrada sintetica e sem pesos pretrained; mede desempenho da arquitetura, nao acuracia."

    times = []
    details = []
    mem_before = process_memory_mb()
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
                        "teste": "torch_modelo_real_inferencia",
                        "modelo": name,
                        "pretrained": pretrained,
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
        "teste": "torch_modelo_real_inferencia",
        "modelo": name,
        "pretrained": pretrained,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", action="store_true", help="Ativa pesos pretrained do torchvision.")
    args = parser.parse_args()

    ensure_dirs()
    config = load_config()
    set_seeds(int(config.get("seed", 42)))
    repetitions = int(config.get("repetitions", 3))
    warmups = int(config.get("warmups", 1))
    batches = [int(x) for x in config.get("batches", [1, 8])]
    names = config.get("vision_models", ["mobilenet_v2", "resnet18"])
    vision = config.get("vision_input", {})
    shape = (int(vision.get("channels", 3)), int(vision.get("height", 224)), int(vision.get("width", 224)))
    pretrained = bool(args.pretrained or config_get(config, "pretrained_vision", False))

    summaries = []
    details = []
    for name in names:
        for device in devices():
            for batch in batches:
                print(f"Modelo real {name} {device} batch {batch}")
                summary, rows = run_model(name, pretrained, device, batch, repetitions, warmups, shape)
                summaries.append(summary)
                details.extend(rows)

    speedups = calculate_speedups(summaries, ["teste", "modelo", "tamanho", "batch", "pretrained"])
    for summary in summaries:
        for item in speedups:
            same = all(summary.get(k) == item.get(k) for k in ["teste", "modelo", "tamanho", "batch", "pretrained"])
            if same and summary.get("hardware") == "CUDA" and item["speedup_cuda_vs_cpu"] != "":
                summary["speedup_cuda_vs_cpu"] = round(float(item["speedup_cuda_vs_cpu"]), 4)

    write_csv(RAW / "benchmark_modelos_reais.csv", summaries)
    write_csv(RAW / "benchmark_modelos_reais_execucoes.csv", details)


if __name__ == "__main__":
    main()
