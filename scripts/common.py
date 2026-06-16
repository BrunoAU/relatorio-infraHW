from __future__ import annotations

import csv
import gc
import json
import math
import random
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "dados" / "raw"
PROCESSED = ROOT / "dados" / "processed"
RESULTS = ROOT / "resultados"
TABLES = RESULTS / "tabelas"
CHARTS = RESULTS / "graficos"
LOGS = RESULTS / "logs"
CONFIG_PATH = Path(__file__).with_name("benchmark_config.json")


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dirs() -> None:
    for path in (RAW, PROCESSED, TABLES, CHARTS, LOGS):
        path.mkdir(parents=True, exist_ok=True)


def load_config() -> dict[str, Any]:
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as file:
            return json.load(file)
    return {}


def config_get(config: dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = config
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def cuda_sync(device: Any = None) -> None:
    try:
        import torch

        if torch.cuda.is_available():
            if device is not None and getattr(device, "type", "") == "cuda":
                torch.cuda.synchronize(device)
            elif device is None:
                torch.cuda.synchronize()
    except Exception:
        pass


def clear_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def reset_cuda_peak(device: Any = None) -> None:
    try:
        import torch

        if torch.cuda.is_available():
            if device is not None and getattr(device, "type", "") == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            elif device is None:
                torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def cuda_peak_mb(device: Any = None) -> Any:
    try:
        import torch

        if not torch.cuda.is_available():
            return "na"
        value = torch.cuda.max_memory_allocated(device) if device is not None else torch.cuda.max_memory_allocated()
        return round(value / (1024**2), 2)
    except Exception:
        return "not_available"


def process_memory_mb() -> float:
    return psutil.Process().memory_info().rss / (1024**2)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[int(index)]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def stats(values: list[float], ndigits: int = 4) -> dict[str, Any]:
    clean = [float(v) for v in values if v is not None and not pd_isna(v)]
    if not clean:
        return {
            "media": "",
            "mediana": "",
            "desvio": "",
            "min": "",
            "max": "",
            "p95": "",
        }
    mean = sum(clean) / len(clean)
    variance = sum((v - mean) ** 2 for v in clean) / (len(clean) - 1) if len(clean) > 1 else 0.0
    return {
        "media": round(mean, ndigits),
        "mediana": round(percentile(clean, 0.5), ndigits),
        "desvio": round(math.sqrt(variance), ndigits),
        "min": round(min(clean), ndigits),
        "max": round(max(clean), ndigits),
        "p95": round(percentile(clean, 0.95), ndigits),
    }


def timing_stats_ms(seconds: list[float]) -> dict[str, Any]:
    values = [v * 1000 for v in seconds]
    s = stats(values)
    return {
        "tempo_medio_ms": s["media"],
        "tempo_mediana_ms": s["mediana"],
        "tempo_desvio_ms": s["desvio"],
        "tempo_min_ms": s["min"],
        "tempo_max_ms": s["max"],
        "tempo_p95_ms": s["p95"],
    }


def calculate_speedups(rows: list[dict[str, Any]], key_fields: list[str]) -> list[dict[str, Any]]:
    by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(row.get(field) for field in key_fields)
        by_key.setdefault(key, {})[str(row.get("hardware", "")).upper()] = row

    output = []
    for key, group in by_key.items():
        cpu = group.get("CPU")
        cuda = group.get("CUDA")
        if not cpu or not cuda:
            continue
        try:
            cpu_ms = float(cpu.get("tempo_medio_ms"))
            cuda_ms = float(cuda.get("tempo_medio_ms"))
            speedup = cpu_ms / cuda_ms if cuda_ms > 0 else ""
        except Exception:
            speedup = ""
        item = {field: value for field, value in zip(key_fields, key)}
        item.update({"tempo_cpu_ms": cpu.get("tempo_medio_ms"), "tempo_cuda_ms": cuda.get("tempo_medio_ms"), "speedup_cuda_vs_cpu": speedup})
        output.append(item)
    return output


def throughput_items_per_second(batch: Any, mean_ms: Any) -> Any:
    try:
        b = float(batch)
        ms = float(mean_ms)
        return round(b / (ms / 1000), 4) if ms > 0 else ""
    except Exception:
        return ""


def write_csv(path: Path, rows: list[dict[str, Any]], preferred_fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    if preferred_fields:
        fields.extend([field for field in preferred_fields if field not in fields])
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def run_command(args: list[str], timeout: int = 10) -> tuple[int, str, str]:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except FileNotFoundError as exc:
        return 127, "", str(exc)
    except subprocess.TimeoutExpired as exc:
        return 124, exc.stdout or "", "timeout"
    except Exception as exc:
        return 1, "", str(exc)


def nvidia_smi_snapshot() -> dict[str, Any]:
    query = "utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
    code, stdout, stderr = run_command(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"], timeout=5)
    if code != 0 or not stdout:
        return {"gpu_available": False, "gpu_error": stderr or "nvidia-smi unavailable"}
    first = stdout.splitlines()[0]
    parts = [part.strip() for part in first.split(",")]
    keys = ["gpu_percent", "vram_used_mb", "vram_total_mb", "gpu_temperature_c", "gpu_power_w"]
    data = {"gpu_available": True}
    for key, value in zip(keys, parts):
        try:
            data[key] = float(value)
        except Exception:
            data[key] = "not_available"
    return data


def hardware_snapshot(process: psutil.Process | None = None) -> dict[str, Any]:
    process = process or psutil.Process()
    vm = psutil.virtual_memory()
    snap = {
        "timestamp": now_iso(),
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_system_used_mb": round((vm.total - vm.available) / (1024**2), 2),
        "ram_system_percent": vm.percent,
        "ram_process_mb": round(process.memory_info().rss / (1024**2), 2),
    }
    snap.update(nvidia_smi_snapshot())
    return snap


class HardwareMonitor:
    def __init__(self, interval_s: float = 1.0, context: dict[str, Any] | None = None):
        self.interval_s = max(0.1, float(interval_s))
        self.context = context or {}
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._start = 0.0
        self._process = psutil.Process()

    def __enter__(self) -> "HardwareMonitor":
        psutil.cpu_percent(interval=None)
        self._start = time.perf_counter()
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = hardware_snapshot(self._process)
            sample.update(self.context)
            sample["tempo_relativo_s"] = round(time.perf_counter() - self._start, 4)
            self.samples.append(sample)
            self._stop.wait(self.interval_s)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(1.0, self.interval_s * 2))


def summarize_hardware(samples: list[dict[str, Any]]) -> dict[str, Any]:
    def collect(key: str) -> list[float]:
        output = []
        for sample in samples:
            value = sample.get(key)
            if isinstance(value, (int, float)):
                output.append(float(value))
        return output

    result: dict[str, Any] = {}
    mappings = {
        "cpu_percent": "cpu",
        "ram_process_mb": "ram_processo_mb",
        "ram_system_used_mb": "ram_sistema_mb",
        "gpu_percent": "gpu",
        "vram_used_mb": "vram_mb",
        "gpu_temperature_c": "temperatura_gpu_c",
        "gpu_power_w": "potencia_gpu_w",
    }
    for source, prefix in mappings.items():
        values = collect(source)
        s = stats(values, ndigits=2)
        result[f"{prefix}_media"] = s["media"] if values else "not_available"
        result[f"{prefix}_pico"] = s["max"] if values else "not_available"
    return result


def infer_compute_path(hw: dict[str, Any]) -> str:
    gpu = hw.get("gpu_media")
    vram = hw.get("vram_mb_pico")
    cpu = hw.get("cpu_media")
    try:
        gpu_v = float(gpu)
    except Exception:
        gpu_v = 0.0
    try:
        vram_v = float(vram)
    except Exception:
        vram_v = 0.0
    try:
        cpu_v = float(cpu)
    except Exception:
        cpu_v = 0.0
    if gpu_v >= 20 or vram_v > 500:
        return "GPU ou misto"
    if cpu_v >= 20:
        return "principalmente CPU"
    return "indisponível ou baixa utilização observada"


def pd_isna(value: Any) -> bool:
    try:
        return bool(value != value)
    except Exception:
        return False
