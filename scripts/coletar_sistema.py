import importlib.metadata
import platform
import sys

from bootstrap import ensure_local_deps

ensure_local_deps()

import psutil
from common import LOGS, RAW, ensure_dirs, now_iso, run_command, timestamp_slug, write_json


def get_gpu_info():
    query = "name,memory.total,driver_version"
    code, stdout, stderr = run_command(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"], timeout=10)
    if code != 0 or not stdout:
        return [{"provider": "nvidia-smi", "status": "not_available", "observacao": stderr or "nvidia-smi indisponível"}]

    gpus = []
    for line in stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        gpus.append(
            {
                "provider": "nvidia-smi",
                "status": "ok",
                "name": parts[0] if len(parts) > 0 else "not_available",
                "memory_total": parts[1] if len(parts) > 1 else "not_available",
                "driver_version": parts[2] if len(parts) > 2 else "not_available",
            }
        )
    return gpus


def get_torch_info():
    info = {
        "installed": False,
        "version": "not_available",
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_devices": [],
        "cuda_version_pytorch": "not_available",
    }
    try:
        import torch

        info["installed"] = True
        info["version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version_pytorch"] = torch.version.cuda or "not_available"
        info["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                info["cuda_devices"].append(
                    {
                        "index": index,
                        "name": torch.cuda.get_device_name(index),
                        "vram_total_gb": round(props.total_memory / (1024**3), 3),
                    }
                )
    except Exception as exc:
        info["error"] = str(exc)
    return info


def package_versions():
    packages = ["numpy", "pandas", "psutil", "matplotlib", "torch", "torchvision", "requests"]
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    return versions


def main():
    ensure_dirs()
    vm = psutil.virtual_memory()
    data = {
        "timestamp": now_iso(),
        "sistema_operacional": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "cpu": {
            "nome": platform.processor() or "not_available",
            "nucleos_fisicos": psutil.cpu_count(logical=False),
            "threads_logicas": psutil.cpu_count(logical=True),
            "frequencia_mhz": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else "not_available",
        },
        "ram": {"total_gb": round(vm.total / (1024**3), 3)},
        "gpu": get_gpu_info(),
        "torch": get_torch_info(),
        "pacotes_principais": package_versions(),
        "npu": {
            "medida_automaticamente": False,
            "observacao": "NPU não é medida automaticamente neste projeto. O estudo experimental atual cobre CPU e CUDA/GPU quando disponíveis.",
        },
    }
    raw_path = RAW / "system_info.json"
    log_path = LOGS / f"system_info_{timestamp_slug()}.json"
    write_json(raw_path, data)
    write_json(log_path, data)
    print(f"Informações do sistema salvas em {raw_path} e {log_path}")


if __name__ == "__main__":
    main()
