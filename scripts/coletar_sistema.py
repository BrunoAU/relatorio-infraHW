import json
import os
import platform
import subprocess
from datetime import datetime
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dados" / "raw" / "system_info.json"


def run_command(command):
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            shell=True,
            timeout=10,
        )

        if result.returncode != 0:
            return None

        return result.stdout.strip()
    except Exception:
        return None


def get_gpu_info():
    gpu_info = []

    nvidia = run_command("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader")
    if nvidia:
        for line in nvidia.splitlines():
            parts = [part.strip() for part in line.split(",")]
            gpu_info.append(
                {
                    "provider": "nvidia-smi",
                    "name": parts[0] if len(parts) > 0 else None,
                    "memory_total": parts[1] if len(parts) > 1 else None,
                    "driver_version": parts[2] if len(parts) > 2 else None,
                }
            )

    if len(gpu_info) == 0 and platform.system().lower() == "windows":
        wmic = run_command("wmic path win32_VideoController get name,AdapterRAM,DriverVersion /format:csv")
        if wmic:
            gpu_info.append({"provider": "wmic", "raw": wmic})

    if len(gpu_info) == 0:
        gpu_info.append({"provider": "not_detected", "name": None})

    return gpu_info


def get_torch_info():
    info = {
        "installed": False,
        "version": None,
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_devices": [],
    }

    try:
        import torch

        info["installed"] = True
        info["version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0

        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                info["cuda_devices"].append(
                    {
                        "index": index,
                        "name": torch.cuda.get_device_name(index),
                        "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
                    }
                )
    except Exception as exc:
        info["error"] = str(exc)

    return info


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        },
        "cpu": {
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_threads": psutil.cpu_count(logical=True),
            "frequency_mhz": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
        },
        "memory": {
            "total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        },
        "gpu": get_gpu_info(),
        "torch": get_torch_info(),
        "npu_note": "NPU não é medida automaticamente por este script. Registrar manualmente se houver suporte real por framework/driver.",
        "environment": {
            "cwd": os.getcwd(),
        },
    }

    with OUT.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

    print(f"Informações do sistema salvas em: {OUT}")


if __name__ == "__main__":
    main()
