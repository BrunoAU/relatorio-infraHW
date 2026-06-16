import json
import subprocess
import sys
import time
from pathlib import Path

from bootstrap import ensure_local_deps

ensure_local_deps()

from common import LOGS, RAW, ensure_dirs, now_iso, write_json

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

STEPS = [
    ("coletar_sistema", "coletar_sistema.py"),
    ("benchmark_numpy_cpu", "benchmark_numpy_cpu.py"),
    ("benchmark_torch_cpu_gpu", "benchmark_torch_cpu_gpu.py"),
    ("benchmark_limite_gpu", "benchmark_limite_gpu.py"),
    ("benchmark_cnn_sintetico", "benchmark_cnn_sintetico.py"),
    ("benchmark_modelos_reais", "benchmark_modelos_reais.py"),
    ("benchmark_ollama_llms", "benchmark_ollama_llms.py"),
    ("analisar_resultados", "analisar_resultados.py"),
    ("gerar_graficos", "gerar_graficos.py"),
]


def can_reuse_ollama_outputs():
    required = [
        RAW / "benchmark_ollama_llms_execucoes.csv",
        RAW / "benchmark_ollama_llms.csv",
        RAW / "ollama_modelos_resumo.csv",
        RAW / "ollama_por_tarefa.csv",
        RAW / "ollama_tokens_por_segundo.csv",
        RAW / "ollama_uso_hardware.csv",
        RAW / "ollama_viabilidade_pratica.csv",
    ]
    for path in required:
        if not path.exists() or path.stat().st_size <= 32:
            return False
    try:
        sample = (RAW / "ollama_modelos_resumo.csv").read_text(encoding="utf-8")
    except Exception:
        return False
    blockers = ["ollama_unavailable", "config_disabled", "Modelo nao instalado"]
    return not any(token in sample for token in blockers)


def run_step(step_name, script_name):
    if script_name == "benchmark_ollama_llms.py" and can_reuse_ollama_outputs():
        return {
            "etapa": step_name,
            "script": script_name,
            "status": "ok",
            "duracao_s": 0.0,
            "returncode": 0,
            "stdout_resumo": "Benchmark Ollama reutilizado a partir dos CSVs existentes em dados/raw.",
            "stderr_resumo": "",
        }
    script_path = SCRIPTS / script_name
    start = time.perf_counter()
    result = subprocess.run([sys.executable, str(script_path)], cwd=ROOT, capture_output=True, text=True)
    duration = round(time.perf_counter() - start, 4)
    status = "ok" if result.returncode == 0 else "erro"
    return {
        "etapa": step_name,
        "script": script_name,
        "status": status,
        "duracao_s": duration,
        "returncode": result.returncode,
        "stdout_resumo": result.stdout[-1500:],
        "stderr_resumo": result.stderr[-1500:],
    }


def generated_files():
    files = []
    for folder in [ROOT / "dados" / "raw", ROOT / "dados" / "processed", ROOT / "resultados" / "tabelas", ROOT / "resultados" / "graficos", ROOT / "resultados" / "logs"]:
        if folder.exists():
            for path in folder.iterdir():
                if path.is_file():
                    files.append(str(path.relative_to(ROOT)))
    return sorted(files)


def main():
    ensure_dirs()
    steps = []
    for step_name, script_name in STEPS:
        print(f"Executando {script_name}...")
        step = run_step(step_name, script_name)
        steps.append(step)
        print(f"{script_name}: {step['status']} em {step['duracao_s']}s")

    system_info = {}
    system_path = RAW / "system_info.json"
    if system_path.exists():
        try:
            system_info = json.loads(system_path.read_text(encoding="utf-8"))
        except Exception:
            system_info = {"status": "nao_lido"}

    manifest = {
        "timestamp": now_iso(),
        "scripts_executados": [script for _, script in STEPS],
        "status_por_etapa": steps,
        "duracao_total_s": round(sum(item["duracao_s"] for item in steps), 4),
        "arquivos_gerados": generated_files(),
        "erros_ou_skips": [item for item in steps if item["status"] != "ok"],
        "resumo_ambiente": {
            "python": system_info.get("python", {}),
            "sistema_operacional": system_info.get("sistema_operacional", {}),
            "cpu": system_info.get("cpu", {}),
            "gpu": system_info.get("gpu", []),
            "torch": system_info.get("torch", {}),
        },
    }
    write_json(LOGS / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()
