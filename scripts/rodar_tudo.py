import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

STEPS = [
    "coletar_sistema.py",
    "benchmark_numpy_cpu.py",
    "benchmark_torch_cpu_gpu.py",
    "gerar_graficos.py",
]


def run_step(script_name):
    script_path = SCRIPTS / script_name
    print(f"\nExecutando {script_name}...")
    result = subprocess.run([sys.executable, str(script_path)], cwd=ROOT)

    if result.returncode != 0:
        raise RuntimeError(f"Falha ao executar {script_name}")


def main():
    for step in STEPS:
        run_step(step)

    print("\nColeta concluída. Verifique as pastas dados/ e resultados/.")


if __name__ == "__main__":
    main()
