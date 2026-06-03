from pathlib import Path
import csv
import statistics
import time

import psutil
import torch


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "dados" / "raw"

TAMANHOS = [2560, 3072, 4096]
REPETICOES = 10
AQUECIMENTOS = 2


def memoria_ram_mb():
    processo = psutil.Process()
    return processo.memory_info().rss / (1024 ** 2)


def sincronizar_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def limpar_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def executar_matmul_cuda(tamanho):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA não está disponível neste ambiente.")

    device = torch.device("cuda")

    limpar_cuda()

    try:
        a = torch.randn((tamanho, tamanho), device=device)
        b = torch.randn((tamanho, tamanho), device=device)

        for _ in range(AQUECIMENTOS):
            c = torch.matmul(a, b)
            sincronizar_cuda()

        tempos = []
        memoria_inicio = memoria_ram_mb()

        for repeticao in range(1, REPETICOES + 1):
            limpar_cuda()

            inicio = time.perf_counter()
            c = torch.matmul(a, b)
            sincronizar_cuda()
            fim = time.perf_counter()

            tempo_ms = (fim - inicio) * 1000
            gpu_peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

            tempos.append({
                "teste": "limite_matmul_cuda",
                "hardware": "CUDA",
                "tamanho": tamanho,
                "batch": "",
                "repeticao": repeticao,
                "tempo_ms": tempo_ms,
                "memoria_ram_mb": memoria_ram_mb(),
                "gpu_peak_mb": gpu_peak_mb,
                "status": "ok",
                "observacao": "teste de limite de matriz em GPU"
            })

        memoria_fim = memoria_ram_mb()
        tempo_lista = [linha["tempo_ms"] for linha in tempos]
        gpu_lista = [linha["gpu_peak_mb"] for linha in tempos]

        resumo = {
            "teste": "limite_matmul_cuda",
            "hardware": "CUDA",
            "tamanho": tamanho,
            "batch": "",
            "repeticoes": REPETICOES,
            "tempo_medio_ms": statistics.mean(tempo_lista),
            "tempo_desvio_ms": statistics.stdev(tempo_lista) if len(tempo_lista) > 1 else 0.0,
            "tempo_min_ms": min(tempo_lista),
            "tempo_max_ms": max(tempo_lista),
            "memoria_inicio_mb": memoria_inicio,
            "memoria_fim_mb": memoria_fim,
            "gpu_peak_mb": max(gpu_lista),
            "status": "ok",
            "observacao": "teste de limite de matriz em GPU"
        }

        del a
        del b
        del c
        limpar_cuda()

        return resumo, tempos

    except RuntimeError as erro:
        limpar_cuda()

        mensagem = str(erro).replace("\n", " ")
        resumo = {
            "teste": "limite_matmul_cuda",
            "hardware": "CUDA",
            "tamanho": tamanho,
            "batch": "",
            "repeticoes": 0,
            "tempo_medio_ms": "",
            "tempo_desvio_ms": "",
            "tempo_min_ms": "",
            "tempo_max_ms": "",
            "memoria_inicio_mb": "",
            "memoria_fim_mb": "",
            "gpu_peak_mb": "",
            "status": "erro",
            "observacao": mensagem[:250]
        }

        return resumo, []


def salvar_csv(path, linhas, campos):
    with open(path, "w", newline="", encoding="utf-8") as arquivo:
        writer = csv.DictWriter(arquivo, fieldnames=campos)
        writer.writeheader()
        writer.writerows(linhas)


def main():
    RAW.mkdir(parents=True, exist_ok=True)

    print("Teste de limite da GPU em multiplicação de matrizes")
    print(f"CUDA disponível: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA indisponível. O teste será encerrado.")
        return

    resumos = []
    execucoes = []

    for tamanho in TAMANHOS:
        print(f"\nRodando teste CUDA com matriz {tamanho}x{tamanho}...")

        resumo, tempos = executar_matmul_cuda(tamanho)

        resumos.append(resumo)
        execucoes.extend(tempos)

        if resumo["status"] == "ok":
            print(
                f"OK | média: {resumo['tempo_medio_ms']:.4f} ms | "
                f"desvio: {resumo['tempo_desvio_ms']:.4f} ms | "
                f"VRAM pico: {resumo['gpu_peak_mb']:.2f} MB"
            )
        else:
            print(f"ERRO | {resumo['observacao']}")

    resumo_path = RAW / "benchmark_limite_gpu.csv"
    execucoes_path = RAW / "benchmark_limite_gpu_execucoes.csv"

    resumo_campos = [
        "teste",
        "hardware",
        "tamanho",
        "batch",
        "repeticoes",
        "tempo_medio_ms",
        "tempo_desvio_ms",
        "tempo_min_ms",
        "tempo_max_ms",
        "memoria_inicio_mb",
        "memoria_fim_mb",
        "gpu_peak_mb",
        "status",
        "observacao"
    ]

    execucoes_campos = [
        "teste",
        "hardware",
        "tamanho",
        "batch",
        "repeticao",
        "tempo_ms",
        "memoria_ram_mb",
        "gpu_peak_mb",
        "status",
        "observacao"
    ]

    salvar_csv(resumo_path, resumos, resumo_campos)
    salvar_csv(execucoes_path, execucoes, execucoes_campos)

    print(f"\nResumo salvo em: {resumo_path}")
    print(f"Execuções individuais salvas em: {execucoes_path}")


if __name__ == "__main__":
    main()
