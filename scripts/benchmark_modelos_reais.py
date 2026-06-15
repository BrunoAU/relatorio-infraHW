from pathlib import Path
import argparse
import csv
import statistics
import time

import psutil
import torch
import torch.nn as nn
import torchvision.models as models


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "dados" / "raw"

BATCHES = [1, 8, 16, 32]
REPETICOES = 10
AQUECIMENTOS = 2
ALTURA = 224
LARGURA = 224
CANAIS = 3


def memoria_ram_mb():
    processo = psutil.Process()
    return processo.memory_info().rss / (1024 ** 2)


def sincronizar(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def limpar_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def criar_dispositivos():
    dispositivos = [torch.device("cpu")]

    if torch.cuda.is_available():
        dispositivos.append(torch.device("cuda"))

    return dispositivos


def criar_modelo(nome_modelo, pretrained):
    if nome_modelo == "mobilenet_v2":
        if pretrained:
            pesos = models.MobileNet_V2_Weights.DEFAULT
            modelo = models.mobilenet_v2(weights=pesos)
        else:
            modelo = models.mobilenet_v2(weights=None)

        return modelo

    if nome_modelo == "resnet18":
        if pretrained:
            pesos = models.ResNet18_Weights.DEFAULT
            modelo = models.resnet18(weights=pesos)
        else:
            modelo = models.resnet18(weights=None)

        return modelo

    raise ValueError(f"Modelo não suportado: {nome_modelo}")


def executar_inferencia(nome_modelo, device, batch, pretrained):
    if device.type == "cuda":
        limpar_cuda()

    modelo = criar_modelo(nome_modelo, pretrained).to(device)
    modelo.eval()

    entrada = torch.randn((batch, CANAIS, ALTURA, LARGURA), device=device)

    with torch.no_grad():
        for _ in range(AQUECIMENTOS):
            saida = modelo(entrada)
            sincronizar(device)

    tempos = []
    memoria_inicio = memoria_ram_mb()

    with torch.no_grad():
        for repeticao in range(1, REPETICOES + 1):
            if device.type == "cuda":
                limpar_cuda()

            inicio = time.perf_counter()
            saida = modelo(entrada)
            sincronizar(device)
            fim = time.perf_counter()

            tempo_ms = (fim - inicio) * 1000

            if device.type == "cuda":
                gpu_peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            else:
                gpu_peak_mb = 0.0

            tempos.append({
                "teste": "torch_modelo_real_inferencia",
                "modelo": nome_modelo,
                "pretrained": pretrained,
                "hardware": "CUDA" if device.type == "cuda" else "CPU",
                "tamanho": f"{CANAIS}x{ALTURA}x{LARGURA}",
                "batch": batch,
                "repeticao": repeticao,
                "tempo_ms": tempo_ms,
                "memoria_ram_mb": memoria_ram_mb(),
                "gpu_peak_mb": gpu_peak_mb,
                "status": "ok",
                "observacao": "inferencia com arquitetura real de visao computacional"
            })

    memoria_fim = memoria_ram_mb()
    tempo_lista = [linha["tempo_ms"] for linha in tempos]
    gpu_lista = [linha["gpu_peak_mb"] for linha in tempos]

    resumo = {
        "teste": "torch_modelo_real_inferencia",
        "modelo": nome_modelo,
        "pretrained": pretrained,
        "hardware": "CUDA" if device.type == "cuda" else "CPU",
        "tamanho": f"{CANAIS}x{ALTURA}x{LARGURA}",
        "batch": batch,
        "repeticoes": REPETICOES,
        "tempo_medio_ms": statistics.mean(tempo_lista),
        "tempo_desvio_ms": statistics.stdev(tempo_lista) if len(tempo_lista) > 1 else 0.0,
        "tempo_min_ms": min(tempo_lista),
        "tempo_max_ms": max(tempo_lista),
        "memoria_inicio_mb": memoria_inicio,
        "memoria_fim_mb": memoria_fim,
        "gpu_peak_mb": max(gpu_lista),
        "status": "ok",
        "observacao": "inferencia com arquitetura real de visao computacional"
    }

    del modelo
    del entrada
    del saida

    if device.type == "cuda":
        limpar_cuda()

    return resumo, tempos


def salvar_csv(path, linhas, campos):
    with open(path, "w", newline="", encoding="utf-8") as arquivo:
        writer = csv.DictWriter(arquivo, fieldnames=campos)
        writer.writeheader()
        writer.writerows(linhas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Baixa e usa pesos pre-treinados do torchvision. Requer internet na primeira execução."
    )
    args = parser.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)

    print("Benchmark com modelos reais de visão computacional")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA disponível: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU detectada: {torch.cuda.get_device_name(0)}")

    if args.pretrained:
        print("Modo pretrained ativado. O torchvision pode baixar pesos na primeira execução.")
    else:
        print("Modo padrão: arquiteturas reais sem baixar pesos. Não depende de internet.")

    modelos = ["mobilenet_v2", "resnet18"]
    dispositivos = criar_dispositivos()

    resumos = []
    execucoes = []

    for nome_modelo in modelos:
        for device in dispositivos:
            for batch in BATCHES:
                hardware = "CUDA" if device.type == "cuda" else "CPU"
                print(f"\nRodando {nome_modelo} em {hardware}, batch {batch}...")

                try:
                    resumo, tempos = executar_inferencia(nome_modelo, device, batch, args.pretrained)

                    resumos.append(resumo)
                    execucoes.extend(tempos)

                    print(
                        f"OK | média: {resumo['tempo_medio_ms']:.4f} ms | "
                        f"desvio: {resumo['tempo_desvio_ms']:.4f} ms | "
                        f"VRAM pico: {resumo['gpu_peak_mb']:.2f} MB"
                    )

                except RuntimeError as erro:
                    mensagem = str(erro).replace("\n", " ")

                    if device.type == "cuda":
                        limpar_cuda()

                    resumo = {
                        "teste": "torch_modelo_real_inferencia",
                        "modelo": nome_modelo,
                        "pretrained": args.pretrained,
                        "hardware": hardware,
                        "tamanho": f"{CANAIS}x{ALTURA}x{LARGURA}",
                        "batch": batch,
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

                    resumos.append(resumo)
                    print(f"ERRO | {resumo['observacao']}")

    resumo_path = RAW / "benchmark_modelos_reais.csv"
    execucoes_path = RAW / "benchmark_modelos_reais_execucoes.csv"

    resumo_campos = [
        "teste",
        "modelo",
        "pretrained",
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
        "modelo",
        "pretrained",
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
