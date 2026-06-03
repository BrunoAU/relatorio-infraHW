from pathlib import Path
import csv
import statistics
import time

import psutil
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "dados" / "raw"

BATCHES = [1, 8, 16, 32]
REPETICOES = 10
AQUECIMENTOS = 2
ALTURA = 224
LARGURA = 224
CANAIS = 3


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
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.rede(x)


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


def executar_cnn(device, batch):
    if device.type == "cuda":
        limpar_cuda()

    modelo = CNNSintetica().to(device)
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
                "teste": "torch_cnn_sintetico",
                "hardware": "CUDA" if device.type == "cuda" else "CPU",
                "tamanho": f"{CANAIS}x{ALTURA}x{LARGURA}",
                "batch": batch,
                "repeticao": repeticao,
                "tempo_ms": tempo_ms,
                "memoria_ram_mb": memoria_ram_mb(),
                "gpu_peak_mb": gpu_peak_mb,
                "status": "ok",
                "observacao": "CNN sintetico com entrada de imagem"
            })

    memoria_fim = memoria_ram_mb()
    tempo_lista = [linha["tempo_ms"] for linha in tempos]
    gpu_lista = [linha["gpu_peak_mb"] for linha in tempos]

    resumo = {
        "teste": "torch_cnn_sintetico",
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
        "observacao": "CNN sintetico com entrada de imagem"
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
    RAW.mkdir(parents=True, exist_ok=True)

    print("Benchmark CNN sintético com entrada de imagem")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA disponível: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU detectada: {torch.cuda.get_device_name(0)}")

    dispositivos = criar_dispositivos()

    resumos = []
    execucoes = []

    for device in dispositivos:
        for batch in BATCHES:
            hardware = "CUDA" if device.type == "cuda" else "CPU"
            print(f"\nRodando CNN em {hardware}, batch {batch}...")

            try:
                resumo, tempos = executar_cnn(device, batch)

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
                    "teste": "torch_cnn_sintetico",
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

    resumo_path = RAW / "benchmark_cnn_sintetico.csv"
    execucoes_path = RAW / "benchmark_cnn_sintetico_execucoes.csv"

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
