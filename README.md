# Relatório de Laboratório — Infraestrutura de Hardware

> Disciplina: Infraestrutura de Hardware | Prof. Ronierison Maciel
> Dupla: Bruno Augusto da Rocha Leite Filho
> Máquina utilizada: Notebook pessoal

## Bloco 0 — Hipóteses iniciais

Antes das medições, eu colocaria assim:

1. Desempenho de CPU em workload multithread
2. Largura de banda e latência da RAM
3. Desempenho do armazenamento em leitura sequencial e aleatória
4. Eficiência da hierarquia de cache
5. Capacidade do barramento/PCIe e impacto no I/O

## 1. Anatomia da CPU (Bloco 1)

### Configuração identificada

| Métrica | Valor |
|--------|-------|
| Modelo da CPU | Intel(R) Core(TM) 7 150U |
| Arquitetura | x86_64 |
| Núcleos físicos / Threads lógicas | 6 / 12 |
| Frequência base (GHz) | Não registrada diretamente nas evidências preservadas |
| Frequência turbo observada (GHz) | 2.611 |
| Cache L1 / L2 / L3 | L1d: 288 KiB, L1i: 192 KiB / 7.5 MiB / 12 MiB |
| Instruction sets relevantes | SSE, SSE2, SSE4.1, SSE4.2, AVX, AVX2, FMA, AES, AVX-VNNI |

### Benchmark de paralelismo

Como não há resultado de Cinebench nas capturas preservadas, os campos abaixo permanecem em branco se o professor exigir especificamente essa ferramenta.

| Métrica | Valor |
|--------|-------|
| Cinebench Single Core | Em branco |
| Cinebench Multi Core | Em branco |
| Fator de escala (Multi/Single) | 5.38 |
| Fator ideal (= nº de threads) | 12 |
| Eficiência paralela (%) | 44.8 |
| Speedup observado em `teste_paralelismo` | 1.88x no melhor caso observado |

### Observações complementares

A CPU identificada foi a Intel(R) Core(TM) 7 150U, com 6 núcleos físicos e 12 threads lógicas, o que indica 2 threads por núcleo.

A frequência base não apareceu explicitamente nas capturas, então esse campo foi mantido sem valor numérico confirmado.

Na hierarquia de cache, foi possível observar:

- L1d: 288 KiB no total
- L1i: 192 KiB no total
- L2: 7.5 MiB
- L3: 12 MiB

Pela topologia mostrada, isso corresponde a:

- 48 KiB de L1d por núcleo
- 32 KiB de L1i por núcleo
- 1280 KiB de L2 por núcleo
- 12 MiB de L3 compartilhado

As extensões relevantes observadas foram SSE, SSE2, SSE4.1, SSE4.2, AVX, AVX2, FMA, AES e AVX-VNNI.

Mesmo sem Cinebench, foi possível calcular o fator de escala e a eficiência paralela com base nos resultados do sysbench:

- single core = 2145.51 events/s
- multi core = 11530.15 events/s
- fator de escala = 11530.15 / 2145.51 ≈ 5.38
- eficiência paralela = 5.38 / 12 × 100 ≈ 44.8%

No teste em Python, o melhor ganho ocorreu com 4 threads:

- 1 thread: 0.90 s
- 2 threads: 0.63 s
- 4 threads: 0.48 s
- 12 threads: 0.50 s

Assim, o melhor speedup observado foi 0.90 / 0.48 ≈ 1.88x. Isso mostra que o paralelismo melhorou o desempenho até certo ponto, mas depois houve saturação.

### Respostas às perguntas reflexivas (Bloco 1)

**Por que o fator de escala é menor que o número de threads?**

Porque o paralelismo real não escala de forma perfeita. Embora a máquina tenha 12 threads lógicas, o ganho observado foi de 5.38x, e não de 12x. Isso acontece por causa de overhead de criação e sincronização de threads ou processos, disputa por cache e memória, partes do programa que continuam seriais e pelo fato de que threads lógicas não equivalem a núcleos físicos totalmente independentes.

**Por que a carga “pula” entre núcleos no teste single-core?**

Porque o sistema operacional pode migrar a thread entre núcleos diferentes para balancear carga, temperatura e consumo de energia. Assim, mesmo em um teste single-core, a execução continua sendo de uma única thread, mas essa thread pode aparecer em núcleos diferentes ao longo do tempo.

**Pipeline com N estágios — ganho ideal e o que quebra?**

O ganho ideal seria aproximadamente N vezes no throughput, porque várias instruções poderiam estar em execução ao mesmo tempo em estágios diferentes do pipeline.

Na prática, esse ganho é reduzido por dependências entre instruções, desvios e branches, stalls e bolhas no pipeline, latência de memória e cache, e desequilíbrio entre os estágios.

**Tradução entre GHz e pontos via CPI?**

A frequência em GHz, sozinha, não determina o desempenho final. O desempenho depende também do CPI.

De forma simplificada:

- desempenho ≈ frequência / CPI
- tempo de execução = instruções × CPI / frequência

Isso significa que mais GHz ajuda, mas não garante sozinho mais desempenho. Se o CPI for alto, houver stalls, cache misses ou baixa eficiência paralela, o desempenho observado fica menor do que o clock isoladamente sugere.

## 2. Hierarquia de Memória (Bloco 2)

### Latência por nível

| Nível | Tamanho | Latência (ns) | Banda (GB/s) |
|------|---------|---------------|--------------|
| L1 | L1d: 288 KiB, L1i: 192 KiB | ~1.0 | ~50.0 |
| L2 | 7.5 MiB | ~7.1 | ~25.0 |
| L3 | 12 MiB | ~45.7 | ~15.0 |
| RAM | 15 GB | ~132.6 | ~13.0 |

### Vazão com `mbw -t0`

| Tamanho do array | Vazão (`mbw -t0`) |
|------------------|-------------------|
| 16 MiB | 7453.995 MiB/s |
| 128 MiB | 8314.928 MiB/s |
| 1024 MiB | 6272.027 MiB/s |

### Experimento de localidade

| `teste_localidade.py` | Tempo (s) |
|-----------------------|-----------|
| Loop A (linhas) | 5.68 |
| Loop B (colunas) | 6.41 |
| Razão B/A | 1.13 |

### Observações do bloco

Os valores de latência e banda foram preenchidos por estimação empírica, com base na análise dos resultados obtidos em tinymembench e mbw, combinados com o comportamento esperado da hierarquia de memória. No ambiente WSL2, não foi possível medir diretamente todos os níveis com contadores de hardware, então os valores foram inferidos a partir das transições observadas e de estimativas gerais coerentes com os dados experimentais.

A estimativa de L2 (~7.1 ns), L3 (~45.7 ns) e RAM (~132.6 ns) foi baseada principalmente nas mudanças de latência observadas no tinymembench conforme o tamanho do bloco aumentava. Já a latência de L1 não foi isolada diretamente pelo benchmark, então foi representada por uma estimativa geral de aproximadamente 1.0 ns. As bandas de L1, L2 e L3 também foram estimadas de forma aproximada, com base em proporção esperada entre níveis e no comportamento observado da banda da memória principal.

No mbw, a vazão ficou em 7453.995 MiB/s para 16 MiB, subiu para 8314.928 MiB/s em 128 MiB e caiu para 6272.027 MiB/s em 1024 MiB. Isso sugere que, à medida que o tamanho do array cresce, o sistema passa a depender mais da memória principal e menos dos níveis superiores da hierarquia, o que reduz a taxa de cópia observada.

A matriz usada no experimento de localidade foi de 6000 × 6000 elementos `float64`, ocupando aproximadamente 274.7 MB em memória, com 36.000.000 operações por loop.

A partir dos tempos medidos:

- acesso linha por linha: 5.68 s
- acesso coluna por coluna: 6.41 s
- razão B/A: 1.13x

Isso mostra que o padrão sequencial foi mais rápido. A diferença observada no ambiente testado não foi extrema, mas ainda confirma a ideia principal do experimento: o padrão de acesso à memória afeta o desempenho mesmo quando a quantidade de operações é a mesma.

### Respostas às perguntas reflexivas (Bloco 2)

**Por que a latência cresce em ordens de grandeza?**

Porque, à medida que descemos na hierarquia, a memória fica maior, mais distante do núcleo e mais cara de acessar em termos de tempo. Registradores e caches pequenos ficam muito próximos da CPU e usam circuitos extremamente rápidos. Já níveis mais baixos, como RAM, priorizam capacidade e custo, sacrificando velocidade. Por isso a latência não cresce de forma linear, e sim em saltos grandes entre os níveis.

**Vetor de 100 MB vs 200 KB — qual é mais rápido em loop sequencial?**

Na máquina analisada:

- L1 total: L1d 288 KiB + L1i 192 KiB
- L2 total: 7.5 MiB
- L3: 12 MiB
- RAM: 15 GB

Assim:

- um vetor de 100 MB não cabe em L1, L2 nem L3, então depende da RAM;
- um vetor de 200 KB cabe nos níveis superiores da hierarquia, especialmente em L1/L2, dependendo do contexto.

O vetor de 200 KB tende a executar mais rápido em loop sequencial, porque muito mais dados permanecem próximos da CPU, reduzindo o custo de acesso.

**Localidade temporal e espacial — exemplos em código**

Localidade temporal é quando um dado acessado agora tende a ser reutilizado em pouco tempo.

Exemplo em C:

```c
int soma = 0;
for (int i = 0; i < 1000000; i++) {
    soma += x;
}
```

Nesse caso, a variável `x` é reutilizada muitas vezes.

Localidade espacial é quando, ao acessar uma posição de memória, logo depois acessamos posições vizinhas.

Exemplo em C:

```c
for (int i = 0; i < n; i++) {
    soma += vetor[i];
}
```

Nesse caso, os acessos são sequenciais no vetor, favorecendo o cache.

**Mesmo número de operações, tempos diferentes — por quê?**

Porque o custo não está só na operação aritmética, mas principalmente em trazer os dados até a CPU. Os dois loops fazem a mesma quantidade de soma, mas com padrões de acesso diferentes. O acesso linha por linha aproveita melhor a localidade espacial e o cache. O acesso coluna por coluna faz saltos maiores na memória e reduz esse aproveitamento. Isso mostra que, muitas vezes, o gargalo do processador não é calcular, e sim acessar memória de forma eficiente.

**Disputa por L3 e fator de escala paralelo**

Quando vários núcleos ou threads executam ao mesmo tempo, eles passam a disputar recursos compartilhados, especialmente o L3 e a largura de banda da memória. Isso reduz o ganho paralelo real. No experimento do Bloco 1, apesar de a máquina ter 12 threads lógicas, o fator de escala medido foi 5.38x, com eficiência paralela de 44.8%. Isso ajuda a explicar por que o paralelismo observado fica abaixo do ideal: mais threads não significam ganho linear quando há disputa por cache e memória.

## 3. RAM, Armazenamento e Memória Virtual (Bloco 3)

### Hierarquia completa

| Componente | Banda medida | Latência típica |
|------------|--------------|-----------------|
| L1 | ~50.0 GB/s | ~1.0 ns |
| L3 | ~15.0 GB/s | ~45.7 ns |
| RAM | ~13.0 GB/s | ~132.6 ns |
| SSD NVMe SEQ1M | 1461 MiB/s | ~682 µs |
| SSD NVMe RND4K | 16.2 MiB/s | ~238 µs |
| SSD SATA (se houver) | N/D | N/D |
| HDD (se houver) | N/D | N/D |

### Resumo no formato do novo modelo

| Métrica | Valor |
|---------|-------|
| RAM total | 16217288 kB |
| RAM disponível em repouso | 15551772 kB |
| Swap total / em uso | 4.0 GiB / 0 B em repouso |
| `fio` seq read (MB/s) | 1532 MB/s |
| `fio` random read 4k (IOPS) | 4157 IOPS |
| Variação de `si`/`so` durante stress-ng | si = 0, so = 783, com swpd = 1282632 |

**Anote o nome do navegador escolhido para `/proc/$PID/status`:** não houve navegador rodando no momento da coleta; foi usado o processo PID 308 do python3 do sistema

### Memória virtual de um processo escolhido

Processo: PID 308 — `/usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal`

| Métrica | Valor |
|---------|-------|
| Working Set / VmRSS | 21888 kB |
| Private Bytes | N/D |
| Virtual Size / VmSize | 107008 kB |

### Pressão de memória — observações

Ao aplicar pressão de memória, o sistema inicialmente ainda ficou sem usar swap no primeiro teste. Depois, no teste em que a pressão foi mais efetiva, o `vmstat` mostrou `swpd = 1282632`, `so = 783` e `si = 0`. Isso indica que houve uso real de swap e ocorrência de swap-out, ou seja, páginas foram empurradas da RAM para a área de swap. O script Python também mostrou uma inflexão clara no tempo de acesso quando os buffers ficaram grandes: o custo saiu de cerca de 0.49 µs/acesso para 1.12 µs/acesso, confirmando piora perceptível de desempenho sob pressão de memória.

Em repouso, a máquina tinha folga considerável de RAM e swap zerado. Sob pressão de memória, houve uso real de swap, com swap-out observado no `vmstat`. Como não havia Firefox em execução no WSL no momento da coleta, foi usado um processo alternativo do sistema para observar `VmSize` e `VmRSS`.

### Respostas às perguntas reflexivas (Bloco 3)

**Virtual Size > RAM física — o que isso revela?**

Isso revela que o espaço de endereçamento de um processo é virtual, não físico. O endereço que aparece no programa em C não é “um lugar direto na RAM”, mas um endereço virtual que o sistema operacional e a MMU traduzem para páginas físicas quando necessário. Por isso, um processo pode enxergar um espaço virtual maior que a RAM instalada: nem tudo está residente ao mesmo tempo, parte pode nem ter sido materializada ainda, e parte pode estar mapeada em arquivo ou swap.

**Penalidade do swap em ordens de grandeza**

Comparando os números medidos, a RAM teve latência de aproximadamente 132.6 ns, enquanto o armazenamento ficou na faixa de 238 µs no RND4K e 682 µs no acesso sequencial.

Fazendo a comparação:

- 238 µs / 132.6 ns ≈ 1795x
- 682 µs / 132.6 ns ≈ 5143x

Logo, quando o sistema sai da RAM e passa a depender do disco/swap, a penalidade fica em torno de 3 a 4 ordens de grandeza.

**Page fault — minor vs major**

Page fault é o evento em que o processo acessa uma página virtual que não está pronta naquele instante na memória física do processo. Nem toda page fault é ruim.

- Minor page fault: a página não estava mapeada no contexto do processo, mas pode ser resolvida sem buscar dados no disco. Exemplo: página já em RAM, zero-fill de uma página nova ou ajuste de mapeamento.
- Major page fault: exige acesso ao disco, porque a página precisa ser trazida de arquivo ou swap. Essa é a mais cara e a que realmente prejudica desempenho.

**Por que `vm.swappiness=1` em servidores de banco?**

Porque banco de dados sofre muito quando páginas quentes saem da RAM e vão para swap. Em servidor de banco, a prioridade é manter o working set do banco residente em memória e evitar ao máximo paginação para disco. Por isso usa-se `swappiness=1`: o kernel só recorre ao swap em situação bem mais extrema.

Em um desktop comum, o padrão 60 busca um equilíbrio mais geral entre responsividade, cache de arquivos e uso de memória ao longo do tempo. O desktop aceita melhor essa política mais balanceada porque a carga costuma ser mais variada e menos sensível que um banco de dados.

**RND4K vs SEQ1M — por que tão diferente?**

Porque no acesso sequencial com blocos grandes o SSD consegue transferir dados em fluxo contínuo, aproveitando melhor paralelismo interno, pré-busca e throughput do controlador. Já no RND4K o custo passa a ser dominado pela latência de cada operação pequena. Cada acesso aleatório de 4 KB paga overhead de busca, tradução no FTL, controle interno da NAND e pouca oportunidade de streaming. Resultado: o SSD continua rápido, mas a banda despenca porque ele passa a resolver milhares de pedidos pequenos e espalhados, em vez de poucos blocos grandes e contínuos.

**"NVMe é melhor que mais RAM" — refute com dados**

A afirmação está parcialmente errada. O NVMe melhora muito o armazenamento, boot, carregamento e leitura/escrita em disco, mas não substitui RAM.

Nos dados medidos:

- RAM: ~13.0 GB/s e ~132.6 ns
- SSD SEQ1M: 1461 MiB/s e ~682 µs
- SSD RND4K: 16.2 MiB/s e ~238 µs

Ou seja, a RAM ainda é várias vezes mais rápida em banda e milhares de vezes mais rápida em latência. Quando faltou memória e o sistema começou a usar swap, houve `swpd = 1282632` e `so = 783`, mostrando que entrar em memória virtual com disco é caro. Então, um NVMe novo ajuda bastante em tarefas dependentes de armazenamento, mas, quando o problema é working set grande demais, mais RAM continua sendo muito mais importante que trocar só o SSD.

## 4. Barramentos, I/O e Interrupções (Bloco 4)

### PCIe identificado

| Dispositivo | Geração | Largura | Banda teórica | Banda real medida |
|-------------|---------|---------|---------------|-------------------|
| SSD NVMe | PCIe 4.0 | x4 | ~7.88 GB/s | 1461 MiB/s (~1.46 GiB/s) sequencial; 16.2 MiB/s no RND4K |
| GPU | Integrada / sem link PCIe útil exposto para a atividade | N/D | N/D | N/D |
| Rede | N/D no ambiente WSL | N/D | N/D | N/D |

### No formato do novo modelo

| Dispositivo PCIe | LnkCap (geração × largura) | LnkSta |
|------------------|----------------------------|--------|
| Controlador NVMe | PCIe 4.0 x4 (16.0 GT/s x4) | PCIe 4.0 x4 (16.0 GT/s x4) |
| GPU | N/D — GPU integrada / sem link PCIe útil exposto para a atividade | N/D |
| Outro (ex: rede 10G) | N/D no ambiente WSL | N/D |

### Interrupções observadas

**Quais IRQs subiram com cada ação?**

- Mexer mouse: IRQ não identificada no WSL
- Digitar: IRQ não identificada no WSL
- Ping: não houve linha claramente correlacionada; apenas leve variação em IRQ 9 (acpi)
- Cópia de arquivo: não houve linha claramente correlacionada; `virtio1-requests.0` era a principal candidata, mas não mostrou subida visível nas capturas

**Top 3 IRQs durante o `dd`:**

1. IRQ 9 (acpi) — foi a única que mostrou variação perceptível nas capturas
2. IRQ 25 (virtio0-virtqueues) — permaneceu estável em 1057
3. IRQ 28 (virtio1-requests.0) — permaneceu estável em 10

Observação: no WSL/Hyper-V, as interrupções observadas são virtualizadas e a correlação direta entre ação e IRQ ficou limitada. Não houve subida clara das linhas esperadas durante o `dd`.

### Respostas às perguntas reflexivas (Bloco 4)

**PCIe 4.0 x16 vs RAM — onde fica o gargalo numa GPU?**

O gargalo depende de qual etapa estamos olhando. Em uma GPU discreta, o barramento PCIe 4.0 x16 tem banda teórica alta, cerca de 31.5 GB/s, mas ainda fica abaixo de uma RAM DDR4-3200 dual-channel teórica de 51.2 GB/s. Além disso, a VRAM da GPU costuma ter banda muito maior que ambas. Então, em renderização contínua, o gargalo normalmente não é a RAM do sistema nem o PCIe isoladamente, mas sim o fluxo interno da própria GPU e da VRAM. Já quando dados precisam ser trazidos da RAM para a GPU, o PCIe pode virar o limite. No nosso cenário, porém, o primeiro gargalo medido foi ainda antes disso: o SSD, que entregou só ~1.46 GiB/s em leitura sequencial prática, muito abaixo do PCIe 4.0 x4 teórico do NVMe.

**Por que existem interrupções? E por que polling é pior?**

Interrupções existem para que o hardware avise a CPU quando algo importante aconteceu, sem que a CPU precise ficar perguntando o tempo todo. Isso economiza ciclos de processamento. Em vez de a CPU ficar em loop verificando teclado, rede, disco e outros periféricos, ela continua executando trabalho útil e só para quando chega um evento. Polling tende a ser pior porque desperdiça tempo de CPU verificando repetidamente algo que quase sempre não mudou. Isso aumenta consumo, reduz eficiência e piora escalabilidade.

**Quando polling é melhor que interrupção?**

Polling pode ser melhor quando o evento acontece com frequência muito alta e previsível, a ponto de o custo de tratar milhares de interrupções ficar maior que simplesmente consultar o dispositivo em loop controlado. Isso também pode acontecer em cenários de baixíssima latência, drivers de alto desempenho, redes muito rápidas, processamento em lote ou hardware dedicado. Ou seja, interrupção é melhor para eventos esparsos; polling pode vencer quando a taxa de eventos é tão alta que a interrupção vira overhead.

**Cadeia tecla-pressionada → tela**

Quando uma tecla é pressionada, o teclado gera um evento elétrico e o controlador correspondente detecta isso. Esse controlador sinaliza a CPU por meio de uma interrupção. O sistema operacional atende a interrupção, lê o código da tecla, processa o evento no driver e entrega esse dado à pilha de entrada do sistema. A aplicação em foco recebe o evento, atualiza seu estado interno e pede alteração da interface. Depois, o subsistema gráfico redesenha a região necessária, envia os dados ao pipeline gráfico e a imagem atualizada aparece na tela. Resumindo: tecla → controlador → interrupção → kernel/driver → aplicação → subsistema gráfico → tela.

**NVMe 7000 MB/s em PCIe 3.0 x4 — qual o teto real?**

O teto de PCIe 3.0 x4 é cerca de ~3.94 GB/s. Então um SSD anunciado como 7000 MB/s não conseguiria entregar isso em PCIe 3.0 x4, porque o barramento já limitaria o throughput antes do SSD chegar ao valor de marketing. Na prática, o teto real ficaria próximo do limite do barramento, não do valor nominal máximo do SSD.

**DMA — o que é e como reduz carga da CPU?**

DMA é Direct Memory Access. É um mecanismo em que o periférico, ou um controlador associado, transfere dados diretamente entre dispositivo e memória principal sem que a CPU precise copiar byte por byte. A CPU participa mais no início e no fim da operação: configura a transferência, autoriza o processo e depois é avisada quando terminou. Isso reduz muito carga de CPU, melhora throughput e permite que disco, rede e outros dispositivos movimentem grandes volumes de dados com mais eficiência.

### Discussão da atividade 4.3 — RAM vs PCIe no cenário analisado

No cenário analisado, o primeiro gargalo mais provável é o SSD, não o PCIe e nem a RAM. Pelos dados que medimos, a RAM ficou em ~13.0 GB/s, o link do NVMe é PCIe 4.0 x4 teórico de ~7.88 GB/s, mas o SSD na prática entregou só 1461 MiB/s (~1.46 GiB/s) em leitura sequencial. Em leitura aleatória 4K, caiu ainda mais para 16.2 MiB/s. Então, quando o jogo precisa carregar texturas pesadas do armazenamento, o disco é claramente o ponto mais lento do caminho.

Se a textura já saiu do SSD e está em RAM, aí o próximo limite passa a ser a transferência para a GPU pelo barramento. Nesse caso, o PCIe 4.0 x4 ainda é bem mais rápido que o SSD medido, mas continua abaixo da RAM. Então a ordem, no nosso cenário, fica assim:

RAM (~13.0 GB/s) > PCIe 4.0 x4 teórico (~7.88 GB/s) > SSD real (~1.46 GiB/s sequencial)

Agora, durante a renderização contínua, em jogos pesados, o gargalo costuma migrar para a VRAM/banda interna da GPU, porque é ela que alimenta a GPU o tempo todo com texturas, buffers e dados gráficos. Só que aqui existe uma limitação importante: nossa máquina não expôs uma GPU discreta útil para medir esse caminho, então não temos como provar isso experimentalmente neste ambiente.

Conclusão final para o bloco:

- carregamento/streaming de texturas do disco: gargalo principal tende a ser o SSD
- cópia RAM → GPU: o gargalo pode passar a ser o PCIe
- renderização já em andamento: em geral o gargalo tende a ser a VRAM/banda da GPU, mas isso não foi medido diretamente neste ambiente

## 5. Síntese e Resolução do Mistério (Bloco 5)

### Workload escolhido

Workload: `workload_completo.py`

### Observação importante

O script `workload_completo.py` não estava inicialmente disponível na pasta `~/infra_hw/scripts`, embora fosse o workload indicado para a atividade. Depois que o script foi obtido e colocado manualmente na pasta correta, ele passou a ser executado no ambiente virtual (`.venv`) e se tornou o workload oficial usado neste bloco. Por isso, a versão final da análise deve considerar este script como referência principal, substituindo o uso anterior do benchmark `7z b`.

### Métricas integradas

| Subsistema | Métrica | Valor medido |
|------------|---------|--------------|
| CPU | IPC | N/D no WSL2 (`cycles` e `instructions` não suportados pelo `perf`) |
| Cache | Taxa de miss (%) | N/D no WSL2 (`cache-references` e `cache-misses` não suportados pelo `perf`) |
| Memória virtual | Page faults / s | ~247 page faults/s |
| Armazenamento | Throughput | 89.75 MB/s |
| Multicore | Threads em uso | 0/12 núcleos > 50% no fim do snapshot |
| PCIe | Saturação | ~1.14% do link PCIe do SSD |

### No formato do novo modelo

| Workload (`workload_completo.py`) | Resultado |
|-----------------------------------|-----------|
| CPU-bound — multiplicações/s | 530.55 |
| Memory-bound — bandwidth (GB/s) | 14.80 GB/s |
| I/O-bound — throughput (MB/s) | 89.75 MB/s |
| Cores >50% no fim | 0/12 |

**Gargalo dominante (segundo o script):** I/O / armazenamento

**Você concorda? Justifique:**

Sim. O próprio script apontou o armazenamento como subsistema mais fraco, porque a fase I/O-bound ficou em apenas `89.75 MB/s`, muito abaixo da fase memory-bound, que atingiu `14.80 GB/s`. Além disso, o script emitiu alerta explícito de disco lento. A fase CPU-bound também apresentou uso alto de CPU (`99.9%`) e throughput consistente, então o ponto mais fraco desse workload específico foi claramente a parte de I/O.

### Hipóteses revisitadas

| Hipótese inicial | Status | Evidência |
|------------------|--------|-----------|
| Desempenho de CPU em workload multithread | Parcial | O `workload_completo.py` exerceu fortemente a CPU na fase CPU-bound (`99.9%` de uso), mas o snapshot final mostrou `0/12` núcleos acima de 50%, e o próprio script indicou que há paralelismo a explorar. Assim, a CPU foi bem exercitada, mas o workload não saturou os núcleos como um teste fortemente multithread faria. |
| Largura de banda e latência da RAM | Confirmada | A fase memory-bound atingiu `14.80 GB/s`, valor coerente com a banda de RAM já observada anteriormente, mostrando que a memória principal teve desempenho saudável. |
| Desempenho do armazenamento (sequencial vs aleatório) | Confirmada | No workload integrado, a fase I/O-bound ficou em `89.75 MB/s`, bem abaixo da memória. Somando isso aos testes anteriores com `fio` (`1532 MB/s` sequencial e `4157 IOPS` em 4K aleatório), fica claro que o armazenamento continua sendo uma fonte importante de limitação dependendo do padrão de acesso. |
| Eficiência da hierarquia de cache | Parcial | A fase CPU-bound foi descrita como operação com dados pequenos que cabem em cache, e a fase memory-bound saiu deliberadamente de qualquer L3. Isso sustenta a importância da hierarquia de cache, mas o WSL2 não expôs `cache-misses` diretamente no `perf`. |
| Capacidade do barramento/PCIe e impacto no I/O | Parcial | O SSD continua ligado em PCIe 4.0 x4, com teto teórico de aproximadamente `7.88 GB/s`, mas no workload integrado a fase I/O ficou em apenas `89.75 MB/s`, o que mostra que o gargalo não era o barramento PCIe e sim a forma de acesso ao armazenamento no teste. |

### Perguntas-síntese

**1. Qual é o gargalo dominante da minha máquina? Justifique com 3 métricas medidas.**

No workload integrado, o gargalo dominante foi o armazenamento. A primeira evidência é que a fase I/O-bound ficou em apenas `89.75 MB/s`. A segunda é que a fase memory-bound chegou a `14.80 GB/s`, muito acima do I/O. A terceira é que a saturação do link PCIe do SSD ficou em apenas cerca de `1.14%`, mostrando que o barramento não era o limite. Assim, neste workload, o subsistema mais fraco foi claramente o de armazenamento.

**2. Onde investir R$ 500 em upgrade? Defesa técnica.**

Eu continuaria priorizando mais RAM como upgrade mais seguro em termos gerais, porque o laboratório mostrou que sair da RAM e cair em swap gera penalidade enorme. Porém, olhando só para este workload integrado, um upgrade de armazenamento também faria sentido, já que a fase I/O-bound foi a pior do teste, com apenas `89.75 MB/s`. Então, a decisão depende do uso: para multitarefa pesada, VMs e aplicações grandes, mais RAM continua sendo o investimento mais defensável; para cargas com muita leitura e escrita síncrona em disco, o armazenamento também merece atenção.

**3. Xeon 32C/2.5GHz vs i9 8C/5.5GHz — quem ganha em qual cenário? Use Lei de Amdahl.**

O Xeon 32C/2.5GHz ganha quando o workload é altamente paralelizável, com fração serial pequena e boa distribuição entre muitos núcleos. O i9 8C/5.5GHz ganha quando o workload depende mais de desempenho por núcleo, clock alto e baixa latência. Pela Lei de Amdahl, se uma parte importante do programa continua serial, o ganho adicional de muitos núcleos diminui rapidamente. Assim, em renderização, compressão paralela e servidores com muitas tarefas simultâneas, o Xeon tende a ganhar. Em jogos, tarefas interativas e cargas com dependência forte de desempenho por núcleo, o i9 tende a ganhar.

**4. Como cada conceito afetou meu workload? (uma frase cada)**

- Pipeline e ILP: a fase CPU-bound usou intensamente a CPU, mas o IPC não pôde ser medido diretamente no WSL2.
- Cache: a fase CPU-bound foi desenhada para caber em cache, enquanto a fase memory-bound forçou a saída do cache para estressar a RAM.
- Memória virtual: page faults continuaram ocorrendo, mas em taxa relativamente baixa para o workload integrado, cerca de `247/s`.
- PCIe / barramento: o link PCIe do NVMe continuou muito acima da taxa real do I/O observado, então não foi o gargalo do workload.
- Multicore: o snapshot final mostrou `0/12` núcleos acima de 50%, sugerindo que o workload completo não explorou bem o paralelismo global da máquina.


## Bloco final — Volte ao Bloco 0

Ao final do laboratório, eu manteria como métricas principais o desempenho de CPU em workload multithread, a largura de banda e a latência da RAM, o desempenho do armazenamento em leitura sequencial e aleatória, e o impacto da memória virtual e do swap. Eu trocaria a formulação genérica “capacidade do barramento PCIe” por uma métrica mais concreta de saturação do link, porque o laboratório mostrou que ter PCIe 4.0 x4 não significa automaticamente alto throughput real do SSD. Também manteria a hierarquia de cache como conceito essencial, mas reconhecendo que, no ambiente WSL2, ela ficou apenas parcialmente observável por ausência de contadores diretos de `cache-misses`.
