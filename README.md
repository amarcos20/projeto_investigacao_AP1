# Trabalho Prático — Machine Learning I (CC2008)

**UC:** Machine Learning I — Universidade do Porto  

---

## O que é este trabalho?

Este trabalho é sobre ensinar um computador a tomar decisões — especificamente a classificar casos como "doente" ou "saudável", "defeito" ou "sem defeito" — em situações onde os dados são muito desequilibrados. Por exemplo: numa base de dados de doenças raras, pode haver 100 pessoas saudáveis para cada pessoa doente. Isso cria um problema grave para os algoritmos tradicionais.

O trabalho tem dois ficheiros principais, cada um focado num algoritmo diferente:

| Ficheiro | Algoritmo | Problema estudado |
|---|---|---|
| `notebook_experimental.ipynb` | Árvore de Decisão | Desequilíbrio de classes ← **este trabalho** |

---

## O problema central: dados desequilibrados

Imagina que tens 1000 exames médicos na base de dados e apenas 10 são de pessoas doentes. Se o computador aprender a dizer sempre "saudável", acerta em 990 dos 1000 casos — mas **falha em todos os doentes**. Do ponto de vista médico, este modelo é inútil.

Isto chama-se **desequilíbrio de classes** (*class imbalance*). A proporção de casos raros é medida pelo **IR** (Imbalance Ratio = número de casos raros / número de casos comuns). Quanto mais baixo for o IR, mais difícil é o problema:

- IR = 0.5 → metade dos casos são raros → problema fácil
- IR = 0.1 → 1 caso raro para cada 10 comuns → problema moderado
- IR = 0.01 → 1 caso raro para cada 100 comuns → problema muito difícil

Nos dados usados neste trabalho, o IR vai de **0.014 a 0.241**.

---

## O algoritmo: Árvore de Decisão

Uma **Árvore de Decisão** funciona exactamente como parece: é uma sequência de perguntas de sim/não que levam a uma conclusão.

```
Temperatura > 38°?
├── Sim → Tosse presente?
│         ├── Sim → Provável gripe
│         └── Não → Verificar outros sintomas
└── Não → Provável saudável
```

O computador aprende automaticamente quais perguntas fazer e em que ordem, analisando os exemplos de treino. O critério que usa para decidir a melhor pergunta chama-se **Gini** — uma medida de "quão misturados" estão os casos depois de cada divisão.

**O problema:** o Gini standard não liga ao desequilíbrio. Se 95% dos casos são saudáveis, ele aprende que a melhor pergunta é... nenhuma, porque dizer sempre "saudável" já está quase certo.

---

## A nossa solução: Gini Ponderado

A modificação proposta é simples de explicar: **dar mais atenção aos casos raros**.

Em vez de tratar todos os casos igualmente, atribuímos um peso maior aos casos minoritários. Um caso raro pode valer, por exemplo, 10× mais do que um caso comum na avaliação de qualidade de cada divisão. Assim, o algoritmo é obrigado a aprender padrões que distinguem os casos raros — mesmo que sejam poucos.

É o equivalente a dizer ao algoritmo: *"Não me interessa que acertes nos saudáveis — já sei que são a maioria. Quero que encontres os doentes."*

---

## Como medimos o sucesso?

A acurácia (percentagem de acertos no total) é enganadora em dados desequilibrados, por isso usamos três métricas diferentes:

| Métrica | O que mede | Porquê é importante |
|---|---|---|
| **AUC-ROC** | Capacidade de distinguir as duas classes | Não depende de limiar de decisão; robusta ao desequilíbrio |
| **F1-Score** | Equilíbrio entre encontrar os casos raros e não gerar falsos alarmes | Directamente afectada por quem se ignora |
| **G-mean** | Média geométrica de acertos em cada classe | Penaliza fortemente quem ignora qualquer uma das classes |

---

## Os dados usados

Foram usados **11 conjuntos de dados reais** de domínios variados — medicina, ciência, engenharia — com diferentes graus de desequilíbrio:

| Dataset | Domínio | IR | Dificuldade |
|---|---|---|---|
| Yeast-ML8 | Biologia | 0.014 | Muito difícil |
| Arsenic-ML | Saúde pública | 0.024 | Muito difícil |
| Arsenic-FL | Saúde pública | 0.035 | Difícil |
| Oil-Spill | Ambiente | 0.046 | Difícil |
| Sick | Medicina | 0.066 | Moderado |
| NeaVote | Política | 0.075 | Moderado |
| Challenger | Engenharia | 0.070 | Moderado |
| AR1 | Software | 0.080 | Moderado |
| Hypothyroid | Medicina | 0.085 | Moderado |
| Backache | Medicina | 0.161 | Suave |
| Chlamydia | Saúde pública | 0.235 | Suave |

Cada conjunto de dados foi testado **5 vezes** com divisões diferentes (validação cruzada), e os resultados são a média dessas 5 experiências.

---

## Resultados (pasta `results_1/`)

Foram gerados dois gráficos:

**`resultados_fase1.png`** — Comparação directa entre o Gini standard (azul) e o Gini Ponderado (laranja) nos 11 datasets, para as três métricas. As barras laranja devem ser consistentemente mais altas do que as azuis nas métricas F1 e G-mean.

**`delta_gini_ponderado.png`** — Diferença de desempenho entre os dois métodos em função do IR. Pontos verdes = Gini Ponderado ganhou; pontos vermelhos = Gini standard foi melhor. Esperamos ver mais verde nos datasets com IR mais baixo (lado esquerdo do gráfico).

---

## Estrutura do projecto

```
projeto_investigacao_AP1/
│
├── notebook_experimental.ipynb   ← O trabalho principal (Decision Tree + Class Imbalance)
├── main.ipynb                    ← Trabalho de referência (SVM + Multiclass)
├── PracticalAssignment_ML1.pdf   ← Enunciado do trabalho
│
├── data/
│   └── class_imbalance/          ← Os 11 (de 50) datasets usados
│
└── results_1/
    ├── resultados_fase1.png      ← Gráfico de barras comparativo
    └── delta_gini_ponderado.png  ← Gráfico de diferença vs IR
```

---

## O que está planeado a seguir (Fase 2)

O Gini Ponderado melhora as divisões internas da árvore, mas ainda usa um limiar de decisão fixo (0.5) para converter probabilidades em respostas binárias. Em dados muito desequilibrados, esse limiar é demasiado alto — a probabilidade de um caso raro raramente chega a 0.5.

A Fase 2 propõe **aprender automaticamente o limiar óptimo** usando os próprios dados de validação, maximizando o G-mean. Mais simples do que parece: em vez de perguntar "a probabilidade é maior que 50%?", perguntamos "qual é o valor de corte que melhor equilibra a detecção das duas classes nestes dados específicos?"
