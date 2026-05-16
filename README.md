# Trabalho Prático — Machine Learning I (CC2008)

**UC:** Machine Learning I — Universidade do Porto  
**Grupo:** Dataset Group 5 — Desequilíbrio de Classes  
**Notebook:** `main.ipynb`

---

## O que é este trabalho?

Este trabalho investiga o problema do **desequilíbrio de classes** em algoritmos de classificação. O objectivo é ensinar um computador a identificar casos raros — por exemplo, detectar defeitos de software ou diagnosticar doenças pouco frequentes — quando os dados de treino têm muito mais exemplos negativos do que positivos.

O algoritmo base é uma **Árvore de Decisão CART** implementada de raiz. O notebook documenta o caminho completo de investigação: da identificação do problema, passando por três modificações independentes, até ao produto final.

---

## O problema: dados desequilibrados

Imagina que tens 1000 análises médicas e apenas 30 são positivas. Um algoritmo que prevê sempre "negativo" acerta em 970 dos 1000 casos — mas **falha em todos os positivos**. Do ponto de vista médico ou de engenharia, esse modelo é inútil.

A proporção de casos raros é medida pelo **IR** (Imbalance Ratio = n_min / n_max). Quanto mais baixo for o IR, mais difícil é o problema. O algoritmo standard falha em dois pontos:

1. **Splits enviesados** — o critério de Gini favorece divisões que isolam a classe maioritária, ignorando a minoritária
2. **Folhas enviesadas** — as probabilidades emitidas pelas folhas estão sistematicamente abaixo de 0.5, pelo que o modelo nunca prevê a classe rara

---

## As quatro abordagens investigadas

| Fase | Abordagem | Mecanismo | Artigo de referência |
|------|-----------|-----------|----------------------|
| **1** | Gini Standard (Baseline) | — | — |
| **2A** | Weighted CART | Pesos $w_i = n/(2n_{y_i})$ balanceiam splits e folhas | Ting (2002), IEEE TKDE |
| **2B** | Reduced Error Pruning | Poda bottom-up por G-Mean (adaptação de Quinlan 1987) | Quinlan (1987), IJMMS |
| **2C** | Cost-Sensitive Tree ← **produto final** | Custos assimétricos $C(\text{FN})=5, C(\text{FP})=1$ + limiar $\tau^*=0.167$ | Elkan (2001), IJCAI |

A Fase 2B serve como **controlo experimental**: por não alterar o critério nem as folhas, os seus resultados indicam se o problema é de estrutura (profundidade) ou de critério — o que justifica as modificações das Fases 2A e 2C.

---

## Porquê a Cost-Sensitive Tree é o produto final

O Weighted Gini (2A) melhora o desempenho tratando o desequilíbrio como um problema de *proporções*. A Cost-Sensitive Tree (2C) vai mais longe: trata-o como um problema de *custos*.

Em aplicações reais — detecção de defeitos de software, diagnóstico médico — perder um caso positivo (Falso Negativo) custa muito mais do que um falso alarme (Falso Positivo). A Cost-Sensitive Tree incorpora esta assimetria directamente no critério de split e usa o **limiar de Bayes óptimo**:

$$\tau^* = \frac{C(\text{FP})}{C(\text{FN}) + C(\text{FP})} = \frac{1}{5+1} \approx 0.167$$

Este limiar é derivado matematicamente da função de custo (Elkan, 2001) — não é um hiperparâmetro tunado por validação. O rácio $C(\text{FN})/C(\text{FP})$ pode ser ajustado ao domínio sem re-treinar o modelo.

---

## Dados e avaliação

- **38 datasets** binários de domínios variados (medicina, software, ambiente), com IR de ~0.05 a ~0.24
- Descoberta automática: datasets com menos de 20 instâncias da classe minoritária são excluídos
- **Stratified 5-Fold CV** em cada fase
- **Métricas:** AUC-ROC, F1 (classe minoritária), G-mean
- **Teste estatístico:** Wilcoxon signed-rank one-sided (H₁: nova > baseline, α=0.05, n=38 pares)

---

## Estrutura do projecto

```
projeto_investigacao_AP1/
│
├── main.ipynb              ← Notebook único com toda a investigação
│
├── data/
│   └── class_imbalance/    ← 50 datasets (38 utilizados após filtro n_min≥20)
│
└── results_main/
    ├── resultados_fase1.png
    ├── comparacao_gini_vs_wg.png
    ├── resultados_fase2_rep.png
    ├── comparacao_cost_sensitive_tree.png
    └── resultados_comparacao.png   ← Gráfico final (ΔF1 e ΔG-mean vs IR)
```

