# Experimento: Optimización Sistemática de la Interacción Anomalía x Monto en Brasil

> **Objetivo:** Poner a prueba todas las variantes de interacción entre los diferentes modelos de anomalías One-Class de Brasil y el monto en BRL (`log1p` vs ratio contra mediana de 19.35 BRL). Evaluar con Proxy Evaluator, seleccionar el Top 5 y someterlas al Forward Stepwise Selection oficial.

- **Población Evaluada (Val BR):** `16,680` transacciones OOT (Prevalencia: `5.102%`).
- **Variantes Evaluadas:** `14` interacciones continuas.
- **Tiempo de Búsqueda:** `421.14` segundos.

## 1. Ranking del Screening Inicial con Proxy Evaluator

| Ranking | Interacción Evaluada | Modelo de Anomalía Base | Escala de Monto | PR-AUC (Val) | Delta vs Baseline | ROC-AUC | Ganancia (Gain) |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| 1 | `anom_logmonto_iforest_perp` **(TOP 5)** | `perp` | log1p(monto) | **0.4430** | `+0.0008` | 0.8784 | 5,678.9 |
| 2 | `anom_logmonto_anomalia_consolidado` **(TOP 5)** | `consolidado` | log1p(monto) | **0.4430** | `+0.0008` | 0.8771 | 11,700.6 |
| 3 | `anom_logmonto_pca_q` **(TOP 5)** | `q` | log1p(monto) | **0.4425** | `+0.0002` | 0.8767 | 906.0 |
| 4 | `anom_logmonto_pca_magnitud` **(TOP 5)** | `magnitud` | log1p(monto) | **0.4424** | `+0.0002` | 0.8774 | 3,407.5 |
| 5 | `anom_logmonto_pca_t2` **(TOP 5)** | `t2` | log1p(monto) | **0.4424** | `+0.0002` | 0.8764 | 4,801.5 |
| 6 | `anom_ratiomonto_pca_q` | `q` | monto / 19.35 | **0.4422** | `+0.0000` | 0.8771 | 1,011.6 |
| 7 | `anom_ratiomonto_pca_t2` | `t2` | monto / 19.35 | **0.4417** | `-0.0005` | 0.8762 | 3,236.6 |
| 8 | `anom_ratiomonto_iforest_perp` | `perp` | monto / 19.35 | **0.4413** | `-0.0010` | 0.8772 | 3,843.9 |
| 9 | `anom_logmonto_lof_manhattan` | `manhattan` | log1p(monto) | **0.4412** | `-0.0010` | 0.8768 | 1,723.8 |
| 10 | `anom_logmonto_iforest_diag` | `diag` | log1p(monto) | **0.4411** | `-0.0011` | 0.8772 | 4,565.9 |
| 11 | `anom_ratiomonto_lof_manhattan` | `manhattan` | monto / 19.35 | **0.4410** | `-0.0013` | 0.8773 | 1,942.6 |
| 12 | `anom_ratiomonto_pca_magnitud` | `magnitud` | monto / 19.35 | **0.4404** | `-0.0018` | 0.8768 | 2,625.7 |
| 13 | `anom_ratiomonto_iforest_diag` | `diag` | monto / 19.35 | **0.4384** | `-0.0039` | 0.8758 | 3,471.9 |
| 14 | `anom_ratiomonto_anomalia_consolidado` | `consolidado` | monto / 19.35 | **0.4371** | `-0.0051` | 0.8773 | 7,252.8 |

## 2. Historial de Forward Selection Oficial con el Top 5

| Paso | Variable Seleccionada | Tipo de Variable | PR-AUC (Val BR) | Delta vs Best | ROC-AUC |
| :---: | :--- | :--- | :---: | :---: | :---: |
| 1 | `score` | BASE CRUDA | **0.2971** | `+0.0000` | 0.8122 |
| 2 | `o` | BASE CRUDA | **0.4263** | `+0.1292` | 0.8764 |
| 3 | `anom_logmonto_iforest_perp` | TOP 5 ANOM x MONTO | **0.4351** | `+0.0088` | 0.8812 |
| 4 | `c` | BASE CRUDA | **0.4460** | `+0.0109` | 0.8819 |
| 5 | `diff_m_d` | INGENIERÍA NATIVA | **0.4539** | `+0.0078` | 0.8824 |
| 6 | `es_madrugada` | INGENIERÍA NATIVA | **0.4639** | `+0.0101` | 0.8860 |
| 7 | `dia_semana` | INGENIERÍA NATIVA | **0.4686** | `+0.0047` | 0.8861 |
| 8 | `ratio_monto_q` | INGENIERÍA NATIVA | **0.4685** | `-0.0001` | 0.8866 |
| 9 | `avg_word_len` | INGENIERÍA NATIVA | **0.4716** | `+0.0030` | 0.8866 |
| 10 | `e` | BASE CRUDA | **0.4720** | `+0.0004` | 0.8867 |
| 11 | `ratio_monto_a_bayes_br` | INGENIERÍA NATIVA | **0.4726** | `+0.0007` | 0.8872 |
| 12 | `ratio_monto_e` | INGENIERÍA NATIVA | **0.4738** | `+0.0012` | 0.8875 |
| 13 | `flag_alto_riesgo_nop` | INGENIERÍA NATIVA | **0.4755** | `+0.0016` | 0.8881 |
| 14 | `text_svd_br_9` | INGENIERÍA NATIVA | **0.4756** | `+0.0002` | 0.8876 |

## 3. Importancia de Variables en el Modelo Definitivo (Ranking por Gain)

| Ranking | Variable | Tipo | Ganancia (Gain) | Splits | % Gain Acumulado |
| :---: | :--- | :--- | :---: | :---: | :---: |
| 1 | `score` | Transaccional | 64,780.2 | 651 | 55.6% |
| 2 | `o` | Transaccional | 26,207.1 | 257 | 78.1% |
| 3 | `anom_logmonto_iforest_perp` | Interacción Anomalía | 5,702.0 | 335 | 83.0% |
| 4 | `c` | Transaccional | 4,896.8 | 496 | 87.2% |
| 5 | `diff_m_d` | Transaccional | 3,380.1 | 374 | 90.1% |
| 6 | `ratio_monto_a_bayes_br` | Transaccional | 2,678.0 | 318 | 92.4% |
| 7 | `ratio_monto_q` | Transaccional | 2,548.9 | 324 | 94.6% |
| 8 | `ratio_monto_e` | Transaccional | 1,557.8 | 200 | 96.0% |
| 9 | `avg_word_len` | Transaccional | 1,459.9 | 230 | 97.2% |
| 10 | `e` | Transaccional | 1,332.3 | 175 | 98.3% |
| 11 | `es_madrugada` | Transaccional | 1,055.5 | 81 | 99.3% |
| 12 | `dia_semana` | Transaccional | 608.1 | 93 | 99.8% |
| 13 | `flag_alto_riesgo_nop` | Transaccional | 258.6 | 38 | 100.0% |

## 4. Conclusiones Técnicas

1. **Ganadora Absoluta del Screening:** `anom_logmonto_iforest_perp` demostró la mayor capacidad de corte univariado en el screening.
2. **Comportamiento en el Forward Selection:** Al competir contra todo el pool de variables, la interacción seleccionada demostró si aporta señal ortogonal frente a variables como `c`, `o` y `score`.
3. **Métricas Finales del Modelo Optimizado:** PR-AUC: **`0.4757`** | ROC-AUC: **`0.8881`** | Mejor F1: **`0.4600`** (umbral 0.2542).
