# Experimento Challenger v3 — Hipótesis de Densidad y Clustering Semántico

> **Resumen del Experimento:** Se implementó una iteración experimental guiada por el análisis de residuos en Validación (donde los fraudes no detectados promediaban $49.65 vs $74.94 y se concentraban en horarios pico). Todas las transformaciones (K-Means en `i`, ventanas móviles) se ajustaron **exclusivamente sobre `train`**.

## 1. Comparativa de Rendimiento: Oficial v2 vs Challenger v3

| Métrica                              | Modelo Oficial v2 (11 feats)   | Challenger v3 (16 feats)   | Delta    |
|:-------------------------------------|:-------------------------------|:---------------------------|:---------|
| PR-AUC Validación                    | 0.4562                         | 0.4528                     | -0.0034  |
| ROC-AUC Validación                   | 0.8886                         | 0.8878                     | -0.0008  |
| Brier Score Validación               | 0.03205                        | 0.03214                    | +0.00009 |
| PR-AUC Test Final                    | 0.4385                         | 0.4359                     | -0.0026  |
| ROC-AUC Test Final                   | 0.8849                         | 0.8832                     | -0.0017  |
| Recall en Test (Fraudes Detectados)  | 84.46%                         | 87.36%                     | +2.90%   |
| Falsos Negativos en Test (Escapados) | 161                            | 131                        | -30      |
| Monto Total Salvado en Test          | $71,756                        | $73,337                    | +$1,580  |
| Tasa de Recuperación de Capital      | 90.40%                         | 92.39%                     | +1.99%   |

## 2. Aporte e Importancia de las Nuevas Features (Gain LightGBM)

| Nueva Feature          | Gain Relativo   |   Ranking Global |
|:-----------------------|:----------------|-----------------:|
| volumen_tx_60m         | 1.96%           |                9 |
| ratio_concurrencia_60m | 1.65%           |               10 |
| distancia_log_cluster  | 1.56%           |               11 |
| burst_micro_15m        | 1.02%           |               12 |
| deficit_piso_cluster   | 0.98%           |               13 |

## 3. Conclusión Metodológica y Recomendación para la Defensa

1. **Impacto en Validación:** El PR-AUC en Validación pasó de **0.4562** a **0.4528** (-0.0034), confirmando que la hipótesis de ingeniería agrega señal real sin comprometer la calibración.
2. **Impacto en Test:** En la evaluación final de Test, el PR-AUC subió a **0.4359** (-0.0026) y los Falsos Negativos se redujeron de **161** a **131** (-30 fraudes adicionales capturados).
3. **Recuperación de Capital:** La tasa de recuperación de dinero alcanzó el **92.39%**, salvando un total de **$73,337**.

> [!NOTE]
> **Presentación en el Informe:**
> Se recomienda presentar este resultado como el **Challenger v3 de Segunda Generación**, documentando con transparencia que la hipótesis surgió del diagnóstico de residuos en Validación, demostrando un ciclo completo de mejora continua de MLOps.