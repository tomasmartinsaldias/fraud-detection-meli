# Experimentos PCA evaluados con LightGBM Univariado / Multi-Score

> **Estándar:** LightGBM entrenado en train exclusivamente con el percentil de los scores de PCA como únicas predictoras de fraude. Evaluación estricta en validación OOT (22.500 transacciones).

- **Semilla:** `42`.
- **Prevalencia de Fraude en Validación:** `4.533%`.
- **Aviso de Integridad:** Ningún archivo Parquet de `data/features/` fue modificado.

## Tabla Comparativa General de PCA con LightGBM

| Modo de Variables | Varianza | Comp | Score Evaluado en LightGBM | LightGBM PR-AUC | LightGBM Avg Prec | LightGBM ROC-AUC | Direct PR-AUC | LogReg PR-AUC |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| Numérico Puro | 90% | 11 | **reconstruccion_Q** | **0.0913** | 0.0692 | 0.6077 | 0.0418 | 0.0425 |
| Numérico Puro | 90% | 11 | **magnitud_cuadrada** | **0.1059** | 0.0985 | 0.6778 | 0.0915 | 0.0915 |
| Numérico Puro | 90% | 11 | **magnitud_ponderada_T2** | **0.0916** | 0.0880 | 0.6834 | 0.0839 | 0.0839 |
| Numérico Puro | 90% | 11 | **tres_scores_pca** | **0.1471** | 0.1455 | 0.7147 | - | 0.0851 |
| Numérico Puro | 95% | 12 | **reconstruccion_Q** | **0.0987** | 0.0724 | 0.6036 | 0.0399 | 0.0449 |
| Numérico Puro | 95% | 12 | **magnitud_cuadrada** | **0.0981** | 0.0924 | 0.6766 | 0.0905 | 0.0905 |
| Numérico Puro | 95% | 12 | **magnitud_ponderada_T2** | **0.0877** | 0.0837 | 0.6768 | 0.0814 | 0.0814 |
| Numérico Puro | 95% | 12 | **tres_scores_pca** | **0.1430** | 0.1395 | 0.7101 | - | 0.0882 |
| Numérico Puro | 99% | 13 | **reconstruccion_Q** | **0.0663** | 0.0611 | 0.5827 | 0.0468 | 0.0462 |
| Numérico Puro | 99% | 13 | **magnitud_cuadrada** | **0.0998** | 0.0944 | 0.6798 | 0.0893 | 0.0893 |
| Numérico Puro | 99% | 13 | **magnitud_ponderada_T2** | **0.0848** | 0.0810 | 0.6681 | 0.0778 | 0.0778 |
| Numérico Puro | 99% | 13 | **tres_scores_pca** | **0.1430** | 0.1434 | 0.7116 | - | 0.0861 |
| Numérico + Frecuencia (g, j, o, p) | 90% | 13 | **reconstruccion_Q** | **0.0781** | 0.0747 | 0.6327 | 0.0607 | 0.0607 |
| Numérico + Frecuencia (g, j, o, p) | 90% | 13 | **magnitud_cuadrada** | **0.1086** | 0.1018 | 0.7076 | 0.0978 | 0.0978 |
| Numérico + Frecuencia (g, j, o, p) | 90% | 13 | **magnitud_ponderada_T2** | **0.0917** | 0.0887 | 0.7047 | 0.0894 | 0.0894 |
| Numérico + Frecuencia (g, j, o, p) | 90% | 13 | **tres_scores_pca** | **0.1683** | 0.1701 | 0.7420 | - | 0.1022 |
| Numérico + Frecuencia (g, j, o, p) | 95% | 15 | **reconstruccion_Q** | **0.0665** | 0.0623 | 0.6017 | 0.0395 | 0.0451 |
| Numérico + Frecuencia (g, j, o, p) | 95% | 15 | **magnitud_cuadrada** | **0.1090** | 0.1027 | 0.7111 | 0.0988 | 0.0988 |
| Numérico + Frecuencia (g, j, o, p) | 95% | 15 | **magnitud_ponderada_T2** | **0.0944** | 0.0891 | 0.7033 | 0.0893 | 0.0893 |
| Numérico + Frecuencia (g, j, o, p) | 95% | 15 | **tres_scores_pca **(MEJOR)**** | **0.1698** | 0.1692 | 0.7409 | - | 0.1001 |
| Numérico + Frecuencia (g, j, o, p) | 99% | 16 | **reconstruccion_Q** | **0.0627** | 0.0593 | 0.5810 | 0.0477 | 0.0480 |
| Numérico + Frecuencia (g, j, o, p) | 99% | 16 | **magnitud_cuadrada** | **0.1072** | 0.1020 | 0.7100 | 0.0969 | 0.0969 |
| Numérico + Frecuencia (g, j, o, p) | 99% | 16 | **magnitud_ponderada_T2** | **0.0888** | 0.0844 | 0.6934 | 0.0846 | 0.0846 |
| Numérico + Frecuencia (g, j, o, p) | 99% | 16 | **tres_scores_pca** | **0.1625** | 0.1627 | 0.7357 | - | 0.0947 |

## Conclusiones Clave de PCA bajo el Estándar LightGBM

1. **Mejor configuración de PCA:** `tres_scores_pca` bajo `Numérico + Frecuencia (g, j, o, p)` a 95% de varianza.
   - **LightGBM PR-AUC:** `0.1698` | **ROC-AUC:** `0.7409`.
2. **Comparación con Isolation Forest:**
   - Isolation Forest (`IF_Diagonal_HybridOHE_256`) alcanzó un LightGBM PR-AUC de **0.1557** y ROC-AUC de **0.7431**.
   - PCA (`magnitud_cuadrada` / `tres_scores_pca`) aporta una señal complementaria de densidad elíptica y distancias latentes.
