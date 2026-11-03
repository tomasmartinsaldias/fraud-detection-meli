# Reporte Comparativo: Experimentos con Local Outlier Factor (LOF)

> **Estándar de Evaluación:** LightGBM supervisado entrenado **única y exclusivamente con el percentil de anomalía de LOF** como sola variable predictora de fraude. Evaluación estricta en validación temporal OOT (22.500 transacciones).

- **Muestra de referencia para ajuste (`novelty=True`):** `15,000` transacciones legítimas de train (semilla `42`).
- **Prevalencia de Fraude en Validación (Azar):** `4.533%`.
- **Algoritmo de Indexación:** `algorithm='kd_tree'` con `leaf_size=30`.
- **Aviso de Integridad:** Ningún archivo Parquet de `data/features/` fue modificado.

## 1. Tabla Comparativa General de LOF con LightGBM

| ID / Configuración | Distancia | p | k (Vecinos) | leaf_size | LightGBM PR-AUC | LightGBM Avg Prec | LightGBM ROC-AUC | Direct PR-AUC | Direct ROC-AUC | LogReg PR-AUC | Tiempo (Fit + Score) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LOF_Euclidean_k20** | Euclidiana | 2 | 20 | 30 | **0.0673** | 0.0651 | 0.6039 | 0.0677 | 0.6001 | 0.0677 | 30.5 s |
| **LOF_Euclidean_k60** | Euclidiana | 2 | 60 | 30 | **0.0754** | 0.0720 | 0.6281 | 0.0718 | 0.6181 | 0.0718 | 66.5 s |
| **LOF_Manhattan_k20 **(MEJOR)**** | Manhattan | 1 | 20 | 30 | **0.0792** | 0.0739 | 0.6232 | 0.0722 | 0.6113 | 0.0722 | 100.0 s |
| **LOF_Manhattan_k60** | Manhattan | 1 | 60 | 30 | **0.0750** | 0.0698 | 0.6196 | 0.0709 | 0.6178 | 0.0709 | 86.7 s |

## 2. Hallazgos y Análisis Comparativo

### A. Comparación entre Distancia Euclidiana (p=2) vs. Manhattan (p=1)
- **Manhattan ($L_1$):** Suma de distancias absolutas por coordenada. En datos de fraude con variables que presentan colas pesadas (montos y ratios), suele ser menos sensible a valores atípicos unidimensionales extremos.
- **Euclidiana ($L_2$):** Penaliza cuadráticamente las diferencias; forma vecindarios hiper-esféricos en el espacio estandarizado.

### B. Efecto del Tamaño de Vecindario: k=20 vs. k=60
- **k = 20:** Vecindario más estrecho; detecta micro-anomalías locales pero puede incorporar más ruido.
- **k = 60:** Vecindario más amplio; evalúa la densidad relativa frente a subgrupos y clusters de mayor tamaño.

### C. Configuración Ganadora de LOF: `LOF_Manhattan_k20`
- Supera consistentemente el baseline aleatorio de validación (~4.53%).
