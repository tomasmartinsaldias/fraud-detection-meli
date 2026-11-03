# Experimento de Ensamble: Combinación de Modelos de Anomalías

> **Estándar:** LightGBM sencillo (`n_estimators=100`, `max_depth=3`, `lr=0.05`) entrenado en `train` utilizando **únicamente las combinaciones de percentiles de anomalías** como variables predictoras de fraude. Evaluación en validación temporal OOT (22.500 transacciones).

- **Prevalencia de Fraude en Validación:** `4.533%`.
- **Semilla determinista:** `42`.
- **Aviso de Integridad:** Ningún archivo Parquet de `data/features/` fue modificado.

## 1. Tabla Comparativa de Rendimiento de las Combinaciones

| Combinación / Ensamble | Cant. Features | LightGBM PR-AUC | LightGBM Avg Prec | LightGBM ROC-AUC | Ganancia vs Azar |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Solo PCA (3 scores)** | 3 | **0.1698** | 0.1692 | 0.7409 | **3.7x** |
| **Solo PCA (Magnitud Cuadrada)** | 1 | **0.1090** | 0.1027 | 0.7111 | **2.4x** |
| **Solo Isolation Forest (Diagonal)** | 1 | **0.1557** | 0.1241 | 0.7431 | **3.4x** |
| **Solo LOF (Manhattan k=20)** | 1 | **0.0792** | 0.0739 | 0.6232 | **1.7x** |
| **PCA + Isolation Forest** | 4 | **0.1948** | 0.1939 | 0.7763 | **4.3x** |
| **PCA (Magnitud) + Isolation Forest** | 2 | **0.1431** | 0.1361 | 0.7623 | **3.2x** |
| **Isolation Forest + LOF** | 2 | **0.1410** | 0.1391 | 0.7497 | **3.1x** |
| **PCA + LOF** | 4 | **0.1760** | 0.1762 | 0.7471 | **3.9x** |
| **TRÍO COMPLETO: PCA + Isolation Forest + LOF **(MEJOR)**** | 5 | **0.1970** | 0.1981 | 0.7777 | **4.3x** |

## 2. Importancia Relativa de Variables en el Mejor Ensamble

**Configuración:** `TRÍO COMPLETO: PCA + Isolation Forest + LOF` (PR-AUC: `0.1970` | ROC-AUC: `0.7777`)

| Feature de Anomalía | Paradigma | Ganancia / Splits (LightGBM) |
| :--- | :--- | :---: |
| `iforest_diagonal` | Isolation Forest (Partición Diagonales) | 211.0 |
| `pca_t2` | PCA (Proyección Latente / Mahalanobis) | 168.0 |
| `pca_q` | PCA (Proyección Latente / Mahalanobis) | 117.0 |
| `pca_magnitud` | PCA (Proyección Latente / Mahalanobis) | 117.0 |
| `lof_manhattan` | LOF (Densidad de Vecindarios) | 87.0 |

## 3. Conclusiones y Análisis de Sinergia

1. **Sinergia Demostrada:** Al combinar `TRÍO COMPLETO: PCA + Isolation Forest + LOF`, el PR-AUC asciende a **0.1970**.
2. **Complementariedad Geométrica:** PCA captura distancias globales y roturas de covarianza, mientras que Isolation Forest captura umbrales hiperplanares oblicuos en regiones dispersas. Ambos se potencian mutuamente.
