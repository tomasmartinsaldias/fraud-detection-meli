# Reporte Comparativo: Experimentos con Isolation Forest

> **Estándar de Evaluación Requerido:** LightGBM supervisado entrenado **única y exclusivamente con el percentil de Isolation Forest** como única predictora de fraude. Evaluación estricta en validación temporal (OOT).

- **Semilla determinista:** `42`.
- **Prevalencia de Fraude en Validación (Línea de base aleatoria):** `4.533%`.
- **Restricción Anti-Leakage:** Isolation Forest ajustado únicamente sobre transacciones legítimas (`fraude == 0`) de `train`.

## 1. Tabla Comparativa de Rendimiento (LightGBM Univariado)

| ID / Configuración | Tipo de Corte | Encoding | max_samples | n_est | LightGBM PR-AUC | LightGBM Avg Prec | LightGBM ROC-AUC | Direct PR-AUC | Direct ROC-AUC | LogReg PR-AUC |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **IF_Orthogonal_Numeric_256** | Ortogonal (Axis-Aligned) | `numeric_only` | 256 | 150 | **0.0871** | 0.0812 | 0.6659 | 0.0833 | 0.6652 | 0.0833 |
| **IF_Orthogonal_Freq_256** | Ortogonal (Axis-Aligned) | `frequency` | 256 | 150 | **0.1016** | 0.0919 | 0.7087 | 0.0958 | 0.7089 | 0.0958 |
| **IF_Orthogonal_Freq_512** | Ortogonal (Axis-Aligned) | `frequency` | 512 | 150 | **0.1013** | 0.0937 | 0.7046 | 0.0967 | 0.7060 | 0.0967 |
| **IF_Orthogonal_HybridOHE_256** | Ortogonal (Axis-Aligned) | `hybrid_ohe` | 256 | 150 | **0.1162** | 0.1086 | 0.7343 | 0.1100 | 0.7341 | 0.1100 |
| **IF_Diagonal_Numeric_256** | Diagonal (Rotated Hyperplanes) | `numeric_only` | 256 | 150 | **0.0904** | 0.0849 | 0.6660 | 0.0843 | 0.6689 | 0.0843 |
| **IF_Diagonal_Freq_256** | Diagonal (Rotated Hyperplanes) | `frequency` | 256 | 150 | **0.0907** | 0.0854 | 0.6944 | 0.0852 | 0.6918 | 0.0852 |
| **IF_Diagonal_Freq_512** | Diagonal (Rotated Hyperplanes) | `frequency` | 512 | 150 | **0.0992** | 0.0938 | 0.7049 | 0.0923 | 0.7022 | 0.0923 |
| **IF_Diagonal_HybridOHE_256 (MEJOR)** | Diagonal (Rotated Hyperplanes) | `hybrid_ohe` | 256 | 150 | **0.1557** | 0.1241 | 0.7431 | 0.0860 | 0.7219 | 0.0860 |

## 2. Hallazgos y Análisis Comparativo

### A. Desempeño del LightGBM Univariado sobre el Percentil de Anomaly Score
- Al utilizar el **percentil empírico de anomalía como única variable en LightGBM**, el modelo aprende de forma óptima los umbrales no lineales de decisión sobre la cola superior de anomalías.
- Todos los modelos superan significativamente la prevalencia aleatoria (~4.53%), demostrando una fuerte capacidad de discriminación.

### B. Cortes Ortogonales vs. Cortes Diagonales (Rotated Hyperplanes)
- **Cortes Ortogonales (Axis-Aligned):** Logran particiones nítidas en variables críticas como monto y score.
- **Cortes Diagonales (Rotated Hyperplanes):** Evalúan combinaciones lineales multidimensionales simultáneamente.

### C. Configuración Seleccionada para Producción
- **Mejor Modelo según LightGBM PR-AUC:** `IF_Diagonal_HybridOHE_256`.
- Este modelo fue persistido en `data/features/scores_anomalias.parquet` con la columna `score_iforest` conteniendo el percentil normalizado en $[0.0, 1.0]$.
