# Experimentos Gaussian Mixture Model (GMM)

> **Estándar:** GMM con `covariance_type='full'`, ajustado estrictamente sobre transacciones legítimas de `train`. Score de anomalía definido como $-\ln p(x)$ y normalizado a percentiles empíricos $[0.0, 1.0]$. Evaluación univariada con LightGBM sobre validación OOT (22.500 transacciones).

- **Prevalencia base de fraude en validación:** `4.533%`
- **Semilla:** `42`

## Resumen Comparativo de Métricas (Validación OOT)

| Modelo (k) | Fit Time (s) | Direct PR-AUC | Direct ROC-AUC | LGBM PR-AUC | LGBM ROC-AUC | Mejor F1 | Precision | Recall | Umbral Opt | BIC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `GMM_k_3` | 2.14 | 0.0846 | 0.6838 | **0.0909** | **0.6961** | **0.1560** | 9.12% | 53.63% | 0.0882 | 354,492 |
| `GMM_k_5` | 3.49 | 0.0707 | 0.6401 | **0.0984** | **0.6898** | **0.1747** | 11.68% | 34.71% | 0.0994 | -1,162,864 |
| `GMM_k_8` | 6.80 | 0.0652 | 0.6055 | **0.0894** | **0.6538** | **0.1539** | 10.85% | 26.47% | 0.0844 | -2,293,803 |
| `GMM_k_12` | 11.91 | 0.0713 | 0.6309 | **0.0933** | **0.6591** | **0.1301** | 8.29% | 30.20% | 0.0682 | -3,088,227 |

## Observaciones Técnicas
1. **Definición de Anomaly Score:** Las observaciones anómalas tienen menor densidad conjunta $p(x) = \sum \pi_k \mathcal{N}_k(x) \to 0^+$, lo cual genera un log-likelihood muy negativo $\ln p(x) \to -\infty$. Por ende, el score $-\ln p(x)$ es alto para anomalías.
2. **Normalización por Percentiles:** El ordenamiento empírico en $[0.0, 1.0]$ sobre el conjunto legítimo permite interpretar el score directamente como el cuantil de rareza estadística.
3. **Capacidad vs Generalización:** A mayor $k$, el GMM modela modas locales más específicas; se evalúa si $k$ grande retiene poder discriminativo sobre fraudes o sobreajusta la densidad del espacio legítimo.