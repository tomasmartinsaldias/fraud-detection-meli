# Experimentos One-Class SVM (SGD + Nystroem Kernel RBF)

> **Estándar:** Nystroem con 300 componentes y kernel RBF (`gamma='scale'`), seguido de `SGDOneClassSVM` ajustado en legítimos de `train`. Evaluación univariada en LightGBM sobre validación OOT (22.500 transacciones).

- **Semilla determinista:** `42`.
- **Prevalencia de Fraude en Validación:** `4.533%`.
- **Aviso de Integridad:** Ningún archivo Parquet de `data/features/` fue modificado.

## Tabla Comparativa de Rendimiento

| Modelo / Configuración | nu (Tolerancia) | n_components | LightGBM PR-AUC | LightGBM ROC-AUC | Mejor F1 | Precision | Recall | Umbral Opt | Direct PR-AUC | Tiempo Total |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **OCSVM_SGD_nu_0.10** | 0.1 | 300 | **0.0943** | 0.6969 | **0.1661** | 12.06% | 26.67% | 0.1083 | 0.0910 | 0.36 s |
| **OCSVM_SGD_nu_0.05** | 0.05 | 300 | **0.0961** | 0.6966 | **0.1683** | 11.50% | 31.37% | 0.0996 | 0.0936 | 0.34 s |
| **OCSVM_SGD_nu_0.03 **(MEJOR)**** | 0.03 | 300 | **0.0986** | 0.6884 | **0.1684** | 12.75% | 24.80% | 0.1044 | 0.0931 | 0.31 s |

## Conclusiones Clave de One-Class SVM con Nystroem

1. **Mejor configuración:** `OCSVM_SGD_nu_0.03` alcanzando **PR-AUC = 0.0986** y **F1 = 0.1684**.
2. **Velocidad y Escalabilidad:** Nystroem con 300 componentes aproxima el espacio RBF de forma continua y el SGD procesa las 100k filas en menos de 1 segundo de fit.
