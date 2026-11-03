# Experimentos PCA — validación temporal

- Entrada: `data\processed\dataset_base.parquet`; semilla: 42.
- Variables numéricas: score, monto, a, b, d, e, f, h, l, m, n, q, r, s.
- Categóricas codificadas por frecuencia: g, j, o, p.
- Train: 105,000 transacciones; 99,554 legítimas y 5,446 fraudes.
- Validación: 22,500 transacciones; 1,020 fraudes; prevalencia 4.533%.
- Medianas, frecuencias, escalado y PCA ajustados solo con legítimos de train. La regresión logística usa ambas clases de train.
- Para cada PCA se prueban tres logísticas individuales y otra conjunta; scores estandarizados con legítimos de train. Test queda reservado.

| Varianza objetivo | Componentes | Varianza real | Indicadores | PR-AUC | Average precision | ROC-AUC |
| ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 90% | 13 | 91.14% | reconstruccion_Q | 0.0607 | 0.0611 | 0.6130 |
| 90% | 13 | 91.14% | magnitud_cuadrada | 0.0978 | 0.0982 | 0.7052 |
| 90% | 13 | 91.14% | magnitud_ponderada_T2 | 0.0894 | 0.0897 | 0.7064 |
| 90% | 13 | 91.14% | tres_scores | 0.0773 | 0.0780 | 0.6281 |
| 95% | 15 | 97.61% | reconstruccion_Q | 0.0451 | 0.0453 | 0.5287 |
| 95% | 15 | 97.61% | magnitud_cuadrada | 0.0988 | 0.0992 | 0.7072 |
| 95% | 15 | 97.61% | magnitud_ponderada_T2 | 0.0893 | 0.0897 | 0.7040 |
| 95% | 15 | 97.61% | tres_scores | 0.1140 | 0.1149 | 0.6122 |
| 99% | 16 | 100.00% | reconstruccion_Q | 0.0477 | 0.0479 | 0.4988 |
| 99% | 16 | 100.00% | magnitud_cuadrada | 0.0969 | 0.0973 | 0.7045 |
| 99% | 16 | 100.00% | magnitud_ponderada_T2 | 0.0846 | 0.0850 | 0.6931 |
| 99% | 16 | 100.00% | tres_scores | 0.1088 | 0.1096 | 0.5996 |

### Coeficientes de las regresiones

| Objetivo | Modelo | Coeficientes por score estandarizado |
| ---: | --- | --- |
| 90% | reconstruccion_Q | reconstruccion_Q: 0.1710 |
| 90% | magnitud_cuadrada | magnitud_cuadrada: 0.0120 |
| 90% | magnitud_ponderada_T2 | magnitud_ponderada_T2: 0.0103 |
| 90% | tres_scores | reconstruccion_Q: 0.1586, magnitud_cuadrada: 10.4146, magnitud_ponderada_T2: -10.4851 |
| 95% | reconstruccion_Q | reconstruccion_Q: -0.3126 |
| 95% | magnitud_cuadrada | magnitud_cuadrada: 0.0128 |
| 95% | magnitud_ponderada_T2 | magnitud_ponderada_T2: 0.0113 |
| 95% | tres_scores | reconstruccion_Q: -0.3137, magnitud_cuadrada: 8.8228, magnitud_ponderada_T2: -8.8100 |
| 99% | reconstruccion_Q | reconstruccion_Q: 0.0077 |
| 99% | magnitud_cuadrada | magnitud_cuadrada: 0.0124 |
| 99% | magnitud_ponderada_T2 | magnitud_ponderada_T2: 0.0108 |
| 99% | tres_scores | reconstruccion_Q: 0.0111, magnitud_cuadrada: 9.8875, magnitud_ponderada_T2: -9.8759 |

### Rango de los scores sin estandarizar

| Objetivo | Score | Train mínimo | Train máximo | Val mínimo | Val máximo |
| ---: | --- | ---: | ---: | ---: | ---: |
| 90% | reconstruccion_Q | 0.000764 | 178 | 0.000738 | 139 |
| 90% | magnitud_cuadrada | 1.75 | 5.44e+04 | 2.34 | 4.66e+05 |
| 90% | magnitud_ponderada_T2 | 1.36 | 5.47e+04 | 1.19 | 4.68e+05 |
| 95% | reconstruccion_Q | 0.000282 | 22.6 | 0.000294 | 16.8 |
| 95% | magnitud_cuadrada | 2.21 | 5.44e+04 | 2.36 | 4.66e+05 |
| 95% | magnitud_ponderada_T2 | 1.51 | 5.47e+04 | 1.22 | 4.68e+05 |
| 99% | reconstruccion_Q | 6.28e-05 | 0.000529 | 5.31e-06 | 0.000494 |
| 99% | magnitud_cuadrada | 2.83 | 5.44e+04 | 3.1 | 4.66e+05 |
| 99% | magnitud_ponderada_T2 | 2.05 | 5.47e+04 | 2.26 | 4.68e+05 |

### Cobertura de categorías

| Variable | Categorías conocidas | Nulos legítimos train | Desconocidos train | Desconocidos val | Nulos val |
| --- | ---: | ---: | ---: | ---: | ---: |
| g | 40 | 99 | 5 | 5 | 43 |
| j | 7489 | 0 | 70 | 534 | 0 |
| o | 2 | 74442 | 0 | 0 | 16379 |
| p | 2 | 0 | 0 | 0 | 0 |

Los nulos reciben su frecuencia propia en legítimos de train; categorías no vistas reciben 0. La frecuencia de train incluye cada legítimo en su propia referencia.

## Lectura

- Q es la suma de residuos cuadrados en el espacio estandarizado; la magnitud es la suma de coordenadas PCA al cuadrado; T² divide cada término por su varianza.
- PR-AUC es el área trapezoidal bajo la curva precisión-recall; average precision usa otra integración. La prevalencia orienta la comparación con un ranking aleatorio.
- El signo del coeficiente indica si la logística asoció scores mayores con más o menos fraude en train; un signo negativo invierte el ranking.
- La logística es un evaluador supervisado, no parte del detector one-class. No mide el aporte incremental al modelo final y sus probabilidades no están necesariamente calibradas.
- Los scores de train son calculados por un PCA ajustado sobre sus legítimos: pueden ser optimistas frente a observaciones nuevas. Las comparaciones son exploratorias y usan la misma validación.
- Confirmar que todas las variables estén disponibles al momento de la transacción; no asumir causalidad por figurar en el baseline.
