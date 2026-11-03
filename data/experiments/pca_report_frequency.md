# Experimentos PCA — validación temporal

- Entrada: `data\processed\dataset_base.parquet`; semilla: 42.
- Variables numéricas: score, monto, a, b, c, d, e, f, h, k, l, m, n, q, r, s.
- Categóricas codificadas por frecuencia: g, j, o, p.
- Train: 105,000 transacciones; 99,554 legítimas y 5,446 fraudes.
- Validación: 22,500 transacciones; 1,020 fraudes; prevalencia 4.533%.
- Medianas, frecuencias, escalado y PCA ajustados solo con legítimos de train. La regresión logística usa ambas clases de train.
- Para cada PCA se prueban tres logísticas individuales y otra conjunta; scores estandarizados con legítimos de train. Test queda reservado.

| Varianza objetivo | Componentes | Varianza real | Indicadores | PR-AUC | Average precision | ROC-AUC |
| ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 95% | 15 | 95.32% | reconstruccion_Q | 0.0445 | 0.0449 | 0.4891 |
| 95% | 15 | 95.32% | magnitud_cuadrada | 0.0964 | 0.0967 | 0.7090 |
| 95% | 15 | 95.32% | magnitud_ponderada_T2 | 0.0930 | 0.0934 | 0.7157 |
| 95% | 15 | 95.32% | tres_scores | 0.0755 | 0.0758 | 0.5988 |
| 99% | 17 | 99.98% | reconstruccion_Q | 0.0674 | 0.0677 | 0.5555 |
| 99% | 17 | 99.98% | magnitud_cuadrada | 0.0932 | 0.0936 | 0.7025 |
| 99% | 17 | 99.98% | magnitud_ponderada_T2 | 0.0842 | 0.0846 | 0.6921 |
| 99% | 17 | 99.98% | tres_scores | 0.0854 | 0.0857 | 0.6174 |

### Coeficientes de las regresiones

| Objetivo | Modelo | Coeficientes por score estandarizado |
| ---: | --- | --- |
| 95% | reconstruccion_Q | reconstruccion_Q: 0.0057 |
| 95% | magnitud_cuadrada | magnitud_cuadrada: 0.0132 |
| 95% | magnitud_ponderada_T2 | magnitud_ponderada_T2: 0.0112 |
| 95% | tres_scores | reconstruccion_Q: -0.0009, magnitud_cuadrada: 2.8183, magnitud_ponderada_T2: -2.8031 |
| 99% | reconstruccion_Q | reconstruccion_Q: 0.0661 |
| 99% | magnitud_cuadrada | magnitud_cuadrada: 0.0132 |
| 99% | magnitud_ponderada_T2 | magnitud_ponderada_T2: 0.0108 |
| 99% | tres_scores | reconstruccion_Q: 0.0759, magnitud_cuadrada: 2.9204, magnitud_ponderada_T2: -2.9054 |

### Rango de los scores sin estandarizar

| Objetivo | Score | Train mínimo | Train máximo | Val mínimo | Val máximo |
| ---: | --- | ---: | ---: | ---: | ---: |
| 95% | reconstruccion_Q | 0.000398 | 86.1 | 0.000493 | 148 |
| 95% | magnitud_cuadrada | 2.45 | 5.45e+04 | 2.58 | 4.66e+05 |
| 95% | magnitud_ponderada_T2 | 1.67 | 5.47e+04 | 1.9 | 4.68e+05 |
| 99% | reconstruccion_Q | 0.00026 | 0.0364 | 0.000201 | 0.036 |
| 99% | magnitud_cuadrada | 2.98 | 5.45e+04 | 3.26 | 4.66e+05 |
| 99% | magnitud_ponderada_T2 | 2.47 | 5.47e+04 | 2.31 | 4.68e+05 |

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
