# Experimentos PCA — validación temporal

- Entrada: `data\processed\dataset_base.parquet`; semilla: 42.
- Variables numéricas: score, monto, a, b, d, e, f, h, l, m, n, q, r, s.
- Train: 105,000 transacciones; 99,554 legítimas y 5,446 fraudes.
- Validación: 22,500 transacciones; 1,020 fraudes; prevalencia 4.533%.
- Imputación mediana, escalado y PCA ajustados solo con legítimos de train. La regresión logística usa ambas clases de train.
- Cada logística utiliza un solo score, estandarizado con legítimos de train. Test queda reservado.

| Varianza objetivo | Componentes | Varianza real | Score | Coeficiente | PR-AUC | Average precision | ROC-AUC |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 95% | 12 | 97.47% | reconstruccion_Q | -0.2307 | 0.0449 | 0.0450 | 0.5253 |
| 95% | 12 | 97.47% | magnitud_cuadrada | 0.0108 | 0.0905 | 0.0908 | 0.6751 |
| 95% | 12 | 97.47% | magnitud_ponderada_T2 | 0.0095 | 0.0814 | 0.0818 | 0.6732 |
| 99% | 13 | 100.00% | reconstruccion_Q | 0.0030 | 0.0468 | 0.0469 | 0.5019 |
| 99% | 13 | 100.00% | magnitud_cuadrada | 0.0107 | 0.0893 | 0.0897 | 0.6743 |
| 99% | 13 | 100.00% | magnitud_ponderada_T2 | 0.0093 | 0.0778 | 0.0781 | 0.6659 |

### Rango de los scores sin estandarizar

| Objetivo | Score | Train mínimo | Train máximo | Val mínimo | Val máximo |
| ---: | --- | ---: | ---: | ---: | ---: |
| 95% | reconstruccion_Q | 0.000263 | 19 | 0.000287 | 11.1 |
| 95% | magnitud_cuadrada | 0.447 | 5.44e+04 | 0.747 | 4.66e+05 |
| 95% | magnitud_ponderada_T2 | 0.467 | 5.47e+04 | 0.857 | 4.68e+05 |
| 99% | reconstruccion_Q | 6.19e-05 | 0.000526 | 5.37e-06 | 0.00049 |
| 99% | magnitud_cuadrada | 1.3 | 5.44e+04 | 1.52 | 4.66e+05 |
| 99% | magnitud_ponderada_T2 | 1.44 | 5.47e+04 | 1.22 | 4.68e+05 |

## Lectura

- Q es la suma de residuos cuadrados en el espacio estandarizado; la magnitud es la suma de coordenadas PCA al cuadrado; T² divide cada término por su varianza.
- PR-AUC es el área trapezoidal bajo la curva precisión-recall; average precision usa otra integración. La prevalencia orienta la comparación con un ranking aleatorio.
- El signo del coeficiente indica si la logística asoció scores mayores con más o menos fraude en train; un signo negativo invierte el ranking.
- La logística es un evaluador supervisado, no parte del detector one-class. No mide el aporte incremental al modelo final y sus probabilidades no están necesariamente calibradas.
- Los scores de train son calculados por un PCA ajustado sobre sus legítimos: pueden ser optimistas frente a observaciones nuevas. Las comparaciones son exploratorias y usan la misma validación.
- Confirmar que todas las variables estén disponibles al momento de la transacción; no asumir causalidad por figurar en el baseline.
