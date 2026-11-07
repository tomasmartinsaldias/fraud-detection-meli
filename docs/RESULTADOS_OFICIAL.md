# Resultados oficiales de modelado — Contrato C

Corrida: `models\supervised\oficial_v2`. Semilla: 42. Features: `features_seleccionadas_end_to_end.parquet`.
Modelo elegido **solo con val_search**: `lgb_full_cmaes`.

## Comparación de búsqueda (val_search)

| Modelo | PR-AUC | AP | ROC-AUC |
|---|---:|---:|---:|
| lgb_full_cmaes | 0.4877 | 0.4880 | 0.8895 |
| lgb_full_tpe | 0.4865 | 0.4868 | 0.8860 |
| lgb_transactional_cmaes | 0.4857 | 0.4860 | 0.8913 |
| lgb_transactional_tpe | 0.4855 | 0.4858 | 0.8929 |
| lgb_reference_transactional | 0.4827 | 0.4830 | 0.8922 |
| average_ensemble | 0.4818 | 0.4821 | 0.8911 |
| lgb_reference_full | 0.4789 | 0.4792 | 0.8918 |
| lgb_raw_tpe | 0.4614 | 0.4618 | 0.8902 |
| lgb_raw_cmaes | 0.4565 | 0.4569 | 0.8867 |
| lgb_reference_raw | 0.4483 | 0.4488 | 0.8882 |
| logistic_raw | 0.3319 | 0.3328 | 0.7789 |

## Búsquedas (TPE + pruning vs. CMA-ES, mismo presupuesto)

| Estudio | Trials | Podados | Mejor PR-AUC | Segundos |
|---|---:|---:|---:|---:|
| trials_full_cmaes | 20 | 10 | 0.4877 | 37.2 |
| trials_full_tpe | 20 | 10 | 0.4865 | 43.8 |
| trials_raw_cmaes | 20 | 12 | 0.4565 | 26.9 |
| trials_raw_tpe | 20 | 10 | 0.4614 | 30.4 |
| trials_transactional_cmaes | 20 | 15 | 0.4857 | 26.0 |
| trials_transactional_tpe | 20 | 12 | 0.4855 | 34.3 |

## Comparación del contrato (probabilidades calibradas)

Para cada nivel se toma el mejor LightGBM de búsqueda según val_search (sin mirar test).

| Nivel | Modelo | PR-AUC val_policy | ROC-AUC val_policy | PR-AUC test | ROC-AUC test |
|---|---|---:|---:|---:|---:|
| Baseline (logística, crudas) | `logistic_raw` | 0.2053 | 0.7376 | 0.3150 | 0.8075 |
| Con features transaccionales | `lgb_transactional_cmaes` | 0.3137 | 0.8619 | 0.4413 | 0.8856 |
| Completo (+ scores de anomalías) | `lgb_full_cmaes` | 0.3122 | 0.8551 | 0.4432 | 0.8859 |

## Negocio — test final

Costo principal: FP=5, FN=200. Umbrales elegidos en val_policy, máximo 100 FP en ese bloque.

| Modelo | PR-AUC calibrado | Umbral | Precision | Recall | FP | FN | Costo | Ahorro vs permitir todo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic_raw | 0.3150 | 0.25132 | 0.415 | 0.334 | 487 | 690 | 140,435 | 66,765 |
| lgb_reference_raw | 0.4313 | 0.23101 | 0.493 | 0.416 | 443 | 605 | 123,215 | 83,985 |
| lgb_raw_tpe | 0.4337 | 0.22384 | 0.478 | 0.426 | 482 | 595 | 121,410 | 85,790 |
| lgb_raw_cmaes | 0.4354 | 0.21601 | 0.469 | 0.427 | 500 | 594 | 121,300 | 85,900 |
| lgb_reference_transactional | 0.4384 | 0.22183 | 0.485 | 0.423 | 466 | 598 | 121,930 | 85,270 |
| lgb_transactional_tpe | 0.4409 | 0.20920 | 0.469 | 0.439 | 516 | 581 | 118,780 | 88,420 |
| lgb_transactional_cmaes | 0.4413 | 0.21450 | 0.482 | 0.436 | 485 | 584 | 119,225 | 87,975 |
| lgb_reference_full | 0.4347 | 0.22091 | 0.485 | 0.434 | 477 | 586 | 119,585 | 87,615 |
| lgb_full_tpe | 0.4422 | 0.20792 | 0.482 | 0.434 | 484 | 586 | 119,620 | 87,580 |
| lgb_full_cmaes | 0.4432 | 0.22159 | 0.492 | 0.436 | 466 | 584 | 119,130 | 88,070 |
| average_ensemble | 0.4452 | 0.21871 | 0.479 | 0.448 | 505 | 572 | 116,925 | 90,275 |

## Interpretación

El modelo elegido detecta 43.6% de los fraudes con precision 49.2% en test final.
Su costo es 119,130, frente a 140,435 de la regresión logística bajo las políticas congeladas.
La diferencia de costo frente a logística es 21,305; un valor negativo significa que el ganador por PR-AUC cuesta más.
El ganador por PR-AUC no necesariamente minimiza el costo con una restricción operativa. No se cambia de ganador por observar test.
Calibración del ganador: Brier sin calibrar 0.032894, con Platt 0.032909.

### Sensibilidad del ganador

| Costo FP | Umbral congelado | FP | FN | Costo total |
|---|---:|---:|---:|---:|
| 5.0 | 0.22159 | 466 | 584 | 119,130 |
| 10.0 | 0.22159 | 466 | 584 | 121,460 |
| 20.0 | 0.22159 | 466 | 584 | 126,120 |

### Sensibilidad del contrato: C_FN = monto defraudado (solo val_policy)

Escenario de CONTRATOS.md: C_FP=5 y C_FN = monto de cada fraude no detectado. El umbral se elige y se mide
en val_policy, por lo que es optimista, y **no se evaluó en test** (decisión explícita para no reabrirlo).

| Modelo | Umbral | FP | FN | Precision | Recall | Pérdida + fricción | Ahorro vs permitir todo |
|---|---:|---:|---:|---:|---:|---:|---:|
| logistic_raw | 0.28021 | 78 | 115 | 0.350 | 0.268 | 6,660 | 3,229 |
| lgb_reference_raw | 0.23101 | 90 | 101 | 0.384 | 0.357 | 5,782 | 4,106 |
| lgb_raw_tpe | 0.22384 | 97 | 99 | 0.374 | 0.369 | 4,942 | 4,946 |
| lgb_raw_cmaes | 0.22677 | 90 | 102 | 0.379 | 0.350 | 7,636 | 2,252 |
| lgb_reference_transactional | 0.25113 | 80 | 101 | 0.412 | 0.357 | 5,155 | 4,733 |
| lgb_transactional_tpe | 0.24194 | 77 | 99 | 0.430 | 0.369 | 4,761 | 5,128 |
| lgb_transactional_cmaes | 0.21450 | 97 | 99 | 0.374 | 0.369 | 5,399 | 4,490 |
| lgb_reference_full | 0.22091 | 100 | 100 | 0.363 | 0.363 | 5,169 | 4,720 |
| lgb_full_tpe | 0.24971 | 81 | 104 | 0.396 | 0.338 | 5,204 | 4,685 |
| lgb_full_cmaes | 0.22878 | 93 | 101 | 0.376 | 0.357 | 5,205 | 4,683 |
| average_ensemble | 0.21871 | 97 | 97 | 0.382 | 0.382 | 5,190 | 4,699 |

### Variables principales (gain, no causalidad)

- `o`: 64,328.2.
- `score`: 59,981.3.
- `ratio_monto_l`: 15,460.0.
- `ratio_d_m`: 10,953.2.
- `g`: 8,303.8.
- `prod_b_log_c`: 7,951.0.
- `score_texto_nb`: 7,915.2.
- `score_iforest_perpendicular`: 6,242.7.

## Límites y próximos pasos

La selección previa de variables ya había consultado val; las cifras de búsqueda son exploratorias.
Entorno de la corrida: Python 3.12.14 con las dependencias fijadas en requirements.txt.
Scores de anomalía: entrega oficial de Franco (`scores_anomalias.parquet`), excluyendo `score_anomalia_consolidado` (consolidador supervisado ajustado in-sample sobre train).
Costos del período evaluado, sin extrapolación mensual. El tope de FP de validación no es una garantía para otro período.
Antes de producción: acordar capacidad real de revisión, monitorear calibración y deriva, medir costos de contacto y pérdida de clientes.
Datos adicionales deseables: identidad de usuario/dispositivo, historial causal, chargebacks confirmados y fecha de confirmación de etiqueta.
No usar resultados finales para una nueva ronda de ajuste: se necesitaría otro período fuera de muestra.

## Curvas

![Curva PR val_policy](figures_oficial/pr_val_policy.svg)

![Curva ROC val_policy](figures_oficial/roc_val_policy.svg)

![Curva PR test final](figures_oficial/pr_test.svg)

![Curva ROC test final](figures_oficial/roc_test.svg)
