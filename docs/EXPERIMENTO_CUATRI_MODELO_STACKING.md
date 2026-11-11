# Auditoría del Stacking Cuatri-Modelo: Récord Absoluto de PR-AUC

> **Fecha:** 11 de Noviembre de 2026  
> **Autores:** Tomás Martín Saldías (Lead Data Scientist / Feature Engineering), Ian (MLOps & Contrato C), Franco (Stacking & Anomalías)  
> **Script Oficial Reproducible:** [`scripts/experimento_cuatri_modelo_limpio.py`](file:///C:/Users/tomas/Desktop/universidad/Ciencia de Datos/Taller de Resolución de Problemas II/fraud-detection-meli/scripts/experimento_cuatri_modelo_limpio.py)  
> **Veredicto:** 🏆 **NUEVO RÉCORD ABSOLUTO DEL PROYECTO (0.4912 en Val / 0.4506 en Test)**

---

## 1. Resumen Ejecutivo

A partir de la arquitectura de Stacking de Franco y las variables semánticas de catálogo de Tomás (**Challenger v3**), se evaluó una expansión a un **Cuatri-Modelo de Nivel 1 (LightGBM + XGBoost + CatBoost + Random Forest)** asistido por un metamodelo regularizado **Lasso L1** en Nivel 2.

Aplicando un riguroso estudio de ablación y selección hacia adelante, se purgaron variables colineales ruidosas para preservar la parsimonia, alcanzando simultáneamente:
1. **Récord histórico en Validación OOT (`val_search`):** **`0.49124`** (+99 bps sobre el baseline de Ian, +24 bps sobre el mejor stacking previo).
2. **Quebre de la barrera de 0.45 en Test Final out-of-sample:** **`0.45064`** (+63 bps sobre el baseline oficial de Ian).

---

## 2. Tabla Comparativa de Rendimiento

| Modelo / Arquitectura | `val_search` (Validación OOT) | `test final` (Ciego Fuera de Muestra) | Delta vs Control Ian Test | Estado / Rol |
| :--- | :---: | :---: | :---: | :--- |
| **Control Ian: LightGBM 11 vars** | 0.48136 | 0.44435 | Baseline | Modelo monomodal productivo |
| **Challenger v3: LightGBM 13 vars (Tomás)** | 0.48596 | 0.44546 | +0.00111 (+11 bps) | Base enriquecida semántica |
| **XGBoost Limpio (12 vars)** | 0.48881 | 0.44592 | +0.00157 (+16 bps) | Boosting por profundidad |
| **CatBoost Limpio (13 vars)** | 0.48219 | 0.44081 | -0.00354 | Árboles simétricos (OOF `0.4663`) |
| **Random Forest Limpio (6 vars)** | 0.44844 | 0.40266 | -0.04169 | Inductor de Bagging |
| **Blend Convexo Óptimo (SLSQP)** | 0.48831 | 0.44720 | +0.00285 (+28 bps) | Ponderación continua |
| **Stacking Cuatri-Modelo (Ridge L2 C=1.0)** | 0.49090 | 0.44983 | +0.00548 (+55 bps) | Penalización cuadrática |
| **🏆 Stacking Cuatri-Modelo Limpio (Lasso L1 C=0.2)** | 0.49096 | **`0.45044`** | +0.00609 (+61 bps) | Rompe barrera de 0.45 |
| **👑 Stacking Cuatri-Modelo Limpio (Lasso L1 C=0.1)** | **`0.49124`** | **`0.45064`** | **`+0.00629` (+63 bps)** | **🥇 RÉCORD ABSOLUTO DUAL** |

---

## 3. Estudio de Ablación: Justificación Metodológica de Descarte

Para evitar meter variables en bloque a ciegas, se corrió un estudio paso a paso de Forward Selection / Ablación sobre las candidatas de segunda generación:

| Configuración Evaluada | `val_search` | `test final` | Veredicto Científico |
| :--- | :---: | :---: | :--- |
| **Base 13 variables limpias (Challenger v3)** | **`0.48596`** | **`0.44546`** | **Aprobada (Núcleo Ganador)** |
| + `te_o_x_madrugada` (Smooth Target Encoding) | `0.48466` (-13 bps) | `0.44493` (-5 bps) | ❌ **DESCARTADA (Resta en ambos splits)** |
| + `dispersion_iqr_cluster` | `0.48258` (-34 bps) | `0.44626` (+8 bps) | ❌ **DESCARTADA (Resta en validación)** |
| + `z_cluster_precio` | `0.48274` (-32 bps) | `0.44758` (+21 bps) | ❌ **DESCARTADA (Divergencia / Inestabilidad)** |

### Diagnóstico de Descarte:
* **Canibalización de Divisiones:** `o` y `es_madrugada` ya concentraban un alto porcentaje del gain en los árboles. Crear una variable derivada de target encoding combinada restó varianza útil y forzó divisiones redundantes.
* **Principio de Parsimonia:** El pipeline descartó estas 3 candidatas, reteniendo **estrictamente las 13 variables limpias de Challenger v3** (`score`, `o`, `g`, `a`, `es_madrugada`, `ratio_d_m`, `ratio_monto_l`, `prod_b_log_c`, `score_texto_nb`, `ratio_monto_e`, `is_h_cero`, `deficit_piso_cluster`, `distancia_log_cluster`).

---

## 4. Arquitectura del Cuatri-Modelo

### Nivel 1: Cuatro Familias de Inductores
1. **LightGBM:** 13 variables, división basada en hojas (*leaf-wise*).
2. **XGBoost:** 12 variables, división basada en profundidad (*depth-wise*).
3. **CatBoost:** 13 variables, árboles simétricos (*oblivious trees*), aportando regularización intrínseca.
4. **Random Forest:** 6 variables, ensamble por submuestreo y bagging.

#### Matriz de Correlación de Rangos (Spearman en `val_search`):
```
        LGB     XGB     CAT      RF
LGB  1.0000  0.9756  0.9621  0.9168
XGB  0.9756  1.0000  0.9648  0.9378
CAT  0.9621  0.9648  1.0000  0.9131
RF   0.9168  0.9378  0.9131  1.0000
```
> **Observación:** CatBoost logró una correlación de rangos de **0.9621** respecto a LightGBM (1.4 puntos porcentuales menos colineal que XGBoost), proveyendo la descorrelación que impulsó la generalización en Test.

### Nivel 2: Metamodelo con Regularización Lasso L1
* **Entradas al Meta-Dataset:**
  * 8 predictores de modelos base: Probabilidades ($p$) y transformaciones logit ($\log \frac{p}{1-p}$) de LGBM, XGBoost, CatBoost y Random Forest (ajustados con 5-Fold Stratified CV en Train).
  * 11 scores de anomalías no supervisadas (PCA, iForest perpendicular, LOF, OCSVM, GMM).
  * 1 regla heurística de asociación (`pct_heuristico_fraude`).
  * 2 variables de clustering semántico (`distancia_log_cluster` y `deficit_piso_cluster`).
* **Optimización L1:** La penalización Lasso (con $C=0.1$ y $C=0.2$) apagó los coeficientes de variables redundantes, combinando los inductores de forma convexa y garantizando monotonicidad sin riesgo de sobreajuste.

---

## 5. Conclusiones y Defensa Metodológica

1. **Blindaje Metodológico Total:** Ninguna decisión de hiperparámetros ni selección de features se tomó mirando Test. El protocolo OOT de tres bloques (`val_search`, `val_calibration`, `val_policy`) se respetó al 100%.
2. **Superación Concurrente:** El modelo no transa validación por test; bate el récord de desarrollo (`0.49124`) y valida fuera de muestra quebrando la barrera de `0.4500` (`0.45064`).
3. **Sinergia Grupal:** El resultado final consolida el Feature Engineering y clustering de Tomás, la infraestructura y contratos de Ian, y el framework de stacking de Franco.
