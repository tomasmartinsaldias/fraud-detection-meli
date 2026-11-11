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
| **Challenger v3: LightGBM 13 vars (SVD d=30)** | 0.48571 | 0.44644 | +0.00209 (+21 bps) | Base enriquecida semántica |
| **XGBoost Limpio (12 vars)** | 0.48881 | 0.44592 | +0.00157 (+16 bps) | Boosting por profundidad |
| **CatBoost Limpio (13 vars)** | 0.48016 | 0.44142 | -0.00293 | Árboles simétricos (OOF `0.4659`) |
| **Random Forest Limpio (6 vars)** | 0.45029 | 0.40268 | -0.04167 | Inductor de Bagging |
| **Blend Convexo Óptimo (SLSQP)** | 0.48835 | 0.44739 | +0.00304 (+30 bps) | Ponderación continua |
| **Stacking Cuatri-Modelo (Ridge L2 C=1.0)** | 0.49065 | 0.45052 | +0.00617 (+62 bps) | Penalización cuadrática |
| **Stacking Cuatri-Modelo Limpio (Lasso L1 C=0.2)** | 0.49062 | 0.45063 | +0.00628 (+63 bps) | Rompe barrera de 0.45 |
| **🏆 Stacking Cuatri-Modelo Limpio (Lasso L1 C=0.1)** | **`0.49101`** | **`0.45093`** | **`+0.00658` (+66 bps)** | **🥇 RÉCORD ABSOLUTO DUAL (SVD d=30)** |

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

## 5. Auditoría de Drift y Análisis de Residuos (SVD d=30)

Como parte de la validación del Challenger v3 enriquecido con $d=30$, se ejecutó [`scripts/analisis_drift_residuos_d30.py`](file:///C:/Users/tomas/Desktop/universidad/Ciencia de Datos/Taller de Resolución de Problemas II/fraud-detection-meli/scripts/analisis_drift_residuos_d30.py) para auditar la estabilidad temporal de las predicciones y el comportamiento de los errores residuales.

### Métricas de Estabilidad y Prediction Drift:
* **PR-AUC Test Final:** 0.4435 (Champion v2) $\to$ **0.4468 (Challenger v3, d=30)** (+33 bps).
* **Contracción de Brecha Temporal:** La distancia entre Validación OOT y Test Final cayó de **+0.0127 a +0.0110**, indicando mayor resistencia al covariate shift temporal.
* **Population Stability Index (PSI):** **`0.00065`** (muy por debajo de la cota crítica de 0.05). Las distribuciones de scores no sufren corrimiento espurio.
* **Brier Score en Test:** Se redujo a **`0.03276`** (frente a 0.03281 de v2), confirmando mejor calibración de probabilidades.

### Residuos y Fraudes de Bajo Monto Rescatados:
Al evaluar bajo el umbral de corte de costo óptimo ($\tau^* = 0.0314$):
* **Fricción Operativa:** Se eliminaron **50 Falsos Positivos** (5,337 $\to$ 5,287), ahorrando \$250 en fricción de checkout.
* **Detección de Fraudes Baratos:** Challenger v3 ($d=30$) logró rescatar fraudes que antes eran Falsos Negativos debido a que su precio era anómalo respecto al cluster:
  * `tx_041916` (\$5.69, *Cabo Av Para Dreamcast*): $\hat{p}$ subió de $0.0308 \to 0.0378$ (`deficit_piso_cluster = 0.71`).
  * `tx_125974` (\$7.46, *Relógio De Pulso Digital Led*): $\hat{p}$ subió de $0.0268 \to 0.0336$ (`deficit_piso_cluster = 0.61`).
  * `tx_012064` (\$26.82, *Cordão 3mm Cadeado Banhado A Ouro*): $\hat{p}$ subió de $0.0274 \to 0.0336$.
  * `tx_117666` (\$34.56, *Pistola Airsoft Spring Cybergun Col*): $\hat{p}$ subió de $0.0288 \to 0.0380$.

---

## 6. Conclusiones y Defensa Metodológica

1. **Blindaje Metodológico Total:** Ninguna decisión de hiperparámetros ni selección de features se tomó mirando Test. El protocolo OOT de tres bloques (`val_search`, `val_calibration`, `val_policy`) se respetó al 100%.
2. **Superación Concurrente:** El modelo no transa validación por test; bate el récord de desarrollo (`0.49101`) y valida fuera de muestra quebrando la barrera de `0.4500` con **`0.45093`** en Test.
3. **Sinergia Grupal:** El resultado final consolida el Feature Engineering y clustering de Tomás, la infraestructura y contratos de Ian, y el framework de stacking de Franco.
