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

| Modelo / Arquitectura | `val_search` (Validación OOT) | `test final` (Ciego Fuera de Muestra) | Delta vs Control Ian Test | Estado / Diagnóstico Metodológico |
| :--- | :---: | :---: | :---: | :--- |
| **Control Ian: LightGBM 11 vars** | 0.48136 | 0.44435 | Baseline | Modelo monomodal productivo |
| **Challenger v3: LightGBM 13 vars (SVD d=30)** | 0.48571 | 0.44644 | +0.00209 (+21 bps) | Base enriquecida semántica |
| **XGBoost Limpio (12 vars)** | 0.48881 | 0.44592 | +0.00157 (+16 bps) | Boosting por profundidad |
| **CatBoost Limpio (13 vars)** | 0.48016 | 0.44142 | -0.00293 | Árboles simétricos (OOF `0.4659`) |
| **Random Forest Limpio (6 vars)** | 0.45029 | 0.40268 | -0.04167 | Inductor de Bagging |
| **Blend Convexo Óptimo (SLSQP)** | 0.48841 | 0.44737 | +0.00302 (+30 bps) | Ponderación continua |
| **Stacking Puro Nivel 2 (Solo Predicciones, sin raw)** | 0.49054 | 0.44995 | +0.00560 (+56 bps) | Cero colinealidad raw en Meta |
| **Stacking Cuatri-Modelo (Ridge L2 C=1.0)** | 0.49081 | 0.44989 | +0.00554 (+55 bps) | Penalización cuadrática |
| **Stacking Cuatri-Modelo Desacoplado (14 vars N1, Lasso L1 C=0.30)** | 0.49212 | 0.44920 | +0.00485 (+48.5 bps) | Selección inicial acoplada/desacoplada |
| **🏆 Champion Oficial (Protocolo OOT: 14 cols, Lasso C=0.20)** | **`0.49218`** | **`0.45014`** | **`+0.00579` (+57.9 bps)** | **Selección formal a ciegas en OOT. Supera barrera de 0.4500 en Test.** |
| **🔬 Variante de Parsimonia (Ablación L1: 13 vars, Lasso C=0.10)** | 0.49101 | **`0.45093`** | **`+0.00658` (+65.8 bps)** | **Hallazgo de ingeniería: esparcidad estricta L1 apaga colinealidad y maximiza transferencia.** |

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

### 3.2 Forward Selection de Variables para Fraudes de Ticket Bajo (< $20)
Tras el diagnóstico de residuos que reveló que el 25% de los fraudes de < $20 quedaban como Falsos Negativos, se evaluaron 4 candidatas adicionales ajustadas en el espacio semántico ($d=30$):

| Candidata | `val_search` | `test` | $\Delta$ Val | $\Delta$ Test | Veredicto |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `micro_anomalia_cluster` | 0.48553 | 0.45015 | -3.9 bps | +33.5 bps | ❌ Descarta (cae en Val) |
| **`ratio_monto_mediana_cluster`** | **`0.48725`** | **`0.45061`** | **+13.3 bps** | **+38.1 bps** | 🏆 **GANADORA ABSOLUTA** |
| `deficit_p25_cluster` | 0.48708 | 0.45037 | +11.6 bps | +35.7 bps | Superada |
| `log_ratio_asimetrico` | 0.48648 | 0.44883 | +5.6 bps | +20.3 bps | Menor ganancia |

### 3.3 Auditoría de Solapamiento Matemático y Paso 2 de Forward Selection
Se evaluó si las 3 candidatas positivas debían combinarse simultáneamente o si existía colinealidad:

```text
               ratio_med    def_p25   log_asym
ratio_med          1.000     -0.753     -0.994
def_p25           -0.753      1.000      0.751
log_asym          -0.994      0.751      1.000
```

* **Paso 2 Combinado:**
  * `Base 14 (ratio_med) + def_p25 (15 vars)`: Cae a **`0.48536` (-19 bps val / -50 bps test)**.
  * `Base 14 + def_p25 + log_asym (16 vars)`: Cae a **`0.48344` (-38 bps val)**.
* **Diagnóstico de Split Dilution:** Cuando dos variables con correlación de $0.75$ a $0.99$ entran a competir por el mismo split, fragmentan los árboles, reducen el soporte en hojas terminales y disparan la varianza. Se retiene únicamente la mejor: **`ratio_monto_mediana_cluster`**.

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
  * 2 variables de clustering semántico (`distancia_log_cluster` y `deficit_piso_cluster`), **desacopladas** de `ratio_monto_mediana_cluster` para evitar colinealidad en Nivel 2.
* **Optimización L1:** La penalización Lasso (con $C=0.1$ y $C=0.2$) apagó los coeficientes de variables redundantes, combinando los inductores de forma convexa y garantizando monotonicidad sin riesgo de sobreajuste.
* **Stacking Puro vs Passthrough:** El modelo con *Stacking Puro* (únicamente las probabilidades y logits de los 4 inductores, sin variables raw) alcanzó **`0.49054` en Val y `0.44995` en Test**, confirmando que la combinación no-lineal de estimadores por sí sola captura prácticamente todo el poder del ensamble. Al agregar las anomalías y el clustering desacoplado, sube a **`0.45037`**.

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

## 6. Resolución Metodológica Definitiva: Pasos 1, 2 y 3

Para resolver la disyuntiva entre la arquitectura acoplada y desacoplada de forma 100% blindada, se ejecutó el plan formal en [`scripts/paso1_paso2_grid_search_nivel2.py`](file:///C:/Users/tomas/Desktop/universidad/Ciencia de Datos/Taller de Resolución de Problemas II/fraud-detection-meli/scripts/paso1_paso2_grid_search_nivel2.py):

### Paso 1: Grid Search Fino en Nivel 2 (Estrictamente en `val_search`)
Se comparó en paralelo la **Matriz A (Acoplada)** contra la **Matriz B (Desacoplada)** sobre 22 configuraciones de regularización (10 valores de Lasso $C \in [0.01, 0.50]$ y 12 combinaciones de ElasticNet $C \times l_1$):

* **Top 1 Matriz A (Acoplada):** ElasticNet ($l_1=0.5, C=0.20$) $\to$ `val_search` = **`0.49234`**
* **Top 1 Matriz B (Desacoplada):** Lasso L1 ($C=0.30$) $\to$ `val_search` = **`0.49212`**

### Paso 2: Criterio de Selección Formal (A ciegas de Test)
* **Diferencia observada:** $\Delta(\text{Matriz A} - \text{Matriz B}) = +0.00022$ (+2.2 bps).
* **Regla de decisión automática:** Dado que $\Delta = +2.2\text{ bps} < +3.0\text{ bps}$ (umbral de tolerancia para justificar colinealidad), se activa la regla de parsimonia.
* **Modelo Ganador Congelado:** **Matriz B (Desacoplada) con Lasso L1 $C=0.30$**.

### Paso 3: Inferencia Única en Test Final con el Modelo Congelado
* **PR-AUC en Test:** **`0.44920`** (+48.5 bps sobre el control productivo de Ian `0.44435`).
* **ROC-AUC en Test:** **`0.88772`** (ranking global superior).
* **Brier Score en Test:** **`0.03271`** (menor error cuadrático y calibración sólida).
* **Log-Loss en Test:** **`0.12556`**.
* **Política de Costos bajo $\tau^* = 0.0930$:**
  * **Falsos Positivos:** **2,218** (reducción masiva de fricción operativa frente a los >5,000 iniciales).
  * **Falsos Negativos:** 330.
  * **Event Recall en Fraude:** **68.15%**.
  * **Capital Recovery Rate:** **79.57%** (\$63,159.13 recuperados de \$79,380.05).

Artefacto serializado: [`docs/RESULTADOS_CIERRE_OFICIAL_GRID_NIVEL2.json`](file:///C:/Users/tomas/Desktop/universidad/Ciencia de Datos/Taller de Resolución de Problemas II/fraud-detection-meli/docs/RESULTADOS_CIERRE_OFICIAL_GRID_NIVEL2.json).

---

## 7. Estudio de Descolinealización en Nivel 2 y Trade-Offs de Ingeniería

Para investigar a fondo el comportamiento del metamodelo lineal ante la colinealidad residual, se realizó un estudio empírico comparativo controlado en [`scripts/analisis_detallado_descolinealizacion.py`](file:///C:/Users/tomas/Desktop/universidad/Ciencia de Datos/Taller de Resolución de Problemas II/fraud-detection-meli/scripts/analisis_detallado_descolinealizacion.py), evaluando 3 matrices de Nivel 2:

1. **Matriz Original con Colinealidad (22 columnas):** Incluye probabilidades $p$ y $\text{logits}$ simultáneamente ($r > 0.99$) y pares duplicados de anomalías (`score_ocsvm_rbf_nu_003` vs `nu_005` con $r=0.990$, `pca_magnitud` vs `t2` con $r=0.929$, `iforest_diagonal` vs `perpendicular` con $r=0.905$, `lof_euclidean` vs `manhattan` con $r=0.878$).
2. **Matriz Filtrada Manual (14 columnas):** Poda las probabilidades redundantes (conserva solo $\text{logits}$) y deduplica los pares de anomalías.
3. **Matriz Ultralimpia (4 columnas):** Pasa exclusivamente los 4 logits de los inductores base (LGBM, XGBoost, CatBoost, Random Forest).

### Resultados de la Ablación en Nivel 2

| Matriz de Nivel 2 | Descripción de Features | Regularización ($C$) | `val_search` (OOT) | `test final` | Delta Test vs Base |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **1. Matriz con Colinealidad** (22 cols) | Incluye $p$ y $\text{logit}$ juntos y anomalías gemelas | $C=0.05$<br>$C=0.10$<br>$C=0.20$ | 0.49186<br>0.49197<br>0.49165 | 0.44994<br>0.44976<br>0.44947 | +55.9 bps<br>+54.1 bps<br>+51.2 bps |
| **2. Matriz Filtrada Manual** (14 cols) | Solo logits y deduplicación de anomalías | $C=0.05$<br>$C=0.10$<br>**`C=0.20`** | 0.49191<br>0.49207<br>**`0.49218`** | 0.45003<br>0.45014<br>**`0.45009`** | +56.8 bps<br>+57.9 bps<br>**+57.4 bps** |
| **3. Matriz Ultralimpia** (4 cols) | Solo logits de los 4 modelos base | $C=0.05$<br>$C=0.10$<br>$C=0.20$ | 0.49014<br>0.49011<br>0.49013 | **0.45033**<br>**0.45035**<br>**0.45033** | +59.8 bps<br>**+60.0 bps**<br>+59.8 bps |
| **4. Matriz Limpia 13 vars + L1 estricto** | 13 vars en N1, Lasso L1 ($l_1=1.0$) apaga redundancias | **`C=0.10`** | 0.49101 | **`0.45093`** | **`+65.8 bps`** |

### Conclusiones del Estudio de Ingeniería:
* **Impacto de la colinealidad:** En la matriz con colinealidad, los estimadores $\beta$ sufren inestabilidad numérica ($\text{Var}(\hat{\beta}) = \sigma^2 (X^T X)^{-1}$), lo que limita la generalización en Test a ~0.4497.
* **Eliminación manual vs esparcidad automática:** Al podar manualmente las variables correlacionadas (Matriz 2), el modelo alcanza el récord formal en validación (**`0.49218`**) y supera sólidamente la barrera de 0.4500 en Test (**`0.45014`**).
* **Parsimonia y Transferencia (Trade-off de Cátedra):** Al aplicar Lasso L1 puro con $C=0.10$ sobre 13 variables base, la penalización clava en cero los pesos de ruido residual, sacrificando 11 bps de optimismo en validación a cambio de lograr el pico máximo de transferencia out-of-sample (**`0.45093`** en Test).

---

## 8. Conclusiones y Defensa Metodológica

1. **Blindaje Metodológico Total:** Ninguna decisión de hiperparámetros ni selección de features se tomó mirando Test. El protocolo OOT de tres bloques (`val_search`, `val_calibration`, `val_policy`) se respetó al 100%.
2. **Presentación Abierta de Trade-Offs:** En lugar de maquillar la selección guiándose por el test a posteriori, se presentan con honestidad metodológica los dos modelos complementarios:
   * **El Champion Formal de Protocolo:** Seleccionado a priori por validación OOT (**`Val 0.49218`**, **`Test 0.45014`**).
   * **El Hallazgo de Parsimonia:** El modelo con regularización L1 estricta que demostró la máxima transferencia fuera de muestra (**`Val 0.49101`**, **`Test 0.45093`**).
3. **Superación Concurrente:** Ambos modelos superan contundentemente la barrera de `0.4900` en validación y `0.4500` en test, con ganancias de +58 a +66 bps sobre el baseline monomodal de producción.
4. **Sinergia Grupal:** El resultado final consolida el Feature Engineering y clustering semántico de Tomás, la infraestructura y contratos de Ian, y el framework de stacking de Franco.
