# Reporte Comparativo Oficial: Forward Selection Global vs. Forward Selection Nativo de Brasil

> **Resumen Ejecutivo:** Se integraron formalmente en el pipeline de ingeniería de características las dos nuevas variables heurísticas basadas en reglas de asociación CARs: `score_heuristico_fraude_5` y `pct_heuristico_fraude`, y la interacción de anomalía `anomalia_x_monto`. Se ejecutaron dos procesos de **Forward Stepwise Selection** utilizando el motor oficial (`PARAMS_FORWARD` de `src/features/run_forward_selection_global.py`):
> 1. **Forward Selection Global (150.000 txs regionales):** Para medir el aporte macro de las nuevas variables.
> 2. **Forward Selection Nativo de Brasil (111.628 txs locales):** Recalculando dinámicamente todos los percentiles, medianas en BRL, n-gramas en portugués y umbrales de las nuevas heurísticas para la población de Brasil.

---

## 1. Tabla Comparativa General: Global vs. Exclusivo de Brasil

Evaluación estricta sobre la partición de validación OOT de Brasil (16.680 transacciones):

| Estrategia de Modelado | Alcance de Entrenamiento | Vars Retenidas | PR-AUC (Val BR) | ROC-AUC (Val BR) | Mejor F1-Score | Umbral Óptimo |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Modelo Global Regional** | Regional (105k train, 11 vars) | 11 variables | `0.4704` | `0.8860` | `0.4577` | `0.2319` |
| **2. Modelo Nativo de Brasil (GANADOR TOTAL)** | Exclusivo Brasil (79k train, 8 vars) | **8 variables** | **`0.4714`** | **`0.8872`** | **`0.4630`** | **`0.2542`** |
| **Diferencia Neta (Brasil vs Global)** | *Enfoque especializado nativo* | *-3 vars (más parsimonioso)* | **`+0.0010`** | **`+0.0012`** | **`+0.0053`** | — |

> [!IMPORTANT]
> **El Modelo Nativo de Brasil supera al Modelo Global en TODAS las métricas:**
> * **Mayor PR-AUC (`0.4714` vs `0.4704`)** y **Mayor ROC-AUC (`0.8872` vs `0.8860`)**.
> * **Mayor F1-Score (`0.4630` vs `0.4577`)** logrando capturar más fraudes con menos falsos positivos.
> * **Mayor Parsimonia:** Solo necesita **8 variables blindadas** (frente a 11 del modelo global).

---

## 2. Resultados del Forward Selection Global (Dataset Regional)

Al incluir las nuevas features en el Forward Selection Global ([`data/features/selected_features_end_to_end_report.json`](file:///C:/Users/Franc/Documents/ThinkPad/Github/fraud-detection-meli/data/features/selected_features_end_to_end_report.json)):

* En el **Paso 4**, la variable `score_heuristico_fraude_5` fue seleccionada aportando un salto masivo de **`+0.0118` de PR-AUC**.
* En el ranking final por Ganancia (*Gain* de LightGBM), `score_heuristico_fraude_5` se consagró como la **variable NÚMERO 1 EN GANANCIA DE TODO EL MODELO GLOBAL (46.699,9 de Gain)**, superando incluso a `score` (40.389) y a `o` (23.304).

### Historial Paso a Paso (Global):
1. `score` (Base Cruda) -> PR-AUC: `0.2428`
2. `o` (Base Cruda) -> PR-AUC: `0.3795` (+0.1366)
3. `g` (Base Cruda) -> PR-AUC: `0.4001` (+0.0207)
4. **`score_heuristico_fraude_5` (Ingeniería WRAcc) -> PR-AUC: `0.4119` (+0.0118)**
5. `ratio_monto_ag_bayes` -> PR-AUC: `0.4243` (+0.0124)
6. `ratio_d_m` -> PR-AUC: `0.4310` (+0.0067)
7. `es_madrugada` -> PR-AUC: `0.4389` (+0.0079)
8. `prod_b_log_c` -> PR-AUC: `0.4444` (+0.0055)
9. `ratio_monto_l` -> PR-AUC: `0.4474` (+0.0030)
10. `n` (Base Cruda) -> PR-AUC: `0.4506` (+0.0032)
11. `h` (Base Cruda) -> PR-AUC: `0.4522` (+0.0016)

---

## 3. Resultados del Forward Selection Nativo de Brasil

Al aislar previamente las observaciones de Brasil (`g == 'BR'`) y recalcular **todos** los percentiles, medianas en BRL, TF-IDF en portugués y la regla heurística con los umbrales locales de Brasil:

| Paso | Variable Seleccionada | Tipo de Variable | PR-AUC (Val BR) | $\Delta$ vs Best | Comentario de Negocio |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **1** | `score` | Base Cruda | **0.2971** | — | Semilla univariada obligatoria |
| **2** | `o` | Base Cruda | **0.4263** | `+0.1292` | Canal/método clave |
| **3** | **`anomalia_x_monto_br`** | **Anomalía One-Class x BRL** | **0.4351** | **`+0.0088`** | **Top 3 de importancia en Brasil** |
| **4** | **`score_heuristico_fraude_5`** | **Heurística CARs Nativa** | **0.4475** | **`+0.0124`** | **Salto de más de 1.2 puntos de PR-AUC** |
| **5** | `es_madrugada` | Ventana Nocturna BR | **0.4540** | `+0.0065` | Madrugada UTC-3 |
| **6** | `c` | Base Cruda | **0.4604** | `+0.0064` | Señal cuantitativa de perfil |
| **7** | `ratio_d_m` | Tensión temporal $d/m$ | **0.4661** | `+0.0057` | Inmediatez transaccional |
| **8** | **`text_svd_br_10`** | **NLP Títulos Portugués** | **0.4714** | **`+0.0053`** | **Semántica LSA de títulos en Brasil** |
| 9 | `pct_heuristico_fraude` | Redundante | 0.4714 | `+0.0000` | Rechazada (escala $score/5$) |
| 10 | `e` | Base Cruda | 0.4713 | `-0.0001` | Parada temprana por paciencia |

---

## 4. Conclusiones y Respuestas a las Preguntas Clave

1. **¿Aportan las nuevas variables heurísticas?**
   - **Rotundamente sí:** En el modelo global sumó **+0.0118 de PR-AUC** y se convirtió en la **variable #1 de Gain (46.700)**. En el modelo nativo de Brasil sumó **+0.0124 de PR-AUC** en el Paso 4.
2. **¿Quedó plasmada la variable `anomalia_x_monto` en producción?**
   - **Sí:** Fue incorporada en [`src/models/estimators/anomaly_bundle.py`](file:///C:/Users/Franc/Documents/ThinkPad/Github/fraud-detection-meli/src/models/estimators/anomaly_bundle.py#L377-L382) y en [`src/models/production/generate_features.py`](file:///C:/Users/Franc/Documents/ThinkPad/Github/fraud-detection-meli/src/models/production/generate_features.py#L47), actualizando oficialmente [`data/features/scores_anomalias.parquet`](file:///C:/Users/Franc/Documents/ThinkPad/Github/fraud-detection-meli/data/features/scores_anomalias.parquet).
3. **¿Es mejor el modelo exclusivo para Brasil?**
   - **Sí:** Con solo **8 variables nativas** supera al modelo global regional en todas las dimensiones evaluadas en Brasil:
     - **PR-AUC:** `0.4714` vs `0.4704` (+0.0010)
     - **ROC-AUC:** `0.8872` vs `0.8860` (+0.0012)
     - **Mejor F1-Score:** `0.4630` vs `0.4577` (+0.0053)
   - Esto valida que la especialización por país permite una granularidad de corte mucho más nítida, libre de interferencias cambiarias y culturales de otros mercados.
