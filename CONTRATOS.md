# CONTRATOS.md — Interfaces Técnicas, Roles y Criterios de Aceptación

> **Proyecto:** Detección de Fraude Transaccional (MELI)  
> **Equipo:** Tomás (Data Eng & Features), Franco (Detección de Anomalías), Ian (Modelado & Negocio)  
> **Dataset Base:** `data/processed/dataset_base.parquet`

---

## 1. Convenciones Globales del Repositorio

* **Identificador Universal Inmutable (`transaction_id`):**  
  Toda fila se identifica de manera unívoca mediante `transaction_id` (string formateado como `tx_000000` a `tx_249999`).  
  > [!IMPORTANT]
  > Ningún integrante puede alterar, reordenar físicamente el índice original, ni eliminar registros sin pasar por la capa de exclusión unificada. `transaction_id` es la única clave primaria para cruces (`merge`) entre tablas.

* **Partición Temporal Estricta (`split`):**  
  La columna `split` (`train`, `val`, `test`) generada en `data/processed/dataset_base.parquet` es la **única fuente de verdad** para entrenamiento, validación y prueba Out-Of-Time (OOT). Queda terminantemente prohibido el remuestreo aleatorio (`train_test_split` global) o K-Fold estratificado aleatorio que viole la causalidad temporal.

* **Formato de Intercambio Tabular:**  
  Todos los artefactos de datos compartidos se exportan y leen en formato `.parquet` con compresión `snappy`, preservando tipos nativos (`int`, `float`, `datetime`, `category`).

* **Principio de Fuente Única de Verdad (Single Source of Truth) e Interoperabilidad R/Python:**  
  Python es el **único entorno autorizado para procesar, transformar y persistir artefactos tabulares compartidos (`.parquet`)** en el repositorio. Queda terminantemente prohibido generar artefactos para modelado desde entornos paralelos en R.  
  * **Pipeline y Escritura:** Todos los filtros de población, particiones temporales (`split`) e ingeniería de variables se computan y persisten exclusivamente mediante scripts de Python (`src/make_dataset.py`, `src/features/build_features.py`) exportando a formato `.parquet` en `data/`.
  * **Consumo y Auditoría:** El uso de R o Quarto (`.Rmd`/`.qmd`) queda restringido a tareas de análisis exploratorio (EDA), auditoría estadística y visualización, consumiendo siempre los archivos `.parquet` procesados vía la librería `arrow` (`arrow::read_parquet()`).
  * **Regla Anti-Duplicación:** Cualquier decisión analítica derivada del EDA en R (transformaciones, descartes o nuevas métricas) debe implementarse formalmente en el script correspondiente de Python antes de ser utilizada por el equipo.

* **Política de Git:**  
  Los archivos binarios (`.parquet`, `.csv`) no se suben a Git (están en `.gitignore`). Cada integrante debe garantizar que su código sea determinista y reproducible desde `data/processed/dataset_base.parquet` ejecutando un comando de CLI reproducible con semilla `random_state=42`.

---

## 2. Arquitectura de Integración de Datos

```mermaid
flowchart TD
    Raw["data/raw/fraud_dataset_v2.csv"] --> Base["src/make_dataset.py<br/><b>data/processed/dataset_base.parquet</b><br/>(transaction_id, split, features crudas)"]
    
    Base --> Rol1["<b>ROL 1: Tomás</b><br/>Data Engineering & Features<br/><code>src/features/build_features.py</code>"]
    Base --> Rol2["<b>ROL 2: Franco</b><br/>Unsupervised Anomaly Detection<br/><code>src/models/anomaly_detection.py</code>"]
    
    Rol1 --> Feat1["<b>data/features/features_transaccionales.parquet</b><br/>(transaction_id, rollings, ratios, frecuencias)"]
    Rol2 --> Feat2["<b>data/features/scores_anomalias.parquet</b><br/>(transaction_id, iforest, pca, consolidado)"]
    
    Base --> Join["<b>ROL 3: Ian</b><br/>Integración & Modelado Supervisado<br/><code>src/models/train.py</code>"]
    Feat1 --> Join
    Feat2 --> Join
    
    Join --> Eval["Evaluación PR-AUC, Calibración &<br/>Curvas de Costo de Negocio"]
```

---

## 3. Contratos de Datos (Interfaces Técnicas Detalladas)

### Contrato A: Integrante 1 $\rightarrow$ Repositorio Común (Tomás)
* **Rol:** Senior Data Engineer / Feature Engineer  
* **Script Generador:** `src/features/build_features.py`  
* **Archivo de Salida:** `data/features/features_transaccionales.parquet`  
* **Frecuencia de Entrega:** Versión preliminar (v1) e iteración final (v2).  
* **Responsabilidad de Implementación Anti-Duplicación:**  
  Toda hipótesis, descarte de registros o transformación validada durante el EDA en R debe implementarse formalmente en el pipeline de Python (`src/features/build_features.py`) antes de exportarse a `data/features/features_transaccionales.parquet`. R no genera datasets intermedios para el resto del equipo.  

#### Esquema Oficial Producción (v2):
| Columna | Tipo | Descripción | Restricción Anti-Leakage / Validación Proxy |
| :--- | :--- | :--- | :--- |
| `transaction_id` | `string` | ID universal (`tx_XXXXXX`) | Clave primaria determinista (150.000 filas) |
| `ratio_monto_ag_bayes` | `float64` | Ratio normalizado de monto por `(a, g)` con suavizado bayesiano | Priors y medias ajustados **estrictamente** sobre `train` |
| `ratio_monto_b` | `float64` | Tensión ticket monetario / tasa continua `b` (`monto / (b + 0.001)`) | Sin leakage. Gain: ~1.884 |
| `prod_b_log_c` | `float64` | Interacción no lineal `b * log1p(c)` | Sin leakage. Gain: ~1.318 |
| `ratio_monto_c` | `float64` | Escala relativa `monto / (c + 1.0)` | Sin leakage. Gain: ~2.300 |
| `ratio_d_m` | `float64` | Proporción relativa `d / (m + 1.0)` | Sin leakage. Gain: ~2.506 |
| `ratio_monto_m` | `float64` | Normalización del ticket respecto a `m` (`monto / (m + 1.0)`) | Sin leakage. Gain: ~3.141 |
| `diff_m_d` | `float64` | Discrepancia directa `m - d` | Sin leakage. Gain: ~4.464 |
| `ratio_monto_f` | `float64` | Ticket normalizado `monto / (max(f, 0) + 1.0)` | Domina outlier 145k. Gain: ~4.540 |
| `ratio_monto_e` | `float64` | Ticket normalizado `monto / (e + 0.01)` | Sin leakage. Gain: ~795 |
| `log_f` | `float64` | Compresión logarítmica `log1p(max(f, 0))` | Mitiga asimetría pesada. Gain: ~300 |
| `is_f_cero_neg` | `int8` | Flag booleano `f <= 0` (segmento crítico 12% fraude) | Sin leakage |
| `is_e_cero` | `int8` | Flag booleano `e == 0` (masa puntual 44%) | Sin leakage |
| `ratio_monto_h` | `float64` | Ticket normalizado `monto / (h + 1.0)` | Tensión ticket / madurez. Gain: ~1.375 |
| `is_h_cero` | `int8` | Flag booleano `h == 0` (segmento crítico 9.6% fraude) | Sin leakage |
| `log_h` | `float64` | Compresión logarítmica `log1p(h)` | Sin leakage |

> [!NOTE]
> **Performance del Pool Candidato:** El pool ampliado de features candidato contiene 15 variables analíticas que alimentarán el proceso final de *Forward Stepwise Feature Selection*. Destacan `ratio_monto_f` (Gain: ~4.540), `diff_m_d` (Gain: ~4.460) y `ratio_monto_m` (Gain: ~3.140) como los principales motores de split.

> [!WARNING]
> **Restricción Estricta de Causalidad:** Toda feature que use agregaciones de tiempo (`rolling`, `expanding`, ventanas deslizantes) debe usar un desplazamiento temporal o `shift(1)` para no incluir información del evento que se está evaluando. Si se usan target encodings, los priors se calculan **únicamente** con datos de `split == 'train'`.

---

### Contrato B: Integrante 2 $\rightarrow$ Repositorio Común (Franco)
* **Rol:** Machine Learning Engineer — Detección de Anomalías No Supervisadas  
* **Script Generador:** `src/models/anomaly_detection.py`  
* **Archivo de Salida:** `data/features/scores_anomalias.parquet`  
* **Modelos Asignados:** Isolation Forest, PCA (Error de Reconstrucción) y Métodos Locales (LOF / Distancia Mahalanobis o k-NN).  

#### Esquema Obligatorio:
| Columna | Tipo | Rango / Unidad | Descripción |
| :--- | :--- | :--- | :--- |
| `transaction_id` | `string` | `tx_XXXXXX` | Clave primaria para merge |
| `score_iforest` | `float64` | $[0.0, 1.0]$ | Score normalizado o percentil de anomalía (mayor = más anómalo) |
| `error_pca` | `float64` | $[0.0, 1.0]$ | Error de reconstrucción normalizado (MinMax sobre train) |
| `distancia_lof_knn` | `float64` | $[0.0, 1.0]$ | Densidad o distancia local normalizada |
| `score_anomalia_consolidado` | `float64` | $[0.0, 1.0]$ | Ensamble / combinación ponderada no supervisada |

> [!CAUTION]
> **Restricción Metodológica Fundamental:**  
> Los modelos de detección de anomalías son no supervisados pero de naturaleza **one-class**: deben entrenarse (`fit`) **exclusivamente** con las transacciones legítimas (`fraude == 0`) pertenecientes al conjunto `split == 'train'`.  
> Queda prohibido:
> 1. Hacer `fit` incluyendo transacciones etiquetadas como fraude (`fraude == 1`).
> 2. Calcular escaladores o fitear modelos sobre `val` o `test` (solo se permite `transform` / `score_samples`).

---

### Contrato C: Integrante 3 $\rightarrow$ Entrega Final (Ian)
* **Rol:** Lead Data Scientist — Modelado Supervisado, Calibración & Negocio  
* **Script de Integración & Entrenamiento:** `src/models/train.py`  
* **Modelos Asignados:**
  * **Baseline:** Regresión Logística y/o Random Forest (solo con features crudas).
  * **Modelo Retador (Challenger):** LightGBM / XGBoost / CatBoost integrando features transaccionales + scores de anomalías.
  * **Optimización de hiperparámetros:** búsqueda bayesiana con Optuna usando **TPE + pruning** como método principal. Probar además **CMA-ES** como método alternativo y comparar resultados bajo el mismo presupuesto de trials.
  * **Calibración:** Isotonic Regression o Platt Scaling sobre `val`.
  * **Optimización de Negocio:** Función de pérdida / costo financiero y cálculo de umbral óptimo $\tau^*$.

#### Contrato de Optimización de Hiperparámetros
* La búsqueda se ejecuta sobre los modelos supervisados asignados (en particular, los modelos Challenger) y maximiza **PR-AUC en `val`**, dada la naturaleza desbalanceada del problema.
* Cada trial entrena con `split == 'train'` y evalúa con `split == 'val'`, respetando el orden temporal. El conjunto `test` queda reservado para la evaluación final y no puede intervenir en la búsqueda, el pruning, la selección del sampler ni la elección de hiperparámetros.
* El método principal será **TPE + pruning**: sampler TPE con semilla reproducible y un pruner de Optuna (por ejemplo, `MedianPruner`). Los modelos con soporte de entrenamiento incremental deben reportar la métrica intermedia de validación en cada iteración/época para permitir detener trials poco prometedores. Si un modelo no permite pruning por iteración, se documentará y se completará el trial sin pruning; no se simulará pruning.
* **CMA-ES** se probará como sampler alternativo de Optuna en los espacios de búsqueda compatibles. Para que la comparación sea justa, se mantendrán iguales el modelo, los datos, la métrica objetivo, los rangos de búsqueda y el presupuesto de trials; se registrará cualquier parámetro categórico que CMA-ES no pueda optimizar directamente y la estrategia usada para tratarlo.
* La búsqueda debe respetar el anti-leakage: preprocesadores se ajustan solo con `train`; `val` solo se usa para calcular la métrica objetivo e informar al pruner. Los datos `test` no se consultan hasta congelar la configuración elegida.
* Se deben guardar el sampler, la semilla, el pruner, el número de trials, el espacio de búsqueda, los parámetros y métricas de cada trial, los trials podados y la mejor configuración. Los resultados deben quedar documentados y ser reproducibles desde CLI con `--seed` (por defecto `42`).

**Criterio de comparación:** reportar la mejor PR-AUC de validación y el costo/tiempo de búsqueda para TPE + pruning y CMA-ES. Seleccionar la configuración final únicamente con `val`; luego reentrenar según el procedimiento definido y evaluar una sola vez en `test`.

#### Código de Integración Estándar (`src/models/train.py`):
```python
import pandas as pd

def build_full_dataset(
    base_path="data/processed/dataset_base.parquet",
    feat_path="data/features/features_transaccionales.parquet",
    anom_path="data/features/scores_anomalias.parquet",
) -> pd.DataFrame:
    df_base = pd.read_parquet(base_path)
    df_feat = pd.read_parquet(feat_path)
    df_anom = pd.read_parquet(anom_path)

    # Merge determinista 1:1 por transaction_id
    df_full = (
        df_base
        .merge(df_feat, on="transaction_id", how="left")
        .merge(df_anom, on="transaction_id", how="left")
    )

    # Aserción de integridad
    assert len(df_full) == len(df_base), "Error: El merge alteró la cantidad de filas."
    assert df_full["transaction_id"].nunique() == len(df_base), "Error: Hay transaction_id duplicados."
    return df_full
```

#### Entregables de Negocio:
1. **Curva PR-AUC & ROC-AUC:** Reporte comparativo en `val` y `test`:
   * Modelo Baseline vs.
   * Modelo con Features Transaccionales vs.
   * Modelo Completo (+ Scores de Anomalías).
2. **Matriz de Costos y Función Económica:**
   $$\text{Costo Total}(\tau) = C_{FN} \times FN(\tau) + C_{FP} \times FP(\tau)$$
   donde $C_{FN}$ representa el monto defraudado no recuperado y $C_{FP}$ el costo operativo / fricción del cliente por falso positivo.
3. **Determinación del Umbral Óptimo $\tau^*$:** Selección del punto de operación maximizando el ahorro neto o minimizando el costo esperado en `test`.

---

## 4. Criterios de Aceptación (Definition of Done — DoD)

| # | Criterio | Validación Obligatoria |
| :-: | :--- | :--- |
| **1** | **Anti-Leakage Estricto** | Todo escalador (`StandardScaler`, etc.), imputador o estimador debe ser fiteado **únicamente** en `train`. En `val` y `test` solo se ejecuta `.transform()`. |
| **2** | **Merge Determinista & Sin NaNs** | El dataset integrado debe conservar exactamente las $150.000$ filas de `dataset_base.parquet` (tras la exclusión justificada del Bloque 2 validada en EDA). No se admiten registros huérfanos ni multiplicados. |
| **3** | **Reproducibilidad Total** | Todos los scripts ejecutables deben aceptar `--seed` con valor por defecto `42`. |
| **4** | **Compatibilidad de Esquemas** | Todo Parquet exportado en `data/features/` debe incluir `transaction_id` como primera columna de tipo `string`. |
| **5** | **Documentación de Experimentos** | Métricas reportadas en `test` deben haber sido congeladas sin reajustes de hiperparámetros post-evaluación. |
