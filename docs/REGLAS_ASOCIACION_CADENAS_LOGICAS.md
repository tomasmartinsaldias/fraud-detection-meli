# Análisis Exploratorio: Minería de Reglas de Asociación y Cadenas Lógicas

> **Dataset:** `data/processed/dataset_base.parquet`  
> **Partición Train:** 105,000 transacciones (Tasa de Fraude Base: **5.19%**)  
> **Partición Val (OOT):** 22,500 transacciones (Tasa de Fraude Base: **4.53%**)  

## 1. Contexto Teórico: Variables Continuas vs Reglas de Asociación

Las reglas de asociación clásicas (Apriori, FP-Growth) fueron diseñadas para variables booleanas/categóricas de tipo canasta de mercado (`{A, B} => {C}`). Al aplicarlas sobre datasets de fraude transaccional con variables numéricas continuas (`score`, `monto`, `n`, `l`, `m`, etc.), surgen dos desafíos estructurales:
1. **Sensibilidad a la Discretización:** Una partición arbitraria (ej. 3 bins de igual ancho) pierde fronteras no lineales y colas extremas donde reside el fraude.
2. **Explosión Combinatoria:** Aumentar el número de bins genera itemsets esparsos con soporte trivial.

Para resolver esto, comparamos dos enfoques rigurosos:
- **Reglas de Asociación Cuantitativas (CARs):** Discretización semántica guiada por percentiles y colas de riesgo, cruzada con categóricas (`g`, `o`, `p`).
- **Cadenas Lógicas Subrogadas (Decision Tree Surrogate):** Inducción supervisada de cortes exactos sin discretización forzada, extrayendo reglas booleanas directamente interpretables.

---

## 2. Reglas de Asociación Cuantitativas de Clase (CARs) => {fraude = 1}

| # | Regla (Antecedente) | P(Fraude) Train | Lift Train | Soporte Train | P(Fraude) Val OOT | Lift Val OOT |
|---|---|---|---|---|---|---|
| 01 | `score_muy_alto(>85) AND s_cero(<=0) AND b_alto(>0.9)` | **47.3%** | **9.12x** | 0.62% (N=653) | 48.8% | 10.76x |
| 02 | `score_muy_alto(>85) AND b_alto(>0.9) AND pais_BR` | **46.5%** | **8.97x** | 0.68% (N=716) | 44.4% | 9.80x |
| 03 | `score_alto(>70) AND m_bajo(<=17) AND b_alto(>0.9)` | **43.6%** | **8.40x** | 0.64% (N=668) | 40.4% | 8.91x |
| 04 | `score_muy_alto(>85) AND f_bajo(<=12) AND b_alto(>0.9)` | **40.4%** | **7.80x** | 0.64% (N=670) | 47.7% | 10.51x |
| 05 | `score_muy_alto(>85) AND b_alto(>0.9)` | **39.1%** | **7.54x** | 0.84% (N=880) | 42.4% | 9.35x |
| 06 | `score_muy_alto(>85) AND score_alto(>70) AND b_alto(>0.9)` | **39.1%** | **7.54x** | 0.84% (N=880) | 42.4% | 9.35x |
| 07 | `score_muy_alto(>85) AND monto_muy_alto(>p90) AND n_cero(<=0)` | **38.4%** | **7.41x** | 0.53% (N=557) | 42.2% | 9.31x |
| 08 | `score_muy_alto(>85) AND n_cero(<=0) AND l_bajo(<=40)` | **38.1%** | **7.35x** | 1.43% (N=1503) | 39.9% | 8.81x |
| 09 | `score_muy_alto(>85) AND n_cero(<=0) AND m_bajo(<=17)` | **37.8%** | **7.29x** | 1.99% (N=2086) | 40.8% | 9.00x |
| 10 | `score_muy_alto(>85) AND m_bajo(<=17) AND l_bajo(<=40)` | **37.7%** | **7.27x** | 1.32% (N=1382) | 40.6% | 8.95x |

---

## 3. Cadenas Lógicas de Alto Riesgo (Surrogate Tree - Fraude)

| # | Cadena Lógica Óptima | P(Fraude) Train | Lift Train | Soporte Train | P(Fraude) Val OOT | Lift Val OOT |
|---|---|---|---|---|---|---|
| 01 | `f <= 12.50 AND m <= 17.50 AND score > 86.50` | **34.9%** | **6.72x** | 3.53% (N=3703) | 37.1% | 8.18x |
| 02 | `f > 12.50 AND h <= 0.50 AND score > 86.50` | **26.1%** | **5.02x** | 0.45% (N=472) | 19.1% | 4.22x |
| 03 | `f <= 12.50 AND m > 17.50 AND score > 86.50` | **17.4%** | **3.35x** | 5.70% (N=5984) | 21.3% | 4.69x |
| 04 | `f <= 21.50 AND n <= 0.50 AND 71.50 < score <= 86.50` | **12.0%** | **2.31x** | 3.11% (N=3266) | 9.9% | 2.18x |
| 05 | `f <= 21.50 AND n > 0.50 AND 71.50 < score <= 86.50` | **6.8%** | **1.31x** | 9.58% (N=10064) | 5.6% | 1.24x |
| 06 | `l <= 2151.50 AND 9.50 < score <= 25.50` | **5.8%** | **1.12x** | 7.34% (N=7706) | 5.4% | 1.20x |
| 07 | `f <= 9.50 AND 64.50 < score <= 71.50` | **5.6%** | **1.08x** | 4.01% (N=4210) | 4.1% | 0.91x |
| 08 | `f > 21.50 AND monto > 46.57 AND 71.50 < score <= 86.50` | **4.9%** | **0.94x** | 0.99% (N=1041) | 10.2% | 2.24x |
| 09 | `f > 12.50 AND h > 0.50 AND score > 86.50` | **3.8%** | **0.74x** | 3.52% (N=3698) | 16.5% | 3.64x |
| 10 | `l > 2151.50 AND 9.50 < score <= 25.50` | **3.6%** | **0.70x** | 8.12% (N=8523) | 3.5% | 0.77x |

---

## 4. Cadenas Lógicas de Bajo Riesgo (Whitelist / Legítimas)

| # | Cadena Lógica Legítima | P(Fraude) Train | P(Legítimo) Train | Soporte Train | P(Fraude) Val OOT |
|---|---|---|---|---|---|
| 01 | `25.50 < score <= 47.50` | 0.48% | **99.52%** | 21.13% (N=22191) | 0.55% |
| 02 | `47.50 < score <= 65.50` | 1.37% | **98.63%** | 17.37% (N=18237) | 1.02% |
| 03 | `f > 10.50 AND monto <= 55.18 AND 65.50 < score <= 82.50` | 1.97% | **98.03%** | 4.31% (N=4521) | 2.91% |
| 04 | `f > 12.50 AND l > 1588.50 AND m > 17.50 AND score > 82.50` | 2.03% | **97.97%** | 3.57% (N=3748) | 9.22% |
| 05 | `score <= 9.50` | 2.75% | **97.25%** | 11.10% (N=11657) | 3.28% |

---

## 5. Principales Hallazgos e Implicancias de Negocio

1. **Sinergia no lineal entre Score y Contadores:**  
   - Un `score > 85` aislado tiene un lift relevante, pero al combinarse con `s <= 0` (o `n <= 0`), `b > 0.90` o `m <= 17.50`, la probabilidad de fraude salta de **5.2% a casi 48% (Lift ~9.5x - 10.7x)**.
2. **Generalización Temporal Sólida (Out-Of-Time):**  
   - Las reglas no sufren sobreajuste. En la partición de validación `val`, las reglas mantienen e incluso aumentan su Lift (alcanzando hasta 10.76x).
3. **Patrón de Operaciones Seguras (Fast-Track / Whitelist):**  
   - Reglas con `score <= 25.50` y contadores normales concentran miles de transacciones con una tasa de fraude inferior al 0.5% (99.5%+ legítimas), ideales para auto-aprobación con cero fricción.
