# Diccionario Integral de Variables — Detección de Fraude Transaccional (MELI)

> **Repositorio:** `fraud-detection-meli`  
> **Autores:** Tomás (Data Eng & Features), Franco (Anomaly Detection), Ian (Modeling & Business)  
> **Última actualización:** Noviembre 2026  
> **Datasets documentados:** 
> - `data/processed/dataset_base.parquet` (150.000 filas × 24 columnas)
> - `data/features/features_transaccionales.parquet` (150.000 filas × 51 columnas)
> - `data/features/scores_anomalias.parquet` (150.000 filas × 13 columnas)
> - `data/features/features_seleccionadas_end_to_end.parquet` (150.000 filas × 13 columnas)

---

## 1. Principios de Diseño y Buenas Prácticas

Todas las variables de este repositorio han sido diseñadas bajo tres principios rectores:

1. **Inmutabilidad de la Clave Primaria (`transaction_id`):** Identificador universal determinista en formato `tx_XXXXXX`. Todos los datasets satélite cruzan estrictamente mediante `merge(on='transaction_id')`.
2. **Causalidad Temporal Estricta (Anti-Leakage):** Todas las transformaciones supervisadas (Target Encoding, Naive Bayes, SVD, medias bayesianas) ajustan sus parámetros (`fit`) **exclusivamente sobre `split == 'train'`**. La partición temporal `val` y `test` solo recibe inferencia (`transform`).
3. **Poda Sistemática por Ganancia Empírica:** Cada variable ingenierizada compitió en esquemas de Forward Stepwise Selection evaluadas mediante PR-AUC (Precision-Recall Area Under Curve), optimizando el trade-off entre interpretabilidad, parsimonia y poder predictivo.

---

## 2. Variables Base y Crudas (`data/processed/dataset_base.parquet`)

Provenientes de la ingesta y desanonimización de `data/raw/fraud_dataset_v2.csv` procesadas por `src/make_dataset.py`.

| Variable | Tipo | Rango / Cardinalidad | Descripción / Significado de Negocio |
| :--- | :---: | :---: | :--- |
| `transaction_id` | `string` | `tx_000000` - `tx_149999` | Clave primaria universal e inmutable. |
| `fecha` | `datetime64[ns]` | Marzo - Noviembre 2020 | Timestamp UTC de la transacción. |
| `monto` | `float64` | $[0.01, 4192.0]$ | Importe monetario de la operación en moneda local estandarizada. Presenta asimetría positiva severa ($\text{skew} = 9.57$). |
| `score` | `float64` | $[0.0, 100.0]$ | Score probabilístico interno de scoring de riesgo transaccional provisto por la pasarela de pagos. Es el predictor crudo individual más potente (Gain: 47.140). |
| `fraude` | `int8` | $\{0, 1\}$ | Target binario. $1 = \text{Fraude confirmado}$ (tasa general $\approx 6.0\%$), $0 = \text{Legítima}$. |
| `split` | `category` | `train` (70%), `val` (15%), `test` (15%) | Partición temporal estricta Out-Of-Time (OOT) basada en percentiles cronológicos de `fecha`. |
| `a` | `category` | 5 categorías (`0.0`, `1.0`, etc.) | Dispositivo / canal de origen de la transacción. |
| `b` | `float64` | $[0.0, 1.0]$ | Tasa o ratio de comportamiento continuo del usuario/dispositivo. |
| `c` | `float64` | Continuo positivo con NAs | Variable sintética con jittering decimal. La parte entera $\lfloor c \rfloor$ corresponde al ID de usuario real (`r`), mientras que los NAs corresponden al checkout como invitado ($r = 260445$). |
| `is_na_bc` | `int8` | $\{0, 1\}$ | Flag indicador de valor faltante en `c` (equivalente a compra sin sesión iniciada / Guest Checkout). |
| `d` | `float64` | Continuo positivo | Magnitud operativa o antigüedad de cuenta. Fuertemente correlacionada con `m`. |
| `e` | `float64` | Continuo ($\ge 0$) | Métrica transaccional con masa puntual en 0 (el 44% de las transacciones tiene $e = 0$). |
| `f` | `float64` | $[ -15, \dots ]$ | Métrica continua con masa puntual en valores negativos/cero (el 12% es $\le 0$, con tasa de fraude elevada). |
| `g` | `category` | 20+ países | País de emisión/origen (predominio de Brasil `BR` 71.4%, Argentina `AR` 24.5%, Uruguay `UY` 2.0%). |
| `h` | `float64` | Continuo ($\ge 0$) | Madurez o historial del comprador. Masa puntual en 0 (9.6% de fraude cuando $h = 0$). |
| `i` | `string` | Texto libre | Título o descripción del producto publicado en el marketplace. |
| `j` | `category` | Categórica de alta cardinalidad | Identificador de vertical o categoría de catálogo del producto. |
| `k` | `float64` | $[0.0, 1.0]$ | Variable sintética de ruido uniforme ($U(0, 1)$), ortogonal al target. Descartable. |
| `l` | `float64` | Continuo ($\ge 0$) | Tiempo transcurrido o antigüedad acumulada de la relación con la plataforma. |
| `m` | `float64` | Continuo ($\ge 0$) | Escala acumulada de volumen transaccional histórico del usuario. |
| `n` | `category` | Categórica binaria/ternaria | Flag de perfil de seguridad del medio de pago. |
| `o` | `category` | Categórica con 72% NAs | Nivel de autenticación de dos factores o validación de identidad adicional. |
| `p` | `category` | Categórica binaria/ternaria | Método o tipo de tarjeta/pasarela utilizada. |
| `q` | `float64` | Continuo ($\ge 0$) | Factor continuo de fricción o límite crediticio del comprador. |

---

## 3. Variables Ingenierizadas Transaccionales (`data/features/features_transaccionales.parquet`)

Generadas por `src/features/build_features.py` (50 variables numéricas y categóricas optimizadas).

### Familia 1: Ratios Monetarios y Tensiones de Escala

Miden la anomalía del ticket actual frente a las capacidades históricas, límites o escalas del perfil transaccional.

| Variable | Tipo | Fórmula Matemática | Propósito y Hallazgo Analítico |
| :--- | :---: | :--- | :--- |
| `ratio_monto_ag_bayes` | `float64` | $\frac{\text{monto}}{\mu_{\text{Bayes}}(a, g)}$ | Ticket normalizado por país y dispositivo con regularización bayesiana ($m=30$) hacia la media del dispositivo. Evita overfitting en países chicos. |
| `ratio_monto_b` | `float64` | $\frac{\text{monto}}{b + 0.001}$ | Tensión entre el ticket monetario y la tasa continua `b`. (Gain: ~1.884). |
| `prod_b_log_c` | `float64` | $b \times \log(1 + \max(c, 0))$ | Interacción no lineal entre factor de perfil `b` y la magnitud de `c`. Seleccionada en el Top 11 global. |
| `ratio_monto_c` | `float64` | $\frac{\text{monto}}{c + 1.0}$ | Proporción del monto actual sobre la escala de la cuenta. |
| `ratio_d_m` | `float64` | $\frac{d}{m + 1.0}$ | Relación de magnitudes entre `d` y `m`. Detecta discrepancias anómalas en perfiles maduros. Seleccionada en el Top 11 global. |
| `ratio_monto_m` | `float64` | $\frac{\text{monto}}{m + 1.0}$ | Intensidad del ticket actual relativo a la escala acumulada `m`. (Gain: ~3.141). |
| `diff_m_d` | `float64` | $m - d$ | Diferencia directa entre variables colineales. Captura rupturas de balance histórico. |
| `ratio_monto_f` | `float64` | $\frac{\text{monto}}{\max(f, 0) + 1.0}$ | Modulación del monto ante el factor de estabilidad `f`. Domina valores atípicos. (Gain: ~4.540). |
| `ratio_monto_e` | `float64` | $\frac{\text{monto}}{e + 0.01}$ | Normalización del ticket sobre la magnitud `e`. Seleccionada en el Top 11 global. |
| `ratio_monto_h` | `float64` | $\frac{\text{monto}}{h + 1.0}$ | Presión del ticket sobre el historial `h`. |
| `ratio_monto_l` | `float64` | $\frac{\text{monto}}{l + 1.0}$ | Tensión entre el monto y la antigüedad acumulada `l`. Seleccionada en el Top 11 global. |
| `ratio_monto_q` | `float64` | $\frac{\text{monto}}{q + 0.01}$ | Relación del ticket frente a la capacidad de fricción `q`. |
| `score_x_monto` | `float64` | $\text{score} \times \text{monto}$ | Métrica de **exposición esperada de pérdida**. Si el score es alto y el monto es grande, el riesgo financiero se multiplica exponencialmente. |
| `ratio_monto_j_bayes` | `float64` | $\frac{\text{monto}}{\text{Mediana}_{\text{Bayes}}(j)}$ | Desvío del ticket actual respecto al precio central típico de la categoría `j` ($m=20$). |
| `log_monto` | `float64` | $\log(1 + \max(\text{monto}, 0))$ | Transformación estabilizadora de varianza para comprimir la asimetría positiva de los montos. |
| `monto_centavos` | `float64` | $\lfloor (\text{monto} \times 100) \pmod{100} \rfloor$ | Extracción de los centavos ($0 \dots 99$). Captura patrones psicológicos de precios (.00, .99) vs importes arbitrarios de fraude. |
| `is_monto_alto` | `int8` | $\mathbb{I}(\text{monto} > 422.0)$ | Bandera binaria para transacciones en el percentil 99 superior de la distribución monetaria. |

---

### Familia 2: Compresiones Logarítmicas y Banderas de Masa Puntual

Capturan comportamientos discretos anómalos originados por ceros estructurales o distribuciones fuertemente asimétricas.

| Variable | Tipo | Fórmula Matemática | Propósito y Hallazgo Analítico |
| :--- | :---: | :--- | :--- |
| `log_f` | `float64` | $\log(1 + \max(f, 0))$ | Compresión logarítmica de `f` mitigando outliers extremos. |
| `is_f_cero_neg` | `int8` | $\mathbb{I}(f \le 0)$ | Indicador de corte estructural. El segmento con $f \le 0$ concentra una tasa de fraude del 12% (el doble de la media). |
| `is_e_cero` | `int8` | $\mathbb{I}(e == 0)$ | Flag de masa puntual en $e = 0$ (44% del total de transacciones). |
| `is_h_cero` | `int8` | $\mathbb{I}(h == 0)$ | Flag de ausencia de historial en `h` (9.6% de fraude). Seleccionada en el Top 11 global. |
| `log_h` | `float64` | $\log(1 + \max(h, 0))$ | Compresión logarítmica del historial `h`. |
| `log_l` | `float64` | $\log(1 + \max(l, 0))$ | Compresión logarítmica de la antigüedad `l`. |
| `log_m` | `float64` | $\log(1 + \max(m, 0))$ | Compresión logarítmica del volumen histórico `m`. |
| `is_o_missing` | `int8` | $\mathbb{I}(o \text{ is NaN})$ | Flag booleano de falta de registro en la variable `o` (72% de nulos). |
| `flag_alto_riesgo_nop`| `int8` | $\mathbb{I}(n=0 \land o=\text{'N'} \land p=\text{'N'})$ | Coincidencia de factores negativos simultáneos en los 3 mecanismos de seguridad. |

---

### Familia 3: Temporales y Ciclos Circadianos

Capturan las variaciones circadianas de la actividad criminal frente a la actividad comercial genuina.

| Variable | Tipo | Fórmula Matemática | Propósito y Hallazgo Analítico |
| :--- | :---: | :--- | :--- |
| `hora_cont` | `float64` | $\text{hora} + \frac{\text{min}}{60} + \frac{\text{seg}}{3600}$ | Tiempo continuo fraccional del día ($[0.0, 24.0)$). Preserva el gradiente sub-horario. |
| `cos_hora` | `float64` | $\cos\left(\frac{2\pi \cdot \text{hora\_cont}}{24}\right)$ | Componente armónica cíclica. Modela la polaridad noche vs día de forma suave. |
| `sin_hora` | `float64` | $\sin\left(\frac{2\pi \cdot \text{hora\_cont}}{24}\right)$ | Componente armónica cíclica. Modela la polaridad mañana vs tarde. |
| `es_madrugada` | `int8` | $\mathbb{I}(\text{hora} \ge 23 \lor \text{hora} \le 5)$ | Ventana nocturna en UTC optimizada empíricamente. En este intervalo la tasa de fraude salta hasta el 17.2% (pico a las 02:00 UTC). Seleccionada en el Top 11 global. |
| `dia_semana` | `float64` | $0.0 \dots 6.0$ | Día de la semana (Lunes=0 a Domingo=6). |

---

### Familia 4: NLP y Minería de Texto sobre Títulos (`i`)

Extraen señales semánticas, estilométricas y de probabilidad de riesgo a partir de las cadenas de texto del producto (`i`).

| Variable | Tipo | Metodología | Propósito y Hallazgo Analítico |
| :--- | :---: | :--- | :--- |
| `char_len` | `float64` | Longitud en caracteres | Métrica estilométrica. Títulos inusualmente cortos o largos correlacionan con publicaciones automatizadas. |
| `word_count` | `float64` | Conteo de palabras | Cantidad de tokens en el título. |
| `avg_word_len` | `float64` | $\frac{\text{char\_len}}{\text{word\_count}}$ | Longitud promedio de palabra. Detecta cadenas aleatorias o títulos técnicos. |
| `digit_ratio` | `float64` | $\frac{\text{dígitos}}{\text{char\_len}}$ | Proporción de números en el texto (típico de números de parte o códigos de gift cards). |
| `text_svd_1` a `text_svd_10` | `float64` | LSA (TF-IDF sublinear + TruncatedSVD) | 10 dimensiones latentes del espacio semántico del catálogo. Ajustadas **estrictamente sobre `train`** (regla del codo de varianza explicada). |
| `score_texto_nb` | `float64` | Naive Bayes Multinomial OOF ($\alpha=2.0$) | Probabilidad Out-Of-Fold de fraude estimada únicamente a partir del vocabulario del título mediante `TimeSeriesSplit` sobre `train`. Seleccionada en el Top 11 global (Gain: ~3.400). |

---

### Familia 5: Frecuencias y Target Encoding Categórico (`j`)

Codifican el riesgo histórico y la popularidad de las categorías de catálogo sin memorización espuria.

| Variable | Tipo | Metodología | Propósito y Hallazgo Analítico |
| :--- | :---: | :--- | :--- |
| `te_j_bayes` | `float64` | Target Encoding Bayesiano Regularizado | Tasa de fraude histórica por categoría $j$, suavizada con prior global ($m=10$) y computada mediante KFold=5 OOF con inyección de jitter gaussiano ($\sigma=0.01$). (Gain: ~7.115). |
| `freq_j` | `float64` | $\log(1 + \text{count}_{\text{train}}(j))$ | Log-frecuencia de observaciones de la categoría en entrenamiento. Separa verticales de venta masiva de nichos o categorías singleton. |

---

### Familia 6: Velocidad y Ráfagas de Mismo Producto

Modelan el comportamiento de bots y carding masivo donde un mismo producto publicado sufre compras repetidas a alta velocidad.

| Variable | Tipo | Metodología Anti-Leakage | Propósito y Hallazgo Analítico |
| :--- | :---: | :--- | :--- |
| `log_delta_mismo_prod` | `float64` | $\log(1 + \Delta t)$ con `shift(1)` causal | Tiempo transcurrido en segundos desde la última transacción del mismo producto `i`. Si el producto nunca se vio antes, se imputa un valor neutro equivalente a 30 días. |
| `burst_mismo_prod_24h` | `int8` | $\mathbb{I}(\Delta t \le 86.400 \text{ seg})$ | Bandera de ráfaga: compras sobre el mismo título en un lapso $\le 24$ horas. Para transacciones repetidas en menos de 10 minutos, la tasa de fraude se dispara al **34.78%** (Lift ~7.0x). |

---

## 4. Scores de Detección de Anomalías No Supervisadas (`data/features/scores_anomalias.parquet`)

Generadas por `src/models/anomaly_detection.py`. Modelos One-Class entrenados **exclusivamente con transacciones legítimas (`fraude == 0`) de `split == 'train'`** sobre las variables numéricas base estandarizadas.

| Variable | Algoritmo | Familia No Supervisada | Interpretación |
| :--- | :--- | :--- | :--- |
| `pca_q` | PCA (Reconstrucción) | Subespacio Lineal | Error cuadrático residual $Q$ (SPE - Square Prediction Error) al proyectar y reconstruir la transacción en el subespacio principal. |
| `pca_magnitud` | PCA (Norma Euclídea) | Subespacio Lineal | Norma del vector de componentes principales en el subespacio retenido. |
| `pca_t2` | PCA (Hotelling $T^2$) | Subespacio Lineal | Distancia de Mahalanobis en el espacio latente de las componentes principales retenidas. |
| `score_iforest_perpendicular` | Isolation Forest | Partición por Hiperplanos | Score de aislamiento con particiones ortogonales a los ejes de coordenadas. Cuanto más corto el camino medio al nodo hoja, mayor el score. |
| `score_iforest_diagonal` | Extended Isolation Forest | Partición por Hiperplanos | Score de aislamiento con cortes inclinados/diagonales aleatorios, eliminando sesgos de ejes coordinados. |
| `score_lof_manhattan` | Local Outlier Factor (LOF) | Densidad Local ($k=30$) | Razón de densidad local respecto a sus vecinos utilizando métrica de distancias Manhattan ($L_1$). |
| `score_lof_euclidean` | Local Outlier Factor (LOF) | Densidad Local ($k=30$) | Razón de densidad local respecto a sus vecinos utilizando métrica Euclídea ($L_2$). |
| `score_ocsvm_rbf_nu_003` | One-Class SVM ($\nu=0.03$) | Frontera no lineal de soporte | Distancia con signo al hiperplano soporte con kernel RBF calibrado a una tasa de contaminación conservadora del 3%. |
| `score_ocsvm_rbf_nu_005` | One-Class SVM ($\nu=0.05$) | Frontera no lineal de soporte | Distancia al hiperplano soporte con kernel RBF calibrado a contaminación del 5%. |
| `score_gmm_k3` | Gaussian Mixture Model | Modelado Paramétrico Probabilístico | Log-verosimilitud negativa ajustando una mezcla de 3 gaussianas multivariadas con matriz de covarianza completa. |
| `score_gmm_k5` | Gaussian Mixture Model | Modelado Paramétrico Probabilístico | Log-verosimilitud negativa con 5 componentes gaussianas. |
| `score_anomalia_consolidado`| Ensamble Rank Average | Combinación No Supervisada | Promedio de percentiles normalizados entre Isolation Forest, PCA $Q$ y LOF. Proporciona una señal consensual libre de ruido puntual. |

---

## 5. Módulo Satélite: Reglas de Asociación Heurísticas CARs (`data/features/features_reglas_asociacion.parquet`)

Generadas por el script satélite independiente `src/features/build_rule_features.py` (desacoplado del pipeline transaccional central). Implementa un panel ortogonal de 10 reglas de asociación de alto impacto neto / WRAcc (Weighted Relative Accuracy).

| Variable | Tipo | Condición Lógica | Justificación de Negocio |
| :--- | :---: | :--- | :--- |
| `score_heuristico_fraude_10` | `int8` | $\sum_{k=1}^{10} r_k \in [0, 10]$ | Conteo acumulado de reglas violadas simultáneamente en la transacción. |
| `pct_heuristico_fraude` | `float64` | $\frac{\text{score\_heuristico\_fraude\_10}}{10.0} \in [0.0, 1.0]$ | Grado porcentual de riesgo heurístico agregado. |
| `rule_car_1_seguridad_fallida` | `int8` | $p == \text{'N'} \land o == \text{'N'}$ | Medio de pago no tokenizado y fallo/ausencia de autenticación 2FA. |
| `rule_car_2_inmediatez_sin_historial` | `int8` | $\text{ratio\_d\_m} \ge 1.0 \lor \text{is\_f\_cero\_neg} == 1$ | Ruptura de historial operativo o ausencia de antigüedad en $f$. |
| `rule_car_3_tension_monto_f` | `int8` | $\text{ratio\_monto\_f} > p_{75} \land \text{is\_o\_missing} == 0$ | Ticket monetario alto sobre factor $f$ con identidad verificada fallida. |
| `rule_car_4_categoria_critica` | `int8` | $\text{te\_j\_bayes} > p_{75} \land \text{is\_o\_missing} == 0$ | Compra en categorías históricamente vulnerables sin autenticación segura. |
| `rule_car_5_madrugada_o_texto_nb` | `int8` | $\text{es\_madrugada} == 1 \lor \text{score\_texto\_nb} > p_{90}$ | Operación en ventana crítica nocturna o con vocabulario de alto riesgo. |
| `rule_car_6_antiguedad_baja_l` | `int8` | $\log(1+l) \le p_{25} \land \text{is\_o\_missing} == 0$ | Cuentas recién creadas en operaciones sin 2FA. |
| `rule_car_7_tension_monto_m` | `int8` | $\text{ratio\_monto\_m} > p_{75} \land \text{is\_o\_missing} == 0$ | Desproporción del importe actual frente al volumen histórico $m$. |
| `rule_car_8_b_alto_joven` | `int8` | $b > p_{90} \land m \le p_{25}$ | Tasa de fricción continua extrema en cuentas de bajo volumen. |
| `rule_car_9_canal_atipico` | `int8` | $a \ne 4.0 \land \text{is\_o\_missing} == 0$ | Canales de transacción atípicos fuera del flujo predominante. |
| `rule_car_10_discrepancia_m_d` | `int8` | $\text{diff\_m\_d} \le p_{10}$ | Quiebre de coherencia matemática entre escalas históricas $m$ y $d$. |

---

## 6. Datasets de Modelado Final (Selección Forward Stepwise)

Para evitar la maldición de la dimensionalidad y el sobreajuste ante la alta colinealidad, se generaron dos variantes de conjuntos curados para modelado supervisado:

### 1. Dataset Recomendado: Forward Stepwise Global Blindado (`data/features/features_seleccionadas_end_to_end.parquet`)

Selección competitiva abierta entre **67 variables candidatas** (17 crudas + 50 ingenierizadas), con criterio de tolerancia $\Delta \ge 0.0003$ y paciencia $k=2$ sobre el split de validación Out-Of-Time.

* **Cantidad de variables:** **11 variables** (4 base + 7 ingenierizadas).
* **PR-AUC alcanzado:** **`0.462206`** (+0.0335 vs baseline de 17 variables crudas).
* **Columnas incluidas:**
  1. `score` (Base - Scoring interno de la pasarela)
  2. `o` (Base - Tipo de autenticación adicional)
  3. `g` (Base - País emisor)
  4. `es_madrugada` (Ingenierizada - Ventana nocturna crítica UTC)
  5. `ratio_d_m` (Ingenierizada - Tensión de balances históricos)
  6. `ratio_monto_l` (Ingenierizada - Ticket normalizado por antigüedad)
  7. `prod_b_log_c` (Ingenierizada - Interacción perfil / usuario)
  8. `score_texto_nb` (Ingenierizada - Probabilidad bayesiana del título)
  9. `ratio_monto_e` (Ingenierizada - Ticket sobre escala `e`)
  10. `is_h_cero` (Ingenierizada - Flag de masa puntual en madurez 0)
  11. `a` (Base - Canal o dispositivo de origen)

*Metadatos y traza del algoritmo:* [data/features/selected_features_end_to_end_report.json](file:///C:/Users/tomas/Desktop/universidad/Ciencia%20de%20Datos/Taller%20de%20Resolución%20de%20Problemas%20II/fraud-detection-meli/data/features/selected_features_end_to_end_report.json).

### 2. Dataset Incremental: Baseline Base + Forward (`data/features/features_seleccionadas_forward.parquet`)

Conserva obligatoriamente las 17 variables crudas base y agrega un núcleo complementario de 7 variables ingenierizadas (`es_madrugada`, `te_j_bayes`, `ratio_monto_l`, `diff_m_d`, `score_x_monto`, `text_svd_7`, `avg_word_len`).

* **Cantidad de variables:** **24 variables** (17 base + 7 ingenierizadas).
* **PR-AUC alcanzado:** **`0.458400`** (+0.0297 vs baseline).
* **Metadatos y traza del algoritmo:** [data/features/selected_features_forward_report.json](file:///C:/Users/tomas/Desktop/universidad/Ciencia%20de%20Datos/Taller%20de%20Resolución%20de%20Problemas%20II/fraud-detection-meli/data/features/selected_features_forward_report.json).
