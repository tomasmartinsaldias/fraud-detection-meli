# Guión de Contenidos: Presentación Detección de Fraude (MELI)
**Universidad Austral · Taller de Resolución de Problemas II**  
*Equipo: Tomás Saldías, Franco Esterellas, Ian Connon*

> **Instrucciones para el equipo:**  
> Este documento sirve como base de contenido para la presentación interactiva. Cada slide tiene su título propuesto (humano y al grano), los datos y métricas oficiales congeladas de los experimentos, y un espacio para que agreguen o modifiquen lo que quieran que se diga al exponer.  
> Cuando terminen de editarlo, el agente volcará todo a la presentación final interactiva (`presentacion/index.html`).

---

## Slide 0: Portada & Presentación del Equipo
* **Quién expone:** Apertura conjunta (10-15 segundos).
* **Título en pantalla:** Detección de Fraude Transaccional y Optimización de Umbral.
* **Subtítulo:** De probabilidades continuas a decisiones financieras en pasarelas de pago.
> Comentario de Tomi: Demasiado formar el subtitulo. Pensaba que podiamos poner algo mas informal como hace el profe. "Moonshot Thinking" en una presentacion. 
* **Integrantes y roles definidos:**
  * **Tomás Saldías:** Feature Engineering & Umbrales de Decisión.
  * **Franco Esterellas:** Detección de Anomalías, Reglas de Asociación & Ensambles.
  * **Ian Connon:** Modelado Supervisado & MLOps.
* **Notas / Qué decir:**
  * [Completar aquí o dejar como presentación institucional rápida]

---

## Slide 1: El Punto de Partida (El Problema de los $140k)
* **Quién expone:** [Ian]
* **Título propuesto:** *El problema inicial: la logística costaba $140.000 y el LightGBM crudo $123.000.*
* **Eyebrow (ceja):** `1. EL PUNTO DE PARTIDA`
* **Datos y evidencia oficial:**
  * **Modelo de la consigna (`logistic_raw`):** Regresión Logística con variables nativas.
  * **Umbral inicial congelado en validación:** $\tau = 0.25132$ (regla de no superar 100 FP en validación).
  * **Costo en Test (5.000 transacciones):**
    * 487 Falsos Positivos $\times \$5 = \$2.435$
    * **690 Falsos Negativos $\times \$200 = \$138.000$** (fuga masiva de fraudes no detectados).
    * **Costo total:** **$140.435** | **PR-AUC:** `0.3150`.
  * **LightGBM de referencia con variables nativas (`lgb_reference_raw`):** PR-AUC `0.4313` | Costo: **$123.215**.
* **Visual sugerida:** Comparativa de números grandes (Hero Metric): $140k vs $123k y la alerta de los 690 fraudes no vistos.
* **Notas / Qué decir:**
  * [Poner acá el énfasis: el negocio sangraba plata no por falsos positivos, sino porque el modelo dejaba escapar casi 700 fraudes].

---

## Slide 2: La Estrategia del Equipo (Objetivos y Método)
* **Quién expone:** [Tomi / Ian]
* **Título propuesto:** *Nuestro objetivo: maximizar PR-AUC y adaptar los umbrales a las pérdidas reales.*
* **Eyebrow (ceja):** `2. LA ESTRATEGIA`
* **Puntos clave y metodología:**
  * **Separación estricta de dos problemas:**
    1. **Fase de Data Science (Modelado):** Maximizar la capacidad pura de ordenamiento y discriminación (`PR-AUC`), completamente independiente del umbral de decisión.
    2. **Fase de Negocio (Toma de Decisiones):** Diseñar reglas de corte basadas en el costo asimétrico ($5 vs $200) y en el importe monetario de cada compra.
  * **Blindaje metodológico:** Validación temporal Out-of-Time (OOT) estricta: todo feature, normalización o prior bayesiano se entrena solo en Train para evitar cualquier data leakage.
* **Visual sugerida:** Diagrama de 2 bloques: Fase 1 (Maximizar PR-AUC sin mirar umbrales) $\to$ Fase 2 (Optimizar decisiones financieras con el modelo congelado).
* **Notas / Qué decir:**
  * [Explicar por qué no mezclamos el umbral con el entrenamiento del modelo].
  > Comentario de Tomi: Podemos dejar en claro que el split se hizo temporal.

---

## Slide 3: Poda y Selección de Variables (Feature Engineering 1)
* **Quién expone:** Tomi
* **Título propuesto:** *De 51 variables a 11: ratios de monto y quiebre de colinealidad.*
* **Eyebrow (ceja):** `3. SELECCIÓN DE VARIABLES & RATIOS`
* **Datos y evidencia oficial:**
  * Se formularon más de 50 hipótesis de negocio (temporales, agregadas, ratios, interacciones).
  * **Algoritmo:** Forward Stepwise Selection con paciencia $k=2$ y tolerancia mínima de $\Delta\text{PR-AUC} > 0.0003$ sobre validación temporal.
  * **Números Big Numbers en pantalla:**
    * **51 &rarr; 11:** Se descartó el 78% de las variables por ser ruido o generar sobreajuste; sobrevivieron 11 variables blindadas (4 nativas + 7 ingenierizadas).
    * **0.4443 (+130 bps):** Salto de generalización inmediato en validación temporal sin sobreajuste.
  * **Monto crudo descartado vs Ratios creados:**
    * El algoritmo descartó `monto` plano porque aislado no predecía.
    * `ratio_monto_l` (monto relativo a la escala $l$) se convirtió en la variable ingenierizada #1 de todo el modelo (**Gain de 9.330**), complementada con `ratio_monto_e`.
  * **Quiebre de colinealidad `d` vs `m` (`ratio_d_m`):**
    * En el EDA $d$ y $m$ crecen fuertemente juntas (**Spearman $r_s = 0.735$**).
    * `ratio_d_m` (+132 bps) atrapa las operaciones atípicas donde esa relación se rompe ($d \gg m$), donde la probabilidad de fraude salta de 4.2% a 11.0% y hasta **28.8% en quiebre extremo (multiplica el fraude x5 a x7)**.
* **Visual:** `figures/fig_ratio_d_m.svg` (Explosión de tasa de fraude de 4.2% a 11.0% y 28.8% al romperse la correlación de Spearman de 0.735).
* **Notas / Qué decir:**
  * **Rigor y Poda:** Destacar que en vez de dos gráficos de barras genéricos, los números hablan por sí solos: podar 78% del ruido llevó el PR-AUC a 0.4443 (+130 bps).
  * **Monto relativo:** Mostrar la tabla en pantalla de las 11 variables y remarcar que `ratio_monto_l` es la variable #1 creada del modelo (Gain 9.330).
  * **Quiebre de colinealidad con el gráfico:** Señalar la figura de `ratio_d_m`: cómo el modelo aprovecha la ruptura de la correlación de Spearman de 0.735 para atrapar fraudes que se multiplican por 5 o más.

---

## Slide 4: Variables Clave y Complejidad Técnica (Feature Engineering 2)
* **Quién expone:** Tomi
* **Título propuesto:** *Las variables más potentes: ventana de madrugada, text-mining y ratios no lineales.*
* **Eyebrow (ceja):** `4. VARIABLES CLAVE`
* **Datos y evidencia oficial:**
  * **1. Ventana circadiana (`es_madrugada`):** Ventana de 23:00 a 05:00 UTC. La tasa de fraude promedio del dataset es 4.5%, pero a las 02:00 UTC trepa abruptamente al **17.2%** (3.8x la prevalencia diurna).
  * **2. NLP bayesiano (`score_texto_nb`):** Modelado probabilístico Naive Bayes sobre tokens de títulos de publicaciones, entrenado estrictamente en Train para evitar fuga.
  * **3. Interacción no lineal (`prod_b_log_c`):** Interacción multiplicativa no lineal entre la variable anónima continua $b$ y la escala logarítmica $\log(c + 1)$, permitiendo capturar dependencias conjuntas complejas que a un árbol le cuesta partir con cortes ortogonales.
* **Visual:** `figures/fig_fe_complejidad.svg` (Reloj circadiano polar de 24h UTC y comparativa de lifts de las variables complejas).
* **Notas / Qué decir:**
  * [Contar brevemente cómo los estafadores operan en horarios desatendidos y cómo el texto del producto aportó señal predictiva].
  * **El Reloj Circadiano (es_madrugada):** Señalar el gráfico circular tipo reloj de 24 hs. De día el fraude está controlado en ~4.1%, pero de madrugada (23:00 a 05:00 UTC) se dispara hasta el 17.2% a las 02:00 UTC (3.8x más). Los estafadores aprovechan los horarios nocturnos desatendidos.
  * **Text-Mining Bayesiano (score_texto_nb):** Explicar cómo minamos los títulos de las publicaciones (`i`) con Naive Bayes ajustados 100% en Train. 
    1. El problema y la solución: El título de la publicación (i) contiene señales críticas de riesgo. Construimos score_texto_nb: un método que condensa un vocabulario de 5.000 términos en una única probabilidad continua entre 0 y 1.
    2. Cómo: TF-IDF con ngram_range=(1, 2). No miramos solo palabras sueltas; capturamos bigramas porque el contexto local cambia el riesgo (ej. "digital" es común, pero "gift card" o "código digital" concentran altísima tasa de fraude). Poda (min_df=30): Eliminamos términos poco frecuentes para que el modelo no memorice palabras únicas de estafas aisladas.Multinomial Naive Bayes con suavizado $\alpha = 2.0$. Si aparece una palabra nueva en Test nunca vista en fraudes previos, evita que la verosimilitud colapse a cero y empuja la estimación hacia la media base.
    3. Blindaje Anti-Leakage: Como usamos el target $fraude$ para crear la feature, calcularla sobre el mismo set de entrenamiento causaría target leakage inmediato. Usamos Out-Of-Fold temporal (TimeSeriesSplit en 5 particiones): entrenamos tanto el TF-IDF como el Naive Bayes exclusivamente con bloques pasados para puntuar el bloque siguiente. Recién al final reentrenamos con el 100% de Train para proyectar a Validación y Test.
  * **Interacción no lineal (prod_b_log_c):** Contar que al multiplicar `b` con el logaritmo de `c`, le dimos al modelo una curvatura directa (+91 bps) que a los árboles de decisión les costaría decenas de particiones ortogonales aproximar. Con b alto y c bajo que tenían un 12.8% de fraude, casi el triple que el resto de los clientes. Esto lo ponemos como una una mini-tarjeta comparativa de 2 datos

---

## Slide 5: Detección de Anomalías y Reglas de Asociación
* **Quién expone:** Franco Esterellas
* **Título propuesto:** *Anomaly Detection y Reglas de Asociación aportan señales ortogonales al ensamble.*
* **Eyebrow (ceja):** `5. ANOMALÍAS & REGLAS`
* **Datos y evidencia oficial:**
  * **Ensamble No Supervisado One-Class:** Entrenado exclusivamente sobre transacciones legítimas (`fraude == 0`) de Train utilizando variables crudas. Detecta anomalías geométricas sin haber visto etiquetas de fraude.
    * Combina **Isolation Forest, PCA y LOF** (*Local Outlier Factor*) mediante una meta-regresión logística.
    * Rendimiento univariado aislado: **`0.1970` de PR-AUC** en validación OOT (**+337 bps** sobre Isolation Forest puro de `0.1557`, y **4.4x** superior a la prevalencia aleatoria de 4.5%).
  * **Reglas de Asociación Heurísticas (CARs):**
    * Minería de reglas evaluadas mediante la métrica **WRAcc** (*Weighted Relative Accuracy*).
    * Generación de un panel de **10 reglas ortogonales de alto riesgo** que resumen patrones combinatorios para alimentar como features al Stacking.
* **Visual:** `figures/fig_anomalias_ensamble_prauc.svg` (Capacidad discriminativa del ensamble no supervisado frente al azar y al Isolation Forest puro).
* **Notas / Qué decir (Franco Esterellas):**
  * **1. Por qué no supervisado:** Explicar que los modelos supervisados se sobreajustan a los patrones de fraude conocidos en el pasado. El ensamble One-Class se entrena solo con compradores legítimos (`fraude == 0`); cualquier transacción que se desvíe en el espacio multidimensional es marcada como anómala.
  * **2. Ensamble de 3 perspectivas geométricas:** Combinar Isolation Forest (cortes hiperrectangulares aleatorios), PCA (reconstrucción por varianza residual) y LOF (densidad local) aporta una señal de anomalía mucho más robusta (`0.1970` PR-AUC) que un solo algoritmo.
  * **3. Reglas de Asociación CARs con WRAcc:** Detallar cómo las 10 reglas minadas no buscan predecir todo el dataset, sino atrapar combinaciones complejas de nicho con altísima pureza de fraude, entregando señales categóricas ortogonales a los árboles supervisados.

---

## Slide 6: Auditoría de Residuos y Fraude Hormiga
* **Quién expone:** Tomi
* **Título propuesto:** *Auditoría de residuos: compras de $22 evadían el modelo por sesgo a montos altos.*
* **Eyebrow (ceja):** `6. AUDITORÍA DE RESIDUOS`
* **Datos y evidencia oficial:**
  * **El punto ciego:** Al auditar los errores del modelo en validación, descubrimos que detectaba muy bien compras grandes (mediana $75, media $98), pero se le fugaban sistemáticamente transacciones chicas (mediana **$22.76**, media $47.50).
  * **Solución semántica de precios:**
    * Clustering de títulos de publicación ($k=50$, embedding $d=30$) para establecer el precio de referencia de cada categoría de producto.
    * Creación de variables de desvío: `deficit_piso_cluster` (+19 bps) y `distancia_log_cluster` (+25 bps).
    * Impacto neto: **+44 bps en Validación y +61 bps en Test**, neutralizando la fuga en compras chicas.
  * **Límite del umbral plano:** La mediana de cazados apenas bajó de **$27.52 a $27.00** (-1.9%), demostrando que el ML solo no alcanza y se requiere un umbral dinámico (la Curva S).
* **Visual:** Panel izquierdo con Big Numbers de diagnóstico ($75 vs $22.76) y límite del umbral plano ($27.52 &rarr; $27.00); panel derecho con tabla de 4 casos reales rescatados (87% a 97% de déficit).
* **Notas / Qué decir:**
  * [Explicar cómo analizar los errores nos permitió diseñar variables quirúrgicas en vez de tirar modelos a ciegas].
  * **1. Auditoría de Residuos (El Diagnóstico):**
  * No nos quedamos solo con la métrica global del modelo de 11 variables; abrimos los errores en Validación
  para ver *qué* se nos estaba escapando. * Descubrimos un patrón clarísimo: los fraudes que el modelo detectaba promediaban casi $100 ($97.96), pero los fraudes que lograban evadirlo tenían una **mediana de apenas $22.76** (y un promedio de $47.50).
  * El modelo tenía un sesgo natural hacia montos altos y dejaba pasar compras de menor cuantía.
  * **2. La Causa Raíz Técnica:**
    * ¿Por qué el modelo se comía los fraudes de mediana de $22? Porque todas las variables de la primera ola
    tenían al monto en el numerador (`ratio_monto_l`, `ratio_monto_e`).
    * Esos ratios estaban matemáticamente diseñados para explotar cuando el monto era alto. Los árboles
    aprendieron a cazar fraudes caros, pero los fraudes de montos moderados o bajos generaban ratios chicos y
    pasaban totalmente desapercibidos.

  * **3. La Solución (Invertir la lógica con 'deficit_piso_cluster'):**
    * Agrupamos los títulos en 50 clusters semánticos para calcular la mediana de precio esperada de cada
    producto (usando TF-IDF + SVD con d=30 y k-means, fiteado 100% en Train)
    * Esto nos permitió calcular la **mediana bayesiana de precio de cada cluster de productos**, y creamos dos variables aprobadas por Forward Selection:
    * `distancia_log_cluster`: desvío logarítmico del precio respecto a su categoría (+25 bps). Difiere de lo habitual?
    * `deficit_piso_cluster`: penalización directa cuando el precio cae muy por debajo del piso de ese cluster específico (+19 bps). Está por debajo de la mediana?
\

* **4. El Resultado Empírico:**
  * El PR-AUC subió de inmediato: **+44 bps en Validación y +61 bps en Test**.
  * Logró rescatar transacciones que antes no se detectaban como fraudes:
    1. Suscripción de PlayStation Plus vendida a $0.40
        • Publicación: "Psn Plus Playstation 14 Dias Para Ps4"
        • Monto: $0.40 | Mediana de su cluster: $14.41 | Déficit: 0.97 (97% de descuento anómalo).
    2. Cuentas de Spotify Premium ofuscadas con asteriscos a $2.22
        • Publicación: "\*\*spoti\*\*fy 1 Mes Pr\*\*miüm\*\*\*personal Oficial!"
        • Monto: $2.22 | Mediana de su cluster: $25.61 | Déficit: 0.91 (91% de descuento).
    3. Videojuegos digitales completos a $2.30 - $2.80
        • Publicaciones: "The Sims 4 Deluxe Party Edition Xbox One Midia Digital" ($2.36) y "Gta 5 Pc - Dinero 250 M
        Personaje Chetado" ($2.62).
        • Monto: ~$2.50 | Mediana del cluster: ~$15 - $20 | Déficit: 0.85 a 0.87.
    4. Códigos de desbloqueo de estéreos de auto a $1.89
        • Publicación: "Code De Desbloqueio Do Rádio Mp3 Da Fiat Palio Com Garantia"
        • Monto: $1.89 | Mediana de su cluster: $19.55 | Déficit: 0.90.

* **5. La Gran Conclusión (Puente hacia la Slide de Umbrales):**
  * Pero acá nos topamos con la lección más importante del proyecto: **con un umbral plano estático, la mediana
  de los fraudes cazados apenas bajó de $27.52 a $27.00.**
  * El Machine Learning solo no alcanza si la regla de corte del negocio sigue siendo una línea recta. Si bajás
  un umbral plano para atrapar las compras de $22, arruinás el negocio llenándote de falsos positivos
  * Para resolverlo de verdad en el bolsillo de la empresa, entendimos que la solución final requería cambiar la
  política de corte: **hacer el umbral elástico al ticket (la Curva S).**

---

## Slide 7: Arquitectura de Stacking y el Récord
* **Quién expone:** Ian Connon / Franco Esterellas / Tomás Saldías
* **Título propuesto:** *Stacking de 4 modelos: 0.4922 en validación y récord de 0.4509 en Test.*
* **Eyebrow (ceja):** `7. STACKING & RÉCORD`
* **Datos y evidencia oficial (Interactivo en 2 pasos):**
  * **Paso 0 (Champion Oficial de Protocolo):**
    * Metamodelo: Regresión Lasso ($C=0.20$) combinando LightGBM, CatBoost, Random Forest y XGBoost con optimización CMA-ES.
    * Métrica Validación OOT: **`0.4922`** (el mejor modelo del equipo, congelado a ciegas).
    * Métrica en Test: **`0.4501`** (supera con rigor la barrera del 0.4500 sin test-snooping).
  * **Paso 1 (Hallazgo de Parsimonia L1 - Tocar flecha derecha):**
    * ¿Qué pasa si forzamos a Lasso a apagar coeficientes redundantes ($C=0.10$)?
    * En Validación cae levemente a `0.4910` (-11 bps, pierde optimismo).
    * En Test salta a **`0.4509` (+195.8 bps)**: récord histórico absoluto de todo el taller, traduciéndose en el **79.9% ($63.4k)** de capital recuperado.
* **Visual:** Tarjetas comparativas interactivas con números en grande (`0.4501` vs `0.4509` / Knaflic style).
* **Notas / Qué decir (Ian Connon & Franco Esterellas):**
  * **1. Protocolo Ciego y Respeto a las Reglas:** Ian: Resaltar que el champion oficial seleccionado en Validación OOT fue el $C=0.20$ con `0.4922`, el cual cruzó la barrera de `0.4500` en Test (`0.4501`) de forma totalmente limpia, sin test-snooping ni data leakage.
  * **2. La Lección de la Parsimonia L1 ($C=0.10$):** Franco: Explicar cómo la regularización extrema apaga predicciones colineales entre familias de árboles similares. Al sacrificar 11 bps de optimismo en validación (`0.4910`), el modelo gana robustez frente al drift temporal y alcanza el pico histórico del proyecto en Test (`0.4509`).
  * **3. El Descarte Justificado: Ablación de 16 Modelos Multi-Loss:**
    * *Anticipando la pregunta del jurado:* "¿Por qué no combinaron más funciones de pérdida?".
    * Franco: Probamos una arquitectura de 16 inductores entrenando XGBoost, LightGBM, CatBoost y Random Forest bajo 4 loss functions: **Log-Loss, Weighted Loss, Focal-Loss y Reranker Loss**.
    * Aplicamos Forward Selection de modelos con regularización; solo sobrevivieron 5 inductores.
    * En Validación OOT subió de `0.4922` a `0.4950` (+3 milésimas / +0.6%), pero en Test OOT rindió exactamente lo mismo: **`0.4501` (+0.0%)**.
    * Por la Navaja de Ockham y criterio de ingeniería en producción, descartamos el ensamble de 16 modelos para evitar costos de inferencia y deuda técnica innecesaria.
  * **4. Experimento con Grafos Bipartitos (Auditoría adicional):** Franco exploró features relacionales de grafo usuario-producto (`in_degree_usuarios_producto_hist`) y dispersión de categorías. En Test rindió `0.4497`, por lo que se mantuvo el rollback al Champion de Parsimonia puro (`0.4509`).

---

## Slide 8: Política Financiera de Umbrales (Curva S + Semáforo 2FA)
* **Quién expone:** Tomás Saldías
* **Título propuesto:** *Curva S y semáforo 2FA: convertimos probabilidades en 80% de capital recuperado.*
* **Eyebrow (ceja):** `8. POLÍTICA FINANCIERA`
* **Datos y evidencia oficial (Interactivo en 2 pasos):**
  * **Paso 0 (Curva S en Backend - Costo Cero):**
    * Umbral elástico según ticket: $\tau(monto) \in [0.18, 0.40]$ con sigmoide centrada en $M_0 = \$23$.
    * Resultado: Rescata **$50.875 de capital (+$6.744 netos)** e incrementa el recupero al **64.1%** con exactamente los mismos ~500 bloqueos duros.
  * **Paso 1 (Semáforo 2FA - Tocar flecha derecha):**
    * Tres carriles de decisión:
      * **Verde (94.2%):** Aprobación directa sin fricción ($\hat{p} < 0.18$).
      * **Amarillo (4.8%):** Reto 2FA suave por WhatsApp/Push ($0.18 \le \hat{p} < \tau_S$).
      * **Rojo (1.0%):** Bloqueo directo fraudulento ($\hat{p} \ge \tau_S$).
    * Resultado: **79.9% de capital recuperado ($63.4k protegidos)** y 33% menos costo formal.
    * **Resiliencia:** Si el servicio de 2FA se cae, degrada automáticamente a la Curva S (piso garantizado del 63.3%).
* **Visual:** Gráfico 2D del Espacio de Decisión con 3 zonas sombreadas (`figures/fig4_semaforo_3zonas.svg`: verde, amarillo y rojo) mostrando las fronteras de corte $\tau_{\text{block}}$ y $\tau_{\text{challenge}}$ y cómo el 2FA rescata los tickets chicos.
* **Notas / Qué decir:**
  * [Tomi: explicar cómo la combinación de ambas cosas protege al negocio de fraudes grandes sin generar fricción en compras chicas].
  
  **1. El problema y la Curva S bajo restricción de capacidad (Fase 1 - Backend)**

  * Un umbral plano ($\tau = 0.255$) ignora la asimetría económica del negocio: bloquear por error cuesta un monto fijo, pero dejar pasar un fraude cuesta el **monto del ticket** ($15 vs. $400).
  * Diseñamos una **sigmoide decreciente** $\tau_S(\text{monto}) \in [0.18, 0.40]$ calibrada con punto de inflexión en la mediana de fraude ($M_0 = \$23$) y suavidad $\sigma = 35$:
  * **Tickets altos:** baja a **0.18** (cerrojo estricto para proteger capital).
  * **Tickets bajos:** sube a **0.40** (no quemamos bloqueos duros en compras chicas donde el riesgo no lo justifica).
  * **Restricción estricta de Falsos Positivos:** Toda la calibración en Validación se hizo bajo la cota dura $\text{FP}_{\text{bloqueo}} \le 100$ (que escala a **504 bloqueos duros** en Test, respetando el presupuesto de fricción de la consigna).
  * **Impacto puro de la Curva S:** Reasigna ese mismo presupuesto de riesgo hacia tickets caros y rescata **+$6.744 adicionales** frente al umbral estático (**$44.131 $\rightarrow$ $50.875**, subiendo el recupero de **55.6% a 64.1%**).

  **2. Cómo resolvemos el fraude de ticket bajo: Semáforo Tripartito 2FA (Fase 2 - Checkout)**

  * *Anticipando la crítica:* La Curva S pura deja pasar compras chicas dudosas para no violar el techo de FPs duros. Para capturarlas sin fricción masiva, rompemos la decisión binaria con un **tercer carril**:
  * 🟢 **Carril Verde ($\hat{p} < \tau_{\text{challenge}}$ — 94.2%):** Aprobación directa en 200 ms. Fricción cero.
  * 🟡 **Carril Amarillo ($\tau_{\text{challenge}} \le \hat{p} < \tau_{\text{block}}$ — 4.8%):** Reto suave **2FA** (Push/WhatsApp a ~$0.05). El cliente legítimo valida en 5 segundos; el estafador sin acceso físico al teléfono queda frenado.
  * 🔴 **Carril Rojo ($\hat{p} \ge \tau_{\text{block}}$ — 1.0%):** Bloqueo duro inmediato.

  **3. Calibración matemática de las fronteras ($\tau_{\text{block}}$ y $\tau_{\text{challenge}}$)**

  * **Techo ($\tau_{\text{block}}$):** Es la Curva S semántica 2D ($\tau_S(\text{monto}) - \alpha \cdot \text{déficit\_piso}$), calibrada primero en Validación para garantizar $\text{FP}_{\text{bloqueo}} \le 100$. Si un ticket chico tiene una ganga irreal (alto déficit), el ajuste 2D le baja el umbral y lo manda directo al reto.
  * **Piso ($\tau_{\text{challenge}}$):** Parametrizado como fracción proporcional del techo ($\tau_{\text{challenge}} = \beta \cdot \tau_{\text{block}}$).
  * **Optimización en Validación:** Mediante *grid search* sobre $\beta \in [0.40, 0.85]$ minimizamos la función de costo operativo real:

  $$\text{Costo}_{\text{total}} = \$25 \cdot \text{FP}_{\text{bloqueo}} + \$0.05 \cdot \text{FP}_{\text{challenge}} + \$200 \cdot \text{FN}_{\text{remanentes}}$$


  El óptimo dio $\beta^* = 0.40$ (equivalente a $\hat{p} \approx 0.18$ en tickets estándar). A ese nivel, el carril amarillo de 2FA absorbió 1.077 transacciones en Test, lo que representa exactamente el 4.8% de todo el tráfico de la plataforma. El reto escalonado de 2FA JAMÁS puede superar el 5% del volumen total de compras.

  * **Salto financiero en Test ($N = 22.500$):** Capturamos **199 fraudes adicionales** (Recall sube de **43.2% a 62.5%**), llevamos el capital recuperado al **79.9% ($63.398)** y reducimos el costo formal de la consigna en **~$40.000** ($119.9k $\rightarrow$ $80.4k).

  **4. Blindaje ante fallos operativos: Stress Test (Cierre)**

  * Evaluamos 5 escenarios adversos (caída de conversión de usuarios, *SIM swapping* o ataques bot/phishing). Incluso con solo 50% de efectividad en el canal 2FA, el recupero se sostiene en **71.6% ($56.9k)**.
  * **Peor caso absoluto (caída del 100% del servicio 2FA):** El Carril Rojo gobernado por la Curva S actúa como *fallback* automático, garantizando un piso de **63.3% ($50.2k)** —siempre por encima del 55.6% del umbral estático.



---

## Slide 9: Propuesta Final y Conclusiones (The Ask)
* **Quién expone:** Cierre conjunto (Ian / Franco / Tomi)
* **Título propuesto:** *Propuesta final: Curva S inmediata en backend y carril 2FA con tiempo.*
* **Eyebrow (ceja):** `9. PROPUESTA FINAL`
* **Datos y plan de acción de negocio:**
  * **Fase 1 (Inmediata - Costo Cero):**
    * Activar la Curva S en el motor de reglas del backend.
    * Captura inmediata de **+$7.200 netos** de recupero sin tocar la experiencia de usuario.
    > A la izquierda
  * **Fase 2 (Con tiempo - Producto & Checkout):**
    * Integrar el carril 2FA para transacciones dudosas.
    * Consolidar el **79.9% de capital recuperado**.
    * Costo inicial del baseline: **$140.435**
    * Costo con nuestra solución final: **$80.385**
    * **Ahorro neto final para Mercado Pago:** **$60.050 (-42.7%)** y **$63.400 salvados**.
    * Viabilidad de Operación: Menos de 94 bloqueos duros diarios (< 2.3% de falsos positivos). El 97.7% de los compradores lícitos de Mercado Pago pasa de forma 100% limpia e instantánea.
    > A la derecha 
* **Visual:** Matriz de decisión en 2 fases con métrica hero de ahorro financiero ($63.4k / -42.7%) y logos de la Universidad Austral.
* **Notas / Qué decir:**
  * [Cierre contundente y apertura a preguntas del profesor y la cátedra].
