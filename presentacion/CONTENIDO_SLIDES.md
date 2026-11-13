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
  > Comeentario de Tomi: Podemos dejar en claro que el split se hizo temporal.

---

## Slide 3: Poda y Selección de Variables (Feature Engineering 1)
* **Quién expone:** Tomás Saldías
* **Título propuesto:** *Feature Engineering: de 51 variables candidatas a un núcleo limpio de 11.*
* **Eyebrow (ceja):** `3. SELECCIÓN DE VARIABLES`
* **Datos y evidencia oficial:**
  * Se formularon más de 50 hipótesis de negocio (temporales, agregadas, ratios, interacciones).
  * **Algoritmo:** Forward Stepwise Selection con paciencia $k=2$ y tolerancia mínima de $\Delta\text{PR-AUC} > 0.0003$ sobre validación temporal.
  * **Resultado:** Se descartó el 78% de las variables por ser ruido o generar sobreajuste; sobrevivieron 11 variables blindadas (4 nativas + 7 ingenierizadas).
  * **Salto de rendimiento:** El PR-AUC subió de inmediato a **`0.4443` (+130 bps)** en validación temporal.
* **Visual:** `figures/fig1_embudo_seleccion.svg` (Embudo del algoritmo de Forward Selection).
* **Notas / Qué decir:**
  * [Mencionar el rigor: cada variable tuvo que ganarse su lugar compitiendo individualmente].

---

## Slide 4: Variables Clave y Complejidad Técnica (Feature Engineering 2)
* **Quién expone:** Tomás Saldías
* **Título propuesto:** *Las variables más potentes: ventana de madrugada, text-mining y ratios no lineales.*
* **Eyebrow (ceja):** `4. VARIABLES CLAVE`
* **Datos y evidencia oficial:**
  * **1. Ventana circadiana (`es_madrugada`):** La tasa de fraude promedio del dataset es 4.5%, pero a las 02:00 UTC trepa abruptamente al **17.2%** (3.8x la prevalencia diurna).
  * **2. NLP bayesiano (`score_texto_nb`):** Modelado probabilístico Naive Bayes con priors Dirichlet sobre tokens de títulos de publicaciones, entrenado estrictamente en Train para evitar fuga.
  * **3. Interacción no lineal (`prod_b_log_c`):** Interacción multiplicativa no lineal entre la variable anónima continua $b$ y la escala logarítmica $\log(c + 1)$, permitiendo capturar dependencias conjuntas complejas que a un árbol le cuesta partir con cortes ortogonales.
* **Visual:** `figures/fig_fe_complejidad.svg` (Gráfico de barras de tasa de fraude por hora UTC + fórmulas de Dirichlet).
* **Notas / Qué decir:**
  * [Contar brevemente cómo los estafadores operan en horarios desatendidos y cómo el texto del producto aportó señal predictiva].

---

## Slide 5: Detección de Anomalías y Reglas de Asociación
* **Quién expone:** Franco Esterellas
* **Título propuesto:** *Isolation Forest y Reglas de Asociación aportan señales ortogonales al ensamble.*
* **Eyebrow (ceja):** `5. ANOMALÍAS & REGLAS`
* **Datos y evidencia oficial:**
  * **Isolation Forest One-Class:** Entrenado exclusivamente sobre transacciones legítimas de Train. Detecta fraudes sin haber visto etiquetas de fraude.
    * Rendimiento univariado aislado: **`0.1557` de PR-AUC** (3.4 veces superior a la prevalencia aleatoria de 4.5%).
  * **Reglas de Asociación Heurísticas (CARs):**
    * Minería de reglas evaluadas mediante la métrica **WRAcc** (*Weighted Relative Accuracy*).
    * Generación de un panel de **10 reglas ortogonales de alto riesgo** que resumen patrones combinatorios para alimentar como features al Stacking.
* **Visual:** `figures/fig_anomalias_comparativa.svg` (Capacidad discriminativa del Isolation Forest vs Azar).
* **Notas / Qué decir:**
  * [Franco: detallar cómo estas señales no supervisadas aportan información distinta a la que ven los árboles supervisados].

---

## Slide 6: Auditoría de Residuos y Fraude Hormiga
* **Quién expone:** Tomás Saldías / Franco Esterellas
* **Título propuesto:** *El fraude hormiga: se nos escapaban las compras chicas de menos de $20.*
* **Eyebrow (ceja):** `6. AUDITORÍA DE RESIDUOS`
* **Datos y evidencia oficial:**
  * **El punto ciego:** Al auditar los errores del modelo en validación, descubrimos que detectaba muy bien compras grandes (mediana $75), pero se le fugaban sistemáticamente transacciones chicas (mediana **$22**).
  * **Solución semántica de precios:**
    * Clustering de títulos de publicación ($k=50$, embedding $d=30$) para establecer el precio de referencia de cada categoría de producto.
    * Creación de variables de desvío: `deficit_piso_cluster` (+19 bps) y `distancia_log_cluster` (+25 bps).
    * Impacto neto: **+26 bps de ganancia final**, neutralizando la fuga en compras chicas.
* **Visual:** `figures/fig2_residuos_fraude_hormiga.svg` (Distribución de montos de fraude detectado vs fraude escapado).
* **Notas / Qué decir:**
  * [Explicar cómo analizar los errores nos permitió diseñar variables quirúrgicas en vez de tirar modelos a ciegas].

---

## Slide 7: Arquitectura de Stacking y el Récord
* **Quién expone:** Ian Connon / Tomás Saldías
* **Título propuesto:** *Stacking de 4 modelos: 0.4922 en validación y récord de 0.4509 en Test.*
* **Eyebrow (ceja):** `7. STACKING & RÉCORD`
* **Datos y evidencia oficial (Interactivo en 2 pasos):**
  * **Paso 0 (Champion Oficial de Protocolo):**
    * Metamodelo: Regresión Lasso ($C=0.20$) combinando LightGBM, CatBoost, Random Forest y XGBoost.
    * Métrica Validación OOT: **`0.4922`** (el mejor modelo del equipo, congelado a ciegas).
    * Métrica en Test: **`0.4501`** (supera con rigor la barrera del 0.4500 sin test-snooping).
  * **Paso 1 (Hallazgo de Parsimonia L1 - Tocar flecha derecha):**
    * ¿Qué pasa si forzamos a Lasso a apagar coeficientes ($C=0.10$)?
    * En Validación cae levemente a `0.4910` (-11 bps, pierde optimismo).
    * En Test salta a **`0.4509` (+195.8 bps)**: récord histórico absoluto de todo el taller.
* **Visual:** Tarjetas comparativas interactivas con números en grande (Knaflic style).
* **Notas / Qué decir:**
  * [Ian: destacar el rigor de respetar el modelo de validación sin hacer trampa en Test, y la lección de cómo la regularización L1 extrema mejora la generalización].

---

## Slide 8: Política Financiera de Umbrales (Curva S + Semáforo 2FA)
* **Quién expone:** Tomás Saldías
* **Título propuesto:** *Curva S y semáforo 2FA: convertimos probabilidades en 80% de capital recuperado.*
* **Eyebrow (ceja):** `8. POLÍTICA FINANCIERA`
* **Datos y evidencia oficial (Interactivo en 2 pasos):**
  * **Paso 0 (Curva S en Backend - Costo Cero):**
    * Umbral elástico según ticket: $\tau(monto) \in [0.18, 0.40]$ con sigmoide centrada en $M_0 = \$23$.
    * Resultado: Rescata **+$7.200 netos** e incrementa el recupero al **64.1%** con exactamente los mismos ~500 bloqueos duros.
  * **Paso 1 (Semáforo 2FA - Tocar flecha derecha):**
    * Tres carriles de decisión:
      * **Verde (94.2%):** Aprobación directa sin fricción ($\hat{p} < 0.18$).
      * **Amarillo (4.8%):** Reto 2FA suave por WhatsApp/Push ($0.18 \le \hat{p} < \tau_S$).
      * **Rojo (1.0%):** Bloqueo directo fraudulento ($\hat{p} \ge \tau_S$).
    * Resultado: **79.9% de capital recuperado ($63.4k protegidos)** y 33% menos costo formal.
    * **Resiliencia:** Si el servicio de 2FA se cae, degrada automáticamente a la Curva S (piso garantizado del 63.3%).
* **Visual:** Gráfico de la Curva S sigmoidea y transición al diagrama de 3 carriles del Semáforo.
* **Notas / Qué decir:**
  * [Tomi: explicar cómo la combinación de ambas cosas protege al negocio de fraudes grandes sin generar fricción en compras chicas].

---

## Slide 9: Propuesta Final y Conclusiones (The Ask)
* **Quién expone:** Cierre conjunto (Ian / Franco / Tomi)
* **Título propuesto:** *Propuesta final: Curva S inmediata en backend y carril 2FA con tiempo.*
* **Eyebrow (ceja):** `9. PROPUESTA FINAL`
* **Datos y plan de acción de negocio:**
  * **Fase 1 (Inmediata - Costo Cero):**
    * Activar la Curva S en el motor de reglas del backend.
    * Captura inmediata de **+$7.200 netos** de recupero sin tocar la experiencia de usuario.
  * **Fase 2 (Con tiempo - Producto & Checkout):**
    * Integrar el carril 2FA para transacciones dudosas.
    * Consolidar el **79.9% de capital recuperado**.
  * **Balance Financiero Consolidado:**
    * Costo inicial del baseline: **$140.435**
    * Costo con nuestra solución final: **$80.385**
    * **Ahorro neto final para Mercado Pago:** **$60.050 (-42.7%)** y **$63.400 salvados**.
* **Visual:** Matriz de decisión en 2 fases con métrica hero de ahorro financiero ($63.4k / -42.7%) y logos de la Universidad Austral.
* **Notas / Qué decir:**
  * [Cierre contundente y apertura a preguntas del profesor y la cátedra].
