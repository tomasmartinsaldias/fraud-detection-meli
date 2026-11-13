# Auditoría de Experimento: Señales de Grafos Bipartitos y Rollback al Champion de Parsimonia

**Autores:** Franco & Equipo de Ciencia de Datos  
**Fecha:** Noviembre 2026  
**Script reproducible:** [`scripts/experimento_cuatri_modelo_con_grafo.py`](file:///C:/Users/Franc/Documents/ThinkPad/Github/fraud-detection-meli/scripts/experimento_cuatri_modelo_con_grafo.py)  
**Semilla:** 42  
**Estado:** ❌ **HIPÓTESIS DESCARTADA — ROLLBACK AL CHAMPION OFICIAL DE PARSIMONIA**

---

## 1. Motivación y Formulación de la Hipótesis

A partir de la desanonimización de la cuenta real ($r = \lfloor c \rfloor$, aislando la entidad comodín de Guest Checkout `260445`), se planteó la hipótesis de que las relaciones relacionales de red en el marketplace podían aportar señal predictiva complementaria sobre el catálogo:

1. **Grafo Bipartito Usuario $\leftrightarrow$ Producto ($\mathcal{G}_{U \times P}$):**  
   * Hipótesis: Los ataques organizados de carding o liquidación masiva canalizan múltiples cuentas compradoras distintas (*cuentas mula* o *botnets*) hacia un mismo producto de alta liquidez.  
   * Variable formulada: `in_degree_usuarios_producto_hist` (conteo acumulado estrictamente causal con `shift(1)` de cuentas únicas que compraron previamente la publicación `i`).
2. **Grafo Bipartito Usuario $\leftrightarrow$ Categoría ($\mathcal{G}_{U \times J}$):**  
   * Hipótesis: Cuentas comprometidas (*Account Takeover*) exhiben compras erráticas en verticales no habituales, elevando la dispersión de categorías.  
   * Variable formulada: `dispersion_categorias_usuario` ($\text{categorías\_distintas\_previas} / (\text{tx\_previas} + 1)$).

---

## 2. Fase 1: Evaluación Inicial en Modelo Proxy (`proxy_evaluator.py`)

Ajustado sobre las 17 variables crudas en `train` y evaluado en `val`:

| Variable Candidata | PR-AUC Val | Delta vs Baseline (`0.43129`) | Veredicto Proxy | Split Gain |
| :--- | :---: | :---: | :---: | :---: |
| `in_degree_usuarios_producto_24h` | 0.43083 | -4.7 bps | `NEUTRO` | 494.1 |
| `in_degree_usuarios_producto_hist` | 0.43126 | -0.4 bps | `NEUTRO` | 1142.7 |
| `dispersion_categorias_usuario` | 0.42853 | -27.6 bps | `DEGRADA / RUIDO` | 234.0 |
| **Pack Grafos (24h + Dispersión)** | **0.43383** | **+25.4 bps** | **`APORTA POSITIVO`** | 730.3 |

*Conclusión del Proxy:* La interacción no lineal superó el umbral del evaluador, habilitando el testeo en el modelo de ensamble de producción.

---

## 3. Fase 2: Integración en el Cuatri-Modelo (Stacking Nivel 1 + Nivel 2)

Se entrenaron los 4 inductores incorporando `in_degree_usuarios_producto_hist`, optimizados con **CMA-ES** (`optuna.samplers.CmaEsSampler`) sobre `tune`, con extracción de probabilidades Out-of-Fold (OOF) mediante **5-Fold Stratified CV** (105.000 observaciones de Train):

### A. Rendimiento de Nivel 1 (Inductores Individuales)
* **LightGBM (15 vars con Grafo):** `val_search: 0.48719` | `test: 0.44547` (+11.2 bps vs control)
* **XGBoost (13 vars con Grafo):** `val_search: 0.48880` | `test: 0.44525` (+9.0 bps vs control)
* **CatBoost (15 vars con Grafo):** `val_search: 0.47989` | `test: 0.44300` (-13.5 bps vs control)
* **Random Forest (7 vars con Grafo):** `val_search: 0.45150` | `test: 0.40388` (-404.7 bps vs control)

### B. Rendimiento de Nivel 2 (Metamodelos de Stacking)

| Arquitectura Evaluada | PR-AUC Val (`tune`) | PR-AUC Test (Ciego) | Delta vs Control Ian |
| :--- | :---: | :---: | :---: |
| **🏆 CHAMPION OFICIAL PREVIO (Cuatri-Modelo L1 $C=0.10$)** | 0.49101 | **`0.45093`** | **`+65.8 bps`** |
| Champion Cuatri-Modelo L2 $C=0.20$ | 0.49218 | 0.45014 | $+57.9$ bps |
| Stacking Cuatri + Grafo (Lasso L1 $C=0.10$) | 0.49128 | **0.44972** | $+53.7$ bps |
| Stacking Cuatri + Grafo (Lasso L1 $C=0.20$) | 0.49120 | 0.44964 | $+52.9$ bps |
| Stacking Cuatri + Grafo (Ridge L2 $C=0.20$) | 0.49120 | 0.44964 | $+52.9$ bps |
| Stacking Cuatri Puro (Solo Preds N1) | 0.49062 | 0.44993 | $+55.8$ bps |
| Blend Convexo SLSQP con Grafo | 0.48860 | 0.44768 | $+33.3$ bps |

---

## 4. Diagnóstico Técnico: ¿Por qué se descarta la hipótesis?

1. **Absorción Completa de Señal en Nivel 1:**  
   Los árboles supervisados (LightGBM y XGBoost) ya explotaron la variable en sus ramas, alcanzando ~0.445 en test. Sus logits resultantes (`logit_xgb: +0.755` y `logit_lgb: +0.738`) ya contienen la información del grafo.
2. **Penalización en Nivel 2:**  
   Al alimentar la feature directamente al metamodelo lineal de Nivel 2, la regularización Lasso L1 detectó colinealidad con los logits de los árboles y contrajo su coeficiente a prácticamente cero (**`+0.00075`**).
3. **Pérdida Neta en Test (-1.2 bps):**  
   El ensamble con grafo alcanzó `0.44972` en Test ciego, quedando por debajo de los **`0.45093`** del Champion de Parsimonia oficial. Random Forest sufrió dispersión de impureza al incluir una feature de cola pesada, agregando varianza en lugar de precisión.

---

## 5. Resolución y Decisión de Rollback

Siguiendo el principio rector de **Navaja de Ockham**:
* **Se rechaza formalmente la incorporación de `in_degree_usuarios_producto_hist`** al pipeline de producción.
* **Se realiza rollback al Champion Oficial de Parsimonia:**
  * **Arquitectura:** Cuatri-Modelo Clásico (LightGBM + XGBoost + CatBoost + Random Forest) sobre las 13 variables limpias de Challenger v3.
  * **Metamodelo:** Regresión logística con regularización Lasso L1 estricta ($C=0.10$).
  * **Métricas Oficiales Vigentes:**  
    * **PR-AUC Val (`tune`):** `0.49101`  
    * **PR-AUC Test Final (Ciego):** **`0.45093`** (+65.8 bps sobre el Control).

El repositorio y sus scripts productivos quedan preservados en su estado Champion, cerrando esta línea de exploración.
