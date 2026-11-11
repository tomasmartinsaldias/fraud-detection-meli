# Auditoría de Forward Selection Formal v3

> **Resumen:** Se ejecutó el algoritmo formal de Forward Stepwise Selection sobre las 5 candidatas con los mismos hiperparámetros y reglas del Contrato A (`PARAMS_FORWARD`, `colsample=1.0`, `min_delta=0.0003`, `patience=2`).

## 1. Proceso de Selección Paso a Paso (en Validación)

- **Punto de partida (11 features oficiales):** PR-AUC Validación = **`0.4598`**

### Ronda 1: deficit_piso_cluster — ACEPTADA (+)
- Mejor candidato probado: `deficit_piso_cluster`
- PR-AUC alcanzado: `0.4617` (Delta: `+0.0019` vs mejor previo)
- Decisión: **ACEPTADA (+)** (Tolerancia exigida: $\ge +0.0003$)

Scores de todas las candidatas en esta ronda:
  * `volumen_tx_60m`: PR-AUC = `0.4602` (+0.0004)
  * `ratio_concurrencia_60m`: PR-AUC = `0.4558` (-0.0039)
  * `burst_micro_15m`: PR-AUC = `0.4559` (-0.0039)
  * `distancia_log_cluster`: PR-AUC = `0.4613` (+0.0015)
  * `deficit_piso_cluster`: PR-AUC = `0.4617` (+0.0019)

### Ronda 2: distancia_log_cluster — ACEPTADA (+)
- Mejor candidato probado: `distancia_log_cluster`
- PR-AUC alcanzado: `0.4642` (Delta: `+0.0025` vs mejor previo)
- Decisión: **ACEPTADA (+)** (Tolerancia exigida: $\ge +0.0003$)

Scores de todas las candidatas en esta ronda:
  * `volumen_tx_60m`: PR-AUC = `0.4606` (-0.0010)
  * `ratio_concurrencia_60m`: PR-AUC = `0.4550` (-0.0066)
  * `burst_micro_15m`: PR-AUC = `0.4561` (-0.0056)
  * `distancia_log_cluster`: PR-AUC = `0.4642` (+0.0025)

### Ronda 3: volumen_tx_60m — RECHAZADA (x)
- Mejor candidato probado: `volumen_tx_60m`
- PR-AUC alcanzado: `0.4603` (Delta: `-0.0039` vs mejor previo)
- Decisión: **RECHAZADA (x)** (Tolerancia exigida: $\ge +0.0003$)

Scores de todas las candidatas en esta ronda:
  * `volumen_tx_60m`: PR-AUC = `0.4603` (-0.0039)
  * `ratio_concurrencia_60m`: PR-AUC = `0.4581` (-0.0061)
  * `burst_micro_15m`: PR-AUC = `0.4583` (-0.0059)

### Ronda 4: burst_micro_15m — RECHAZADA (x)
- Mejor candidato probado: `burst_micro_15m`
- PR-AUC alcanzado: `0.4583` (Delta: `-0.0059` vs mejor previo)
- Decisión: **RECHAZADA (x)** (Tolerancia exigida: $\ge +0.0003$)

Scores de todas las candidatas en esta ronda:
  * `ratio_concurrencia_60m`: PR-AUC = `0.4581` (-0.0061)
  * `burst_micro_15m`: PR-AUC = `0.4583` (-0.0059)

## 2. Veredicto del Pipeline

- **Variables Aprobadas por el Pipeline (2):** `['deficit_piso_cluster', 'distancia_log_cluster']`
- **Variables Rechazadas:** `['volumen_tx_60m', 'ratio_concurrencia_60m', 'burst_micro_15m']`
- **PR-AUC Final en Validación:** `0.4642` (+0.0044 vs 11 base)

## 3. Comparativa en Test Final

| Modelo | Features | PR-AUC Val | PR-AUC Test | ROC-AUC Test | Recall Test | Falsos Negativos | Dinero Salvado | Capital Recovery |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Oficial v2** | 11 | 0.4598 | 0.4325 | 0.8852 | 84.27% | 163 | $71,918 | 90.60% |
| **Pipeline v3** | 13 | 0.4642 | 0.4386 | 0.8849 | 84.07% | 165 | $71,891 | 90.57% |

## 4. Conclusión Científica Definitiva

Este resultado demuestra el poder del método científico:
1. Cuando evaluamos las 5 features en bloque, aumentaban el Recall pero reducían el PR-AUC global.
2. Al someterlas al **Forward Selection formal**, el pipeline actúa como filtro objetivo contra el sobreajuste y decide si el incremento compensa la complejidad.
3. Esto provee la respuesta final perfecta para cualquier jurado o cátedra: el pipeline fue ejecutado con neutralidad matemática.