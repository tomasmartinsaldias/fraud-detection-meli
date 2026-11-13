# Deteccion de fraude MELI

## Entorno Python en Windows

La version acordada se declara en `.python-version` (actualmente 3.14); se acepta cualquier revision 3.14.x.
Instalar Python con el launcher `py` antes de comenzar. No hace falta activar
manualmente un entorno virtual.

Desde PowerShell:

```powershell
.\run.bat src/make_dataset.py --help
.\run.bat src/make_dataset.py --input_path data/raw/fraud_dataset_v2.csv
```

El lanzador selecciona Python 3.14, crea `.venv` si falta, comprueba las versiones
de `requirements.txt`, instala o corrige las dependencias si es necesario y
ejecuta `pip check` antes de ejecutar el script. La primera instalacion requiere
acceso a internet. Los argumentos se pasan al script y su codigo de salida se
conserva. Los scripts se ejecutan desde la raiz del repositorio.

Si `.venv` ya existe, el lanzador verifica que su Python sea ejecutable y tenga
la version acordada. Si no es compatible, lo conserva como `.venv-backup` (o
un nombre numerado) y crea un entorno nuevo. Si todas las
dependencias tienen las versiones requeridas, no ejecuta `pip install` ni
descarga paquetes: solo compara las versiones locales y ejecuta `pip check`.
El lanzador no queda encendido ni se activa automaticamente al ejecutar otros
comandos: cada script debe iniciarse mediante `run.bat` para aplicar el control.

Si falta Python 3.14, el launcher `py` informa el error. Al cambiar la version acordada, actualizar `.python-version`
y revisar la compatibilidad de las dependencias.

Usar `run.bat` como punto de entrada para todos los scripts del proyecto.
Ejecutar `python script.py` directamente saltea estas comprobaciones. En un IDE,
seleccionar `.venv/Scripts/python.exe` como interprete.

Las dependencias directas estan fijadas; `requirements.txt` no es un lock completo
de las dependencias transitivas. Los datos no se incluyen en Git: obtener el
dataset base oficial del equipo o generar el Parquet desde el CSV original.

## Síntesis Oficial de Modelado y Rendimiento

A continuación se resume la evolución y el desempeño comparativo de los hitos principales del equipo, evaluados bajo el protocolo cronológico out-of-time (OOT) sobre el mismo split temporal (105k Train / 22.5k Val / 22.5k Test) y confirmados a ciegas en Test:

| Solución / Hito | Arquitectura | Features (N1 / N2) | PR-AUC Val (OOT) | PR-AUC Test | Δ Acumulado vs Crudo | ¿Qué aporta este paso? |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **0. Árbol Crudo de Partida** | LightGBM base de fábrica (sin FE) | 16 / — | 0.44830 | 0.43135 | — | 📌 **Punto Cero:** LightGBM alimentado con las variables brutas tal como vienen del CSV. |
| **1. FE Transaccional y Temporal** | LightGBM 11 variables seleccionadas | 11 / — | 0.48136 | 0.44435 | **+130.0 bps** | 🧠 **Impacto FE Ola 1:** Ratios continuos, ventana nocturna, interacciones y Naive Bayes. |
| **2. Baseline Comercial (Finetuneado)** | LightGBM Optuna con semántica ($d=30$) | 13 / — | 0.48850 | **0.44700** | **+156.5 bps** | 🚀 **Impacto FE Ola 2:** Incorpora clustering semántico NLP. Inferencia ultra-rápida (< 4 ms). |
| **3. 🏆 CHAMPION OFICIAL (Protocolo OOT)** | Cuatri-Modelo Stacking (Lasso $C=0.20$) | 13 / 14 | **`0.49218`** | **0.45014** | **+187.9 bps** | 🎯 **Aporte del Ensamble:** Récord OOT (`0.49218`). Deduplica en N2 y supera 0.45 en Test. |
| **4. 🔬 Récord de Generalización (L1)** | Cuatri-Modelo Stacking (Lasso $C=0.10$) | 13 / Descol. | 0.49101 | **`0.45093`** | **+195.8 bps** | 🥇 **Aporte de la Parsimonia:** Esparcidad estricta $L_1$ logra el récord histórico del proyecto. |

## Arquitectura de Decisión y Políticas de Riesgo (Del Score al Negocio)

Para trasladar las probabilidades del Champion Oficial a políticas de riesgo accionables en pasarela de pagos, se estructuró una **jerarquía metodológica en tres niveles**, evaluada fuera de muestra sobre los 22.500 casos de Test:

| Nivel | Rol Metodológico | Política Operativa | FP Duro (Bloqueos) | Recall Fraude | Capital Rescatado | Costo del Sistema |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **1** | **El Contrato Académico** | Umbral Estático ($\tau=0.2550$) | 477 | 43.24% | $47.164 (55.6%) | $119.985 |
| **2** | **Optimización Matemática Pura** | Curva S Acotada ($\tau \in [0.18, 0.40]$) | 496 | 44.21% | $54.372 (64.1%) | $118.080 |
| **3** | **Visión Estratégica FinTech** | Tripartita con 2FA (Verde / Amarillo / Rojo) | 504 | **62.45%** | **$67.747 (79.9%)** | **$80.385 (-33%)** |

> 📖 **Auditoría Completa:** Consultar la [Sección 1.2 de Resultados Oficiales](docs/RESULTADOS_OFICIAL.md#12-arquitectura-de-decisión-y-umbrales-en-producción-la-trilogía-metodológica) para las fórmulas matemáticas, curvas de decisión, stress test de fricción/phishing y la propiedad de fail-safe del carril rojo.

## Mapa de Documentación

* [Resultados Oficiales y Contrato C](docs/RESULTADOS_OFICIAL.md): Registro consolidado de corridas oficiales, hiperparámetros de búsqueda y métricas financieras de política.
* [Cuatri-Modelo Stacking y Selección en Nivel 2](docs/EXPERIMENTO_CUATRI_MODELO_STACKING.md): Ensamble de 4 familias de inductores (LightGBM, XGBoost, CatBoost, Random Forest), forward selection de variables semánticas, desacople arquitectónico Nivel 1 vs Nivel 2 y grid search fino.
* [Modelo Challenger v3 y Clustering Semántico](docs/MODELO_CHALLENGER_V3.md): Diseño de features a partir de títulos `i` mediante TF-IDF, SVD y K-Means, y estudio de ablación.
* [Diagnóstico de Drift y Anatomía de Residuos](docs/DIAGNOSTICO_DRIFT.md): Análisis de Covariate Shift, Prediction Drift (PSI = 0.00065) y optimización de umbral bajo costo monetario (-58.4% de falsos positivos en compras lícitas).
* [Metodología de Modelado y Pipelines](docs/MODELADO_IAN.md): Pipeline oficial de entrenamiento, calibración Platt y optimización de hiperparámetros con Optuna.
* [Diccionario de Variables](docs/DICCIONARIO_VARIABLES.md): Especificación técnica de features raw, transformadas, temporales y semánticas.
* [Archivo Histórico](docs/archive/README.md): Registro de reportes y pruebas de etapas exploratorias preliminares.

## Arquitectura del Repositorio

El código fuente en `src/` y los experimentos se organizan con responsabilidades desacopladas:

```plaintext
src/
├── features/                  # Ingeniería de features transaccionales, temporales y semánticas
├── models/
│   ├── train.py               # Pipeline de entrenamiento supervisado y Optuna
│   ├── supervised.py          # Módulo core: métricas, splits OOT y calibración Platt
│   ├── prepare_supervised.py  # Preparación y ensamblado de datasets
│   ├── report_supervised.py   # Generación de reportes y curvas de decisión
│   ├── anomaly_detection.py   # Entrypoint unificado de anomalías
│   ├── compare_team.py        # Benchmarking cruzado de integrantes
│   ├── estimators/            # Estimadores de anomalías serializados (.joblib)
│   ├── production/            # Generación de features de anomalías en inferencia
│   ├── evaluation/            # Evaluadores y métricas de soporte
│   ├── anomaly_experiments/   # Pruebas de laboratorio por familia (IForest, LOF, PCA, OCSVM)
│   └── archive/               # Ensayos exploratorios archivados (stackings preliminares, SMOTE)
scripts/                       # Scripts ejecutables reproducibles del champion y ablaciones
docs/                          # Documentación metodológica canónica y memorias técnicas
```

Las dependencias de pruebas están en `requirements-dev.txt`.
