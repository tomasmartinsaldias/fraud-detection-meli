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

## Síntesis Oficial de Modelado y Resultados

A continuación se resume la evolución y el desempeño comparativo de los hitos principales del equipo, evaluados bajo el protocolo cronológico out-of-time (OOT) y confirmados a ciegas en Test:

| Hito / Arquitectura | Autor / Fase | Features | PR-AUC Val (OOT) | PR-AUC Test | Δ vs Base | Diagnóstico Metodológico |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Control Baseline (Ian)** | Ian | 11 | 0.48136 | 0.44435 | — | Modelo productivo monomodal base (LightGBM). |
| **Tri-Modelo Campeón Previo** | Franco | 11 | 0.48887 | 0.44450 | +1 bp | Ensamble preliminar (LGBM + XGB + RF). Techo histórico inicial de Val. |
| **Challenger v3 (Semántico)** | Tomás | 14 | 0.48725 | 0.44661 | +23 bps | Incorporación de SVD ($d=30$), K-Means ($k=50$) y ratio de precio continuo. Rompe la barrera monomodal. |
| **Cuatri-Modelo Stacking Oficial** | Equipo Consolidado | 14 (Nivel 1) / Desacoplado | **`0.49212`** | **`0.44920`** | **`+48.5 bps`** | 🏆 **Champion Oficial:** Stacking (LGBM + XGB + CatBoost + RF) con metamodelo Lasso ($L_1, C=0.30$). Récord en Validación y máxima ganancia generalizada. |

## Mapa de Documentación

* [Resultados Oficiales y Contrato C](docs/RESULTADOS_OFICIAL.md): Registro consolidado de corridas oficiales, hiperparámetros de búsqueda y métricas financieras de política.
* [Cuatri-Modelo Stacking y Selección en Nivel 2](docs/EXPERIMENTO_CUATRI_MODELO_STACKING.md): Ensamble de 4 familias de inductores (LightGBM, XGBoost, CatBoost, Random Forest), forward selection de variables semánticas, desacople arquitectónico Nivel 1 vs Nivel 2 y grid search fino.
* [Modelo Challenger v3 y Clustering Semántico](docs/MODELO_CHALLENGER_V3.md): Diseño de features a partir de títulos `i` mediante TF-IDF, SVD y K-Means, y estudio de ablación.
* [Diagnóstico de Drift y Anatomía de Residuos](docs/DIAGNOSTICO_DRIFT.md): Análisis de Covariate Shift, Prediction Drift (PSI = 0.00065) y reducción de Falsos Positivos (-58.4%).
* [Metodología de Modelado y Pipelines](docs/MODELADO_IAN.md): Pipeline oficial de entrenamiento, calibración Platt y optimización de hiperparámetros con Optuna.
* [Diccionario de Variables](docs/DICCIONARIO_VARIABLES.md): Especificación técnica de features raw, transformadas, temporales y semánticas.
* [Archivo Histórico](docs/archive/README.md): Registro de reportes y pruebas de etapas exploratorias preliminares.

Las dependencias de pruebas están en `requirements-dev.txt`.
