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

## Modelado supervisado (integrante 3)

El pipeline de Ian está en `src/models/train.py`. Incluye regresión logística,
LightGBM con Optuna (TPE + pruning y CMA-ES), calibración Platt, ensamble y costos
de la consigna (FP=5, FN=200), con umbral elegido en validación.
Consultar [metodología y comandos](docs/MODELADO_IAN.md) y los
[resultados oficiales](docs/RESULTADOS_OFICIAL.md) (features de Tomás + scores de Franco).
La primera corrida está en [RESULTADOS_IAN.md](docs/RESULTADOS_IAN.md) y la comparación
exploratoria del equipo en [COMPARACION_EQUIPO.md](docs/COMPARACION_EQUIPO.md).
Los modelos y datos generados quedan locales, ignorados por Git.

## Diagnóstico de Drift, Residuos y Modelo Challenger v3

- [Auditoría Metodológica de Drift y Anatomía del Error (Val vs Test)](docs/DIAGNOSTICO_DRIFT.md): análisis de covariate shift (PSI, $D_{KS}$), intervalos bootstrap al 95%, desfasaje de días de la semana (corte del lunes 13/04 y ausencia de miércoles) y métrica de Capital Recovery Rate (90.4%).
- [Modelo Challenger v3 — Hipótesis, Forward Selection y Negocio](docs/MODELO_CHALLENGER_V3.md): informe unificado que consolida la hipótesis de clustering semántico en títulos `i`, la traza paso a paso del Forward Stepwise Selection formal (`colsample=1.0`, `min_delta=0.0003`, `patience=2`), la optimización de $k=50$ clusters y la evaluación final en Test elevando el PR-AUC a 0.4642 en Validación y 0.4386 en Test (+61 bps).
- [Stacking Cuatri-Modelo y Récord Absoluto de PR-AUC](docs/EXPERIMENTO_CUATRI_MODELO_STACKING.md): ensamble de Nivel 1 con LightGBM, XGBoost, CatBoost y Random Forest sobre las 13 variables limpias de Challenger v3, combinado con Metamodelo Lasso L1 de Nivel 2 (probabilidades OOF + 11 anomalías no supervisadas + reglas heurísticas). Alcanza **0.4912 en Validación** y **0.4506 en Test**, quebrando por primera vez la barrera de 0.4500 fuera de muestra.

Las dependencias de pruebas están en `requirements-dev.txt`.
