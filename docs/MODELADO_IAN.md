# Modelado supervisado y negocio — Ian (integrante 3)

Rama local: `feat/ian-modelado-supervisado`. No se realizan commits ni push.

## Alcance y decisiones

Implementación de Contrato C: integración por `transaction_id`, regresión logística,
LightGBM, comparación de variables crudas / seleccionadas / seleccionadas con anomalías,
Optuna TPE y CMA-ES, calibración Platt, promedio de probabilidades, política de bloqueo,
curvas PR/ROC/costo, sensibilidad, exportación del modelo y evaluación final separada.

La consigna primaria es el PDF `Trabajo Práctico Integrador — Examen Parcial.pdf`
dentro del ZIP `Trabajo práctico integrador para cerrar la cursada-20261107.zip`
entregado por el usuario. Define **costo = 5 × FP + 200 × FN**. La moneda no se
especifica más allá del símbolo `$`; no se afirma que sean dólares estadounidenses.
Esto prevalece sobre la pérdida variable por monto sugerida en CONTRATOS.md.
El módulo económico también admite montos al usar `fn_cost=None` desde Python.

El contrato contiene una contradicción: propone elegir el umbral en test y a la vez
mantener test sellado. Se resuelve eligiendo **todos los umbrales en validación**,
coincidiendo con la consigna docente. No se cambia de ganador después de observar test.

## Datos y generación

Se encontró el CSV en la carpeta vecina `MELI-fraude`; para la ejecución se utiliza
el CSV comprimido dentro del ZIP que indicó el usuario, como fuente identificable.
Se conserva la exclusión del segundo bloque establecida por el EDA del equipo:
150.000 transacciones, train/val/test temporal 70/15/15. No es una decisión nueva
basada en el rendimiento de los modelos.

`src/models/prepare_supervised.py` reutiliza las funciones de ingesta y las fórmulas
de las once variables seleccionadas, sin repetir la selección sobre validación.
Genera `dataset_base.parquet`, `ian_features_selected.parquet` y
`ian_scores_anomalias.parquet`. No sobrescribe artefactos existentes. No calcula
métricas ni usa etiquetas de val/test para ajustar transformaciones.

Correcciones y límites de los insumos:

- El score de texto original completaba el primer bloque temporal con la prevalencia
  de todo train. La variante de Ian deja ese bloque como faltante y entrena TF-IDF
  y Naive Bayes únicamente con fechas anteriores a cada bloque OOF. LightGBM maneja
  ese faltante sin incorporar etiquetas futuras. El resto usa Naive Bayes ajustado
  solamente en train. Se guardan vectorizador y estimador.
- Se preservan faltantes naturales (como la caída de b/c). Se rechazan IDs ausentes,
  duplicados o inconsistencias al unir; la ausencia natural de una medición no es
  lo mismo que una fila perdida en el merge. Logística imputa exclusivamente con train.
- No estaban los Parquet de Franco. Se implementa una alternativa explícita con
  Isolation Forest, PCA y distancia media de 20 vecinos sobre muestra de 10.000
  legítimas, ajustados únicamente con legítimas de train. Se normalizan con percentiles
  de ese mismo train y se promedian sin etiquetas para el score consolidado.
  Se excluye `score` como predictor de estas anomalías. **No se presenta esta variante
  como el bundle completo de Franco.** Puede sustituirse por su Parquet mediante CLI.
- El bundle existente de Franco incluye un consolidador supervisado entrenado sobre
  todo train. Su salida in-sample no debe incorporarse como stacking sin OOF; esta
  ejecución usa un promedio no supervisado para evitar ese problema.
- Las once variables y la ventana nocturna fueron seleccionadas previamente con val.
  Dividir ahora val no elimina ese sesgo previo; las métricas de desarrollo siguen
  siendo exploratorias. Test es la evaluación independiente final.

## Entrenamiento y selección

Se ajusta todo modelo exclusivamente en train. Val se ordena por fecha y se divide
sin separar timestamps iguales:

| Bloque | Fracción aproximada de val | Uso |
|---|---:|---|
| val_search | 60% | Hiperparámetros, early stopping, elección del ganador |
| val_calibration | 20% | Ajustar Platt sobre logit de probabilidad |
| val_policy | 20% | Elegir umbral, costos y límite de FP |

La métrica primaria es **PR-AUC trapezoidal**, compatible con los reportes existentes.
También se reporta Average Precision por separado: no son exactamente la misma métrica.
Se incluyen ROC-AUC, Brier, log loss, precision, recall, FP/FN/TP/TN y ahorro frente
a permitir todas las transacciones. No se usa accuracy como criterio de elección.

Cada grupo de variables tiene un LightGBM de referencia fijo y dos búsquedas
con el mismo presupuesto, rangos y datos. Por defecto son 20 trials por sampler y
grupo (120 trials si están los tres grupos). TPE y CMA-ES usan MedianPruner,
reportando PR-AUC en cada iteración, con 5 trials iniciales y 50 iteraciones de
calentamiento. Early stopping: 60 iteraciones, máximo 800 árboles. Los rangos
incluyen learning rate, hojas, mínimo de muestras por hoja, submuestreo de filas
y columnas y regularización L1/L2. No hay parámetros categóricos incompatibles con CMA-ES.
Los trials son secuenciales y la semilla es 42.

Se reconstruye el mejor modelo de cada búsqueda en train con la cantidad elegida
de árboles. No se reentrena con val: así se conserva la validez de calibración y
umbrales. El promedio de todos los candidatos es un candidato adicional con pesos
iguales; no es stacking aprendido. Se elige el mayor PR-AUC en val_search, nunca en test.
No se promete un óptimo global; es el mejor candidato de esta comparación acotada.

## Política de negocio

El escenario principal es FP=5, FN=200. Sensibilidad: FP=10 y FP=20, manteniendo FN=200.
La CLI usa un límite propuesto de **100 FP en val_policy** (aproximadamente 4.500
transacciones), equivalente a un presupuesto de contacto de 500 unidades monetarias
en ese bloque bajo el escenario principal. Es un supuesto inicial de capacidad;
no hay dotación real de operadores informada. Se puede cambiar con `--max-fp`.

Se minimiza el costo entre todos los umbrales distintos que cumplen ese límite,
incluyendo permitir todo. Los empates de probabilidad se bloquean juntos. No se
optimiza separando transacciones que tienen el mismo score. El límite se verifica
en validación y **no garantiza** el mismo número de FP en test o producción: depende
del volumen y del cambio de distribución. Hay que reportar esa diferencia, no retocar
el umbral al verla. Los costos son totales del período evaluado, no costos mensuales
sin ajustar por la duración y volumen reales del período.

Con probabilidades perfectamente calibradas y sin restricción operativa, el umbral
teórico del costo fijo es 5/(5+200) ≈ 0,02439. El umbral empírico puede diferir por
calibración imperfecta, variación muestral y la restricción de FP.

## Comandos reproducibles

El entorno oficial del repositorio es Python **3.14**, según `.python-version` y README.
CONTRATOS.md aún menciona 3.13: esa referencia está desactualizada. Usar `run.bat`
con Python 3.14 instalado para la ejecución oficial.

```powershell
# Preparación, una sola vez; ajustar la ubicación del ZIP.
.\run.bat src/models/prepare_supervised.py --source-zip '..\Trabajo práctico integrador para cerrar la cursada-20261107.zip'

# Desarrollo: no evalúa test.
.\run.bat src/models/train.py --features data/features/ian_features_selected.parquet --anomalies data/features/ian_scores_anomalias.parquet --output models/supervised/ian_v1 --seed 42 --trials 20 --fp-costs 5 10 20 --fn-cost 200 --max-fp 100

# Evaluación final explícita, solo una vez tras congelar modelo y política.
.\run.bat src/models/train.py --evaluate-test --bundle models/supervised/ian_v1/bundle.joblib

# Informe legible y curvas SVG desde los resultados congelados.
.\run.bat src/models/report_supervised.py --run models/supervised/ian_v1

# Pruebas específicas.
.\run.bat tests/test_supervised.py
```

Para usar las entregas oficiales de los compañeros, pasar sus rutas a `--features`
y `--anomalies`. Para correr solo con lo disponible, cada una acepta `none`, de forma
explícita. No hay omisiones silenciosas. Cada corrida requiere un directorio nuevo.

Durante esta sesión no había `py` ni Python 3.14 instalados en PATH. Se creó un
`.venv` local con Python **3.12.14** del runtime auxiliar, usando las mismas versiones
fijadas de dependencias. Los comandos de sesión usan `.venv\Scripts\python.exe`;
esto es una excepción documentada para poder verificar y entrenar ahora, no un
cambio de la versión acordada. `run.bat` conservará ese entorno como backup cuando
se use con 3.14. Falta verificar específicamente el runtime 3.14.

## Artefactos y trazabilidad

En `models/supervised/<corrida>/` (ignorado por Git):

- `bundle.joblib`: estimadores, preprocesadores, calibradores, umbrales y ensamble.
- `manifest.json`: ganador, versiones, filas por bloque, métricas, costos y hashes.
- `trials_*.json`: sampler, pruner, rangos, parámetros, valores intermedios, duración
  y estado de cada trial, incluyendo podados, y mejor trial.
- `report.md`: comparación de modelos en búsqueda.
- `*_importance.json`: importancia gain de variables para interpretación descriptiva;
  no implica causalidad.
- `*_pr.csv`, `*_roc.csv`, `*_cost_*.csv`: puntos completos para curvas.
- `*_val_predictions.parquet`: probabilidades por transaction_id en val_policy.
- `test_report.json`: evaluación final con modelo y umbrales previamente congelados.

Antes de evaluar test se verifica el hash del bundle y que los datos de desarrollo
no hayan cambiado. Se escribe una marca exclusiva antes de leer test, bloqueando
repeticiones accidentales. Si falla una evaluación iniciada, la marca se conserva:
hay que diagnosticar la causa antes de decidir una recuperación manual. Esta es una
protección por carpeta de corrida, no un registro central de todas las ejecuciones.

## Pruebas y pendientes

Las pruebas cubren integridad del merge, categorías desconocidas, costos con empates,
la fórmula fija de la consigna y un límite de capacidad,
particiones temporales y un flujo completo con datos sintéticos, ambos samplers,
serialización, evaluación test y rechazo de repetición. El test intercepta lecturas
para comprobar que desarrollo pide exclusivamente train/val. Los resultados sintéticos
no se usan como evidencia de calidad predictiva.

Pendientes de cierre de equipo: verificar procedencia y versión de la entrega definitiva
de Franco, acordar capacidad operativa real y completar la presentación conjunta.
La detección exploratoria de casos nuevos corresponde al rol de anomalías; no se
identifican transacciones legítimas como fraude confirmado solo por un score alto.

## Referencias técnicas

- [LightGBM: callbacks y métricas de evaluación](https://lightgbm.readthedocs.io/en/v4.6.0/pythonapi/lightgbm.LGBMClassifier.html).
- [Optuna: CMA-ES y limitaciones de parámetros categóricos](https://optuna.readthedocs.io/en/v4.3.0/reference/samplers/generated/optuna.samplers.CmaEsSampler.html).

## Resultados de esta sesión

Corrida real completada: `models/supervised/ian_v1`. Se entrenaron 120 trials
(68 podados), las referencias fijas, regresión logística y promedio de probabilidades.
Train: 105.000 filas; val_search: 13.499; val_calibration: 4.500; val_policy: 4.501;
test final: 22.500. No se retocó el modelo ni la política después de abrir test.

Ganador en val_search: **LightGBM con las once variables seleccionadas, TPE**,
PR-AUC 0,4852. Las anomalías y el promedio no ganaron en esa comparación.
En test: PR-AUC **0,4432**, precision **47,2%**, recall **45,7%**, **530 FP** y
**563 FN**, umbral **0,20581**. Costo: **115.250**, frente a **140.435** de logística:
reducción **17,9%** bajo las políticas congeladas. Son resultados de ese período,
no una promesa de rendimiento futuro.

El ensamble tuvo PR-AUC ligeramente superior en test (0,4436), pero mayor costo
(118.730); no se cambia de ganador por observar esa evaluación. Platt mejoró Brier
en val_policy, pero lo empeoró ligeramente en test (0,032846 a 0,032881). Se informa
la limitación y se conserva el método congelado.

El tope de 100 FP se cumplió en val_policy (99). Test tiene aproximadamente cinco
veces más filas: proporcionalmente serían unos 500 FP, frente a los 530 observados.
Es evidencia de variación entre períodos y de que no hay garantía dura de capacidad
fuera de muestra. Los tres escenarios conservan el mismo umbral porque domina esa
restricción; al subir FP de 5 a 20 el costo final sube a 123.200.

Suite completa: **25 passed, 1 skipped**, con 9 advertencias de los tests existentes
por muestras pequeñas. El test omitido requiere el Parquet de reglas de asociación,
que no está generado. `pip check` pasó y `git diff --check` no reportó problemas.
La comprobación se realizó con Python 3.12.14; no se afirma validación en 3.14.

Tablas, tiempos por sampler, sensibilidad, variables importantes y curvas están en
[RESULTADOS_IAN.md](RESULTADOS_IAN.md). Los modelos binarios y datos permanecen locales.


## Actualizacion: integracion oficial y ensambles

Se completo la integracion de los scripts de Tomas y Franco. El CSV original esta
en data/raw; las 50 variables, las once seleccionadas y los doce scores del bundle
original ya estan generados. Fue necesario corregir la incompatibilidad r/s del
preprocesador de Franco con la base limpia. Se preservaron las variantes Ian y
la corrida ian_v1 para trazabilidad.

Se compararon 24 candidatos, con 72 nuevos trials y ensambles de probabilidades.
La recomendacion sigue siendo ian_v1: el candidato elegido antes de la comprobacion
posterior aumento FP de 99 a 112, FN de 94 a 97 y costo de 19295 a 19960.
No se volvio a evaluar test ni se entreno con sus etiquetas. Esta segunda corrida
es exploratoria: no hay una nueva muestra independiente para confirmar mejoras.

Ver [COMPARACION_EQUIPO.md](COMPARACION_EQUIPO.md) para la comparacion, sensibilidad
y limitaciones. Pruebas tras la integracion: 30 aprobadas, sin omisiones.
