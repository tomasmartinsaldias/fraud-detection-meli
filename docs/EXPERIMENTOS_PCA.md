# Banco exploratorio de PCA

Desde la raíz del repositorio, ejecutar:

```powershell
.\run.bat src/models/pca_experiments.py --seed 42
```

Para usar otro Parquet y excluir columnas adicionales, indicando la lista completa de exclusiones:

```powershell
.\run.bat src/models/pca_experiments.py --input data/processed/otro_dataset.parquet --exclude c k i columna_extra
```

Por defecto se excluyen `c`, `k` e `i`; `--exclude` reemplaza esa lista. Siempre se excluyen `transaction_id`, `fecha`, `split` y `fraude`. El resto de las columnas numéricas se usa como numéricas; las de texto, categóricas o booleanas se codifican por frecuencia. Revisar cuidadosamente la selección mostrada por consola: detectar el tipo no demuestra que una variable sea causalmente válida. El archivo de entrada debe contener `split` (`train`/`val`) y `fraude` (0/1) con ambas clases en cada partición. Se requiere al menos una variable numérica y un Parquet compatible con el filtro temporal.

Para guardar las probabilidades de validación, usar el parámetro binario `--export-probs 1`; `0` es el valor por defecto y no escribe ese archivo:

```powershell
.\run.bat src/models/pca_experiments.py --export-probs 1 --probs-output data/experiments/pca_probabilidades_val.parquet
```

El Parquet Snappy contiene una fila por observación de `val`, en su orden de entrada, con `transaction_id` primero y 12 columnas `pca_<porcentaje>_<indicador>`; `tres_scores` identifica la logística conjunta. El identificador debe ser string, único y no nulo en `val`. El archivo se reemplaza al repetir la ejecución con la misma ruta. Las probabilidades son salidas exploratorias de las logísticas, no necesariamente probabilidades calibradas ni scores del contrato de intercambio.

La consola muestra al inicio una referencia aproximada de duración y al final el tiempo real del script, además del informe. El tiempo de referencia se basa en dos corridas anteriores del dataset de este proyecto y puede cambiar con el tamaño y el entorno. El reporte se escribe en `data/experiments/pca_report_frequency_90_95_99.md` y se reemplaza en cada ejecución. Los reportes anteriores permanecen disponibles; el de `pca_report_frequency.md` incluía `c` y `k` por error y no debe interpretarse como una comparación aislada del encoding. Para una selección explícita de variables o para conservar otro reporte:

```powershell
.\run.bat src/models/pca_experiments.py --features monto a b --categorical g o --output data/experiments/pca_alternativo.md
```

Si se usa `--features` o `--categorical`, solo se usan las columnas explícitas en ambos argumentos y se siguen aplicando las exclusiones. Los valores numéricos faltantes se imputan por mediana; el encoding categórico usa proporciones de legítimos de train, tratando nulos como categoría y categorías no vistas como frecuencia 0. Las frecuencias, medianas y el escalado se ajustan exclusivamente con legítimos de train. El reporte contabiliza categorías no vistas. El mapa fijo de entrenamiento incluye la propia transacción legítima en su frecuencia, por lo que no simula información histórica disponible para cada fila.

Se ajustan tres PCA, con mínimos de varianza explicada del 90 %, 95 % y 99 %. Para cada uno se prueban tres scores: error de reconstrucción Q, suma de coordenadas al cuadrado y suma ponderada por el inverso de la varianza de cada componente (T²). Cada score alimenta una regresión logística independiente y una cuarta logística recibe los tres scores juntos. Las logísticas se entrenan con las dos clases de train; el escalado previo de cada score usa solo legítimos de train. Las métricas se calculan en val, sin leer filas de test. El informe conserva los rangos originales y los coeficientes para ayudar a interpretar los resultados.

Las doce comparaciones son exploratorias: usar repetidamente val para elegir variantes puede sobreajustar las decisiones a ese período. Una logística de una variable también puede sobreajustarse, aprender una dirección equivocada o fallar ante cambios temporales. En la logística conjunta los coeficientes pueden ser inestables si los indicadores están correlacionados. Sus métricas no prueban que la feature mejore al modelo final cuando se combine con otras. Los scores de train se calculan con un PCA que vio a los legítimos de ese conjunto, por lo que pueden ser optimistas frente a datos nuevos.

Este banco no produce el artefacto compartido de `CONTRATOS.md`: faltan decisiones de normalización, otros detectores e integración. El signo negativo de una regresión tampoco cambia la definición del score de anomalía ni prueba que un score alto implique menos fraude en otros períodos.
