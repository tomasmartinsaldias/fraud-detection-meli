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
