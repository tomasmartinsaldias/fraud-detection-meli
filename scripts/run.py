"""Prepare the project environment and run a Python script from the repo root."""

from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
import venv


ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / ".venv"
PYTHON = ENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
REQUIREMENTS = ROOT / "requirements.txt"


def main() -> int:
    expected = (ROOT / ".python-version").read_text().strip()
    actual = f"{sys.version_info.major}.{sys.version_info.minor}"
    if actual != expected:
        print(f"Se requiere Python {expected}; se recibio {actual}.", file=sys.stderr)
        return 1
    if len(sys.argv) < 2:
        print("Uso: run.bat <script.py> [argumentos del script]")
        print("Ejemplo: run.bat src/make_dataset.py --help")
        return 0

    script = (ROOT / sys.argv[1]).resolve()
    if not script.is_file():
        print(f"No existe el script: {script}", file=sys.stderr)
        return 1

    if Path(sys.prefix).resolve() != ENV.resolve():
        if not PYTHON.exists():
            print(f"Creando .venv con Python {expected}...", flush=True)
            venv.create(ENV, with_pip=True)
        # Re-enter inside the environment so metadata checks inspect its packages.
        return subprocess.call(
            [str(PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]], cwd=ROOT
        )

    missing = []
    for line in REQUIREMENTS.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, version = line.split("==", 1)
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        if installed != version:
            missing.append(line)

    if missing:
        print("Instalando dependencias de requirements.txt...", flush=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)],
            cwd=ROOT, check=True,
        )
    subprocess.run([sys.executable, "-m", "pip", "check"], cwd=ROOT, check=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.call([sys.executable, str(script), *sys.argv[2:]], cwd=ROOT, env=env)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print("No se pudo preparar el entorno. Revisa el error anterior.", file=sys.stderr)
        raise SystemExit(exc.returncode)
