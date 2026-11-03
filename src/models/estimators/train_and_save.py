"""
Script de entrenamiento único y serialización de todos los modelos de anomalías.
Ajusta los estimadores sobre la partición 'train' y guarda el binario en models/anomaly_bundle.joblib.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import pandas as pd

from src.models.estimators.anomaly_bundle import AnomalyPipelineBundle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_and_save")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/dataset_base.parquet"),
        help="Ruta al dataset base.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("models/anomaly_bundle.joblib"),
        help="Ruta donde serializar el bundle de modelos.",
    )
    parser.add_argument(
        "--lof-sample-size",
        type=int,
        default=20000,
        help="Cantidad de observaciones legítimas de train para LOF (default: 20000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla determinista.",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    if not args.input.exists():
        raise FileNotFoundError(f"No se encontró el dataset base en: {args.input.resolve()}")

    logger.info(f"Cargando partición 'train' desde: {args.input}")
    df_train = pd.read_parquet(args.input, filters=[("split", "==", "train")])
    logger.info(f"Filas de train cargadas: {len(df_train):,}")

    bundle = AnomalyPipelineBundle(
        pca_variance=0.95,
        iforest_n_estimators=150,
        iforest_max_samples=256,
        lof_sample_size=args.lof_sample_size,
        seed=args.seed,
    )
    bundle.fit(df_train)
    bundle.save(args.model_output)

    print(f"\nEntrenamiento y serialización completados en {time.perf_counter() - started:.2f} s.")


if __name__ == "__main__":
    main()
