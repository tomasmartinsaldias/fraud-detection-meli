"""
Pipeline productivo de generación de features de anomalías no supervisadas.
Toma los estimadores congelados desde models/anomaly_bundle.joblib, transforma
todas las observaciones de dataset_base.parquet (train, val, test) a percentiles
normalizados [0.0, 1.0] y exporta data/features/scores_anomalias.parquet.
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

import numpy as np
import pandas as pd

from src.models.estimators.anomaly_bundle import AnomalyPipelineBundle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("production_pipeline")

EXPECTED_OUTPUT_COLUMNS = [
    "transaction_id",
    "pca_q",
    "pca_magnitud",
    "pca_t2",
    "score_iforest_perpendicular",
    "score_iforest_diagonal",
    "score_lof_manhattan",
    "score_lof_euclidean",
    "score_ocsvm_rbf_nu_003",
    "score_ocsvm_rbf_nu_005",
    "score_gmm_k3",
    "score_gmm_k5",
    "score_anomalia_consolidado",
    "anomalia_x_monto",
    "extended_iforest_x_monto",
]


def validate_output_dataframe(out_df: pd.DataFrame, expected_rows: int) -> None:
    """Ejecuta aserciones estrictas de calidad e integridad antes de exportar a Parquet."""
    if len(out_df) != expected_rows:
        raise ValueError(f"Cantidad de filas inválida: esperado {expected_rows}, obtenido {len(out_df)}.")

    if list(out_df.columns) != EXPECTED_OUTPUT_COLUMNS:
        raise ValueError(
            f"Esquema de columnas inválido.\nEsperado: {EXPECTED_OUTPUT_COLUMNS}\nObtenido: {list(out_df.columns)}"
        )

    if out_df["transaction_id"].isna().any() or out_df["transaction_id"].duplicated().any():
        raise ValueError("La columna 'transaction_id' contiene valores nulos o duplicados.")

    for col in EXPECTED_OUTPUT_COLUMNS[1:]:
        vals = out_df[col].to_numpy()
        if not np.all(np.isfinite(vals)):
            raise ValueError(f"La columna '{col}' contiene valores NaN o infinitos.")
        if np.any(vals < 0.0):
            raise ValueError(f"La columna '{col}' tiene valores negativos no válidos.")
        # Los percentiles deben acotarse en [0.0, 1.0], mientras que las interacciones anomalia_x_monto son no-negativas
        if col not in ("anomalia_x_monto", "extended_iforest_x_monto") and np.any(vals > 1.0):
            raise ValueError(f"La columna '{col}' tiene valores fuera del rango normalizado [0.0, 1.0].")


def run_production_pipeline(
    input_path: Path,
    model_path: Path,
    output_path: Path,
    seed: int = 42,
) -> pd.DataFrame:
    """Ejecuta el pipeline productivo completo y persiste el archivo Parquet."""
    if not input_path.exists():
        raise FileNotFoundError(f"No se encontró el dataset base en: {input_path.resolve()}")

    logger.info(f"Cargando dataset completo desde: {input_path}")
    df = pd.read_parquet(input_path)
    logger.info(f"Total de registros cargados: {len(df):,}")

    if not model_path.exists():
        logger.info(f"Binario no encontrado en {model_path}. Entrenando en partición 'train'...")
        df_train = df.loc[df["split"] == "train"].copy()
        bundle = AnomalyPipelineBundle(seed=seed).fit(df_train)
        bundle.save(model_path)
    else:
        logger.info(f"Cargando estimadores desde binario: {model_path}")
        bundle = AnomalyPipelineBundle.load(model_path)

    logger.info("Generando features de anomalías normalizadas [0.0, 1.0] para todo el dataset...")
    scores_df = bundle.predict_scores_df(df)

    logger.info("Validando integridad y esquema del DataFrame de salida...")
    validate_output_dataframe(scores_df, expected_rows=len(df))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores_df.to_parquet(output_path, compression="snappy", index=False)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Archivo exportado exitosamente en: {output_path} ({len(scores_df):,} filas, {size_mb:.2f} MB)")
    return scores_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/dataset_base.parquet"),
        help="Ruta al dataset base procesado.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/anomaly_bundle.joblib"),
        help="Ruta al archivo binario con los modelos entrenados.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/features/scores_anomalias.parquet"),
        help="Ruta de salida para el Parquet de anomalías.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla determinista.",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    run_production_pipeline(
        input_path=args.input,
        model_path=args.model_path,
        output_path=args.output,
        seed=args.seed,
    )
    print(f"\nPipeline productivo finalizado con éxito en {time.perf_counter() - started:.2f} s.")


if __name__ == "__main__":
    main()
