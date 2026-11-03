"""
Evaluación numérica de los modelos de detección de anomalías sobre la partición de validación OOT.
Emite el reporte de métricas (PR-AUC, ROC-AUC, Mejor F1-Score, Precision y Recall asociados)
exclusivamente por consola al finalizar la ejecución.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

from src.models.estimators.anomaly_bundle import AnomalyPipelineBundle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluate_models")


@dataclass
class MetricRow:
    score_name: str
    pr_auc: float
    roc_auc: float
    best_f1: float
    precision_at_best_f1: float
    recall_at_best_f1: float
    optimal_threshold: float


def compute_score_metrics(y_true: np.ndarray, scores: np.ndarray, score_name: str) -> MetricRow:
    """
    Calcula PR-AUC, ROC-AUC y el punto de operación de máximo F1-Score
    (junto a su Precision, Recall y umbral óptimo).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    pr_auc = float(auc(recall, precision))
    roc_auc = float(roc_auc_score(y_true, scores))

    # Calcular F1 para cada umbral válido (precision y recall tienen len = len(thresholds) + 1)
    prec_t = precision[:-1]
    rec_t = recall[:-1]
    denom = prec_t + rec_t
    f1_scores = np.where(denom > 0, 2.0 * (prec_t * rec_t) / denom, 0.0)

    best_idx = int(np.argmax(f1_scores))
    return MetricRow(
        score_name=score_name,
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        best_f1=float(f1_scores[best_idx]),
        precision_at_best_f1=float(prec_t[best_idx]),
        recall_at_best_f1=float(rec_t[best_idx]),
        optimal_threshold=float(thresholds[best_idx]),
    )


def print_console_report(
    rows: List[MetricRow],
    n_val: int,
    n_frauds: int,
    prevalence: float,
) -> None:
    """Imprime en consola la tabla comparativa de evaluación de anomalías."""
    sep = "=" * 108
    subsep = "-" * 108
    print("\n" + sep)
    print("                 REPORTE DE EVALUACIÓN DE MODELOS DE ANOMALÍAS (VALIDACIÓN OOT)")
    print(sep)
    print(f"  * Total transacciones evaluadas (val): {n_val:,}")
    print(f"  * Total fraudes reales en val:         {n_frauds:,} (Prevalencia base: {prevalence:.3%})")
    print(subsep)
    header = (
        f"{'Indicador / Modelo':<32} | {'PR-AUC':>8} | {'ROC-AUC':>8} | "
        f"{'Mejor F1':>9} | {'Precision':>10} | {'Recall':>8} | {'Umbral Opt':>10}"
    )
    print(header)
    print(subsep)

    for r in rows:
        is_consolidado = r.score_name == "score_anomalia_consolidado"
        if is_consolidado:
            print(subsep)
        line = (
            f"{r.score_name:<32} | {r.pr_auc:>8.4f} | {r.roc_auc:>8.4f} | "
            f"{r.best_f1:>9.4f} | {r.precision_at_best_f1:>9.2%} | {r.recall_at_best_f1:>7.2%} | {r.optimal_threshold:>10.4f}"
        )
        print(line)

    print(sep + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/dataset_base.parquet"),
        help="Ruta al dataset base.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/anomaly_bundle.joblib"),
        help="Ruta al archivo binario con los modelos entrenados.",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    if not args.input.exists():
        raise FileNotFoundError(f"No se encontró el dataset base en: {args.input.resolve()}")

    if not args.model_path.exists():
        logger.info(f"No se encontró el binario en {args.model_path}. Entrenando estimadores en 'train'...")
        df_train = pd.read_parquet(args.input, filters=[("split", "==", "train")])
        bundle = AnomalyPipelineBundle().fit(df_train)
        bundle.save(args.model_path)
    else:
        logger.info(f"Cargando estimadores serializados desde: {args.model_path}")
        bundle = AnomalyPipelineBundle.load(args.model_path)

    logger.info("Cargando partición 'val' para evaluación OOT...")
    df_val = pd.read_parquet(args.input, filters=[("split", "==", "val")])
    y_val = df_val["fraude"].to_numpy()

    logger.info(f"Infiriendo scores sobre {len(df_val):,} transacciones de validación...")
    scores_df = bundle.predict_scores_df(df_val)

    score_columns = AnomalyPipelineBundle.INDIVIDUAL_SCORE_COLS + ["score_anomalia_consolidado"]
    metric_rows = [
        compute_score_metrics(y_val, scores_df[col].to_numpy(), col)
        for col in score_columns
    ]

    print_console_report(
        rows=metric_rows,
        n_val=len(df_val),
        n_frauds=int(y_val.sum()),
        prevalence=float(y_val.mean()),
    )
    logger.info(f"Evaluación completada en {time.perf_counter() - started:.2f} s.")


if __name__ == "__main__":
    main()
