"""
Experimentos de Gaussian Mixture Model (GMM) para detección de anomalías.

Parámetros evaluados:
- n_components (k): in [3, 5, 8, 12]
- covariance_type: 'full'
- Preprocesamiento: 14 numéricas + Frequency Encoding de g, j, o, p + StandardScaler
- Ajuste estricto en legítimos de train (split == 'train' & fraude == 0)
- Score de anomalía: - score_samples(X) = - ln p(x), normalizado a percentiles [0.0, 1.0]
- Evaluación en validación temporal OOT (22.500 transacciones)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("gmm_experiments")

DEFAULT_NUMERIC = ["score", "monto", "a", "b", "d", "e", "f", "h", "l", "m", "n", "q", "r", "s"]
DEFAULT_CATEGORICAL = ["g", "j", "o", "p"]


def frequency_encode(train: pd.Series, val: pd.Series, normal_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    reference = train.loc[normal_mask]
    counts = reference.value_counts(dropna=True)
    frequencies = counts / len(reference)
    null_frequency = float(reference.isna().mean())

    def transform(values: pd.Series) -> np.ndarray:
        missing = values.isna()
        unknown = ~missing & ~values.isin(counts.index)
        encoded = values.map(frequencies).astype(float)
        encoded.loc[missing] = null_frequency
        encoded.loc[unknown] = 0.0
        return encoded.to_numpy()

    return transform(train), transform(val)


def prepare_features(
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_mask = (df["split"] == "train").to_numpy()
    val_mask = (df["split"] == "val").to_numpy()

    y_train = df.loc[train_mask, "fraude"].to_numpy()
    y_val = df.loc[val_mask, "fraude"].to_numpy()
    normal_train = y_train == 0

    x_num_train = df.loc[train_mask, numeric_cols].to_numpy(dtype=float, na_value=np.nan)
    x_num_val = df.loc[val_mask, numeric_cols].to_numpy(dtype=float, na_value=np.nan)

    imputer = SimpleImputer(strategy="median").fit(x_num_train[normal_train])
    x_num_train = imputer.transform(x_num_train)
    x_num_val = imputer.transform(x_num_val)

    cat_blocks_tr = []
    cat_blocks_val = []
    for col in categorical_cols:
        tr_enc, v_enc = frequency_encode(df.loc[train_mask, col], df.loc[val_mask, col], normal_train)
        cat_blocks_tr.append(tr_enc[:, None])
        cat_blocks_val.append(v_enc[:, None])

    x_train_full = np.column_stack([x_num_train] + cat_blocks_tr)
    x_val_full = np.column_stack([x_num_val] + cat_blocks_val)

    scaler = StandardScaler().fit(x_train_full[normal_train])
    return scaler.transform(x_train_full), scaler.transform(x_val_full), y_train, y_val


def to_percentiles(raw_train: np.ndarray, raw_val: np.ndarray, normal_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    sorted_legit = np.sort(raw_train[normal_train])
    n_legit = len(sorted_legit)
    pct_train = np.searchsorted(sorted_legit, raw_train) / n_legit
    pct_val = np.searchsorted(sorted_legit, raw_val) / n_legit
    return np.clip(pct_train, 0.0, 1.0), np.clip(pct_val, 0.0, 1.0)


@dataclass
class GMMExperimentResult:
    config_name: str
    k: int
    bic: float
    aic: float
    fit_time_sec: float
    score_time_sec: float
    direct_pr_auc: float
    direct_roc_auc: float
    lgb_pr_auc: float
    lgb_avg_prec: float
    lgb_roc_auc: float
    best_f1: float
    precision_at_best_f1: float
    recall_at_best_f1: float
    optimal_threshold: float


def run_single_gmm(
    x_train: np.ndarray,
    x_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    k: int,
    seed: int = 42,
) -> GMMExperimentResult:
    config_name = f"GMM_k_{k}"
    logger.info(f"--- Ejecutando {config_name}: k={k}, covariance_type='full' ---")

    normal_train = y_train == 0
    x_train_legit = x_train[normal_train]

    gmm = GaussianMixture(
        n_components=k,
        covariance_type="full",
        max_iter=100,
        random_state=seed,
    )

    t0 = time.perf_counter()
    gmm.fit(x_train_legit)
    fit_time = time.perf_counter() - t0

    bic = float(gmm.bic(x_train_legit))
    aic = float(gmm.aic(x_train_legit))

    t0 = time.perf_counter()
    # Score de anomalía crudo = - ln p(x). Mayor valor = menor densidad = más anómalo
    raw_train = -gmm.score_samples(x_train)
    raw_val = -gmm.score_samples(x_val)
    score_time = time.perf_counter() - t0

    # Normalización a percentiles empíricos [0.0, 1.0]
    pct_train, pct_val = to_percentiles(raw_train, raw_val, normal_train)

    # 1. Métricas directas no supervisadas en val
    prec, rec, _ = precision_recall_curve(y_val, pct_val)
    direct_pr_auc = float(auc(rec, prec))
    direct_roc_auc = float(roc_auc_score(y_val, pct_val))

    # 2. Evaluación con LightGBM univariado
    X_tr_lgb = pd.DataFrame({"percentil_gmm": pct_train})
    X_val_lgb = pd.DataFrame({"percentil_gmm": pct_val})

    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=seed,
        verbose=-1,
        n_jobs=-1,
    )
    lgb_model.fit(X_tr_lgb, y_train)
    probs_val = lgb_model.predict_proba(X_val_lgb)[:, 1]

    p_lgb, r_lgb, thresholds = precision_recall_curve(y_val, probs_val)
    lgb_pr_auc = float(auc(r_lgb, p_lgb))
    lgb_avg_prec = float(average_precision_score(y_val, probs_val))
    lgb_roc_auc = float(roc_auc_score(y_val, probs_val))

    # Mejor F1 sobre probabilidades calibradas
    prec_t = p_lgb[:-1]
    rec_t = r_lgb[:-1]
    denom = prec_t + rec_t
    f1_scores = np.divide(2.0 * (prec_t * rec_t), denom, out=np.zeros_like(denom), where=denom > 0)
    best_idx = int(np.argmax(f1_scores))

    res = GMMExperimentResult(
        config_name=config_name,
        k=k,
        bic=bic,
        aic=aic,
        fit_time_sec=fit_time,
        score_time_sec=score_time,
        direct_pr_auc=direct_pr_auc,
        direct_roc_auc=direct_roc_auc,
        lgb_pr_auc=lgb_pr_auc,
        lgb_avg_prec=lgb_avg_prec,
        lgb_roc_auc=lgb_roc_auc,
        best_f1=float(f1_scores[best_idx]),
        precision_at_best_f1=float(prec_t[best_idx]),
        recall_at_best_f1=float(rec_t[best_idx]),
        optimal_threshold=float(thresholds[best_idx]),
    )

    logger.info(
        f"[{config_name}] LGBM PR-AUC: {lgb_pr_auc:.4f} | ROC-AUC: {lgb_roc_auc:.4f} | "
        f"Mejor F1: {res.best_f1:.4f} (Fit: {fit_time:.2f}s, Score: {score_time:.2f}s)"
    )
    return res


def format_report(results: List[GMMExperimentResult], val_prev: float, seed: int) -> str:
    lines = [
        "# Experimentos Gaussian Mixture Model (GMM)",
        "",
        "> **Estándar:** GMM con `covariance_type='full'`, ajustado estrictamente sobre transacciones legítimas de `train`. Score de anomalía definido como $-\\ln p(x)$ y normalizado a percentiles empíricos $[0.0, 1.0]$. Evaluación univariada con LightGBM sobre validación OOT (22.500 transacciones).",
        "",
        f"- **Prevalencia base de fraude en validación:** `{val_prev * 100:.3f}%`",
        f"- **Semilla:** `{seed}`",
        "",
        "## Resumen Comparativo de Métricas (Validación OOT)",
        "",
        "| Modelo (k) | Fit Time (s) | Direct PR-AUC | Direct ROC-AUC | LGBM PR-AUC | LGBM ROC-AUC | Mejor F1 | Precision | Recall | Umbral Opt | BIC |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for r in results:
        lines.append(
            f"| `{r.config_name}` | {r.fit_time_sec:.2f} | {r.direct_pr_auc:.4f} | {r.direct_roc_auc:.4f} | "
            f"**{r.lgb_pr_auc:.4f}** | **{r.lgb_roc_auc:.4f}** | **{r.best_f1:.4f}** | "
            f"{r.precision_at_best_f1 * 100:.2f}% | {r.recall_at_best_f1 * 100:.2f}% | {r.optimal_threshold:.4f} | {r.bic:,.0f} |"
        )

    lines.extend([
        "",
        "## Observaciones Técnicas",
        "1. **Definición de Anomaly Score:** Las observaciones anómalas tienen menor densidad conjunta $p(x) = \\sum \\pi_k \\mathcal{N}_k(x) \\to 0^+$, lo cual genera un log-likelihood muy negativo $\\ln p(x) \\to -\\infty$. Por ende, el score $-\\ln p(x)$ es alto para anomalías.",
        "2. **Normalización por Percentiles:** El ordenamiento empírico en $[0.0, 1.0]$ sobre el conjunto legítimo permite interpretar el score directamente como el cuantil de rareza estadística.",
        "3. **Capacidad vs Generalización:** A mayor $k$, el GMM modela modas locales más específicas; se evalúa si $k$ grande retiene poder discriminativo sobre fraudes o sobreajusta la densidad del espacio legítimo.",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Experimentos de GMM para detección de anomalías.")
    parser.add_argument("--data-path", default="data/processed/dataset_base.parquet", help="Ruta al dataset base")
    parser.add_argument("--output-report", default="data/experiments/gmm_report.md", help="Ruta de salida del reporte")
    parser.add_argument("--seed", type=int, default=42, help="Semilla de reproducibilidad")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.error(f"No existe el archivo de datos: {data_path}")
        sys.exit(1)

    logger.info(f"Cargando dataset base desde: {data_path}")
    df = pd.read_parquet(data_path)

    logger.info("Preparando features con imputación y frequency encoding...")
    x_train, x_val, y_train, y_val = prepare_features(df, DEFAULT_NUMERIC, DEFAULT_CATEGORICAL)
    val_prev = float(np.mean(y_val))
    logger.info(f"Features listos: train shape={x_train.shape}, val shape={x_val.shape}, prevalencia val={val_prev:.4f}")

    k_values = [3, 5, 8, 12]
    results: List[GMMExperimentResult] = []

    for k in k_values:
        res = run_single_gmm(
            x_train=x_train,
            x_val=x_val,
            y_train=y_train,
            y_val=y_val,
            k=k,
            seed=args.seed,
        )
        results.append(res)

    report_content = format_report(results, val_prev, args.seed)
    out_path = Path(args.output_report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_content, encoding="utf-8")
    logger.info(f"Reporte de experimentos exportado en: {out_path}")

    # Imprimir resumen final en consola
    print("\n" + "=" * 108)
    print("                 REPORTE DE EXPERIMENTOS DE GAUSSIAN MIXTURE MODEL (GMM)")
    print("=" * 108)
    print(f"  * Total transacciones evaluadas (val): {len(y_val):,}")
    print(f"  * Fraudes reales en val:               {int(np.sum(y_val)):,} (Prevalencia: {val_prev * 100:.3f}%)")
    print("-" * 108)
    print("Modelo (k)   | Fit(s) | Score(s) | Direct PR | Direct ROC | LGBM PR-AUC | ROC-AUC | Mejor F1 |  Precision |   Recall")
    print("-" * 108)
    for r in results:
        print(
            f"{r.config_name:<12} | {r.fit_time_sec:>6.2f} | {r.score_time_sec:>8.2f} | "
            f"{r.direct_pr_auc:>9.4f} | {r.direct_roc_auc:>10.4f} | {r.lgb_pr_auc:>11.4f} | "
            f"{r.lgb_roc_auc:>7.4f} | {r.best_f1:>8.4f} | {r.precision_at_best_f1 * 100:>9.2f}% | "
            f"{r.recall_at_best_f1 * 100:>7.2f}%"
        )
    print("=" * 108 + "\n")


if __name__ == "__main__":
    main()
