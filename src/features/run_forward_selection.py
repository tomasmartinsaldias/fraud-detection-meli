"""
Forward Stepwise Selection Definitivo con LightGBM.

Hiperparámetros:
- n_estimators: 120
- learning_rate: 0.06
- num_leaves: 31
- max_depth: 6
- subsample: 0.8
- colsample_bytree: 1.0 (Determinístico)
- min_delta (epsilon): 0.0003
- patience: 2 rondas
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("forward_selection")

PARAMS_FORWARD = {
    "n_estimators": 120,
    "learning_rate": 0.06,
    "num_leaves": 31,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 1.0,  # Determinístico
    "n_jobs": -1,
    "random_state": 42,
    "importance_type": "gain",
    "verbose": -1,
}

BASE_NUMERIC_COLS = ["score", "monto", "b", "c", "d", "e", "f", "h", "l", "m", "q"]
BASE_CATEGORICAL_COLS = ["a", "g", "n", "o", "p", "is_na_bc"]


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str], np.ndarray, np.ndarray, pd.Series, pd.Series]:
    df_base = pd.read_parquet("data/processed/dataset_base.parquet")
    df_feat = pd.read_parquet("data/features/features_transaccionales.parquet")

    full_df = df_base.merge(df_feat, on="transaction_id")
    for cat in BASE_CATEGORICAL_COLS:
        if cat in full_df.columns:
            full_df[cat] = full_df[cat].astype("category")

    train_mask = full_df["split"] == "train"
    val_mask = full_df["split"] == "val"

    y_train = full_df.loc[train_mask, "fraude"].values
    y_val = full_df.loc[val_mask, "fraude"].values

    base_cols = [c for c in (BASE_NUMERIC_COLS + BASE_CATEGORICAL_COLS) if c in full_df.columns]
    cand_cols = [c for c in df_feat.columns if c != "transaction_id"]

    return full_df, df_feat, base_cols, cand_cols, y_train, y_val, train_mask, val_mask


def fit_eval(
    full_df: pd.DataFrame,
    feature_subset: List[str],
    y_train: np.ndarray,
    y_val: np.ndarray,
    train_mask: pd.Series,
    val_mask: pd.Series,
) -> Tuple[float, Dict[str, float]]:
    X_train = full_df.loc[train_mask, feature_subset]
    X_val = full_df.loc[val_mask, feature_subset]

    clf = lgb.LGBMClassifier(**PARAMS_FORWARD)
    clf.fit(X_train, y_train)

    probs = clf.predict_proba(X_val)[:, 1]
    precision, recall, _ = precision_recall_curve(y_val, probs)
    pr_auc = float(auc(recall, precision))

    importances = dict(zip(feature_subset, clf.feature_importances_))
    return pr_auc, importances


def run_forward():
    full_df, df_feat, base_cols, cand_cols, y_train, y_val, train_mask, val_mask = load_data()

    logger.info("================================================================")
    logger.info(" INICIANDO FORWARD STEPWISE SELECTION DEFINITIVO")
    logger.info(" colsample_bytree = 1.0 | min_delta = 0.0003 | patience = 2")
    logger.info("================================================================")

    t0 = time.time()
    base_pr, _ = fit_eval(full_df, base_cols, y_train, y_val, train_mask, val_mask)
    logger.info(f"Paso  0 (Baseline Crudo 17 cols): PR-AUC = {base_pr:.4f}")

    current_best_pr = base_pr
    best_selected: List[str] = []
    
    selected: List[str] = []
    remaining = list(cand_cols)
    patience_counter = 0
    patience_limit = 2
    min_delta = 0.0003
    history = []
    step = 1

    while remaining and patience_counter < patience_limit:
        best_cand = None
        best_cand_score = -1.0

        for c in remaining:
            score, _ = fit_eval(
                full_df, base_cols + selected + [c],
                y_train, y_val, train_mask, val_mask
            )
            if score > best_cand_score:
                best_cand_score = score
                best_cand = c

        delta_vs_best = best_cand_score - current_best_pr
        
        selected.append(best_cand)
        remaining.remove(best_cand)

        if delta_vs_best >= min_delta:
            current_best_pr = best_cand_score
            best_selected = list(selected)
            patience_counter = 0
            logger.info(
                f"Paso {step:2d}: + {best_cand:25s} -> PR-AUC: {best_cand_score:.4f} "
                f"(Delta vs best: {delta_vs_best:+.5f}) [MEJORA ACEPTADA]"
            )
        else:
            patience_counter += 1
            logger.info(
                f"Paso {step:2d}: + {best_cand:25s} -> PR-AUC: {best_cand_score:.4f} "
                f"(Delta vs best: {delta_vs_best:+.5f}) [SIN MEJORA - Paciencia {patience_counter}/{patience_limit}]"
            )

        history.append({
            "step": step,
            "feature": best_cand,
            "score": best_cand_score,
            "delta_vs_best": delta_vs_best,
            "patience": patience_counter,
            "accepted": (delta_vs_best >= min_delta),
        })
        step += 1

    logger.info("----------------------------------------------------------------")
    logger.info(f"Búsqueda finalizada tras {step-1} pasos.")
    logger.info(f"Mejor subconjunto retiene {len(best_selected)} variables: {best_selected}")

    final_pr, final_imp = fit_eval(
        full_df, base_cols + best_selected, y_train, y_val, train_mask, val_mask
    )
    elapsed = time.time() - t0
    logger.info(
        f"Forward completado en {elapsed:.1f}s. PR-AUC Final: {final_pr:.4f} "
        f"(Delta total: {final_pr - base_pr:+.5f})"
    )

    # Exportar dataset de features seleccionadas forward
    out_cols = ["transaction_id"] + best_selected
    df_out = df_feat[out_cols].copy()
    out_path = Path("data/features/features_seleccionadas_forward.parquet")
    tmp_path = out_path.with_suffix(".parquet.tmp")
    df_out.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, out_path)
    logger.info(f"Guardado dataset forward: {out_path} ({df_out.shape})")

    report = {
        "method": "Forward Stepwise Selection Definitivo",
        "params": PARAMS_FORWARD,
        "min_delta": min_delta,
        "patience": patience_limit,
        "baseline_pr_auc": base_pr,
        "final_pr_auc": final_pr,
        "delta_pr_auc": final_pr - base_pr,
        "selected_count": len(best_selected),
        "selected_features": best_selected,
        "feature_gains": {c: final_imp.get(c, 0.0) for c in best_selected},
        "history": history,
        "elapsed_seconds": round(elapsed, 2),
    }

    report_path = Path("data/features/selected_features_forward_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Reporte guardado en {report_path}")

    # Imprimir tabla final
    print("\n" + "=" * 70)
    print("RESUMEN FORWARD STEPWISE SELECTION DEFINITIVO")
    print("=" * 70)
    print(f"Baseline PR-AUC (17 crudas): {base_pr:.4f}")
    print(f"Final PR-AUC ({len(best_selected)} vars):      {final_pr:.4f} (+{final_pr - base_pr:.4f})")
    print(f"Tiempo total:                {elapsed:.1f} segundos")
    print("-" * 70)
    print(f"{'Orden':<6} | {'Variable':<26} | {'Gain LightGBM':>15}")
    print("-" * 70)
    for idx, f in enumerate(best_selected, 1):
        print(f"{idx:<6} | {f:<26} | {final_imp.get(f, 0.0):>15.2f}")
    print("=" * 70)


if __name__ == "__main__":
    run_forward()
