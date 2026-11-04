"""
Forward Stepwise Selection Global End-to-End (Competencia abierta 65 variables).

Ejecuta selección hacia adelante poniendo a competir las 65 variables (17 base + 48 ingenierizadas)
arrancando desde la variable continua de máxima señal ('score'), con:
- colsample_bytree: 1.0 (Determinístico)
- min_delta: 0.0003
- patience: 2 rondas
- Poda estructural en Paso 12 (descartando ruido de cola para mitigar Validation Overfitting)

Genera:
- data/features/features_seleccionadas_end_to_end.parquet (12 variables blindadas)
- data/features/selected_features_end_to_end_report.json
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
logger = logging.getLogger("forward_global")

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
BASE_COLS = BASE_NUMERIC_COLS + BASE_CATEGORICAL_COLS


def load_data() -> Tuple[pd.DataFrame, List[str], List[str], List[str], np.ndarray, np.ndarray, pd.Series, pd.Series]:
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

    base_cols = [c for c in BASE_COLS if c in full_df.columns]
    eng_cols = [c for c in df_feat.columns if c != "transaction_id"]
    all_cand_cols = base_cols + eng_cols

    return full_df, base_cols, eng_cols, all_cand_cols, y_train, y_val, train_mask, val_mask


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


def run_global_forward(max_steps: int = 12):
    full_df, base_cols, eng_cols, all_cand_cols, y_train, y_val, train_mask, val_mask = load_data()

    logger.info("================================================================")
    logger.info(" INICIANDO FORWARD STEPWISE GLOBAL BLINDADO (65 VARIABLES)")
    logger.info(f" Pool: {len(base_cols)} base + {len(eng_cols)} ingenierizadas | Max Steps: {max_steps}")
    logger.info(" colsample_bytree = 1.0 | min_delta = 0.0003 | patience = 2")
    logger.info("================================================================")

    t0 = time.time()
    
    # Baseline 17 crudas juntas como referencia
    base_all_pr, _ = fit_eval(full_df, base_cols, y_train, y_val, train_mask, val_mask)
    logger.info(f"Referencia (17 crudas base juntas): PR-AUC = {base_all_pr:.4f}")

    # Semilla en la variable continua univariada más potente
    seed_col = "score"
    selected = [seed_col]
    remaining = [c for c in all_cand_cols if c != seed_col]

    seed_pr, _ = fit_eval(full_df, selected, y_train, y_val, train_mask, val_mask)
    logger.info(f"Paso  1: + {seed_col:24s} [BASE CRUDA   ] -> PR-AUC: {seed_pr:.4f} [SEMILLA INICIAL]")

    current_best_pr = seed_pr
    best_selected = list(selected)
    patience_counter = 0
    patience_limit = 2
    min_delta = 0.0003
    history = [{
        "step": 1,
        "feature": seed_col,
        "type": "BASE CRUDA",
        "score": seed_pr,
        "delta_vs_best": 0.0,
        "patience": 0,
        "accepted": True,
    }]
    step = 2

    while remaining and patience_counter < patience_limit and step <= max_steps:
        best_cand = None
        best_cand_score = -1.0

        for c in remaining:
            score, _ = fit_eval(
                full_df, selected + [c],
                y_train, y_val, train_mask, val_mask
            )
            if score > best_cand_score:
                best_cand_score = score
                best_cand = c

        delta_vs_best = best_cand_score - current_best_pr
        cand_type = "BASE CRUDA" if best_cand in base_cols else "INGENIERIZADA"
        
        selected.append(best_cand)
        remaining.remove(best_cand)

        if delta_vs_best >= min_delta:
            current_best_pr = best_cand_score
            best_selected = list(selected)
            patience_counter = 0
            logger.info(
                f"Paso {step:2d}: + {best_cand:24s} [{cand_type:13s}] -> PR-AUC: {best_cand_score:.4f} "
                f"(Delta vs best: {delta_vs_best:+.5f}) [MEJORA ACEPTADA]"
            )
        else:
            patience_counter += 1
            logger.info(
                f"Paso {step:2d}: + {best_cand:24s} [{cand_type:13s}] -> PR-AUC: {best_cand_score:.4f} "
                f"(Delta vs best: {delta_vs_best:+.5f}) [SIN MEJORA - Paciencia {patience_counter}/{patience_limit}]"
            )

        history.append({
            "step": step,
            "feature": best_cand,
            "type": cand_type,
            "score": best_cand_score,
            "delta_vs_best": delta_vs_best,
            "patience": patience_counter,
            "accepted": (delta_vs_best >= min_delta),
        })
        step += 1

    logger.info("----------------------------------------------------------------")
    logger.info(f"Búsqueda finalizada. Subconjunto blindado retiene {len(best_selected)} variables.")

    final_pr, final_imp = fit_eval(
        full_df, best_selected, y_train, y_val, train_mask, val_mask
    )
    elapsed = time.time() - t0

    surviving_base = [c for c in best_selected if c in base_cols]
    discarded_base = [c for c in base_cols if c not in best_selected]
    selected_eng = [c for c in best_selected if c in eng_cols]

    logger.info(f"Completado en {elapsed:.1f}s. PR-AUC Final: {final_pr:.4f} (+{final_pr - base_all_pr:.4f} vs 17 base)")
    logger.info(f"Base Retenidas ({len(surviving_base)}): {surviving_base}")
    logger.info(f"Ingenierizadas Seleccionadas ({len(selected_eng)}): {selected_eng}")

    # Exportar dataset definitivo end-to-end
    out_cols = ["transaction_id"] + best_selected
    df_out = full_df[out_cols].copy()
    out_path = Path("data/features/features_seleccionadas_end_to_end.parquet")
    tmp_path = out_path.with_suffix(".parquet.tmp")
    df_out.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, out_path)
    logger.info(f"Guardado dataset blindado: {out_path} ({df_out.shape})")

    report = {
        "method": "Forward Stepwise Global End-to-End Blindado",
        "params": PARAMS_FORWARD,
        "max_steps": max_steps,
        "reference_17_base_pr_auc": base_all_pr,
        "final_pr_auc_val": final_pr,
        "delta_vs_reference_17_base": final_pr - base_all_pr,
        "total_selected_count": len(best_selected),
        "surviving_base_count": len(surviving_base),
        "discarded_base_count": len(discarded_base),
        "selected_eng_count": len(selected_eng),
        "surviving_base_features": surviving_base,
        "discarded_base_features": discarded_base,
        "selected_eng_features": selected_eng,
        "all_selected_features_order": best_selected,
        "feature_gains": {c: final_imp.get(c, 0.0) for c in best_selected},
        "history": history,
        "elapsed_seconds": round(elapsed, 2),
    }

    report_path = Path("data/features/selected_features_end_to_end_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Reporte guardado en {report_path}")

    # Tabla resumen en consola
    print("\n" + "=" * 80)
    print("RESUMEN FORWARD STEPWISE GLOBAL BLINDADO (END-TO-END)")
    print("=" * 80)
    print(f"Referencia (17 base crudas): {base_all_pr:.4f}")
    print(f"PR-AUC Final ({len(best_selected)} vars blindadas):       {final_pr:.4f} ({final_pr - base_all_pr:+.4f})")
    print(f"Base Retenidas: {len(surviving_base)}/17 | Ingenierizadas Seleccionadas: {len(selected_eng)}")
    print(f"Tiempo total: {elapsed:.1f} segundos")
    print("-" * 80)
    print(f"{'Orden':<6} | {'Variable':<24} | {'Tipo':<15} | {'Gain LightGBM':>15}")
    print("-" * 80)
    for idx, f in enumerate(best_selected, 1):
        f_type = "Base Cruda" if f in base_cols else "Ingenierizada"
        print(f"{idx:<6} | {f:<24} | {f_type:<15} | {final_imp.get(f, 0.0):>15.2f}")
    print("=" * 80)


if __name__ == "__main__":
    run_global_forward(max_steps=12)
