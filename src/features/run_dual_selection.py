"""
Script de Selección Dual de Variables (Forward Stepwise + Backward RFE).

Ejecuta ambos métodos bajo la configuración de hiperparámetros especificada:
- n_estimators: 120
- learning_rate: 0.06
- num_leaves: 31
- max_depth: 6
- subsample: 0.8
- colsample_bytree: 0.8
- n_jobs: -1
- random_state: 42

Guarda datasets curados, reportes JSON y el análisis de intersección.
"""

from __future__ import annotations

import json
import logging
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
logger = logging.getLogger("dual_selection")

PARAMS_BUSQUEDA = {
    "n_estimators": 120,
    "learning_rate": 0.06,
    "num_leaves": 31,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
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

    clf = lgb.LGBMClassifier(**PARAMS_BUSQUEDA)
    clf.fit(X_train, y_train)

    probs = clf.predict_proba(X_val)[:, 1]
    precision, recall, _ = precision_recall_curve(y_val, probs)
    pr_auc = float(auc(recall, precision))

    importances = dict(zip(feature_subset, clf.feature_importances_))
    return pr_auc, importances


def run_forward_selection(
    full_df: pd.DataFrame,
    base_cols: List[str],
    cand_cols: List[str],
    y_train: np.ndarray,
    y_val: np.ndarray,
    train_mask: pd.Series,
    val_mask: pd.Series,
    min_delta: float = 0.0002,
) -> Tuple[List[str], Dict]:
    logger.info("==================================================")
    logger.info("   INICIANDO MÉTODO 1: FORWARD STEPWISE SELECTION")
    logger.info("==================================================")
    t0 = time.time()

    base_pr, _ = fit_eval(full_df, base_cols, y_train, y_val, train_mask, val_mask)
    logger.info(f"Paso 0 (Baseline Crudo): PR-AUC = {base_pr:.4f}")

    selected: List[str] = []
    remaining = list(cand_cols)
    current_pr = base_pr
    history = []
    step = 1

    while remaining:
        best_cand = None
        best_score = current_pr

        for c in remaining:
            score, _ = fit_eval(
                full_df, base_cols + selected + [c],
                y_train, y_val, train_mask, val_mask
            )
            if score > best_score:
                best_score = score
                best_cand = c

        delta = best_score - current_pr
        if best_cand is not None and delta >= min_delta:
            selected.append(best_cand)
            remaining.remove(best_cand)
            current_pr = best_score
            logger.info(
                f"Paso {step:2d}: + {best_cand:25s} -> PR-AUC: {best_score:.4f} (Delta: {delta:+.5f})"
            )
            history.append({
                "step": step,
                "added_feature": best_cand,
                "pr_auc": best_score,
                "delta": delta,
            })
            step += 1
        else:
            logger.info(f"Parada Forward en paso {step}. Criterio de mejora marginal alcanzado.")
            break

    final_pr, final_imp = fit_eval(
        full_df, base_cols + selected, y_train, y_val, train_mask, val_mask
    )
    elapsed = time.time() - t0
    logger.info(
        f"Forward finalizado en {elapsed:.1f}s. PR-AUC: {final_pr:.4f} "
        f"({len(selected)} variables seleccionadas de {len(cand_cols)})"
    )

    report = {
        "method": "Forward Stepwise Selection",
        "params": PARAMS_BUSQUEDA,
        "baseline_pr_auc": base_pr,
        "final_pr_auc": final_pr,
        "delta_pr_auc": final_pr - base_pr,
        "selected_count": len(selected),
        "selected_features": selected,
        "feature_gains": {c: final_imp.get(c, 0.0) for c in selected},
        "history": history,
        "elapsed_seconds": round(elapsed, 2),
    }
    return selected, report


def run_backward_selection(
    full_df: pd.DataFrame,
    base_cols: List[str],
    cand_cols: List[str],
    y_train: np.ndarray,
    y_val: np.ndarray,
    train_mask: pd.Series,
    val_mask: pd.Series,
    tolerance: float = 0.0002,
    min_features: int = 12,
) -> Tuple[List[str], Dict]:
    logger.info("==================================================")
    logger.info("   INICIANDO MÉTODO 2: BACKWARD RECURSIVE ELIMINATION")
    logger.info("==================================================")
    t0 = time.time()

    base_pr, _ = fit_eval(full_df, base_cols, y_train, y_val, train_mask, val_mask)
    active_cands = list(cand_cols)
    current_pr, current_imp = fit_eval(
        full_df, base_cols + active_cands, y_train, y_val, train_mask, val_mask
    )
    logger.info(f"Paso 0: Pool Completo ({len(active_cands)} candidatas) -> PR-AUC: {current_pr:.4f}")

    history = []
    step = 1

    while len(active_cands) > min_features:
        # Ordenar candidatos activos por Gain
        cands_sorted = sorted(active_cands, key=lambda c: current_imp.get(c, 0.0))
        worst_cand = cands_sorted[0]
        worst_gain = current_imp.get(worst_cand, 0.0)

        # Probar remover la de menor Gain
        test_cands = [c for c in active_cands if c != worst_cand]
        new_pr, new_imp = fit_eval(
            full_df, base_cols + test_cands, y_train, y_val, train_mask, val_mask
        )
        delta = new_pr - current_pr

        if delta >= -tolerance:
            active_cands = test_cands
            current_imp = new_imp
            current_pr = max(current_pr, new_pr)
            logger.info(
                f"Paso {step:2d}: - Removida {worst_cand:22s} (Gain: {worst_gain:7.1f}) "
                f"-> PR-AUC: {new_pr:.4f} (Delta: {delta:+.5f}) [Restan: {len(active_cands)}]"
            )
            history.append({
                "step": step,
                "removed_feature": worst_cand,
                "gain_before": worst_gain,
                "pr_auc": new_pr,
                "delta": delta,
            })
            step += 1
        else:
            # Probar si la segunda peor puede removerse
            if len(cands_sorted) > 1:
                second_worst = cands_sorted[1]
                second_gain = current_imp.get(second_worst, 0.0)
                test_cands_2 = [c for c in active_cands if c != second_worst]
                new_pr_2, new_imp_2 = fit_eval(
                    full_df, base_cols + test_cands_2, y_train, y_val, train_mask, val_mask
                )
                delta_2 = new_pr_2 - current_pr

                if delta_2 >= -tolerance:
                    active_cands = test_cands_2
                    current_imp = new_imp_2
                    current_pr = max(current_pr, new_pr_2)
                    logger.info(
                        f"Paso {step:2d} (alt): - Removida {second_worst:22s} (Gain: {second_gain:7.1f}) "
                        f"-> PR-AUC: {new_pr_2:.4f} (Delta: {delta_2:+.5f}) [Restan: {len(active_cands)}]"
                    )
                    history.append({
                        "step": step,
                        "removed_feature": second_worst,
                        "gain_before": second_gain,
                        "pr_auc": new_pr_2,
                        "delta": delta_2,
                    })
                    step += 1
                    continue

            logger.info(
                f"Parada Backward en paso {step}: remover '{worst_cand}' degrada en {delta:+.5f}."
            )
            break

    final_pr, final_imp = fit_eval(
        full_df, base_cols + active_cands, y_train, y_val, train_mask, val_mask
    )
    elapsed = time.time() - t0
    logger.info(
        f"Backward finalizado en {elapsed:.1f}s. PR-AUC: {final_pr:.4f} "
        f"({len(active_cands)} variables seleccionadas de {len(cand_cols)})"
    )

    report = {
        "method": "Backward Recursive Elimination",
        "params": PARAMS_BUSQUEDA,
        "baseline_pr_auc": base_pr,
        "final_pr_auc": final_pr,
        "delta_pr_auc": final_pr - base_pr,
        "selected_count": len(active_cands),
        "selected_features": active_cands,
        "feature_gains": {c: final_imp.get(c, 0.0) for c in active_cands},
        "history": history,
        "elapsed_seconds": round(elapsed, 2),
    }
    return active_cands, report


def main() -> None:
    logger.info("Iniciando pipeline de Selección Dual...")
    full_df, df_feat, base_cols, cand_cols, y_train, y_val, train_mask, val_mask = load_data()

    # 1. Ejecutar Forward Selection
    forward_selected, forward_report = run_forward_selection(
        full_df, base_cols, cand_cols, y_train, y_val, train_mask, val_mask
    )

    # 2. Ejecutar Backward Selection
    backward_selected, backward_report = run_backward_selection(
        full_df, base_cols, cand_cols, y_train, y_val, train_mask, val_mask
    )

    # 3. Análisis de Intersección y Comparación
    intersection = sorted(list(set(forward_selected).intersection(set(backward_selected))))
    logger.info("==================================================")
    logger.info("               SÍNTESIS COMPARATIVA")
    logger.info("==================================================")
    logger.info(f"Forward PR-AUC:  {forward_report['final_pr_auc']:.4f} ({len(forward_selected)} features)")
    logger.info(f"Backward PR-AUC: {backward_report['final_pr_auc']:.4f} ({len(backward_selected)} features)")
    logger.info(f"Núcleo Común (Intersección): {len(intersection)} features: {intersection}")

    # 4. Exportar Datasets Parquet Curados
    out_dir = Path("data/features")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Exportar Forward
    df_fwd = df_feat[["transaction_id"] + forward_selected]
    df_fwd.to_parquet(out_dir / "features_seleccionadas_forward.parquet", engine="pyarrow", index=False)
    logger.info(f"Guardado: {out_dir / 'features_seleccionadas_forward.parquet'}")

    # Exportar Backward
    df_bwd = df_feat[["transaction_id"] + backward_selected]
    df_bwd.to_parquet(out_dir / "features_seleccionadas_backward.parquet", engine="pyarrow", index=False)
    logger.info(f"Guardado: {out_dir / 'features_seleccionadas_backward.parquet'}")

    # Exportar Reportes JSON
    with open(out_dir / "selected_features_forward_report.json", "w", encoding="utf-8") as f:
        json.dump(forward_report, f, indent=2, ensure_ascii=False)

    with open(out_dir / "selected_features_backward_report.json", "w", encoding="utf-8") as f:
        json.dump(backward_report, f, indent=2, ensure_ascii=False)

    dual_summary = {
        "params_busqueda": PARAMS_BUSQUEDA,
        "forward": {
            "pr_auc": forward_report["final_pr_auc"],
            "delta_pr_auc": forward_report["delta_pr_auc"],
            "count": len(forward_selected),
            "features": forward_selected,
            "elapsed_seconds": forward_report["elapsed_seconds"],
        },
        "backward": {
            "pr_auc": backward_report["final_pr_auc"],
            "delta_pr_auc": backward_report["delta_pr_auc"],
            "count": len(backward_selected),
            "features": backward_selected,
            "elapsed_seconds": backward_report["elapsed_seconds"],
        },
        "intersection": {
            "count": len(intersection),
            "features": intersection,
        },
    }
    with open(out_dir / "dual_selection_report.json", "w", encoding="utf-8") as f:
        json.dump(dual_summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Reporte comparativo guardado: {out_dir / 'dual_selection_report.json'}")


if __name__ == "__main__":
    main()
