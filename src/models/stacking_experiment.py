"""Stacking Meta-Model Experiment:
Level 1: lgb_full_cmaes (21 features: 11 transactional + 10 anomaly scores)
Level 2: Meta-dataset [p_lgb, pct_heuristico_fraude] -> Logistic Regression
Evaluated strictly on validation temporal splits (val_search, val_calibration, val_policy).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.models.supervised import (
    BASE_CATEGORICAL, BASE_NUMERIC, FrameEncoder, calibrate, check_ids,
    cost_curve, logit, metrics, policy_metrics, pr_auc,
    validate_development, validation_blocks,
)
from src.models.train import load_data, lgb_model


def run_stacking_experiment(
    base_path="data/processed/dataset_base.parquet",
    feat_path="data/features/features_seleccionadas_oficiales_11.parquet",
    anom_path="data/features/scores_anomalias.parquet",
    rules_path="data/features/features_reglas_asociacion.parquet",
    output_dir="models/experiments/exp_stacking_meta",
    seed=42,
    jobs=4,
):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    out_path = ROOT / output_dir
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("INICIANDO EXPERIMENTO DE STACKING DETERMINÍSTICO (Semilla 42)")
    print("=" * 70)

    # 1. Cargar datos
    print("\n[1/6] Cargando datos y splits de desarrollo...")
    exclude_anom = ["score_anomalia_consolidado", "anomalia_x_monto", "extended_iforest_x_monto"]
    rule_cols = ["pct_heuristico_fraude"]

    df, feat_cols, anom_cols, rules_cols = load_data(
        ROOT / base_path,
        ROOT / feat_path,
        ROOT / anom_path,
        ["train", "val"],
        exclude_anomaly_cols=exclude_anom,
        rules_path=ROOT / rules_path,
        rule_cols=rule_cols,
    )
    validate_development(df)

    train = df.loc[df.split == "train"].copy()
    tune, cal, policy = validation_blocks(df.loc[df.split == "val"])

    # Features de Nivel 1: 21 variables oficiales
    l1_features = list(dict.fromkeys(feat_cols + anom_cols))
    print(f"Features Nivel 1 ({len(l1_features)}): {l1_features}")
    print(f"Features Nivel 2: ['p_lgb', 'pct_heuristico_fraude']")

    # 2. Hiperparámetros óptimos de control lgb_full_cmaes
    # Obtenidos del benchmark de 20 trials CMA-ES
    best_params = {
        "learning_rate": 0.03486992722440761,
        "num_leaves": 28,
        "min_child_samples": 100,
        "colsample_bytree": 0.8140488199767426,
        "subsample": 0.7573516799506432,
        "reg_alpha": 0.013018028720806228,
        "reg_lambda": 0.5270037815983998,
        "n_estimators": 261,
    }
    print(f"\n[2/6] Parámetros óptimos Nivel 1 (lgb_full_cmaes):\n{json.dumps(best_params, indent=2)}")

    encoder = FrameEncoder().fit(train[l1_features])
    x_train = encoder.transform(train[l1_features])
    x_tune = encoder.transform(tune[l1_features])
    x_cal = encoder.transform(cal[l1_features])
    x_policy = encoder.transform(policy[l1_features])

    # 3. Entrenar Modelo Base Nivel 1 en todo train
    print("\n[3/6] Ajustando Modelo Base de Nivel 1 en train completo...")
    t0 = time.perf_counter()
    lgb_full = lgb_model(best_params, seed, jobs).fit(x_train, train.fraude)
    fit_time = time.perf_counter() - t0
    print(f"Modelo Nivel 1 entrenado en {fit_time:.2f}s.")

    # Inferencia de Nivel 1 en validación
    p_lgb_tune = lgb_full.predict_proba(x_tune)[:, 1]
    p_lgb_cal = lgb_full.predict_proba(x_cal)[:, 1]
    p_lgb_policy = lgb_full.predict_proba(x_policy)[:, 1]

    l1_metrics_tune = metrics(tune.fraude, p_lgb_tune)
    print(f"Métricas Nivel 1 en val_search:")
    print(f"  PR-AUC: {l1_metrics_tune['pr_auc']:.6f} (AP: {l1_metrics_tune['average_precision']:.6f}, ROC-AUC: {l1_metrics_tune['roc_auc']:.6f})")

    # 4. Generación OOF en train (5-Fold Stratified K-Fold)
    print("\n[4/6] Generando predicciones Out-Of-Fold (OOF) con 5-Fold Stratified CV en train...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    p_lgb_oof_train = np.zeros(len(train), dtype=float)

    t_cv_0 = time.perf_counter()
    for fold, (train_idx, val_idx) in enumerate(skf.split(x_train, train.fraude), 1):
        x_tr_fold = x_train.iloc[train_idx]
        y_tr_fold = train.fraude.iloc[train_idx]
        x_va_fold = x_train.iloc[val_idx]

        model_fold = lgb_model(best_params, seed, jobs).fit(x_tr_fold, y_tr_fold)
        p_lgb_oof_train[val_idx] = model_fold.predict_proba(x_va_fold)[:, 1]
        fold_pr_auc = pr_auc(train.fraude.iloc[val_idx], p_lgb_oof_train[val_idx])
        print(f"  Fold {fold}/5 finalizado | PR-AUC OOF: {fold_pr_auc:.6f}")

    cv_time = time.perf_counter() - t_cv_0
    oof_metrics_train = metrics(train.fraude, p_lgb_oof_train)
    print(f"OOF Completo en train ({cv_time:.2f}s) | PR-AUC: {oof_metrics_train['pr_auc']:.6f} | ROC-AUC: {oof_metrics_train['roc_auc']:.6f}")

    # 5. Construcción del Meta-Dataset y Entrenamiento de Metamodelos
    print("\n[5/6] Construyendo Meta-Datasets y entrenando Metamodelos de Nivel 2...")

    # Preparar matrices Nivel 2
    # Caso 1: Probabilidad directa
    X_meta_train_prob = pd.DataFrame({
        "p_lgb": p_lgb_oof_train,
        "pct_heuristico_fraude": train["pct_heuristico_fraude"].to_numpy(),
    })
    X_meta_tune_prob = pd.DataFrame({
        "p_lgb": p_lgb_tune,
        "pct_heuristico_fraude": tune["pct_heuristico_fraude"].to_numpy(),
    })
    X_meta_cal_prob = pd.DataFrame({
        "p_lgb": p_lgb_cal,
        "pct_heuristico_fraude": cal["pct_heuristico_fraude"].to_numpy(),
    })
    X_meta_policy_prob = pd.DataFrame({
        "p_lgb": p_lgb_policy,
        "pct_heuristico_fraude": policy["pct_heuristico_fraude"].to_numpy(),
    })

    # Caso 2: Espacio Logit (log-odds natural)
    eps = 1e-7
    logit_oof_train = np.log(np.clip(p_lgb_oof_train, eps, 1 - eps) / (1 - np.clip(p_lgb_oof_train, eps, 1 - eps)))
    logit_tune = np.log(np.clip(p_lgb_tune, eps, 1 - eps) / (1 - np.clip(p_lgb_tune, eps, 1 - eps)))
    logit_cal = np.log(np.clip(p_lgb_cal, eps, 1 - eps) / (1 - np.clip(p_lgb_cal, eps, 1 - eps)))
    logit_policy = np.log(np.clip(p_lgb_policy, eps, 1 - eps) / (1 - np.clip(p_lgb_policy, eps, 1 - eps)))

    X_meta_train_logit = pd.DataFrame({
        "logit_lgb": logit_oof_train,
        "pct_heuristico_fraude": train["pct_heuristico_fraude"].to_numpy(),
    })
    X_meta_tune_logit = pd.DataFrame({
        "logit_lgb": logit_tune,
        "pct_heuristico_fraude": tune["pct_heuristico_fraude"].to_numpy(),
    })
    X_meta_cal_logit = pd.DataFrame({
        "logit_lgb": logit_cal,
        "pct_heuristico_fraude": cal["pct_heuristico_fraude"].to_numpy(),
    })
    X_meta_policy_logit = pd.DataFrame({
        "logit_lgb": logit_policy,
        "pct_heuristico_fraude": policy["pct_heuristico_fraude"].to_numpy(),
    })

    # Modelos a evaluar
    meta_models = {
        "lgb_full_cmaes (Control)": {
            "type": "control",
            "p_tune": p_lgb_tune,
            "p_cal": p_lgb_cal,
            "p_policy": p_lgb_policy,
        },
        "stacking_lr_probs (C=1.0)": {
            "type": "meta",
            "model": LogisticRegression(C=1.0, max_iter=1000, random_state=seed),
            "X_tr": X_meta_train_prob,
            "X_tu": X_meta_tune_prob,
            "X_ca": X_meta_cal_prob,
            "X_po": X_meta_policy_prob,
        },
        "stacking_lr_probs_unreg (C=1e5)": {
            "type": "meta",
            "model": LogisticRegression(C=1e5, max_iter=1000, random_state=seed),
            "X_tr": X_meta_train_prob,
            "X_tu": X_meta_tune_prob,
            "X_ca": X_meta_cal_prob,
            "X_po": X_meta_policy_prob,
        },
        "stacking_lr_logit (C=1.0)": {
            "type": "meta",
            "model": LogisticRegression(C=1.0, max_iter=1000, random_state=seed),
            "X_tr": X_meta_train_logit,
            "X_tu": X_meta_tune_logit,
            "X_ca": X_meta_cal_logit,
            "X_po": X_meta_policy_logit,
        },
        "stacking_lr_logit_unreg (C=1e5)": {
            "type": "meta",
            "model": LogisticRegression(C=1e5, max_iter=1000, random_state=seed),
            "X_tr": X_meta_train_logit,
            "X_tu": X_meta_tune_logit,
            "X_ca": X_meta_cal_logit,
            "X_po": X_meta_policy_logit,
        },
    }

    # Ajustar cada meta-modelo y calcular predicciones
    results_comparison = []
    policy_results = {}

    for name, spec in meta_models.items():
        if spec["type"] == "meta":
            m = spec["model"]
            m.fit(spec["X_tr"], train.fraude)
            p_tu = m.predict_proba(spec["X_tu"])[:, 1]
            p_ca = m.predict_proba(spec["X_ca"])[:, 1]
            p_po = m.predict_proba(spec["X_po"])[:, 1]
            spec["p_tune"] = p_tu
            spec["p_cal"] = p_ca
            spec["p_policy"] = p_po

            # Inspeccionar coeficientes
            coefs = dict(zip(spec["X_tr"].columns, m.coef_[0].tolist()))
            intercept = float(m.intercept_[0])
            print(f"\n--- {name} ---")
            print(f"  Intercept: {intercept:.5f}")
            for col, val in coefs.items():
                print(f"  Coef {col}: {val:.5f}")
            spec["weights"] = {"intercept": intercept, "coefs": coefs}
        else:
            p_tu = spec["p_tune"]
            p_ca = spec["p_cal"]
            p_po = spec["p_policy"]

        # Métricas de validación de búsqueda
        m_tune = metrics(tune.fraude, p_tu)
        results_comparison.append({
            "model": name,
            "pr_auc": m_tune["pr_auc"],
            "ap": m_tune["average_precision"],
            "roc_auc": m_tune["roc_auc"],
            "brier": m_tune["brier"],
            "log_loss": m_tune["log_loss"],
        })

        # Calibración en val_calibration
        calibrator = LogisticRegression(C=1e6, max_iter=1000, random_state=seed).fit(
            logit(p_ca), cal.fraude
        )
        p_po_cal = calibrate(p_po, calibrator)
        m_policy_raw = metrics(policy.fraude, p_po)
        m_policy_cal = metrics(policy.fraude, p_po_cal)

        # Escenarios de costo en val_policy (límite operativo: FP <= 100)
        # Costo estándar: 5 * FP + 200 * FN
        curve_5 = cost_curve(policy.fraude, p_po_cal, policy.monto, fp_cost=5.0, loss_fraction=1.0, fn_cost=200.0)
        feasible_5 = curve_5.loc[curve_5["fp"] <= 100]
        best_5 = feasible_5.loc[feasible_5["cost"].idxmin()] if not feasible_5.empty else curve_5.iloc[0]

        # Costo monto
        curve_monto = cost_curve(policy.fraude, p_po_cal, policy.monto, fp_cost=5.0, loss_fraction=1.0, fn_cost=None)
        feasible_monto = curve_monto.loc[curve_monto["fp"] <= 100]
        best_monto = feasible_monto.loc[feasible_monto["cost"].idxmin()] if not feasible_monto.empty else curve_monto.iloc[0]

        policy_results[name] = {
            "val_policy_pr_auc_raw": m_policy_raw["pr_auc"],
            "val_policy_pr_auc_cal": m_policy_cal["pr_auc"],
            "cost_5_200": {
                "threshold": float(best_5["threshold"]),
                "fp": int(best_5["fp"]),
                "tp": int(best_5["tp"]),
                "fn": int(best_5["fn"]),
                "cost": float(best_5["cost"]),
                "savings": float(best_5["savings_vs_allow_all"]),
            },
            "cost_monto": {
                "threshold": float(best_monto["threshold"]),
                "fp": int(best_monto["fp"]),
                "tp": int(best_monto["tp"]),
                "fn": int(best_monto["fn"]),
                "cost": float(best_monto["cost"]),
                "savings": float(best_monto["savings_vs_allow_all"]),
            },
        }

    # 6. Reporte Final y Exportación
    print("\n" + "=" * 70)
    print("COMPARACIÓN FINAL DE RESULTADOS EN VALIDACIÓN")
    print("=" * 70)
    res_df = pd.DataFrame(results_comparison)
    for idx, row in res_df.iterrows():
        name = row["model"]
        pol = policy_results[name]
        c5 = pol["cost_5_200"]
        cm = pol["cost_monto"]
        print(f"\n* Modelo: {name}")
        print(f"  val_search -> PR-AUC: {row['pr_auc']:.6f} | AP: {row['ap']:.6f} | ROC-AUC: {row['roc_auc']:.6f}")
        print(f"  val_policy -> PR-AUC Cal: {pol['val_policy_pr_auc_cal']:.6f} | Costo ($5FP+200FN): ${c5['cost']:.0f} (FP: {c5['fp']}, TP: {c5['tp']}, FN: {c5['fn']}) | Costo Monto: ${cm['cost']:.0f}")

    # Guardar resultados JSON
    summary = {
        "comparison_val_search": results_comparison,
        "policy_results_val_policy": policy_results,
        "meta_models_weights": {k: v.get("weights") for k, v in meta_models.items() if "weights" in v},
        "seed": seed,
        "oof_train_pr_auc": oof_metrics_train["pr_auc"],
    }
    with open(out_path / "stacking_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Guardar predicciones de validación
    preds_df = pd.DataFrame({
        "transaction_id": tune["transaction_id"].tolist() + cal["transaction_id"].tolist() + policy["transaction_id"].tolist(),
        "split_block": ["val_search"] * len(tune) + ["val_calibration"] * len(cal) + ["val_policy"] * len(policy),
        "fraude": tune["fraude"].tolist() + cal["fraude"].tolist() + policy["fraude"].tolist(),
        "pct_heuristico_fraude": tune["pct_heuristico_fraude"].tolist() + cal["pct_heuristico_fraude"].tolist() + policy["pct_heuristico_fraude"].tolist(),
    })
    for name, spec in meta_models.items():
        clean_name = name.split()[0].replace("(", "").replace(")", "")
        preds_df[f"p_{clean_name}"] = np.r_[spec["p_tune"], spec["p_cal"], spec["p_policy"]]

    preds_df.to_parquet(out_path / "validation_predictions.parquet", index=False)
    print(f"\nResultados guardados exitosamente en: {out_path}")
    return summary


if __name__ == "__main__":
    run_stacking_experiment()
