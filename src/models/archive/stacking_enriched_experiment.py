"""Enriched Stacking Meta-Model Experiment:
Level 1: LightGBM on 11 Forward-Selected transactional features, tuned with CMA-ES.
Level 2: Enriched Meta-Dataset containing:
  - p_lgb (OOF on train)
  - logit_lgb
  - Franco's 11 Anomaly Scores (PCA, iForest, LOF, OCSVM, GMM)
  - Franco's Heuristic Rule Scores (pct_heuristico_fraude, score_heuristico_fraude_10)
Meta-Models:
  - LogisticRegression (L2, C=1.0)
  - LogisticRegression (L1 / Lasso feature selection)
  - LightGBM Meta-Learner (shallow non-linear interactions)
Evaluation:
  Strictly on temporal validation: val_search, val_calibration, val_policy.
"""
from __future__ import annotations

import json
from pathlib import Path
import os
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.models.supervised import (
    BASE_CATEGORICAL, BASE_NUMERIC, FrameEncoder, calibrate, check_ids,
    cost_curve, logit, metrics, policy_metrics, pr_auc,
    validate_development, validation_blocks,
)
from src.models.train import load_data, lgb_model


def run_enriched_stacking(
    base_path="data/processed/dataset_base.parquet",
    feat_path="data/features/features_seleccionadas_oficiales_11.parquet",
    anom_path="data/features/scores_anomalias.parquet",
    rules_path="data/features/features_reglas_asociacion.parquet",
    output_dir="models/experiments/exp_stacking_enriched_meta",
    seed=42,
    jobs=4,
):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    out_path = ROOT / output_dir
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EXPERIMENTO DE STACKING ENRIQUECIDO DETERMINÍSTICO (Semilla 42)")
    print("=" * 80)

    # 1. Cargar datos
    print("\n[1/6] Cargando datos y particiones de desarrollo...")
    exclude_anom = ["score_anomalia_consolidado", "anomalia_x_monto", "extended_iforest_x_monto"]
    rule_cols = ["pct_heuristico_fraude", "score_heuristico_fraude_10"]

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

    # Features Nivel 1: Las 11 variables transaccionales puras de Forward Selection
    l1_features = list(feat_cols)
    print(f"Features Nivel 1 (Forward Selection {len(l1_features)} vars): {l1_features}")
    print(f"Features Anomalías Franco ({len(anom_cols)} vars): {anom_cols}")
    print(f"Features Reglas Heurísticas Franco ({len(rules_cols)} vars): {rules_cols}")

    # 2. Hiperparámetros óptimos CMA-ES para las 11 variables
    best_params_l1 = {
        "learning_rate": 0.03486992722440761,
        "num_leaves": 28,
        "min_child_samples": 100,
        "colsample_bytree": 0.8140488199767426,
        "subsample": 0.7573516799506432,
        "reg_alpha": 0.013018028720806228,
        "reg_lambda": 0.5270037815983998,
        "n_estimators": 253,
    }
    print(f"\n[2/6] Parámetros CMA-ES Modelo Nivel 1 (11 variables):\n{json.dumps(best_params_l1, indent=2)}")

    encoder = FrameEncoder().fit(train[l1_features])
    x_train_l1 = encoder.transform(train[l1_features])
    x_tune_l1 = encoder.transform(tune[l1_features])
    x_cal_l1 = encoder.transform(cal[l1_features])
    x_policy_l1 = encoder.transform(policy[l1_features])

    # 3. Entrenar Modelo Base Nivel 1 en train completo
    print("\n[3/6] Ajustando Modelo Base Nivel 1 en train completo...")
    t0 = time.perf_counter()
    lgb_l1 = lgb_model(best_params_l1, seed, jobs).fit(x_train_l1, train.fraude)
    print(f"Modelo Nivel 1 ajustado en {time.perf_counter() - t0:.2f}s.")

    p_lgb_tune = lgb_l1.predict_proba(x_tune_l1)[:, 1]
    p_lgb_cal = lgb_l1.predict_proba(x_cal_l1)[:, 1]
    p_lgb_policy = lgb_l1.predict_proba(x_policy_l1)[:, 1]

    m_l1_tune = metrics(tune.fraude, p_lgb_tune)
    print(f"Rendimiento Modelo Nivel 1 (11 vars) en val_search:")
    print(f"  PR-AUC: {m_l1_tune['pr_auc']:.6f} | AP: {m_l1_tune['average_precision']:.6f} | ROC-AUC: {m_l1_tune['roc_auc']:.6f}")

    # 4. Generar OOF en train con 5-Fold Stratified CV
    print("\n[4/6] Generando predicciones Out-Of-Fold (OOF) con 5-Fold Stratified CV en train...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    p_lgb_oof_train = np.zeros(len(train), dtype=float)

    t_cv = time.perf_counter()
    for fold, (tr_idx, va_idx) in enumerate(skf.split(x_train_l1, train.fraude), 1):
        x_tr_fold = x_train_l1.iloc[tr_idx]
        y_tr_fold = train.fraude.iloc[tr_idx]
        x_va_fold = x_train_l1.iloc[va_idx]

        m_fold = lgb_model(best_params_l1, seed, jobs).fit(x_tr_fold, y_tr_fold)
        p_lgb_oof_train[va_idx] = m_fold.predict_proba(x_va_fold)[:, 1]
        print(f"  Fold {fold}/5 finalizado | PR-AUC OOF: {pr_auc(train.fraude.iloc[va_idx], p_lgb_oof_train[va_idx]):.6f}")

    print(f"OOF Completo en {time.perf_counter() - t_cv:.2f}s | PR-AUC Global OOF: {pr_auc(train.fraude, p_lgb_oof_train):.6f}")

    # 5. Construcción del Meta-Dataset Enriquecido
    print("\n[5/6] Construyendo Meta-Datasets Enriquecidos...")
    eps = 1e-7
    logit_oof_train = np.log(np.clip(p_lgb_oof_train, eps, 1 - eps) / (1 - np.clip(p_lgb_oof_train, eps, 1 - eps)))
    logit_tune = np.log(np.clip(p_lgb_tune, eps, 1 - eps) / (1 - np.clip(p_lgb_tune, eps, 1 - eps)))
    logit_cal = np.log(np.clip(p_lgb_cal, eps, 1 - eps) / (1 - np.clip(p_lgb_cal, eps, 1 - eps)))
    logit_policy = np.log(np.clip(p_lgb_policy, eps, 1 - eps) / (1 - np.clip(p_lgb_policy, eps, 1 - eps)))

    meta_feature_cols = ["p_lgb", "logit_lgb"] + anom_cols + rules_cols
    print(f"Total Columnas Predictoras en Meta-Dataset ({len(meta_feature_cols)}):")
    for idx, c in enumerate(meta_feature_cols, 1):
        print(f"  {idx:2d}. {c}")

    def make_meta_df(split_df, p_vec, logit_vec):
        d = {
            "p_lgb": p_vec,
            "logit_lgb": logit_vec,
        }
        for col in anom_cols + rules_cols:
            d[col] = split_df[col].to_numpy(dtype=float)
        return pd.DataFrame(d)

    X_meta_train = make_meta_df(train, p_lgb_oof_train, logit_oof_train)
    X_meta_tune = make_meta_df(tune, p_lgb_tune, logit_tune)
    X_meta_cal = make_meta_df(cal, p_lgb_cal, logit_cal)
    X_meta_policy = make_meta_df(policy, p_lgb_policy, logit_policy)

    # Estandarización para modelos lineales
    scaler = StandardScaler().fit(X_meta_train)
    X_meta_train_scaled = pd.DataFrame(scaler.transform(X_meta_train), columns=meta_feature_cols)
    X_meta_tune_scaled = pd.DataFrame(scaler.transform(X_meta_tune), columns=meta_feature_cols)
    X_meta_cal_scaled = pd.DataFrame(scaler.transform(X_meta_cal), columns=meta_feature_cols)
    X_meta_policy_scaled = pd.DataFrame(scaler.transform(X_meta_policy), columns=meta_feature_cols)

    # 6. Definición y Ajuste de Metamodelos
    print("\n[6/6] Ajustando y evaluando metamodelos...")
    meta_candidates = {
        "lgb_transactional_cmaes (L1 Solo 11 Vars)": {
            "type": "control",
            "p_tune": p_lgb_tune,
            "p_cal": p_lgb_cal,
            "p_policy": p_lgb_policy,
        },
        "meta_lr_l2 (C=1.0)": {
            "type": "linear",
            "model": LogisticRegression(C=1.0, max_iter=1000, random_state=seed),
            "X_tr": X_meta_train_scaled,
            "X_tu": X_meta_tune_scaled,
            "X_ca": X_meta_cal_scaled,
            "X_po": X_meta_policy_scaled,
        },
        "meta_lr_l1_lasso (C=0.1)": {
            "type": "linear",
            "model": LogisticRegression(penalty="l1", solver="saga", C=0.1, max_iter=1000, random_state=seed),
            "X_tr": X_meta_train_scaled,
            "X_tu": X_meta_tune_scaled,
            "X_ca": X_meta_cal_scaled,
            "X_po": X_meta_policy_scaled,
        },
        "meta_lightgbm (Shallow Trees)": {
            "type": "tree",
            "model": lgb.LGBMClassifier(
                n_estimators=80,
                learning_rate=0.03,
                max_depth=3,
                num_leaves=7,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
                n_jobs=jobs,
                verbosity=-1,
            ),
            "X_tr": X_meta_train,
            "X_tu": X_meta_tune,
            "X_ca": X_meta_cal,
            "X_po": X_meta_policy,
        },
    }

    comparison_results = []
    policy_results = {}
    coefficients = {}

    for name, spec in meta_candidates.items():
        if spec["type"] in ("linear", "tree"):
            m = spec["model"]
            m.fit(spec["X_tr"], train.fraude)
            p_tu = m.predict_proba(spec["X_tu"])[:, 1]
            p_ca = m.predict_proba(spec["X_ca"])[:, 1]
            p_po = m.predict_proba(spec["X_po"])[:, 1]
            spec["p_tune"] = p_tu
            spec["p_cal"] = p_ca
            spec["p_policy"] = p_po

            if spec["type"] == "linear":
                coefs = dict(zip(meta_feature_cols, m.coef_[0].tolist()))
                coefficients[name] = {"intercept": float(m.intercept_[0]), "coefs": coefs}
                print(f"\n--- Coeficientes {name} ---")
                sorted_coefs = sorted(coefs.items(), key=lambda kv: abs(kv[1]), reverse=True)
                for col, val in sorted_coefs[:8]:
                    print(f"  {col:<30s}: {val:+.5f}")
            elif spec["type"] == "tree":
                imps = dict(zip(meta_feature_cols, m.feature_importances_.tolist()))
                coefficients[name] = {"importances": imps}
                print(f"\n--- Importancias {name} ---")
                sorted_imp = sorted(imps.items(), key=lambda kv: kv[1], reverse=True)
                for col, val in sorted_imp[:8]:
                    print(f"  {col:<30s}: {val:5d}")
        else:
            p_tu = spec["p_tune"]
            p_ca = spec["p_cal"]
            p_po = spec["p_policy"]

        # Métricas en val_search
        m_tune = metrics(tune.fraude, p_tu)
        comparison_results.append({
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
        m_policy_cal = metrics(policy.fraude, p_po_cal)

        # Costos en val_policy (FP <= 100)
        curve_5 = cost_curve(policy.fraude, p_po_cal, policy.monto, fp_cost=5.0, loss_fraction=1.0, fn_cost=200.0)
        feas_5 = curve_5.loc[curve_5["fp"] <= 100]
        b5 = feas_5.loc[feas_5["cost"].idxmin()] if not feas_5.empty else curve_5.iloc[0]

        curve_monto = cost_curve(policy.fraude, p_po_cal, policy.monto, fp_cost=5.0, loss_fraction=1.0, fn_cost=None)
        feas_monto = curve_monto.loc[curve_monto["fp"] <= 100]
        bm = feas_monto.loc[feas_monto["cost"].idxmin()] if not feas_monto.empty else curve_monto.iloc[0]

        policy_results[name] = {
            "val_policy_pr_auc_cal": m_policy_cal["pr_auc"],
            "cost_5_200": {
                "threshold": float(b5["threshold"]),
                "fp": int(b5["fp"]),
                "tp": int(b5["tp"]),
                "fn": int(b5["fn"]),
                "cost": float(b5["cost"]),
                "savings": float(b5["savings_vs_allow_all"]),
            },
            "cost_monto": {
                "threshold": float(bm["threshold"]),
                "fp": int(bm["fp"]),
                "tp": int(bm["tp"]),
                "fn": int(bm["fn"]),
                "cost": float(bm["cost"]),
                "savings": float(bm["savings_vs_allow_all"]),
            },
        }

    # Resumen impreso
    print("\n" + "=" * 80)
    print("RESULTADOS FINALES DEL STACKING ENRIQUECIDO EN VALIDACIÓN OOT")
    print("=" * 80)
    res_df = pd.DataFrame(comparison_results)
    for idx, row in res_df.iterrows():
        name = row["model"]
        pol = policy_results[name]
        c5 = pol["cost_5_200"]
        cm = pol["cost_monto"]
        print(f"\n* Modelo: {name}")
        print(f"  val_search -> PR-AUC: {row['pr_auc']:.6f} | AP: {row['ap']:.6f} | ROC-AUC: {row['roc_auc']:.6f}")
        print(f"  val_policy -> PR-AUC Cal: {pol['val_policy_pr_auc_cal']:.6f} | Costo ($5FP+200FN): ${c5['cost']:.0f} (FP: {c5['fp']}, TP: {c5['tp']}, FN: {c5['fn']}) | Costo Monto: ${cm['cost']:.0f}")

    # Guardar resumen en JSON
    summary = {
        "comparison_val_search": comparison_results,
        "policy_results_val_policy": policy_results,
        "coefficients": coefficients,
        "seed": seed,
        "meta_features": meta_feature_cols,
    }
    with open(out_path / "enriched_stacking_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nResultados y artefactos guardados en: {out_path}")
    return summary


if __name__ == "__main__":
    run_enriched_stacking()
