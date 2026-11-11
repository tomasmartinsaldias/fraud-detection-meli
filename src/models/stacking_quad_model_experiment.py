"""Quad-Model Stacking Experiment (LGBM + XGBoost + Random Forest + Naive Bayes):
Level 1:
  - Model 1: LightGBM (11 Forward-Selected vars, CMA-ES tuned)
  - Model 2: XGBoost (12 Forward-Selected vars, CMA-ES tuned)
  - Model 3: Random Forest (14 Forward-Selected vars, CMA-ES tuned)
  - Model 4: Naive Bayes (Forward-Selected vars, CMA-ES tuned var_smoothing)
  - Full 4x4 Correlation Analysis (Pearson & Spearman) on val_search.
Level 2:
  - Meta-Dataset with 20 columns:
    [p_lgb, logit_lgb, p_xgb, logit_xgb, p_rf, logit_rf, p_nb, logit_nb] +
    11 Anomaly Scores (PCA, iForest, LOF, OCSVM, GMM) +
    pct_heuristico_fraude.
Meta-Models:
  - Equal-weight Ensemble (p_lgb + p_xgb + p_rf + p_nb) / 4
  - Logistic Regression L2 (C=1.0)
  - Logistic Regression L1 / Lasso (C=0.1)
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
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
import xgboost as xgb

from src.models.supervised import (
    calibrate, cost_curve, logit, metrics, policy_metrics, pr_auc,
    validate_development, validation_blocks,
)
from src.models.train import load_data, lgb_model

BASE_CATEGORICAL_COLS = ["a", "g", "n", "o", "p"]
BASE_NUMERIC_COLS = ["score", "monto", "b", "c", "d", "e", "f", "h", "l", "m", "q"]


def make_nb_pipeline(var_smoothing=1e-9):
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    # Preprocessor will be built per column set
    return numeric_transformer, categorical_transformer


def build_nb_estimator(cols: List[str], var_smoothing=1e-9):
    cat = [c for c in cols if c in BASE_CATEGORICAL_COLS]
    num = [c for c in cols if c not in cat]

    transformers = []
    if num:
        transformers.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sca", StandardScaler())]), num))
    if cat:
        transformers.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat))

    preprocessor = ColumnTransformer(transformers=transformers)
    return Pipeline([
        ("prep", preprocessor),
        ("gnb", GaussianNB(var_smoothing=var_smoothing)),
    ])


def run_forward_selection_nb(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    candidate_cols: List[str],
    seed: int = 42,
    patience_limit: int = 2,
    min_delta: float = 0.0003,
    max_steps: int = 15,
) -> Tuple[List[str], float]:
    print("\n--- Ejecutando Forward Stepwise Selection para Naive Bayes (Proxy) ---")
    seed_col = "score"
    selected = [seed_col]
    remaining = [c for c in candidate_cols if c != seed_col]

    # Evaluar semilla
    pipe0 = build_nb_estimator([seed_col], var_smoothing=1e-7).fit(X_train[[seed_col]], y_train)
    p0 = pipe0.predict_proba(X_val[[seed_col]])[:, 1]
    best_pr = pr_auc(y_val, p0)
    print(f"Paso 1: + {seed_col:<24s} -> PR-AUC: {best_pr:.5f} [SEMILLA INICIAL]")

    best_selected = list(selected)
    patience = 0
    step = 2

    def eval_cand(cand):
        cols = selected + [cand]
        m = build_nb_estimator(cols, var_smoothing=1e-7).fit(X_train[cols], y_train)
        p = m.predict_proba(X_val[cols])[:, 1]
        return cand, pr_auc(y_val, p)

    while remaining and patience < patience_limit and step <= max_steps:
        scores = joblib.Parallel(n_jobs=8)(
            joblib.delayed(eval_cand)(c) for c in remaining
        )
        best_cand, best_cand_pr = max(scores, key=lambda kv: kv[1])
        delta = best_cand_pr - best_pr

        selected.append(best_cand)
        remaining.remove(best_cand)

        if delta >= min_delta:
            best_pr = best_cand_pr
            best_selected = list(selected)
            patience = 0
            print(f"Paso {step:2d}: + {best_cand:<24s} -> PR-AUC: {best_cand_pr:.5f} (Delta: {delta:+.5f}) [ACEPTADA]")
        else:
            patience += 1
            print(f"Paso {step:2d}: + {best_cand:<24s} -> PR-AUC: {best_cand_pr:.5f} (Delta: {delta:+.5f}) [SIN MEJORA - Paciencia {patience}/{patience_limit}]")
        step += 1

    print(f"Naive Bayes Forward Selection finalizado. Variables ({len(best_selected)}): {best_selected}")
    return best_selected, best_pr


def tune_nb_cmaes(X_train, y_train, X_val, y_val, cols, seed=42, n_trials=15):
    print(f"\n--- Optimizando Hiperparámetros de Naive Bayes con CMA-ES ({n_trials} trials) ---")
    sampler = optuna.samplers.CmaEsSampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial):
        var_smoothing = trial.suggest_float("var_smoothing", 1e-11, 1.0, log=True)
        pipe = build_nb_estimator(cols, var_smoothing=var_smoothing).fit(X_train[cols], y_train)
        p = pipe.predict_proba(X_val[cols])[:, 1]
        return pr_auc(y_val, p)

    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    best_var_smoothing = study.best_params["var_smoothing"]
    print(f"Mejor Naive Bayes PR-AUC: {study.best_value:.5f} con var_smoothing: {best_var_smoothing:.2e}")
    return best_var_smoothing


def run_quad_model_stacking():
    seed = 42
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    output_dir = ROOT / "models/experiments/exp_quad_model_stacking"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EXPERIMENTO QUAD-MODEL STACKING (LGBM + XGBOOST + RF + NAIVE BAYES)")
    print("=" * 80)

    # 1. Cargar datos
    print("\n[1/7] Cargando base de datos, transaccionales, anomalías y reglas...")
    base_df = pd.read_parquet(ROOT / "data/processed/dataset_base.parquet", filters=[("split", "in", ["train", "val"])])
    feat_df = pd.read_parquet(ROOT / "data/features/features_transaccionales.parquet")
    anom_df = pd.read_parquet(ROOT / "data/features/scores_anomalias.parquet")

    exclude_anom = ["score_anomalia_consolidado", "anomalia_x_monto", "extended_iforest_x_monto"]
    anom_cols = [c for c in anom_df.columns if c not in ["transaction_id"] + exclude_anom]

    if "score_heuristico_fraude_10" in feat_df.columns:
        feat_df = feat_df.drop(columns=["score_heuristico_fraude_10"])

    df = base_df.merge(feat_df, on="transaction_id").merge(anom_df[["transaction_id"] + anom_cols], on="transaction_id")
    validate_development(df)

    train = df.loc[df.split == "train"].copy()
    tune, cal, policy = validation_blocks(df.loc[df.split == "val"])

    y_train = train.fraude.to_numpy()
    y_tune = tune.fraude.to_numpy()
    y_cal = cal.fraude.to_numpy()
    y_policy = policy.fraude.to_numpy()

    base_cols_all = [c for c in BASE_NUMERIC_COLS + BASE_CATEGORICAL_COLS if c in df.columns]
    eng_cols = [c for c in feat_df.columns if c not in ["transaction_id", "pct_heuristico_fraude"]]
    candidate_cols = list(dict.fromkeys(base_cols_all + eng_cols))

    # Formatos de datos para XGB, RF y NB
    df_xgb = df.copy()
    for cat in BASE_CATEGORICAL_COLS:
        if cat in df_xgb.columns:
            df_xgb[cat] = df_xgb[cat].astype("category")

    ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df_rf = df.copy()
    cat_present = [c for c in BASE_CATEGORICAL_COLS if c in df_rf.columns]
    df_rf[cat_present] = ord_enc.fit_transform(df_rf[cat_present])
    for c in candidate_cols:
        if df_rf[c].isna().any():
            med = df_rf.loc[df_rf.split == "train", c].median()
            df_rf[c] = df_rf[c].fillna(med)

    train_xgb, tune_xgb, cal_xgb, policy_xgb = df_xgb.loc[df.split == "train"], df_xgb.loc[tune.index], df_xgb.loc[cal.index], df_xgb.loc[policy.index]
    train_rf, tune_rf, cal_rf, policy_rf = df_rf.loc[df.split == "train"], df_rf.loc[tune.index], df_rf.loc[cal.index], df_rf.loc[policy.index]

    # =========================================================================
    # MODELO 1: LightGBM (11 vars)
    # =========================================================================
    print("\n[2/7] Ajustando Modelo 1: LightGBM...")
    lgb_cols = ["score", "o", "g", "a", "es_madrugada", "ratio_d_m", "ratio_monto_l", "prod_b_log_c", "score_texto_nb", "ratio_monto_e", "is_h_cero"]
    best_params_lgb = {
        "learning_rate": 0.03486992722440761,
        "num_leaves": 28,
        "min_child_samples": 100,
        "colsample_bytree": 0.8140488199767426,
        "subsample": 0.7573516799506432,
        "reg_alpha": 0.013018028720806228,
        "reg_lambda": 0.5270037815983998,
        "n_estimators": 253,
    }
    m_lgb = lgb_model(best_params_lgb, seed, 4).fit(train_xgb[lgb_cols], y_train)
    p_lgb_tune = m_lgb.predict_proba(tune_xgb[lgb_cols])[:, 1]
    p_lgb_cal = m_lgb.predict_proba(cal_xgb[lgb_cols])[:, 1]
    p_lgb_policy = m_lgb.predict_proba(policy_xgb[lgb_cols])[:, 1]
    print(f"LGBM 11 vars -> val_search PR-AUC: {pr_auc(y_tune, p_lgb_tune):.5f}")

    # =========================================================================
    # MODELO 2: XGBoost (12 vars)
    # =========================================================================
    print("\n[3/7] Ajustando Modelo 2: XGBoost...")
    xgb_cols = ["score", "o", "te_j_bayes", "es_madrugada", "ratio_monto_l", "q", "score_x_monto", "score_texto_nb", "c", "diff_m_d", "g", "flag_alto_riesgo_nop"]
    xgb_params = {
        "learning_rate": 0.03880453457276846,
        "max_depth": 5,
        "min_child_weight": 25,
        "colsample_bytree": 0.8896276959161779,
        "subsample": 0.807191755093485,
        "reg_alpha": 0.012021842230622207,
        "reg_lambda": 0.2515850109156839,
        "n_estimators": 180,
        "tree_method": "hist",
        "enable_categorical": True,
        "random_state": seed,
        "n_jobs": 4,
        "verbosity": 0,
    }
    m_xgb = xgb.XGBClassifier(**xgb_params).fit(train_xgb[xgb_cols], y_train)
    p_xgb_tune = m_xgb.predict_proba(tune_xgb[xgb_cols])[:, 1]
    p_xgb_cal = m_xgb.predict_proba(cal_xgb[xgb_cols])[:, 1]
    p_xgb_policy = m_xgb.predict_proba(policy_xgb[xgb_cols])[:, 1]
    print(f"XGBoost 12 vars -> val_search PR-AUC: {pr_auc(y_tune, p_xgb_tune):.5f}")

    # =========================================================================
    # MODELO 3: Random Forest (14 vars)
    # =========================================================================
    print("\n[4/7] Ajustando Modelo 3: Random Forest...")
    rf_cols = ["score", "o", "te_j_bayes", "es_madrugada", "q", "ratio_monto_c", "ratio_monto_l", "score_texto_nb", "hora_cont", "dia_semana", "n", "is_e_cero", "g", "is_o_missing"]
    rf_params = {
        "max_depth": 11,
        "min_samples_leaf": 35,
        "max_features": 0.5647846214083654,
        "n_estimators": 66,
        "random_state": seed,
        "n_jobs": 4,
    }
    m_rf = RandomForestClassifier(**rf_params).fit(train_rf[rf_cols], y_train)
    p_rf_tune = m_rf.predict_proba(tune_rf[rf_cols])[:, 1]
    p_rf_cal = m_rf.predict_proba(cal_rf[rf_cols])[:, 1]
    p_rf_policy = m_rf.predict_proba(policy_rf[rf_cols])[:, 1]
    print(f"Random Forest 14 vars -> val_search PR-AUC: {pr_auc(y_tune, p_rf_tune):.5f}")

    # =========================================================================
    # MODELO 4: Naive Bayes (Forward Selection & CMA-ES)
    # =========================================================================
    print("\n[5/7] Configurando y Optimizando Modelo 4: Naive Bayes...")
    nb_cols, _ = run_forward_selection_nb(
        train, y_train, tune, y_tune, candidate_cols, seed=seed, patience_limit=2, min_delta=0.0003
    )
    best_smoothing = tune_nb_cmaes(train, y_train, tune, y_tune, nb_cols, seed=seed, n_trials=15)

    pipe_nb = build_nb_estimator(nb_cols, var_smoothing=best_smoothing).fit(train[nb_cols], y_train)
    p_nb_tune = pipe_nb.predict_proba(tune[nb_cols])[:, 1]
    p_nb_cal = pipe_nb.predict_proba(cal[nb_cols])[:, 1]
    p_nb_policy = pipe_nb.predict_proba(policy[nb_cols])[:, 1]
    print(f"Naive Bayes {len(nb_cols)} vars -> val_search PR-AUC: {pr_auc(y_tune, p_nb_tune):.5f}")

    # =========================================================================
    # MATRICES DE CORRELACIÓN 4X4 (val_search)
    # =========================================================================
    print("\n[6/7] Analizando la correlación de predicciones de los 4 modelos en val_search...")
    preds_val_df = pd.DataFrame({
        "p_lgb": p_lgb_tune,
        "p_xgb": p_xgb_tune,
        "p_rf": p_rf_tune,
        "p_nb": p_nb_tune,
    })
    pearson_corr = preds_val_df.corr(method="pearson")
    spearman_corr = preds_val_df.corr(method="spearman")

    print("\n" + "=" * 60)
    print("--- MATRIZ DE CORRELACIÓN DE PEARSON (val_search) ---")
    print(pearson_corr.round(4))
    print("\n--- MATRIZ DE CORRELACIÓN DE SPEARMAN (RANK) (val_search) ---")
    print(spearman_corr.round(4))
    print("=" * 60)

    # =========================================================================
    # OOF CV 5-FOLD EN TRAIN PARA TODOS LOS MODELOS
    # =========================================================================
    print("\nGenerando predicciones Out-Of-Fold (OOF) con 5-Fold CV...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    p_lgb_oof = np.zeros(len(train), dtype=float)
    p_xgb_oof = np.zeros(len(train), dtype=float)
    p_rf_oof = np.zeros(len(train), dtype=float)
    p_nb_oof = np.zeros(len(train), dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y_train), 1):
        # LGB
        m_lgb_f = lgb_model(best_params_lgb, seed, 4).fit(train_xgb[lgb_cols].iloc[tr_idx], y_train[tr_idx])
        p_lgb_oof[va_idx] = m_lgb_f.predict_proba(train_xgb[lgb_cols].iloc[va_idx])[:, 1]

        # XGB
        m_xgb_f = xgb.XGBClassifier(**xgb_params).fit(train_xgb[xgb_cols].iloc[tr_idx], y_train[tr_idx])
        p_xgb_oof[va_idx] = m_xgb_f.predict_proba(train_xgb[xgb_cols].iloc[va_idx])[:, 1]

        # RF
        m_rf_f = RandomForestClassifier(**rf_params).fit(train_rf[rf_cols].iloc[tr_idx], y_train[tr_idx])
        p_rf_oof[va_idx] = m_rf_f.predict_proba(train_rf[rf_cols].iloc[va_idx])[:, 1]

        # NB
        m_nb_f = build_nb_estimator(nb_cols, var_smoothing=best_smoothing).fit(train[nb_cols].iloc[tr_idx], y_train[tr_idx])
        p_nb_oof[va_idx] = m_nb_f.predict_proba(train[nb_cols].iloc[va_idx])[:, 1]

    print(f"OOF PR-AUC -> LGB: {pr_auc(y_train, p_lgb_oof):.5f} | XGB: {pr_auc(y_train, p_xgb_oof):.5f} | RF: {pr_auc(y_train, p_rf_oof):.5f} | NB: {pr_auc(y_train, p_nb_oof):.5f}")

    # =========================================================================
    # META-DATASET DE NIVEL 2 (20 columnas)
    # =========================================================================
    print("\n[7/7] Construyendo Meta-Dataset de 20 variables y evaluando metamodelos...")
    eps = 1e-7

    def build_meta_quad(p_lgb_vec, p_xgb_vec, p_rf_vec, p_nb_vec, part_df):
        d = {
            "p_lgb": p_lgb_vec,
            "logit_lgb": np.log(np.clip(p_lgb_vec, eps, 1 - eps) / (1 - np.clip(p_lgb_vec, eps, 1 - eps))),
            "p_xgb": p_xgb_vec,
            "logit_xgb": np.log(np.clip(p_xgb_vec, eps, 1 - eps) / (1 - np.clip(p_xgb_vec, eps, 1 - eps))),
            "p_rf": p_rf_vec,
            "logit_rf": np.log(np.clip(p_rf_vec, eps, 1 - eps) / (1 - np.clip(p_rf_vec, eps, 1 - eps))),
            "p_nb": p_nb_vec,
            "logit_nb": np.log(np.clip(p_nb_vec, eps, 1 - eps) / (1 - np.clip(p_nb_vec, eps, 1 - eps))),
        }
        for col in anom_cols:
            d[col] = part_df[col].to_numpy(dtype=float)
        d["pct_heuristico_fraude"] = part_df["pct_heuristico_fraude"].to_numpy(dtype=float)
        return pd.DataFrame(d)

    X_meta_train = build_meta_quad(p_lgb_oof, p_xgb_oof, p_rf_oof, p_nb_oof, train)
    X_meta_tune = build_meta_quad(p_lgb_tune, p_xgb_tune, p_rf_tune, p_nb_tune, tune)
    X_meta_cal = build_meta_quad(p_lgb_cal, p_xgb_cal, p_rf_cal, p_nb_cal, cal)
    X_meta_policy = build_meta_quad(p_lgb_policy, p_xgb_policy, p_rf_policy, p_nb_policy, policy)

    meta_cols = list(X_meta_train.columns)
    print(f"Total Columnas en Meta-Dataset Cuádruple: {len(meta_cols)}")

    scaler = StandardScaler().fit(X_meta_train)
    X_meta_tr_sc = pd.DataFrame(scaler.transform(X_meta_train), columns=meta_cols)
    X_meta_tu_sc = pd.DataFrame(scaler.transform(X_meta_tune), columns=meta_cols)
    X_meta_ca_sc = pd.DataFrame(scaler.transform(X_meta_cal), columns=meta_cols)
    X_meta_po_sc = pd.DataFrame(scaler.transform(X_meta_policy), columns=meta_cols)

    p_avg_tune = (p_lgb_tune + p_xgb_tune + p_rf_tune + p_nb_tune) / 4.0
    p_avg_cal = (p_lgb_cal + p_xgb_cal + p_rf_cal + p_nb_cal) / 4.0
    p_avg_policy = (p_lgb_policy + p_xgb_policy + p_rf_policy + p_nb_policy) / 4.0

    eval_candidates = {
        "Base LightGBM (11 vars)": {"p_tu": p_lgb_tune, "p_ca": p_lgb_cal, "p_po": p_lgb_policy},
        "Base XGBoost (12 vars)": {"p_tu": p_xgb_tune, "p_ca": p_xgb_cal, "p_po": p_xgb_policy},
        "Base Random Forest (14 vars)": {"p_tu": p_rf_tune, "p_ca": p_rf_cal, "p_po": p_rf_policy},
        "Base Naive Bayes": {"p_tu": p_nb_tune, "p_ca": p_nb_cal, "p_po": p_nb_policy},
        "Ensamble Equiponderado (4 modelos)": {"p_tu": p_avg_tune, "p_ca": p_avg_cal, "p_po": p_avg_policy},
    }

    # Meta-LR L2
    m_l2 = LogisticRegression(C=1.0, max_iter=1000, random_state=seed).fit(X_meta_tr_sc, y_train)
    eval_candidates["Meta-LR L2 (C=1.0)"] = {
        "p_tu": m_l2.predict_proba(X_meta_tu_sc)[:, 1],
        "p_ca": m_l2.predict_proba(X_meta_ca_sc)[:, 1],
        "p_po": m_l2.predict_proba(X_meta_po_sc)[:, 1],
        "model": m_l2,
    }

    # Meta-LR Lasso
    m_l1 = LogisticRegression(penalty="l1", solver="saga", C=0.1, max_iter=1000, random_state=seed).fit(X_meta_tr_sc, y_train)
    eval_candidates["Meta-LR Lasso (C=0.1)"] = {
        "p_tu": m_l1.predict_proba(X_meta_tu_sc)[:, 1],
        "p_ca": m_l1.predict_proba(X_meta_ca_sc)[:, 1],
        "p_po": m_l1.predict_proba(X_meta_po_sc)[:, 1],
        "model": m_l1,
    }

    # Imprimir coeficientes de los metamodelos
    for mname in ["Meta-LR L2 (C=1.0)", "Meta-LR Lasso (C=0.1)"]:
        m = eval_candidates[mname]["model"]
        c_dict = dict(zip(meta_cols, m.coef_[0].tolist()))
        print(f"\n--- Coeficientes {mname} ---")
        for k, v in sorted(c_dict.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10]:
            print(f"  {k:<30s}: {v:+.5f}")

    # Tabla resumen de validación
    res_table = []
    for name, spec in eval_candidates.items():
        m_tu = metrics(y_tune, spec["p_tu"])
        calibrator = LogisticRegression(C=1e6, max_iter=1000, random_state=seed).fit(
            logit(spec["p_ca"]), y_cal
        )
        p_po_cal = calibrate(spec["p_po"], calibrator)
        m_po_cal = metrics(y_policy, p_po_cal)

        c5 = cost_curve(y_policy, p_po_cal, policy.monto, fp_cost=5.0, loss_fraction=1.0, fn_cost=200.0)
        feas5 = c5.loc[c5["fp"] <= 100]
        b5 = feas5.loc[feas5["cost"].idxmin()] if not feas5.empty else c5.iloc[0]

        cm = cost_curve(y_policy, p_po_cal, policy.monto, fp_cost=5.0, loss_fraction=1.0, fn_cost=None)
        feasm = cm.loc[cm["fp"] <= 100]
        bm = feasm.loc[feasm["cost"].idxmin()] if not feasm.empty else cm.iloc[0]

        res_table.append({
            "model": name,
            "pr_auc_search": m_tu["pr_auc"],
            "ap_search": m_tu["average_precision"],
            "roc_auc_search": m_tu["roc_auc"],
            "pr_auc_policy_cal": m_po_cal["pr_auc"],
            "cost_5_200": float(b5["cost"]),
            "fp": int(b5["fp"]),
            "tp": int(b5["tp"]),
            "fn": int(b5["fn"]),
            "cost_monto": float(bm["cost"]),
        })

    summary_df = pd.DataFrame(res_table)
    print("\n" + "=" * 90)
    print("RESULTADOS COMPARATIVOS FINALES DEL ENFOQUE CUÁDRUPLE (INCLUYENDO NAIVE BAYES)")
    print("=" * 90)
    for _, row in summary_df.iterrows():
        print(f"* {row['model']:<36s} | val_search PR-AUC: {row['pr_auc_search']:.5f} | Costo ($5FP+200FN): ${row['cost_5_200']:<5.0f} (FP:{row['fp']}, TP:{row['tp']}, FN:{row['fn']}) | Costo Monto: ${row['cost_monto']:.0f}")

    # Guardar resumen en JSON
    summary_data = {
        "summary": res_table,
        "pearson_corr": pearson_corr.to_dict(),
        "spearman_corr": spearman_corr.to_dict(),
        "nb_selected_features": nb_cols,
        "nb_best_var_smoothing": best_smoothing,
        "meta_features": meta_cols,
    }
    with open(output_dir / "quad_model_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    print(f"\nResultados y reporte guardados en: {output_dir}")
    return summary_data


if __name__ == "__main__":
    run_quad_model_stacking()
