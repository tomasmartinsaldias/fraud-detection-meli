"""
Experimento de Stacking Tri-Modelo (LGBM + XGBoost + RF)
incorporando el Clustering Semántico de Títulos (Challenger v3).
================================================================
Combina la arquitectura de Stacking de Franco con las 2 variables
semánticas del Challenger v3:
- deficit_piso_cluster
- distancia_log_cluster

Evalúa el impacto en:
1. Modelos Base de Nivel 1 (LightGBM, XGBoost, Random Forest).
2. Metamodelo de Nivel 2 (Meta-LR L2 y Meta-LR Lasso).
3. Split temporal OOT: val_search, val_calibration, val_policy, y test final.
"""

from __future__ import annotations
import json
from pathlib import Path
import os
import random
import sys
import time
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
import xgboost as xgb

from src.models.supervised import (
    calibrate, cost_curve, logit, metrics, policy_metrics, pr_auc,
    validation_blocks,
)
from src.models.train import lgb_model


BASE_CATEGORICAL_COLS = ["a", "g", "n", "o", "p"]
BASE_NUMERIC_COLS = ["score", "monto", "b", "c", "d", "e", "f", "h", "l", "m", "q"]


def build_clustering_features(df_train: pd.DataFrame, df_all: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Genera deficit_piso_cluster y distancia_log_cluster estrictamente ajustados en Train."""
    print("  -> Generando clustering semántico k=50 sobre títulos...")
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=5, sublinear_tf=True, strip_accents="unicode")
    X_tr_tfidf = vec.fit_transform(df_train["i"].fillna(""))
    svd = TruncatedSVD(n_components=10, random_state=42)
    X_tr_svd = svd.fit_transform(X_tr_tfidf)

    kmeans = MiniBatchKMeans(n_clusters=50, random_state=42, batch_size=2048)
    train_clusters = kmeans.fit_predict(X_tr_svd)

    X_all_tfidf = vec.transform(df_all["i"].fillna(""))
    X_all_svd = svd.transform(X_all_tfidf)
    all_clusters = kmeans.predict(X_all_svd)

    temp_train = pd.DataFrame({"cluster_i": train_clusters, "monto": df_train["monto"].values})
    cluster_stats = temp_train.groupby("cluster_i")["monto"].agg(["median", "count"])
    global_median = float(df_train["monto"].median())
    m_prior = 10.0
    cluster_stats["bayes_median"] = (
        cluster_stats["count"] * cluster_stats["median"] + m_prior * global_median
    ) / (cluster_stats["count"] + m_prior)
    median_map = cluster_stats["bayes_median"].to_dict()

    all_medians = np.array([median_map.get(c, global_median) for c in all_clusters], dtype=float)
    all_montos = df_all["monto"].values.astype(float)

    distancia_log = (np.log1p(all_montos) - np.log1p(all_medians)).astype(float)
    deficit_piso = np.maximum(0.0, (all_medians - all_montos) / (all_medians + 1e-4)).astype(float)
    return distancia_log, deficit_piso


def run_forward_selection_xgb(
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
    print("\n--- Forward Selection para XGBoost (con candidatas + clustering) ---")
    seed_col = "score"
    selected = [seed_col]
    remaining = [c for c in candidate_cols if c != seed_col]

    proxy_params = {
        "n_estimators": 40,
        "max_depth": 4,
        "learning_rate": 0.08,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "enable_categorical": True,
        "random_state": seed,
        "n_jobs": 1,
        "verbosity": 0,
    }

    m0 = xgb.XGBClassifier(**proxy_params).fit(X_train[[seed_col]], y_train)
    p0 = m0.predict_proba(X_val[[seed_col]])[:, 1]
    best_pr = pr_auc(y_val, p0)
    print(f"Paso 1: + {seed_col:<28s} -> PR-AUC: {best_pr:.5f} [SEMILLA]")

    best_selected = list(selected)
    patience = 0
    step = 2

    def eval_cand(cand):
        cols = selected + [cand]
        m = xgb.XGBClassifier(**proxy_params).fit(X_train[cols], y_train)
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
            print(f"Paso {step:2d}: + {best_cand:<28s} -> PR-AUC: {best_cand_pr:.5f} ({delta:+.5f}) [ACEPTADA]")
        else:
            patience += 1
            print(f"Paso {step:2d}: + {best_cand:<28s} -> PR-AUC: {best_cand_pr:.5f} ({delta:+.5f}) [Paciencia {patience}/{patience_limit}]")
        step += 1

    print(f"XGBoost Variables Seleccionadas ({len(best_selected)}): {best_selected}")
    return best_selected, best_pr


def run_forward_selection_rf(
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
    print("\n--- Forward Selection para Random Forest (con candidatas + clustering) ---")
    seed_col = "score"
    selected = [seed_col]
    remaining = [c for c in candidate_cols if c != seed_col]

    proxy_params = {
        "n_estimators": 30,
        "max_depth": 8,
        "min_samples_leaf": 30,
        "random_state": seed,
        "n_jobs": 1,
    }

    m0 = RandomForestClassifier(**proxy_params).fit(X_train[[seed_col]], y_train)
    p0 = m0.predict_proba(X_val[[seed_col]])[:, 1]
    best_pr = pr_auc(y_val, p0)
    print(f"Paso 1: + {seed_col:<28s} -> PR-AUC: {best_pr:.5f} [SEMILLA]")

    best_selected = list(selected)
    patience = 0
    step = 2

    def eval_cand(cand):
        cols = selected + [cand]
        m = RandomForestClassifier(**proxy_params).fit(X_train[cols], y_train)
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
            print(f"Paso {step:2d}: + {best_cand:<28s} -> PR-AUC: {best_cand_pr:.5f} ({delta:+.5f}) [ACEPTADA]")
        else:
            patience += 1
            print(f"Paso {step:2d}: + {best_cand:<28s} -> PR-AUC: {best_cand_pr:.5f} ({delta:+.5f}) [Paciencia {patience}/{patience_limit}]")
        step += 1

    print(f"Random Forest Variables Seleccionadas ({len(best_selected)}): {best_selected}")
    return best_selected, best_pr


def tune_xgb_cmaes(X_train, y_train, X_val, y_val, cols, seed=42, n_trials=8, jobs=4):
    print(f"\n--- Fine-Tuning XGBoost con CMA-ES ({n_trials} trials) ---")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.CmaEsSampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 100),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
            "n_estimators": 180,
            "tree_method": "hist",
            "enable_categorical": True,
            "random_state": seed,
            "n_jobs": jobs,
            "verbosity": 0,
        }
        m = xgb.XGBClassifier(**params).fit(X_train[cols], y_train, eval_set=[(X_val[cols], y_val)], verbose=False)
        p = m.predict_proba(X_val[cols])[:, 1]
        return pr_auc(y_val, p)

    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    best_params = dict(study.best_params, n_estimators=180, tree_method="hist", enable_categorical=True, random_state=seed, n_jobs=jobs, verbosity=0)
    print(f"Mejor XGBoost PR-AUC: {study.best_value:.5f}")
    return best_params


def tune_rf_cmaes(X_train, y_train, X_val, y_val, cols, seed=42, n_trials=8, jobs=4):
    print(f"\n--- Fine-Tuning Random Forest con CMA-ES ({n_trials} trials) ---")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.CmaEsSampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 6, 14),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 80),
            "max_features": trial.suggest_float("max_features", 0.4, 0.95),
            "n_estimators": trial.suggest_int("n_estimators", 50, 100),
            "random_state": seed,
            "n_jobs": jobs,
        }
        m = RandomForestClassifier(**params).fit(X_train[cols], y_train)
        p = m.predict_proba(X_val[cols])[:, 1]
        return pr_auc(y_val, p)

    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    best_params = dict(study.best_params, random_state=seed, n_jobs=jobs)
    print(f"Mejor Random Forest PR-AUC: {study.best_value:.5f}")
    return best_params


def main():
    seed = 42
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    t0 = time.time()

    print("=" * 80)
    print("ENSAMBLE STACKING TRI-MODELO + CLUSTERING SEMÁNTICO V3 (CHALLENGER)")
    print("=" * 80)

    # 1. Cargar bases
    print("\n1. Cargando datos...")
    base_df = pd.read_parquet(ROOT / "data/processed/dataset_base.parquet")
    feat_df = pd.read_parquet(ROOT / "data/features/features_transaccionales.parquet")
    anom_df = pd.read_parquet(ROOT / "data/features/scores_anomalias.parquet")
    rules_df = pd.read_parquet(ROOT / "data/features/features_reglas_asociacion.parquet")[["transaction_id", "pct_heuristico_fraude"]]

    # Exclusiones de anomalías limpias
    exclude_anom = ["score_anomalia_consolidado", "anomalia_x_monto", "extended_iforest_x_monto"]
    anom_cols = [c for c in anom_df.columns if c not in ["transaction_id"] + exclude_anom]

    # Regla: SOLAMENTE pct_heuristico_fraude
    if "score_heuristico_fraude_10" in feat_df.columns:
        feat_df = feat_df.drop(columns=["score_heuristico_fraude_10"])

    # 2. Generar Clustering Semántico v3 estrictamente ajustado en Train
    train_mask = base_df["split"] == "train"
    dist_log, def_piso = build_clustering_features(base_df[train_mask], base_df)
    base_df["distancia_log_cluster"] = dist_log
    base_df["deficit_piso_cluster"] = def_piso

    # Merge
    df = (
        base_df
        .merge(feat_df, on="transaction_id")
        .merge(anom_df[["transaction_id"] + anom_cols], on="transaction_id")
        .merge(rules_df, on="transaction_id")
    )
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values(["fecha", "transaction_id"]).reset_index(drop=True)

    train = df.loc[df.split == "train"].copy()
    val_full = df.loc[df.split == "val"].copy()
    test = df.loc[df.split == "test"].copy()
    tune, cal, policy = validation_blocks(val_full)

    y_train = train.fraude.to_numpy()
    y_tune = tune.fraude.to_numpy()
    y_cal = cal.fraude.to_numpy()
    y_policy = policy.fraude.to_numpy()
    y_test = test.fraude.to_numpy()

    # Predecir categorías
    base_cols_all = [c for c in BASE_NUMERIC_COLS + BASE_CATEGORICAL_COLS if c in df.columns]
    eng_cols = [c for c in feat_df.columns if c not in ["transaction_id", "pct_heuristico_fraude"]]
    candidate_cols = list(dict.fromkeys(base_cols_all + eng_cols + ["distancia_log_cluster", "deficit_piso_cluster"]))

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

    train_xgb = df_xgb.loc[df.split == "train"]
    tune_xgb = df_xgb.loc[tune.index]
    cal_xgb = df_xgb.loc[cal.index]
    policy_xgb = df_xgb.loc[policy.index]
    test_xgb = df_xgb.loc[test.index]

    train_rf = df_rf.loc[df.split == "train"]
    tune_rf = df_rf.loc[tune.index]
    cal_rf = df_rf.loc[cal.index]
    policy_rf = df_rf.loc[policy.index]
    test_rf = df_rf.loc[test.index]

    # =========================================================================
    # MODELO 1: LightGBM con y sin Clustering
    # =========================================================================
    print("\n2. Evaluando LightGBM (Base 11 vs Challenger 13 con Clustering)...")
    lgb_11 = ["score", "o", "g", "a", "es_madrugada", "ratio_d_m", "ratio_monto_l", "prod_b_log_c", "score_texto_nb", "ratio_monto_e", "is_h_cero"]
    lgb_13 = lgb_11 + ["deficit_piso_cluster", "distancia_log_cluster"]

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

    # Control 11 vars
    m_lgb_11 = lgb_model(best_params_lgb, seed, 4).fit(train_xgb[lgb_11], y_train)
    p_lgb_11_tu = m_lgb_11.predict_proba(tune_xgb[lgb_11])[:, 1]
    p_lgb_11_te = m_lgb_11.predict_proba(test_xgb[lgb_11])[:, 1]
    print(f"  * LightGBM Control (11 vars) -> val_search: {pr_auc(y_tune, p_lgb_11_tu):.5f} | test: {pr_auc(y_test, p_lgb_11_te):.5f}")

    # Challenger 13 vars
    m_lgb_13 = lgb_model(best_params_lgb, seed, 4).fit(train_xgb[lgb_13], y_train)
    p_lgb_13_tu = m_lgb_13.predict_proba(tune_xgb[lgb_13])[:, 1]
    p_lgb_13_ca = m_lgb_13.predict_proba(cal_xgb[lgb_13])[:, 1]
    p_lgb_13_po = m_lgb_13.predict_proba(policy_xgb[lgb_13])[:, 1]
    p_lgb_13_te = m_lgb_13.predict_proba(test_xgb[lgb_13])[:, 1]
    print(f"  * LightGBM Challenger (13 vars) -> val_search: {pr_auc(y_tune, p_lgb_13_tu):.5f} | test: {pr_auc(y_test, p_lgb_13_te):.5f} (Delta: {pr_auc(y_test, p_lgb_13_te) - pr_auc(y_test, p_lgb_11_te):+.5f})")

    # =========================================================================
    # MODELO 2: XGBoost
    # =========================================================================
    print("\n3. Configurando y Fine-Tuning de XGBoost...")
    # Features seleccionadas en la corrida previa con inclusión de clustering
    xgb_cols = ['score', 'o', 'te_j_bayes', 'es_madrugada', 'ratio_monto_l', 'diff_m_d', 'score_texto_nb', 'is_h_cero', 'flag_alto_riesgo_nop', 'ratio_monto_b', 'g', 'b']
    xgb_params = tune_xgb_cmaes(train_xgb, y_train, tune_xgb, y_tune, xgb_cols, seed=seed, n_trials=6, jobs=4)
    m_xgb = xgb.XGBClassifier(**xgb_params).fit(train_xgb[xgb_cols], y_train)
    p_xgb_tu = m_xgb.predict_proba(tune_xgb[xgb_cols])[:, 1]
    p_xgb_ca = m_xgb.predict_proba(cal_xgb[xgb_cols])[:, 1]
    p_xgb_po = m_xgb.predict_proba(policy_xgb[xgb_cols])[:, 1]
    p_xgb_te = m_xgb.predict_proba(test_xgb[xgb_cols])[:, 1]
    print(f"  * XGBoost ({len(xgb_cols)} vars) -> val_search: {pr_auc(y_tune, p_xgb_tu):.5f} | test: {pr_auc(y_test, p_xgb_te):.5f}")

    # =========================================================================
    # MODELO 3: Random Forest
    # =========================================================================
    print("\n4. Configurando y Fine-Tuning de Random Forest...")
    # Features seleccionadas orgánicamente (incluye distancia_log_cluster de Challenger v3)
    rf_cols = ['score', 'o', 'te_j_bayes', 'es_madrugada', 'diff_m_d', 'distancia_log_cluster']
    rf_params = tune_rf_cmaes(train_rf, y_train, tune_rf, y_tune, rf_cols, seed=seed, n_trials=6, jobs=4)
    m_rf = RandomForestClassifier(**rf_params).fit(train_rf[rf_cols], y_train)
    p_rf_tu = m_rf.predict_proba(tune_rf[rf_cols])[:, 1]
    p_rf_ca = m_rf.predict_proba(cal_rf[rf_cols])[:, 1]
    p_rf_po = m_rf.predict_proba(policy_rf[rf_cols])[:, 1]
    p_rf_te = m_rf.predict_proba(test_rf[rf_cols])[:, 1]
    print(f"  * Random Forest ({len(rf_cols)} vars) -> val_search: {pr_auc(y_tune, p_rf_tu):.5f} | test: {pr_auc(y_test, p_rf_te):.5f}")

    # =========================================================================
    # OOF PREDICTIONS (5-Fold Stratified CV en Train)
    # =========================================================================
    print("\n5. Generando predicciones Out-Of-Fold (OOF) con 5-Fold Stratified CV...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    p_lgb_oof = np.zeros(len(train), dtype=float)
    p_xgb_oof = np.zeros(len(train), dtype=float)
    p_rf_oof = np.zeros(len(train), dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y_train), 1):
        m_lgb_f = lgb_model(best_params_lgb, seed, 4).fit(train_xgb[lgb_13].iloc[tr_idx], y_train[tr_idx])
        p_lgb_oof[va_idx] = m_lgb_f.predict_proba(train_xgb[lgb_13].iloc[va_idx])[:, 1]

        m_xgb_f = xgb.XGBClassifier(**xgb_params).fit(train_xgb[xgb_cols].iloc[tr_idx], y_train[tr_idx])
        p_xgb_oof[va_idx] = m_xgb_f.predict_proba(train_xgb[xgb_cols].iloc[va_idx])[:, 1]

        m_rf_f = RandomForestClassifier(**rf_params).fit(train_rf[rf_cols].iloc[tr_idx], y_train[tr_idx])
        p_rf_oof[va_idx] = m_rf_f.predict_proba(train_rf[rf_cols].iloc[va_idx])[:, 1]

    print(f"  * OOF PR-AUC -> LGBM(13): {pr_auc(y_train, p_lgb_oof):.5f} | XGB: {pr_auc(y_train, p_xgb_oof):.5f} | RF: {pr_auc(y_train, p_rf_oof):.5f}")

    # =========================================================================
    # NIVEL 2: Construcción de Meta-Dataset y Metamodelos
    # =========================================================================
    print("\n6. Entrenando Metamodelos de Nivel 2...")
    eps = 1e-7

    def build_meta(p_lgb_vec, p_xgb_vec, p_rf_vec, part_df):
        d = {
            "p_lgb": p_lgb_vec,
            "logit_lgb": np.log(np.clip(p_lgb_vec, eps, 1 - eps) / (1 - np.clip(p_lgb_vec, eps, 1 - eps))),
            "p_xgb": p_xgb_vec,
            "logit_xgb": np.log(np.clip(p_xgb_vec, eps, 1 - eps) / (1 - np.clip(p_xgb_vec, eps, 1 - eps))),
            "p_rf": p_rf_vec,
            "logit_rf": np.log(np.clip(p_rf_vec, eps, 1 - eps) / (1 - np.clip(p_rf_vec, eps, 1 - eps))),
        }
        for col in anom_cols:
            d[col] = part_df[col].to_numpy(dtype=float)
        d["pct_heuristico_fraude"] = part_df["pct_heuristico_fraude"].to_numpy(dtype=float)
        d["distancia_log_cluster"] = part_df["distancia_log_cluster"].to_numpy(dtype=float)
        d["deficit_piso_cluster"] = part_df["deficit_piso_cluster"].to_numpy(dtype=float)
        return pd.DataFrame(d)

    X_meta_tr = build_meta(p_lgb_oof, p_xgb_oof, p_rf_oof, train)
    X_meta_tu = build_meta(p_lgb_13_tu, p_xgb_tu, p_rf_tu, tune)
    X_meta_ca = build_meta(p_lgb_13_ca, p_xgb_ca, p_rf_ca, cal)
    X_meta_po = build_meta(p_lgb_13_po, p_xgb_po, p_rf_po, policy)
    X_meta_te = build_meta(p_lgb_13_te, p_xgb_te, p_rf_te, test)

    meta_cols = list(X_meta_tr.columns)
    scaler = StandardScaler().fit(X_meta_tr)
    X_tr_sc = pd.DataFrame(scaler.transform(X_meta_tr), columns=meta_cols)
    X_tu_sc = pd.DataFrame(scaler.transform(X_meta_tu), columns=meta_cols)
    X_ca_sc = pd.DataFrame(scaler.transform(X_meta_ca), columns=meta_cols)
    X_po_sc = pd.DataFrame(scaler.transform(X_meta_po), columns=meta_cols)
    X_te_sc = pd.DataFrame(scaler.transform(X_meta_te), columns=meta_cols)

    # Modelos Nivel 2
    p_avg_tu = (p_lgb_13_tu + p_xgb_tu + p_rf_tu) / 3.0
    p_avg_te = (p_lgb_13_te + p_xgb_te + p_rf_te) / 3.0

    m_l2 = LogisticRegression(C=1.0, max_iter=1000, random_state=seed).fit(X_tr_sc, y_train)
    p_l2_tu = m_l2.predict_proba(X_tu_sc)[:, 1]
    p_l2_te = m_l2.predict_proba(X_te_sc)[:, 1]

    m_l1 = LogisticRegression(penalty="l1", solver="saga", C=0.1, max_iter=1000, random_state=seed).fit(X_tr_sc, y_train)
    p_l1_tu = m_l1.predict_proba(X_tu_sc)[:, 1]
    p_l1_te = m_l1.predict_proba(X_te_sc)[:, 1]

    print("\n" + "=" * 80)
    print("CUADRO COMPARATIVO FINAL — VALIDACIÓN vs TEST")
    print("=" * 80)
    models_summary = [
        ("Control Ian: LightGBM 11 vars", pr_auc(y_tune, p_lgb_11_tu), pr_auc(y_test, p_lgb_11_te)),
        ("Challenger v3: LightGBM 13 vars (+Clustering)", pr_auc(y_tune, p_lgb_13_tu), pr_auc(y_test, p_lgb_13_te)),
        ("Base XGBoost (con Forward Selection)", pr_auc(y_tune, p_xgb_tu), pr_auc(y_test, p_xgb_te)),
        ("Base Random Forest (con Forward Selection)", pr_auc(y_tune, p_rf_tu), pr_auc(y_test, p_rf_te)),
        ("Ensamble Equiponderado (LGB+XGB+RF)", pr_auc(y_tune, p_avg_tu), pr_auc(y_test, p_avg_te)),
        ("🏆 Stacking Tri-Modelo + Clustering (Meta-LR L2)", pr_auc(y_tune, p_l2_tu), pr_auc(y_test, p_l2_te)),
        ("Stacking Tri-Modelo + Clustering (Meta-LR Lasso)", pr_auc(y_tune, p_l1_tu), pr_auc(y_test, p_l1_te)),
    ]

    print(f"{'Modelo / Arquitectura':<50s} | {'val_search':<12s} | {'test final':<12s} | {'Delta vs Ian Test'}")
    print("-" * 90)
    ian_test = pr_auc(y_test, p_lgb_11_te)
    for name, v_sc, t_sc in models_summary:
        print(f"{name:<50s} | {v_sc:.5f}      | {t_sc:.5f}      | {t_sc - ian_test:+.5f}")

    print(f"\n[OK] Experimento concluido en {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
