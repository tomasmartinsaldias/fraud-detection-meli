"""
Experimento Cuatri-Modelo (LGBM + XGBoost + CatBoost + Random Forest)
con Features Enriquecidas de Catálogo y Target Encoding Libre de Leakage.
========================================================================
Objetivo: Superar 0.50 en Validación (val_search) y consolidar >0.45 en Test final.

Palancas implementadas:
1. Palanca 3: Dispersión IQR normalizada por cluster y z-score de desvío de catálogo.
2. Palanca 4: Smooth Target Encoding de (o x es_madrugada) estrictamente ajustado en Train (m=25).
3. Palanca 2: CatBoost (árboles simétricos) integrado en Nivel 1 para romper correlación.
4. Palanca 1: Blend Convexo Óptimo (SLSQP / Nelder-Mead) + Meta-LR Lasso L1 en Nivel 2.
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

from catboost import CatBoostClassifier
import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from scipy.optimize import minimize
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


def compute_cluster_features(df_train: pd.DataFrame, df_all: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Genera variables de cluster semántico, piso, dispersión IQR y desvío normalizado estrictamente en Train."""
    print("  -> Ajustando clustering semántico k=50 y estadísticas de dispersión en Train...")
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
    
    # Estadísticos robustos por cluster
    def calc_iqr(x):
        q75, q25 = np.percentile(x, 75), np.percentile(x, 25)
        med = np.median(x)
        return (q75 - q25) / (med + 1e-4)

    stats = temp_train.groupby("cluster_i")["monto"].agg(
        median="median",
        count="count",
        q10=lambda x: np.percentile(x, 10),
        iqr_norm=calc_iqr,
    )

    global_median = float(df_train["monto"].median())
    global_iqr = float(calc_iqr(df_train["monto"].values))
    m_prior = 10.0

    stats["bayes_median"] = (stats["count"] * stats["median"] + m_prior * global_median) / (stats["count"] + m_prior)
    median_map = stats["bayes_median"].to_dict()
    iqr_map = stats["iqr_norm"].to_dict()

    all_medians = np.array([median_map.get(c, global_median) for c in all_clusters], dtype=float)
    all_iqrs = np.array([iqr_map.get(c, global_iqr) for c in all_clusters], dtype=float)
    all_montos = df_all["monto"].values.astype(float)

    dist_log = (np.log1p(all_montos) - np.log1p(all_medians)).astype(float)
    def_piso = np.maximum(0.0, (all_medians - all_montos) / (all_medians + 1e-4)).astype(float)
    z_cluster = (dist_log / (all_iqrs + 0.1)).astype(float)

    return dist_log, def_piso, all_iqrs, z_cluster


def compute_smooth_target_encoding(df_train: pd.DataFrame, df_all: pd.DataFrame, m: float = 25.0) -> np.ndarray:
    """Calcula Target Encoding con m-smoothing de (o x es_madrugada) estrictamente derivado de Train."""
    print(f"  -> Calculando Smooth Target Encoding (o x es_madrugada) en Train con m={m}...")
    global_rate = float(df_train["fraude"].mean())

    train_group = df_train.groupby(["o", "es_madrugada"]).agg(
        total=("fraude", "count"),
        fraudes=("fraude", "sum"),
    )
    # Suavizado bayesiano
    train_group["smooth_risk"] = (train_group["fraudes"] + m * global_rate) / (train_group["total"] + m)
    risk_dict = train_group["smooth_risk"].to_dict()

    te_values = np.array([
        risk_dict.get((o_val, madr_val), global_rate)
        for o_val, madr_val in zip(df_all["o"], df_all["es_madrugada"])
    ], dtype=float)
    return te_values


def optimize_blend_weights(y_val: np.ndarray, preds_list: List[np.ndarray]) -> np.ndarray:
    """Optimiza pesos convexos (sum=1, w>=0) para maximizar PR-AUC en validación."""
    n_models = len(preds_list)
    init_weights = np.ones(n_models) / n_models

    def loss(weights):
        w = np.clip(weights, 0, None)
        if w.sum() == 0:
            return 1.0
        w = w / w.sum()
        blend = sum(w[i] * preds_list[i] for i in range(n_models))
        return -pr_auc(y_val, blend)

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, 1.0) for _ in range(n_models)]

    res = minimize(loss, init_weights, method="SLSQP", bounds=bounds, constraints=constraints)
    w_opt = np.clip(res.x, 0, None)
    w_opt = w_opt / w_opt.sum()
    return w_opt


def main():
    seed = 42
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    t0 = time.time()

    print("=" * 85)
    print("OBJETIVO 0.50: CUATRI-MODELO (LGB + XGB + CATBOOST + RF) + ENRIQUECIMIENTO")
    print("=" * 85)

    # 1. Cargar bases
    print("\n1. Cargando datasets base, transaccionales, anomalías y reglas...")
    base_df = pd.read_parquet(ROOT / "data/processed/dataset_base.parquet")
    feat_df = pd.read_parquet(ROOT / "data/features/features_transaccionales.parquet")
    anom_df = pd.read_parquet(ROOT / "data/features/scores_anomalias.parquet")
    rules_df = pd.read_parquet(ROOT / "data/features/features_reglas_asociacion.parquet")[["transaction_id", "pct_heuristico_fraude"]]

    exclude_anom = ["score_anomalia_consolidado", "anomalia_x_monto", "extended_iforest_x_monto"]
    anom_cols = [c for c in anom_df.columns if c not in ["transaction_id"] + exclude_anom]

    if "score_heuristico_fraude_10" in feat_df.columns:
        feat_df = feat_df.drop(columns=["score_heuristico_fraude_10"])

    # 2. Generación de Nuevas Features (Palancas 3 y 4)
    train_mask = base_df["split"] == "train"
    dist_log, def_piso, iqr_cluster, z_cluster = compute_cluster_features(base_df[train_mask], base_df)
    base_df["distancia_log_cluster"] = dist_log
    base_df["deficit_piso_cluster"] = def_piso
    base_df["dispersion_iqr_cluster"] = iqr_cluster
    base_df["z_cluster_precio"] = z_cluster

    # Merge inicial
    df = (
        base_df
        .merge(feat_df, on="transaction_id")
        .merge(anom_df[["transaction_id"] + anom_cols], on="transaction_id")
        .merge(rules_df, on="transaction_id")
    )
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values(["fecha", "transaction_id"]).reset_index(drop=True)

    # Palanca 4: Smooth Target Encoding (o x es_madrugada)
    te_madr = compute_smooth_target_encoding(df[df["split"] == "train"], df, m=25.0)
    df["te_o_x_madrugada"] = te_madr

    train = df.loc[df.split == "train"].copy()
    val_full = df.loc[df.split == "val"].copy()
    test = df.loc[df.split == "test"].copy()
    tune, cal, policy = validation_blocks(val_full)

    y_train = train.fraude.to_numpy()
    y_tune = tune.fraude.to_numpy()
    y_cal = cal.fraude.to_numpy()
    y_policy = policy.fraude.to_numpy()
    y_test = test.fraude.to_numpy()

    # Predecir variables y codificación
    df_xgb = df.copy()
    for cat in BASE_CATEGORICAL_COLS:
        if cat in df_xgb.columns:
            df_xgb[cat] = df_xgb[cat].astype("category")

    ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df_rf = df.copy()
    cat_present = [c for c in BASE_CATEGORICAL_COLS if c in df_rf.columns]
    df_rf[cat_present] = ord_enc.fit_transform(df_rf[cat_present])

    candidate_cols = [c for c in df.columns if c not in ["transaction_id", "split", "fraude", "fecha", "i"]]
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
    # MODELO 1: LightGBM (15 variables con enriquecimiento)
    # =========================================================================
    print("\n2. Entrenando Modelo 1: LightGBM Enriquecido...")
    lgb_cols = [
        "score", "o", "g", "a", "es_madrugada", "ratio_d_m", "ratio_monto_l",
        "prod_b_log_c", "score_texto_nb", "ratio_monto_e", "is_h_cero",
        "deficit_piso_cluster", "distancia_log_cluster", "z_cluster_precio", "te_o_x_madrugada"
    ]
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
    p_lgb_tu = m_lgb.predict_proba(tune_xgb[lgb_cols])[:, 1]
    p_lgb_ca = m_lgb.predict_proba(cal_xgb[lgb_cols])[:, 1]
    p_lgb_po = m_lgb.predict_proba(policy_xgb[lgb_cols])[:, 1]
    p_lgb_te = m_lgb.predict_proba(test_xgb[lgb_cols])[:, 1]
    print(f"  * LightGBM ({len(lgb_cols)} vars) -> val_search: {pr_auc(y_tune, p_lgb_tu):.5f} | test: {pr_auc(y_test, p_lgb_te):.5f}")

    # =========================================================================
    # MODELO 2: XGBoost Enriquecido
    # =========================================================================
    print("\n3. Entrenando Modelo 2: XGBoost...")
    xgb_cols = [
        'score', 'o', 'te_j_bayes', 'es_madrugada', 'ratio_monto_l', 'diff_m_d',
        'score_texto_nb', 'is_h_cero', 'flag_alto_riesgo_nop', 'ratio_monto_b',
        'g', 'b', 'distancia_log_cluster', 'te_o_x_madrugada'
    ]
    best_params_xgb = {
        "learning_rate": 0.052,
        "max_depth": 5,
        "min_child_weight": 35,
        "colsample_bytree": 0.80,
        "subsample": 0.80,
        "reg_alpha": 0.1,
        "reg_lambda": 5.0,
        "n_estimators": 180,
        "tree_method": "hist",
        "enable_categorical": True,
        "random_state": seed,
        "n_jobs": 4,
        "verbosity": 0,
    }
    m_xgb = xgb.XGBClassifier(**best_params_xgb).fit(train_xgb[xgb_cols], y_train)
    p_xgb_tu = m_xgb.predict_proba(tune_xgb[xgb_cols])[:, 1]
    p_xgb_ca = m_xgb.predict_proba(cal_xgb[xgb_cols])[:, 1]
    p_xgb_po = m_xgb.predict_proba(policy_xgb[xgb_cols])[:, 1]
    p_xgb_te = m_xgb.predict_proba(test_xgb[xgb_cols])[:, 1]
    print(f"  * XGBoost ({len(xgb_cols)} vars)  -> val_search: {pr_auc(y_tune, p_xgb_tu):.5f} | test: {pr_auc(y_test, p_xgb_te):.5f}")

    # =========================================================================
    # MODELO 3: CatBoost (Palanca 2 — Árboles Simétricos Oblivious)
    # =========================================================================
    print("\n4. Entrenando Modelo 3: CatBoost (Árboles Simétricos)...")
    cat_features = ["a", "g", "n", "o", "p"]
    # Para CatBoost, rellenamos NaNs en categóricas como string "MISSING"
    df_cat = df.copy()
    for cat in cat_features:
        df_cat[cat] = df_cat[cat].fillna("MISSING").astype(str)

    cat_cols = [
        "score", "monto", "b", "c", "d", "e", "f", "h", "l", "m",
        "a", "g", "n", "o", "p",
        "ratio_d_m", "ratio_monto_l", "prod_b_log_c", "es_madrugada",
        "distancia_log_cluster", "deficit_piso_cluster", "te_o_x_madrugada"
    ]
    cat_cat_indices = [cat_cols.index(c) for c in cat_features if c in cat_cols]

    train_cat = df_cat.loc[df.split == "train"]
    tune_cat = df_cat.loc[tune.index]
    cal_cat = df_cat.loc[cal.index]
    policy_cat = df_cat.loc[policy.index]
    test_cat = df_cat.loc[test.index]

    m_cat = CatBoostClassifier(
        iterations=280,
        learning_rate=0.045,
        depth=6,
        l2_leaf_reg=6.0,
        cat_features=cat_cat_indices,
        random_seed=seed,
        thread_count=4,
        verbose=0,
    )
    m_cat.fit(train_cat[cat_cols], y_train, eval_set=(tune_cat[cat_cols], y_tune), early_stopping_rounds=30)
    p_cat_tu = m_cat.predict_proba(tune_cat[cat_cols])[:, 1]
    p_cat_ca = m_cat.predict_proba(cal_cat[cat_cols])[:, 1]
    p_cat_po = m_cat.predict_proba(policy_cat[cat_cols])[:, 1]
    p_cat_te = m_cat.predict_proba(test_cat[cat_cols])[:, 1]
    print(f"  * CatBoost ({len(cat_cols)} vars) -> val_search: {pr_auc(y_tune, p_cat_tu):.5f} | test: {pr_auc(y_test, p_cat_te):.5f}")

    # =========================================================================
    # MODELO 4: Random Forest
    # =========================================================================
    print("\n5. Entrenando Modelo 4: Random Forest...")
    rf_cols = ['score', 'o', 'te_j_bayes', 'es_madrugada', 'diff_m_d', 'distancia_log_cluster', 'te_o_x_madrugada']
    m_rf = RandomForestClassifier(n_estimators=80, max_depth=10, min_samples_leaf=30, max_features=0.7, random_state=seed, n_jobs=4)
    m_rf.fit(train_rf[rf_cols], y_train)
    p_rf_tu = m_rf.predict_proba(tune_rf[rf_cols])[:, 1]
    p_rf_ca = m_rf.predict_proba(cal_rf[rf_cols])[:, 1]
    p_rf_po = m_rf.predict_proba(policy_rf[rf_cols])[:, 1]
    p_rf_te = m_rf.predict_proba(test_rf[rf_cols])[:, 1]
    print(f"  * Random Forest ({len(rf_cols)} vars) -> val_search: {pr_auc(y_tune, p_rf_tu):.5f} | test: {pr_auc(y_test, p_rf_te):.5f}")

    # =========================================================================
    # CORRELACIÓN ENTRE LOS 4 MODELOS BASE
    # =========================================================================
    preds_val_df = pd.DataFrame({"LGB": p_lgb_tu, "XGB": p_xgb_tu, "CAT": p_cat_tu, "RF": p_rf_tu})
    spearman = preds_val_df.corr(method="spearman").round(4)
    print("\n--- MATRIZ DE CORRELACIÓN DE SPEARMAN (RANK) EN VAL_SEARCH ---")
    print(spearman)

    # =========================================================================
    # 5-FOLD STRATIFIED CV PARA GENERAR OOF EN TRAIN
    # =========================================================================
    print("\n6. Generando predicciones Out-Of-Fold (OOF) con 5-Fold Stratified CV...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    p_lgb_oof = np.zeros(len(train), dtype=float)
    p_xgb_oof = np.zeros(len(train), dtype=float)
    p_cat_oof = np.zeros(len(train), dtype=float)
    p_rf_oof = np.zeros(len(train), dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y_train), 1):
        # LGB
        m_lgb_f = lgb_model(best_params_lgb, seed, 4).fit(train_xgb[lgb_cols].iloc[tr_idx], y_train[tr_idx])
        p_lgb_oof[va_idx] = m_lgb_f.predict_proba(train_xgb[lgb_cols].iloc[va_idx])[:, 1]
        # XGB
        m_xgb_f = xgb.XGBClassifier(**best_params_xgb).fit(train_xgb[xgb_cols].iloc[tr_idx], y_train[tr_idx])
        p_xgb_oof[va_idx] = m_xgb_f.predict_proba(train_xgb[xgb_cols].iloc[va_idx])[:, 1]
        # CAT
        m_cat_f = CatBoostClassifier(iterations=250, learning_rate=0.045, depth=6, l2_leaf_reg=6.0, cat_features=cat_cat_indices, random_seed=seed, thread_count=4, verbose=0)
        m_cat_f.fit(train_cat[cat_cols].iloc[tr_idx], y_train[tr_idx])
        p_cat_oof[va_idx] = m_cat_f.predict_proba(train_cat[cat_cols].iloc[va_idx])[:, 1]
        # RF
        m_rf_f = RandomForestClassifier(n_estimators=80, max_depth=10, min_samples_leaf=30, max_features=0.7, random_state=seed, n_jobs=4)
        m_rf_f.fit(train_rf[rf_cols].iloc[tr_idx], y_train[tr_idx])
        p_rf_oof[va_idx] = m_rf_f.predict_proba(train_rf[rf_cols].iloc[va_idx])[:, 1]

    print(f"  * OOF PR-AUC -> LGB: {pr_auc(y_train, p_lgb_oof):.5f} | XGB: {pr_auc(y_train, p_xgb_oof):.5f} | CAT: {pr_auc(y_train, p_cat_oof):.5f} | RF: {pr_auc(y_train, p_rf_oof):.5f}")

    # =========================================================================
    # NIVEL 2: METAMODELOS (Blend Convexo Óptimo + Lasso L1)
    # =========================================================================
    print("\n7. Evaluando Metamodelos de Nivel 2...")

    # A) Blend Convexo Óptimo (SLSQP)
    w_opt = optimize_blend_weights(y_tune, [p_lgb_tu, p_xgb_tu, p_cat_tu, p_rf_tu])
    print(f"  * Pesos Óptimos Blend Convexo -> LGB: {w_opt[0]:.3f} | XGB: {w_opt[1]:.3f} | CAT: {w_opt[2]:.3f} | RF: {w_opt[3]:.3f}")

    p_blend_tu = w_opt[0] * p_lgb_tu + w_opt[1] * p_xgb_tu + w_opt[2] * p_cat_tu + w_opt[3] * p_rf_tu
    p_blend_te = w_opt[0] * p_lgb_te + w_opt[1] * p_xgb_te + w_opt[2] * p_cat_te + w_opt[3] * p_rf_te

    # B) Meta-Dataset para Lasso y Ridge
    eps = 1e-7
    def build_meta(p1, p2, p3, p4, part_df):
        d = {
            "p_lgb": p1, "logit_lgb": np.log(np.clip(p1, eps, 1-eps) / (1-np.clip(p1, eps, 1-eps))),
            "p_xgb": p2, "logit_xgb": np.log(np.clip(p2, eps, 1-eps) / (1-np.clip(p2, eps, 1-eps))),
            "p_cat": p3, "logit_cat": np.log(np.clip(p3, eps, 1-eps) / (1-np.clip(p3, eps, 1-eps))),
            "p_rf": p4,  "logit_rf": np.log(np.clip(p4, eps, 1-eps) / (1-np.clip(p4, eps, 1-eps))),
        }
        for col in anom_cols:
            d[col] = part_df[col].to_numpy(dtype=float)
        d["pct_heuristico_fraude"] = part_df["pct_heuristico_fraude"].to_numpy(dtype=float)
        d["distancia_log_cluster"] = part_df["distancia_log_cluster"].to_numpy(dtype=float)
        d["deficit_piso_cluster"] = part_df["deficit_piso_cluster"].to_numpy(dtype=float)
        d["z_cluster_precio"] = part_df["z_cluster_precio"].to_numpy(dtype=float)
        d["te_o_x_madrugada"] = part_df["te_o_x_madrugada"].to_numpy(dtype=float)
        return pd.DataFrame(d)

    X_meta_tr = build_meta(p_lgb_oof, p_xgb_oof, p_cat_oof, p_rf_oof, train)
    X_meta_tu = build_meta(p_lgb_tu, p_xgb_tu, p_cat_tu, p_rf_tu, tune)
    X_meta_te = build_meta(p_lgb_te, p_xgb_te, p_cat_te, p_rf_te, test)

    meta_cols = list(X_meta_tr.columns)
    scaler = StandardScaler().fit(X_meta_tr)
    X_tr_sc = pd.DataFrame(scaler.transform(X_meta_tr), columns=meta_cols)
    X_tu_sc = pd.DataFrame(scaler.transform(X_meta_tu), columns=meta_cols)
    X_te_sc = pd.DataFrame(scaler.transform(X_meta_te), columns=meta_cols)

    # Lasso L1 (C=0.1)
    m_lasso = LogisticRegression(solver="saga", l1_ratio=1.0, C=0.1, max_iter=1000, random_state=seed).fit(X_tr_sc, y_train)
    p_lasso_tu = m_lasso.predict_proba(X_tu_sc)[:, 1]
    p_lasso_te = m_lasso.predict_proba(X_te_sc)[:, 1]

    # Ridge L2 (C=1.0)
    m_ridge = LogisticRegression(C=1.0, max_iter=1000, random_state=seed).fit(X_tr_sc, y_train)
    p_ridge_tu = m_ridge.predict_proba(X_tu_sc)[:, 1]
    p_ridge_te = m_ridge.predict_proba(X_te_sc)[:, 1]

    # Ensamble Equiponderado simple
    p_avg_tu = (p_lgb_tu + p_xgb_tu + p_cat_tu + p_rf_tu) / 4.0
    p_avg_te = (p_lgb_te + p_xgb_te + p_cat_te + p_rf_te) / 4.0

    print("\n" + "=" * 90)
    print("TABLA COMPARATIVA FINAL — VALIDACIÓN vs TEST (OBJETIVO 0.50 / 0.45)")
    print("=" * 90)

    models_summary = [
        ("Control Ian: LightGBM 11 vars", 0.48136, 0.44435),
        ("Franco Campeón Previo: Tri-Modelo (H4)", 0.48887, 0.44450),
        ("LightGBM Enriquecido (15 vars)", pr_auc(y_tune, p_lgb_tu), pr_auc(y_test, p_lgb_te)),
        ("XGBoost Enriquecido (14 vars)", pr_auc(y_tune, p_xgb_tu), pr_auc(y_test, p_xgb_te)),
        ("CatBoost Nuevo (Palanca 2 Oblivious)", pr_auc(y_tune, p_cat_tu), pr_auc(y_test, p_cat_te)),
        ("Random Forest Enriquecido (7 vars)", pr_auc(y_tune, p_rf_tu), pr_auc(y_test, p_rf_te)),
        ("Promedio Equiponderado (4 modelos)", pr_auc(y_tune, p_avg_tu), pr_auc(y_test, p_avg_te)),
        ("Blend Convexo Óptimo SLSQP (Palanca 1)", pr_auc(y_tune, p_blend_tu), pr_auc(y_test, p_blend_te)),
        ("🏆 Stacking Cuatri-Modelo (Meta-LR Lasso L1)", pr_auc(y_tune, p_lasso_tu), pr_auc(y_test, p_lasso_te)),
        ("Stacking Cuatri-Modelo (Meta-LR Ridge L2)", pr_auc(y_tune, p_ridge_tu), pr_auc(y_test, p_ridge_te)),
    ]

    print(f"{'Modelo / Arquitectura':<45s} | {'val_search':<12s} | {'test final':<12s} | {'Delta vs Ian Test'}")
    print("-" * 90)
    for name, v_sc, t_sc in models_summary:
        print(f"{name:<45s} | {v_sc:.5f}      | {t_sc:.5f}      | {t_sc - 0.44435:+.5f}")

    print(f"\n[OK] Experimento concluido en {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
