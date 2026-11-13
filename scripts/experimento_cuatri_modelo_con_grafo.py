"""
Experimento Maestro: Cuatri-Modelo Stacking con Señal de Grafo Bipartito (in_degree_usuarios_producto_hist)
===========================================================================================================
- Nivel 1: LightGBM, XGBoost, CatBoost y Random Forest
- Feature agregada: in_degree_usuarios_producto_hist (co-ocurrencia causal acumulada de usuarios por producto)
- Fine-Tuning: CMA-ES (Optuna CmaEsSampler) sobre val_search (tune)
- Extracción OOF: 5-Fold Stratified CV determinista en train (105k)
- Nivel 2: Metamodelo Decisor con regularización Lasso L1 (C=0.10 y C=0.20), Ridge L2 y Blend SLSQP
- Evaluación ciego out-of-time: Comparativa directa contra el Champion Oficial y Récord de Parsimonia
"""

from __future__ import annotations
import json
import logging
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from catboost import CatBoostClassifier
import lightgbm as lgb
import numpy as np
import optuna
from optuna.samplers import CmaEsSampler
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
import xgboost as xgb

from src.models.supervised import logit, pr_auc, validation_blocks
from src.models.train import lgb_model

optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_CATEGORICAL_COLS = ["a", "g", "n", "o", "p"]
BASE_NUMERIC_COLS = ["score", "monto", "b", "c", "d", "e", "f", "h", "l", "m", "q"]


def prob_to_logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p_clipped = np.clip(p, eps, 1.0 - eps)
    return np.log(p_clipped / (1.0 - p_clipped))


def build_clustering_features(df_train: pd.DataFrame, df_all: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Genera deficit_piso_cluster, distancia_log_cluster y ratio_monto_mediana_cluster ajustados en Train con SVD d=30."""
    print("  -> Generando clustering semántico k=50 (SVD d=30 L2-norm) sobre títulos...")
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=5, sublinear_tf=True, strip_accents="unicode")
    X_tr_tfidf = vec.fit_transform(df_train["i"].fillna(""))
    svd = TruncatedSVD(n_components=30, random_state=42)
    X_tr_svd = normalize(svd.fit_transform(X_tr_tfidf))

    kmeans = MiniBatchKMeans(n_clusters=50, random_state=42, batch_size=2048)
    train_clusters = kmeans.fit_predict(X_tr_svd)

    X_all_tfidf = vec.transform(df_all["i"].fillna(""))
    X_all_svd = normalize(svd.transform(X_all_tfidf))
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
    ratio_monto_mediana = (all_montos / (all_medians + 1e-4)).astype(float)
    return distancia_log, deficit_piso, ratio_monto_mediana


def compute_in_degree_product(df: pd.DataFrame) -> np.ndarray:
    """Calcula in_degree_usuarios_producto_hist: usuarios distintos acumulados previos por producto."""
    print("  -> Generando variable de grafo bipartito in_degree_usuarios_producto_hist...")
    guest_mask = (df["r"] == 260445) | (df["c"].isna())
    effective_user = np.where(guest_mask, np.nan, df["r"])

    prod_seen_users_hist = {}
    in_deg_hist = np.zeros(len(df), dtype=np.int32)
    i_vals = df["i"].values

    for idx in range(len(df)):
        prod = i_vals[idx]
        u = effective_user[idx]
        seen = prod_seen_users_hist.get(prod)
        if seen is None:
            prod_seen_users_hist[prod] = {u} if pd.notna(u) else set()
            in_deg_hist[idx] = 0
        else:
            in_deg_hist[idx] = len(seen)
            if pd.notna(u):
                seen.add(u)

    return in_deg_hist


def optimize_blend_weights(y_val: np.ndarray, preds_list: List[np.ndarray]) -> np.ndarray:
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

    print("=" * 95)
    print("EXPERIMENTO MAESTRO: CUATRI-MODELO STACKING CON GRAFO BIPARTITO")
    print("OPTIMIZACIÓN CMA-ES + 5-FOLD CV OOF + NIVEL 2 CON PARSIMONIA LASSO L1")
    print("=" * 95)

    # 1. Cargar bases oficiales
    print("\n1. Cargando datos...")
    base_df = pd.read_parquet(ROOT / "data/processed/dataset_base.parquet")
    feat_df = pd.read_parquet(ROOT / "data/features/features_transaccionales.parquet")
    anom_df = pd.read_parquet(ROOT / "data/features/scores_anomalias.parquet")

    exclude_anom = ["score_anomalia_consolidado", "anomalia_x_monto", "extended_iforest_x_monto"]
    anom_cols = [c for c in anom_df.columns if c not in ["transaction_id"] + exclude_anom]

    if "score_heuristico_fraude_10" in feat_df.columns:
        feat_df = feat_df.drop(columns=["score_heuristico_fraude_10"])

    # 2. Generar Clustering Semántico v3 ajustado en Train
    train_mask_raw = base_df["split"] == "train"
    dist_log, def_piso, ratio_med = build_clustering_features(base_df[train_mask_raw], base_df)
    base_df["distancia_log_cluster"] = dist_log
    base_df["deficit_piso_cluster"] = def_piso
    base_df["ratio_monto_mediana_cluster"] = ratio_med

    # 3. Generar la señal de grafo bipartito
    base_df["in_degree_usuarios_producto_hist"] = compute_in_degree_product(base_df)

    # 4. Mergear datasets
    df = (
        base_df
        .merge(feat_df, on="transaction_id")
        .merge(anom_df[["transaction_id"] + anom_cols], on="transaction_id")
    )
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values(["fecha", "transaction_id"]).reset_index(drop=True)

    train = df.loc[df.split == "train"].copy()
    val_full = df.loc[df.split == "val"].copy()
    test = df.loc[df.split == "test"].copy()
    tune, cal, policy = validation_blocks(val_full)

    y_train = train.fraude.to_numpy()
    y_tune = tune.fraude.to_numpy()
    y_test = test.fraude.to_numpy()

    # Predecir categorías unificadas para XGBoost
    df_xgb = df.copy()
    for cat in BASE_CATEGORICAL_COLS:
        cats = sorted(df[cat].dropna().astype(str).unique())
        df_xgb[cat] = pd.Categorical(df_xgb[cat].astype(str), categories=cats)

    # Preparar Random Forest
    ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df_rf = df.copy()
    cat_present = [c for c in BASE_CATEGORICAL_COLS if c in df_rf.columns]
    df_rf[cat_present] = ord_enc.fit_transform(df_rf[cat_present])
    candidate_cols = [c for c in df.columns if c not in ["transaction_id", "split", "fraude", "fecha", "i"]]
    for c in candidate_cols:
        if df_rf[c].isna().any():
            med = df_rf.loc[df_rf.split == "train", c].median()
            df_rf[c] = df_rf[c].fillna(med)

    # Preparar CatBoost
    df_cat = df.copy()
    for cat in BASE_CATEGORICAL_COLS:
        df_cat[cat] = df_cat[cat].fillna("MISSING").astype(str)

    train_xgb = df_xgb.loc[df.split == "train"]
    tune_xgb = df_xgb.loc[tune.index]
    test_xgb = df_xgb.loc[test.index]

    train_rf = df_rf.loc[df.split == "train"]
    tune_rf = df_rf.loc[tune.index]
    test_rf = df_rf.loc[test.index]

    train_cat = df_cat.loc[df.split == "train"]
    tune_cat = df_cat.loc[tune.index]
    test_cat = df_cat.loc[test.index]

    # Definición de features para cada modelo
    lgb_cols = [
        "score", "o", "g", "a", "es_madrugada", "ratio_d_m", "ratio_monto_l",
        "prod_b_log_c", "score_texto_nb", "ratio_monto_e", "is_h_cero",
        "deficit_piso_cluster", "distancia_log_cluster", "ratio_monto_mediana_cluster",
        "in_degree_usuarios_producto_hist"
    ]
    xgb_cols = [
        "score", "o", "te_j_bayes", "es_madrugada", "ratio_monto_l", "diff_m_d",
        "score_texto_nb", "is_h_cero", "flag_alto_riesgo_nop", "ratio_monto_b", "g", "b",
        "in_degree_usuarios_producto_hist"
    ]
    cat_cols = list(lgb_cols)
    cat_indices = [cat_cols.index(c) for c in BASE_CATEGORICAL_COLS if c in cat_cols]
    rf_cols = [
        "score", "o", "te_j_bayes", "es_madrugada", "diff_m_d", "distancia_log_cluster",
        "in_degree_usuarios_producto_hist"
    ]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    print("\n" + "=" * 95)
    print("2. ENTRENAMIENTO Y FINE-TUNING CON CMA-ES DE LOS 4 INDUCTORES (NIVEL 1)")
    print("=" * 95)

    # =========================================================================
    # MODELO 1: LightGBM (15 variables con Grafo) + CMA-ES
    # =========================================================================
    print("\n--- [MODELO 1/4: LightGBM con in_degree_usuarios_producto_hist] ---")
    def obj_lgb(trial):
        p = {
            "learning_rate": trial.suggest_float("lr", 0.02, 0.06),
            "num_leaves": trial.suggest_int("nl", 20, 36),
            "min_child_samples": trial.suggest_int("mc", 70, 140),
            "colsample_bytree": trial.suggest_float("col", 0.70, 0.90),
            "subsample": trial.suggest_float("sub", 0.70, 0.90),
            "reg_alpha": trial.suggest_float("ra", 1e-3, 0.5, log=True),
            "reg_lambda": trial.suggest_float("rl", 1e-2, 3.0, log=True),
            "n_estimators": 240, "objective": "binary", "verbosity": -1, "random_state": seed, "n_jobs": 4,
        }
        m = lgb.LGBMClassifier(**p).fit(train_xgb[lgb_cols], y_train)
        return pr_auc(y_tune, m.predict_proba(tune_xgb[lgb_cols])[:, 1])

    s_lgb = optuna.create_study(direction="maximize", sampler=CmaEsSampler(seed=seed))
    s_lgb.optimize(obj_lgb, n_trials=8)
    bp_lgb = s_lgb.best_params
    p_lgb = {
        "learning_rate": bp_lgb["lr"], "num_leaves": bp_lgb["nl"], "min_child_samples": bp_lgb["mc"],
        "colsample_bytree": bp_lgb["col"], "subsample": bp_lgb["sub"], "reg_alpha": bp_lgb["ra"],
        "reg_lambda": bp_lgb["rl"], "n_estimators": 250, "objective": "binary", "verbosity": -1, "random_state": seed, "n_jobs": 4,
    }
    m_lgb = lgb.LGBMClassifier(**p_lgb).fit(train_xgb[lgb_cols], y_train)
    p_lgb_tu = m_lgb.predict_proba(tune_xgb[lgb_cols])[:, 1]
    p_lgb_te = m_lgb.predict_proba(test_xgb[lgb_cols])[:, 1]
    print(f"  * LightGBM ({len(lgb_cols)} vars) -> val_search: {pr_auc(y_tune, p_lgb_tu):.5f} | test: {pr_auc(y_test, p_lgb_te):.5f}")

    print("    Generando OOF 5-Fold para LightGBM...")
    p_lgb_oof = np.zeros(len(train), dtype=float)
    for tr_i, va_i in skf.split(train, y_train):
        m_f = lgb.LGBMClassifier(**p_lgb).fit(train_xgb[lgb_cols].iloc[tr_i], y_train[tr_i])
        p_lgb_oof[va_i] = m_f.predict_proba(train_xgb[lgb_cols].iloc[va_i])[:, 1]

    # =========================================================================
    # MODELO 2: XGBoost (13 variables con Grafo) + CMA-ES
    # =========================================================================
    print("\n--- [MODELO 2/4: XGBoost con in_degree_usuarios_producto_hist] ---")
    def obj_xgb(trial):
        p = {
            "learning_rate": trial.suggest_float("lr", 0.03, 0.07),
            "max_depth": trial.suggest_int("md", 4, 6),
            "min_child_weight": trial.suggest_int("mcw", 25, 50),
            "colsample_bytree": trial.suggest_float("col", 0.70, 0.90),
            "subsample": trial.suggest_float("sub", 0.70, 0.90),
            "reg_alpha": trial.suggest_float("ra", 0.05, 1.0, log=True),
            "reg_lambda": trial.suggest_float("rl", 1.0, 10.0, log=True),
            "n_estimators": 180, "tree_method": "hist", "enable_categorical": True,
            "random_state": seed, "n_jobs": 4, "verbosity": 0,
        }
        m = xgb.XGBClassifier(**p).fit(train_xgb[xgb_cols], y_train)
        return pr_auc(y_tune, m.predict_proba(tune_xgb[xgb_cols])[:, 1])

    s_xgb = optuna.create_study(direction="maximize", sampler=CmaEsSampler(seed=seed))
    s_xgb.optimize(obj_xgb, n_trials=8)
    bp_xgb = s_xgb.best_params
    p_xgb = {
        "learning_rate": bp_xgb["lr"], "max_depth": bp_xgb["md"], "min_child_weight": bp_xgb["mcw"],
        "colsample_bytree": bp_xgb["col"], "subsample": bp_xgb["sub"], "reg_alpha": bp_xgb["ra"],
        "reg_lambda": bp_xgb["rl"], "n_estimators": 180, "tree_method": "hist", "enable_categorical": True,
        "random_state": seed, "n_jobs": 4, "verbosity": 0,
    }
    m_xgb = xgb.XGBClassifier(**p_xgb).fit(train_xgb[xgb_cols], y_train)
    p_xgb_tu = m_xgb.predict_proba(tune_xgb[xgb_cols])[:, 1]
    p_xgb_te = m_xgb.predict_proba(test_xgb[xgb_cols])[:, 1]
    print(f"  * XGBoost ({len(xgb_cols)} vars)  -> val_search: {pr_auc(y_tune, p_xgb_tu):.5f} | test: {pr_auc(y_test, p_xgb_te):.5f}")

    print("    Generando OOF 5-Fold para XGBoost...")
    p_xgb_oof = np.zeros(len(train), dtype=float)
    for tr_i, va_i in skf.split(train, y_train):
        m_f = xgb.XGBClassifier(**p_xgb).fit(train_xgb[xgb_cols].iloc[tr_i], y_train[tr_i])
        p_xgb_oof[va_i] = m_f.predict_proba(train_xgb[xgb_cols].iloc[va_i])[:, 1]

    # =========================================================================
    # MODELO 3: CatBoost (15 variables con Grafo)
    # =========================================================================
    print("\n--- [MODELO 3/4: CatBoost con in_degree_usuarios_producto_hist] ---")
    p_cat = {
        "iterations": 260,
        "learning_rate": 0.045,
        "depth": 6,
        "l2_leaf_reg": 6.0,
        "cat_features": cat_indices,
        "random_seed": seed,
        "thread_count": 4,
        "verbose": 0,
    }
    m_cat = CatBoostClassifier(**p_cat).fit(train_cat[cat_cols], y_train, eval_set=(tune_cat[cat_cols], y_tune), early_stopping_rounds=30)
    p_cat_tu = m_cat.predict_proba(tune_cat[cat_cols])[:, 1]
    p_cat_te = m_cat.predict_proba(test_cat[cat_cols])[:, 1]
    print(f"  * CatBoost ({len(cat_cols)} vars) -> val_search: {pr_auc(y_tune, p_cat_tu):.5f} | test: {pr_auc(y_test, p_cat_te):.5f}")

    print("    Generando OOF 5-Fold para CatBoost...")
    p_cat_oof = np.zeros(len(train), dtype=float)
    for tr_i, va_i in skf.split(train, y_train):
        m_f = CatBoostClassifier(iterations=240, learning_rate=0.045, depth=6, l2_leaf_reg=6.0, cat_features=cat_indices, random_seed=seed, thread_count=4, verbose=0)
        m_f.fit(train_cat[cat_cols].iloc[tr_i], y_train[tr_i])
        p_cat_oof[va_i] = m_f.predict_proba(train_cat[cat_cols].iloc[va_i])[:, 1]

    # =========================================================================
    # MODELO 4: Random Forest (7 variables con Grafo)
    # =========================================================================
    print("\n--- [MODELO 4/4: Random Forest con in_degree_usuarios_producto_hist] ---")
    p_rf = {
        "n_estimators": 80,
        "max_depth": 10,
        "min_samples_leaf": 30,
        "max_features": 0.7,
        "random_state": seed,
        "n_jobs": 4,
    }
    m_rf = RandomForestClassifier(**p_rf).fit(train_rf[rf_cols], y_train)
    p_rf_tu = m_rf.predict_proba(tune_rf[rf_cols])[:, 1]
    p_rf_te = m_rf.predict_proba(test_rf[rf_cols])[:, 1]
    print(f"  * Random Forest ({len(rf_cols)} vars) -> val_search: {pr_auc(y_tune, p_rf_tu):.5f} | test: {pr_auc(y_test, p_rf_te):.5f}")

    print("    Generando OOF 5-Fold para Random Forest...")
    p_rf_oof = np.zeros(len(train), dtype=float)
    for tr_i, va_i in skf.split(train, y_train):
        m_f = RandomForestClassifier(**p_rf).fit(train_rf[rf_cols].iloc[tr_i], y_train[tr_i])
        p_rf_oof[va_i] = m_f.predict_proba(train_rf[rf_cols].iloc[va_i])[:, 1]

    print("\n  [RESUMEN OOF EN TRAIN]:")
    print(f"    * LGBM OOF: {pr_auc(y_train, p_lgb_oof):.5f}")
    print(f"    * XGB  OOF: {pr_auc(y_train, p_xgb_oof):.5f}")
    print(f"    * CAT  OOF: {pr_auc(y_train, p_cat_oof):.5f}")
    print(f"    * RF   OOF: {pr_auc(y_train, p_rf_oof):.5f}")

    # =========================================================================
    # 3. CONSTRUCCIÓN DE DATASET DE NIVEL 2 Y EVALUACIÓN DE STACKING
    # =========================================================================
    print("\n" + "=" * 95)
    print("3. CONSTRUCCIÓN DEL META-DATASET Y ENTRENAMIENTO DE NIVEL 2")
    print("=" * 95)

    def build_meta(p1, p2, p3, p4, part_df):
        d = {
            "p_lgb": p1, "logit_lgb": prob_to_logit(p1),
            "p_xgb": p2, "logit_xgb": prob_to_logit(p2),
            "p_cat": p3, "logit_cat": prob_to_logit(p3),
            "p_rf": p4,  "logit_rf": prob_to_logit(p4),
        }
        for col in anom_cols:
            d[col] = part_df[col].to_numpy(dtype=float)
        d["pct_heuristico_fraude"] = part_df["pct_heuristico_fraude"].to_numpy(dtype=float)
        d["distancia_log_cluster"] = part_df["distancia_log_cluster"].to_numpy(dtype=float)
        d["deficit_piso_cluster"] = part_df["deficit_piso_cluster"].to_numpy(dtype=float)
        d["in_degree_usuarios_producto_hist"] = part_df["in_degree_usuarios_producto_hist"].to_numpy(dtype=float)
        return pd.DataFrame(d)

    X_meta_tr = build_meta(p_lgb_oof, p_xgb_oof, p_cat_oof, p_rf_oof, train)
    X_meta_tu = build_meta(p_lgb_tu, p_xgb_tu, p_cat_tu, p_rf_tu, tune)
    X_meta_te = build_meta(p_lgb_te, p_xgb_te, p_cat_te, p_rf_te, test)

    meta_cols = list(X_meta_tr.columns)
    scaler = StandardScaler().fit(X_meta_tr)
    X_tr_sc = pd.DataFrame(scaler.transform(X_meta_tr), columns=meta_cols)
    X_tu_sc = pd.DataFrame(scaler.transform(X_meta_tu), columns=meta_cols)
    X_te_sc = pd.DataFrame(scaler.transform(X_meta_te), columns=meta_cols)

    # A. Stacking con Lasso L1 C=0.10 (Champion de Parsimonia)
    m_lasso_01 = LogisticRegression(solver="saga", l1_ratio=1.0, C=0.10, max_iter=1000, random_state=seed).fit(X_tr_sc, y_train)
    p_l1_01_tu = m_lasso_01.predict_proba(X_tu_sc)[:, 1]
    p_l1_01_te = m_lasso_01.predict_proba(X_te_sc)[:, 1]

    # B. Stacking con Lasso L1 C=0.20
    m_lasso_02 = LogisticRegression(solver="saga", l1_ratio=1.0, C=0.20, max_iter=1000, random_state=seed).fit(X_tr_sc, y_train)
    p_l1_02_tu = m_lasso_02.predict_proba(X_tu_sc)[:, 1]
    p_l1_02_te = m_lasso_02.predict_proba(X_te_sc)[:, 1]

    # C. Stacking con Ridge L2 C=0.20
    m_ridge_02 = LogisticRegression(solver="saga", penalty="l2", C=0.20, max_iter=1000, random_state=seed).fit(X_tr_sc, y_train)
    p_r_02_tu = m_ridge_02.predict_proba(X_tu_sc)[:, 1]
    p_r_02_te = m_ridge_02.predict_proba(X_te_sc)[:, 1]

    # D. Blend Convexo Óptimo (SLSQP)
    w_opt = optimize_blend_weights(y_tune, [p_lgb_tu, p_xgb_tu, p_cat_tu, p_rf_tu])
    p_blend_tu = w_opt[0] * p_lgb_tu + w_opt[1] * p_xgb_tu + w_opt[2] * p_cat_tu + w_opt[3] * p_rf_tu
    p_blend_te = w_opt[0] * p_lgb_te + w_opt[1] * p_xgb_te + w_opt[2] * p_cat_te + w_opt[3] * p_rf_te

    # E. Stacking Puro de Nivel 2 (Solo Logits/Preds)
    pred_cols = ["p_lgb", "logit_lgb", "p_xgb", "logit_xgb", "p_cat", "logit_cat", "p_rf", "logit_rf"]
    m_pure = LogisticRegression(solver="saga", l1_ratio=1.0, C=0.20, max_iter=1000, random_state=seed).fit(X_tr_sc[pred_cols], y_train)
    p_pure_tu = m_pure.predict_proba(X_tu_sc[pred_cols])[:, 1]
    p_pure_te = m_pure.predict_proba(X_te_sc[pred_cols])[:, 1]

    # =========================================================================
    # 4. TABLERO COMPARATIVO FINAL CONTRA HITOS LÍDERES
    # =========================================================================
    print("\n" + "=" * 105)
    print("RESUMEN COMPARATIVO: CUATRI-MODELO CON GRAFO BIPARTITO VS HISTÓRICOS DEL PROYECTO")
    print("=" * 105)

    summary_table = [
        ("0. Control Ian (LightGBM 11 vars)", 0.48136, 0.44435, "+0.0 bps", "Baseline FE"),
        ("1. Challenger v3 Oficial (LGBM 13 vars)", 0.48850, 0.44700, "+26.5 bps", "Semántica"),
        ("2. Champion Oficial (Cuatri-Modelo C=0.20)", 0.49218, 0.45014, "+57.9 bps", "Champion N2"),
        ("3. Récord Parsimonia Previo (Cuatri L1 C=0.10)", 0.49101, 0.45093, "+65.8 bps", "🏆 RECORD PREVIO"),
        ("4. Forward Selection Multi-Loss (5 modelos)", 0.49502, 0.45010, "+57.5 bps", "Multi-Loss"),
        ("-" * 45, 0.0, 0.0, "", ""),
        ("5. Nuevo LightGBM + Grafo Bipartito", pr_auc(y_tune, p_lgb_tu), pr_auc(y_test, p_lgb_te), f"{(pr_auc(y_test, p_lgb_te)-0.44435)*10000:+.1f} bps", "N1 Inductor"),
        ("6. Nuevo XGBoost + Grafo Bipartito", pr_auc(y_tune, p_xgb_tu), pr_auc(y_test, p_xgb_te), f"{(pr_auc(y_test, p_xgb_te)-0.44435)*10000:+.1f} bps", "N1 Inductor"),
        ("7. Nuevo CatBoost + Grafo Bipartito", pr_auc(y_tune, p_cat_tu), pr_auc(y_test, p_cat_te), f"{(pr_auc(y_test, p_cat_te)-0.44435)*10000:+.1f} bps", "N1 Inductor"),
        ("8. Nuevo Random Forest + Grafo Bipartito", pr_auc(y_tune, p_rf_tu), pr_auc(y_test, p_rf_te), f"{(pr_auc(y_test, p_rf_te)-0.44435)*10000:+.1f} bps", "N1 Inductor"),
        ("9. Nuevo Blend Convexo SLSQP con Grafo", pr_auc(y_tune, p_blend_tu), pr_auc(y_test, p_blend_te), f"{(pr_auc(y_test, p_blend_te)-0.44435)*10000:+.1f} bps", "Ensamble"),
        ("10. Stacking Cuatri-Modelo Puro (Solo Preds)", pr_auc(y_tune, p_pure_tu), pr_auc(y_test, p_pure_te), f"{(pr_auc(y_test, p_pure_te)-0.44435)*10000:+.1f} bps", "N2 Puro"),
        ("11. Stacking Cuatri + Grafo (Ridge L2 C=0.20)", pr_auc(y_tune, p_r_02_tu), pr_auc(y_test, p_r_02_te), f"{(pr_auc(y_test, p_r_02_te)-0.44435)*10000:+.1f} bps", "N2 Ridge"),
        ("12. 🏆 Stacking Cuatri + Grafo (Lasso L1 C=0.20)", pr_auc(y_tune, p_l1_02_tu), pr_auc(y_test, p_l1_02_te), f"{(pr_auc(y_test, p_l1_02_te)-0.44435)*10000:+.1f} bps", "N2 Lasso"),
        ("13. 🏆 Stacking Cuatri + Grafo (Lasso L1 C=0.10)", pr_auc(y_tune, p_l1_01_tu), pr_auc(y_test, p_l1_01_te), f"{(pr_auc(y_test, p_l1_01_te)-0.44435)*10000:+.1f} bps", "N2 Parsimonia"),
    ]

    print(f"{'Hito / Arquitectura':<47s} | {'PR-AUC Val':<12s} | {'PR-AUC Test':<12s} | {'Delta vs Ian':<14s} | {'Rol / Estado'}")
    print("-" * 105)
    for row in summary_table:
        if row[0].startswith("-"):
            print("-" * 105)
            continue
        name, v_sc, t_sc, d_ian, rol = row
        print(f"{name:<47s} | {v_sc:.5f}      | {t_sc:.5f}      | {d_ian:<14s} | {rol}")

    # Coeficientes del metamodelo ganador
    print("\n" + "=" * 95)
    print("COEFICIENTES DEL METAMODELO LASSO L1 (C=0.10) EN NIVEL 2:")
    print("=" * 95)
    weights = pd.Series(m_lasso_01.coef_[0], index=meta_cols).sort_values(ascending=False)
    for feat, w in weights.items():
        if abs(w) > 1e-4:
            print(f"  * {feat:<35s}: {w:+.5f}")
        else:
            print(f"  * {feat:<35s}:  0.00000 (APAGADO POR L1)")

    print(f"\n[OK] Experimento completado exitosamente en {time.time() - t0:.1f} segundos.")


if __name__ == "__main__":
    main()
