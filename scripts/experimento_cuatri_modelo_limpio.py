"""
Experimento Definitivo: Stacking Cuatri-Modelo Limpio (LGB + XGB + CatBoost + RF)
================================================================================
Aplica el principio de Forward Selection:
- Conserva estrictamente las 13 variables limpias de Challenger v3.
- Descarta te_o_x_madrugada y dispersion_iqr (comprobadas ruidosas en ablación).
- Nivel 1: LightGBM (13 vars), XGBoost (12 vars), CatBoost (13 vars), Random Forest (6 vars).
- Nivel 2: Metamodelo Lasso L1 (probabilidades OOF + 11 anomalías + pct_heuristico + clustering).
- Evalúa val_search, val_calibration, val_policy y test final.
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
import lightgbm as lgb
import numpy as np
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

from src.models.supervised import (
    calibrate, cost_curve, logit, metrics, policy_metrics, pr_auc,
    validation_blocks,
)
from src.models.train import lgb_model


BASE_CATEGORICAL_COLS = ["a", "g", "n", "o", "p"]
BASE_NUMERIC_COLS = ["score", "monto", "b", "c", "d", "e", "f", "h", "l", "m", "q"]


def build_clustering_features(df_train: pd.DataFrame, df_all: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Genera deficit_piso_cluster, distancia_log_cluster y ratio_monto_mediana_cluster estrictamente ajustados en Train con SVD d=30."""
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

    print("=" * 85)
    print("STACKING CUATRI-MODELO LIMPIO (LGBM + XGBOOST + CATBOOST + RF)")
    print("ENTRENADO ESTRICTAMENTE SOBRE LAS 13 VARIABLES DE CHALLENGER V3")
    print("=" * 85)

    # 1. Cargar bases
    print("\n1. Cargando datos...")
    base_df = pd.read_parquet(ROOT / "data/processed/dataset_base.parquet")
    feat_df = pd.read_parquet(ROOT / "data/features/features_transaccionales.parquet")
    anom_df = pd.read_parquet(ROOT / "data/features/scores_anomalias.parquet")
    rules_df = pd.read_parquet(ROOT / "data/features/features_reglas_asociacion.parquet")[["transaction_id", "pct_heuristico_fraude"]]

    exclude_anom = ["score_anomalia_consolidado", "anomalia_x_monto", "extended_iforest_x_monto"]
    anom_cols = [c for c in anom_df.columns if c not in ["transaction_id"] + exclude_anom]

    if "score_heuristico_fraude_10" in feat_df.columns:
        feat_df = feat_df.drop(columns=["score_heuristico_fraude_10"])

    # 2. Generar Clustering Semántico v3 estrictamente ajustado en Train
    train_mask = base_df["split"] == "train"
    dist_log, def_piso, ratio_med = build_clustering_features(base_df[train_mask], base_df)
    base_df["distancia_log_cluster"] = dist_log
    base_df["deficit_piso_cluster"] = def_piso
    base_df["ratio_monto_mediana_cluster"] = ratio_med

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

    # Predecir categorías unificadas
    df_xgb = df.copy()
    for cat in BASE_CATEGORICAL_COLS:
        cats = sorted(df[cat].dropna().astype(str).unique())
        df_xgb[cat] = pd.Categorical(df_xgb[cat].astype(str), categories=cats)

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

    # Preparar CatBoost
    df_cat = df.copy()
    for cat in BASE_CATEGORICAL_COLS:
        df_cat[cat] = df_cat[cat].fillna("MISSING").astype(str)
    train_cat = df_cat.loc[df.split == "train"]
    tune_cat = df_cat.loc[tune.index]
    cal_cat = df_cat.loc[cal.index]
    policy_cat = df_cat.loc[policy.index]
    test_cat = df_cat.loc[test.index]

    # =========================================================================
    # MODELO 1: LightGBM (14 variables de Challenger v3 con ratio_monto_mediana_cluster)
    # =========================================================================
    print("\n2. Entrenando Modelo 1: LightGBM Challenger v3 (14 variables)...")
    lgb_14 = [
        "score", "o", "g", "a", "es_madrugada", "ratio_d_m", "ratio_monto_l",
        "prod_b_log_c", "score_texto_nb", "ratio_monto_e", "is_h_cero",
        "deficit_piso_cluster", "distancia_log_cluster", "ratio_monto_mediana_cluster"
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
    m_lgb = lgb_model(best_params_lgb, seed, 4).fit(train_xgb[lgb_14], y_train)
    p_lgb_tu = m_lgb.predict_proba(tune_xgb[lgb_14])[:, 1]
    p_lgb_ca = m_lgb.predict_proba(cal_xgb[lgb_14])[:, 1]
    p_lgb_po = m_lgb.predict_proba(policy_xgb[lgb_14])[:, 1]
    p_lgb_te = m_lgb.predict_proba(test_xgb[lgb_14])[:, 1]
    print(f"  * LightGBM (14 vars) -> val_search: {pr_auc(y_tune, p_lgb_tu):.5f} | test: {pr_auc(y_test, p_lgb_te):.5f}")

    # =========================================================================
    # MODELO 2: XGBoost (12 variables)
    # =========================================================================
    print("\n3. Entrenando Modelo 2: XGBoost...")
    xgb_cols = ['score', 'o', 'te_j_bayes', 'es_madrugada', 'ratio_monto_l', 'diff_m_d', 'score_texto_nb', 'is_h_cero', 'flag_alto_riesgo_nop', 'ratio_monto_b', 'g', 'b']
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
    # MODELO 3: CatBoost (con 14 variables de Challenger v3)
    # =========================================================================
    print("\n4. Entrenando Modelo 3: CatBoost...")
    cat_cols = [c for c in lgb_14]
    cat_indices = [cat_cols.index(c) for c in BASE_CATEGORICAL_COLS if c in cat_cols]

    m_cat = CatBoostClassifier(
        iterations=280,
        learning_rate=0.045,
        depth=6,
        l2_leaf_reg=6.0,
        cat_features=cat_indices,
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
    # MODELO 4: Random Forest (6 variables con clustering)
    # =========================================================================
    print("\n5. Entrenando Modelo 4: Random Forest...")
    rf_cols = ['score', 'o', 'te_j_bayes', 'es_madrugada', 'diff_m_d', 'distancia_log_cluster']
    m_rf = RandomForestClassifier(n_estimators=80, max_depth=10, min_samples_leaf=30, max_features=0.7, random_state=seed, n_jobs=4)
    m_rf.fit(train_rf[rf_cols], y_train)
    p_rf_tu = m_rf.predict_proba(tune_rf[rf_cols])[:, 1]
    p_rf_ca = m_rf.predict_proba(cal_rf[rf_cols])[:, 1]
    p_rf_po = m_rf.predict_proba(policy_rf[rf_cols])[:, 1]
    p_rf_te = m_rf.predict_proba(test_rf[rf_cols])[:, 1]
    print(f"  * Random Forest ({len(rf_cols)} vars) -> val_search: {pr_auc(y_tune, p_rf_tu):.5f} | test: {pr_auc(y_test, p_rf_te):.5f}")

    # =========================================================================
    # OOF PREDICTIONS (5-Fold Stratified CV en Train)
    # =========================================================================
    print("\n6. Generando probabilidades OOF con 5-Fold Stratified CV...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    p_lgb_oof = np.zeros(len(train), dtype=float)
    p_xgb_oof = np.zeros(len(train), dtype=float)
    p_cat_oof = np.zeros(len(train), dtype=float)
    p_rf_oof = np.zeros(len(train), dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y_train), 1):
        # LGB
        m_lgb_f = lgb_model(best_params_lgb, seed, 4).fit(train_xgb[lgb_14].iloc[tr_idx], y_train[tr_idx])
        p_lgb_oof[va_idx] = m_lgb_f.predict_proba(train_xgb[lgb_14].iloc[va_idx])[:, 1]
        # XGB
        m_xgb_f = xgb.XGBClassifier(**best_params_xgb).fit(train_xgb[xgb_cols].iloc[tr_idx], y_train[tr_idx])
        p_xgb_oof[va_idx] = m_xgb_f.predict_proba(train_xgb[xgb_cols].iloc[va_idx])[:, 1]
        # CAT
        m_cat_f = CatBoostClassifier(iterations=250, learning_rate=0.045, depth=6, l2_leaf_reg=6.0, cat_features=cat_indices, random_seed=seed, thread_count=4, verbose=0)
        m_cat_f.fit(train_cat[cat_cols].iloc[tr_idx], y_train[tr_idx])
        p_cat_oof[va_idx] = m_cat_f.predict_proba(train_cat[cat_cols].iloc[va_idx])[:, 1]
        # RF
        m_rf_f = RandomForestClassifier(n_estimators=80, max_depth=10, min_samples_leaf=30, max_features=0.7, random_state=seed, n_jobs=4)
        m_rf_f.fit(train_rf[rf_cols].iloc[tr_idx], y_train[tr_idx])
        p_rf_oof[va_idx] = m_rf_f.predict_proba(train_rf[rf_cols].iloc[va_idx])[:, 1]

    print(f"  * OOF PR-AUC -> LGB: {pr_auc(y_train, p_lgb_oof):.5f} | XGB: {pr_auc(y_train, p_xgb_oof):.5f} | CAT: {pr_auc(y_train, p_cat_oof):.5f} | RF: {pr_auc(y_train, p_rf_oof):.5f}")

    # =========================================================================
    # NIVEL 2: Construcción de Meta-Dataset
    # =========================================================================
    print("\n7. Construyendo Meta-Dataset de Nivel 2 y evaluando Stacking...")
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
        d["ratio_monto_mediana_cluster"] = part_df["ratio_monto_mediana_cluster"].to_numpy(dtype=float)
        return pd.DataFrame(d)

    X_meta_tr = build_meta(p_lgb_oof, p_xgb_oof, p_cat_oof, p_rf_oof, train)
    X_meta_tu = build_meta(p_lgb_tu, p_xgb_tu, p_cat_tu, p_rf_tu, tune)
    X_meta_te = build_meta(p_lgb_te, p_xgb_te, p_cat_te, p_rf_te, test)

    meta_cols = list(X_meta_tr.columns)
    scaler = StandardScaler().fit(X_meta_tr)
    X_tr_sc = pd.DataFrame(scaler.transform(X_meta_tr), columns=meta_cols)
    X_tu_sc = pd.DataFrame(scaler.transform(X_meta_tu), columns=meta_cols)
    X_te_sc = pd.DataFrame(scaler.transform(X_meta_te), columns=meta_cols)

    # 1. Lasso L1 C=0.1
    m_lasso_01 = LogisticRegression(solver="saga", l1_ratio=1.0, C=0.1, max_iter=1000, random_state=seed).fit(X_tr_sc, y_train)
    p_l1_tu = m_lasso_01.predict_proba(X_tu_sc)[:, 1]
    p_l1_te = m_lasso_01.predict_proba(X_te_sc)[:, 1]

    # 2. Lasso L1 C=0.2
    m_lasso_02 = LogisticRegression(solver="saga", l1_ratio=1.0, C=0.2, max_iter=1000, random_state=seed).fit(X_tr_sc, y_train)
    p_l2_tu = m_lasso_02.predict_proba(X_tu_sc)[:, 1]
    p_l2_te = m_lasso_02.predict_proba(X_te_sc)[:, 1]

    # 3. Ridge L2 C=1.0
    m_ridge = LogisticRegression(C=1.0, max_iter=1000, random_state=seed).fit(X_tr_sc, y_train)
    p_r_tu = m_ridge.predict_proba(X_tu_sc)[:, 1]
    p_r_te = m_ridge.predict_proba(X_te_sc)[:, 1]

    # 4. Blend Convexo SLSQP
    w_opt = optimize_blend_weights(y_tune, [p_lgb_tu, p_xgb_tu, p_cat_tu, p_rf_tu])
    p_blend_tu = w_opt[0] * p_lgb_tu + w_opt[1] * p_xgb_tu + w_opt[2] * p_cat_tu + w_opt[3] * p_rf_tu
    p_blend_te = w_opt[0] * p_lgb_te + w_opt[1] * p_xgb_te + w_opt[2] * p_cat_te + w_opt[3] * p_rf_te

    print("\n" + "=" * 95)
    print("RESULTADOS FINALES LIMPIOS: CUATRI-MODELO CON VARIABLES PURAS DE CHALLENGER V3")
    print("=" * 95)
    models_summary = [
        ("Control Ian: LightGBM 11 vars", 0.48136, 0.44435),
        ("Challenger v3 Oficial: LightGBM 13 vars", pr_auc(y_tune, p_lgb_tu), pr_auc(y_test, p_lgb_te)),
        ("XGBoost Limpio (12 vars)", pr_auc(y_tune, p_xgb_tu), pr_auc(y_test, p_xgb_te)),
        ("CatBoost Limpio (13 vars Challenger v3)", pr_auc(y_tune, p_cat_tu), pr_auc(y_test, p_cat_te)),
        ("Random Forest Limpio (6 vars)", pr_auc(y_tune, p_rf_tu), pr_auc(y_test, p_rf_te)),
        ("Blend Convexo Óptimo (SLSQP)", pr_auc(y_tune, p_blend_tu), pr_auc(y_test, p_blend_te)),
        ("Stacking Cuatri-Modelo (Ridge L2 C=1.0)", pr_auc(y_tune, p_r_tu), pr_auc(y_test, p_r_te)),
        ("🏆 Stacking Cuatri-Modelo Limpio (Lasso L1 C=0.1)", pr_auc(y_tune, p_l1_tu), pr_auc(y_test, p_l1_te)),
        ("🏆 Stacking Cuatri-Modelo Limpio (Lasso L1 C=0.2)", pr_auc(y_tune, p_l2_tu), pr_auc(y_test, p_l2_te)),
    ]

    print(f"{'Modelo / Arquitectura':<52s} | {'val_search':<12s} | {'test final':<12s} | {'Delta vs Ian Test'}")
    print("-" * 95)
    for name, v_sc, t_sc in models_summary:
        print(f"{name:<52s} | {v_sc:.5f}      | {t_sc:.5f}      | {t_sc - 0.44435:+.5f}")

    print(f"\n[OK] Experimento concluido en {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
