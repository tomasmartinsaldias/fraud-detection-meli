"""
Script de Comprobación: Impacto de Regularización L1 vs L2 en Ensamble Multi-Loss
Evalúa el efecto de podar coeficientes con Lasso L1 (C=0.05, 0.10, 0.20) vs Ridge L2
"""

from __future__ import annotations
import json
from pathlib import Path
import os
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from catboost import CatBoostRanker, Pool
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
import xgboost as xgb

from src.models.supervised import logit, pr_auc, validation_blocks

def prob_to_logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p_clipped = np.clip(p, eps, 1.0 - eps)
    return np.log(p_clipped / (1.0 - p_clipped))

BASE_CATEGORICAL_COLS = ["a", "g", "n", "o", "p"]

def make_lgb_focal_loss(gamma=2.0, alpha=0.5):
    def focal_obj(y_true, y_pred):
        p = 1.0 / (1.0 + np.exp(-np.clip(y_pred, -25.0, 25.0)))
        pt = np.where(y_true == 1, p, 1.0 - p)
        alpha_t = np.where(y_true == 1, alpha, 1.0 - alpha)
        focal_weight = alpha_t * ((1.0 - pt) ** gamma)
        grad = focal_weight * (p - y_true)
        hess = np.maximum(focal_weight * p * (1.0 - p), 1e-5)
        return grad, hess
    return focal_obj

def make_xgb_focal_loss(gamma=2.0, alpha=0.5):
    def xgb_focal_obj(y_true, y_pred):
        p = 1.0 / (1.0 + np.exp(-np.clip(y_pred, -25.0, 25.0)))
        pt = np.where(y_true == 1, p, 1.0 - p)
        alpha_t = np.where(y_true == 1, alpha, 1.0 - alpha)
        focal_weight = alpha_t * ((1.0 - pt) ** gamma)
        grad = focal_weight * (p - y_true)
        hess = np.maximum(focal_weight * p * (1.0 - p), 1e-5)
        return grad, hess
    return xgb_focal_obj

def normalize_scores_to_margin(scores):
    std = np.std(scores)
    return (scores - np.mean(scores)) / (std if std > 1e-6 else 1.0)

def main():
    seed = 42
    print("Cargando datos...")
    base_df = pd.read_parquet(ROOT / "data/processed/dataset_base.parquet")
    feat_df = pd.read_parquet(ROOT / "data/features/features_transaccionales.parquet")
    anom_df = pd.read_parquet(ROOT / "data/features/scores_anomalias.parquet")
    rules_df = pd.read_parquet(ROOT / "data/features/features_reglas_asociacion.parquet")[["transaction_id", "pct_heuristico_fraude"]]

    exclude_anom = ["score_anomalia_consolidado", "anomalia_x_monto", "extended_iforest_x_monto"]
    anom_cols = [c for c in anom_df.columns if c not in ["transaction_id"] + exclude_anom]
    if "score_heuristico_fraude_10" in feat_df.columns:
        feat_df = feat_df.drop(columns=["score_heuristico_fraude_10"])

    # Clustering semántico
    train_mask = base_df["split"] == "train"
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=5, sublinear_tf=True, strip_accents="unicode")
    X_tr_tfidf = vec.fit_transform(base_df[train_mask]["i"].fillna(""))
    svd = TruncatedSVD(n_components=30, random_state=42)
    kmeans = MiniBatchKMeans(n_clusters=50, random_state=42, batch_size=2048)
    kmeans.fit(normalize(svd.fit_transform(X_tr_tfidf)))

    all_clusters = kmeans.predict(normalize(svd.transform(vec.transform(base_df["i"].fillna("")))))
    temp_train = pd.DataFrame({"cluster_i": all_clusters[train_mask], "monto": base_df[train_mask]["monto"].values})
    cluster_stats = temp_train.groupby("cluster_i")["monto"].agg(["median", "count"])
    global_median = float(base_df[train_mask]["monto"].median())
    cluster_stats["bayes_median"] = (cluster_stats["count"] * cluster_stats["median"] + 10.0 * global_median) / (cluster_stats["count"] + 10.0)
    median_map = cluster_stats["bayes_median"].to_dict()
    all_medians = np.array([median_map.get(c, global_median) for c in all_clusters], dtype=float)
    all_montos = base_df["monto"].values.astype(float)

    base_df["distancia_log_cluster"] = (np.log1p(all_montos) - np.log1p(all_medians)).astype(float)
    base_df["deficit_piso_cluster"] = np.maximum(0.0, (all_medians - all_montos) / (all_medians + 1e-4)).astype(float)

    df = base_df.merge(feat_df, on="transaction_id").merge(anom_df[["transaction_id"] + anom_cols], on="transaction_id").merge(rules_df, on="transaction_id")
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values(["fecha", "transaction_id"]).reset_index(drop=True)

    train = df.loc[df.split == "train"].copy()
    val_full = df.loc[df.split == "val"].copy()
    test = df.loc[df.split == "test"].copy()
    tune, cal, policy = validation_blocks(val_full)

    y_train = train.fraude.to_numpy()
    y_tune = tune.fraude.to_numpy()
    y_test = test.fraude.to_numpy()

    # Features para modelos
    lgb_14 = ["score", "monto", "b", "c", "d", "e", "f", "h", "l", "m", "q", "ratio_monto_e", "score_texto_nb", "ratio_monto_l"]
    xgb_cols = lgb_14 + ["ratio_d_m", "prod_b_log_c", "es_madrugada", "is_h_cero"]
    cat_cols = list(xgb_cols)
    cat_indices = [cat_cols.index(c) for c in ["es_madrugada", "is_h_cero"] if c in cat_cols]

    # Predecir categorías para XGBoost
    df_xgb = df.copy()
    for cat in BASE_CATEGORICAL_COLS:
        cats = sorted(df[cat].dropna().astype(str).unique())
        df_xgb[cat] = pd.Categorical(df_xgb[cat].astype(str), categories=cats)

    train_xgb = df_xgb.loc[df_xgb.split == "train"]
    tune_xgb = df_xgb.loc[tune.index]
    test_xgb = df_xgb.loc[df_xgb.split == "test"]

    train_cat = df.loc[df.split == "train"]
    tune_cat = df.loc[tune.index]
    test_cat = df.loc[df.split == "test"]

    date_series = train["fecha"].dt.date
    train_qid_xgb = date_series.map({d: i for i, d in enumerate(date_series.unique())}).values

    print("Entrenando los 5 modelos ganadores del Forward Selection...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # 1. XGB_LogLoss
    p_xgb_ll = {"learning_rate": 0.05, "max_depth": 5, "min_child_weight": 35, "colsample_bytree": 0.8, "subsample": 0.8, "n_estimators": 180, "tree_method": "hist", "random_state": seed, "n_jobs": 4, "verbosity": 0}
    m1 = xgb.XGBClassifier(**p_xgb_ll).fit(train_xgb[xgb_cols], y_train)
    tu_xgb_ll = prob_to_logit(m1.predict_proba(tune_xgb[xgb_cols])[:, 1])
    te_xgb_ll = prob_to_logit(m1.predict_proba(test_xgb[xgb_cols])[:, 1])
    oof_xgb_ll = np.zeros(len(train))
    for tr_i, va_i in skf.split(train, y_train):
        m = xgb.XGBClassifier(**p_xgb_ll).fit(train_xgb[xgb_cols].iloc[tr_i], y_train[tr_i])
        oof_xgb_ll[va_i] = prob_to_logit(m.predict_proba(train_xgb[xgb_cols].iloc[va_i])[:, 1])

    # 2. LGB_Focal
    p_lgb_fc = {"learning_rate": 0.04, "num_leaves": 31, "min_child_samples": 80, "colsample_bytree": 0.8, "subsample": 0.8, "n_estimators": 220, "objective": make_lgb_focal_loss(2.0, 0.5), "verbosity": -1, "random_state": seed, "n_jobs": 4}
    m2 = lgb.LGBMClassifier(**p_lgb_fc).fit(train_xgb[lgb_14], y_train)
    tu_lgb_fc = m2.predict_proba(tune_xgb[lgb_14])
    te_lgb_fc = m2.predict_proba(test_xgb[lgb_14])
    oof_lgb_fc = np.zeros(len(train))
    for tr_i, va_i in skf.split(train, y_train):
        m = lgb.LGBMClassifier(**p_lgb_fc).fit(train_xgb[lgb_14].iloc[tr_i], y_train[tr_i])
        oof_lgb_fc[va_i] = m.predict_proba(train_xgb[lgb_14].iloc[va_i])

    # 3. XGB_Focal
    p_xgb_fc = {"learning_rate": 0.05, "max_depth": 5, "min_child_weight": 35, "colsample_bytree": 0.8, "subsample": 0.8, "n_estimators": 180, "objective": make_xgb_focal_loss(2.0, 0.5), "tree_method": "hist", "random_state": seed, "n_jobs": 4, "verbosity": 0}
    m3 = xgb.XGBClassifier(**p_xgb_fc).fit(train_xgb[xgb_cols], y_train)
    tu_xgb_fc = m3.predict(tune_xgb[xgb_cols], output_margin=True)
    te_xgb_fc = m3.predict(test_xgb[xgb_cols], output_margin=True)
    oof_xgb_fc = np.zeros(len(train))
    for tr_i, va_i in skf.split(train, y_train):
        m = xgb.XGBClassifier(**p_xgb_fc).fit(train_xgb[xgb_cols].iloc[tr_i], y_train[tr_i])
        oof_xgb_fc[va_i] = m.predict(train_xgb[xgb_cols].iloc[va_i], output_margin=True)

    # 4. CAT_Ranker (YetiRank)
    p_cat_rk = {"iterations": 180, "learning_rate": 0.05, "depth": 6, "l2_leaf_reg": 6.0, "loss_function": "YetiRank", "cat_features": cat_indices, "random_seed": seed, "thread_count": 4, "verbose": 0}
    pool_tr = Pool(train_cat[cat_cols], y_train, group_id=train_qid_xgb, cat_features=cat_indices)
    m4 = CatBoostRanker(**p_cat_rk).fit(pool_tr)
    tu_cat_rk = normalize_scores_to_margin(m4.predict(tune_cat[cat_cols]))
    te_cat_rk = normalize_scores_to_margin(m4.predict(test_cat[cat_cols]))
    oof_cat_rk = np.zeros(len(train))
    for tr_i, va_i in skf.split(train, y_train):
        p_f = Pool(train_cat[cat_cols].iloc[tr_i], y_train[tr_i], group_id=train_qid_xgb[tr_i], cat_features=cat_indices)
        m = CatBoostRanker(**p_cat_rk).fit(p_f)
        oof_cat_rk[va_i] = m.predict(train_cat[cat_cols].iloc[va_i])
    oof_cat_rk = normalize_scores_to_margin(oof_cat_rk)

    # 5. LGB_LogLoss
    p_lgb_ll = {"learning_rate": 0.04, "num_leaves": 31, "min_child_samples": 80, "colsample_bytree": 0.8, "subsample": 0.8, "n_estimators": 220, "verbosity": -1, "random_state": seed, "n_jobs": 4}
    m5 = lgb.LGBMClassifier(**p_lgb_ll).fit(train_xgb[lgb_14], y_train)
    tu_lgb_ll = prob_to_logit(m5.predict_proba(tune_xgb[lgb_14])[:, 1])
    te_lgb_ll = prob_to_logit(m5.predict_proba(test_xgb[lgb_14])[:, 1])
    oof_lgb_ll = np.zeros(len(train))
    for tr_i, va_i in skf.split(train, y_train):
        m = lgb.LGBMClassifier(**p_lgb_ll).fit(train_xgb[lgb_14].iloc[tr_i], y_train[tr_i])
        oof_lgb_ll[va_i] = prob_to_logit(m.predict_proba(train_xgb[lgb_14].iloc[va_i])[:, 1])

    # Construir datasets de Nivel 2
    fixed_base_cols = anom_cols + ["pct_heuristico_fraude", "distancia_log_cluster", "deficit_piso_cluster"]
    models_oof = {"XGB_LogLoss": oof_xgb_ll, "LGB_Focal": oof_lgb_fc, "XGB_Focal": oof_xgb_fc, "CAT_Ranker": oof_cat_rk, "LGB_LogLoss": oof_lgb_ll}
    models_tu = {"XGB_LogLoss": tu_xgb_ll, "LGB_Focal": tu_lgb_fc, "XGB_Focal": tu_xgb_fc, "CAT_Ranker": tu_cat_rk, "LGB_LogLoss": tu_lgb_ll}
    models_te = {"XGB_LogLoss": te_xgb_ll, "LGB_Focal": te_lgb_fc, "XGB_Focal": te_xgb_fc, "CAT_Ranker": te_cat_rk, "LGB_LogLoss": te_lgb_ll}

    X_tr = pd.DataFrame({c: train[c].to_numpy(dtype=float) for c in fixed_base_cols})
    X_tu = pd.DataFrame({c: tune[c].to_numpy(dtype=float) for c in fixed_base_cols})
    X_te = pd.DataFrame({c: test[c].to_numpy(dtype=float) for c in fixed_base_cols})

    for k in models_oof:
        X_tr[k] = models_oof[k]
        X_tu[k] = models_tu[k]
        X_te[k] = models_te[k]

    cols = list(X_tr.columns)
    sc = StandardScaler().fit(X_tr)
    X_tr_sc = pd.DataFrame(sc.transform(X_tr), columns=cols)
    X_tu_sc = pd.DataFrame(sc.transform(X_tu), columns=cols)
    X_te_sc = pd.DataFrame(sc.transform(X_te), columns=cols)

    print("\n" + "=" * 80)
    print("EVALUACIÓN COMPARATIVA: IMPACTO DE PENALIZACIÓN L1 VS L2 EN NIVEL 2")
    print("=" * 80)

    # 1. Ridge L2 (lo que ejecutó el script por omisión de penalty)
    m_l2 = LogisticRegression(penalty="l2", C=0.20, solver="saga", max_iter=1000, random_state=seed).fit(X_tr_sc, y_train)
    v_l2 = pr_auc(y_tune, m_l2.predict_proba(X_tu_sc)[:, 1])
    t_l2 = pr_auc(y_test, m_l2.predict_proba(X_te_sc)[:, 1])
    n_zeros_l2 = np.sum(np.abs(m_l2.coef_[0]) < 1e-4)

    print(f"Ridge L2 (C=0.20)              | val_search: {v_l2:.5f} | test: {t_l2:.5f} | Features apagadas: {n_zeros_l2}/19")

    # 2. Lasso L1 verdadero con varios C
    for c_val in [0.20, 0.10, 0.05, 0.02]:
        m_l1 = LogisticRegression(penalty="l1", C=c_val, solver="saga", max_iter=1000, random_state=seed).fit(X_tr_sc, y_train)
        v_l1 = pr_auc(y_tune, m_l1.predict_proba(X_tu_sc)[:, 1])
        t_l1 = pr_auc(y_test, m_l1.predict_proba(X_te_sc)[:, 1])
        n_zeros_l1 = np.sum(np.abs(m_l1.coef_[0]) < 1e-4)
        print(f"Lasso L1 (C={c_val:<4.2f}, penalty='l1') | val_search: {v_l1:.5f} | test: {t_l1:.5f} | Features apagadas: {n_zeros_l1}/19")

if __name__ == "__main__":
    main()
