"""
Prueba de Concepto (PoC): Stacking Multi-Loss con 4 Funciones de Pérdida en LightGBM
=====================================================================================
Hipótesis de Franco:
1. Evaluar si entrenar LightGBM bajo distintas funciones de pérdida (Log-Loss estándar,
   Weighted Cross-Entropy, Focal Loss y LambdaMART Ranking Loss) reduce la correlación
   entre predictores de árboles.
2. Afinar cada variante con CMA-ES sobre val_search (10 trials ágiles).
3. Construir los logits de cada modelo mediante 5-Fold Stratified CV OOF.
4. Comparar el Champion actual (Cuatri-modelo) vs Ensamble Multi-Loss de 7 modelos en Nivel 2
   usando la base fija de anomalías, reglas y semántica con regularización Lasso L1 y Ridge L2.
"""

from __future__ import annotations
import json
from pathlib import Path
import os
import random
import sys
import time
from typing import List, Tuple, Dict, Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from catboost import CatBoostClassifier
import lightgbm as lgb
import numpy as np
import optuna
from optuna.samplers import CmaEsSampler
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
import xgboost as xgb

from src.models.supervised import (
    calibrate, logit, pr_auc, validation_blocks,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_CATEGORICAL_COLS = ["a", "g", "n", "o", "p"]
BASE_NUMERIC_COLS = ["score", "monto", "b", "c", "d", "e", "f", "h", "l", "m", "q"]


def build_clustering_features(df_train: pd.DataFrame, df_all: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Genera deficit_piso_cluster, distancia_log_cluster y ratio_monto_mediana_cluster ajustados en Train con SVD d=30."""
    print("  -> Generando clustering semantico k=50 (SVD d=30 L2-norm) sobre titulos...")
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


def make_focal_loss(gamma: float = 2.0, alpha: float = 0.5):
    """Implementa Focal Loss robusta para LightGBM devolviendo gradiente y hessiano vectorizados."""
    def focal_obj(y_true: np.ndarray, y_pred: np.ndarray):
        p = 1.0 / (1.0 + np.exp(-np.clip(y_pred, -25.0, 25.0)))
        pt = np.where(y_true == 1, p, 1.0 - p)
        alpha_t = np.where(y_true == 1, alpha, 1.0 - alpha)
        focal_weight = alpha_t * ((1.0 - pt) ** gamma)
        grad = focal_weight * (p - y_true)
        hess = np.maximum(focal_weight * p * (1.0 - p), 1e-5)
        return grad, hess
    return focal_obj


def prob_to_logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p_clipped = np.clip(p, eps, 1.0 - eps)
    return np.log(p_clipped / (1.0 - p_clipped))


def main():
    seed = 42
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    t0 = time.time()

    print("=" * 90)
    print("PRUEBA DE CONCEPTO: STACKING MULTI-LOSS CON 4 FUNCIONES DE PERDIDA EN LIGHTGBM")
    print("Evaluando hipotesis de descorrelacion y aporte incremental a ciegas de Test")
    print("=" * 90)

    # 1. Cargar bases y variables
    print("\n1. Cargando datos...")
    base_df = pd.read_parquet(ROOT / "data/processed/dataset_base.parquet")
    feat_df = pd.read_parquet(ROOT / "data/features/features_transaccionales.parquet")
    anom_df = pd.read_parquet(ROOT / "data/features/scores_anomalias.parquet")
    rules_df = pd.read_parquet(ROOT / "data/features/features_reglas_asociacion.parquet")[["transaction_id", "pct_heuristico_fraude"]]

    exclude_anom = ["score_anomalia_consolidado", "anomalia_x_monto", "extended_iforest_x_monto"]
    anom_cols = [c for c in anom_df.columns if c not in ["transaction_id"] + exclude_anom]

    if "score_heuristico_fraude_10" in feat_df.columns:
        feat_df = feat_df.drop(columns=["score_heuristico_fraude_10"])

    # Generar clustering semántico Challenger v3
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
    test_xgb = df_xgb.loc[test.index]

    train_rf = df_rf.loc[df.split == "train"]
    tune_rf = df_rf.loc[tune.index]
    test_rf = df_rf.loc[test.index]

    df_cat = df.copy()
    for cat in BASE_CATEGORICAL_COLS:
        df_cat[cat] = df_cat[cat].fillna("MISSING").astype(str)
    train_cat = df_cat.loc[df.split == "train"]
    tune_cat = df_cat.loc[tune.index]
    test_cat = df_cat.loc[test.index]

    lgb_14 = [
        "score", "o", "g", "a", "es_madrugada", "ratio_d_m", "ratio_monto_l",
        "prod_b_log_c", "score_texto_nb", "ratio_monto_e", "is_h_cero",
        "deficit_piso_cluster", "distancia_log_cluster", "ratio_monto_mediana_cluster"
    ]
    xgb_cols = ['score', 'o', 'te_j_bayes', 'es_madrugada', 'ratio_monto_l', 'diff_m_d', 'score_texto_nb', 'is_h_cero', 'flag_alto_riesgo_nop', 'ratio_monto_b', 'g', 'b']
    cat_cols = [c for c in lgb_14]
    cat_indices = [cat_cols.index(c) for c in BASE_CATEGORICAL_COLS if c in cat_cols]
    rf_cols = ['score', 'o', 'te_j_bayes', 'es_madrugada', 'diff_m_d', 'distancia_log_cluster']

    # Grupos por fecha para LambdaMART
    train_groups = train.groupby(train["fecha"].dt.date, sort=False).size().values
    tune_groups = tune.groupby(tune["fecha"].dt.date, sort=False).size().values

    # =========================================================================
    # 2. FINE-TUNING CON CMA-ES DE LAS 4 VARIANTES DE LIGHTGBM
    # =========================================================================
    print("\n2. Fine-tuning ágil con CMA-ES (10 trials por variante) sobre val_search...")

    # A) Loss 1: Log-Loss Estándar (Binary)
    print("  -> Optimizando Variante 1: LightGBM (Standard Log-Loss)...")
    def obj_logloss(trial):
        params = {
            "learning_rate": trial.suggest_float("lr", 0.02, 0.07),
            "num_leaves": trial.suggest_int("num_leaves", 20, 45),
            "min_child_samples": trial.suggest_int("min_child_samples", 60, 150),
            "colsample_bytree": trial.suggest_float("colsample", 0.65, 0.95),
            "subsample": trial.suggest_float("subsample", 0.65, 0.95),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 5.0, log=True),
            "n_estimators": 220,
            "objective": "binary",
            "verbosity": -1,
            "random_state": seed,
            "n_jobs": 4,
        }
        m = lgb.LGBMClassifier(**params).fit(train_xgb[lgb_14], y_train)
        p = m.predict_proba(tune_xgb[lgb_14])[:, 1]
        return pr_auc(y_tune, p)

    study_logloss = optuna.create_study(direction="maximize", sampler=CmaEsSampler(seed=seed))
    study_logloss.optimize(obj_logloss, n_trials=10)
    best_logloss_params = study_logloss.best_params
    print(f"     Mejor PR-AUC val_search (Log-Loss): {study_logloss.best_value:.5f}")

    # B) Loss 2: Weighted Cross-Entropy (scale_pos_weight)
    print("  -> Optimizando Variante 2: LightGBM (Weighted Cross-Entropy)...")
    def obj_weighted(trial):
        params = {
            "learning_rate": trial.suggest_float("lr", 0.02, 0.07),
            "num_leaves": trial.suggest_int("num_leaves", 20, 45),
            "min_child_samples": trial.suggest_int("min_child_samples", 60, 150),
            "colsample_bytree": trial.suggest_float("colsample", 0.65, 0.95),
            "subsample": trial.suggest_float("subsample", 0.65, 0.95),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 5.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 2.0, 9.0),
            "n_estimators": 220,
            "objective": "binary",
            "verbosity": -1,
            "random_state": seed,
            "n_jobs": 4,
        }
        m = lgb.LGBMClassifier(**params).fit(train_xgb[lgb_14], y_train)
        p = m.predict_proba(tune_xgb[lgb_14])[:, 1]
        return pr_auc(y_tune, p)

    study_weighted = optuna.create_study(direction="maximize", sampler=CmaEsSampler(seed=seed))
    study_weighted.optimize(obj_weighted, n_trials=10)
    best_weighted_params = study_weighted.best_params
    print(f"     Mejor PR-AUC val_search (Weighted): {study_weighted.best_value:.5f}")

    # C) Loss 3: Focal Loss
    print("  -> Optimizando Variante 3: LightGBM (Focal Loss)...")
    def obj_focal(trial):
        gamma = trial.suggest_float("gamma", 1.0, 3.0)
        alpha = trial.suggest_float("alpha", 0.25, 0.75)
        params = {
            "learning_rate": trial.suggest_float("lr", 0.02, 0.07),
            "num_leaves": trial.suggest_int("num_leaves", 20, 45),
            "min_child_samples": trial.suggest_int("min_child_samples", 60, 150),
            "colsample_bytree": trial.suggest_float("colsample", 0.65, 0.95),
            "subsample": trial.suggest_float("subsample", 0.65, 0.95),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 5.0, log=True),
            "n_estimators": 220,
            "objective": make_focal_loss(gamma, alpha),
            "verbosity": -1,
            "random_state": seed,
            "n_jobs": 4,
        }
        m = lgb.LGBMClassifier(**params).fit(train_xgb[lgb_14], y_train)
        raw = m.predict_proba(tune_xgb[lgb_14])
        p = 1.0 / (1.0 + np.exp(-np.clip(raw, -25.0, 25.0)))
        return pr_auc(y_tune, p)

    study_focal = optuna.create_study(direction="maximize", sampler=CmaEsSampler(seed=seed))
    study_focal.optimize(obj_focal, n_trials=10)
    best_focal_params = study_focal.best_params
    print(f"     Mejor PR-AUC val_search (Focal):    {study_focal.best_value:.5f}")

    # D) Loss 4: LambdaMART (Ranking)
    print("  -> Optimizando Variante 4: LightGBM (LambdaMART Ranker)...")
    def obj_ranker(trial):
        params = {
            "learning_rate": trial.suggest_float("lr", 0.02, 0.07),
            "num_leaves": trial.suggest_int("num_leaves", 20, 45),
            "min_child_samples": trial.suggest_int("min_child_samples", 60, 150),
            "colsample_bytree": trial.suggest_float("colsample", 0.65, 0.95),
            "subsample": trial.suggest_float("subsample", 0.65, 0.95),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 5.0, log=True),
            "n_estimators": 220,
            "objective": "lambdarank",
            "verbosity": -1,
            "random_state": seed,
            "n_jobs": 4,
        }
        ranker = lgb.LGBMRanker(**params).fit(train_xgb[lgb_14], y_train, group=train_groups)
        preds = ranker.predict(tune_xgb[lgb_14])
        return pr_auc(y_tune, preds)

    study_ranker = optuna.create_study(direction="maximize", sampler=CmaEsSampler(seed=seed))
    study_ranker.optimize(obj_ranker, n_trials=10)
    best_ranker_params = study_ranker.best_params
    print(f"     Mejor PR-AUC val_search (Ranker):   {study_ranker.best_value:.5f}")

    # =========================================================================
    # 3. ENTRENAMIENTO COMPLETO DE LOS 7 MODELOS
    # =========================================================================
    print("\n3. Entrenando los 7 modelos sobre el Train completo...")

    # 1. LGB Log-Loss
    p_logloss = {
        "learning_rate": best_logloss_params["lr"],
        "num_leaves": best_logloss_params["num_leaves"],
        "min_child_samples": best_logloss_params["min_child_samples"],
        "colsample_bytree": best_logloss_params["colsample"],
        "subsample": best_logloss_params["subsample"],
        "reg_alpha": best_logloss_params["reg_alpha"],
        "reg_lambda": best_logloss_params["reg_lambda"],
        "n_estimators": 250,
        "objective": "binary",
        "verbosity": -1,
        "random_state": seed,
        "n_jobs": 4,
    }
    m_lgb_logloss = lgb.LGBMClassifier(**p_logloss).fit(train_xgb[lgb_14], y_train)
    p_ll_tu = m_lgb_logloss.predict_proba(tune_xgb[lgb_14])[:, 1]
    p_ll_te = m_lgb_logloss.predict_proba(test_xgb[lgb_14])[:, 1]

    # 2. LGB Weighted
    p_weighted = {
        "learning_rate": best_weighted_params["lr"],
        "num_leaves": best_weighted_params["num_leaves"],
        "min_child_samples": best_weighted_params["min_child_samples"],
        "colsample_bytree": best_weighted_params["colsample"],
        "subsample": best_weighted_params["subsample"],
        "reg_alpha": best_weighted_params["reg_alpha"],
        "reg_lambda": best_weighted_params["reg_lambda"],
        "scale_pos_weight": best_weighted_params["scale_pos_weight"],
        "n_estimators": 250,
        "objective": "binary",
        "verbosity": -1,
        "random_state": seed,
        "n_jobs": 4,
    }
    m_lgb_weighted = lgb.LGBMClassifier(**p_weighted).fit(train_xgb[lgb_14], y_train)
    p_wt_tu = m_lgb_weighted.predict_proba(tune_xgb[lgb_14])[:, 1]
    p_wt_te = m_lgb_weighted.predict_proba(test_xgb[lgb_14])[:, 1]

    # 3. LGB Focal
    p_focal = {
        "learning_rate": best_focal_params["lr"],
        "num_leaves": best_focal_params["num_leaves"],
        "min_child_samples": best_focal_params["min_child_samples"],
        "colsample_bytree": best_focal_params["colsample"],
        "subsample": best_focal_params["subsample"],
        "reg_alpha": best_focal_params["reg_alpha"],
        "reg_lambda": best_focal_params["reg_lambda"],
        "n_estimators": 250,
        "objective": make_focal_loss(best_focal_params["gamma"], best_focal_params["alpha"]),
        "verbosity": -1,
        "random_state": seed,
        "n_jobs": 4,
    }
    m_lgb_focal = lgb.LGBMClassifier(**p_focal).fit(train_xgb[lgb_14], y_train)
    raw_fc_tu = m_lgb_focal.predict_proba(tune_xgb[lgb_14])
    p_fc_tu = 1.0 / (1.0 + np.exp(-np.clip(raw_fc_tu, -25.0, 25.0)))
    raw_fc_te = m_lgb_focal.predict_proba(test_xgb[lgb_14])
    p_fc_te = 1.0 / (1.0 + np.exp(-np.clip(raw_fc_te, -25.0, 25.0)))

    # 4. LGB LambdaMART
    p_ranker = {
        "learning_rate": best_ranker_params["lr"],
        "num_leaves": best_ranker_params["num_leaves"],
        "min_child_samples": best_ranker_params["min_child_samples"],
        "colsample_bytree": best_ranker_params["colsample"],
        "subsample": best_ranker_params["subsample"],
        "reg_alpha": best_ranker_params["reg_alpha"],
        "reg_lambda": best_ranker_params["reg_lambda"],
        "n_estimators": 250,
        "objective": "lambdarank",
        "verbosity": -1,
        "random_state": seed,
        "n_jobs": 4,
    }
    m_lgb_ranker = lgb.LGBMRanker(**p_ranker).fit(train_xgb[lgb_14], y_train, group=train_groups)
    p_rk_tu = m_lgb_ranker.predict(tune_xgb[lgb_14])
    p_rk_te = m_lgb_ranker.predict(test_xgb[lgb_14])

    # 5. XGBoost (Campeón existente)
    best_params_xgb = {
        "learning_rate": 0.052, "max_depth": 5, "min_child_weight": 35,
        "colsample_bytree": 0.80, "subsample": 0.80, "reg_alpha": 0.1,
        "reg_lambda": 5.0, "n_estimators": 180, "tree_method": "hist",
        "enable_categorical": True, "random_state": seed, "n_jobs": 4, "verbosity": 0,
    }
    m_xgb = xgb.XGBClassifier(**best_params_xgb).fit(train_xgb[xgb_cols], y_train)
    p_xgb_tu = m_xgb.predict_proba(tune_xgb[xgb_cols])[:, 1]
    p_xgb_te = m_xgb.predict_proba(test_xgb[xgb_cols])[:, 1]

    # 6. CatBoost (Campeón existente)
    m_cat = CatBoostClassifier(
        iterations=280, learning_rate=0.045, depth=6, l2_leaf_reg=6.0,
        cat_features=cat_indices, random_seed=seed, thread_count=4, verbose=0,
    )
    m_cat.fit(train_cat[cat_cols], y_train, eval_set=(tune_cat[cat_cols], y_tune), early_stopping_rounds=30)
    p_cat_tu = m_cat.predict_proba(tune_cat[cat_cols])[:, 1]
    p_cat_te = m_cat.predict_proba(test_cat[cat_cols])[:, 1]

    # 7. Random Forest (Campeón existente)
    m_rf = RandomForestClassifier(n_estimators=80, max_depth=10, min_samples_leaf=30, max_features=0.7, random_state=seed, n_jobs=4)
    m_rf.fit(train_rf[rf_cols], y_train)
    p_rf_tu = m_rf.predict_proba(tune_rf[rf_cols])[:, 1]
    p_rf_te = m_rf.predict_proba(test_rf[rf_cols])[:, 1]

    print("\n  -> PR-AUC Individual en val_search:")
    print(f"     1. LGBM Log-Loss:     {pr_auc(y_tune, p_ll_tu):.5f} | test: {pr_auc(y_test, p_ll_te):.5f}")
    print(f"     2. LGBM Weighted:     {pr_auc(y_tune, p_wt_tu):.5f} | test: {pr_auc(y_test, p_wt_te):.5f}")
    print(f"     3. LGBM Focal Loss:   {pr_auc(y_tune, p_fc_tu):.5f} | test: {pr_auc(y_test, p_fc_te):.5f}")
    print(f"     4. LGBM LambdaRank:   {pr_auc(y_tune, p_rk_tu):.5f} | test: {pr_auc(y_test, p_rk_te):.5f}")
    print(f"     5. XGBoost (Control): {pr_auc(y_tune, p_xgb_tu):.5f} | test: {pr_auc(y_test, p_xgb_te):.5f}")
    print(f"     6. CatBoost (Control):{pr_auc(y_tune, p_cat_tu):.5f} | test: {pr_auc(y_test, p_cat_te):.5f}")
    print(f"     7. Random Forest:     {pr_auc(y_tune, p_rf_tu):.5f} | test: {pr_auc(y_test, p_rf_te):.5f}")

    # =========================================================================
    # 4. AUDITORIA DE CORRELACION DE SPEARMAN (TEST CENTRAL DE LA HIPOTESIS)
    # =========================================================================
    print("\n" + "=" * 90)
    print("4. AUDITORIA DE CORRELACION DE SPEARMAN EN VAL_SEARCH")
    print("=" * 90)

    model_preds_tune = {
        "LGB_LogLoss": p_ll_tu,
        "LGB_Weighted": p_wt_tu,
        "LGB_Focal": p_fc_tu,
        "LGB_Ranker": p_rk_tu,
        "XGBoost": p_xgb_tu,
        "CatBoost": p_cat_tu,
        "RandomForest": p_rf_tu,
    }

    names = list(model_preds_tune.keys())
    corr_matrix = pd.DataFrame(index=names, columns=names, dtype=float)
    for n1 in names:
        for n2 in names:
            rho, _ = spearmanr(model_preds_tune[n1], model_preds_tune[n2])
            corr_matrix.loc[n1, n2] = rho

    print(corr_matrix.to_string(float_format=lambda x: f"{x:.4f}"))

    # =========================================================================
    # 5. GENERACIÓN OOF CON 5-FOLD STRATIFIED CV EN TRAIN
    # =========================================================================
    print("\n5. Generando probabilidades y logits OOF (5-Fold Stratified CV en Train)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    oof_ll = np.zeros(len(train), dtype=float)
    oof_wt = np.zeros(len(train), dtype=float)
    oof_fc = np.zeros(len(train), dtype=float)
    oof_rk = np.zeros(len(train), dtype=float)
    oof_xgb = np.zeros(len(train), dtype=float)
    oof_cat = np.zeros(len(train), dtype=float)
    oof_rf = np.zeros(len(train), dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y_train), 1):
        # 1. LogLoss
        m1 = lgb.LGBMClassifier(**p_logloss).fit(train_xgb[lgb_14].iloc[tr_idx], y_train[tr_idx])
        oof_ll[va_idx] = m1.predict_proba(train_xgb[lgb_14].iloc[va_idx])[:, 1]

        # 2. Weighted
        m2 = lgb.LGBMClassifier(**p_weighted).fit(train_xgb[lgb_14].iloc[tr_idx], y_train[tr_idx])
        oof_wt[va_idx] = m2.predict_proba(train_xgb[lgb_14].iloc[va_idx])[:, 1]

        # 3. Focal
        m3 = lgb.LGBMClassifier(**p_focal).fit(train_xgb[lgb_14].iloc[tr_idx], y_train[tr_idx])
        raw_fc = m3.predict_proba(train_xgb[lgb_14].iloc[va_idx])
        oof_fc[va_idx] = 1.0 / (1.0 + np.exp(-np.clip(raw_fc, -25.0, 25.0)))

        # 4. LambdaRank (ordenar tr_idx por fecha para el grupo)
        tr_df = train.iloc[tr_idx].sort_values("fecha")
        tr_order = tr_df.index
        tr_groups_fold = tr_df.groupby(tr_df["fecha"].dt.date, sort=False).size().values
        m4 = lgb.LGBMRanker(**p_ranker).fit(train_xgb[lgb_14].loc[tr_order], y_train[train.index.get_indexer(tr_order)], group=tr_groups_fold)
        oof_rk[va_idx] = m4.predict(train_xgb[lgb_14].iloc[va_idx])

        # 5. XGB
        m5 = xgb.XGBClassifier(**best_params_xgb).fit(train_xgb[xgb_cols].iloc[tr_idx], y_train[tr_idx])
        oof_xgb[va_idx] = m5.predict_proba(train_xgb[xgb_cols].iloc[va_idx])[:, 1]

        # 6. CatBoost
        m6 = CatBoostClassifier(iterations=250, learning_rate=0.045, depth=6, l2_leaf_reg=6.0, cat_features=cat_indices, random_seed=seed, thread_count=4, verbose=0)
        m6.fit(train_cat[cat_cols].iloc[tr_idx], y_train[tr_idx])
        oof_cat[va_idx] = m6.predict_proba(train_cat[cat_cols].iloc[va_idx])[:, 1]

        # 7. RF
        m7 = RandomForestClassifier(n_estimators=80, max_depth=10, min_samples_leaf=30, max_features=0.7, random_state=seed, n_jobs=4)
        m7.fit(train_rf[rf_cols].iloc[tr_idx], y_train[tr_idx])
        oof_rf[va_idx] = m7.predict_proba(train_rf[rf_cols].iloc[va_idx])[:, 1]

    print("  * OOF PR-AUC -> LogLoss: {:.5f} | Weighted: {:.5f} | Focal: {:.5f} | Ranker: {:.5f} | XGB: {:.5f} | CAT: {:.5f} | RF: {:.5f}".format(
        pr_auc(y_train, oof_ll), pr_auc(y_train, oof_wt), pr_auc(y_train, oof_fc),
        pr_auc(y_train, oof_rk), pr_auc(y_train, oof_xgb), pr_auc(y_train, oof_cat), pr_auc(y_train, oof_rf)
    ))

    # =========================================================================
    # 6. CONSTRUCCIÓN DE METADATASETS DE NIVEL 2 Y EVALUACIÓN DE STACKING
    # =========================================================================
    print("\n6. Construyendo Meta-Datasets en espacio de Logits con Base Fija...")

    def build_meta(p_ll, p_wt, p_fc, p_rk, p_x, p_c, p_r, part_df):
        d = {
            # Logits de las 4 pérdidas de LightGBM
            "logit_lgb_logloss": prob_to_logit(p_ll),
            "logit_lgb_weighted": prob_to_logit(p_wt),
            "logit_lgb_focal": prob_to_logit(p_fc),
            "logit_lgb_ranker": (p_rk - np.mean(p_rk)) / (np.std(p_rk) + 1e-4), # Ranker Margin Z-Score
            # Logits de los otros 3 inductores
            "logit_xgb": prob_to_logit(p_x),
            "logit_cat": prob_to_logit(p_c),
            "logit_rf": prob_to_logit(p_r),
        }
        # Base fija de contexto (Anomalías, Reglas, Semántica)
        for col in anom_cols:
            d[col] = part_df[col].to_numpy(dtype=float)
        d["pct_heuristico_fraude"] = part_df["pct_heuristico_fraude"].to_numpy(dtype=float)
        d["distancia_log_cluster"] = part_df["distancia_log_cluster"].to_numpy(dtype=float)
        d["deficit_piso_cluster"] = part_df["deficit_piso_cluster"].to_numpy(dtype=float)
        return pd.DataFrame(d)

    X_meta_tr = build_meta(oof_ll, oof_wt, oof_fc, oof_rk, oof_xgb, oof_cat, oof_rf, train)
    X_meta_tu = build_meta(p_ll_tu, p_wt_tu, p_fc_tu, p_rk_tu, p_xgb_tu, p_cat_tu, p_rf_tu, tune)
    X_meta_te = build_meta(p_ll_te, p_wt_te, p_fc_te, p_rk_te, p_xgb_te, p_cat_te, p_rf_te, test)

    fixed_base_cols = anom_cols + ["pct_heuristico_fraude", "distancia_log_cluster", "deficit_piso_cluster"]

    # Definir subconjuntos de modelos para comparar
    cols_champion = ["logit_lgb_logloss", "logit_xgb", "logit_cat", "logit_rf"] + fixed_base_cols
    cols_lgb_multiloss = ["logit_lgb_logloss", "logit_lgb_weighted", "logit_lgb_focal", "logit_lgb_ranker"] + fixed_base_cols
    cols_septa_total = [
        "logit_lgb_logloss", "logit_lgb_weighted", "logit_lgb_focal", "logit_lgb_ranker",
        "logit_xgb", "logit_cat", "logit_rf"
    ] + fixed_base_cols

    def evaluate_stacking(cols: List[str], label: str):
        scaler = StandardScaler().fit(X_meta_tr[cols])
        X_tr = pd.DataFrame(scaler.transform(X_meta_tr[cols]), columns=cols)
        X_tu = pd.DataFrame(scaler.transform(X_meta_tu[cols]), columns=cols)
        X_te = pd.DataFrame(scaler.transform(X_meta_te[cols]), columns=cols)

        results = {}
        # Lasso L1 C=0.1
        m_l1_01 = LogisticRegression(solver="saga", l1_ratio=1.0, C=0.10, max_iter=1000, random_state=seed).fit(X_tr, y_train)
        results["L1_C01"] = (pr_auc(y_tune, m_l1_01.predict_proba(X_tu)[:, 1]), pr_auc(y_test, m_l1_01.predict_proba(X_te)[:, 1]), m_l1_01)

        # Lasso L1 C=0.2
        m_l1_02 = LogisticRegression(solver="saga", l1_ratio=1.0, C=0.20, max_iter=1000, random_state=seed).fit(X_tr, y_train)
        results["L1_C02"] = (pr_auc(y_tune, m_l1_02.predict_proba(X_tu)[:, 1]), pr_auc(y_test, m_l1_02.predict_proba(X_te)[:, 1]), m_l1_02)

        # Ridge L2 C=1.0
        m_l2 = LogisticRegression(C=1.0, max_iter=1000, random_state=seed).fit(X_tr, y_train)
        results["L2_C10"] = (pr_auc(y_tune, m_l2.predict_proba(X_tu)[:, 1]), pr_auc(y_test, m_l2.predict_proba(X_te)[:, 1]), m_l2)
        return results

    res_champ = evaluate_stacking(cols_champion, "Champion Base (4 Modelos)")
    res_multilgb = evaluate_stacking(cols_lgb_multiloss, "LGBM Multi-Loss (4 LGBMs)")
    res_septa = evaluate_stacking(cols_septa_total, "Septa-Modelo (7 Modelos)")

    # =========================================================================
    # 7. REPORTE COMPARATIVO FINAL
    # =========================================================================
    print("\n" + "=" * 95)
    print("RESUMEN COMPARATIVO DEL EXPERIMENTO: CHAMPION VS MULTI-LOSS STACKING")
    print("=" * 95)

    summary_rows = [
        ("Control Ian (LightGBM 11 vars)", 0.48136, 0.44435),
        ("Champion Base (Lasso L1 C=0.10)", res_champ["L1_C01"][0], res_champ["L1_C01"][1]),
        ("Champion Base (Lasso L1 C=0.20)", res_champ["L1_C02"][0], res_champ["L1_C02"][1]),
        ("Champion Base (Ridge L2 C=1.00)", res_champ["L2_C10"][0], res_champ["L2_C10"][1]),
        ("LGBM Multi-Loss (Lasso L1 C=0.10)", res_multilgb["L1_C01"][0], res_multilgb["L1_C01"][1]),
        ("LGBM Multi-Loss (Lasso L1 C=0.20)", res_multilgb["L1_C02"][0], res_multilgb["L1_C02"][1]),
        ("🏆 Septa-Modelo 7-Inductores (Lasso L1 C=0.10)", res_septa["L1_C01"][0], res_septa["L1_C01"][1]),
        ("🏆 Septa-Modelo 7-Inductores (Lasso L1 C=0.20)", res_septa["L1_C02"][0], res_septa["L1_C02"][1]),
        ("Septa-Modelo 7-Inductores (Ridge L2 C=1.00)", res_septa["L2_C10"][0], res_septa["L2_C10"][1]),
    ]

    print(f"{'Configuracion de Stacking':<50s} | {'val_search':<12s} | {'test final':<12s} | {'Delta vs Ian Test'}")
    print("-" * 95)
    for name, v_sc, t_sc in summary_rows:
        delta = t_sc - 0.44435
        print(f"{name:<50s} | {v_sc:.5f}      | {t_sc:.5f}      | {delta:+.5f} ({delta*10000:+.1f} bps)")

    # Coeficientes en Nivel 2 del Septa-Modelo
    print("\n" + "=" * 90)
    print("COEFICIENTES DEL METAMODELO LASSO L1 (C=0.20) EN EL SEPTA-MODELO:")
    print("=" * 90)
    septa_model = res_septa["L1_C02"][2]
    weights = pd.Series(septa_model.coef_[0], index=cols_septa_total).sort_values(ascending=False)
    for feat, w in weights.items():
        if abs(w) > 1e-4:
            print(f"  * {feat:<30s}: {w:+.5f}")
        else:
            print(f"  * {feat:<30s}:  0.00000 (APAGADO POR L1)")

    print(f"\n[OK] PoC concluida en {time.time() - t0:.1f} segundos.")


if __name__ == "__main__":
    main()
