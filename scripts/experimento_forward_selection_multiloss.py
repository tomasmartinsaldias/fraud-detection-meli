"""
Experimento Maestro: Multi-Loss Stacking y Forward Model Selection en Nivel 2
=============================================================================
Autores: Franco & Integrante 2
Objetivo:
1. Entrenar 16 variantes de modelos supervisados combinando 4 familias (LightGBM,
   XGBoost, CatBoost, Random Forest) con sus respectivas funciones de pérdida
   compatibles (Log-Loss, Weighted Loss, Focal Loss, Ranking Loss/Criteria).
2. Afinar cada modelo ágilmente con CMA-ES sobre val_search (tune).
3. Obtener los logits mediante 5-Fold Stratified CV OOF en Train.
4. Ejecutar Forward Selection Greedy de modelos en Nivel 2:
   - Base Fija inicial: 11 anomalías + pct_heuristico + semántica de catálogo.
   - Criterio de parada: Ganancia mínima requerida de +5 BPS (+0.0005 de PR-AUC)
     en val_search.
5. Evaluar a ciegas la generalización del ensamble ganador congelado en Test.
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

from catboost import CatBoostClassifier, CatBoostRanker, Pool
import lightgbm as lgb
import numpy as np
import optuna
from optuna.samplers import CmaEsSampler
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
import xgboost as xgb

from src.models.supervised import (
    logit, pr_auc, validation_blocks,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_CATEGORICAL_COLS = ["a", "g", "n", "o", "p"]
BASE_NUMERIC_COLS = ["score", "monto", "b", "c", "d", "e", "f", "h", "l", "m", "q"]


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


def make_lgb_focal_loss(gamma: float = 2.0, alpha: float = 0.5):
    def focal_obj(y_true: np.ndarray, y_pred: np.ndarray):
        p = 1.0 / (1.0 + np.exp(-np.clip(y_pred, -25.0, 25.0)))
        pt = np.where(y_true == 1, p, 1.0 - p)
        alpha_t = np.where(y_true == 1, alpha, 1.0 - alpha)
        focal_weight = alpha_t * ((1.0 - pt) ** gamma)
        grad = focal_weight * (p - y_true)
        hess = np.maximum(focal_weight * p * (1.0 - p), 1e-5)
        return grad, hess
    return focal_obj


def make_xgb_focal_loss(gamma: float = 2.0, alpha: float = 0.5):
    def xgb_focal_obj(y_true: np.ndarray, y_pred: np.ndarray):
        p = 1.0 / (1.0 + np.exp(-np.clip(y_pred, -25.0, 25.0)))
        pt = np.where(y_true == 1, p, 1.0 - p)
        alpha_t = np.where(y_true == 1, alpha, 1.0 - alpha)
        focal_weight = alpha_t * ((1.0 - pt) ** gamma)
        grad = focal_weight * (p - y_true)
        hess = np.maximum(focal_weight * p * (1.0 - p), 1e-5)
        return grad, hess
    return xgb_focal_obj


class CatBoostFocalObjective(object):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.5):
        self.gamma = gamma
        self.alpha = alpha

    def calc_ders_range(self, approxes, targets, weights):
        approxes = np.array(approxes)
        targets = np.array(targets)
        p = 1.0 / (1.0 + np.exp(-np.clip(approxes, -20.0, 20.0)))
        pt = np.where(targets == 1, p, 1.0 - p)
        alpha_t = np.where(targets == 1, self.alpha, 1.0 - self.alpha)
        focal_weight = alpha_t * ((1.0 - pt) ** self.gamma)
        der1 = (targets - p) * focal_weight  # negative gradient
        der2 = np.maximum(focal_weight * p * (1.0 - p), 1e-5)
        return list(zip(der1, der2))


def prob_to_logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p_clipped = np.clip(p, eps, 1.0 - eps)
    return np.log(p_clipped / (1.0 - p_clipped))


def normalize_scores_to_margin(scores: np.ndarray) -> np.ndarray:
    """Estandariza los scores continuos de rankers a márgenes con varianza unitaria."""
    std = np.std(scores)
    return (scores - np.mean(scores)) / (std if std > 1e-6 else 1.0)


def main():
    seed = 42
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    t0 = time.time()

    print("=" * 95)
    print("FORWARD MODEL SELECTION CON 16 VARIANTES MULTI-LOSS EN NIVEL 2")
    print("LightGBM + XGBoost + CatBoost + Random Forest bajo 4 Funciones Objetivo")
    print("=" * 95)

    # 1. Cargar bases y generar features
    print("\n1. Cargando datos y features...")
    base_df = pd.read_parquet(ROOT / "data/processed/dataset_base.parquet")
    feat_df = pd.read_parquet(ROOT / "data/features/features_transaccionales.parquet")
    anom_df = pd.read_parquet(ROOT / "data/features/scores_anomalias.parquet")
    rules_df = pd.read_parquet(ROOT / "data/features/features_reglas_asociacion.parquet")[["transaction_id", "pct_heuristico_fraude"]]

    exclude_anom = ["score_anomalia_consolidado", "anomalia_x_monto", "extended_iforest_x_monto"]
    anom_cols = [c for c in anom_df.columns if c not in ["transaction_id"] + exclude_anom]

    if "score_heuristico_fraude_10" in feat_df.columns:
        feat_df = feat_df.drop(columns=["score_heuristico_fraude_10"])

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

    # Predecir categorías para XGBoost
    df_xgb = df.copy()
    for cat in BASE_CATEGORICAL_COLS:
        cats = sorted(df[cat].dropna().astype(str).unique())
        df_xgb[cat] = pd.Categorical(df_xgb[cat].astype(str), categories=cats)

    # Predecir categorías para Random Forest
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

    # Grupos temporales contiguos por fecha para ranking
    train_groups_lgb = train.groupby(train["fecha"].dt.date, sort=False).size().values
    train_qid_xgb = train["fecha"].dt.date.astype("category").cat.codes.values
    tune_qid_xgb = tune["fecha"].dt.date.astype("category").cat.codes.values

    # Estructuras para almacenar predictores (logits OOF, tune y test)
    candidate_oof: Dict[str, np.ndarray] = {}
    candidate_tune: Dict[str, np.ndarray] = {}
    candidate_test: Dict[str, np.ndarray] = {}
    candidate_val_prauc: Dict[str, float] = {}

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    print("\n" + "=" * 95)
    print("2. ENTRENAMIENTO, FINE-TUNING CON CMA-ES Y EXTRACCIÓN OOF DE 16 CANDIDATOS")
    print("=" * 95)

    # -------------------------------------------------------------------------
    # FAMILIA 1: LIGHTGBM (4 Variantes)
    # -------------------------------------------------------------------------
    print("\n--- [FAMILIA 1: LIGHTGBM] ---")

    # 1. LGBM Log-Loss
    print("  1/16. Optimizando LGBM Log-Loss...")
    def obj_lgb_ll(trial):
        p = {
            "learning_rate": trial.suggest_float("lr", 0.02, 0.07),
            "num_leaves": trial.suggest_int("nl", 20, 40),
            "min_child_samples": trial.suggest_int("mc", 60, 150),
            "colsample_bytree": trial.suggest_float("col", 0.65, 0.95),
            "subsample": trial.suggest_float("sub", 0.65, 0.95),
            "reg_alpha": trial.suggest_float("ra", 1e-3, 1.0, log=True),
            "reg_lambda": trial.suggest_float("rl", 1e-2, 5.0, log=True),
            "n_estimators": 220, "objective": "binary", "verbosity": -1, "random_state": seed, "n_jobs": 4,
        }
        m = lgb.LGBMClassifier(**p).fit(train_xgb[lgb_14], y_train)
        return pr_auc(y_tune, m.predict_proba(tune_xgb[lgb_14])[:, 1])

    s_lgb_ll = optuna.create_study(direction="maximize", sampler=CmaEsSampler(seed=seed))
    s_lgb_ll.optimize(obj_lgb_ll, n_trials=8)
    bp_lgb_ll = s_lgb_ll.best_params
    p_lgb_ll = {
        "learning_rate": bp_lgb_ll["lr"], "num_leaves": bp_lgb_ll["nl"], "min_child_samples": bp_lgb_ll["mc"],
        "colsample_bytree": bp_lgb_ll["col"], "subsample": bp_lgb_ll["sub"], "reg_alpha": bp_lgb_ll["ra"],
        "reg_lambda": bp_lgb_ll["rl"], "n_estimators": 240, "objective": "binary", "verbosity": -1, "random_state": seed, "n_jobs": 4,
    }
    m = lgb.LGBMClassifier(**p_lgb_ll).fit(train_xgb[lgb_14], y_train)
    p_tu = m.predict_proba(tune_xgb[lgb_14])[:, 1]
    p_te = m.predict_proba(test_xgb[lgb_14])[:, 1]
    candidate_tune["LGB_LogLoss"] = prob_to_logit(p_tu)
    candidate_test["LGB_LogLoss"] = prob_to_logit(p_te)
    candidate_val_prauc["LGB_LogLoss"] = pr_auc(y_tune, p_tu)
    print(f"       -> val_search PR-AUC: {candidate_val_prauc['LGB_LogLoss']:.5f}")

    oof = np.zeros(len(train), dtype=float)
    for tr_i, va_i in skf.split(train, y_train):
        m_f = lgb.LGBMClassifier(**p_lgb_ll).fit(train_xgb[lgb_14].iloc[tr_i], y_train[tr_i])
        oof[va_i] = m_f.predict_proba(train_xgb[lgb_14].iloc[va_i])[:, 1]
    candidate_oof["LGB_LogLoss"] = prob_to_logit(oof)

    # 2. LGBM Weighted
    print("  2/16. Optimizando LGBM Weighted Loss...")
    def obj_lgb_wt(trial):
        p = {
            "learning_rate": trial.suggest_float("lr", 0.02, 0.07),
            "num_leaves": trial.suggest_int("nl", 20, 40),
            "min_child_samples": trial.suggest_int("mc", 60, 150),
            "colsample_bytree": trial.suggest_float("col", 0.65, 0.95),
            "subsample": trial.suggest_float("sub", 0.65, 0.95),
            "reg_alpha": trial.suggest_float("ra", 1e-3, 1.0, log=True),
            "reg_lambda": trial.suggest_float("rl", 1e-2, 5.0, log=True),
            "scale_pos_weight": trial.suggest_float("spw", 2.0, 8.0),
            "n_estimators": 220, "objective": "binary", "verbosity": -1, "random_state": seed, "n_jobs": 4,
        }
        m = lgb.LGBMClassifier(**p).fit(train_xgb[lgb_14], y_train)
        return pr_auc(y_tune, m.predict_proba(tune_xgb[lgb_14])[:, 1])

    s_lgb_wt = optuna.create_study(direction="maximize", sampler=CmaEsSampler(seed=seed))
    s_lgb_wt.optimize(obj_lgb_wt, n_trials=8)
    bp_lgb_wt = s_lgb_wt.best_params
    p_lgb_wt = {
        "learning_rate": bp_lgb_wt["lr"], "num_leaves": bp_lgb_wt["nl"], "min_child_samples": bp_lgb_wt["mc"],
        "colsample_bytree": bp_lgb_wt["col"], "subsample": bp_lgb_wt["sub"], "reg_alpha": bp_lgb_wt["ra"],
        "reg_lambda": bp_lgb_wt["rl"], "scale_pos_weight": bp_lgb_wt["spw"], "n_estimators": 240,
        "objective": "binary", "verbosity": -1, "random_state": seed, "n_jobs": 4,
    }
    m = lgb.LGBMClassifier(**p_lgb_wt).fit(train_xgb[lgb_14], y_train)
    p_tu = m.predict_proba(tune_xgb[lgb_14])[:, 1]
    p_te = m.predict_proba(test_xgb[lgb_14])[:, 1]
    candidate_tune["LGB_Weighted"] = prob_to_logit(p_tu)
    candidate_test["LGB_Weighted"] = prob_to_logit(p_te)
    candidate_val_prauc["LGB_Weighted"] = pr_auc(y_tune, p_tu)
    print(f"       -> val_search PR-AUC: {candidate_val_prauc['LGB_Weighted']:.5f}")

    oof = np.zeros(len(train), dtype=float)
    for tr_i, va_i in skf.split(train, y_train):
        m_f = lgb.LGBMClassifier(**p_lgb_wt).fit(train_xgb[lgb_14].iloc[tr_i], y_train[tr_i])
        oof[va_i] = m_f.predict_proba(train_xgb[lgb_14].iloc[va_i])[:, 1]
    candidate_oof["LGB_Weighted"] = prob_to_logit(oof)

    # 3. LGBM Focal Loss
    print("  3/16. Optimizando LGBM Focal Loss...")
    def obj_lgb_fc(trial):
        gamma = trial.suggest_float("gamma", 1.0, 3.0)
        alpha = trial.suggest_float("alpha", 0.3, 0.7)
        p = {
            "learning_rate": trial.suggest_float("lr", 0.02, 0.07),
            "num_leaves": trial.suggest_int("nl", 20, 40),
            "min_child_samples": trial.suggest_int("mc", 60, 150),
            "colsample_bytree": trial.suggest_float("col", 0.65, 0.95),
            "subsample": trial.suggest_float("sub", 0.65, 0.95),
            "reg_alpha": trial.suggest_float("ra", 1e-3, 1.0, log=True),
            "reg_lambda": trial.suggest_float("rl", 1e-2, 5.0, log=True),
            "n_estimators": 220, "objective": make_lgb_focal_loss(gamma, alpha), "verbosity": -1, "random_state": seed, "n_jobs": 4,
        }
        m = lgb.LGBMClassifier(**p).fit(train_xgb[lgb_14], y_train)
        raw = m.predict_proba(tune_xgb[lgb_14])
        return pr_auc(y_tune, 1.0 / (1.0 + np.exp(-np.clip(raw, -25.0, 25.0))))

    s_lgb_fc = optuna.create_study(direction="maximize", sampler=CmaEsSampler(seed=seed))
    s_lgb_fc.optimize(obj_lgb_fc, n_trials=8)
    bp_lgb_fc = s_lgb_fc.best_params
    p_lgb_fc = {
        "learning_rate": bp_lgb_fc["lr"], "num_leaves": bp_lgb_fc["nl"], "min_child_samples": bp_lgb_fc["mc"],
        "colsample_bytree": bp_lgb_fc["col"], "subsample": bp_lgb_fc["sub"], "reg_alpha": bp_lgb_fc["ra"],
        "reg_lambda": bp_lgb_fc["rl"], "n_estimators": 240,
        "objective": make_lgb_focal_loss(bp_lgb_fc["gamma"], bp_lgb_fc["alpha"]), "verbosity": -1, "random_state": seed, "n_jobs": 4,
    }
    m = lgb.LGBMClassifier(**p_lgb_fc).fit(train_xgb[lgb_14], y_train)
    raw_tu = m.predict_proba(tune_xgb[lgb_14])
    raw_te = m.predict_proba(test_xgb[lgb_14])
    candidate_tune["LGB_Focal"] = raw_tu
    candidate_test["LGB_Focal"] = raw_te
    candidate_val_prauc["LGB_Focal"] = pr_auc(y_tune, 1.0 / (1.0 + np.exp(-np.clip(raw_tu, -25.0, 25.0))))
    print(f"       -> val_search PR-AUC: {candidate_val_prauc['LGB_Focal']:.5f}")

    oof = np.zeros(len(train), dtype=float)
    for tr_i, va_i in skf.split(train, y_train):
        m_f = lgb.LGBMClassifier(**p_lgb_fc).fit(train_xgb[lgb_14].iloc[tr_i], y_train[tr_i])
        oof[va_i] = m_f.predict_proba(train_xgb[lgb_14].iloc[va_i])
    candidate_oof["LGB_Focal"] = oof

    # 4. LGBM LambdaMART
    print("  4/16. Optimizando LGBM LambdaMART...")
    def obj_lgb_rk(trial):
        p = {
            "learning_rate": trial.suggest_float("lr", 0.02, 0.07),
            "num_leaves": trial.suggest_int("nl", 20, 40),
            "min_child_samples": trial.suggest_int("mc", 60, 150),
            "colsample_bytree": trial.suggest_float("col", 0.65, 0.95),
            "subsample": trial.suggest_float("sub", 0.65, 0.95),
            "reg_alpha": trial.suggest_float("ra", 1e-3, 1.0, log=True),
            "reg_lambda": trial.suggest_float("rl", 1e-2, 5.0, log=True),
            "n_estimators": 220, "objective": "lambdarank", "verbosity": -1, "random_state": seed, "n_jobs": 4,
        }
        ranker = lgb.LGBMRanker(**p).fit(train_xgb[lgb_14], y_train, group=train_groups_lgb)
        return pr_auc(y_tune, ranker.predict(tune_xgb[lgb_14]))

    s_lgb_rk = optuna.create_study(direction="maximize", sampler=CmaEsSampler(seed=seed))
    s_lgb_rk.optimize(obj_lgb_rk, n_trials=8)
    bp_lgb_rk = s_lgb_rk.best_params
    p_lgb_rk = {
        "learning_rate": bp_lgb_rk["lr"], "num_leaves": bp_lgb_rk["nl"], "min_child_samples": bp_lgb_rk["mc"],
        "colsample_bytree": bp_lgb_rk["col"], "subsample": bp_lgb_rk["sub"], "reg_alpha": bp_lgb_rk["ra"],
        "reg_lambda": bp_lgb_rk["rl"], "n_estimators": 240, "objective": "lambdarank", "verbosity": -1, "random_state": seed, "n_jobs": 4,
    }
    ranker = lgb.LGBMRanker(**p_lgb_rk).fit(train_xgb[lgb_14], y_train, group=train_groups_lgb)
    s_tu = ranker.predict(tune_xgb[lgb_14])
    s_te = ranker.predict(test_xgb[lgb_14])
    candidate_tune["LGB_Ranker"] = normalize_scores_to_margin(s_tu)
    candidate_test["LGB_Ranker"] = normalize_scores_to_margin(s_te)
    candidate_val_prauc["LGB_Ranker"] = pr_auc(y_tune, s_tu)
    print(f"       -> val_search PR-AUC: {candidate_val_prauc['LGB_Ranker']:.5f}")

    oof = np.zeros(len(train), dtype=float)
    for tr_i, va_i in skf.split(train, y_train):
        tr_df = train.iloc[tr_i].sort_values("fecha")
        tr_idx_order = tr_df.index
        grp = tr_df.groupby(tr_df["fecha"].dt.date, sort=False).size().values
        m_f = lgb.LGBMRanker(**p_lgb_rk).fit(train_xgb[lgb_14].loc[tr_idx_order], y_train[train.index.get_indexer(tr_idx_order)], group=grp)
        oof[va_i] = m_f.predict(train_xgb[lgb_14].iloc[va_i])
    candidate_oof["LGB_Ranker"] = normalize_scores_to_margin(oof)

    # -------------------------------------------------------------------------
    # FAMILIA 2: XGBOOST (4 Variantes)
    # -------------------------------------------------------------------------
    print("\n--- [FAMILIA 2: XGBOOST] ---")

    # 5. XGB Log-Loss
    print("  5/16. Optimizando XGBoost Log-Loss...")
    def obj_xgb_ll(trial):
        p = {
            "learning_rate": trial.suggest_float("lr", 0.03, 0.08),
            "max_depth": trial.suggest_int("md", 4, 6),
            "min_child_weight": trial.suggest_int("mcw", 20, 60),
            "colsample_bytree": trial.suggest_float("col", 0.65, 0.90),
            "subsample": trial.suggest_float("sub", 0.65, 0.90),
            "reg_alpha": trial.suggest_float("ra", 1e-2, 1.0, log=True),
            "reg_lambda": trial.suggest_float("rl", 0.5, 10.0, log=True),
            "n_estimators": 160, "tree_method": "hist", "enable_categorical": True,
            "random_state": seed, "n_jobs": 4, "verbosity": 0,
        }
        m = xgb.XGBClassifier(**p).fit(train_xgb[xgb_cols], y_train)
        return pr_auc(y_tune, m.predict_proba(tune_xgb[xgb_cols])[:, 1])

    s_xgb_ll = optuna.create_study(direction="maximize", sampler=CmaEsSampler(seed=seed))
    s_xgb_ll.optimize(obj_xgb_ll, n_trials=8)
    bp_xgb_ll = s_xgb_ll.best_params
    p_xgb_ll = {
        "learning_rate": bp_xgb_ll["lr"], "max_depth": bp_xgb_ll["md"], "min_child_weight": bp_xgb_ll["mcw"],
        "colsample_bytree": bp_xgb_ll["col"], "subsample": bp_xgb_ll["sub"], "reg_alpha": bp_xgb_ll["ra"],
        "reg_lambda": bp_xgb_ll["rl"], "n_estimators": 180, "tree_method": "hist", "enable_categorical": True,
        "random_state": seed, "n_jobs": 4, "verbosity": 0,
    }
    m = xgb.XGBClassifier(**p_xgb_ll).fit(train_xgb[xgb_cols], y_train)
    p_tu = m.predict_proba(tune_xgb[xgb_cols])[:, 1]
    p_te = m.predict_proba(test_xgb[xgb_cols])[:, 1]
    candidate_tune["XGB_LogLoss"] = prob_to_logit(p_tu)
    candidate_test["XGB_LogLoss"] = prob_to_logit(p_te)
    candidate_val_prauc["XGB_LogLoss"] = pr_auc(y_tune, p_tu)
    print(f"       -> val_search PR-AUC: {candidate_val_prauc['XGB_LogLoss']:.5f}")

    oof = np.zeros(len(train), dtype=float)
    for tr_i, va_i in skf.split(train, y_train):
        m_f = xgb.XGBClassifier(**p_xgb_ll).fit(train_xgb[xgb_cols].iloc[tr_i], y_train[tr_i])
        oof[va_i] = m_f.predict_proba(train_xgb[xgb_cols].iloc[va_i])[:, 1]
    candidate_oof["XGB_LogLoss"] = prob_to_logit(oof)

    # 6. XGB Weighted
    print("  6/16. Optimizando XGBoost Weighted Loss...")
    def obj_xgb_wt(trial):
        p = {
            "learning_rate": trial.suggest_float("lr", 0.03, 0.08),
            "max_depth": trial.suggest_int("md", 4, 6),
            "min_child_weight": trial.suggest_int("mcw", 20, 60),
            "colsample_bytree": trial.suggest_float("col", 0.65, 0.90),
            "subsample": trial.suggest_float("sub", 0.65, 0.90),
            "reg_alpha": trial.suggest_float("ra", 1e-2, 1.0, log=True),
            "reg_lambda": trial.suggest_float("rl", 0.5, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("spw", 2.0, 8.0),
            "n_estimators": 160, "tree_method": "hist", "enable_categorical": True,
            "random_state": seed, "n_jobs": 4, "verbosity": 0,
        }
        m = xgb.XGBClassifier(**p).fit(train_xgb[xgb_cols], y_train)
        return pr_auc(y_tune, m.predict_proba(tune_xgb[xgb_cols])[:, 1])

    s_xgb_wt = optuna.create_study(direction="maximize", sampler=CmaEsSampler(seed=seed))
    s_xgb_wt.optimize(obj_xgb_wt, n_trials=8)
    bp_xgb_wt = s_xgb_wt.best_params
    p_xgb_wt = {
        "learning_rate": bp_xgb_wt["lr"], "max_depth": bp_xgb_wt["md"], "min_child_weight": bp_xgb_wt["mcw"],
        "colsample_bytree": bp_xgb_wt["col"], "subsample": bp_xgb_wt["sub"], "reg_alpha": bp_xgb_wt["ra"],
        "reg_lambda": bp_xgb_wt["rl"], "scale_pos_weight": bp_xgb_wt["spw"], "n_estimators": 180,
        "tree_method": "hist", "enable_categorical": True, "random_state": seed, "n_jobs": 4, "verbosity": 0,
    }
    m = xgb.XGBClassifier(**p_xgb_wt).fit(train_xgb[xgb_cols], y_train)
    p_tu = m.predict_proba(tune_xgb[xgb_cols])[:, 1]
    p_te = m.predict_proba(test_xgb[xgb_cols])[:, 1]
    candidate_tune["XGB_Weighted"] = prob_to_logit(p_tu)
    candidate_test["XGB_Weighted"] = prob_to_logit(p_te)
    candidate_val_prauc["XGB_Weighted"] = pr_auc(y_tune, p_tu)
    print(f"       -> val_search PR-AUC: {candidate_val_prauc['XGB_Weighted']:.5f}")

    oof = np.zeros(len(train), dtype=float)
    for tr_i, va_i in skf.split(train, y_train):
        m_f = xgb.XGBClassifier(**p_xgb_wt).fit(train_xgb[xgb_cols].iloc[tr_i], y_train[tr_i])
        oof[va_i] = m_f.predict_proba(train_xgb[xgb_cols].iloc[va_i])[:, 1]
    candidate_oof["XGB_Weighted"] = prob_to_logit(oof)

    # 7. XGB Focal Loss
    print("  7/16. Optimizando XGBoost Focal Loss...")
    def obj_xgb_fc(trial):
        gamma = trial.suggest_float("gamma", 1.0, 3.0)
        alpha = trial.suggest_float("alpha", 0.3, 0.7)
        p = {
            "learning_rate": trial.suggest_float("lr", 0.03, 0.08),
            "max_depth": trial.suggest_int("md", 4, 6),
            "min_child_weight": trial.suggest_int("mcw", 20, 60),
            "colsample_bytree": trial.suggest_float("col", 0.65, 0.90),
            "subsample": trial.suggest_float("sub", 0.65, 0.90),
            "reg_alpha": trial.suggest_float("ra", 1e-2, 1.0, log=True),
            "reg_lambda": trial.suggest_float("rl", 0.5, 10.0, log=True),
            "n_estimators": 160, "tree_method": "hist", "enable_categorical": True,
            "objective": make_xgb_focal_loss(gamma, alpha), "random_state": seed, "n_jobs": 4, "verbosity": 0,
        }
        m = xgb.XGBClassifier(**p).fit(train_xgb[xgb_cols], y_train)
        raw = m.predict(tune_xgb[xgb_cols], output_margin=True)
        return pr_auc(y_tune, 1.0 / (1.0 + np.exp(-np.clip(raw, -25.0, 25.0))))

    s_xgb_fc = optuna.create_study(direction="maximize", sampler=CmaEsSampler(seed=seed))
    s_xgb_fc.optimize(obj_xgb_fc, n_trials=8)
    bp_xgb_fc = s_xgb_fc.best_params
    p_xgb_fc = {
        "learning_rate": bp_xgb_fc["lr"], "max_depth": bp_xgb_fc["md"], "min_child_weight": bp_xgb_fc["mcw"],
        "colsample_bytree": bp_xgb_fc["col"], "subsample": bp_xgb_fc["sub"], "reg_alpha": bp_xgb_fc["ra"],
        "reg_lambda": bp_xgb_fc["rl"], "n_estimators": 180, "tree_method": "hist", "enable_categorical": True,
        "objective": make_xgb_focal_loss(bp_xgb_fc["gamma"], bp_xgb_fc["alpha"]), "random_state": seed, "n_jobs": 4, "verbosity": 0,
    }
    m = xgb.XGBClassifier(**p_xgb_fc).fit(train_xgb[xgb_cols], y_train)
    raw_tu = m.predict(tune_xgb[xgb_cols], output_margin=True)
    raw_te = m.predict(test_xgb[xgb_cols], output_margin=True)
    candidate_tune["XGB_Focal"] = raw_tu
    candidate_test["XGB_Focal"] = raw_te
    candidate_val_prauc["XGB_Focal"] = pr_auc(y_tune, 1.0 / (1.0 + np.exp(-np.clip(raw_tu, -25.0, 25.0))))
    print(f"       -> val_search PR-AUC: {candidate_val_prauc['XGB_Focal']:.5f}")

    oof = np.zeros(len(train), dtype=float)
    for tr_i, va_i in skf.split(train, y_train):
        m_f = xgb.XGBClassifier(**p_xgb_fc).fit(train_xgb[xgb_cols].iloc[tr_i], y_train[tr_i])
        oof[va_i] = m_f.predict(train_xgb[xgb_cols].iloc[va_i], output_margin=True)
    candidate_oof["XGB_Focal"] = oof

    # 8. XGB Ranker (Pairwise)
    print("  8/16. Optimizando XGBoost Ranker...")
    def obj_xgb_rk(trial):
        p = {
            "learning_rate": trial.suggest_float("lr", 0.03, 0.08),
            "max_depth": trial.suggest_int("md", 4, 6),
            "colsample_bytree": trial.suggest_float("col", 0.65, 0.90),
            "subsample": trial.suggest_float("sub", 0.65, 0.90),
            "reg_alpha": trial.suggest_float("ra", 1e-2, 1.0, log=True),
            "reg_lambda": trial.suggest_float("rl", 0.5, 10.0, log=True),
            "n_estimators": 160, "tree_method": "hist", "enable_categorical": True,
            "objective": "rank:pairwise", "random_state": seed, "n_jobs": 4, "verbosity": 0,
        }
        ranker = xgb.XGBRanker(**p).fit(train_xgb[xgb_cols], y_train, qid=train_qid_xgb)
        return pr_auc(y_tune, ranker.predict(tune_xgb[xgb_cols]))

    s_xgb_rk = optuna.create_study(direction="maximize", sampler=CmaEsSampler(seed=seed))
    s_xgb_rk.optimize(obj_xgb_rk, n_trials=8)
    bp_xgb_rk = s_xgb_rk.best_params
    p_xgb_rk = {
        "learning_rate": bp_xgb_rk["lr"], "max_depth": bp_xgb_rk["md"],
        "colsample_bytree": bp_xgb_rk["col"], "subsample": bp_xgb_rk["sub"], "reg_alpha": bp_xgb_rk["ra"],
        "reg_lambda": bp_xgb_rk["rl"], "n_estimators": 180, "tree_method": "hist", "enable_categorical": True,
        "objective": "rank:pairwise", "random_state": seed, "n_jobs": 4, "verbosity": 0,
    }
    ranker = xgb.XGBRanker(**p_xgb_rk).fit(train_xgb[xgb_cols], y_train, qid=train_qid_xgb)
    s_tu = ranker.predict(tune_xgb[xgb_cols])
    s_te = ranker.predict(test_xgb[xgb_cols])
    candidate_tune["XGB_Ranker"] = normalize_scores_to_margin(s_tu)
    candidate_test["XGB_Ranker"] = normalize_scores_to_margin(s_te)
    candidate_val_prauc["XGB_Ranker"] = pr_auc(y_tune, s_tu)
    print(f"       -> val_search PR-AUC: {candidate_val_prauc['XGB_Ranker']:.5f}")

    oof = np.zeros(len(train), dtype=float)
    for tr_i, va_i in skf.split(train, y_train):
        m_f = xgb.XGBRanker(**p_xgb_rk).fit(train_xgb[xgb_cols].iloc[tr_i], y_train[tr_i], qid=train_qid_xgb[tr_i])
        oof[va_i] = m_f.predict(train_xgb[xgb_cols].iloc[va_i])
    candidate_oof["XGB_Ranker"] = normalize_scores_to_margin(oof)

    # -------------------------------------------------------------------------
    # FAMILIA 3: CATBOOST (4 Variantes)
    # -------------------------------------------------------------------------
    print("\n--- [FAMILIA 3: CATBOOST] ---")

    # 9. CatBoost Log-Loss
    print("  9/16. Entrenando CatBoost Log-Loss...")
    p_cat_ll = {
        "iterations": 220, "learning_rate": 0.045, "depth": 6, "l2_leaf_reg": 6.0,
        "cat_features": cat_indices, "random_seed": seed, "thread_count": 4, "verbose": 0,
    }
    m = CatBoostClassifier(**p_cat_ll).fit(train_cat[cat_cols], y_train)
    p_tu = m.predict_proba(tune_cat[cat_cols])[:, 1]
    p_te = m.predict_proba(test_cat[cat_cols])[:, 1]
    candidate_tune["CAT_LogLoss"] = prob_to_logit(p_tu)
    candidate_test["CAT_LogLoss"] = prob_to_logit(p_te)
    candidate_val_prauc["CAT_LogLoss"] = pr_auc(y_tune, p_tu)
    print(f"       -> val_search PR-AUC: {candidate_val_prauc['CAT_LogLoss']:.5f}")

    oof = np.zeros(len(train), dtype=float)
    for tr_i, va_i in skf.split(train, y_train):
        m_f = CatBoostClassifier(**p_cat_ll).fit(train_cat[cat_cols].iloc[tr_i], y_train[tr_i])
        oof[va_i] = m_f.predict_proba(train_cat[cat_cols].iloc[va_i])[:, 1]
    candidate_oof["CAT_LogLoss"] = prob_to_logit(oof)

    # 10. CatBoost Weighted (class_weights)
    print("  10/16. Entrenando CatBoost Weighted Loss (Ratio 3.5)...")
    p_cat_wt = {
        "iterations": 220, "learning_rate": 0.045, "depth": 6, "l2_leaf_reg": 6.0,
        "class_weights": [1.0, 3.5], "cat_features": cat_indices, "random_seed": seed, "thread_count": 4, "verbose": 0,
    }
    m = CatBoostClassifier(**p_cat_wt).fit(train_cat[cat_cols], y_train)
    p_tu = m.predict_proba(tune_cat[cat_cols])[:, 1]
    p_te = m.predict_proba(test_cat[cat_cols])[:, 1]
    candidate_tune["CAT_Weighted"] = prob_to_logit(p_tu)
    candidate_test["CAT_Weighted"] = prob_to_logit(p_te)
    candidate_val_prauc["CAT_Weighted"] = pr_auc(y_tune, p_tu)
    print(f"       -> val_search PR-AUC: {candidate_val_prauc['CAT_Weighted']:.5f}")

    oof = np.zeros(len(train), dtype=float)
    for tr_i, va_i in skf.split(train, y_train):
        m_f = CatBoostClassifier(**p_cat_wt).fit(train_cat[cat_cols].iloc[tr_i], y_train[tr_i])
        oof[va_i] = m_f.predict_proba(train_cat[cat_cols].iloc[va_i])[:, 1]
    candidate_oof["CAT_Weighted"] = prob_to_logit(oof)

    # 11. CatBoost Focal Loss
    print("  11/16. Entrenando CatBoost Focal Loss (gamma=2.0, alpha=0.5)...")
    p_cat_fc = {
        "iterations": 180, "learning_rate": 0.05, "depth": 6, "l2_leaf_reg": 6.0,
        "loss_function": CatBoostFocalObjective(2.0, 0.5), "eval_metric": "Logloss",
        "cat_features": cat_indices, "random_seed": seed, "thread_count": 4, "verbose": 0,
    }
    m = CatBoostClassifier(**p_cat_fc).fit(train_cat[cat_cols], y_train)
    raw_tu = m.predict(tune_cat[cat_cols], prediction_type="RawFormulaVal")
    raw_te = m.predict(test_cat[cat_cols], prediction_type="RawFormulaVal")
    candidate_tune["CAT_Focal"] = raw_tu
    candidate_test["CAT_Focal"] = raw_te
    candidate_val_prauc["CAT_Focal"] = pr_auc(y_tune, 1.0 / (1.0 + np.exp(-np.clip(raw_tu, -20.0, 20.0))))
    print(f"       -> val_search PR-AUC: {candidate_val_prauc['CAT_Focal']:.5f}")

    oof = np.zeros(len(train), dtype=float)
    for tr_i, va_i in skf.split(train, y_train):
        m_f = CatBoostClassifier(**p_cat_fc).fit(train_cat[cat_cols].iloc[tr_i], y_train[tr_i])
        oof[va_i] = m_f.predict(train_cat[cat_cols].iloc[va_i], prediction_type="RawFormulaVal")
    candidate_oof["CAT_Focal"] = oof

    # 12. CatBoost YetiRank
    print("  12/16. Entrenando CatBoost YetiRank...")
    p_cat_rk = {
        "iterations": 180, "learning_rate": 0.05, "depth": 6, "l2_leaf_reg": 6.0,
        "loss_function": "YetiRank", "cat_features": cat_indices, "random_seed": seed, "thread_count": 4, "verbose": 0,
    }
    pool_tr = Pool(train_cat[cat_cols], y_train, group_id=train_qid_xgb, cat_features=cat_indices)
    ranker = CatBoostRanker(**p_cat_rk).fit(pool_tr)
    s_tu = ranker.predict(tune_cat[cat_cols])
    s_te = ranker.predict(test_cat[cat_cols])
    candidate_tune["CAT_Ranker"] = normalize_scores_to_margin(s_tu)
    candidate_test["CAT_Ranker"] = normalize_scores_to_margin(s_te)
    candidate_val_prauc["CAT_Ranker"] = pr_auc(y_tune, s_tu)
    print(f"       -> val_search PR-AUC: {candidate_val_prauc['CAT_Ranker']:.5f}")

    oof = np.zeros(len(train), dtype=float)
    for tr_i, va_i in skf.split(train, y_train):
        pool_f = Pool(train_cat[cat_cols].iloc[tr_i], y_train[tr_i], group_id=train_qid_xgb[tr_i], cat_features=cat_indices)
        m_f = CatBoostRanker(**p_cat_rk).fit(pool_f)
        oof[va_i] = m_f.predict(train_cat[cat_cols].iloc[va_i])
    candidate_oof["CAT_Ranker"] = normalize_scores_to_margin(oof)

    # -------------------------------------------------------------------------
    # FAMILIA 4: RANDOM FOREST (4 Criterios Compatibles)
    # -------------------------------------------------------------------------
    print("\n--- [FAMILIA 4: RANDOM FOREST] ---")

    rf_variants = [
        ("RF_Gini", {"criterion": "gini", "class_weight": None}),
        ("RF_Entropy", {"criterion": "log_loss", "class_weight": None}),
        ("RF_Weighted_Gini", {"criterion": "gini", "class_weight": "balanced"}),
        ("RF_Weighted_Subsample", {"criterion": "log_loss", "class_weight": "balanced_subsample"}),
    ]

    for idx, (rf_name, rf_kwargs) in enumerate(rf_variants, 13):
        print(f"  {idx}/16. Entrenando Random Forest ({rf_name})...")
        m = RandomForestClassifier(n_estimators=70, max_depth=10, min_samples_leaf=30, max_features=0.7, random_state=seed, n_jobs=4, **rf_kwargs)
        m.fit(train_rf[rf_cols], y_train)
        p_tu = m.predict_proba(tune_rf[rf_cols])[:, 1]
        p_te = m.predict_proba(test_rf[rf_cols])[:, 1]
        candidate_tune[rf_name] = prob_to_logit(p_tu)
        candidate_test[rf_name] = prob_to_logit(p_te)
        candidate_val_prauc[rf_name] = pr_auc(y_tune, p_tu)
        print(f"       -> val_search PR-AUC: {candidate_val_prauc[rf_name]:.5f}")

        oof = np.zeros(len(train), dtype=float)
        for tr_i, va_i in skf.split(train, y_train):
            m_f = RandomForestClassifier(n_estimators=70, max_depth=10, min_samples_leaf=30, max_features=0.7, random_state=seed, n_jobs=4, **rf_kwargs)
            m_f.fit(train_rf[rf_cols].iloc[tr_i], y_train[tr_i])
            oof[va_i] = m_f.predict_proba(train_rf[rf_cols].iloc[va_i])[:, 1]
        candidate_oof[rf_name] = prob_to_logit(oof)

    # =========================================================================
    # 3. FORWARD MODEL SELECTION EN NIVEL 2
    # =========================================================================
    print("\n" + "=" * 95)
    print("3. FORWARD SELECTION DE MODELOS EN NIVEL 2 (BASE FIJA + MODELOS ADICIONALES)")
    print("Regla de Adjudicacion: Incremento minimo de +5 BPS (+0.0005 PR-AUC en val_search)")
    print("=" * 95)

    # Base fija (Sin ningun modelo de ML supervisado)
    fixed_base_cols = anom_cols + ["pct_heuristico_fraude", "distancia_log_cluster", "deficit_piso_cluster"]

    # Construir dataframes base fija
    df_fixed_tr = pd.DataFrame({c: train[c].to_numpy(dtype=float) for c in fixed_base_cols})
    df_fixed_tu = pd.DataFrame({c: tune[c].to_numpy(dtype=float) for c in fixed_base_cols})
    df_fixed_te = pd.DataFrame({c: test[c].to_numpy(dtype=float) for c in fixed_base_cols})

    # Ajustar Baseline Nivel 2
    scaler_base = StandardScaler().fit(df_fixed_tr)
    X_tr_base = pd.DataFrame(scaler_base.transform(df_fixed_tr), columns=fixed_base_cols)
    X_tu_base = pd.DataFrame(scaler_base.transform(df_fixed_tu), columns=fixed_base_cols)
    X_te_base = pd.DataFrame(scaler_base.transform(df_fixed_te), columns=fixed_base_cols)

    m_base = LogisticRegression(solver="saga", l1_ratio=1.0, C=0.20, max_iter=1000, random_state=seed).fit(X_tr_base, y_train)
    prauc_base_val = pr_auc(y_tune, m_base.predict_proba(X_tu_base)[:, 1])
    prauc_base_test = pr_auc(y_test, m_base.predict_proba(X_te_base)[:, 1])

    print(f"\n[PASO 0: BASELINE FIJO SIN MODELOS SUPERVISADOS]")
    print(f"Features: {len(fixed_base_cols)} anomalías, reglas y clusters.")
    print(f"val_search PR-AUC: {prauc_base_val:.5f} | test PR-AUC: {prauc_base_test:.5f}")

    selected_models: List[str] = []
    remaining_candidates = list(candidate_oof.keys())
    current_best_val = prauc_base_val
    current_best_test = prauc_base_test

    step = 1
    selection_history = [("Base Fija (Solo Anomalías + Reglas)", prauc_base_val, prauc_base_test, 0.0)]

    while remaining_candidates:
        best_candidate = None
        best_candidate_val = -1.0
        best_candidate_test = -1.0

        for cand in remaining_candidates:
            # Evaluar metamodelo con Base Fija + Modelos ya seleccionados + cand
            test_cols = list(selected_models) + [cand]
            X_tr_cand = df_fixed_tr.copy()
            X_tu_cand = df_fixed_tu.copy()
            X_te_cand = df_fixed_te.copy()

            for col in test_cols:
                X_tr_cand[col] = candidate_oof[col]
                X_tu_cand[col] = candidate_tune[col]
                X_te_cand[col] = candidate_test[col]

            all_cols = list(X_tr_cand.columns)
            sc = StandardScaler().fit(X_tr_cand)
            X_tr_sc = pd.DataFrame(sc.transform(X_tr_cand), columns=all_cols)
            X_tu_sc = pd.DataFrame(sc.transform(X_tu_cand), columns=all_cols)
            X_te_sc = pd.DataFrame(sc.transform(X_te_cand), columns=all_cols)

            m_meta = LogisticRegression(solver="saga", l1_ratio=1.0, C=0.20, max_iter=1000, random_state=seed).fit(X_tr_sc, y_train)
            val_sc = pr_auc(y_tune, m_meta.predict_proba(X_tu_sc)[:, 1])
            te_sc = pr_auc(y_test, m_meta.predict_proba(X_te_sc)[:, 1])

            if val_sc > best_candidate_val:
                best_candidate_val = val_sc
                best_candidate_test = te_sc
                best_candidate = cand

        delta_bps = (best_candidate_val - current_best_val) * 10000.0

        print(f"\n[PASO {step}] Mejor candidato evaluado: {best_candidate}")
        print(f"  * val_search PR-AUC: {best_candidate_val:.5f} (Ganancia: {delta_bps:+.1f} BPS)")

        if delta_bps >= 5.0:  # Umbral de ganancia mínima obligatoria
            selected_models.append(best_candidate)
            remaining_candidates.remove(best_candidate)
            current_best_val = best_candidate_val
            current_best_test = best_candidate_test
            selection_history.append((f"+ {best_candidate}", current_best_val, current_best_test, delta_bps))
            print(f"  >>> ACEPTADO en el ensamble (supera umbral de +5.0 BPS)")
            step += 1
        else:
            print(f"  >>> RECHAZADO: Ganancia de {delta_bps:+.1f} BPS es menor a la cota de +5.0 BPS.")
            print("  >>> CRITERIO DE PARADA ALCANZADO: El ensamble se ha estabilizado.")
            break

    # =========================================================================
    # 4. REPORTE FINAL Y DIAGNÓSTICO
    # =========================================================================
    print("\n" + "=" * 95)
    print("4. TRAYECTORIA AUDITADA DEL FORWARD SELECTION DE MODELOS")
    print("=" * 95)
    print(f"{'Paso / Modelo Agregado':<42s} | {'val_search':<12s} | {'test final':<12s} | {'Delta BPS Val'}")
    print("-" * 95)
    for name, v_sc, t_sc, d_bps in selection_history:
        print(f"{name:<42s} | {v_sc:.5f}      | {t_sc:.5f}      | {d_bps:+.1f} bps")

    print("\n" + "=" * 95)
    print("5. ENSAMBLE GANADOR CONGELADO Y COEFICIENTES EN NIVEL 2")
    print("=" * 95)
    print(f"Modelos Supervisados Seleccionados: {selected_models}")

    # Ajustar metamodelo final
    final_tr = df_fixed_tr.copy()
    final_tu = df_fixed_tu.copy()
    final_te = df_fixed_te.copy()
    for col in selected_models:
        final_tr[col] = candidate_oof[col]
        final_tu[col] = candidate_tune[col]
        final_te[col] = candidate_test[col]

    final_cols = list(final_tr.columns)
    sc = StandardScaler().fit(final_tr)
    X_tr_sc = pd.DataFrame(sc.transform(final_tr), columns=final_cols)
    X_tu_sc = pd.DataFrame(sc.transform(final_tu), columns=final_cols)
    X_te_sc = pd.DataFrame(sc.transform(final_te), columns=final_cols)

    m_final = LogisticRegression(solver="saga", l1_ratio=1.0, C=0.20, max_iter=1000, random_state=seed).fit(X_tr_sc, y_train)
    weights = pd.Series(m_final.coef_[0], index=final_cols).sort_values(ascending=False)

    print("\nPesos del Metamodelo Lasso L1 (C=0.20):")
    for feat, w in weights.items():
        if abs(w) > 1e-4:
            print(f"  * {feat:<30s}: {w:+.5f}")
        else:
            print(f"  * {feat:<30s}:  0.00000 (APAGADO POR L1)")

    final_val_prauc = pr_auc(y_tune, m_final.predict_proba(X_tu_sc)[:, 1])
    final_test_prauc = pr_auc(y_test, m_final.predict_proba(X_te_sc)[:, 1])

    print("\n" + "=" * 95)
    print("COMPARATIVA HISTÓRICA DEL PROYECTO")
    print("=" * 95)
    comparisons = [
        ("Control Ian (LightGBM 11 vars monomodal)", 0.48136, 0.44435),
        ("Champion Anterior (Cuatri-Modelo Clásico)", 0.49218, 0.45014),
        ("🔬 Variante Parsimonia Previa (Lasso C=0.10)", 0.49101, 0.45093),
        ("🏆 NUEVO GANADOR: Forward Selection Multi-Loss", final_val_prauc, final_test_prauc),
    ]
    print(f"{'Hito / Solución':<48s} | {'val_search':<12s} | {'test final':<12s} | {'Delta vs Ian'}")
    print("-" * 95)
    for name, v_sc, t_sc in comparisons:
        delta = t_sc - 0.44435
        print(f"{name:<48s} | {v_sc:.5f}      | {t_sc:.5f}      | {delta:+.5f} ({delta*10000:+.1f} bps)")

    print(f"\n[OK] Experimento maestro completado en {time.time() - t0:.1f} segundos.")


if __name__ == "__main__":
    main()
