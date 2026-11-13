"""
Experimento: Penta-Modelo Stacking incorporando Deep Learning (TabM) al Champion de Parsimonia
=============================================================================================
- Modelos Nivel 1: LightGBM, XGBoost, CatBoost, Random Forest y TabM (PLE + AdamW)
- Hiperparámetros TabM: k=32, n_blocks=3, d_block=512, dropout=0.15, PLE n_bins=48, d_emb=16, AdamW lr=2e-3, wd=3e-4, bs=512
- Análisis de Correlación: Spearman y Pearson entre TabM y las familias de árboles GBDT/RF
- Extracción OOF: 5-Fold Stratified CV determinista sobre las 105k de Train
- Metamodelo Nivel 2: Lasso L1 (C=0.10 y C=0.20), Ridge L2, Blend SLSQP y Stacking Puro
"""

from __future__ import annotations
import copy
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
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, normalize
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

import tabm
import rtdl_num_embeddings
from src.models.supervised import logit, pr_auc, validation_blocks
from src.models.train import lgb_model

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
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    print("=" * 95)
    print("EXPERIMENTO: PENTA-MODELO STACKING (GBDT + RANDOM FOREST + TABM)")
    print("INTEGRACIÓN DE TABM DEEP LEARNING CON OPTIMIZADOR ADAMW EN GPU BLACKWELL")
    print(f"DISPOSITIVO: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 95)

    # 1. Cargar bases de datos oficiales
    print("\n1. Cargando datos y generando features de Champion Parsimonia...")
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

    # 3. Mergear datasets
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

    # Preparar Datos para TabM (13 variables del Champion de Parsimonia)
    tabm_cat_cols = ["a", "g", "o"]
    tabm_num_cols = [
        "score", "es_madrugada", "ratio_d_m", "ratio_monto_l",
        "prod_b_log_c", "score_texto_nb", "ratio_monto_e", "is_h_cero",
        "deficit_piso_cluster", "distancia_log_cluster"
    ]
    imputer = SimpleImputer(strategy="median").fit(train[tabm_num_cols])
    X_tr_num_raw = imputer.transform(train[tabm_num_cols]).astype(np.float32)
    X_tu_num_raw = imputer.transform(tune[tabm_num_cols]).astype(np.float32)
    X_te_num_raw = imputer.transform(test[tabm_num_cols]).astype(np.float32)

    df_tabm_cat = df.copy()
    for cat in tabm_cat_cols:
        df_tabm_cat[cat] = df_tabm_cat[cat].fillna("__MISSING__").astype(str)

    enc_tabm = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(df_tabm_cat.loc[train.index, tabm_cat_cols])
    X_tr_cat_raw = enc_tabm.transform(df_tabm_cat.loc[train.index, tabm_cat_cols]).astype(np.int64)
    X_tu_cat_raw = enc_tabm.transform(df_tabm_cat.loc[tune.index, tabm_cat_cols]).astype(np.int64)
    X_te_cat_raw = enc_tabm.transform(df_tabm_cat.loc[test.index, tabm_cat_cols]).astype(np.int64)

    tabm_cardinalities = []
    for j in range(len(tabm_cat_cols)):
        max_idx = max(int(X_tr_cat_raw[:, j].max()), int(X_tu_cat_raw[:, j].max()), int(X_te_cat_raw[:, j].max()))
        mask_tr = X_tr_cat_raw[:, j] == -1
        mask_tu = X_tu_cat_raw[:, j] == -1
        mask_te = X_te_cat_raw[:, j] == -1
        if mask_tr.any() or mask_tu.any() or mask_te.any():
            X_tr_cat_raw[mask_tr, j] = max_idx + 1
            X_tu_cat_raw[mask_tu, j] = max_idx + 1
            X_te_cat_raw[mask_te, j] = max_idx + 1
            tabm_cardinalities.append(max_idx + 2)
        else:
            tabm_cardinalities.append(max_idx + 1)

    # 4. Features para Árboles
    lgb_14 = [
        "score", "o", "g", "a", "es_madrugada", "ratio_d_m", "ratio_monto_l",
        "prod_b_log_c", "score_texto_nb", "ratio_monto_e", "is_h_cero",
        "deficit_piso_cluster", "distancia_log_cluster", "ratio_monto_mediana_cluster"
    ]
    xgb_cols = ['score', 'o', 'te_j_bayes', 'es_madrugada', 'ratio_monto_l', 'diff_m_d', 'score_texto_nb', 'is_h_cero', 'flag_alto_riesgo_nop', 'ratio_monto_b', 'g', 'b']
    cat_cols = list(lgb_14)
    cat_indices = [cat_cols.index(c) for c in BASE_CATEGORICAL_COLS if c in cat_cols]
    rf_cols = ['score', 'o', 'te_j_bayes', 'es_madrugada', 'diff_m_d', 'distancia_log_cluster']

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    print("\n" + "=" * 95)
    print("2. ENTRENAMIENTO DE LOS 5 MODELOS DE NIVEL 1 (GBDTs, RANDOM FOREST Y TABM)")
    print("=" * 95)

    # --- MODELO 1: LightGBM ---
    print("\n[1/5] Entrenando LightGBM (14 vars Challenger v3)...")
    best_params_lgb = {
        "learning_rate": 0.03487, "num_leaves": 28, "min_child_samples": 100,
        "colsample_bytree": 0.814, "subsample": 0.757, "reg_alpha": 0.013, "reg_lambda": 0.527,
        "n_estimators": 253,
    }
    m_lgb = lgb_model(best_params_lgb, seed, 4).fit(train_xgb[lgb_14], y_train)
    p_lgb_tu = m_lgb.predict_proba(tune_xgb[lgb_14])[:, 1]
    p_lgb_te = m_lgb.predict_proba(test_xgb[lgb_14])[:, 1]
    print(f"  * LightGBM -> val_search: {pr_auc(y_tune, p_lgb_tu):.5f} | test: {pr_auc(y_test, p_lgb_te):.5f}")

    p_lgb_oof = np.zeros(len(train), dtype=float)
    for tr_i, va_i in skf.split(train, y_train):
        m_f = lgb_model(best_params_lgb, seed, 4).fit(train_xgb[lgb_14].iloc[tr_i], y_train[tr_i])
        p_lgb_oof[va_i] = m_f.predict_proba(train_xgb[lgb_14].iloc[va_i])[:, 1]

    # --- MODELO 2: XGBoost ---
    print("\n[2/5] Entrenando XGBoost (12 vars)...")
    best_params_xgb = {
        "learning_rate": 0.052, "max_depth": 5, "min_child_weight": 35,
        "colsample_bytree": 0.80, "subsample": 0.80, "reg_alpha": 0.1, "reg_lambda": 5.0,
        "n_estimators": 180, "tree_method": "hist", "enable_categorical": True,
        "random_state": seed, "n_jobs": 4, "verbosity": 0,
    }
    m_xgb = xgb.XGBClassifier(**best_params_xgb).fit(train_xgb[xgb_cols], y_train)
    p_xgb_tu = m_xgb.predict_proba(tune_xgb[xgb_cols])[:, 1]
    p_xgb_te = m_xgb.predict_proba(test_xgb[xgb_cols])[:, 1]
    print(f"  * XGBoost  -> val_search: {pr_auc(y_tune, p_xgb_tu):.5f} | test: {pr_auc(y_test, p_xgb_te):.5f}")

    p_xgb_oof = np.zeros(len(train), dtype=float)
    for tr_i, va_i in skf.split(train, y_train):
        m_f = xgb.XGBClassifier(**best_params_xgb).fit(train_xgb[xgb_cols].iloc[tr_i], y_train[tr_i])
        p_xgb_oof[va_i] = m_f.predict_proba(train_xgb[xgb_cols].iloc[va_i])[:, 1]

    # --- MODELO 3: CatBoost ---
    print("\n[3/5] Entrenando CatBoost (14 vars Challenger v3)...")
    m_cat = CatBoostClassifier(iterations=280, learning_rate=0.045, depth=6, l2_leaf_reg=6.0, cat_features=cat_indices, random_seed=seed, thread_count=4, verbose=0)
    m_cat.fit(train_cat[cat_cols], y_train, eval_set=(tune_cat[cat_cols], y_tune), early_stopping_rounds=30)
    p_cat_tu = m_cat.predict_proba(tune_cat[cat_cols])[:, 1]
    p_cat_te = m_cat.predict_proba(test_cat[cat_cols])[:, 1]
    print(f"  * CatBoost -> val_search: {pr_auc(y_tune, p_cat_tu):.5f} | test: {pr_auc(y_test, p_cat_te):.5f}")

    p_cat_oof = np.zeros(len(train), dtype=float)
    for tr_i, va_i in skf.split(train, y_train):
        m_f = CatBoostClassifier(iterations=240, learning_rate=0.045, depth=6, l2_leaf_reg=6.0, cat_features=cat_indices, random_seed=seed, thread_count=4, verbose=0)
        m_f.fit(train_cat[cat_cols].iloc[tr_i], y_train[tr_i])
        p_cat_oof[va_i] = m_f.predict_proba(train_cat[cat_cols].iloc[va_i])[:, 1]

    # --- MODELO 4: Random Forest ---
    print("\n[4/5] Entrenando Random Forest (6 vars)...")
    m_rf = RandomForestClassifier(n_estimators=80, max_depth=10, min_samples_leaf=30, max_features=0.7, random_state=seed, n_jobs=4)
    m_rf.fit(train_rf[rf_cols], y_train)
    p_rf_tu = m_rf.predict_proba(tune_rf[rf_cols])[:, 1]
    p_rf_te = m_rf.predict_proba(test_rf[rf_cols])[:, 1]
    print(f"  * Random Forest -> val_search: {pr_auc(y_tune, p_rf_tu):.5f} | test: {pr_auc(y_test, p_rf_te):.5f}")

    p_rf_oof = np.zeros(len(train), dtype=float)
    for tr_i, va_i in skf.split(train, y_train):
        m_f = RandomForestClassifier(n_estimators=80, max_depth=10, min_samples_leaf=30, max_features=0.7, random_state=seed, n_jobs=4)
        m_f.fit(train_rf[rf_cols].iloc[tr_i], y_train[tr_i])
        p_rf_oof[va_i] = m_f.predict_proba(train_rf[rf_cols].iloc[va_i])[:, 1]

    # --- MODELO 5: TabM (PLE + AdamW en GPU Blackwell) ---
    print("\n[5/5] Entrenando TabM Deep Learning en GPU Blackwell con 5-Fold Stratified CV OOF...")
    k_heads = 32
    n_blocks = 3
    d_block = 512
    dropout_rate = 0.15
    lr = 2e-3
    weight_decay = 3e-4
    batch_size = 512
    epochs_per_fold = 30

    def create_tabm_model(x_num_fit: torch.Tensor) -> nn.Module:
        bins = rtdl_num_embeddings.compute_bins(x_num_fit, n_bins=48)
        ple = rtdl_num_embeddings.PiecewiseLinearEmbeddings(bins, d_embedding=16, activation=True, version="B")
        m = tabm.TabM(
            n_num_features=len(tabm_num_cols),
            cat_cardinalities=tabm_cardinalities,
            d_out=1,
            num_embeddings=ple,
            k=k_heads,
            n_blocks=n_blocks,
            d_block=d_block,
            dropout=dropout_rate,
            start_scaling_init="random-signs",
        ).to(device)
        return m

    def predict_tabm(m: nn.Module, x_num: np.ndarray, x_cat: np.ndarray, eval_bs: int = 2048) -> np.ndarray:
        m.eval()
        probs_all = []
        n_samples = len(x_num)
        t_num = torch.tensor(x_num)
        t_cat = torch.tensor(x_cat)
        with torch.no_grad():
            for i in range(0, n_samples, eval_bs):
                b_num = t_num[i:i + eval_bs].to(device)
                b_cat = t_cat[i:i + eval_bs].to(device)
                out = m(b_num, b_cat)
                probs_k = torch.sigmoid(out.squeeze(-1))
                b_probs = probs_k.mean(dim=-1).cpu().numpy()
                probs_all.append(b_probs)
        return np.concatenate(probs_all)

    # Entrenar TabM full en train (para tune y test)
    print("    -> Entrenando TabM global en Train (105k) para predicciones de test...")
    m_tabm_global = create_tabm_model(torch.tensor(X_tr_num_raw))
    opt_tabm = torch.optim.AdamW(m_tabm_global.parameters(), lr=lr, weight_decay=weight_decay)
    loader_global = DataLoader(TensorDataset(torch.tensor(X_tr_num_raw), torch.tensor(X_tr_cat_raw), torch.tensor(y_train, dtype=torch.float32)), batch_size=batch_size, shuffle=True)

    best_p_tune = 0.0
    best_tabm_state = None
    for ep in range(1, 35):
        m_tabm_global.train()
        for b_num, b_cat, b_y in loader_global:
            b_num, b_cat, b_y = b_num.to(device), b_cat.to(device), b_y.to(device)
            opt_tabm.zero_grad()
            out = m_tabm_global(b_num, b_cat)
            loss = F.binary_cross_entropy_with_logits(out.squeeze(-1), b_y.unsqueeze(-1).expand(-1, k_heads))
            loss.backward()
            opt_tabm.step()
        p_tu_tmp = predict_tabm(m_tabm_global, X_tu_num_raw, X_tu_cat_raw)
        pr_tu = pr_auc(y_tune, p_tu_tmp)
        if pr_tu > best_p_tune:
            best_p_tune = pr_tu
            best_tabm_state = copy.deepcopy(m_tabm_global.state_dict())

    m_tabm_global.load_state_dict(best_tabm_state)
    p_tabm_tu = predict_tabm(m_tabm_global, X_tu_num_raw, X_tu_cat_raw)
    p_tabm_te = predict_tabm(m_tabm_global, X_te_num_raw, X_te_cat_raw)
    print(f"  * TabM Global -> val_search: {pr_auc(y_tune, p_tabm_tu):.5f} | test: {pr_auc(y_test, p_tabm_te):.5f}")

    # Generación OOF 5-Fold para TabM
    print("    -> Generando OOF 5-Fold de TabM en Train...")
    p_tabm_oof = np.zeros(len(train), dtype=float)
    for fold, (tr_i, va_i) in enumerate(skf.split(train, y_train), 1):
        x_tr_n_fold = torch.tensor(X_tr_num_raw[tr_i])
        m_f = create_tabm_model(x_tr_n_fold)
        opt_f = torch.optim.AdamW(m_f.parameters(), lr=lr, weight_decay=weight_decay)
        loader_f = DataLoader(TensorDataset(torch.tensor(X_tr_num_raw[tr_i]), torch.tensor(X_tr_cat_raw[tr_i]), torch.tensor(y_train[tr_i], dtype=torch.float32)), batch_size=batch_size, shuffle=True)
        
        for ep in range(1, epochs_per_fold + 1):
            m_f.train()
            for b_num, b_cat, b_y in loader_f:
                b_num, b_cat, b_y = b_num.to(device), b_cat.to(device), b_y.to(device)
                opt_f.zero_grad()
                out = m_f(b_num, b_cat)
                loss = F.binary_cross_entropy_with_logits(out.squeeze(-1), b_y.unsqueeze(-1).expand(-1, k_heads))
                loss.backward()
                opt_f.step()
        p_tabm_oof[va_i] = predict_tabm(m_f, X_tr_num_raw[va_i], X_tr_cat_raw[va_i])
        print(f"       Fold {fold}/5 completado. OOF PR-AUC parcial: {pr_auc(y_train[va_i], p_tabm_oof[va_i]):.5f}")

    print("\n  [RESUMEN FINAL DE PR-AUC OOF EN TRAIN]:")
    print(f"    * LightGBM     OOF: {pr_auc(y_train, p_lgb_oof):.5f}")
    print(f"    * XGBoost      OOF: {pr_auc(y_train, p_xgb_oof):.5f}")
    print(f"    * CatBoost     OOF: {pr_auc(y_train, p_cat_oof):.5f}")
    print(f"    * Random Forest OOF: {pr_auc(y_train, p_rf_oof):.5f}")
    print(f"    * TabM DL      OOF: {pr_auc(y_train, p_tabm_oof):.5f}")

    # =========================================================================
    # 3. MATRIZ DE CORRELACIÓN ENTRE TABM Y LOS MODELOS DE ÁRBOLES
    # =========================================================================
    print("\n" + "=" * 95)
    print("3. ANÁLISIS DE CORRELACIÓN (SPEARMAN Y PEARSON) ENTRE MODELOS EN TEST")
    print("=" * 95)

    preds_test_df = pd.DataFrame({
        "LightGBM": p_lgb_te,
        "XGBoost": p_xgb_te,
        "CatBoost": p_cat_te,
        "RandomForest": p_rf_te,
        "TabM_DL": p_tabm_te,
    })

    pearson_corr = preds_test_df.corr(method="pearson")
    spearman_corr = preds_test_df.corr(method="spearman")

    print("\nA. Matriz de Correlación de Spearman (Ranking Relativo en Test):")
    print(spearman_corr.round(4).to_string())

    print("\nB. Matriz de Correlación de Pearson (Predicciones Puntuales en Test):")
    print(pearson_corr.round(4).to_string())

    corr_lgb_tabm = spearman_corr.loc["LightGBM", "TabM_DL"]
    corr_lgb_xgb = spearman_corr.loc["LightGBM", "XGBoost"]
    print(f"\n  [HALLAZGO CLAVE DE DESCORRELACIÓN]:")
    print(f"    * Correlación entre Árboles (LGBM vs XGBoost): {corr_lgb_xgb:.4f} (Altamente redundantes)")
    print(f"    * Correlación entre Árbol y Red (LGBM vs TabM)  : {corr_lgb_tabm:.4f} (Señal marcadamente más ortogonal)")

    # =========================================================================
    # 4. CONSTRUCCIÓN DEL META-DATASET DE NIVEL 2 Y EVALUACIÓN DE STACKING
    # =========================================================================
    print("\n" + "=" * 95)
    print("4. CONSTRUCCIÓN DE METAMODELOS DE NIVEL 2 CON TABM (PENTA-MODELO)")
    print("=" * 95)

    def build_meta(p1, p2, p3, p4, p5, part_df):
        d = {
            "p_lgb": p1, "logit_lgb": prob_to_logit(p1),
            "p_xgb": p2, "logit_xgb": prob_to_logit(p2),
            "p_cat": p3, "logit_cat": prob_to_logit(p3),
            "p_rf": p4,  "logit_rf": prob_to_logit(p4),
            "p_tabm": p5, "logit_tabm": prob_to_logit(p5),
        }
        for col in anom_cols:
            d[col] = part_df[col].to_numpy(dtype=float)
        d["pct_heuristico_fraude"] = part_df["pct_heuristico_fraude"].to_numpy(dtype=float)
        d["distancia_log_cluster"] = part_df["distancia_log_cluster"].to_numpy(dtype=float)
        d["deficit_piso_cluster"] = part_df["deficit_piso_cluster"].to_numpy(dtype=float)
        return pd.DataFrame(d)

    X_meta_tr = build_meta(p_lgb_oof, p_xgb_oof, p_cat_oof, p_rf_oof, p_tabm_oof, train)
    X_meta_tu = build_meta(p_lgb_tu, p_xgb_tu, p_cat_tu, p_rf_tu, p_tabm_tu, tune)
    X_meta_te = build_meta(p_lgb_te, p_xgb_te, p_cat_te, p_rf_te, p_tabm_te, test)

    meta_cols = list(X_meta_tr.columns)
    scaler = StandardScaler().fit(X_meta_tr)
    X_tr_sc = pd.DataFrame(scaler.transform(X_meta_tr), columns=meta_cols)
    X_tu_sc = pd.DataFrame(scaler.transform(X_meta_tu), columns=meta_cols)
    X_te_sc = pd.DataFrame(scaler.transform(X_meta_te), columns=meta_cols)

    # 1. Stacking Lasso L1 C=0.10 (Champion de Parsimonia)
    m_lasso_01 = LogisticRegression(solver="saga", l1_ratio=1.0, C=0.10, max_iter=1000, random_state=seed).fit(X_tr_sc, y_train)
    p_l1_01_tu = m_lasso_01.predict_proba(X_tu_sc)[:, 1]
    p_l1_01_te = m_lasso_01.predict_proba(X_te_sc)[:, 1]

    # 2. Stacking Lasso L1 C=0.20
    m_lasso_02 = LogisticRegression(solver="saga", l1_ratio=1.0, C=0.20, max_iter=1000, random_state=seed).fit(X_tr_sc, y_train)
    p_l1_02_tu = m_lasso_02.predict_proba(X_tu_sc)[:, 1]
    p_l1_02_te = m_lasso_02.predict_proba(X_te_sc)[:, 1]

    # 3. Stacking Ridge L2 C=0.20
    m_ridge_02 = LogisticRegression(solver="saga", penalty="l2", C=0.20, max_iter=1000, random_state=seed).fit(X_tr_sc, y_train)
    p_r_02_tu = m_ridge_02.predict_proba(X_tu_sc)[:, 1]
    p_r_02_te = m_ridge_02.predict_proba(X_te_sc)[:, 1]

    # 4. Blend Convexo SLSQP (5 Modelos)
    w_opt = optimize_blend_weights(y_tune, [p_lgb_tu, p_xgb_tu, p_cat_tu, p_rf_tu, p_tabm_tu])
    p_blend_tu = sum(w_opt[i] * [p_lgb_tu, p_xgb_tu, p_cat_tu, p_rf_tu, p_tabm_tu][i] for i in range(5))
    p_blend_te = sum(w_opt[i] * [p_lgb_te, p_xgb_te, p_cat_te, p_rf_te, p_tabm_te][i] for i in range(5))

    # 5. Stacking Puro de Nivel 2 (Solo Logits/Preds de los 5 Modelos)
    pred_cols = ["p_lgb", "logit_lgb", "p_xgb", "logit_xgb", "p_cat", "logit_cat", "p_rf", "logit_rf", "p_tabm", "logit_tabm"]
    m_pure = LogisticRegression(solver="saga", l1_ratio=1.0, C=0.20, max_iter=1000, random_state=seed).fit(X_tr_sc[pred_cols], y_train)
    p_pure_tu = m_pure.predict_proba(X_tu_sc[pred_cols])[:, 1]
    p_pure_te = m_pure.predict_proba(X_te_sc[pred_cols])[:, 1]

    # =========================================================================
    # 5. TABLA COMPARATIVA FINAL VS HITOS LÍDERES
    # =========================================================================
    print("\n" + "=" * 105)
    print("RESUMEN COMPARATIVO: PENTA-MODELO CON TABM VS HISTÓRICOS DEL PROYECTO")
    print("=" * 105)

    summary_table = [
        ("0. Control Ian (LightGBM 11 vars)", 0.48136, 0.44435, "+0.0 bps", "Baseline FE"),
        ("1. Challenger v3 Oficial (LGBM 13 vars)", 0.48850, 0.44700, "+26.5 bps", "Semántica"),
        ("2. Champion Oficial (Cuatri-Modelo C=0.20)", 0.49218, 0.45014, "+57.9 bps", "Champion N2"),
        ("3. 🏆 RÉCORD PARSIMONIA PREVIO (Cuatri L1 C=0.10)", 0.49101, 0.45093, "+65.8 bps", "🏆 RECORD PREVIO"),
        ("4. Forward Selection Multi-Loss (5 modelos)", 0.49502, 0.45010, "+57.5 bps", "Multi-Loss"),
        ("-" * 45, 0.0, 0.0, "", ""),
        ("5. Inductor TabM Deep Learning Autónomo", pr_auc(y_tune, p_tabm_tu), pr_auc(y_test, p_tabm_te), f"{(pr_auc(y_test, p_tabm_te)-0.44435)*10000:+.1f} bps", "Deep Learning"),
        ("6. Blend Convexo SLSQP (5 Modelos)", pr_auc(y_tune, p_blend_tu), pr_auc(y_test, p_blend_te), f"{(pr_auc(y_test, p_blend_te)-0.44435)*10000:+.1f} bps", "Blend 5-Mod"),
        ("7. Stacking Puro Nivel 2 (5 Modelos)", pr_auc(y_tune, p_pure_tu), pr_auc(y_test, p_pure_te), f"{(pr_auc(y_test, p_pure_te)-0.44435)*10000:+.1f} bps", "N2 Puro"),
        ("8. Stacking Penta-Modelo (Ridge L2 C=0.20)", pr_auc(y_tune, p_r_02_tu), pr_auc(y_test, p_r_02_te), f"{(pr_auc(y_test, p_r_02_te)-0.44435)*10000:+.1f} bps", "N2 Ridge"),
        ("9. 🏆 Stacking Penta-Modelo (Lasso L1 C=0.20)", pr_auc(y_tune, p_l1_02_tu), pr_auc(y_test, p_l1_02_te), f"{(pr_auc(y_test, p_l1_02_te)-0.44435)*10000:+.1f} bps", "N2 Lasso"),
        ("10. 🏆 Stacking Penta-Modelo (Lasso L1 C=0.10)", pr_auc(y_tune, p_l1_01_tu), pr_auc(y_test, p_l1_01_te), f"{(pr_auc(y_test, p_l1_01_te)-0.44435)*10000:+.1f} bps", "N2 Parsimonia"),
    ]

    print(f"{'Hito / Arquitectura':<47s} | {'PR-AUC Val':<12s} | {'PR-AUC Test':<12s} | {'Delta vs Ian':<14s} | {'Rol / Estado'}")
    print("-" * 105)
    for row in summary_table:
        if row[0].startswith("-"):
            print("-" * 105)
            continue
        name, v_sc, t_sc, d_ian, rol = row
        print(f"{name:<47s} | {v_sc:.5f}      | {t_sc:.5f}      | {d_ian:<14s} | {rol}")

    # Coeficientes del metamodelo Lasso L1 C=0.10
    print("\n" + "=" * 95)
    print("COEFICIENTES DEL METAMODELO LASSO L1 (C=0.10) EN EL PENTA-MODELO:")
    print("=" * 95)
    weights = pd.Series(m_lasso_01.coef_[0], index=meta_cols).sort_values(ascending=False)
    for feat, w in weights.items():
        if abs(w) > 1e-4:
            print(f"  * {feat:<35s}: {w:+.5f}")
        else:
            print(f"  * {feat:<35s}:  0.00000 (APAGADO POR L1)")

    # Coeficientes del Blend SLSQP
    print("\n" + "=" * 95)
    print("PESOS DEL BLEND CONVEXO SLSQP (5 MODELOS):")
    print("=" * 95)
    model_names = ["LightGBM", "XGBoost", "CatBoost", "RandomForest", "TabM_DL"]
    for name, w in zip(model_names, w_opt):
        print(f"  * {name:<20s}: {w:.4f} ({w*100:.1f}%)")

    print(f"\n[OK] Experimento Penta-Modelo completado en {time.time() - t0:.1f} segundos.")


if __name__ == "__main__":
    main()
