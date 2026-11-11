"""
Paso 1, 2 y 3: Grid Search Fino en Nivel 2, Selección Formal a Ciegas de Test y Cierre Oficial.

Metodología:
- Nivel 1: Cuatri-Modelo con 14 variables de Challenger v3 (LGBM, XGBoost, CatBoost, RF).
- Paso 1: Grid Search fino en Nivel 2 sobre:
    * Matriz A (Acoplada): Predicciones + Anomalias + distancia_log + deficit_piso + ratio_mediana.
    * Matriz B (Desacoplada): Predicciones + Anomalias + distancia_log + deficit_piso (sin ratio_mediana).
    * Grilla: Lasso (10 valores de C), ElasticNet (12 combinaciones C x l1_ratio).
    * Evaluación: Estrictamente a ciegas de Test, sobre val_search (OOT).
- Paso 2: Selección Formal:
    * Si Matriz A supera a B por Delta >= +0.0003 (+3 bps), gana A.
    * Caso contrario (empate o victoria de B), gana B (parsimonia y menor colinealidad).
    * Congelamiento del modelo ganador.
- Paso 3: Inferencia Única en Test Final:
    * PR-AUC, ROC-AUC, Brier Score, Log-Loss.
    * Análisis de Política Financiera (Capital Recovery y Event Recall) con tau* de validación.
"""
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.supervised import (
    validation_blocks, pr_auc,
)
from src.models.train import lgb_model


BASE_CATEGORICAL_COLS = ["a", "g", "n", "o", "p"]


def build_clustering_features(df_train: pd.DataFrame, df_all: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def main():
    seed = 42
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    t0 = time.time()

    print("=" * 85)
    print("EJECUCIÓN DEL PLAN METODOLÓGICO: PASOS 1, 2 Y 3")
    print("GRID SEARCH FINO EN NIVEL 2 Y SELECCIÓN FORMAL A CIEGAS DE TEST")
    print("=" * 85)

    # 1. Cargar bases
    print("\n1. Cargando datos...")
    base_df = pd.read_parquet(ROOT / "data/processed/dataset_base.parquet")
    feat_df = pd.read_parquet(ROOT / "data/features/features_transaccionales.parquet")
    anom_df = pd.read_parquet(ROOT / "data/features/scores_anomalias.parquet")
    rules_df = pd.read_parquet(ROOT / "data/features/features_reglas_asociacion.parquet")[["transaction_id", "pct_heuristico_fraude"]]

    anom_cols = [c for c in anom_df.columns if c != "transaction_id"]
    if "score_heuristico_fraude_10" in feat_df.columns:
        feat_df = feat_df.drop(columns=["score_heuristico_fraude_10"])

    # 2. Clustering Semántico SVD d=30
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

    # Preprocesamiento por familia de modelos
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

    # Modelos Nivel 1
    print("\n2. Entrenando Nivel 1 (LGBM, XGBoost, CatBoost, RF)...")
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
    p_lgb_te = m_lgb.predict_proba(test_xgb[lgb_14])[:, 1]

    xgb_cols = ['score', 'o', 'te_j_bayes', 'es_madrugada', 'ratio_monto_l', 'diff_m_d', 'score_texto_nb', 'is_h_cero', 'flag_alto_riesgo_nop', 'ratio_monto_b', 'g', 'b']
    best_params_xgb = {
        "learning_rate": 0.052,
        "max_depth": 5,
        "min_child_weight": 35,
        "colsample_bytree": 0.80,
        "subsample": 0.80,
        "reg_alpha": 0.05,
        "reg_lambda": 1.2,
        "n_estimators": 210,
        "enable_categorical": True,
        "tree_method": "hist",
        "random_state": seed,
        "n_jobs": 4,
    }
    m_xgb = xgb.XGBClassifier(**best_params_xgb).fit(train_xgb[xgb_cols], y_train)
    p_xgb_tu = m_xgb.predict_proba(tune_xgb[xgb_cols])[:, 1]
    p_xgb_te = m_xgb.predict_proba(test_xgb[xgb_cols])[:, 1]

    cat_cols = [c for c in lgb_14]
    cat_indices = [cat_cols.index(c) for c in BASE_CATEGORICAL_COLS if c in cat_cols]
    m_cat = CatBoostClassifier(
        iterations=280, learning_rate=0.045, depth=6, l2_leaf_reg=6.0,
        cat_features=cat_indices, random_seed=seed, thread_count=4, verbose=0,
    )
    m_cat.fit(train_cat[cat_cols], y_train, eval_set=(tune_cat[cat_cols], y_tune), early_stopping_rounds=30)
    p_cat_tu = m_cat.predict_proba(tune_cat[cat_cols])[:, 1]
    p_cat_te = m_cat.predict_proba(test_cat[cat_cols])[:, 1]

    rf_cols = ['score', 'o', 'te_j_bayes', 'es_madrugada', 'diff_m_d', 'distancia_log_cluster']
    m_rf = RandomForestClassifier(n_estimators=80, max_depth=10, min_samples_leaf=30, max_features=0.7, random_state=seed, n_jobs=4)
    m_rf.fit(train_rf[rf_cols], y_train)
    p_rf_tu = m_rf.predict_proba(tune_rf[rf_cols])[:, 1]
    p_rf_te = m_rf.predict_proba(test_rf[rf_cols])[:, 1]

    # OOF Predictions
    print("\n3. Generando probabilidades OOF con 5-Fold Stratified CV...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    p_lgb_oof = np.zeros(len(train), dtype=float)
    p_xgb_oof = np.zeros(len(train), dtype=float)
    p_cat_oof = np.zeros(len(train), dtype=float)
    p_rf_oof = np.zeros(len(train), dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y_train), 1):
        m_lgb_f = lgb_model(best_params_lgb, seed, 4).fit(train_xgb[lgb_14].iloc[tr_idx], y_train[tr_idx])
        p_lgb_oof[va_idx] = m_lgb_f.predict_proba(train_xgb[lgb_14].iloc[va_idx])[:, 1]

        m_xgb_f = xgb.XGBClassifier(**best_params_xgb).fit(train_xgb[xgb_cols].iloc[tr_idx], y_train[tr_idx])
        p_xgb_oof[va_idx] = m_xgb_f.predict_proba(train_xgb[xgb_cols].iloc[va_idx])[:, 1]

        m_cat_f = CatBoostClassifier(iterations=250, learning_rate=0.045, depth=6, l2_leaf_reg=6.0, cat_features=cat_indices, random_seed=seed, thread_count=4, verbose=0)
        m_cat_f.fit(train_cat[cat_cols].iloc[tr_idx], y_train[tr_idx])
        p_cat_oof[va_idx] = m_cat_f.predict_proba(train_cat[cat_cols].iloc[va_idx])[:, 1]

        m_rf_f = RandomForestClassifier(n_estimators=80, max_depth=10, min_samples_leaf=30, max_features=0.7, random_state=seed, n_jobs=4)
        m_rf_f.fit(train_rf[rf_cols].iloc[tr_idx], y_train[tr_idx])
        p_rf_oof[va_idx] = m_rf_f.predict_proba(train_rf[rf_cols].iloc[va_idx])[:, 1]

    # 4. Construcción de Matriz A (Acoplada) y Matriz B (Desacoplada)
    print("\n4. Construyendo Matrices A y B para Nivel 2...")
    eps = 1e-7

    def get_meta_frames(include_ratio_med: bool):
        def build_df(p1, p2, p3, p4, part_df):
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
            if include_ratio_med:
                d["ratio_monto_mediana_cluster"] = part_df["ratio_monto_mediana_cluster"].to_numpy(dtype=float)
            return pd.DataFrame(d)

        X_tr = build_df(p_lgb_oof, p_xgb_oof, p_cat_oof, p_rf_oof, train)
        X_tu = build_df(p_lgb_tu, p_xgb_tu, p_cat_tu, p_rf_tu, tune)
        X_te = build_df(p_lgb_te, p_xgb_te, p_cat_te, p_rf_te, test)

        scaler = StandardScaler().fit(X_tr)
        cols = list(X_tr.columns)
        X_tr_sc = pd.DataFrame(scaler.transform(X_tr), columns=cols)
        X_tu_sc = pd.DataFrame(scaler.transform(X_tu), columns=cols)
        X_te_sc = pd.DataFrame(scaler.transform(X_te), columns=cols)
        return X_tr_sc, X_tu_sc, X_te_sc

    X_tr_A, X_tu_A, X_te_A = get_meta_frames(include_ratio_med=True)
    X_tr_B, X_tu_B, X_te_B = get_meta_frames(include_ratio_med=False)

    # =========================================================================
    # PASO 1: GRID SEARCH FINO EN NIVEL 2 (ESTRICTAMENTE EN VAL_SEARCH)
    # =========================================================================
    print("\n" + "=" * 85)
    print("PASO 1: GRID SEARCH FINO EN NIVEL 2 (EVALUADO ESTRICTAMENTE EN VAL_SEARCH)")
    print("=" * 85)

    c_lasso_vals = [0.01, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.50]
    c_elastic_vals = [0.05, 0.10, 0.20, 0.50]
    l1_ratios = [0.2, 0.5, 0.8]

    def run_grid(X_tr, X_tu, label_matriz):
        records = []
        # Lasso L1
        for c in c_lasso_vals:
            m = LogisticRegression(solver="saga", l1_ratio=1.0, C=c, max_iter=1000, random_state=seed).fit(X_tr, y_train)
            p_val = m.predict_proba(X_tu)[:, 1]
            score_val = pr_auc(y_tune, p_val)
            records.append({
                "matriz": label_matriz,
                "familia": "Lasso L1",
                "C": c,
                "l1_ratio": 1.0,
                "val_search": score_val,
                "model": m,
            })

        # ElasticNet L1+L2
        for c in c_elastic_vals:
            for l1_r in l1_ratios:
                m = LogisticRegression(solver="saga", l1_ratio=l1_r, C=c, penalty="elasticnet", max_iter=1000, random_state=seed).fit(X_tr, y_train)
                p_val = m.predict_proba(X_tu)[:, 1]
                score_val = pr_auc(y_tune, p_val)
                records.append({
                    "matriz": label_matriz,
                    "familia": f"ElasticNet (l1={l1_r})",
                    "C": c,
                    "l1_ratio": l1_r,
                    "val_search": score_val,
                    "model": m,
                })
        return records

    results_A = run_grid(X_tr_A, X_tu_A, "Matriz A (Acoplada)")
    results_B = run_grid(X_tr_B, X_tu_B, "Matriz B (Desacoplada)")

    df_res_A = pd.DataFrame(results_A)
    df_res_B = pd.DataFrame(results_B)

    best_A = df_res_A.sort_values("val_search", ascending=False).iloc[0]
    best_B = df_res_B.sort_values("val_search", ascending=False).iloc[0]

    print("\n--- TOP 5 CONFIGURACIONES EN MATRIZ A (ACOPLADA) EN VAL_SEARCH ---")
    for _, row in df_res_A.sort_values("val_search", ascending=False).head(5).iterrows():
        print(f"  * {row['familia']:<22} C={row['C']:<5.2f} -> val_search: {row['val_search']:.5f}")

    print("\n--- TOP 5 CONFIGURACIONES EN MATRIZ B (DESACOPLADA) EN VAL_SEARCH ---")
    for _, row in df_res_B.sort_values("val_search", ascending=False).head(5).iterrows():
        print(f"  * {row['familia']:<22} C={row['C']:<5.2f} -> val_search: {row['val_search']:.5f}")

    # =========================================================================
    # PASO 2: CRITERIO DE SELECCIÓN FORMAL (A CIEGAS DE TEST)
    # =========================================================================
    print("\n" + "=" * 85)
    print("PASO 2: CRITERIO DE SELECCIÓN FORMAL (A CIEGAS DE TEST)")
    print("=" * 85)

    delta_A_vs_B = best_A["val_search"] - best_B["val_search"]
    print(f"Mejor Matriz A: {best_A['familia']} C={best_A['C']} -> val_search = {best_A['val_search']:.5f}")
    print(f"Mejor Matriz B: {best_B['familia']} C={best_B['C']} -> val_search = {best_B['val_search']:.5f}")
    print(f"Diferencia (Delta A - B): {delta_A_vs_B:+.5f} ({delta_A_vs_B*10000:+.1f} bps)")

    TOLERANCIA = 0.0003  # +3 bps requeridos para justificar colinealidad en Nivel 2
    if delta_A_vs_B >= TOLERANCIA:
        ganador = "Matriz A (Acoplada)"
        modelo_ganador = best_A["model"]
        X_te_ganador = X_te_A
        X_tu_ganador = X_tu_A
        best_cfg = best_A
        justificacion = f"Matriz A superó a B por {delta_A_vs_B*10000:+.1f} bps >= +3.0 bps de tolerancia."
    else:
        ganador = "Matriz B (Desacoplada)"
        modelo_ganador = best_B["model"]
        X_te_ganador = X_te_B
        X_tu_ganador = X_tu_B
        best_cfg = best_B
        justificacion = f"Matriz B empató o superó el umbral de parsimonia (Delta = {delta_A_vs_B*10000:+.1f} bps < +3.0 bps). Se adopta la arquitectura desacoplada para evitar colinealidad en Nivel 2."

    print(f"\n[VEREDICTO DE SELECCIÓN CONGELADO]: {ganador}")
    print(f"Configuración Seleccionada: {best_cfg['familia']} con C={best_cfg['C']}")
    print(f"Justificación Formal: {justificacion}")

    # =========================================================================
    # PASO 3: INFERENCIA ÚNICA EN TEST FINAL Y CIERRE DEL REPORTE
    # =========================================================================
    print("\n" + "=" * 85)
    print("PASO 3: INFERENCIA ÚNICA EN TEST FINAL CON EL MODELO GANADOR CONGELADO")
    print("=" * 85)

    p_val_final = modelo_ganador.predict_proba(X_tu_ganador)[:, 1]
    p_test_final = modelo_ganador.predict_proba(X_te_ganador)[:, 1]

    pr_auc_test = pr_auc(y_test, p_test_final)
    roc_auc_test = roc_auc_score(y_test, p_test_final)
    brier_test = brier_score_loss(y_test, p_test_final)
    logloss_test = log_loss(y_test, p_test_final)

    # Umbral de costo óptimo fijado en validación
    # Minimizar Costo = 5 * FP + sum(monto_FN) en el split de validación
    best_cost = float("inf")
    best_tau = 0.03
    for tau in np.linspace(0.01, 0.10, 91):
        fp = ((p_val_final >= tau) & (y_tune == 0)).sum()
        fn_montos = tune.loc[tune.fraude == 1, "monto"].values[p_val_final[y_tune == 1] < tau]
        costo = 5.0 * fp + fn_montos.sum()
        if costo < best_cost:
            best_cost = costo
            best_tau = tau

    # Métricas de Política en Test con tau congelado
    test_montos_fraude = test.loc[test.fraude == 1, "monto"].values
    total_capital_fraude = test_montos_fraude.sum()

    preds_test_bin = (p_test_final >= best_tau).astype(int)
    tp_test = ((preds_test_bin == 1) & (y_test == 1)).sum()
    fp_test = ((preds_test_bin == 1) & (y_test == 0)).sum()
    fn_test = ((preds_test_bin == 0) & (y_test == 1)).sum()

    montos_salvados = test.loc[(test.fraude == 1) & (preds_test_bin == 1), "monto"].sum()
    montos_perdidos = test.loc[(test.fraude == 1) & (preds_test_bin == 0), "monto"].sum()
    capital_recovery_rate = (montos_salvados / total_capital_fraude) * 100.0
    event_recall = (tp_test / (tp_test + fn_test)) * 100.0

    print(f"\n--- MÉTRICAS OFICIALES DE TEST FINAL ---")
    print(f"  * PR-AUC en Test:                 {pr_auc_test:.5f} (Delta vs Control Ian: {pr_auc_test - 0.44435:+.5f})")
    print(f"  * ROC-AUC en Test:                {roc_auc_test:.5f}")
    print(f"  * Brier Score en Test:            {brier_test:.5f}")
    print(f"  * Log-Loss en Test:               {logloss_test:.5f}")
    print(f"  * Umbral Óptimo Congelado (tau*): {best_tau:.4f}")
    print(f"  * Falsos Positivos en Test:       {fp_test}")
    print(f"  * Falsos Negativos en Test:       {fn_test}")
    print(f"  * Event Recall en Fraude:         {event_recall:.2f}%")
    print(f"  * Capital Recovery Rate:          {capital_recovery_rate:.2f}% (${montos_salvados:,.2f} recuperados de ${total_capital_fraude:,.2f})")

    report_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ganador_paso2": ganador,
        "familia_optima": best_cfg["familia"],
        "C_optimo": float(best_cfg["C"]),
        "l1_ratio_optimo": float(best_cfg["l1_ratio"]),
        "val_search_ganador": float(best_cfg["val_search"]),
        "justificacion": justificacion,
        "metricas_test": {
            "pr_auc": float(pr_auc_test),
            "delta_vs_ian_test": float(pr_auc_test - 0.44435),
            "roc_auc": float(roc_auc_test),
            "brier_score": float(brier_test),
            "log_loss": float(logloss_test),
            "tau_optimo": float(best_tau),
            "tp": int(tp_test),
            "fp": int(fp_test),
            "fn": int(fn_test),
            "event_recall_pct": float(event_recall),
            "capital_recovery_pct": float(capital_recovery_rate),
            "capital_salvado": float(montos_salvados),
            "capital_total": float(total_capital_fraude),
        }
    }
    out_path = ROOT / "docs/RESULTADOS_CIERRE_OFICIAL_GRID_NIVEL2.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)
    print(f"\n[OK] Resultados guardados en {out_path}")
    print(f"[OK] Proceso completado en {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
