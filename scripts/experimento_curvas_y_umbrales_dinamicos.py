"""
Generación de Curvas Comparativas de Alta Calidad (PR y ROC) y
Experimento de Umbrales Operativos Dinámicos / Segmentados por Monto (tau*(monto)).
"""
from __future__ import annotations
import json
from pathlib import Path
import os
import random
import sys
import time
from typing import List, Tuple, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc,
    brier_score_loss, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.supervised import pr_auc, validation_blocks
from src.models.train import lgb_model

BASE_CATEGORICAL_COLS = ["a", "g", "n", "o", "p"]
BASE_NUMERIC_COLS = ["score", "monto", "b", "c", "d", "e", "f", "h", "l", "m", "q"]


def build_clustering_features(df_train: pd.DataFrame, df_all: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    print("  -> Calculando clustering semántico SVD d=30 k=50...")
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
    print("1. CARGA DE DATOS Y FEATURE ENGINEERING")
    print("=" * 85)

    base_df = pd.read_parquet(ROOT / "data/processed/dataset_base.parquet")
    feat_df = pd.read_parquet(ROOT / "data/features/features_transaccionales.parquet")
    anom_df = pd.read_parquet(ROOT / "data/features/scores_anomalias.parquet")
    rules_df = pd.read_parquet(ROOT / "data/features/features_reglas_asociacion.parquet")[["transaction_id", "pct_heuristico_fraude"]]

    anom_cols = [c for c in anom_df.columns if c != "transaction_id"]
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
    y_policy = policy.fraude.to_numpy()
    y_test = test.fraude.to_numpy()

    # Preprocesamiento categórico
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
    policy_xgb = df_xgb.loc[policy.index]
    test_xgb = df_xgb.loc[test.index]

    train_rf = df_rf.loc[df.split == "train"]
    tune_rf = df_rf.loc[tune.index]
    policy_rf = df_rf.loc[policy.index]
    test_rf = df_rf.loc[test.index]

    df_cat = df.copy()
    for cat in BASE_CATEGORICAL_COLS:
        df_cat[cat] = df_cat[cat].fillna("MISSING").astype(str)
    train_cat = df_cat.loc[df.split == "train"]
    tune_cat = df_cat.loc[tune.index]
    policy_cat = df_cat.loc[policy.index]
    test_cat = df_cat.loc[test.index]

    print("\n" + "=" * 85)
    print("2. ENTRENAMIENTO DE MODELOS PARA COMPARATIVA DE CURVAS")
    print("=" * 85)

    raw_cols = BASE_NUMERIC_COLS + [c for c in BASE_CATEGORICAL_COLS if c != "i"]
    selected_11_cols = [
        "score", "o", "g", "a", "es_madrugada", "ratio_d_m",
        "ratio_monto_l", "prod_b_log_c", "score_texto_nb", "ratio_monto_e", "is_h_cero"
    ]
    challenger_13_cols = selected_11_cols + ["deficit_piso_cluster", "distancia_log_cluster"]
    champion_14_cols = challenger_13_cols + ["ratio_monto_mediana_cluster"]

    # Modelo 0: Árbol Crudo (LightGBM sin FE)
    print("  -> Modelo 0: LightGBM Crudo (16 variables de fábrica)...")
    m_raw = lgb_model({}, seed, 4).fit(train_xgb[raw_cols], y_train)
    p_raw_test = m_raw.predict_proba(test_xgb[raw_cols])[:, 1]
    p_raw_val = m_raw.predict_proba(tune_xgb[raw_cols])[:, 1]

    # Modelo 1: Control Ian (LightGBM 11 vars seleccionadas)
    print("  -> Modelo 1: LightGBM Control Ian (11 variables Ola 1)...")
    m_11 = lgb_model({}, seed, 4).fit(train_xgb[selected_11_cols], y_train)
    p_11_test = m_11.predict_proba(test_xgb[selected_11_cols])[:, 1]
    p_11_val = m_11.predict_proba(tune_xgb[selected_11_cols])[:, 1]

    # Modelo 2: Baseline Comercial (LightGBM 13 vars Optuna)
    print("  -> Modelo 2: LightGBM Baseline Comercial (13 vars Optuna)...")
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
    m_13 = lgb_model(best_params_lgb, seed, 4).fit(train_xgb[challenger_13_cols], y_train)
    p_13_test = m_13.predict_proba(test_xgb[challenger_13_cols])[:, 1]
    p_13_val = m_13.predict_proba(tune_xgb[challenger_13_cols])[:, 1]

    # Modelos Base de Nivel 1 para Stacking Cuatri-Modelo
    print("  -> Entrenando inductores N1 para Stacking Champion...")
    m_lgb_champ = lgb_model(best_params_lgb, seed, 4).fit(train_xgb[champion_14_cols], y_train)
    p_lgb_tu = m_lgb_champ.predict_proba(tune_xgb[champion_14_cols])[:, 1]
    p_lgb_pol = m_lgb_champ.predict_proba(policy_xgb[champion_14_cols])[:, 1]
    p_lgb_te = m_lgb_champ.predict_proba(test_xgb[champion_14_cols])[:, 1]

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
    p_xgb_pol = m_xgb.predict_proba(policy_xgb[xgb_cols])[:, 1]
    p_xgb_te = m_xgb.predict_proba(test_xgb[xgb_cols])[:, 1]

    cat_cols = [c for c in champion_14_cols]
    cat_indices = [cat_cols.index(c) for c in BASE_CATEGORICAL_COLS if c in cat_cols]
    m_cat = CatBoostClassifier(
        iterations=280, learning_rate=0.045, depth=6, l2_leaf_reg=6.0,
        cat_features=cat_indices, random_seed=seed, thread_count=4, verbose=0,
    )
    m_cat.fit(train_cat[cat_cols], y_train, eval_set=(tune_cat[cat_cols], y_tune), early_stopping_rounds=30)
    p_cat_tu = m_cat.predict_proba(tune_cat[cat_cols])[:, 1]
    p_cat_pol = m_cat.predict_proba(policy_cat[cat_cols])[:, 1]
    p_cat_te = m_cat.predict_proba(test_cat[cat_cols])[:, 1]

    rf_cols = ['score', 'o', 'te_j_bayes', 'es_madrugada', 'diff_m_d', 'distancia_log_cluster']
    m_rf = RandomForestClassifier(n_estimators=80, max_depth=10, min_samples_leaf=30, max_features=0.7, random_state=seed, n_jobs=4)
    m_rf.fit(train_rf[rf_cols], y_train)
    p_rf_tu = m_rf.predict_proba(tune_rf[rf_cols])[:, 1]
    p_rf_pol = m_rf.predict_proba(policy_rf[rf_cols])[:, 1]
    p_rf_te = m_rf.predict_proba(test_rf[rf_cols])[:, 1]

    # OOF de Train para Stacking N2
    print("  -> Generando OOF de Train con Stratified 5-Fold...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    p_lgb_oof = np.zeros(len(train), dtype=float)
    p_xgb_oof = np.zeros(len(train), dtype=float)
    p_cat_oof = np.zeros(len(train), dtype=float)
    p_rf_oof = np.zeros(len(train), dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y_train), 1):
        m_lgb_f = lgb_model(best_params_lgb, seed, 4).fit(train_xgb[champion_14_cols].iloc[tr_idx], y_train[tr_idx])
        p_lgb_oof[va_idx] = m_lgb_f.predict_proba(train_xgb[champion_14_cols].iloc[va_idx])[:, 1]

        m_xgb_f = xgb.XGBClassifier(**best_params_xgb).fit(train_xgb[xgb_cols].iloc[tr_idx], y_train[tr_idx])
        p_xgb_oof[va_idx] = m_xgb_f.predict_proba(train_xgb[xgb_cols].iloc[va_idx])[:, 1]

        m_cat_f = CatBoostClassifier(iterations=250, learning_rate=0.045, depth=6, l2_leaf_reg=6.0, cat_features=cat_indices, random_seed=seed, thread_count=4, verbose=0)
        m_cat_f.fit(train_cat[cat_cols].iloc[tr_idx], y_train[tr_idx])
        p_cat_oof[va_idx] = m_cat_f.predict_proba(train_cat[cat_cols].iloc[va_idx])[:, 1]

        m_rf_f = RandomForestClassifier(n_estimators=80, max_depth=10, min_samples_leaf=30, max_features=0.7, random_state=seed, n_jobs=4)
        m_rf_f.fit(train_rf[rf_cols].iloc[tr_idx], y_train[tr_idx])
        p_rf_oof[va_idx] = m_rf_f.predict_proba(train_rf[rf_cols].iloc[va_idx])[:, 1]

    # Construcción de la Matriz Filtrada Manual (14 cols) para el Champion
    # Poda: Solo logits de modelos base (sin p duplicada) + anomalías no redundantes
    eps = 1e-7
    ratio_d_m_median = float(train["ratio_d_m"].median())

    def get_meta_frames_filtered():
        def build_df(p1, p2, p3, p4, part_df):
            d = {
                # 4 Logits de Nivel 1 (función de enlace lineal)
                "logit_lgb": np.log(np.clip(p1, eps, 1-eps) / (1-np.clip(p1, eps, 1-eps))),
                "logit_xgb": np.log(np.clip(p2, eps, 1-eps) / (1-np.clip(p2, eps, 1-eps))),
                "logit_cat": np.log(np.clip(p3, eps, 1-eps) / (1-np.clip(p3, eps, 1-eps))),
                "logit_rf":  np.log(np.clip(p4, eps, 1-eps) / (1-np.clip(p4, eps, 1-eps))),
                # 4 Anomalías Ortogonales Deduplicadas
                "pca_q": part_df["pca_q"].to_numpy(dtype=float),
                "score_ocsvm_rbf_nu_003": part_df["score_ocsvm_rbf_nu_003"].to_numpy(dtype=float),
                "score_iforest_perpendicular": part_df["score_iforest_perpendicular"].to_numpy(dtype=float),
                "score_lof_manhattan": part_df["score_lof_manhattan"].to_numpy(dtype=float),
                # 6 Variables Contextuales de Passthrough (imputando NaNs residuales si existieran)
                "pct_heuristico_fraude": part_df["pct_heuristico_fraude"].to_numpy(dtype=float),
                "distancia_log_cluster": part_df["distancia_log_cluster"].to_numpy(dtype=float),
                "deficit_piso_cluster": part_df["deficit_piso_cluster"].to_numpy(dtype=float),
                "ratio_d_m": part_df["ratio_d_m"].fillna(ratio_d_m_median).to_numpy(dtype=float),
                "es_madrugada": part_df["es_madrugada"].to_numpy(dtype=float),
                "score": part_df["score"].to_numpy(dtype=float),
            }
            return pd.DataFrame(d)

        X_tr = build_df(p_lgb_oof, p_xgb_oof, p_cat_oof, p_rf_oof, train)
        X_tu = build_df(p_lgb_tu, p_xgb_tu, p_cat_tu, p_rf_tu, tune)
        X_pol = build_df(p_lgb_pol, p_xgb_pol, p_cat_pol, p_rf_pol, policy)
        X_te = build_df(p_lgb_te, p_xgb_te, p_cat_te, p_rf_te, test)

        scaler = StandardScaler().fit(X_tr)
        cols = list(X_tr.columns)
        return (
            pd.DataFrame(scaler.transform(X_tr), columns=cols),
            pd.DataFrame(scaler.transform(X_tu), columns=cols),
            pd.DataFrame(scaler.transform(X_pol), columns=cols),
            pd.DataFrame(scaler.transform(X_te), columns=cols),
        )

    X_tr_F, X_tu_F, X_pol_F, X_te_F = get_meta_frames_filtered()

    # Champion Oficial: Lasso L1 C=0.20 sobre Matriz Filtrada (liblinear es instantáneo y exacto)
    print("  -> Ajustando Metamodelo Champion Oficial (Lasso C=0.20)...")
    m_champ = LogisticRegression(solver="liblinear", penalty="l1", C=0.20, random_state=seed).fit(X_tr_F, y_train)
    p_champ_val = m_champ.predict_proba(X_tu_F)[:, 1]
    p_champ_pol = m_champ.predict_proba(X_pol_F)[:, 1]
    p_champ_test = m_champ.predict_proba(X_te_F)[:, 1]

    # Variante Parsimonia: Lasso L1 C=0.10
    print("  -> Ajustando Metamodelo Parsimonia (Lasso C=0.10)...")
    m_parsim = LogisticRegression(solver="liblinear", penalty="l1", C=0.10, random_state=seed).fit(X_tr_F, y_train)
    p_parsim_test = m_parsim.predict_proba(X_te_F)[:, 1]

    # Persistencia de Predicciones del Champion para experimentación ultrarrápida
    pred_export = pd.concat([
        pd.DataFrame({
            "transaction_id": tune["transaction_id"].values, "split": "val_tune",
            "fraude": y_tune, "monto": tune["monto"].values,
            "deficit_piso_cluster": tune["deficit_piso_cluster"].values,
            "p_champ": p_champ_val,
        }),
        pd.DataFrame({
            "transaction_id": policy["transaction_id"].values, "split": "val_policy",
            "fraude": y_policy, "monto": policy["monto"].values,
            "deficit_piso_cluster": policy["deficit_piso_cluster"].values,
            "p_champ": p_champ_pol,
        }),
        pd.DataFrame({
            "transaction_id": test["transaction_id"].values, "split": "test",
            "fraude": y_test, "monto": test["monto"].values,
            "deficit_piso_cluster": test["deficit_piso_cluster"].values,
            "p_champ": p_champ_test,
        }),
    ], ignore_index=True)
    pred_export.to_parquet(ROOT / "data/processed/champion_predictions.parquet", index=False)
    print("  [OK] Predicciones del Champion persistidas en data/processed/champion_predictions.parquet")

    print("\n" + "=" * 85)
    print("3. GENERACIÓN DE CURVAS VECTORIALES (PR & ROC) PARA EL REPORTE")
    print("=" * 85)

    figures_dir = ROOT / "docs/figures_champion"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    colors = {
        "raw": "#757575",
        "control": "#1976D2",
        "challenger": "#F57C00",
        "champion": "#2E7D32",
        "parsimonia": "#7B1FA2",
    }

    models_curves = [
        ("0. Árbol Crudo (sin FE)", p_raw_test, colors["raw"], "--", 1.8),
        ("1. Control Ian (11 vars)", p_11_test, colors["control"], "-.", 2.0),
        ("2. Baseline Comercial (13 vars Optuna)", p_13_test, colors["challenger"], "-", 2.2),
        ("3. Champion Oficial (Stacking C=0.20)", p_champ_test, colors["champion"], "-", 2.8),
        ("4. Variante Parsimonia (Stacking C=0.10)", p_parsim_test, colors["parsimonia"], ":", 2.2),
    ]

    # --- FIGURA 1: CURVA PRECISION-RECALL EN TEST ---
    fig, ax = plt.subplots(figsize=(9, 6.5), dpi=300)
    base_rate = float(y_test.mean())
    ax.axhline(base_rate, color="#D32F2F", linestyle=":", linewidth=1.5, label=f"Clasificador Aleatorio ({base_rate*100:.2f}%)")

    for name, p_preds, color, ls, lw in models_curves:
        prec, rec, _ = precision_recall_curve(y_test, p_preds)
        score = pr_auc(y_test, p_preds)
        ax.plot(rec, prec, label=f"{name} (PR-AUC = {score:.4f})", color=color, linestyle=ls, linewidth=lw)

    ax.set_title("Curva Precision-Recall Fuera de Muestra (Test Final, N=22.500)\nEvolución Metodológica del Proyecto MELI Fraude", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Recall (Sensibilidad a Fraudes)", fontsize=11, labelpad=8)
    ax.set_ylabel("Precision (Valor Predictivo Positivo)", fontsize=11, labelpad=8)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=9.5)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    pr_svg = figures_dir / "curva_pr_comparativa_test.svg"
    pr_png = figures_dir / "curva_pr_comparativa_test.png"
    fig.savefig(str(pr_svg), format="svg", bbox_inches="tight")
    fig.savefig(str(pr_png), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Curva PR guardada en: {pr_svg} y {pr_png}")

    # --- FIGURA 2: CURVA ROC EN TEST ---
    fig, ax = plt.subplots(figsize=(9, 6.5), dpi=300)
    ax.plot([0, 1], [0, 1], color="#9E9E9E", linestyle=":", linewidth=1.5, label="Diagonal de Azar (ROC-AUC = 0.5000)")

    for name, p_preds, color, ls, lw in models_curves:
        fpr, tpr, _ = roc_curve(y_test, p_preds)
        roc_val = roc_auc_score(y_test, p_preds)
        ax.plot(fpr, tpr, label=f"{name} (ROC-AUC = {roc_val:.4f})", color=color, linestyle=ls, linewidth=lw)

    ax.set_title("Curva ROC Fuera de Muestra (Test Final, N=22.500)\nComparativa de Separabilidad Global", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Tasa de Falsos Positivos (1 - Especificidad)", fontsize=11, labelpad=8)
    ax.set_ylabel("Tasa de Verdaderos Positivos (Recall)", fontsize=11, labelpad=8)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=9.5)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    roc_svg = figures_dir / "curva_roc_comparativa_test.svg"
    roc_png = figures_dir / "curva_roc_comparativa_test.png"
    fig.savefig(str(roc_svg), format="svg", bbox_inches="tight")
    fig.savefig(str(roc_png), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Curva ROC guardada en: {roc_svg} y {roc_png}")

    # =========================================================================
    print("\n" + "=" * 85)
    print("4. EXPERIMENTO: ARQUITECTURA DE DECISIÓN PRODUCTIVA Y UMBRALES DINÁMICOS")
    print("   (S-CURVE, UMBRAL 2D SEMÁNTICO Y DECISIÓN TRIPARTITA / STEP-UP AUTH)")
    print("=" * 85)
    # =========================================================================

    montos_val = policy["monto"].values
    deficit_val = policy["deficit_piso_cluster"].values
    montos_test = test["monto"].values
    deficit_test = test["deficit_piso_cluster"].values
    total_fraud_test_capital = float(montos_test[y_test == 1].sum())

    # -------------------------------------------------------------------------
    # POLÍTICA 0: Umbral Estático Consigna (Control OOT, FP_val <= 100)
    # -------------------------------------------------------------------------
    taus = np.linspace(0.01, 0.90, 891)
    best_cost_p0 = float("inf")
    tau_p0 = 0.2550
    for tau in taus:
        preds = (p_champ_pol >= tau).astype(int)
        fp = ((preds == 1) & (y_policy == 0)).sum()
        if fp <= 100:
            fn = ((preds == 0) & (y_policy == 1)).sum()
            costo = 5.0 * fp + 200.0 * fn
            if costo < best_cost_p0:
                best_cost_p0 = costo
                tau_p0 = tau

    # -------------------------------------------------------------------------
    # POLÍTICA 1: Segmentado Bimodal Ticket ($25, FP_val <= 100)
    # -------------------------------------------------------------------------
    SPLIT_TICKET = 25.0
    val_low_mask = montos_val <= SPLIT_TICKET
    val_high_mask = montos_val > SPLIT_TICKET

    best_cost_bimodal = float("inf")
    tau_low_opt = tau_p0
    tau_high_opt = tau_p0

    for t_low in np.linspace(0.15, 0.45, 31):
        preds_low = (p_champ_pol[val_low_mask] >= t_low).astype(int)
        fp_low = ((preds_low == 1) & (y_policy[val_low_mask] == 0)).sum()
        for t_high in np.linspace(0.08, 0.35, 28):
            preds_high = (p_champ_pol[val_high_mask] >= t_high).astype(int)
            fp_high = ((preds_high == 1) & (y_policy[val_high_mask] == 0)).sum()
            if (fp_low + fp_high) <= 100:
                fn_low = ((preds_low == 0) & (y_policy[val_low_mask] == 1)).sum()
                fn_high = ((preds_high == 0) & (y_policy[val_high_mask] == 1)).sum()
                costo_bimodal = 5.0 * (fp_low + fp_high) + 200.0 * (fn_low + fn_high)
                if costo_bimodal < best_cost_bimodal:
                    best_cost_bimodal = costo_bimodal
                    tau_low_opt = t_low
                    tau_high_opt = t_high

    # -------------------------------------------------------------------------
    # POLÍTICA 2: Umbral Sigmoideo Acotado (S-Curve: tau_min a tau_max con FP_val <= 100)
    # tau(monto) = tau_min + (tau_max - tau_min) / (1 + exp((monto - M0) / sigma_m))
    # -------------------------------------------------------------------------
    M0 = float(np.median(montos_val[y_policy == 1]))  # ~ $49.65 mediana de fraude
    best_scurve_cost = float("inf")
    scurve_params = (0.35, 0.10, 30.0)

    for t_max in [0.30, 0.35, 0.40]:
        for t_min in [0.08, 0.10, 0.14, 0.18]:
            for sigma in [20.0, 35.0, 50.0]:
                tau_s = t_min + (t_max - t_min) / (1.0 + np.exp((montos_val - M0) / sigma))
                preds = (p_champ_pol >= tau_s).astype(int)
                fp = ((preds == 1) & (y_policy == 0)).sum()
                if fp <= 100:
                    fn = ((preds == 0) & (y_policy == 1)).sum()
                    costo = 5.0 * fp + 200.0 * fn
                    if costo < best_scurve_cost:
                        best_scurve_cost = costo
                        scurve_params = (t_max, t_min, sigma)

    t_max_opt, t_min_opt, sigma_opt = scurve_params
    def compute_tau_scurve(m):
        return t_min_opt + (t_max_opt - t_min_opt) / (1.0 + np.exp((m - M0) / sigma_opt))

    # -------------------------------------------------------------------------
    # POLÍTICA 3: Umbral 2D Semántico: tau(monto, deficit) = tau_S(monto) - alpha * deficit
    # Calibrado con FP_val <= 100
    # -------------------------------------------------------------------------
    best_2d_cost = float("inf")
    alpha_opt = 0.05
    for a_cand in np.linspace(0.01, 0.15, 29):
        tau_2d_val = np.clip(compute_tau_scurve(montos_val) - a_cand * deficit_val, 0.05, 0.90)
        preds = (p_champ_pol >= tau_2d_val).astype(int)
        fp = ((preds == 1) & (y_policy == 0)).sum()
        if fp <= 100:
            fn = ((preds == 0) & (y_policy == 1)).sum()
            costo = 5.0 * fp + 200.0 * fn
            if costo < best_2d_cost:
                best_2d_cost = costo
                alpha_opt = a_cand

    def compute_tau_2d(m, def_c):
        return np.clip(compute_tau_scurve(m) - alpha_opt * def_c, 0.05, 0.90)

    # -------------------------------------------------------------------------
    # POLÍTICA 4: Decisión Tripartita (Three-Way Decision / Step-up Auth 2FA)
    # - Green Lane (p < tau_challenge): Fricción cero
    # - Yellow Lane (tau_challenge <= p < tau_block): Desafío 2FA/Biometría (costo fricción $0.05)
    #   * Cliente legítimo pasa el 2FA con 95% éxito (costo $0.05).
    #   * Fraude es detenido en el 2FA con 90% éxito.
    # - Red Lane (p >= tau_block): Bloqueo directo (costo de FP = $25 por daño de marca)
    # -------------------------------------------------------------------------
    best_3way_cost = float("inf")
    beta_challenge_opt = 0.60
    for beta in np.linspace(0.40, 0.85, 19):
        t_block_val = compute_tau_2d(montos_val, deficit_val)
        t_chal_val = beta * t_block_val
        preds_block = (p_champ_pol >= t_block_val).astype(int)
        preds_chal = ((p_champ_pol >= t_chal_val) & (p_champ_pol < t_block_val)).astype(int)
        fp_block = ((preds_block == 1) & (y_policy == 0)).sum()
        if fp_block <= 100:
            fp_chal = ((preds_chal == 1) & (y_policy == 0)).sum()
            # Fraudes detenidos: todos los del bloque rojo + 90% de los del bloque amarillo
            tp_detenidos = ((preds_block == 1) & (y_policy == 1)).sum() + 0.90 * ((preds_chal == 1) & (y_policy == 1)).sum()
            total_fraude_val = (y_policy == 1).sum()
            fn_remanentes = total_fraude_val - tp_detenidos
            # Función de costo del sistema: 25 * FP_bloqueo + 0.05 * FP_challenge + 200 * FN
            costo_3way = 25.0 * fp_block + 0.05 * fp_chal + 200.0 * fn_remanentes
            if costo_3way < best_3way_cost:
                best_3way_cost = costo_3way
                beta_challenge_opt = beta

    # -------------------------------------------------------------------------
    # EVALUACIÓN DE TODAS LAS POLÍTICAS EN TEST FINAL (FUERA DE MUESTRA)
    # -------------------------------------------------------------------------
    def eval_policy_test(decision_rule_fn, name):
        preds_bin = decision_rule_fn(p_champ_test, montos_test, deficit_test)
        tp = int(((preds_bin == 1) & (y_test == 1)).sum())
        fp = int(((preds_bin == 1) & (y_test == 0)).sum())
        fn = int(((preds_bin == 0) & (y_test == 1)).sum())
        costo_consigna = 5.0 * fp + 200.0 * fn
        perdida_fraude = float(montos_test[(preds_bin == 0) & (y_test == 1)].sum())
        salvado_fraude = float(montos_test[(preds_bin == 1) & (y_test == 1)].sum())
        costo_financiero = 5.0 * fp + perdida_fraude
        recovery_rate = (salvado_fraude / total_fraud_test_capital) * 100.0
        recall_eventos = (tp / (tp + fn)) * 100.0
        precision = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0

        return {
            "politica": name,
            "tp": tp, "fp": fp, "fn": fn,
            "precision_pct": precision,
            "recall_pct": recall_eventos,
            "costo_consigna": costo_consigna,
            "costo_financiero": costo_financiero,
            "capital_salvado": salvado_fraude,
            "capital_perdido": perdida_fraude,
            "capital_recovery_rate": recovery_rate,
        }

    res_p0 = eval_policy_test(
        lambda p, m, d: (p >= tau_p0).astype(int),
        f"0. Umbral Estático Consigna (tau = {tau_p0:.4f})"
    )

    res_bimodal = eval_policy_test(
        lambda p, m, d: np.where(m <= SPLIT_TICKET, p >= tau_low_opt, p >= tau_high_opt).astype(int),
        f"1. Segmentado Ticket ($25): Low={tau_low_opt:.4f}, High={tau_high_opt:.4f}"
    )

    res_scurve = eval_policy_test(
        lambda p, m, d: (p >= compute_tau_scurve(m)).astype(int),
        f"2. Curva S Acotada (t_max={t_max_opt:.2f}, t_min={t_min_opt:.2f})"
    )

    res_2d = eval_policy_test(
        lambda p, m, d: (p >= compute_tau_2d(m, d)).astype(int),
        f"3. Umbral 2D Semántico (S-Curve - {alpha_opt:.2f}*deficit)"
    )

    # Evaluación especializada de la Decisión Tripartita (Three-Way Decision)
    t_blk_te = compute_tau_2d(montos_test, deficit_test)
    t_chl_te = beta_challenge_opt * t_blk_te
    block_mask = p_champ_test >= t_blk_te
    chal_mask = (p_champ_test >= t_chl_te) & (~block_mask)

    # Fraudes atrapados: los bloqueados + el 90% de los desafiados por 2FA
    tp_block = int((block_mask & (y_test == 1)).sum())
    tp_chal = int((chal_mask & (y_test == 1)).sum())
    tp_3way_effective = int(tp_block + round(0.90 * tp_chal))
    fp_block_te = int((block_mask & (y_test == 0)).sum())
    fp_chal_te = int((chal_mask & (y_test == 0)).sum())
    fn_3way_effective = int((y_test == 1).sum() - tp_3way_effective)

    # Dinero salvado: bloqueo total + 90% del dinero de fraudes en 2FA
    m_salv_block = float(montos_test[block_mask & (y_test == 1)].sum())
    m_salv_chal = float(montos_test[chal_mask & (y_test == 1)].sum())
    salvado_3way = m_salv_block + 0.90 * m_salv_chal
    perdido_3way = total_fraud_test_capital - salvado_3way
    costo_consigna_3way = 5.0 * fp_block_te + 0.05 * fp_chal_te + 200.0 * fn_3way_effective
    costo_real_3way = 25.0 * fp_block_te + 0.05 * fp_chal_te + perdido_3way
    recovery_rate_3way = (salvado_3way / total_fraud_test_capital) * 100.0
    recall_3way = (tp_3way_effective / (y_test == 1).sum()) * 100.0

    res_3way = {
        "politica": f"4. Decisión Tripartita 2FA (beta_chal={beta_challenge_opt:.2f})",
        "tp": tp_3way_effective, "fp": fp_block_te, "fn": fn_3way_effective,
        "precision_pct": (tp_3way_effective / (tp_3way_effective + fp_block_te)) * 100.0,
        "recall_pct": recall_3way,
        "costo_consigna": costo_consigna_3way,
        "costo_financiero": costo_real_3way,
        "capital_salvado": salvado_3way,
        "capital_perdido": perdido_3way,
        "capital_recovery_rate": recovery_rate_3way,
    }

    df_policies = pd.DataFrame([res_p0, res_bimodal, res_scurve, res_2d, res_3way])

    print("-" * 120)
    print(f"{'Política Operativa':<48s} | {'FP Duro':<7s} | {'FN':<5s} | {'Recall':<7s} | {'Costo Formal':<14s} | {'Costo Real':<12s} | {'Recupero %'}")
    print("-" * 120)
    for _, r in df_policies.iterrows():
        print(f"{r['politica']:<48s} | {r['fp']:<7d} | {r['fn']:<5d} | {r['recall_pct']:<6.2f}% | ${r['costo_consigna']:<13,.0f} | ${r['costo_financiero']:<11,.2f} | {r['capital_recovery_rate']:.2f}%")
    print("-" * 120)

    policy_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tau_estatico_consigna": float(tau_p0),
        "bimodal_params": {
            "corte_ticket": float(SPLIT_TICKET),
            "tau_low": float(tau_low_opt),
            "tau_high": float(tau_high_opt)
        },
        "scurve_params": {
            "t_max": float(t_max_opt),
            "t_min": float(t_min_opt),
            "sigma": float(sigma_opt),
            "M0": float(M0)
        },
        "semantico_2d_alpha": float(alpha_opt),
        "tripartita_beta_challenge": float(beta_challenge_opt),
        "resultados_test": df_policies.to_dict(orient="records")
    }
    out_json = ROOT / "docs/RESULTADOS_EXPERIMENTO_POLITICA_UMBRALES.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(policy_report, f, indent=2)
    print(f"\n[OK] Reporte completo guardado en {out_json}")
    print(f"[OK] Tiempo total de ejecución: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
