"""
Experimento EBM (Explainable Boosting Machines) vs Modelos del Stacking
========================================================================
- Evalúa EBM en el set de Validación cronológico oficial (val_full y tune).
- Compara con LightGBM (14 vars), XGBoost (12 vars), CatBoost (14 vars) y Random Forest (6 vars).
- Evalúa EBM con las 14 variables de Challenger v3 y con las 12 variables de XGBoost.
- Mide estadísticas descriptivas de predicción en Validación (separación fraude vs no fraude).
- Computa la matriz de correlación de Spearman y Pearson entre todos los modelos.
- Configuración ultra-optimizada para ejecutar en menos de 5 minutos.
"""

from __future__ import annotations
import json
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
from interpret.glassbox import ExplainableBoostingClassifier
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, pearsonr, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import OrdinalEncoder, normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
import xgboost as xgb

from src.models.supervised import pr_auc, validation_blocks
from src.models.train import lgb_model

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


def compute_prediction_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calcula métricas de discriminación y estadísticas percentilares en fraude vs no fraude."""
    p_legit = y_pred[y_true == 0]
    p_fraud = y_pred[y_true == 1]
    ks_res = ks_2samp(p_fraud, p_legit)

    return {
        "pr_auc": float(pr_auc(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, y_pred)),
        "brier_score": float(brier_score_loss(y_true, y_pred)),
        "ks_stat": float(ks_res.statistic),
        "mean_all": float(np.mean(y_pred)),
        "mean_legit": float(np.mean(p_legit)),
        "mean_fraud": float(np.mean(p_fraud)),
        "ratio_mean_separation": float(np.mean(p_fraud) / (np.mean(p_legit) + 1e-9)),
        "p50_legit": float(np.percentile(p_legit, 50)),
        "p95_legit": float(np.percentile(p_legit, 95)),
        "p50_fraud": float(np.percentile(p_fraud, 50)),
        "p95_fraud": float(np.percentile(p_fraud, 95)),
    }


def main():
    seed = 42
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    t0_global = time.time()

    print("=" * 90)
    print("EXPERIMENTO: EXPLAINABLE BOOSTING MACHINES (EBM) VS MODELOS DEL STACKING")
    print("EVALUACIÓN EN VALIDACIÓN CRONOLÓGICA (VAL SEARCH Y VAL FULL) Y CORRELACIÓN")
    print("=" * 90)

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

    # 2. Generar Clustering Semántico v3 estrictamente en Train
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
    y_val_full = val_full.fraude.to_numpy()
    y_test = test.fraude.to_numpy()

    # Preprocesamiento específico por tipo de modelo
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
    val_xgb = df_xgb.loc[val_full.index]
    test_xgb = df_xgb.loc[test.index]

    train_rf = df_rf.loc[df.split == "train"]
    tune_rf = df_rf.loc[tune.index]
    val_rf = df_rf.loc[val_full.index]
    test_rf = df_rf.loc[test.index]

    df_cat = df.copy()
    for cat in BASE_CATEGORICAL_COLS:
        df_cat[cat] = df_cat[cat].fillna("MISSING").astype(str)
    train_cat = df_cat.loc[df.split == "train"]
    tune_cat = df_cat.loc[tune.index]
    val_cat = df_cat.loc[val_full.index]
    test_cat = df_cat.loc[test.index]

    # Preprocesamiento específico para EBM
    # Convertimos categorías a string para que EBM las reconozca como nominales nativas con manejo de nulos
    df_ebm = df.copy()
    for cat in BASE_CATEGORICAL_COLS:
        # Dejamos NaNs como NaNs o strings 'None'
        df_ebm[cat] = df_ebm[cat].astype("category")

    train_ebm = df_ebm.loc[df.split == "train"]
    tune_ebm = df_ebm.loc[tune.index]
    val_ebm = df_ebm.loc[val_full.index]
    test_ebm = df_ebm.loc[test.index]

    # =========================================================================
    # DEFINICIÓN DE SUBCONJUNTOS DE FEATURES
    # =========================================================================
    lgb_14 = [
        "score", "o", "g", "a", "es_madrugada", "ratio_d_m", "ratio_monto_l",
        "prod_b_log_c", "score_texto_nb", "ratio_monto_e", "is_h_cero",
        "deficit_piso_cluster", "distancia_log_cluster", "ratio_monto_mediana_cluster"
    ]
    xgb_cols = [
        'score', 'o', 'te_j_bayes', 'es_madrugada', 'ratio_monto_l', 'diff_m_d',
        'score_texto_nb', 'is_h_cero', 'flag_alto_riesgo_nop', 'ratio_monto_b', 'g', 'b'
    ]
    rf_cols = ['score', 'o', 'te_j_bayes', 'es_madrugada', 'diff_m_d', 'distancia_log_cluster']
    cat_cols = list(lgb_14)
    cat_indices = [cat_cols.index(c) for c in BASE_CATEGORICAL_COLS if c in cat_cols]

    predictions_val = {}
    predictions_tune = {}
    predictions_test = {}
    timing_info = {}

    # =========================================================================
    # 2. MODELOS BASE DEL STACKING
    # =========================================================================
    print("\n2. Entrenando Modelos Base del Stacking (Nivel 1)...")

    # [1] LightGBM (14 vars)
    t_start = time.time()
    best_params_lgb = {
        "learning_rate": 0.03487, "num_leaves": 28, "min_child_samples": 100,
        "colsample_bytree": 0.814, "subsample": 0.757, "reg_alpha": 0.013, "reg_lambda": 0.527,
        "n_estimators": 253,
    }
    m_lgb = lgb_model(best_params_lgb, seed, 4).fit(train_xgb[lgb_14], y_train)
    p_lgb_tu = m_lgb.predict_proba(tune_xgb[lgb_14])[:, 1]
    p_lgb_va = m_lgb.predict_proba(val_xgb[lgb_14])[:, 1]
    p_lgb_te = m_lgb.predict_proba(test_xgb[lgb_14])[:, 1]
    predictions_tune["LightGBM"] = p_lgb_tu
    predictions_val["LightGBM"] = p_lgb_va
    predictions_test["LightGBM"] = p_lgb_te
    timing_info["LightGBM"] = time.time() - t_start
    print(f"  * [1/6] LightGBM (14 vars) -> Val PR-AUC: {pr_auc(y_val_full, p_lgb_va):.5f} | Val Search: {pr_auc(y_tune, p_lgb_tu):.5f} ({timing_info['LightGBM']:.1f}s)")

    # [2] XGBoost (12 vars)
    t_start = time.time()
    best_params_xgb = {
        "learning_rate": 0.052, "max_depth": 5, "min_child_weight": 35,
        "colsample_bytree": 0.80, "subsample": 0.80, "reg_alpha": 0.1, "reg_lambda": 5.0,
        "n_estimators": 180, "tree_method": "hist", "enable_categorical": True,
        "random_state": seed, "n_jobs": 4, "verbosity": 0,
    }
    m_xgb = xgb.XGBClassifier(**best_params_xgb).fit(train_xgb[xgb_cols], y_train)
    p_xgb_tu = m_xgb.predict_proba(tune_xgb[xgb_cols])[:, 1]
    p_xgb_va = m_xgb.predict_proba(val_xgb[xgb_cols])[:, 1]
    p_xgb_te = m_xgb.predict_proba(test_xgb[xgb_cols])[:, 1]
    predictions_tune["XGBoost"] = p_xgb_tu
    predictions_val["XGBoost"] = p_xgb_va
    predictions_test["XGBoost"] = p_xgb_te
    timing_info["XGBoost"] = time.time() - t_start
    print(f"  * [2/6] XGBoost  (12 vars) -> Val PR-AUC: {pr_auc(y_val_full, p_xgb_va):.5f} | Val Search: {pr_auc(y_tune, p_xgb_tu):.5f} ({timing_info['XGBoost']:.1f}s)")

    # [3] CatBoost (14 vars)
    t_start = time.time()
    m_cat = CatBoostClassifier(iterations=280, learning_rate=0.045, depth=6, l2_leaf_reg=6.0, cat_features=cat_indices, random_seed=seed, thread_count=4, verbose=0)
    m_cat.fit(train_cat[cat_cols], y_train, eval_set=(tune_cat[cat_cols], y_tune), early_stopping_rounds=30)
    p_cat_tu = m_cat.predict_proba(tune_cat[cat_cols])[:, 1]
    p_cat_va = m_cat.predict_proba(val_cat[cat_cols])[:, 1]
    p_cat_te = m_cat.predict_proba(test_cat[cat_cols])[:, 1]
    predictions_tune["CatBoost"] = p_cat_tu
    predictions_val["CatBoost"] = p_cat_va
    predictions_test["CatBoost"] = p_cat_te
    timing_info["CatBoost"] = time.time() - t_start
    print(f"  * [3/6] CatBoost (14 vars) -> Val PR-AUC: {pr_auc(y_val_full, p_cat_va):.5f} | Val Search: {pr_auc(y_tune, p_cat_tu):.5f} ({timing_info['CatBoost']:.1f}s)")

    # [4] Random Forest (6 vars)
    t_start = time.time()
    m_rf = RandomForestClassifier(n_estimators=80, max_depth=10, min_samples_leaf=30, max_features=0.7, random_state=seed, n_jobs=4)
    m_rf.fit(train_rf[rf_cols], y_train)
    p_rf_tu = m_rf.predict_proba(tune_rf[rf_cols])[:, 1]
    p_rf_va = m_rf.predict_proba(val_rf[rf_cols])[:, 1]
    p_rf_te = m_rf.predict_proba(test_rf[rf_cols])[:, 1]
    predictions_tune["RandomForest"] = p_rf_tu
    predictions_val["RandomForest"] = p_rf_va
    predictions_test["RandomForest"] = p_rf_te
    timing_info["RandomForest"] = time.time() - t_start
    print(f"  * [4/6] RandomForest (6 vars) -> Val PR-AUC: {pr_auc(y_val_full, p_rf_va):.5f} | Val Search: {pr_auc(y_tune, p_rf_tu):.5f} ({timing_info['RandomForest']:.1f}s)")

    # =========================================================================
    # 3. ENTRENAMIENTO DE EXPLAINABLE BOOSTING MACHINES (EBM)
    # =========================================================================
    print("\n3. Entrenando Explainable Boosting Machines (EBM) optimizado (< 5 min)...")

    # Helper para tipos de EBM
    def get_ebm_feature_types(columns: List[str]) -> List[str]:
        return ["nominal" if c in BASE_CATEGORICAL_COLS else "continuous" for c in columns]

    # [5] EBM - Configuración Challenger v3 (14 variables de LGBM/CatBoost)
    t_start = time.time()
    ebm_types_14 = get_ebm_feature_types(lgb_14)
    ebm_14 = ExplainableBoostingClassifier(
        feature_types=ebm_types_14,
        max_bins=128,
        max_interaction_bins=32,
        interactions=8,
        outer_bags=4,
        inner_bags=0,
        learning_rate=0.04,
        early_stopping_rounds=30,
        n_jobs=-1,
        random_state=seed
    )
    print("  -> Ajustando EBM_14 (14 vars, 8 interacciones FAST, outer_bags=4)...")
    ebm_14.fit(train_ebm[lgb_14], y_train)
    p_ebm14_tu = ebm_14.predict_proba(tune_ebm[lgb_14])[:, 1]
    p_ebm14_va = ebm_14.predict_proba(val_ebm[lgb_14])[:, 1]
    p_ebm14_te = ebm_14.predict_proba(test_ebm[lgb_14])[:, 1]
    predictions_tune["EBM_14vars"] = p_ebm14_tu
    predictions_val["EBM_14vars"] = p_ebm14_va
    predictions_test["EBM_14vars"] = p_ebm14_te
    timing_info["EBM_14vars"] = time.time() - t_start
    print(f"  * [5/6] EBM_14vars (14 vars) -> Val PR-AUC: {pr_auc(y_val_full, p_ebm14_va):.5f} | Val Search: {pr_auc(y_tune, p_ebm14_tu):.5f} ({timing_info['EBM_14vars']:.1f}s)")

    # [6] EBM - Configuración XGBoost (12 variables)
    t_start = time.time()
    ebm_types_12 = get_ebm_feature_types(xgb_cols)
    ebm_12 = ExplainableBoostingClassifier(
        feature_types=ebm_types_12,
        max_bins=128,
        max_interaction_bins=32,
        interactions=8,
        outer_bags=4,
        inner_bags=0,
        learning_rate=0.04,
        early_stopping_rounds=30,
        n_jobs=-1,
        random_state=seed
    )
    print("  -> Ajustando EBM_12 (12 vars XGB, 8 interacciones FAST, outer_bags=4)...")
    ebm_12.fit(train_ebm[xgb_cols], y_train)
    p_ebm12_tu = ebm_12.predict_proba(tune_ebm[xgb_cols])[:, 1]
    p_ebm12_va = ebm_12.predict_proba(val_ebm[xgb_cols])[:, 1]
    p_ebm12_te = ebm_12.predict_proba(test_ebm[xgb_cols])[:, 1]
    predictions_tune["EBM_12vars"] = p_ebm12_tu
    predictions_val["EBM_12vars"] = p_ebm12_va
    predictions_test["EBM_12vars"] = p_ebm12_te
    timing_info["EBM_12vars"] = time.time() - t_start
    print(f"  * [6/6] EBM_12vars (12 vars) -> Val PR-AUC: {pr_auc(y_val_full, p_ebm12_va):.5f} | Val Search: {pr_auc(y_tune, p_ebm12_tu):.5f} ({timing_info['EBM_12vars']:.1f}s)")

    # =========================================================================
    # 4. ESTADÍSTICAS DETALLADAS EN VALIDACIÓN (VAL_FULL Y VAL_SEARCH)
    # =========================================================================
    print("\n" + "=" * 90)
    print("4. TABLA DE STATS Y PERFORMANCE EN VALIDATION (VAL_FULL, 22.500 REGISTROS)")
    print("=" * 90)

    val_stats = {}
    for name, p_val in predictions_val.items():
        val_stats[name] = compute_prediction_stats(y_val_full, p_val)

    stats_df = pd.DataFrame(val_stats).T
    cols_display = ["pr_auc", "roc_auc", "brier_score", "log_loss", "ks_stat", "mean_legit", "mean_fraud", "ratio_mean_separation"]
    print(stats_df[cols_display].round(5).to_string())

    print("\n" + "-" * 90)
    print("ESTADÍSTICAS PERCENTILARES EN VALIDACIÓN (PREDICCIONES DE FRAUDE VS LEGÍTIMO)")
    print("-" * 90)
    for name, p_val in [("LightGBM", predictions_val["LightGBM"]), ("EBM_14vars", predictions_val["EBM_14vars"]), ("EBM_12vars", predictions_val["EBM_12vars"])]:
        p_legit = p_val[y_val_full == 0]
        p_fraud = p_val[y_val_full == 1]
        print(f"[{name}]")
        print(f"  * No Fraude (y=0) | p25: {np.percentile(p_legit, 25):.4f} | Mediana: {np.percentile(p_legit, 50):.4f} | p95: {np.percentile(p_legit, 95):.4f} | Max: {np.max(p_legit):.4f}")
        print(f"  * Fraude (y=1)    | p25: {np.percentile(p_fraud, 25):.4f} | Mediana: {np.percentile(p_fraud, 50):.4f} | p95: {np.percentile(p_fraud, 95):.4f} | Max: {np.max(p_fraud):.4f}")

    # =========================================================================
    # 5. MATRICES DE CORRELACIÓN EN VALIDACIÓN
    # =========================================================================
    print("\n" + "=" * 90)
    print("5. MATRIZ DE CORRELACIÓN DE SPEARMAN (RANKING) EN VALIDACIÓN")
    print("=" * 90)

    val_preds_df = pd.DataFrame(predictions_val)
    spearman_mat = val_preds_df.corr(method="spearman")
    print(spearman_mat.round(4).to_string())

    print("\n" + "=" * 90)
    print("6. MATRIZ DE CORRELACIÓN LINEAL DE PEARSON EN VALIDACIÓN")
    print("=" * 90)
    pearson_mat = val_preds_df.corr(method="pearson")
    print(pearson_mat.round(4).to_string())

    # =========================================================================
    # 6. HALLAZGOS DE EXPLICABILIDAD DE EBM (RANKING DE TÉRMINOS)
    # =========================================================================
    print("\n" + "=" * 90)
    print("7. EXPLICABILIDAD DE EBM_14vars (TOP TÉRMINOS E INTERACCIONES POR IMPORTANCIA GLOBAL)")
    print("=" * 90)

    # term_importances() da la importancia global de cada término (univariado o interacción)
    ebm_term_names = ebm_14.term_names_
    ebm_importances = ebm_14.term_importances()
    terms_df = pd.DataFrame({
        "termino": ebm_term_names,
        "importancia_media": ebm_importances,
        "tipo": ["Interacción (2D)" if " & " in t or " x " in t else "Efecto Principal (1D)" for t in ebm_term_names]
    }).sort_values("importancia_media", ascending=False).reset_index(drop=True)

    print(terms_df.head(15).to_string())

    # =========================================================================
    # 7. TIEMPOS Y CONCLUSIÓN
    # =========================================================================
    t_global = time.time() - t0_global
    print("\n" + "=" * 90)
    print(f"TIEMPO TOTAL DE EJECUCIÓN DEL EXPERIMENTO: {t_global:.1f} segundos ({t_global/60:.2f} minutos)")
    print("=" * 90)

    # Persistir artefactos en outputs/ebm/
    out_dir = ROOT / "outputs/ebm"
    out_dir.mkdir(parents=True, exist_ok=True)
    spearman_mat.to_csv(out_dir / "spearman_correlation_validation.csv")
    pearson_mat.to_csv(out_dir / "pearson_correlation_validation.csv")
    stats_df.to_csv(out_dir / "stats_validation_models.csv")
    terms_df.to_csv(out_dir / "ebm_feature_importances.csv")

    summary_json = {
        "tiempo_total_segundos": t_global,
        "tiempos_por_modelo": timing_info,
        "stats_val_full": val_stats,
        "spearman_correlation": spearman_mat.to_dict(),
        "pearson_correlation": pearson_mat.to_dict(),
    }
    with open(out_dir / "ebm_experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    print(f"Artefactos y tablas exportados exitosamente en: {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
