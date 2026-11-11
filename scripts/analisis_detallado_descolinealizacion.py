"""
Análisis detallado: Comparación del metamodelo Nivel 2 con y sin variables correlacionadas.
Compara:
1. Matriz Completa Original (22 columnas: p y logits + todas las anomalías + clustering)
2. Matriz Filtrada Descolinealizada (14 columnas: solo logits + anomalías no redundantes + clustering)
3. Matriz Ultralimpia (4 columnas: solo logits de los 4 modelos)
"""
import os, sys, random
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.supervised import validation_blocks, pr_auc
from src.models.train import lgb_model

BASE_CATEGORICAL_COLS = ["a", "g", "n", "o", "p"]
BASE_NUMERIC_COLS = ["score", "monto", "b", "c", "d", "e", "f", "h", "l", "m", "q"]


def build_clustering_features(df_train: pd.DataFrame, df_all: pd.DataFrame):
    """Genera deficit_piso_cluster y distancia_log_cluster estrictamente ajustados en Train con SVD d=30."""
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
    return distancia_log, deficit_piso
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
import xgboost as xgb

def main():
    seed = 42
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    base_df = pd.read_parquet(ROOT / "data/processed/dataset_base.parquet")
    feat_df = pd.read_parquet(ROOT / "data/features/features_transaccionales.parquet")
    anom_df = pd.read_parquet(ROOT / "data/features/scores_anomalias.parquet")
    rules_df = pd.read_parquet(ROOT / "data/features/features_reglas_asociacion.parquet")[["transaction_id", "pct_heuristico_fraude"]]

    anom_cols = [c for c in anom_df.columns if c != "transaction_id"]
    if "score_heuristico_fraude_10" in feat_df.columns:
        feat_df = feat_df.drop(columns=["score_heuristico_fraude_10"])

    train_mask = base_df["split"] == "train"
    dist_log, def_piso = build_clustering_features(base_df[train_mask], base_df)
    base_df["distancia_log_cluster"] = dist_log
    base_df["deficit_piso_cluster"] = def_piso

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

    # Preprocesamiento N1
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

    lgb_13 = ["score", "o", "g", "a", "es_madrugada", "ratio_d_m", "ratio_monto_l", "prod_b_log_c", "score_texto_nb", "ratio_monto_e", "is_h_cero", "deficit_piso_cluster", "distancia_log_cluster"]
    best_params_lgb = {
        "learning_rate": 0.03486992722440761, "num_leaves": 28, "min_child_samples": 100,
        "colsample_bytree": 0.8140488199767426, "subsample": 0.7573516799506432,
        "reg_alpha": 0.013018028720806228, "reg_lambda": 0.5270037815983998, "n_estimators": 253,
    }
    m_lgb = lgb_model(best_params_lgb, seed, 4).fit(train_xgb[lgb_13], y_train)
    p_lgb_tu = m_lgb.predict_proba(tune_xgb[lgb_13])[:, 1]
    p_lgb_te = m_lgb.predict_proba(test_xgb[lgb_13])[:, 1]

    xgb_cols = ['score', 'o', 'te_j_bayes', 'es_madrugada', 'ratio_monto_l', 'diff_m_d', 'score_texto_nb', 'is_h_cero', 'flag_alto_riesgo_nop', 'ratio_monto_b', 'g', 'b']
    best_params_xgb = {
        "learning_rate": 0.052, "max_depth": 5, "min_child_weight": 35,
        "colsample_bytree": 0.80, "subsample": 0.80, "reg_alpha": 0.05,
        "reg_lambda": 1.2, "n_estimators": 210, "enable_categorical": True,
        "tree_method": "hist", "random_state": seed, "n_jobs": 4,
    }
    m_xgb = xgb.XGBClassifier(**best_params_xgb).fit(train_xgb[xgb_cols], y_train)
    p_xgb_tu = m_xgb.predict_proba(tune_xgb[xgb_cols])[:, 1]
    p_xgb_te = m_xgb.predict_proba(test_xgb[xgb_cols])[:, 1]

    cat_cols = [c for c in lgb_13]
    cat_indices = [cat_cols.index(c) for c in BASE_CATEGORICAL_COLS if c in cat_cols]
    m_cat = CatBoostClassifier(iterations=280, learning_rate=0.045, depth=6, l2_leaf_reg=6.0, cat_features=cat_indices, random_seed=seed, thread_count=4, verbose=0)
    m_cat.fit(train_cat[cat_cols], y_train, eval_set=(tune_cat[cat_cols], y_tune), early_stopping_rounds=30)
    p_cat_tu = m_cat.predict_proba(tune_cat[cat_cols])[:, 1]
    p_cat_te = m_cat.predict_proba(test_cat[cat_cols])[:, 1]

    rf_cols = ['score', 'o', 'te_j_bayes', 'es_madrugada', 'diff_m_d', 'distancia_log_cluster']
    m_rf = RandomForestClassifier(n_estimators=80, max_depth=10, min_samples_leaf=30, max_features=0.7, random_state=seed, n_jobs=4)
    m_rf.fit(train_rf[rf_cols], y_train)
    p_rf_tu = m_rf.predict_proba(tune_rf[rf_cols])[:, 1]
    p_rf_te = m_rf.predict_proba(test_rf[rf_cols])[:, 1]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    p_lgb_oof = np.zeros(len(train), dtype=float)
    p_xgb_oof = np.zeros(len(train), dtype=float)
    p_cat_oof = np.zeros(len(train), dtype=float)
    p_rf_oof = np.zeros(len(train), dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y_train), 1):
        m_lgb_f = lgb_model(best_params_lgb, seed, 4).fit(train_xgb[lgb_13].iloc[tr_idx], y_train[tr_idx])
        p_lgb_oof[va_idx] = m_lgb_f.predict_proba(train_xgb[lgb_13].iloc[va_idx])[:, 1]
        m_xgb_f = xgb.XGBClassifier(**best_params_xgb).fit(train_xgb[xgb_cols].iloc[tr_idx], y_train[tr_idx])
        p_xgb_oof[va_idx] = m_xgb_f.predict_proba(train_xgb[xgb_cols].iloc[va_idx])[:, 1]
        m_cat_f = CatBoostClassifier(iterations=250, learning_rate=0.045, depth=6, l2_leaf_reg=6.0, cat_features=cat_indices, random_seed=seed, thread_count=4, verbose=0)
        m_cat_f.fit(train_cat[cat_cols].iloc[tr_idx], y_train[tr_idx])
        p_cat_oof[va_idx] = m_cat_f.predict_proba(train_cat[cat_cols].iloc[va_idx])[:, 1]
        m_rf_f = RandomForestClassifier(n_estimators=80, max_depth=10, min_samples_leaf=30, max_features=0.7, random_state=seed, n_jobs=4)
        m_rf_f.fit(train_rf[rf_cols].iloc[tr_idx], y_train[tr_idx])
        p_rf_oof[va_idx] = m_rf_f.predict_proba(train_rf[rf_cols].iloc[va_idx])[:, 1]

    eps = 1e-7

    # 1. Matriz Original (22 columnas: p y logits + todas anomalias + clustering)
    def build_meta_original(p1, p2, p3, p4, part_df):
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
        return pd.DataFrame(d)

    # 2. Matriz Filtrada (14 columnas: SOLO logits + anomalias no redundantes + clustering)
    anom_no_redundantes = [c for c in anom_cols if c not in [
        "pca_magnitud", "score_ocsvm_rbf_nu_003", "score_iforest_diagonal", "score_lof_euclidean"
    ]]
    def build_meta_filtrada(p1, p2, p3, p4, part_df):
        d = {
            "logit_lgb": np.log(np.clip(p1, eps, 1-eps) / (1-np.clip(p1, eps, 1-eps))),
            "logit_xgb": np.log(np.clip(p2, eps, 1-eps) / (1-np.clip(p2, eps, 1-eps))),
            "logit_cat": np.log(np.clip(p3, eps, 1-eps) / (1-np.clip(p3, eps, 1-eps))),
            "logit_rf":  np.log(np.clip(p4, eps, 1-eps) / (1-np.clip(p4, eps, 1-eps))),
        }
        for col in anom_no_redundantes:
            d[col] = part_df[col].to_numpy(dtype=float)
        d["pct_heuristico_fraude"] = part_df["pct_heuristico_fraude"].to_numpy(dtype=float)
        d["distancia_log_cluster"] = part_df["distancia_log_cluster"].to_numpy(dtype=float)
        d["deficit_piso_cluster"] = part_df["deficit_piso_cluster"].to_numpy(dtype=float)
        return pd.DataFrame(d)

    # 3. Matriz Ultralimpia (4 columnas: solo logits de los 4 inductores)
    def build_meta_pure(p1, p2, p3, p4, part_df):
        return pd.DataFrame({
            "logit_lgb": np.log(np.clip(p1, eps, 1-eps) / (1-np.clip(p1, eps, 1-eps))),
            "logit_xgb": np.log(np.clip(p2, eps, 1-eps) / (1-np.clip(p2, eps, 1-eps))),
            "logit_cat": np.log(np.clip(p3, eps, 1-eps) / (1-np.clip(p3, eps, 1-eps))),
            "logit_rf":  np.log(np.clip(p4, eps, 1-eps) / (1-np.clip(p4, eps, 1-eps))),
        })

    matrices = {
        "1. Matriz Original (22 cols c/ colinealidad)": build_meta_original,
        "2. Matriz Filtrada (14 cols sin redundancias)": build_meta_filtrada,
        "3. Matriz Ultralimpia (4 cols solo logits)": build_meta_pure,
    }

    print("\n" + "=" * 90)
    print("COMPARATIVA DE DESCOLINEALIZACIÓN EN NIVEL 2 CON LASSO L1 (C=0.10 y C=0.20)")
    print("=" * 90)
    print(f"{'Matriz de Nivel 2':<48} | {'C':<5} | {'val_search':<12} | {'test':<12} | {'Delta Test vs Ian'}")
    print("-" * 90)

    for mat_name, builder in matrices.items():
        X_tr = builder(p_lgb_oof, p_xgb_oof, p_cat_oof, p_rf_oof, train)
        X_tu = builder(p_lgb_tu, p_xgb_tu, p_cat_tu, p_rf_tu, tune)
        X_te = builder(p_lgb_te, p_xgb_te, p_cat_te, p_rf_te, test)

        scaler = StandardScaler().fit(X_tr)
        cols = list(X_tr.columns)
        X_tr_sc = pd.DataFrame(scaler.transform(X_tr), columns=cols)
        X_tu_sc = pd.DataFrame(scaler.transform(X_tu), columns=cols)
        X_te_sc = pd.DataFrame(scaler.transform(X_te), columns=cols)

        for c in [0.05, 0.10, 0.20]:
            m = LogisticRegression(solver="saga", l1_ratio=1.0, C=c, max_iter=1000, random_state=seed).fit(X_tr_sc, y_train)
            pr_val = pr_auc(y_tune, m.predict_proba(X_tu_sc)[:, 1])
            pr_te = pr_auc(y_test, m.predict_proba(X_te_sc)[:, 1])
            print(f"{mat_name:<48} | {c:<5.2f} | {pr_val:<12.5f} | {pr_te:<12.5f} | {pr_te - 0.44435:+.5f}")

if __name__ == "__main__":
    main()
