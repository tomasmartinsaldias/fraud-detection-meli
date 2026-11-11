"""
Prueba empírica de solapamiento y competencia combinada de las 3 features positivas:
1. ratio_monto_mediana_cluster
2. deficit_p25_cluster
3. log_ratio_asimetrico
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.cluster import MiniBatchKMeans

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.supervised import validation_blocks, pr_auc
from src.models.train import lgb_model

def main():
    base_df = pd.read_parquet(ROOT / "data/processed/dataset_base.parquet")
    feats_v2 = pd.read_parquet(ROOT / "data/features/features_seleccionadas_end_to_end.parquet")

    train_mask = base_df["split"] == "train"
    val_mask = base_df["split"] == "val"
    test_mask = base_df["split"] == "test"

    train = base_df[train_mask]
    tune, cal, policy = validation_blocks(base_df[val_mask])
    test = base_df[test_mask]

    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=5, sublinear_tf=True, strip_accents="unicode")
    X_tr_tfidf = vec.fit_transform(train["i"].fillna(""))
    svd = TruncatedSVD(n_components=30, random_state=42)
    X_tr_svd = normalize(svd.fit_transform(X_tr_tfidf))

    kmeans = MiniBatchKMeans(n_clusters=50, random_state=42, batch_size=2048)
    train_clusters = kmeans.fit_predict(X_tr_svd)
    tune_clusters = kmeans.predict(normalize(svd.transform(vec.transform(tune["i"].fillna("")))))
    test_clusters = kmeans.predict(normalize(svd.transform(vec.transform(test["i"].fillna("")))))

    t_df = pd.DataFrame({"c": train_clusters, "m": train["monto"].values})
    st = t_df.groupby("c")["m"].agg(["median", "count", lambda x: np.percentile(x, 25)])
    st.columns = ["median", "count", "p25"]
    gm = float(train["monto"].median())
    gp25 = float(np.percentile(train["monto"], 25))

    st["bayes_med"] = (st["count"] * st["median"] + 10.0 * gm) / (st["count"] + 10.0)
    st["bayes_p25"] = (st["count"] * st["p25"] + 10.0 * gp25) / (st["count"] + 10.0)

    med_map = st["bayes_med"].to_dict()
    p25_map = st["bayes_p25"].to_dict()

    tr_med = np.array([med_map.get(c, gm) for c in train_clusters], dtype=float)
    tu_med = np.array([med_map.get(c, gm) for c in tune_clusters], dtype=float)
    te_med = np.array([med_map.get(c, gm) for c in test_clusters], dtype=float)

    tr_p25 = np.array([p25_map.get(c, gp25) for c in train_clusters], dtype=float)
    tu_p25 = np.array([p25_map.get(c, gp25) for c in tune_clusters], dtype=float)
    te_p25 = np.array([p25_map.get(c, gp25) for c in test_clusters], dtype=float)

    # Las 3 variables
    f1_tr = train["monto"].values / (tr_med + 1e-4) # ratio_monto_mediana_cluster
    f1_tu = tune["monto"].values / (tu_med + 1e-4)
    f1_te = test["monto"].values / (te_med + 1e-4)

    f2_tr = np.maximum(0.0, (tr_p25 - train["monto"].values) / (tr_p25 + 1e-4)) # deficit_p25_cluster
    f2_tu = np.maximum(0.0, (tu_p25 - tune["monto"].values) / (tu_p25 + 1e-4))
    f2_te = np.maximum(0.0, (te_p25 - test["monto"].values) / (te_p25 + 1e-4))

    f3_tr = np.log1p(tr_med) / (np.log1p(train["monto"].values) + 1.0) # log_ratio_asimetrico
    f3_tu = np.log1p(tu_med) / (np.log1p(tune["monto"].values) + 1.0)
    f3_te = np.log1p(te_med) / (np.log1p(test["monto"].values) + 1.0)

    # Correlación de Spearman en Train
    corr_df = pd.DataFrame({
        "ratio_med": f1_tr,
        "def_p25": f2_tr,
        "log_asym": f3_tr,
    })
    print("\n--- MATRIZ DE CORRELACIÓN DE SPEARMAN ENTRE LAS 3 CANDIDATAS ---")
    print(corr_df.corr(method="spearman").round(3))

    overlap = [c for c in feats_v2.columns if c in base_df.columns and c != "transaction_id"]
    df = base_df.drop(columns=overlap).merge(feats_v2, on="transaction_id")
    for c in ["o", "g", "a"]:
        if c in df.columns:
            df[c] = df[c].astype("category")

    train_df = df[train_mask].copy()
    tune_df = df.loc[tune.index].copy()
    test_df = df[test_mask].copy()

    train_df["deficit_piso_cluster"] = np.maximum(0.0, (tr_med - train["monto"].values) / (tr_med + 1e-4))
    tune_df["deficit_piso_cluster"] = np.maximum(0.0, (tu_med - tune["monto"].values) / (tu_med + 1e-4))
    test_df["deficit_piso_cluster"] = np.maximum(0.0, (te_med - test["monto"].values) / (te_med + 1e-4))

    train_df["distancia_log_cluster"] = np.log1p(train["monto"].values) - np.log1p(tr_med)
    tune_df["distancia_log_cluster"] = np.log1p(tune["monto"].values) - np.log1p(tu_med)
    test_df["distancia_log_cluster"] = np.log1p(test["monto"].values) - np.log1p(te_med)

    base_13 = [c for c in feats_v2.columns if c != "transaction_id"] + ["deficit_piso_cluster", "distancia_log_cluster"]

    train_df["ratio_med"] = f1_tr
    tune_df["ratio_med"] = f1_tu
    test_df["ratio_med"] = f1_te

    train_df["def_p25"] = f2_tr
    tune_df["def_p25"] = f2_tu
    test_df["def_p25"] = f2_te

    train_df["log_asym"] = f3_tr
    tune_df["log_asym"] = f3_tu
    test_df["log_asym"] = f3_te

    y_train = train_df["fraude"].values
    y_tune = tune_df["fraude"].values
    y_test = test_df["fraude"].values

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

    configs = {
        "Base 13": base_13,
        "Base 13 + ratio_med (14 vars)": base_13 + ["ratio_med"],
        "Base 13 + def_p25 (14 vars)": base_13 + ["def_p25"],
        "Base 13 + log_asym (14 vars)": base_13 + ["log_asym"],
        "Base 14 (ratio_med) + def_p25 (15 vars)": base_13 + ["ratio_med", "def_p25"],
        "Base 14 (ratio_med) + log_asym (15 vars)": base_13 + ["ratio_med", "log_asym"],
        "Base 14 + def_p25 + log_asym (16 vars)": base_13 + ["ratio_med", "def_p25", "log_asym"],
    }

    print("\n--- RESULTADOS DE COMPETENCIA COMBINADA (FORWARD SELECTION PASO 2) ---")
    print(f"{'Configuración':<45} | {'val_search':<10} | {'test':<10} | {'Delta Val (bps)':<15}")
    print("-" * 85)
    for name, cols in configs.items():
        m = lgb_model(best_params_lgb, 42, 4).fit(train_df[cols], y_train)
        p_tu = m.predict_proba(tune_df[cols])[:, 1]
        p_te = m.predict_proba(test_df[cols])[:, 1]
        pr_tu = pr_auc(y_tune, p_tu)
        pr_te = pr_auc(y_test, p_te)
        delta_val = (pr_tu - 0.48592) * 10000
        print(f"{name:<45} | {pr_tu:<10.5f} | {pr_te:<10.5f} | {delta_val:+8.1f} bps")

if __name__ == "__main__":
    main()
