"""
Evaluación de Features para Detección de Fraude en Tickets Bajos (< $20).
Objetivo: Diseñar una feature relativa al cluster que incremente la capacidad
de discriminar fraudes de bajo ticket sin perjudicar la calibración ni el PR-AUC global.
Protocolo: Evaluación rigurosa en val_search (OOT).
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
    print("=" * 80)
    print("EXPERIMENTO: INGENIERÍA DE FEATURES PARA FRAUDES DE TICKET BAJO")
    print("=" * 80)

    base_df = pd.read_parquet(ROOT / "data/processed/dataset_base.parquet")
    feats_v2 = pd.read_parquet(ROOT / "data/features/features_seleccionadas_end_to_end.parquet")

    train_mask = base_df["split"] == "train"
    val_mask = base_df["split"] == "val"
    test_mask = base_df["split"] == "test"

    train = base_df[train_mask]
    tune, cal, policy = validation_blocks(base_df[val_mask])
    test = base_df[test_mask]

    print("\n1. Generando SVD d=30 + MiniBatchKMeans k=50 estrictamente en Train...")
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=5, sublinear_tf=True, strip_accents="unicode")
    X_tr_tfidf = vec.fit_transform(train["i"].fillna(""))
    svd = TruncatedSVD(n_components=30, random_state=42)
    X_tr_svd = normalize(svd.fit_transform(X_tr_tfidf))

    kmeans = MiniBatchKMeans(n_clusters=50, random_state=42, batch_size=2048)
    train_clusters = kmeans.fit_predict(X_tr_svd)

    X_tu_tfidf = vec.transform(tune["i"].fillna(""))
    X_tu_svd = normalize(svd.transform(X_tu_tfidf))
    tune_clusters = kmeans.predict(X_tu_svd)

    X_te_tfidf = vec.transform(test["i"].fillna(""))
    X_te_svd = normalize(svd.transform(X_te_tfidf))
    test_clusters = kmeans.predict(X_te_svd)

    # Estadísticas por cluster ajustadas SOLO en Train
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

    # Base de 13 variables oficiales
    tr_dist_log = np.log1p(train["monto"].values) - np.log1p(tr_med)
    tu_dist_log = np.log1p(tune["monto"].values) - np.log1p(tu_med)
    te_dist_log = np.log1p(test["monto"].values) - np.log1p(te_med)

    tr_deficit = np.maximum(0.0, (tr_med - train["monto"].values) / (tr_med + 1e-4))
    tu_deficit = np.maximum(0.0, (tu_med - tune["monto"].values) / (tu_med + 1e-4))
    te_deficit = np.maximum(0.0, (te_med - test["monto"].values) / (te_med + 1e-4))

    # Candidata 1: micro_anomalia_cluster = (monto < 20) & (med_cluster > 40)
    tr_micro = ((train["monto"].values < 20.0) & (tr_med > 40.0)).astype(float)
    tu_micro = ((tune["monto"].values < 20.0) & (tu_med > 40.0)).astype(float)
    te_micro = ((test["monto"].values < 20.0) & (te_med > 40.0)).astype(float)

    # Candidata 2: ratio_monto_mediana_cluster continuo (monto / (med_cluster + eps))
    tr_ratio_med = train["monto"].values / (tr_med + 1e-4)
    tu_ratio_med = tune["monto"].values / (tu_med + 1e-4)
    te_ratio_med = test["monto"].values / (te_med + 1e-4)

    # Candidata 3: deficit_p25_cluster = max(0, (p25_cluster - monto) / (p25_cluster + eps))
    tr_def_p25 = np.maximum(0.0, (tr_p25 - train["monto"].values) / (tr_p25 + 1e-4))
    tu_def_p25 = np.maximum(0.0, (tu_p25 - tune["monto"].values) / (tu_p25 + 1e-4))
    te_def_p25 = np.maximum(0.0, (te_p25 - test["monto"].values) / (te_p25 + 1e-4))

    # Candidata 4: log_ratio_asimetrico = log(1 + med_cluster) / log(1 + monto + 1)
    tr_asym = np.log1p(tr_med) / (np.log1p(train["monto"].values) + 1.0)
    tu_asym = np.log1p(tu_med) / (np.log1p(tune["monto"].values) + 1.0)
    te_asym = np.log1p(te_med) / (np.log1p(test["monto"].values) + 1.0)

    # Unir con features seleccionadas
    overlap = [c for c in feats_v2.columns if c in base_df.columns and c != "transaction_id"]
    df = base_df.drop(columns=overlap).merge(feats_v2, on="transaction_id")
    for c in ["o", "g", "a"]:
        if c in df.columns:
            df[c] = df[c].astype("category")

    train_df = df[train_mask].copy()
    tune_df = df.loc[tune.index].copy()
    test_df = df[test_mask].copy()

    train_df["deficit_piso_cluster"] = tr_deficit
    tune_df["deficit_piso_cluster"] = tu_deficit
    test_df["deficit_piso_cluster"] = te_deficit

    train_df["distancia_log_cluster"] = tr_dist_log
    tune_df["distancia_log_cluster"] = tu_dist_log
    test_df["distancia_log_cluster"] = te_dist_log

    base_cols = [c for c in feats_v2.columns if c != "transaction_id"] + ["deficit_piso_cluster", "distancia_log_cluster"]

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

    print("\n2. Evaluando Base 13 variables en LightGBM...")
    m_base = lgb_model(best_params_lgb, 42, 4).fit(train_df[base_cols], y_train)
    p_tune_base = m_base.predict_proba(tune_df[base_cols])[:, 1]
    p_test_base = m_base.predict_proba(test_df[base_cols])[:, 1]
    pr_tune_base = pr_auc(y_tune, p_tune_base)
    pr_test_base = pr_auc(y_test, p_test_base)
    print(f"  * Base 13 vars -> val_search: {pr_tune_base:.5f} | test: {pr_test_base:.5f}")

    candidates = {
        "micro_anomalia_cluster": (tr_micro, tu_micro, te_micro),
        "ratio_monto_mediana_cluster": (tr_ratio_med, tu_ratio_med, te_ratio_med),
        "deficit_p25_cluster": (tr_def_p25, tu_def_p25, te_def_p25),
        "log_ratio_asimetrico": (tr_asym, tu_asym, te_asym),
    }

    print("\n3. Evaluando Candidatas Adicionales (Forward Selection)...")
    results = []
    for name, (tr_feat, tu_feat, te_feat) in candidates.items():
        train_df[name] = tr_feat
        tune_df[name] = tu_feat
        test_df[name] = te_feat

        cand_cols = base_cols + [name]
        m_cand = lgb_model(best_params_lgb, 42, 4).fit(train_df[cand_cols], y_train)
        p_tu = m_cand.predict_proba(tune_df[cand_cols])[:, 1]
        p_te = m_cand.predict_proba(test_df[cand_cols])[:, 1]

        pr_tu = pr_auc(y_tune, p_tu)
        pr_te = pr_auc(y_test, p_te)
        delta_tu = pr_tu - pr_tune_base
        delta_te = pr_te - pr_test_base

        results.append({
            "feature": name,
            "val_search": pr_tu,
            "test": pr_te,
            "delta_val_bps": delta_tu * 10000,
            "delta_test_bps": delta_te * 10000,
        })
        print(f"  * + {name:<28} -> val: {pr_tu:.5f} ({delta_tu*10000:+.1f} bps) | test: {pr_te:.5f} ({delta_te*10000:+.1f} bps)")

    print("\n" + "=" * 80)
    print("RESUMEN DE FORWARD SELECTION PARA CANDIDATAS DE TICKET BAJO:")
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))


if __name__ == "__main__":
    main()
