"""Rebuild Contract C inputs from the course ZIP, without evaluating test labels.

Produces separate Ian feature artifacts; existing teammates' outputs are untouched.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.make_dataset import add_traceability_features, filter_population, apply_temporal_split, clean_final_columns, save_processed_dataset
from src.features.build_features import compute_profile_interactions, compute_dm_interactions, compute_ef_interactions, compute_h_interactions, compute_additional_candidates
from src.models.supervised import BASE_NUMERIC


def text_scores(df):
    train = df.loc[df.split == "train"].sort_values(["fecha", "transaction_id"])
    text = train.i.fillna("").astype(str)
    result = pd.Series(np.nan, index=df.index)

    def fit(part):
        vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=30,
                                    sublinear_tf=True, strip_accents="unicode")
        x = vectorizer.fit_transform(text.iloc[part])
        model = MultinomialNB(alpha=2.).fit(x, train.fraude.iloc[part])
        return vectorizer, model

    for tr, va in TimeSeriesSplit(n_splits=5).split(train):
        # Exclude training rows sharing the first validation timestamp.
        tr = tr[train.fecha.iloc[tr].to_numpy() < train.fecha.iloc[va[0]]]
        vec, model = fit(tr)
        result.loc[train.index[va]] = model.predict_proba(vec.transform(text.iloc[va]))[:, 1]
    # No past labels exist for the initial block: keep it missing, never fill
    # with future/global target prevalence. LightGBM handles this explicitly.
    vec, model = fit(np.arange(len(train)))
    heldout = df.split != "train"
    result.loc[heldout] = model.predict_proba(vec.transform(df.loc[heldout, "i"].fillna("").astype(str)))[:, 1]
    return result, {"vectorizer": vec, "model": model}


def anomaly_scores(df, seed):
    # A transparent contract-compatible fallback while Franco's artifact is absent.
    # Neither score nor fraude are predictors; only legitimate train fits models.
    cols = [c for c in BASE_NUMERIC if c != "score"]
    legitimate = (df.split == "train") & (df.fraude == 0)
    values = df[cols].replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(values.loc[legitimate])
    scaler = StandardScaler().fit(imputer.transform(values.loc[legitimate]))
    x = scaler.transform(imputer.transform(values))
    reference = x[legitimate]
    pca = PCA(n_components=.95, svd_solver="full").fit(reference)
    forest = IsolationForest(n_estimators=150, max_samples=256, random_state=seed, n_jobs=4).fit(reference)
    rng = np.random.default_rng(seed)
    sample = reference[rng.choice(len(reference), min(10000, len(reference)), replace=False)]
    knn = NearestNeighbors(n_neighbors=20, n_jobs=4).fit(sample)

    def raw_scores(block):
        return {"score_iforest": -forest.score_samples(block),
                "error_pca": ((block - pca.inverse_transform(pca.transform(block))) ** 2).sum(axis=1),
                "distancia_lof_knn": knn.kneighbors(block)[0].mean(axis=1)}

    raw = raw_scores(x)
    out = pd.DataFrame({"transaction_id": df.transaction_id.astype("string")})
    refs = {}
    for col, scores in raw.items():
        refs[col] = np.sort(scores[legitimate])
        out[col] = np.searchsorted(refs[col], scores) / len(refs[col])
    out["score_anomalia_consolidado"] = out[list(raw)].mean(axis=1)
    return out, {"columns": cols, "imputer": imputer, "scaler": scaler, "pca": pca,
                 "forest": forest, "knn": knn, "reference_percentiles": refs, "seed": seed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zip", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    base_path = Path("data/processed/dataset_base.parquet")
    feat_path = Path("data/features/ian_features_selected.parquet")
    anom_path = Path("data/features/ian_scores_anomalias.parquet")
    if any(p.exists() for p in (base_path, feat_path, anom_path)):
        raise FileExistsError("Ya hay artefactos de entrada; no se sobrescriben automáticamente.")
    with zipfile.ZipFile(args.source_zip) as archive:
        nested_name = next(n for n in archive.namelist() if n.endswith(".csv.zip"))
        with zipfile.ZipFile(io.BytesIO(archive.read(nested_name))) as nested:
            csv_name = next(n for n in nested.namelist() if n.endswith(".csv") and not n.startswith("__MACOSX"))
            df = pd.read_csv(io.BytesIO(nested.read(csv_name)))
    df = add_traceability_features(df)
    df, _ = filter_population(df, drop_second_block=True)
    df = clean_final_columns(apply_temporal_split(df))
    save_processed_dataset(df, base_path)
    # Access labels only from train for feature fitting. Held-out labels masked.
    feature_input = df.copy()
    feature_input.loc[feature_input.split != "train", "fraude"] = np.nan
    selected = df[["transaction_id", "score", "o", "g", "a"]].copy()
    selected["transaction_id"] = selected.transaction_id.astype("string")
    for func, cols in ((compute_profile_interactions, ["prod_b_log_c"]),
                       (compute_dm_interactions, ["ratio_d_m"]),
                       (compute_ef_interactions, ["ratio_monto_e"]),
                       (compute_h_interactions, ["is_h_cero"]),
                       (compute_additional_candidates, ["es_madrugada", "ratio_monto_l"])):
        block = func(feature_input)
        for col in cols:
            selected[col] = block[col]
    selected["score_texto_nb"], text_bundle = text_scores(feature_input)
    print("Variables seleccionadas reconstruidas; entrenando anomalías...", flush=True)
    anomalies, anomaly_bundle = anomaly_scores(feature_input, args.seed)
    feat_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(feat_path, index=False, compression="snappy")
    anomalies.to_parquet(anom_path, index=False, compression="snappy")
    out = Path("models/supervised/preparation")
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(text_bundle, out / "text.joblib")
    joblib.dump(anomaly_bundle, out / "anomalies.joblib")
    (out / "provenance.json").write_text(json.dumps({"source": str(args.source_zip.resolve()), "seed": args.seed,
        "rows": len(df), "test_labels_used_for_fit": False,
        "note": "Selected formulas from Contract A; causal text warmup missing. Fallback IF/PCA/kNN, not Franco's full bundle."}, indent=2), encoding="utf-8")
    print(f"Listo: {len(df)} filas. No se calcularon métricas de test.", flush=True)


if __name__ == "__main__":
    main()
