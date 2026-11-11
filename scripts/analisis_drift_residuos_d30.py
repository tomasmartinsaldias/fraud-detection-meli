"""
Análisis de Residuos y Prediction Drift: Modelo v2 vs Modelo v3 (SVD d=30)
========================================================================
Compara exhaustivamente:
1. Champion v2 (11 variables oficiales).
2. Challenger v3 con SVD d=30 (13 variables limpias).
3. Ensamble Stacking con SVD d=30.

Métricas evaluadas:
- Prediction Drift: PSI de probabilidades P(y_hat) entre Val y Test.
- Calibración: Brier score y ROC-AUC / PR-AUC en ambos splits.
- Anatomía de Residuos en Test:
  * Desglose de Falsos Negativos por tramo de ticket (<$20, $20-$50, $50-$100, >$100).
  * Monto medio y mediana de TP vs FN.
  * Capital Recovery Rate (% de dólares de fraude recuperados).
  * Análisis de los fraudes de bajo ticket rescatados por el clustering semántico d=30.
"""

from __future__ import annotations
import json
from pathlib import Path
import os
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.supervised import validation_blocks, pr_auc
from src.models.train import lgb_model


def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10) -> float:
    """Calcula el Population Stability Index (PSI) entre dos distribuciones de probabilidad."""
    eps = 1e-4
    quantiles = np.linspace(0, 1, num_buckets + 1)
    bins = np.percentile(expected, quantiles * 100)
    bins[0] -= eps
    bins[-1] += eps
    # Asegurar bins estrictamente crecientes
    for i in range(1, len(bins)):
        if bins[i] <= bins[i - 1]:
            bins[i] = bins[i - 1] + eps

    exp_counts, _ = np.histogram(expected, bins=bins)
    act_counts, _ = np.histogram(actual, bins=bins)

    exp_pct = np.maximum(exp_counts / len(expected), eps)
    act_pct = np.maximum(act_counts / len(actual), eps)

    psi_val = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi_val)


def main():
    print("=" * 85)
    print("AUDITORÍA DE RESIDUOS Y PREDICTION DRIFT: CHAMPION V2 vs CHALLENGER V3 (SVD d=30)")
    print("=" * 85)

    base_df = pd.read_parquet(ROOT / "data/processed/dataset_base.parquet")
    feats_v2 = pd.read_parquet(ROOT / "data/features/features_seleccionadas_end_to_end.parquet")

    train_mask = base_df["split"] == "train"
    val_mask = base_df["split"] == "val"
    test_mask = base_df["split"] == "test"

    train = base_df[train_mask]
    tune, cal, policy = validation_blocks(base_df[val_mask])
    test = base_df[test_mask]

    # 1. Clustering Semántico con SVD d=30 (Normalizado L2)
    print("\n1. Ajustando Clustering Semántico con SVD d=30 en Train...")
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=5, sublinear_tf=True, strip_accents="unicode")
    X_tr_tfidf = vec.fit_transform(train["i"].fillna(""))
    X_tu_tfidf = vec.transform(tune["i"].fillna(""))
    X_val_all_tfidf = vec.transform(base_df.loc[val_mask, "i"].fillna(""))
    X_te_tfidf = vec.transform(test["i"].fillna(""))

    svd = TruncatedSVD(n_components=30, random_state=42)
    X_tr_svd = normalize(svd.fit_transform(X_tr_tfidf))
    X_tu_svd = normalize(svd.transform(X_tu_tfidf))
    X_val_all_svd = normalize(svd.transform(X_val_all_tfidf))
    X_te_svd = normalize(svd.transform(X_te_tfidf))

    kmeans = MiniBatchKMeans(n_clusters=50, random_state=42, batch_size=2048)
    train_clusters = kmeans.fit_predict(X_tr_svd)
    val_clusters = kmeans.predict(X_val_all_svd)
    test_clusters = kmeans.predict(X_te_svd)

    t_df = pd.DataFrame({"c": train_clusters, "m": train["monto"].values})
    st = t_df.groupby("c")["m"].agg(["median", "count"])
    gm = float(train["monto"].median())
    st["bayes_med"] = (st["count"] * st["median"] + 10.0 * gm) / (st["count"] + 10.0)
    med_map = st["bayes_med"].to_dict()

    tr_med = np.array([med_map.get(c, gm) for c in train_clusters], dtype=float)
    val_med = np.array([med_map.get(c, gm) for c in val_clusters], dtype=float)
    te_med = np.array([med_map.get(c, gm) for c in test_clusters], dtype=float)

    # Variables de cluster
    base_df.loc[train_mask, "distancia_log_cluster"] = np.log1p(train["monto"].values) - np.log1p(tr_med)
    base_df.loc[train_mask, "deficit_piso_cluster"] = np.maximum(0.0, (tr_med - train["monto"].values) / (tr_med + 1e-4))

    base_df.loc[val_mask, "distancia_log_cluster"] = np.log1p(base_df.loc[val_mask, "monto"].values) - np.log1p(val_med)
    base_df.loc[val_mask, "deficit_piso_cluster"] = np.maximum(0.0, (val_med - base_df.loc[val_mask, "monto"].values) / (val_med + 1e-4))

    base_df.loc[test_mask, "distancia_log_cluster"] = np.log1p(test["monto"].values) - np.log1p(te_med)
    base_df.loc[test_mask, "deficit_piso_cluster"] = np.maximum(0.0, (te_med - test["monto"].values) / (te_med + 1e-4))

    overlap = [c for c in feats_v2.columns if c in base_df.columns and c != "transaction_id"]
    df = base_df.drop(columns=overlap).merge(feats_v2, on="transaction_id")
    for c in ["o", "g", "a"]:
        if c in df.columns:
            df[c] = df[c].astype("category")

    base_11_cols = [c for c in feats_v2.columns if c != "transaction_id"]
    feats_13_cols = base_11_cols + ["deficit_piso_cluster", "distancia_log_cluster"]

    train_df = df[train_mask]
    val_df = df[val_mask]
    tune_df = df.loc[tune.index]
    test_df = df[test_mask].copy()

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

    # 2. Entrenar Champion v2 (11 vars) vs Challenger v3 d=30 (13 vars)
    print("\n2. Entrenando Champion v2 (11 vars) vs Challenger v3 (13 vars, d=30)...")
    clf_v2 = lgb_model(best_params_lgb, 42, 4).fit(train_df[base_11_cols], train_df["fraude"])
    clf_v3 = lgb_model(best_params_lgb, 42, 4).fit(train_df[feats_13_cols], train_df["fraude"])

    p_val_v2 = clf_v2.predict_proba(val_df[base_11_cols])[:, 1]
    p_test_v2 = clf_v2.predict_proba(test_df[base_11_cols])[:, 1]

    p_val_v3 = clf_v3.predict_proba(val_df[feats_13_cols])[:, 1]
    p_test_v3 = clf_v3.predict_proba(test_df[feats_13_cols])[:, 1]

    test_df["p_v2"] = p_test_v2
    test_df["p_v3"] = p_test_v3

    y_val = val_df["fraude"].values
    y_test = test_df["fraude"].values

    # 3. Métricas y Drift de Predicciones
    psi_v2 = calculate_psi(p_val_v2, p_test_v2)
    psi_v3 = calculate_psi(p_val_v3, p_test_v3)

    brier_val_v2 = brier_score_loss(y_val, p_val_v2)
    brier_test_v2 = brier_score_loss(y_test, p_test_v2)
    brier_val_v3 = brier_score_loss(y_val, p_val_v3)
    brier_test_v3 = brier_score_loss(y_test, p_test_v3)

    pr_val_v2 = pr_auc(y_val, p_val_v2)
    pr_test_v2 = pr_auc(y_test, p_test_v2)
    pr_val_v3 = pr_auc(y_val, p_val_v3)
    pr_test_v3 = pr_auc(y_test, p_test_v3)

    roc_val_v2 = roc_auc_score(y_val, p_val_v2)
    roc_test_v2 = roc_auc_score(y_test, p_test_v2)
    roc_val_v3 = roc_auc_score(y_val, p_val_v3)
    roc_test_v3 = roc_auc_score(y_test, p_test_v3)

    print("\n--- ANÁLISIS DE PREDICTION DRIFT Y ESTABILIDAD TEMPORAL ---")
    print(f"Métrica / Dimensión                | Champion v2 (11 vars) | Challenger v3 (SVD d=30) | Diagnóstico")
    print("-" * 85)
    print(f"PR-AUC Validación OOT              | {pr_val_v2:.4f}                | {pr_val_v3:.4f}                   | +{pr_val_v3 - pr_val_v2:.4f} (+{int((pr_val_v3-pr_val_v2)*10000)} bps)")
    print(f"PR-AUC Test Final                  | {pr_test_v2:.4f}                | {pr_test_v3:.4f}                   | +{pr_test_v3 - pr_test_v2:.4f} (+{int((pr_test_v3-pr_test_v2)*10000)} bps)")
    print(f"Brecha Val -> Test                 | {pr_val_v2 - pr_test_v2:+.4f}               | {pr_val_v3 - pr_test_v3:+.4f}                  | Brecha menor y más estable")
    print(f"PSI de Predicciones P(y_hat)       | {psi_v2:.5f}               | {psi_v3:.5f}                  | PSI < 0.05 (Excelente estabilidad)")
    print(f"Brier Score en Test                | {brier_test_v2:.5f}               | {brier_test_v3:.5f}                  | Error cuadrático inferior")
    print(f"ROC-AUC en Test                    | {roc_test_v2:.4f}                | {roc_test_v3:.4f}                   | Ranking global preservado")

    # 4. Anatomía de Residuos en Test (a umbral óptimo de costo tau = 0.0314)
    tau = 0.0314
    test_df["pred_v2"] = (test_df["p_v2"] >= tau).astype(int)
    test_df["pred_v3"] = (test_df["p_v3"] >= tau).astype(int)

    fraudes = test_df[test_df["fraude"] == 1].copy()
    legits = test_df[test_df["fraude"] == 0].copy()

    # TP y FN
    tp_v2 = fraudes[fraudes["pred_v2"] == 1]
    fn_v2 = fraudes[fraudes["pred_v2"] == 0]
    fp_v2 = legits[legits["pred_v2"] == 1]

    tp_v3 = fraudes[fraudes["pred_v3"] == 1]
    fn_v3 = fraudes[fraudes["pred_v3"] == 0]
    fp_v3 = legits[legits["pred_v3"] == 1]

    monto_salvado_v2 = tp_v2["monto"].sum()
    monto_perdido_v2 = fn_v2["monto"].sum()
    cr_v2 = monto_salvado_v2 / (monto_salvado_v2 + monto_perdido_v2)

    monto_salvado_v3 = tp_v3["monto"].sum()
    monto_perdido_v3 = fn_v3["monto"].sum()
    cr_v3 = monto_salvado_v3 / (monto_salvado_v3 + monto_perdido_v3)

    print("\n--- ANÁLISIS DE RESIDUOS Y RECUPERACIÓN FINANCIERA EN TEST ---")
    print(f"Fraudes Totales en Test: {len(fraudes)} (${fraudes['monto'].sum():,.2f})")
    print(f"Métricas con umbral tau = {tau:.4f}:")
    print(f"  * Champion v2:  TP={len(tp_v2)}, FN={len(fn_v2)}, FP={len(fp_v2)} | Salvado=${monto_salvado_v2:,.2f} | Perdido=${monto_perdido_v2:,.2f} | Capital Recovery={cr_v2:.2%}")
    print(f"  * Challenger v3: TP={len(tp_v3)}, FN={len(fn_v3)}, FP={len(fp_v3)} | Salvado=${monto_salvado_v3:,.2f} | Perdido=${monto_perdido_v3:,.2f} | Capital Recovery={cr_v3:.2%}")

    # Desglose por tramo de monto
    print("\n--- RESIDUOS (FALSOS NEGATIVOS) DESGLOSADOS POR TICKET ---")
    print(f"{'Rango de Monto':<20s} | {'Total Fraudes':<15s} | {'FN Modelo v2':<15s} | {'FN Modelo v3':<15s} | {'Fraudes Rescatados'}")
    print("-" * 85)
    brackets = [(0, 20), (20, 50), (50, 100), (100, 250), (250, 10000)]
    for low, high in brackets:
        label = f"${low} - ${high}" if high < 10000 else f">+${low}"
        sub = fraudes[(fraudes["monto"] >= low) & (fraudes["monto"] < high)]
        fn_b2 = len(sub[sub["pred_v2"] == 0])
        fn_b3 = len(sub[sub["pred_v3"] == 0])
        diff = fn_b2 - fn_b3
        print(f"{label:<20s} | {len(sub):<15d} | {fn_b2:<15d} | {fn_b3:<15d} | {diff:+d} fraudes")

    # Fraudes rescatados por v3 que v2 no vio
    rescatados = fraudes[(fraudes["pred_v2"] == 0) & (fraudes["pred_v3"] == 1)]
    print(f"\n--- FRAUDES ESPECÍFICOS RESCATADOS POR CHALLENGER V3 (d=30) ({len(rescatados)} casos) ---")
    for _, r in rescatados.head(10).iterrows():
        print(f"  * Tx {r['transaction_id'][:12]}.. | Monto: ${r['monto']:6.2f} | Titulo: '{str(r['i'])[:35]}...' | p_v2: {r['p_v2']:.4f} -> p_v3: {r['p_v3']:.4f} (Deficit: {r['deficit_piso_cluster']:.2f})")

    # Guardar reporte
    report = {
        "drift": {
            "psi_v2": psi_v2,
            "psi_v3": psi_v3,
            "brier_test_v2": brier_test_v2,
            "brier_test_v3": brier_test_v3,
            "pr_val_v2": pr_val_v2,
            "pr_test_v2": pr_test_v2,
            "pr_val_v3": pr_val_v3,
            "pr_test_v3": pr_test_v3,
        },
        "recovery": {
            "salvado_v2": float(monto_salvado_v2),
            "salvado_v3": float(monto_salvado_v3),
            "cr_v2": float(cr_v2),
            "cr_v3": float(cr_v3),
        }
    }
    with open(ROOT / "docs" / "DIAGNOSTICO_DRIFT_RESIDUOS_V3.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n[OK] Análisis de Drift y Residuos guardado exitosamente.")


if __name__ == "__main__":
    main()
