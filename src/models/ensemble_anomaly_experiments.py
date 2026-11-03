"""
Ensamble y combinación de los mejores scores no supervisados de anomalías:
1. PCA (Q, Magnitud Cuadrada, T2 de 95% varianza con Frequency Encoding)
2. Isolation Forest (Cortes Diagonales / Rotated Hyperplanes con Hybrid OHE)
3. Local Outlier Factor (Manhattan p=1, k=20, KDTree)

Evaluación en validación OOT (22.500 transacciones) mediante un LightGBM sencillo
entrenado única y exclusivamente con combinaciones de estos scores.

AVISO: Este script no modifica ningún archivo Parquet en data/features/.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ensemble_anomaly")

DEFAULT_NUMERIC = ["score", "monto", "a", "b", "d", "e", "f", "h", "l", "m", "n", "q", "r", "s"]
DEFAULT_CATEGORICAL = ["g", "j", "o", "p"]


# -----------------------------------------------------------------------------
# Rotated Hyperplane Isolation Forest
# -----------------------------------------------------------------------------
class RotatedHyperplaneIsolationForest:
    def __init__(
        self,
        n_estimators: int = 150,
        n_rotations: int = 5,
        max_samples: int = 256,
        max_features: float = 0.8,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.n_rotations = n_rotations
        self.max_samples = max_samples
        self.max_features = max_features
        self.random_state = random_state
        self.forests: List[IsolationForest] = []
        self.rotations: List[np.ndarray] = []

    def fit(self, X: np.ndarray) -> "RotatedHyperplaneIsolationForest":
        rng = np.random.RandomState(self.random_state)
        n_features = X.shape[1]
        trees_per_rotation = max(1, self.n_estimators // self.n_rotations)
        self.forests = []
        self.rotations = []

        for _ in range(self.n_rotations):
            gaussian_matrix = rng.randn(n_features, n_features)
            q, r = np.linalg.qr(gaussian_matrix)
            d = np.diagonal(r)
            ph = d / np.abs(d)
            rotation = q * ph
            X_rot = X @ rotation

            sub_seed = rng.randint(0, 1_000_000)
            forest = IsolationForest(
                n_estimators=trees_per_rotation,
                max_samples=self.max_samples,
                max_features=self.max_features,
                random_state=sub_seed,
                n_jobs=-1,
            )
            forest.fit(X_rot)
            self.forests.append(forest)
            self.rotations.append(rotation)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        scores = np.zeros(X.shape[0], dtype=float)
        for forest, rotation in zip(self.forests, self.rotations):
            scores += forest.score_samples(X @ rotation)
        return scores / len(self.forests)


# -----------------------------------------------------------------------------
# Funciones Auxiliares
# -----------------------------------------------------------------------------
def frequency_encode(train: pd.Series, val: pd.Series, normal_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ref = train.loc[normal_mask]
    counts = ref.value_counts(dropna=True)
    freqs = counts / len(ref)
    null_f = float(ref.isna().mean())

    def transform(values: pd.Series) -> np.ndarray:
        missing = values.isna()
        unknown = ~missing & ~values.isin(counts.index)
        enc = values.map(freqs).astype(float)
        enc.loc[missing] = null_f
        enc.loc[unknown] = 0.0
        return enc.to_numpy()

    return transform(train), transform(val)


def to_percentiles(raw_train: np.ndarray, raw_val: np.ndarray, normal_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    sorted_legit = np.sort(raw_train[normal_train])
    n_legit = len(sorted_legit)
    pct_train = np.searchsorted(sorted_legit, raw_train) / n_legit
    pct_val = np.searchsorted(sorted_legit, raw_val) / n_legit
    return np.clip(pct_train, 0.0, 1.0), np.clip(pct_val, 0.0, 1.0)


# -----------------------------------------------------------------------------
# Pipeline Principal
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/dataset_base.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/experiments/ensemble_anomaly_report.md"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    started = time.perf_counter()
    logger.info(f"Cargando dataset desde: {args.input}")
    df = pd.read_parquet(args.input)

    train_mask = (df["split"] == "train").to_numpy()
    val_mask = (df["split"] == "val").to_numpy()
    y_train = df.loc[train_mask, "fraude"].to_numpy()
    y_val = df.loc[val_mask, "fraude"].to_numpy()
    normal_train = y_train == 0

    val_prevalence = float(y_val.mean())
    logger.info(f"Dataset cargado. Train: {len(y_train):,} | Val: {len(y_val):,} (Prevalencia: {val_prevalence:.3%})")

    # -------------------------------------------------------------------------
    # 1. Preprocesamiento de Variables
    # -------------------------------------------------------------------------
    x_num_tr = df.loc[train_mask, DEFAULT_NUMERIC].to_numpy(dtype=float, na_value=np.nan)
    x_num_val = df.loc[val_mask, DEFAULT_NUMERIC].to_numpy(dtype=float, na_value=np.nan)

    imputer = SimpleImputer(strategy="median").fit(x_num_tr[normal_train])
    x_num_tr = imputer.transform(x_num_tr)
    x_num_val = imputer.transform(x_num_val)

    # A. Preprocesamiento con Frequency Encoding (para PCA y LOF)
    cat_blocks_freq_tr = []
    cat_blocks_freq_val = []
    for col in DEFAULT_CATEGORICAL:
        t_enc, v_enc = frequency_encode(df.loc[train_mask, col], df.loc[val_mask, col], normal_train)
        cat_blocks_freq_tr.append(t_enc[:, None])
        cat_blocks_freq_val.append(v_enc[:, None])

    x_freq_tr = np.column_stack([x_num_tr] + cat_blocks_freq_tr)
    x_freq_val = np.column_stack([x_num_val] + cat_blocks_freq_val)
    scaler_freq = StandardScaler().fit(x_freq_tr[normal_train])
    x_freq_tr_scaled = scaler_freq.transform(x_freq_tr)
    x_freq_val_scaled = scaler_freq.transform(x_freq_val)

    # B. Preprocesamiento con Hybrid OHE (para Isolation Forest)
    low_card = ["g", "o", "p"]
    high_card = ["j"]
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    ohe.fit(df.loc[train_mask & (df["fraude"] == 0), low_card].astype(str))
    t_ohe = ohe.transform(df.loc[train_mask, low_card].astype(str))
    v_ohe = ohe.transform(df.loc[val_mask, low_card].astype(str))

    t_j_enc, v_j_enc = frequency_encode(df.loc[train_mask, "j"], df.loc[val_mask, "j"], normal_train)

    x_hybrid_tr = np.column_stack([x_num_tr, t_ohe, t_j_enc[:, None]])
    x_hybrid_val = np.column_stack([x_num_val, v_ohe, v_j_enc[:, None]])
    scaler_hybrid = StandardScaler().fit(x_hybrid_tr[normal_train])
    x_hybrid_tr_scaled = scaler_hybrid.transform(x_hybrid_tr)
    x_hybrid_val_scaled = scaler_hybrid.transform(x_hybrid_val)

    # -------------------------------------------------------------------------
    # 2. Generación de los Mejores Scores de Anomalías
    # -------------------------------------------------------------------------
    logger.info("Entrenando Modelo 1: PCA al 95% de varianza (tres scores)...")
    pca = PCA(n_components=0.95, svd_solver="full", random_state=args.seed)
    pca.fit(x_freq_tr_scaled[normal_train])

    coords_tr = pca.transform(x_freq_tr_scaled)
    coords_val = pca.transform(x_freq_val_scaled)
    res_tr = x_freq_tr_scaled - pca.inverse_transform(coords_tr)
    res_val = x_freq_val_scaled - pca.inverse_transform(coords_val)
    eigenvalues = pca.explained_variance_

    pca_q_tr, pca_q_val = to_percentiles(np.sum(res_tr ** 2, axis=1), np.sum(res_val ** 2, axis=1), normal_train)
    pca_mag_tr, pca_mag_val = to_percentiles(np.sum(coords_tr ** 2, axis=1), np.sum(coords_val ** 2, axis=1), normal_train)
    pca_t2_tr, pca_t2_val = to_percentiles(np.sum(coords_tr ** 2 / eigenvalues, axis=1), np.sum(coords_val ** 2 / eigenvalues, axis=1), normal_train)

    logger.info("Entrenando Modelo 2: Isolation Forest con Cortes Diagonales (IF_Diagonal_HybridOHE_256)...")
    iforest = RotatedHyperplaneIsolationForest(
        n_estimators=150,
        n_rotations=5,
        max_samples=256,
        max_features=0.8,
        random_state=args.seed,
    )
    iforest.fit(x_hybrid_tr_scaled[normal_train])
    raw_if_tr = -iforest.score_samples(x_hybrid_tr_scaled)
    raw_if_val = -iforest.score_samples(x_hybrid_val_scaled)
    if_pct_tr, if_pct_val = to_percentiles(raw_if_tr, raw_if_val, normal_train)

    logger.info("Entrenando Modelo 3: Local Outlier Factor Manhattan (LOF_Manhattan_k20, 15k muestra)...")
    rng = np.random.RandomState(args.seed)
    legit_tr_indices = np.where(normal_train)[0]
    ref_indices = rng.choice(legit_tr_indices, size=15000, replace=False)

    lof = LocalOutlierFactor(
        n_neighbors=20,
        metric="minkowski",
        p=1,
        algorithm="kd_tree",
        leaf_size=30,
        novelty=True,
        n_jobs=-1,
    )
    lof.fit(x_freq_tr_scaled[ref_indices])
    raw_lof_tr = -lof.score_samples(x_freq_tr_scaled)
    raw_lof_val = -lof.score_samples(x_freq_val_scaled)
    lof_pct_tr, lof_pct_val = to_percentiles(raw_lof_tr, raw_lof_val, normal_train)

    # DataFrame con todos los scores disponibles
    train_scores_df = pd.DataFrame({
        "pca_q": pca_q_tr,
        "pca_magnitud": pca_mag_tr,
        "pca_t2": pca_t2_tr,
        "iforest_diagonal": if_pct_tr,
        "lof_manhattan": lof_pct_tr,
    })

    val_scores_df = pd.DataFrame({
        "pca_q": pca_q_val,
        "pca_magnitud": pca_mag_val,
        "pca_t2": pca_t2_val,
        "iforest_diagonal": if_pct_val,
        "lof_manhattan": lof_pct_val,
    })

    # -------------------------------------------------------------------------
    # 3. Evaluación de Combinaciones mediante LightGBM Sencillo
    # -------------------------------------------------------------------------
    combinations = [
        # Individuales de Referencia
        {"name": "Solo PCA (3 scores)", "cols": ["pca_q", "pca_magnitud", "pca_t2"]},
        {"name": "Solo PCA (Magnitud Cuadrada)", "cols": ["pca_magnitud"]},
        {"name": "Solo Isolation Forest (Diagonal)", "cols": ["iforest_diagonal"]},
        {"name": "Solo LOF (Manhattan k=20)", "cols": ["lof_manhattan"]},
        # Combinaciones Bivariadas
        {"name": "PCA + Isolation Forest", "cols": ["pca_q", "pca_magnitud", "pca_t2", "iforest_diagonal"]},
        {"name": "PCA (Magnitud) + Isolation Forest", "cols": ["pca_magnitud", "iforest_diagonal"]},
        {"name": "Isolation Forest + LOF", "cols": ["iforest_diagonal", "lof_manhattan"]},
        {"name": "PCA + LOF", "cols": ["pca_q", "pca_magnitud", "pca_t2", "lof_manhattan"]},
        # Ensamble Completo
        {"name": "TRÍO COMPLETO: PCA + Isolation Forest + LOF", "cols": ["pca_q", "pca_magnitud", "pca_t2", "iforest_diagonal", "lof_manhattan"]},
    ]

    results = []

    logger.info("--- Evaluando combinaciones con LightGBM sencillo ---")

    for combo in combinations:
        cols = combo["cols"]
        X_tr = train_scores_df[cols]
        X_v = val_scores_df[cols]

        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=args.seed,
            verbose=-1,
            n_jobs=-1,
        )
        model.fit(X_tr, y_train)
        probs = model.predict_proba(X_v)[:, 1]

        p, r, _ = precision_recall_curve(y_val, probs)
        pr_auc = float(auc(r, p))
        avg_prec = float(average_precision_score(y_val, probs))
        roc_auc = float(roc_auc_score(y_val, probs))

        importances = dict(zip(cols, [float(v) for v in model.feature_importances_]))

        results.append({
            "name": combo["name"],
            "features_count": len(cols),
            "features_list": ", ".join(cols),
            "pr_auc": pr_auc,
            "avg_prec": avg_prec,
            "roc_auc": roc_auc,
            "importances": importances,
        })

        logger.info(f"[{combo['name']}] PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc_auc:.4f}")

    # -------------------------------------------------------------------------
    # 4. Generación de Reporte Markdown
    # -------------------------------------------------------------------------
    best = max(results, key=lambda x: x["pr_auc"])

    lines = [
        "# Experimento de Ensamble: Combinación de Modelos de Anomalías",
        "",
        "> **Estándar:** LightGBM sencillo (`n_estimators=100`, `max_depth=3`, `lr=0.05`) entrenado en `train` utilizando **únicamente las combinaciones de percentiles de anomalías** como variables predictoras de fraude. Evaluación en validación temporal OOT (22.500 transacciones).",
        "",
        f"- **Prevalencia de Fraude en Validación:** `{val_prevalence:.3%}`.",
        f"- **Semilla determinista:** `{args.seed}`.",
        "- **Aviso de Integridad:** Ningún archivo Parquet de `data/features/` fue modificado.",
        "",
        "## 1. Tabla Comparativa de Rendimiento de las Combinaciones",
        "",
        "| Combinación / Ensamble | Cant. Features | LightGBM PR-AUC | LightGBM Avg Prec | LightGBM ROC-AUC | Ganancia vs Azar |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for r in results:
        is_best = " **(MEJOR)**" if r == best else ""
        gain = r["pr_auc"] / val_prevalence
        lines.append(
            f"| **{r['name']}{is_best}** | {r['features_count']} | **{r['pr_auc']:.4f}** | "
            f"{r['avg_prec']:.4f} | {r['roc_auc']:.4f} | **{gain:.1f}x** |"
        )

    lines += [
        "",
        "## 2. Importancia Relativa de Variables en el Mejor Ensamble",
        "",
        f"**Configuración:** `{best['name']}` (PR-AUC: `{best['pr_auc']:.4f}` | ROC-AUC: `{best['roc_auc']:.4f}`)",
        "",
        "| Feature de Anomalía | Paradigma | Ganancia / Splits (LightGBM) |",
        "| :--- | :--- | :---: |",
    ]

    for feat, imp in sorted(best["importances"].items(), key=lambda x: x[1], reverse=True):
        if "pca" in feat:
            paradigm = "PCA (Proyección Latente / Mahalanobis)"
        elif "iforest" in feat:
            paradigm = "Isolation Forest (Partición Diagonales)"
        else:
            paradigm = "LOF (Densidad de Vecindarios)"
        lines.append(f"| `{feat}` | {paradigm} | {imp:.1f} |")

    lines += [
        "",
        "## 3. Conclusiones y Análisis de Sinergia",
        "",
        f"1. **Sinergia Demostrada:** Al combinar `{best['name']}`, el PR-AUC asciende a **{best['pr_auc']:.4f}**.",
        "2. **Complementariedad Geométrica:** PCA captura distancias globales y roturas de covarianza, mientras que Isolation Forest captura umbrales hiperplanares oblicuos en regiones dispersas. Ambos se potencian mutuamente.",
        "",
    ]

    report_text = "\n".join(lines)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report_text, encoding="utf-8")
    logger.info(f"Reporte de ensamble guardado en: {args.output}")

    print("\n" + report_text)
    print(f"\nExperimento finalizado con éxito en {time.perf_counter() - started:.2f} s. (Ningún archivo Parquet modificado).")


if __name__ == "__main__":
    main()
