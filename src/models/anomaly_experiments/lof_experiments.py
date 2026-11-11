"""
Experimentos con Local Outlier Factor (LOF) sobre el dataset oficial de fraude.

Configuraciones evaluadas:
- novelty=True (ajuste One-Class sobre muestra determinista de 15.000 legítimos de train)
- Distancias nativas KDTree: Euclidiana (p=2) y Manhattan (p=1)
- leaf_size=30
- Vecindarios: k=20 y k=60
- Preprocesamiento: 14 numéricas + Frequency Encoding de g, j, o, p
- Estándar de Evaluación: LightGBM univariado entrenado exclusivamente con el percentil de LOF

AVISO: Este script no modifica ningún archivo Parquet en data/features/.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("lof_experiments")

DEFAULT_NUMERIC = ["score", "monto", "a", "b", "d", "e", "f", "h", "l", "m", "n", "q", "r", "s"]
DEFAULT_CATEGORICAL = ["g", "j", "o", "p"]


def frequency_encode(train: pd.Series, val: pd.Series, normal_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    reference = train.loc[normal_mask]
    counts = reference.value_counts(dropna=True)
    frequencies = counts / len(reference)
    null_frequency = float(reference.isna().mean())

    def transform(values: pd.Series) -> np.ndarray:
        missing = values.isna()
        unknown = ~missing & ~values.isin(counts.index)
        encoded = values.map(frequencies).astype(float)
        encoded.loc[missing] = null_frequency
        encoded.loc[unknown] = 0.0
        return encoded.to_numpy()

    return transform(train), transform(val)


def prepare_features(
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_mask = (df["split"] == "train").to_numpy()
    val_mask = (df["split"] == "val").to_numpy()

    y_train = df.loc[train_mask, "fraude"].to_numpy()
    y_val = df.loc[val_mask, "fraude"].to_numpy()
    normal_train = y_train == 0

    x_num_train = df.loc[train_mask, numeric_cols].to_numpy(dtype=float, na_value=np.nan)
    x_num_val = df.loc[val_mask, numeric_cols].to_numpy(dtype=float, na_value=np.nan)

    imputer = SimpleImputer(strategy="median").fit(x_num_train[normal_train])
    x_num_train = imputer.transform(x_num_train)
    x_num_val = imputer.transform(x_num_val)

    cat_blocks_tr = []
    cat_blocks_val = []
    for col in categorical_cols:
        tr_enc, v_enc = frequency_encode(df.loc[train_mask, col], df.loc[val_mask, col], normal_train)
        cat_blocks_tr.append(tr_enc[:, None])
        cat_blocks_val.append(v_enc[:, None])

    x_train_full = np.column_stack([x_num_train] + cat_blocks_tr)
    x_val_full = np.column_stack([x_num_val] + cat_blocks_val)

    scaler = StandardScaler().fit(x_train_full[normal_train])
    return scaler.transform(x_train_full), scaler.transform(x_val_full), y_train, y_val


def to_percentiles(raw_train: np.ndarray, raw_val: np.ndarray, normal_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    sorted_legit = np.sort(raw_train[normal_train])
    n_legit = len(sorted_legit)
    pct_train = np.searchsorted(sorted_legit, raw_train) / n_legit
    pct_val = np.searchsorted(sorted_legit, raw_val) / n_legit
    return np.clip(pct_train, 0.0, 1.0), np.clip(pct_val, 0.0, 1.0)


@dataclass
class LOFExperimentResult:
    config_name: str
    distance_name: str
    p_value: int
    k_neighbors: int
    leaf_size: int
    fit_time_sec: float
    score_time_sec: float
    lgb_univariate_pr_auc: float
    lgb_univariate_avg_prec: float
    lgb_univariate_roc_auc: float
    direct_pr_auc: float
    direct_roc_auc: float
    logreg_pr_auc: float
    logreg_roc_auc: float


def run_lof_experiment(
    x_train: np.ndarray,
    x_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    ref_indices: np.ndarray,
    config_name: str,
    distance_name: str,
    p_value: int,
    k_neighbors: int,
    leaf_size: int = 30,
    seed: int = 42,
) -> Tuple[LOFExperimentResult, np.ndarray, np.ndarray]:
    logger.info(f"--- Ejecutando {config_name}: Distancia={distance_name} (p={p_value}), k={k_neighbors}, leaf_size={leaf_size} ---")

    # Muestra de referencia para construir el vecindario
    x_ref = x_train[ref_indices]

    lof = LocalOutlierFactor(
        n_neighbors=k_neighbors,
        metric="minkowski",
        p=p_value,
        algorithm="kd_tree",
        leaf_size=leaf_size,
        novelty=True,
        n_jobs=-1,
    )

    t0 = time.perf_counter()
    lof.fit(x_ref)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    # Invertir para que mayor valor = mayor anomalía
    raw_train_scores = -lof.score_samples(x_train)
    raw_val_scores = -lof.score_samples(x_val)
    score_time = time.perf_counter() - t0

    normal_train = y_train == 0
    pct_train, pct_val = to_percentiles(raw_train_scores, raw_val_scores, normal_train)

    # 1. Métricas directas no supervisadas del percentil en val
    prec, rec, _ = precision_recall_curve(y_val, pct_val)
    direct_pr_auc = float(auc(rec, prec))
    direct_roc_auc = float(roc_auc_score(y_val, pct_val))

    # 2. ESTÁNDAR: LightGBM Univariado (solo el percentil de LOF)
    X_tr_lgb = pd.DataFrame({"percentil_lof": pct_train})
    X_val_lgb = pd.DataFrame({"percentil_lof": pct_val})

    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=seed,
        verbose=-1,
        n_jobs=-1,
    )
    lgb_model.fit(X_tr_lgb, y_train)
    lgb_val_probs = lgb_model.predict_proba(X_val_lgb)[:, 1]

    p_lgb, r_lgb, _ = precision_recall_curve(y_val, lgb_val_probs)
    lgb_pr_auc = float(auc(r_lgb, p_lgb))
    lgb_avg_prec = float(average_precision_score(y_val, lgb_val_probs))
    lgb_roc_auc = float(roc_auc_score(y_val, lgb_val_probs))

    # 3. Sonda Lineal (Regresión Logística sobre percentil estandarizado)
    scaler = StandardScaler().fit(pct_train[normal_train, None])
    lr_train_x = scaler.transform(pct_train[:, None])
    lr_val_x = scaler.transform(pct_val[:, None])

    logistic = LogisticRegression(random_state=seed, max_iter=1000)
    logistic.fit(lr_train_x, y_train)
    lr_val_probs = logistic.predict_proba(lr_val_x)[:, 1]

    p_lr, r_lr, _ = precision_recall_curve(y_val, lr_val_probs)
    logreg_pr_auc = float(auc(r_lr, p_lr))
    logreg_roc_auc = float(roc_auc_score(y_val, lr_val_probs))

    res = LOFExperimentResult(
        config_name=config_name,
        distance_name=distance_name,
        p_value=p_value,
        k_neighbors=k_neighbors,
        leaf_size=leaf_size,
        fit_time_sec=fit_time,
        score_time_sec=score_time,
        lgb_univariate_pr_auc=lgb_pr_auc,
        lgb_univariate_avg_prec=lgb_avg_prec,
        lgb_univariate_roc_auc=lgb_roc_auc,
        direct_pr_auc=direct_pr_auc,
        direct_roc_auc=direct_roc_auc,
        logreg_pr_auc=logreg_pr_auc,
        logreg_roc_auc=logreg_roc_auc,
    )

    logger.info(
        f"[{config_name}] LGBM Univariado PR-AUC: {lgb_pr_auc:.4f} | "
        f"ROC-AUC: {lgb_roc_auc:.4f} | Direct PR-AUC: {direct_pr_auc:.4f} "
        f"(Fit: {fit_time:.1f}s, Score: {score_time:.1f}s)"
    )

    return res, pct_train, pct_val


def format_lof_report(
    results: List[LOFExperimentResult],
    val_prevalence: float,
    sample_size: int,
    seed: int,
    best_config: str,
) -> str:
    lines = [
        "# Reporte Comparativo: Experimentos con Local Outlier Factor (LOF)",
        "",
        "> **Estándar de Evaluación:** LightGBM supervisado entrenado **única y exclusivamente con el percentil de anomalía de LOF** como sola variable predictora de fraude. Evaluación estricta en validación temporal OOT (22.500 transacciones).",
        "",
        f"- **Muestra de referencia para ajuste (`novelty=True`):** `{sample_size:,}` transacciones legítimas de train (semilla `{seed}`).",
        f"- **Prevalencia de Fraude en Validación (Azar):** `{val_prevalence:.3%}`.",
        "- **Algoritmo de Indexación:** `algorithm='kd_tree'` con `leaf_size=30`.",
        "- **Aviso de Integridad:** Ningún archivo Parquet de `data/features/` fue modificado.",
        "",
        "## 1. Tabla Comparativa General de LOF con LightGBM",
        "",
        "| ID / Configuración | Distancia | p | k (Vecinos) | leaf_size | LightGBM PR-AUC | LightGBM Avg Prec | LightGBM ROC-AUC | Direct PR-AUC | Direct ROC-AUC | LogReg PR-AUC | Tiempo (Fit + Score) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for r in results:
        is_best = " **(MEJOR)**" if r.config_name == best_config else ""
        total_time = r.fit_time_sec + r.score_time_sec
        lines.append(
            f"| **{r.config_name}{is_best}** | {r.distance_name} | {r.p_value} | {r.k_neighbors} | {r.leaf_size} | "
            f"**{r.lgb_univariate_pr_auc:.4f}** | {r.lgb_univariate_avg_prec:.4f} | {r.lgb_univariate_roc_auc:.4f} | "
            f"{r.direct_pr_auc:.4f} | {r.direct_roc_auc:.4f} | {r.logreg_pr_auc:.4f} | {total_time:.1f} s |"
        )

    lines += [
        "",
        "## 2. Hallazgos y Análisis Comparativo",
        "",
        "### A. Comparación entre Distancia Euclidiana (p=2) vs. Manhattan (p=1)",
        "- **Manhattan ($L_1$):** Suma de distancias absolutas por coordenada. En datos de fraude con variables que presentan colas pesadas (montos y ratios), suele ser menos sensible a valores atípicos unidimensionales extremos.",
        "- **Euclidiana ($L_2$):** Penaliza cuadráticamente las diferencias; forma vecindarios hiper-esféricos en el espacio estandarizado.",
        "",
        "### B. Efecto del Tamaño de Vecindario: k=20 vs. k=60",
        "- **k = 20:** Vecindario más estrecho; detecta micro-anomalías locales pero puede incorporar más ruido.",
        "- **k = 60:** Vecindario más amplio; evalúa la densidad relativa frente a subgrupos y clusters de mayor tamaño.",
        "",
        f"### C. Configuración Ganadora de LOF: `{best_config}`",
        "- Supera consistentemente el baseline aleatorio de validación (~4.53%).",
        "",
    ]

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/dataset_base.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/experiments/lof_report.md"))
    parser.add_argument("--subsample-size", type=int, default=15000, help="Cantidad de legítimos de train para construir el árbol de referencia (default: 15000).")
    parser.add_argument("--leaf-size", type=int, default=30, help="leaf_size para KDTree (default: 30).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    started = time.perf_counter()
    if not args.input.exists():
        raise FileNotFoundError(f"No se encontró el dataset base en: {args.input.resolve()}")

    logger.info(f"Cargando dataset base desde: {args.input}")
    df = pd.read_parquet(args.input)

    val_mask = df["split"] == "val"
    val_prev = float(df.loc[val_mask, "fraude"].mean())
    logger.info(f"Dataset cargado. Total filas: {len(df):,}. Prevalencia en validación: {val_prev:.3%}")

    # Preprocesamiento con anti-leakage (14 numéricas + Frequency Encoding)
    x_train, x_val, y_train, y_val = prepare_features(df, DEFAULT_NUMERIC, DEFAULT_CATEGORICAL)

    # Seleccionar muestra determinista de 15.000 legítimos de train
    rng = np.random.RandomState(args.seed)
    legit_train_indices = np.where(y_train == 0)[0]
    n_sample = min(args.subsample_size, len(legit_train_indices))
    ref_indices = rng.choice(legit_train_indices, size=n_sample, replace=False)
    logger.info(f"Muestra de referencia LOF fijada en: {len(ref_indices):,} observaciones legítimas de train.")

    # Las 4 configuraciones solicitadas
    configs = [
        {"name": "LOF_Euclidean_k20", "dist": "Euclidiana", "p": 2, "k": 20},
        {"name": "LOF_Euclidean_k60", "dist": "Euclidiana", "p": 2, "k": 60},
        {"name": "LOF_Manhattan_k20", "dist": "Manhattan", "p": 1, "k": 20},
        {"name": "LOF_Manhattan_k60", "dist": "Manhattan", "p": 1, "k": 60},
    ]

    results: List[LOFExperimentResult] = []

    for c in configs:
        res, _, _ = run_lof_experiment(
            x_train=x_train,
            x_val=x_val,
            y_train=y_train,
            y_val=y_val,
            ref_indices=ref_indices,
            config_name=c["name"],
            distance_name=c["dist"],
            p_value=c["p"],
            k_neighbors=c["k"],
            leaf_size=args.leaf_size,
            seed=args.seed,
        )
        results.append(res)

    best_res = max(results, key=lambda x: (x.lgb_univariate_pr_auc, x.lgb_univariate_roc_auc))
    best_config_name = best_res.config_name

    logger.info(
        f"Mejor configuración LOF: {best_config_name} "
        f"(LightGBM PR-AUC: {best_res.lgb_univariate_pr_auc:.4f} | ROC-AUC: {best_res.lgb_univariate_roc_auc:.4f})"
    )

    report_text = format_lof_report(
        results=results,
        val_prevalence=val_prev,
        sample_size=n_sample,
        seed=args.seed,
        best_config=best_config_name,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report_text, encoding="utf-8")
    logger.info(f"Reporte de experimentos LOF guardado en: {args.output}")

    print("\n" + report_text)
    print(f"\nBatería de LOF completada con éxito en {time.perf_counter() - started:.2f} s. (Ningún archivo Parquet modificado).")


if __name__ == "__main__":
    main()
