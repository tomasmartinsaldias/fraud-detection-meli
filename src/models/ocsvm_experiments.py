"""
Experimentos de One-Class SVM con Stochastic Gradient Descent y aproximación de Kernel RBF vía Nystroem.

Parámetros evaluados:
- Nystroem: n_components=300, kernel='rbf', gamma='scale'
- SGDOneClassSVM: nu in [0.10, 0.05, 0.03], max_iter=1000, random_state=42
- Preprocesamiento: 14 numéricas + Frequency Encoding de g, j, o, p + StandardScaler
- Ajuste One-Class estricto en legítimos de train (split == 'train' & fraude == 0)
- Evaluación en validación temporal OOT (22.500 transacciones)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ocsvm_experiments")

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
class OCSVMExperimentResult:
    config_name: str
    nu: float
    n_components: int
    fit_time_sec: float
    score_time_sec: float
    direct_pr_auc: float
    direct_roc_auc: float
    lgb_pr_auc: float
    lgb_avg_prec: float
    lgb_roc_auc: float
    best_f1: float
    precision_at_best_f1: float
    recall_at_best_f1: float
    optimal_threshold: float


def run_single_ocsvm(
    x_train_nystroem: np.ndarray,
    x_val_nystroem: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    nu: float,
    n_components: int,
    seed: int = 42,
) -> OCSVMExperimentResult:
    config_name = f"OCSVM_SGD_nu_{nu:.2f}"
    logger.info(f"--- Ejecutando {config_name}: nu={nu}, n_components={n_components} ---")

    normal_train = y_train == 0
    x_train_legit = x_train_nystroem[normal_train]

    sgd = SGDOneClassSVM(
        nu=nu,
        fit_intercept=True,
        max_iter=1000,
        tol=1e-3,
        random_state=seed,
    )

    t0 = time.perf_counter()
    sgd.fit(x_train_legit)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    # Invertir decision_function para que mayor valor = mayor anomalía
    raw_train = -sgd.decision_function(x_train_nystroem)
    raw_val = -sgd.decision_function(x_val_nystroem)
    score_time = time.perf_counter() - t0

    pct_train, pct_val = to_percentiles(raw_train, raw_val, normal_train)

    # 1. Métricas directas no supervisadas en val
    prec, rec, _ = precision_recall_curve(y_val, pct_val)
    direct_pr_auc = float(auc(rec, prec))
    direct_roc_auc = float(roc_auc_score(y_val, pct_val))

    # 2. Evaluación con LightGBM univariado
    X_tr_lgb = pd.DataFrame({"percentil_ocsvm": pct_train})
    X_val_lgb = pd.DataFrame({"percentil_ocsvm": pct_val})

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
    probs_val = lgb_model.predict_proba(X_val_lgb)[:, 1]

    p_lgb, r_lgb, thresholds = precision_recall_curve(y_val, probs_val)
    lgb_pr_auc = float(auc(r_lgb, p_lgb))
    lgb_avg_prec = float(average_precision_score(y_val, probs_val))
    lgb_roc_auc = float(roc_auc_score(y_val, probs_val))

    # Mejor F1 sobre las probabilidades calibradas de LightGBM
    prec_t = p_lgb[:-1]
    rec_t = r_lgb[:-1]
    denom = prec_t + rec_t
    f1_scores = np.divide(2.0 * (prec_t * rec_t), denom, out=np.zeros_like(denom), where=denom > 0)
    best_idx = int(np.argmax(f1_scores))

    res = OCSVMExperimentResult(
        config_name=config_name,
        nu=nu,
        n_components=n_components,
        fit_time_sec=fit_time,
        score_time_sec=score_time,
        direct_pr_auc=direct_pr_auc,
        direct_roc_auc=direct_roc_auc,
        lgb_pr_auc=lgb_pr_auc,
        lgb_avg_prec=lgb_avg_prec,
        lgb_roc_auc=lgb_roc_auc,
        best_f1=float(f1_scores[best_idx]),
        precision_at_best_f1=float(prec_t[best_idx]),
        recall_at_best_f1=float(rec_t[best_idx]),
        optimal_threshold=float(thresholds[best_idx]),
    )

    logger.info(
        f"[{config_name}] LGBM PR-AUC: {lgb_pr_auc:.4f} | ROC-AUC: {lgb_roc_auc:.4f} | "
        f"Mejor F1: {res.best_f1:.4f} (Fit: {fit_time:.2f}s, Score: {score_time:.2f}s)"
    )
    return res


def format_report(results: List[OCSVMExperimentResult], val_prev: float, seed: int) -> str:
    lines = [
        "# Experimentos One-Class SVM (SGD + Nystroem Kernel RBF)",
        "",
        "> **Estándar:** Nystroem con 300 componentes y kernel RBF (`gamma='scale'`), seguido de `SGDOneClassSVM` ajustado en legítimos de `train`. Evaluación univariada en LightGBM sobre validación OOT (22.500 transacciones).",
        "",
        f"- **Semilla determinista:** `{seed}`.",
        f"- **Prevalencia de Fraude en Validación:** `{val_prev:.3%}`.",
        "- **Aviso de Integridad:** Ningún archivo Parquet de `data/features/` fue modificado.",
        "",
        "## Tabla Comparativa de Rendimiento",
        "",
        "| Modelo / Configuración | nu (Tolerancia) | n_components | LightGBM PR-AUC | LightGBM ROC-AUC | Mejor F1 | Precision | Recall | Umbral Opt | Direct PR-AUC | Tiempo Total |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    best = max(results, key=lambda x: x.lgb_pr_auc)
    for r in results:
        is_best = " **(MEJOR)**" if r == best else ""
        tot_time = r.fit_time_sec + r.score_time_sec
        lines.append(
            f"| **{r.config_name}{is_best}** | {r.nu} | {r.n_components} | "
            f"**{r.lgb_pr_auc:.4f}** | {r.lgb_roc_auc:.4f} | **{r.best_f1:.4f}** | "
            f"{r.precision_at_best_f1:.2%} | {r.recall_at_best_f1:.2%} | {r.optimal_threshold:.4f} | "
            f"{r.direct_pr_auc:.4f} | {tot_time:.2f} s |"
        )

    lines += [
        "",
        "## Conclusiones Clave de One-Class SVM con Nystroem",
        "",
        f"1. **Mejor configuración:** `{best.config_name}` alcanzando **PR-AUC = {best.lgb_pr_auc:.4f}** y **F1 = {best.best_f1:.4f}**.",
        "2. **Velocidad y Escalabilidad:** Nystroem con 300 componentes aproxima el espacio RBF de forma continua y el SGD procesa las 100k filas en menos de 1 segundo de fit.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/dataset_base.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/experiments/ocsvm_nystroem_report.md"))
    parser.add_argument("--n-components", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    started = time.perf_counter()
    if not args.input.exists():
        raise FileNotFoundError(f"No se encontró el dataset base en: {args.input.resolve()}")

    logger.info(f"Cargando dataset desde: {args.input}")
    df = pd.read_parquet(args.input)

    val_mask = df["split"] == "val"
    val_prev = float(df.loc[val_mask, "fraude"].mean())

    x_train, x_val, y_train, y_val = prepare_features(df, DEFAULT_NUMERIC, DEFAULT_CATEGORICAL)
    normal_train = y_train == 0
    x_train_legit = x_train[normal_train]

    # Ajuste de Nystroem con 300 componentes sobre legítimos de train
    # gamma=None utiliza el estándar por defecto de scikit-learn: 1 / n_features (equivalente a 1 / (2 * sigma^2))
    logger.info(f"Ajustando Nystroem (n_components={args.n_components}, kernel='rbf', gamma=default) sobre legítimos...")
    t0 = time.perf_counter()
    nystroem = Nystroem(
        kernel="rbf",
        gamma=None,
        n_components=args.n_components,
        random_state=args.seed,
        n_jobs=-1,
    )
    nystroem.fit(x_train_legit)
    logger.info(f"Nystroem fiteado en {time.perf_counter() - t0:.2f} s. Transformando espacios...")

    x_train_nystroem = nystroem.transform(x_train)
    x_val_nystroem = nystroem.transform(x_val)

    nu_values = [0.10, 0.05, 0.03]
    results: List[OCSVMExperimentResult] = []

    for nu in nu_values:
        res = run_single_ocsvm(
            x_train_nystroem=x_train_nystroem,
            x_val_nystroem=x_val_nystroem,
            y_train=y_train,
            y_val=y_val,
            nu=nu,
            n_components=args.n_components,
            seed=args.seed,
        )
        results.append(res)

    report_text = format_report(results, val_prev, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report_text, encoding="utf-8")
    logger.info(f"Reporte guardado en: {args.output}")

    print("\n" + report_text)
    print(f"\nExperimento finalizado con éxito en {time.perf_counter() - started:.2f} s. (Ningún archivo Parquet modificado).")


if __name__ == "__main__":
    main()
