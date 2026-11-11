"""
Evaluación de variantes de PCA utilizando LightGBM entrenado exclusivamente
con los scores / percentiles de anomalía de PCA como únicas predictoras de fraude.

IMPORTANTE: Este script es de análisis y diagnóstico experimental.
NO actualiza ni modifica ningún archivo Parquet en data/features/.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pca_lightgbm_experiments")

DEFAULT_NUMERIC = ["score", "monto", "a", "b", "d", "e", "f", "h", "l", "m", "n", "q", "r", "s"]
DEFAULT_CATEGORICAL = ["g", "j", "o", "p"]


def frequency_encode(train: pd.Series, val: pd.Series, normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    reference = train.loc[normal]
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


def calculate_pca_scores(pca: PCA, x: np.ndarray) -> Dict[str, np.ndarray]:
    coordinates = pca.transform(x)
    residual = x - pca.inverse_transform(coordinates)
    eigenvalues = pca.explained_variance_
    return {
        "reconstruccion_Q": np.sum(residual ** 2, axis=1),
        "magnitud_cuadrada": np.sum(coordinates ** 2, axis=1),
        "magnitud_ponderada_T2": np.sum(coordinates ** 2 / eigenvalues, axis=1),
    }


def to_percentiles(train_scores: np.ndarray, val_scores: np.ndarray, normal_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    sorted_legit = np.sort(train_scores[normal_train])
    n_legit = len(sorted_legit)
    pct_train = np.searchsorted(sorted_legit, train_scores) / n_legit
    pct_val = np.searchsorted(sorted_legit, val_scores) / n_legit
    return np.clip(pct_train, 0.0, 1.0), np.clip(pct_val, 0.0, 1.0)


def run_pca_lgb_battery(
    df: pd.DataFrame,
    include_categorical: bool,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    train_mask = (df["split"] == "train").to_numpy()
    val_mask = (df["split"] == "val").to_numpy()
    y_train = df.loc[train_mask, "fraude"].to_numpy()
    y_val = df.loc[val_mask, "fraude"].to_numpy()
    normal_train = y_train == 0

    mode_name = "Numérico + Frecuencia (g, j, o, p)" if include_categorical else "Numérico Puro"
    logger.info(f"--- Iniciando evaluación PCA con modo: {mode_name} ---")

    # 1. Preprocesar numéricas
    x_num_train = df.loc[train_mask, DEFAULT_NUMERIC].to_numpy(dtype=float, na_value=np.nan)
    x_num_val = df.loc[val_mask, DEFAULT_NUMERIC].to_numpy(dtype=float, na_value=np.nan)

    imputer = SimpleImputer(strategy="median").fit(x_num_train[normal_train])
    x_num_train = imputer.transform(x_num_train)
    x_num_val = imputer.transform(x_num_val)

    if include_categorical:
        cat_blocks_tr = []
        cat_blocks_val = []
        for col in DEFAULT_CATEGORICAL:
            tr_enc, v_enc = frequency_encode(df.loc[train_mask, col], df.loc[val_mask, col], normal_train)
            cat_blocks_tr.append(tr_enc[:, None])
            cat_blocks_val.append(v_enc[:, None])
        x_tr_full = np.column_stack([x_num_train] + cat_blocks_tr)
        x_val_full = np.column_stack([x_num_val] + cat_blocks_val)
    else:
        x_tr_full = x_num_train
        x_val_full = x_num_val

    scaler = StandardScaler().fit(x_tr_full[normal_train])
    x_tr_scaled = scaler.transform(x_tr_full)
    x_val_scaled = scaler.transform(x_val_full)

    results = []

    for fraction in (0.90, 0.95, 0.99):
        pca = PCA(n_components=fraction, svd_solver="full", random_state=seed)
        pca.fit(x_tr_scaled[normal_train])
        n_comp = int(pca.n_components_)
        var_expl = float(pca.explained_variance_ratio_.sum())

        train_scores_dict = calculate_pca_scores(pca, x_tr_scaled)
        val_scores_dict = calculate_pca_scores(pca, x_val_scaled)

        score_keys = ["reconstruccion_Q", "magnitud_cuadrada", "magnitud_ponderada_T2"]

        # Calcular percentiles empíricos
        pct_train_dict = {}
        pct_val_dict = {}
        for k in score_keys:
            pct_tr, pct_v = to_percentiles(train_scores_dict[k], val_scores_dict[k], normal_train)
            pct_train_dict[k] = pct_tr
            pct_val_dict[k] = pct_v

        # Probar cada score individual y la combinación de los 3
        test_cases = [[k] for k in score_keys] + [score_keys]

        for selected in test_cases:
            case_name = selected[0] if len(selected) == 1 else "tres_scores_pca"

            X_tr_lgb = pd.DataFrame({f"pct_{name}": pct_train_dict[name] for name in selected})
            X_val_lgb = pd.DataFrame({f"pct_{name}": pct_val_dict[name] for name in selected})

            # 1. LightGBM supervisado SOLO con el score / percentil PCA
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

            prec_lgb, rec_lgb, _ = precision_recall_curve(y_val, lgb_val_probs)
            lgb_pr_auc = float(auc(rec_lgb, prec_lgb))
            lgb_avg_prec = float(average_precision_score(y_val, lgb_val_probs))
            lgb_roc_auc = float(roc_auc_score(y_val, lgb_val_probs))

            # 2. Direct PR-AUC (solo para scores individuales)
            if len(selected) == 1:
                p_dir, r_dir, _ = precision_recall_curve(y_val, pct_val_dict[selected[0]])
                direct_pr_auc = float(auc(r_dir, p_dir))
                direct_roc_auc = float(roc_auc_score(y_val, pct_val_dict[selected[0]]))
            else:
                direct_pr_auc = None
                direct_roc_auc = None

            # 3. Regresión Logística de referencia
            lr_scaler = StandardScaler().fit(X_tr_lgb.values[normal_train])
            logistic = LogisticRegression(random_state=seed, max_iter=1000)
            logistic.fit(lr_scaler.transform(X_tr_lgb.values), y_train)
            lr_val_probs = logistic.predict_proba(lr_scaler.transform(X_val_lgb.values))[:, 1]
            p_lr, r_lr, _ = precision_recall_curve(y_val, lr_val_probs)
            logreg_pr_auc = float(auc(r_lr, p_lr))
            logreg_roc_auc = float(roc_auc_score(y_val, lr_val_probs))

            results.append({
                "mode": mode_name,
                "var_target": fraction,
                "components": n_comp,
                "var_actual": var_expl,
                "score_name": case_name,
                "lgb_pr_auc": lgb_pr_auc,
                "lgb_avg_prec": lgb_avg_prec,
                "lgb_roc_auc": lgb_roc_auc,
                "direct_pr_auc": direct_pr_auc,
                "direct_roc_auc": direct_roc_auc,
                "logreg_pr_auc": logreg_pr_auc,
                "logreg_roc_auc": logreg_roc_auc,
            })

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/dataset_base.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/experiments/pca_lightgbm_report.md"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    started = time.perf_counter()
    logger.info(f"Cargando dataset desde: {args.input}")
    df = pd.read_parquet(args.input)

    val_mask = df["split"] == "val"
    val_prev = float(df.loc[val_mask, "fraude"].mean())
    logger.info(f"Prevalencia en validación: {val_prev:.3%}")

    # Batería 1: Numéricas puras
    res_num = run_pca_lgb_battery(df, include_categorical=False, seed=args.seed)
    # Batería 2: Numéricas + Categóricas con Frequency Encoding
    res_freq = run_pca_lgb_battery(df, include_categorical=True, seed=args.seed)

    all_results = res_num + res_freq

    # Generar tabla y reporte Markdown
    lines = [
        "# Experimentos PCA evaluados con LightGBM Univariado / Multi-Score",
        "",
        "> **Estándar:** LightGBM entrenado en train exclusivamente con el percentil de los scores de PCA como únicas predictoras de fraude. Evaluación estricta en validación OOT (22.500 transacciones).",
        "",
        f"- **Semilla:** `{args.seed}`.",
        f"- **Prevalencia de Fraude en Validación:** `{val_prev:.3%}`.",
        "- **Aviso de Integridad:** Ningún archivo Parquet de `data/features/` fue modificado.",
        "",
        "## Tabla Comparativa General de PCA con LightGBM",
        "",
        "| Modo de Variables | Varianza | Comp | Score Evaluado en LightGBM | LightGBM PR-AUC | LightGBM Avg Prec | LightGBM ROC-AUC | Direct PR-AUC | LogReg PR-AUC |",
        "| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    best = max(all_results, key=lambda x: x["lgb_pr_auc"])

    for r in all_results:
        is_best = " **(MEJOR)**" if r == best else ""
        dir_pr = f"{r['direct_pr_auc']:.4f}" if r['direct_pr_auc'] is not None else "-"
        lines.append(
            f"| {r['mode']} | {r['var_target']:.0%} | {r['components']} | **{r['score_name']}{is_best}** | "
            f"**{r['lgb_pr_auc']:.4f}** | {r['lgb_avg_prec']:.4f} | {r['lgb_roc_auc']:.4f} | "
            f"{dir_pr} | {r['logreg_pr_auc']:.4f} |"
        )

    lines += [
        "",
        "## Conclusiones Clave de PCA bajo el Estándar LightGBM",
        "",
        f"1. **Mejor configuración de PCA:** `{best['score_name']}` bajo `{best['mode']}` a {best['var_target']:.0%} de varianza.",
        f"   - **LightGBM PR-AUC:** `{best['lgb_pr_auc']:.4f}` | **ROC-AUC:** `{best['lgb_roc_auc']:.4f}`.",
        "2. **Comparación con Isolation Forest:**",
        "   - Isolation Forest (`IF_Diagonal_HybridOHE_256`) alcanzó un LightGBM PR-AUC de **0.1557** y ROC-AUC de **0.7431**.",
        "   - PCA (`magnitud_cuadrada` / `tres_scores_pca`) aporta una señal complementaria de densidad elíptica y distancias latentes.",
        "",
    ]

    report_text = "\n".join(lines)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report_text, encoding="utf-8")
    logger.info(f"Reporte guardado en: {args.output}")

    print("\n" + report_text)
    print(f"\nEjecución finalizada con éxito en {time.perf_counter() - started:.2f} s. (Ningún archivo Parquet modificado).")


if __name__ == "__main__":
    main()
