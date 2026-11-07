"""
Diagnóstico de Scoring y Curvas de Desempeño (K-S, Precision-Recall y Bimodalidad).

Calcula las métricas de negocio estándar de la industria bancaria y de riesgo (Kolmogorov-Smirnov D_ks,
PR-AUC, ROC-AUC) contrastando el score propietario crudo de pasarela contra el modelo óptimo
de 11 variables seleccionadas, evaluado estrictamente en validación temporal Out-Of-Time (OOT).

Exporta gráficos con paleta corporativa de Mercado Libre a reports/figures/.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("scoring_diagnostics")

# Paleta corporativa de Mercado Libre
COL_AMARILLO_MELI = "#FFE600"
COL_AMARILLO_TEXTO = "#9A8400"
COL_AZUL_OSCURO = "#030C7D"
COL_AZUL_CLARO = "#3483FA"
COL_NARANJA = "#D95F02"
COL_FONDO = "#F9F9F9"


def compute_ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Calcula el estadístico K-S y el umbral de máxima separación."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    diff = np.abs(tpr - fpr)
    max_idx = np.argmax(diff)
    return float(diff[max_idx]), float(thresholds[max_idx])


def run_diagnostics(
    base_path: Path,
    features_path: Path,
    output_dir: Path,
    save_plots: bool = True,
) -> None:
    logger.info("Cargando datasets para diagnóstico de scoring...")
    df_base = pd.read_parquet(base_path)
    df_feat = pd.read_parquet(features_path)

    # Identificar las variables del modelo óptimo
    exclude_cols = {"transaction_id", "split", "fraude", "fecha"}
    model_cols = [c for c in df_feat.columns if c not in exclude_cols]
    logger.info(f"Modelo cargado con {len(model_cols)} variables: {model_cols}")

    # Merge limpio sin columnas duplicadas (_x, _y)
    cols_to_add = ["transaction_id"]
    for c in ["split", "fraude", "score"]:
        if c not in df_feat.columns:
            cols_to_add.append(c)

    full_df = df_feat.merge(df_base[cols_to_add], on="transaction_id")

    for col in ["a", "g", "n", "o", "p", "is_na_bc"]:
        if col in full_df.columns:
            full_df[col] = full_df[col].astype("category")

    train_mask = full_df["split"] == "train"
    val_mask = full_df["split"] == "val"

    X_train = full_df.loc[train_mask, model_cols]
    y_train = full_df.loc[train_mask, "fraude"].to_numpy()

    X_val = full_df.loc[val_mask, model_cols]
    y_val = full_df.loc[val_mask, "fraude"].to_numpy()
    score_val_raw = full_df.loc[val_mask, "score"].to_numpy()

    logger.info(f"Entrenando LightGBM (120 árboles) en train ({len(y_train):,} filas)...")
    clf = lgb.LGBMClassifier(
        n_estimators=120,
        learning_rate=0.06,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=1.0,
        random_state=42,
        verbose=-1,
    )
    clf.fit(X_train, y_train)

    p_val_model = clf.predict_proba(X_val)[:, 1]

    # 1. Métricas de Ranking y Discriminación
    roc_raw = roc_auc_score(y_val, score_val_raw)
    p_raw, r_raw, _ = precision_recall_curve(y_val, score_val_raw)
    pr_raw = float(auc(r_raw, p_raw))
    ks_raw, ks_thresh_raw = compute_ks_statistic(y_val, score_val_raw)

    roc_model = roc_auc_score(y_val, p_val_model)
    p_model, r_model, _ = precision_recall_curve(y_val, p_val_model)
    pr_model = float(auc(r_model, p_model))
    ks_model, ks_thresh_model = compute_ks_statistic(y_val, p_val_model)

    prevalencia_val = float(np.mean(y_val))

    print("\n" + "=" * 80)
    print("REPORTE EJECUTIVO: DIAGNÓSTICO DE SCORING EN VALIDACIÓN OUT-OF-TIME (OOT)")
    print("=" * 80)
    print(f"Población evaluada: {len(y_val):,} transacciones | Prevalencia Basal de Fraude: {prevalencia_val*100:.2f}%")
    print("-" * 80)
    print(f"{'Métrica':<25s} | {'Score Crudo (Pasarela)':<24s} | {'Modelo 11 Variables':<24s} | {'Mejora (Delta)':<15s}")
    print("-" * 80)
    print(f"{'PR-AUC':<25s} | {pr_raw:24.4f} | {pr_model:24.4f} | {pr_model - pr_raw:+15.4f}")
    print(f"{'ROC-AUC':<25s} | {roc_raw:24.4f} | {roc_model:24.4f} | {roc_model - roc_raw:+15.4f}")
    print(f"{'K-S Stat (Separación)':<25s} | {ks_raw*100:23.2f}% | {ks_model*100:23.2f}% | {(ks_model - ks_raw)*100:+14.2f}%")
    print(f"{'Umbral Óptimo K-S':<25s} | {f'Score = {ks_thresh_raw:.1f}':<24s} | {f'p = {ks_thresh_model:.4f}':<24s} | {'—':<15s}")
    print("=" * 80)

    # 2. Análisis Bimodal del Fraude en Validación
    print("\n--- ANÁLISIS BIMODAL DEL FRAUDE (COMPORTAMIENTO DEL SCORE) ---")
    bins = [0, 30, 60, 100]
    labels = ["Bajo Score (< 30) - Fraude Sigiloso", "Valle Medio [30, 60] - Zona Segura (Whitelist)", "Alto Score (> 60) - Fraude Evidente"]
    val_score_series = pd.Series(score_val_raw)
    cat_bins = pd.cut(val_score_series, bins=bins, labels=labels, include_lowest=True)

    df_bimodal = pd.DataFrame({"rango": cat_bins, "fraude": y_val})
    resumen_bimodal = df_bimodal.groupby("rango", observed=True).agg(
        total=("fraude", "count"),
        fraudes=("fraude", "sum"),
        tasa_fraude=("fraude", "mean"),
    ).reset_index()

    for _, row in resumen_bimodal.iterrows():
        pct_pob = row["total"] / len(y_val) * 100
        pct_fr = row["fraudes"] / np.sum(y_val) * 100
        print(f"• {row['rango']:<45s} -> Fraudes: {row['fraudes']:4d} ({pct_fr:5.1f}% del total) | Tasa Fraude: {row['tasa_fraude']*100:5.2f}% (Población: {pct_pob:5.1f}%)")

    # 3. Generación y guardado de gráficos corporativos
    if save_plots:
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

        # Gráfico 1: Curva K-S Comparativa
        fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
        fig.patch.set_facecolor(COL_FONDO)
        ax.set_facecolor("white")

        eval_grid = np.linspace(0, 1, 300)
        fpr_m, tpr_m, _ = roc_curve(y_val, p_val_model)
        ax.plot(fpr_m, tpr_m, color=COL_AZUL_OSCURO, lw=2.2, label=f"Modelo 11 Vars (ROC = {roc_model:.3f})")
        
        fpr_r, tpr_r, _ = roc_curve(y_val, score_val_raw / 100.0)
        ax.plot(fpr_r, tpr_r, color="gray", lw=1.8, linestyle="--", label=f"Score Crudo (ROC = {roc_raw:.3f})")
        ax.plot([0, 1], [0, 1], color="red", linestyle=":", lw=1.2, label="Línea de Azar (0.50)")

        ax.set_title("Curva ROC Comparativa — Modelo Óptimo vs. Score Crudo", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("1 - Especificidad (Tasa de Falsos Positivos)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Sensibilidad (Tasa de Verdaderos Positivos)", fontsize=10, fontweight="bold")
        ax.legend(loc="lower right", frameon=True)
        fig.tight_layout()
        ks_path = output_dir / "roc_curve_comparison.png"
        fig.savefig(ks_path)
        plt.close(fig)
        logger.info(f"Gráfico ROC guardado en: {ks_path}")

        # Gráfico 2: Curva Precision-Recall vs Azar
        fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
        fig.patch.set_facecolor(COL_FONDO)
        ax.set_facecolor("white")

        ax.plot(r_model, p_model, color=COL_AZUL_OSCURO, lw=2.2, label=f"Modelo 11 Vars (PR-AUC = {pr_model:.4f})")
        ax.plot(r_raw, p_raw, color="gray", lw=1.8, linestyle="--", label=f"Score Crudo (PR-AUC = {pr_raw:.4f})")
        ax.axhline(prevalencia_val, color="red", linestyle=":", lw=1.4, label=f"Azar / Prevalencia Basal ({prevalencia_val*100:.1f}%)")

        ax.set_title("Curva Precision-Recall — Elevación sobre el Azar", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Recall (Cobertura de Fraudes)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Precisión (Aciertos Reales)", fontsize=10, fontweight="bold")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(-0.02, 1.02)
        ax.legend(loc="upper right", frameon=True)
        fig.tight_layout()
        pr_path = output_dir / "pr_curve_comparison.png"
        fig.savefig(pr_path)
        plt.close(fig)
        logger.info(f"Gráfico PR-AUC guardado en: {pr_path}")

        # Gráfico 3: Bimodalidad del Fraude
        fig, ax = plt.subplots(figsize=(8, 5.0), dpi=150)
        fig.patch.set_facecolor(COL_FONDO)
        ax.set_facecolor("white")

        ax.hist(
            score_val_raw[y_val == 0],
            bins=30,
            density=True,
            alpha=0.5,
            color=COL_AZUL_CLARO,
            label="Legítimas (Fraude = 0)",
        )
        ax.hist(
            score_val_raw[y_val == 1],
            bins=30,
            density=True,
            alpha=0.75,
            color=COL_NARANJA,
            label="Fraudes (Fraude = 1)",
        )

        ax.set_title("Distribución del Score: Detección de la Moda Oculta de Fraude", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Score de Riesgo de Pasarela (0 - 100)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Densidad Empírica", fontsize=10, fontweight="bold")
        ax.legend(loc="upper center", frameon=True)
        fig.tight_layout()
        bimodal_path = output_dir / "bimodal_fraud_distribution.png"
        fig.savefig(bimodal_path)
        plt.close(fig)
        logger.info(f"Gráfico de Bimodalidad guardado en: {bimodal_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluación diagnóstica de scoring y curvas K-S / PR.")
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("data/processed/dataset_base.parquet"),
        help="Ruta a dataset_base.parquet",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/features/features_seleccionadas_end_to_end.parquet"),
        help="Ruta a features_seleccionadas_end_to_end.parquet",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("reports/figures"),
        help="Directorio destino para guardar los gráficos generados.",
    )
    args = parser.parse_args()

    run_diagnostics(
        base_path=args.base,
        features_path=args.features,
        output_dir=args.output_dir,
        save_plots=True,
    )


if __name__ == "__main__":
    main()
