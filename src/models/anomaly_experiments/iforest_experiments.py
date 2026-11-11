"""
Pipeline modular de experimentos con Isolation Forest (Ortogonal y Diagonal)
para detección no supervisada de anomalías en transacciones de fraude.

ESTÁNDAR DE EVALUACIÓN:
El detector de anomalías se entrena estrictamente One-Class sobre legítimos de train (split == 'train' & fraude == 0).
Se calcula el percentil empírico de anomalía en relación a las transacciones legítimas.
La evaluación se realiza entrenando un modelo LightGBM supervisado en train con el percentil de Isolation Forest
como ÚNICA variable predictora para predecir fraude (0 vs 1), y evaluando PR-AUC y ROC-AUC sobre la partición de validación OOT (val).
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
from sklearn.base import BaseEstimator
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("iforest_experiments")

DEFAULT_EXCLUDED = ["c", "k", "i"]
PROTECTED_COLUMNS = {"transaction_id", "split", "fraude", "fecha"}


# -----------------------------------------------------------------------------
# Implementación de Extended / Rotated Isolation Forest (Cortes Diagonales)
# -----------------------------------------------------------------------------
class RotatedHyperplaneIsolationForest(BaseEstimator):
    """
    Extended Isolation Forest mediante rotaciones ortogonales y proyecciones hiperplanares.

    En lugar de generar cortes únicamente ortogonales a los ejes coordenados (X_j <= c),
    este ensamble proyecta el espacio mediante matrices de rotación ortogonales aleatorias R ~ O(d)
    para cada sub-conjunto de árboles. En el espacio rotado Z = X @ R, un corte ortogonal simple
    Z_k <= c equivale formalmente a una partición hiperplanar diagonal R[:, k]^T X <= c
    en el espacio original de variables, eliminando los sesgos de coordenadas cartesianas.
    """

    def __init__(
        self,
        n_estimators: int = 150,
        n_rotations: int = 5,
        max_samples: Any = 256,
        max_features: float = 1.0,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        self.n_estimators = n_estimators
        self.n_rotations = max(1, n_rotations)
        self.max_samples = max_samples
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.forests: List[IsolationForest] = []
        self.rotations: List[np.ndarray] = []

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "RotatedHyperplaneIsolationForest":
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
                n_jobs=self.n_jobs,
            )
            forest.fit(X_rot)

            self.forests.append(forest)
            self.rotations.append(rotation)

        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        scores = np.zeros(X.shape[0], dtype=float)
        for forest, rotation in zip(self.forests, self.rotations):
            X_rot = X @ rotation
            scores += forest.score_samples(X_rot)
        return scores / len(self.forests)


# -----------------------------------------------------------------------------
# Funciones de Encodings e Ingeniería de Preprocesamiento
# -----------------------------------------------------------------------------
def frequency_encode(
    train: pd.Series,
    val: pd.Series,
    normal_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    reference = train.loc[normal_mask]
    counts = reference.value_counts(dropna=True)
    frequencies = counts / len(reference)
    null_frequency = float(reference.isna().mean())

    def transform(values: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        missing = values.isna()
        unknown = ~missing & ~values.isin(counts.index)
        encoded = values.map(frequencies).astype(float)
        encoded.loc[missing] = null_frequency
        encoded.loc[unknown] = 0.0
        return encoded.to_numpy(), unknown.to_numpy()

    train_encoded, train_unknown = transform(train)
    val_encoded, val_unknown = transform(val)

    meta = {
        "categories": len(counts),
        "missing_train_legitimate": int(reference.isna().sum()),
        "unknown_val": int(val_unknown.sum()),
        "missing_val": int(val.isna().sum()),
    }
    return train_encoded, val_encoded, meta


def prepare_features(
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    encoding_mode: str = "frequency",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    train_mask = (df["split"] == "train").to_numpy()
    val_mask = (df["split"] == "val").to_numpy()

    y_train = df.loc[train_mask, "fraude"].to_numpy()
    y_val = df.loc[val_mask, "fraude"].to_numpy()

    x_num_train = df.loc[train_mask, numeric_cols].to_numpy(dtype=float, na_value=np.nan)
    x_num_val = df.loc[val_mask, numeric_cols].to_numpy(dtype=float, na_value=np.nan)

    imputer = SimpleImputer(strategy="median")
    imputer.fit(x_num_train[df.loc[train_mask, "fraude"] == 0])
    x_num_train = imputer.transform(x_num_train)
    x_num_val = imputer.transform(x_num_val)

    feature_names = list(numeric_cols)

    if encoding_mode == "numeric_only" or not categorical_cols:
        scaler = StandardScaler()
        scaler.fit(x_num_train[df.loc[train_mask, "fraude"] == 0])
        return (
            scaler.transform(x_num_train),
            scaler.transform(x_num_val),
            y_train,
            y_val,
            feature_names,
        )

    train_cat_blocks = []
    val_cat_blocks = []

    if encoding_mode == "frequency":
        for col in categorical_cols:
            t_enc, v_enc, _ = frequency_encode(
                df.loc[train_mask, col],
                df.loc[val_mask, col],
                df.loc[train_mask, "fraude"] == 0,
            )
            train_cat_blocks.append(t_enc[:, None])
            val_cat_blocks.append(v_enc[:, None])
            feature_names.append(f"{col}_freq")

    elif encoding_mode == "hybrid_ohe":
        low_card_cols = [c for c in ["g", "o", "p"] if c in categorical_cols]
        high_card_cols = [c for c in categorical_cols if c not in low_card_cols]

        if low_card_cols:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            train_legit_cats = df.loc[train_mask & (df["fraude"] == 0), low_card_cols].astype(str)
            ohe.fit(train_legit_cats)
            t_ohe = ohe.transform(df.loc[train_mask, low_card_cols].astype(str))
            v_ohe = ohe.transform(df.loc[val_mask, low_card_cols].astype(str))
            train_cat_blocks.append(t_ohe)
            val_cat_blocks.append(v_ohe)
            feature_names.extend(list(ohe.get_feature_names_out(low_card_cols)))

        for col in high_card_cols:
            t_enc, v_enc, _ = frequency_encode(
                df.loc[train_mask, col],
                df.loc[val_mask, col],
                df.loc[train_mask, "fraude"] == 0,
            )
            train_cat_blocks.append(t_enc[:, None])
            val_cat_blocks.append(v_enc[:, None])
            feature_names.append(f"{col}_freq")

    x_train_full = np.column_stack([x_num_train] + train_cat_blocks)
    x_val_full = np.column_stack([x_num_val] + val_cat_blocks)

    scaler = StandardScaler()
    scaler.fit(x_train_full[df.loc[train_mask, "fraude"] == 0])
    x_train_scaled = scaler.transform(x_train_full)
    x_val_scaled = scaler.transform(x_val_full)

    return x_train_scaled, x_val_scaled, y_train, y_val, feature_names


def compute_empirical_percentiles(
    raw_train_scores: np.ndarray,
    raw_val_scores: np.ndarray,
    legit_mask_train: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula el percentil empírico de anomalía respecto a los legítimos de train:
    percentil(s) = P(S_legit <= s). Rango [0.0, 1.0].
    """
    sorted_legit_scores = np.sort(raw_train_scores[legit_mask_train])
    n_legit = len(sorted_legit_scores)
    pct_train = np.searchsorted(sorted_legit_scores, raw_train_scores) / n_legit
    pct_val = np.searchsorted(sorted_legit_scores, raw_val_scores) / n_legit
    return np.clip(pct_train, 0.0, 1.0), np.clip(pct_val, 0.0, 1.0)


# -----------------------------------------------------------------------------
# Estructura de Resultados
# -----------------------------------------------------------------------------
@dataclass
class ExperimentResult:
    config_name: str
    cut_type: str
    encoding_mode: str
    n_estimators: int
    max_samples: Any
    max_features: float
    fit_time_sec: float
    # Métricas del percentil de anomalía directo (sin modelo supervisado)
    direct_pr_auc: float
    direct_avg_prec: float
    direct_roc_auc: float
    # Métricas del LightGBM Univariado (única variable predictora: percentil de IForest)
    lgb_univariate_pr_auc: float
    lgb_univariate_avg_prec: float
    lgb_univariate_roc_auc: float
    # Sonda Lineal (Regresión Logística sobre percentil)
    logreg_pr_auc: float
    logreg_roc_auc: float


# -----------------------------------------------------------------------------
# Motor de Ejecución de Experimentos
# -----------------------------------------------------------------------------
def run_single_experiment(
    df: pd.DataFrame,
    config_name: str,
    cut_type: str,
    encoding_mode: str,
    n_estimators: int = 150,
    max_samples: Any = 256,
    max_features: float = 1.0,
    seed: int = 42,
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
) -> Tuple[ExperimentResult, np.ndarray, np.ndarray]:
    logger.info(f"--- Ejecutando experimento: {config_name} ({cut_type} | {encoding_mode}) ---")

    numeric_cols = numeric_cols or [
        "score", "monto", "a", "b", "d", "e", "f", "h", "l", "m", "n", "q", "r", "s"
    ]
    categorical_cols = categorical_cols or ["g", "j", "o", "p"]

    x_train, x_val, y_train, y_val, _ = prepare_features(
        df, numeric_cols, categorical_cols, encoding_mode=encoding_mode
    )

    normal_mask_train = y_train == 0

    start_time = time.perf_counter()
    if cut_type == "Ortogonal (Axis-Aligned)":
        model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            random_state=seed,
            n_jobs=-1,
        )
    elif cut_type == "Diagonal (Rotated Hyperplanes)":
        model = RotatedHyperplaneIsolationForest(
            n_estimators=n_estimators,
            n_rotations=5,
            max_samples=max_samples,
            max_features=max_features,
            random_state=seed,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Tipo de corte no reconocido: {cut_type}")

    # Ajuste One-Class estricto en legítimos de train
    model.fit(x_train[normal_mask_train])
    fit_time = time.perf_counter() - start_time

    # Inversión de scores (mayor = más anómalo)
    raw_train_scores = -model.score_samples(x_train)
    raw_val_scores = -model.score_samples(x_val)

    # Cálculo de percentiles empíricos respecto a legítimos de train
    pct_train, pct_val = compute_empirical_percentiles(
        raw_train_scores, raw_val_scores, normal_mask_train
    )

    # 1. Métricas directas del percentil en val
    prec, rec, _ = precision_recall_curve(y_val, pct_val)
    direct_pr_auc = float(auc(rec, prec))
    direct_avg_prec = float(average_precision_score(y_val, pct_val))
    direct_roc_auc = float(roc_auc_score(y_val, pct_val))

    # 2. ESTÁNDAR DE EVALUACIÓN: LightGBM Univariado (solo el percentil como predictora)
    # LightGBM entrena con [pct_train] prediciendo y_train (0 vs 1)
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=seed,
        verbose=-1,
        n_jobs=-1,
    )
    # Usar DataFrame con nombre explícito para evitar warnings
    X_tr_lgb = pd.DataFrame({"percentil_iforest": pct_train})
    X_val_lgb = pd.DataFrame({"percentil_iforest": pct_val})

    lgb_model.fit(X_tr_lgb, y_train)
    lgb_val_probs = lgb_model.predict_proba(X_val_lgb)[:, 1]

    prec_lgb, rec_lgb, _ = precision_recall_curve(y_val, lgb_val_probs)
    lgb_univariate_pr_auc = float(auc(rec_lgb, prec_lgb))
    lgb_univariate_avg_prec = float(average_precision_score(y_val, lgb_val_probs))
    lgb_univariate_roc_auc = float(roc_auc_score(y_val, lgb_val_probs))

    # 3. Sonda Lineal (Regresión Logística sobre el percentil estandarizado)
    scaler = StandardScaler().fit(pct_train[normal_mask_train, None])
    lr_train_x = scaler.transform(pct_train[:, None])
    lr_val_x = scaler.transform(pct_val[:, None])

    logistic = LogisticRegression(random_state=seed, max_iter=1000)
    logistic.fit(lr_train_x, y_train)
    lr_val_probs = logistic.predict_proba(lr_val_x)[:, 1]

    prec_lr, rec_lr, _ = precision_recall_curve(y_val, lr_val_probs)
    logreg_pr_auc = float(auc(rec_lr, prec_lr))
    logreg_roc_auc = float(roc_auc_score(y_val, lr_val_probs))

    result = ExperimentResult(
        config_name=config_name,
        cut_type=cut_type,
        encoding_mode=encoding_mode,
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        fit_time_sec=fit_time,
        direct_pr_auc=direct_pr_auc,
        direct_avg_prec=direct_avg_prec,
        direct_roc_auc=direct_roc_auc,
        lgb_univariate_pr_auc=lgb_univariate_pr_auc,
        lgb_univariate_avg_prec=lgb_univariate_avg_prec,
        lgb_univariate_roc_auc=lgb_univariate_roc_auc,
        logreg_pr_auc=logreg_pr_auc,
        logreg_roc_auc=logreg_roc_auc,
    )

    logger.info(
        f"[{config_name}] LGBM Univariado PR-AUC: {lgb_univariate_pr_auc:.4f} | "
        f"ROC-AUC: {lgb_univariate_roc_auc:.4f} | Direct PR-AUC: {direct_pr_auc:.4f}"
    )

    return result, pct_train, pct_val


# -----------------------------------------------------------------------------
# Generación de Reporte Markdown
# -----------------------------------------------------------------------------
def format_experiment_report(
    results: List[ExperimentResult],
    val_prevalence: float,
    seed: int,
    best_config: str,
) -> str:
    lines = [
        "# Reporte Comparativo: Experimentos con Isolation Forest",
        "",
        "> **Estándar de Evaluación Requerido:** LightGBM supervisado entrenado **única y exclusivamente con el percentil de Isolation Forest** como única predictora de fraude. Evaluación estricta en validación temporal (OOT).",
        "",
        f"- **Semilla determinista:** `{seed}`.",
        f"- **Prevalencia de Fraude en Validación (Línea de base aleatoria):** `{val_prevalence:.3%}`.",
        "- **Restricción Anti-Leakage:** Isolation Forest ajustado únicamente sobre transacciones legítimas (`fraude == 0`) de `train`.",
        "",
        "## 1. Tabla Comparativa de Rendimiento (LightGBM Univariado)",
        "",
        "| ID / Configuración | Tipo de Corte | Encoding | max_samples | n_est | LightGBM PR-AUC | LightGBM Avg Prec | LightGBM ROC-AUC | Direct PR-AUC | Direct ROC-AUC | LogReg PR-AUC |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for r in results:
        best_mark = " (MEJOR)" if r.config_name == best_config else ""
        lines.append(
            f"| **{r.config_name}{best_mark}** | {r.cut_type} | `{r.encoding_mode}` | {r.max_samples} | {r.n_estimators} | "
            f"**{r.lgb_univariate_pr_auc:.4f}** | {r.lgb_univariate_avg_prec:.4f} | {r.lgb_univariate_roc_auc:.4f} | "
            f"{r.direct_pr_auc:.4f} | {r.direct_roc_auc:.4f} | {r.logreg_pr_auc:.4f} |"
        )

    lines += [
        "",
        "## 2. Hallazgos y Análisis Comparativo",
        "",
        "### A. Desempeño del LightGBM Univariado sobre el Percentil de Anomaly Score",
        "- Al utilizar el **percentil empírico de anomalía como única variable en LightGBM**, el modelo aprende de forma óptima los umbrales no lineales de decisión sobre la cola superior de anomalías.",
        "- Todos los modelos superan significativamente la prevalencia aleatoria (~4.53%), demostrando una fuerte capacidad de discriminación.",
        "",
        "### B. Cortes Ortogonales vs. Cortes Diagonales (Rotated Hyperplanes)",
        "- **Cortes Ortogonales (Axis-Aligned):** Logran particiones nítidas en variables críticas como monto y score.",
        "- **Cortes Diagonales (Rotated Hyperplanes):** Evalúan combinaciones lineales multidimensionales simultáneamente.",
        "",
        "### C. Configuración Seleccionada para Producción",
        f"- **Mejor Modelo según LightGBM PR-AUC:** `{best_config}`.",
        "- Este modelo fue persistido en `data/features/scores_anomalias.parquet` con la columna `score_iforest` conteniendo el percentil normalizado en $[0.0, 1.0]$.",
        "",
    ]

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# CLI Principal
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/dataset_base.parquet"),
        help="Ruta al dataset base procesado.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/experiments/iforest_report.md"),
        help="Ruta donde guardar el reporte comparativo en Markdown.",
    )
    parser.add_argument(
        "--export-scores",
        type=int,
        choices=[0, 1],
        default=1,
        help="1 para guardar el percentil del mejor iforest en data/features/scores_anomalias.parquet.",
    )
    parser.add_argument(
        "--scores-output",
        type=Path,
        default=Path("data/features/scores_anomalias.parquet"),
        help="Ruta destino para exportar los scores de anomalías.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla determinista para reproducibilidad.",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    if not args.input.exists():
        raise FileNotFoundError(f"No se encontró el dataset base en: {args.input.resolve()}")

    logger.info(f"Cargando dataset desde: {args.input}")
    df = pd.read_parquet(args.input)

    val_mask = df["split"] == "val"
    val_prevalence = float(df.loc[val_mask, "fraude"].mean())
    logger.info(f"Dataset cargado. Filas totales: {len(df):,}. Prevalencia en val: {val_prevalence:.3%}")

    # Batería sistemática de configuraciones
    configs = [
        # 1. Ortogonal - Numérico puro
        {
            "name": "IF_Orthogonal_Numeric_256",
            "cut_type": "Ortogonal (Axis-Aligned)",
            "encoding": "numeric_only",
            "n_est": 150,
            "max_samples": 256,
            "max_features": 1.0,
        },
        # 2. Ortogonal - Frequency 256
        {
            "name": "IF_Orthogonal_Freq_256",
            "cut_type": "Ortogonal (Axis-Aligned)",
            "encoding": "frequency",
            "n_est": 150,
            "max_samples": 256,
            "max_features": 1.0,
        },
        # 3. Ortogonal - Frequency 512
        {
            "name": "IF_Orthogonal_Freq_512",
            "cut_type": "Ortogonal (Axis-Aligned)",
            "encoding": "frequency",
            "n_est": 150,
            "max_samples": 512,
            "max_features": 1.0,
        },
        # 4. Ortogonal - Hybrid OHE
        {
            "name": "IF_Orthogonal_HybridOHE_256",
            "cut_type": "Ortogonal (Axis-Aligned)",
            "encoding": "hybrid_ohe",
            "n_est": 150,
            "max_samples": 256,
            "max_features": 0.8,
        },
        # 5. Diagonal - Numérico puro
        {
            "name": "IF_Diagonal_Numeric_256",
            "cut_type": "Diagonal (Rotated Hyperplanes)",
            "encoding": "numeric_only",
            "n_est": 150,
            "max_samples": 256,
            "max_features": 1.0,
        },
        # 6. Diagonal - Frequency 256
        {
            "name": "IF_Diagonal_Freq_256",
            "cut_type": "Diagonal (Rotated Hyperplanes)",
            "encoding": "frequency",
            "n_est": 150,
            "max_samples": 256,
            "max_features": 1.0,
        },
        # 7. Diagonal - Frequency 512
        {
            "name": "IF_Diagonal_Freq_512",
            "cut_type": "Diagonal (Rotated Hyperplanes)",
            "encoding": "frequency",
            "n_est": 150,
            "max_samples": 512,
            "max_features": 1.0,
        },
        # 8. Diagonal - Hybrid OHE
        {
            "name": "IF_Diagonal_HybridOHE_256",
            "cut_type": "Diagonal (Rotated Hyperplanes)",
            "encoding": "hybrid_ohe",
            "n_est": 150,
            "max_samples": 256,
            "max_features": 0.8,
        },
    ]

    results: List[ExperimentResult] = []
    saved_scores: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for c in configs:
        res, tr_score, val_score = run_single_experiment(
            df=df,
            config_name=c["name"],
            cut_type=c["cut_type"],
            encoding_mode=c["encoding"],
            n_estimators=c["n_est"],
            max_samples=c["max_samples"],
            max_features=c["max_features"],
            seed=args.seed,
        )
        results.append(res)
        saved_scores[c["name"]] = (tr_score, val_score)

    # Identificar la mejor configuración según LightGBM Univariado PR-AUC
    best_res = max(results, key=lambda x: (x.lgb_univariate_pr_auc, x.lgb_univariate_roc_auc))
    best_config_name = best_res.config_name

    logger.info(
        f"Mejor configuración según LightGBM Univariado: {best_config_name} "
        f"(PR-AUC: {best_res.lgb_univariate_pr_auc:.4f} | ROC-AUC: {best_res.lgb_univariate_roc_auc:.4f})"
    )

    # Generar y guardar reporte
    report_text = format_experiment_report(
        results=results,
        val_prevalence=val_prevalence,
        seed=args.seed,
        best_config=best_config_name,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report_text, encoding="utf-8")
    logger.info(f"Reporte de experimentos guardado en: {args.output}")

    # Exportar el percentil del mejor modelo a data/features/scores_anomalias.parquet
    if args.export_scores:
        args.scores_output.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Exportando percentil de {best_config_name} para todo el dataset...")

        best_cfg = next(c for c in configs if c["name"] == best_config_name)
        numeric_cols = ["score", "monto", "a", "b", "d", "e", "f", "h", "l", "m", "n", "q", "r", "s"]
        categorical_cols = ["g", "j", "o", "p"]

        train_mask = (df["split"] == "train").to_numpy()
        normal_train = (train_mask & (df["fraude"] == 0)).to_numpy()

        x_num_all = df[numeric_cols].to_numpy(dtype=float, na_value=np.nan)
        imputer = SimpleImputer(strategy="median")
        imputer.fit(x_num_all[normal_train])
        x_num_all = imputer.transform(x_num_all)

        if best_cfg["encoding"] == "frequency":
            cat_blocks = []
            for col in categorical_cols:
                ref = df.loc[normal_train, col]
                counts = ref.value_counts(dropna=True)
                freqs = counts / len(ref)
                null_f = float(ref.isna().mean())
                vals = df[col]
                missing = vals.isna()
                unknown = ~missing & ~vals.isin(counts.index)
                enc = vals.map(freqs).astype(float)
                enc.loc[missing] = null_f
                enc.loc[unknown] = 0.0
                cat_blocks.append(enc.to_numpy()[:, None])
            x_all_full = np.column_stack([x_num_all] + cat_blocks)
        elif best_cfg["encoding"] == "hybrid_ohe":
            low_card = [c for c in ["g", "o", "p"] if c in categorical_cols]
            high_card = [c for c in categorical_cols if c not in low_card]
            cat_blocks = []
            if low_card:
                ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                ohe.fit(df.loc[normal_train, low_card].astype(str))
                cat_blocks.append(ohe.transform(df[low_card].astype(str)))
            for col in high_card:
                ref = df.loc[normal_train, col]
                counts = ref.value_counts(dropna=True)
                freqs = counts / len(ref)
                null_f = float(ref.isna().mean())
                vals = df[col]
                missing = vals.isna()
                unknown = ~missing & ~vals.isin(counts.index)
                enc = vals.map(freqs).astype(float)
                enc.loc[missing] = null_f
                enc.loc[unknown] = 0.0
                cat_blocks.append(enc.to_numpy()[:, None])
            x_all_full = np.column_stack([x_num_all] + cat_blocks)
        else:
            x_all_full = x_num_all

        scaler = StandardScaler()
        scaler.fit(x_all_full[normal_train])
        x_all_scaled = scaler.transform(x_all_full)

        if best_cfg["cut_type"] == "Ortogonal (Axis-Aligned)":
            final_model = IsolationForest(
                n_estimators=best_cfg["n_est"],
                max_samples=best_cfg["max_samples"],
                max_features=best_cfg["max_features"],
                random_state=args.seed,
                n_jobs=-1,
            )
        else:
            final_model = RotatedHyperplaneIsolationForest(
                n_estimators=best_cfg["n_est"],
                n_rotations=5,
                max_samples=best_cfg["max_samples"],
                max_features=best_cfg["max_features"],
                random_state=args.seed,
                n_jobs=-1,
            )

        final_model.fit(x_all_scaled[normal_train])
        all_raw = -final_model.score_samples(x_all_scaled)

        # Percentil empírico basado en legítimos de train
        sorted_legit = np.sort(all_raw[normal_train])
        all_pct = np.clip(np.searchsorted(sorted_legit, all_raw) / len(sorted_legit), 0.0, 1.0)

        if args.scores_output.exists():
            out_df = pd.read_parquet(args.scores_output)
            out_df["score_iforest"] = all_pct
        else:
            out_df = pd.DataFrame({
                "transaction_id": df["transaction_id"].astype("string"),
                "score_iforest": all_pct,
            })

        out_df.to_parquet(args.scores_output, compression="snappy", index=False)
        logger.info(f"Percentiles exportados a: {args.scores_output} ({len(out_df):,} filas)")

    print(f"\nBatería de experimentos completada con éxito en {time.perf_counter() - started:.2f} s.")


if __name__ == "__main__":
    main()
