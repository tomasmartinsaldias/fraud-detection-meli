"""
Pipeline modular de ingeniería de características transaccionales.

Genera variables analíticas validadas previamente mediante el modelo proxy,
respetando de forma estricta las restricciones anti-leakage temporal y el contrato de datos.

Salida oficial: data/features/features_transaccionales.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("build_features")


def compute_bayesian_monto_ratio(
    df: pd.DataFrame,
    m_weight: float = 30.0,
) -> pd.Series:
    """
    Calcula el ratio de monto normalizado por (a, g) con suavizado bayesiano hacia la media de 'a'.
    
    Restricción Anti-Leakage:
    Las medias a nivel 'a' y a nivel '(a, g)' se calculan EXCLUSIVAMENTE sobre split == 'train'.

    Args:
        df: DataFrame completo con 'transaction_id', 'split', 'monto', 'a' y 'g'.
        m_weight: Peso bayesiano para regularizar celdas con pocas observaciones.

    Returns:
        pd.Series: Serie con el ratio bayesiano alineada al índice de df.
    """
    logger.info("Calculando feature: ratio_monto_ag_bayes...")
    train_mask = df["split"] == "train"
    train_df = df[train_mask]

    # Medias de train
    mean_a = train_df.groupby("a")["monto"].mean().to_dict()
    stats_ag = train_df.groupby(["a", "g"])["monto"].agg(["count", "mean"]).reset_index()
    stats_ag["global_a"] = stats_ag["a"].map(mean_a)
    stats_ag["bayes_mean"] = (
        stats_ag["count"] * stats_ag["mean"] + m_weight * stats_ag["global_a"]
    ) / (stats_ag["count"] + m_weight)

    map_bayes = dict(zip(zip(stats_ag["a"], stats_ag["g"]), stats_ag["bayes_mean"]))

    # Mapeo sobre todo el dataset (train, val, test) usando fallback seguro a media de a
    ratios = [
        monto_val / map_bayes.get((row_a, row_g), mean_a.get(row_a, np.nan))
        for monto_val, row_a, row_g in zip(df["monto"], df["a"], df["g"])
    ]
    return pd.Series(ratios, index=df.index, name="ratio_monto_ag_bayes", dtype=np.float64)


def compute_profile_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula transformaciones no lineales e interacciones de las variables continuas de perfil b y c.
    
    Selección optimizada por el Modelo Proxy:
    - ratio_monto_b: Tensión entre el ticket monetario y la tasa continua b (+1700 gain).
    - prod_b_log_c: Interacción multiplicativa no lineal entre el factor b y log(c + 1) (+1800 gain).
    - ratio_monto_c: Proporción del monto sobre la escala cuantitativa de c (+1400 gain).

    Args:
        df: DataFrame base con 'monto', 'b' y 'c'.

    Returns:
        pd.DataFrame: DataFrame con las 3 features de interacción más potentes.
    """
    logger.info("Calculando transformaciones e interacciones de b y c...")
    feats = pd.DataFrame(index=df.index)

    log_c = np.log1p(df["c"].clip(lower=0)).astype(np.float64)

    # 1. ratio_monto_b: Tensión entre el ticket monetario y la tasa continua b
    feats["ratio_monto_b"] = (df["monto"] / (df["b"] + 0.001)).astype(np.float64)

    # 2. prod_b_log_c: Interacción multiplicativa no lineal entre el factor b y log(c + 1)
    feats["prod_b_log_c"] = (df["b"] * log_c).astype(np.float64)

    # 3. ratio_monto_c: Proporción del monto actual sobre la escala de c
    feats["ratio_monto_c"] = (df["monto"] / (df["c"] + 1.0)).astype(np.float64)

    return feats


def compute_dm_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula transformaciones no lineales e interacciones de las variables d y m.
    
    Selección optimizada y validada mediante el Modelo Proxy (Delta PR-AUC +0.0073 conjunto):
    - ratio_d_m: Proporción d / (m + 1.0). Captura la tensión entre ambas escalas relativas (+2500 a 4000 gain).
    - ratio_monto_m: Normalización monto / (m + 1.0). Relación entre el importe y la magnitud de m (+3100 gain).
    - diff_m_d: Diferencia directa m - d. Identifica discrepancias anómalas entre variables correlacionadas (+4400 gain).

    Args:
        df: DataFrame base con 'monto', 'd' y 'm'.

    Returns:
        pd.DataFrame: DataFrame con las 3 features calculadas.
    """
    logger.info("Calculando transformaciones e interacciones de d y m...")
    feats = pd.DataFrame(index=df.index)

    # 1. ratio_d_m: Escala relativa entre d y m (+1.0 evita división por cero cuando m == 0)
    feats["ratio_d_m"] = (df["d"] / (df["m"] + 1.0)).astype(np.float64)

    # 2. ratio_monto_m: Normalización del ticket respecto a m
    feats["ratio_monto_m"] = (df["monto"] / (df["m"] + 1.0)).astype(np.float64)

    # 3. diff_m_d: Discrepancia directa entre m y d
    feats["diff_m_d"] = (df["m"] - df["d"]).astype(np.float64)

    return feats


def build_transactional_features(
    input_base_path: Path,
    output_features_path: Path,
) -> pd.DataFrame:
    """
    Función orquestadora que carga el dataset base oficial, computa las features
    validadas y las exporta a Parquet con transaction_id como clave primaria.

    Args:
        input_base_path: Ruta a data/processed/dataset_base.parquet.
        output_features_path: Ruta a data/features/features_transaccionales.parquet.

    Returns:
        pd.DataFrame: Tabla generada y exportada.
    """
    if not input_base_path.exists():
        raise FileNotFoundError(f"No se encontró el dataset base en: {input_base_path.resolve()}")

    logger.info(f"Cargando dataset base desde: {input_base_path}")
    df_base = pd.read_parquet(input_base_path)

    # DataFrame receptor con clave primaria obligatoria
    df_features = pd.DataFrame({"transaction_id": df_base["transaction_id"]})

    # 1. Feature de país y tipo de transacción (Suavizado Bayesiano)
    df_features["ratio_monto_ag_bayes"] = compute_bayesian_monto_ratio(df_base)

    # 2. Features de interacción de perfil (b y c)
    interactions = compute_profile_interactions(df_base)
    for col in interactions.columns:
        df_features[col] = interactions[col]

    # 3. Features de interacción matemática de d y m
    dm_interactions = compute_dm_interactions(df_base)
    for col in dm_interactions.columns:
        df_features[col] = dm_interactions[col]

    # Validaciones de integridad según Contrato A
    assert len(df_features) == len(df_base), "Error: Se alteró el número de filas."
    assert df_features["transaction_id"].is_unique, "Error: transaction_id duplicado."

    # Exportación atómica a Parquet
    output_features_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_features_path.with_suffix(".parquet.tmp")
    df_features.to_parquet(tmp_path, engine="pyarrow", index=False)
    import os
    os.replace(tmp_path, output_features_path)

    size_mb = output_features_path.stat().st_size / (1024 * 1024)
    logger.info(
        f"Features transaccionales generadas exitosamente en: {output_features_path} "
        f"({len(df_features):,} filas x {df_features.shape[1]} columnas, {size_mb:.2f} MB)"
    )
    return df_features


def main() -> None:
    """CLI para ejecución reproducible."""
    parser = argparse.ArgumentParser(
        description="Generador de features transaccionales para detección de fraude."
    )
    parser.add_argument(
        "--input_base",
        type=Path,
        default=Path("data/processed/dataset_base.parquet"),
        help="Ruta al dataset base procesado.",
    )
    parser.add_argument(
        "--output_features",
        type=Path,
        default=Path("data/features/features_transaccionales.parquet"),
        help="Ruta destino para guardar las features generadas.",
    )
    args = parser.parse_args()

    build_transactional_features(
        input_base_path=args.input_base,
        output_features_path=args.output_features,
    )


if __name__ == "__main__":
    main()
