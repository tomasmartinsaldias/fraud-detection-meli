"""
Pipeline modular de extracción de características basadas en Reglas de Asociación de Clase (CARs).

Permite incorporar el panel heurístico de 10 reglas de alto impacto (WRAcc) como un artefacto
satélite independiente ('data/features/features_reglas_asociacion.parquet'), desacoplado del
pipeline transaccional central para preservar la modularidad y prevenir colisiones entre contratos.

Salida oficial: data/features/features_reglas_asociacion.parquet
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("build_rule_features")


def compute_heuristic_risk_features(
    df_base: pd.DataFrame,
    df_features: pd.DataFrame,
    include_individual_rules: bool = True,
) -> pd.DataFrame:
    """
    Calcula el Score Heurístico y Porcentaje de Violaciones de Fraude basado en un panel
    ortogonal de 10 reglas de asociación de alto WRAcc / impacto neto:
    - r1 (Perfil/Canal): p == 'N' AND o == 'N' (seguridad del medio de pago y autenticación ausentes)
    - r2 (Inmediatez/Sin historial): ratio_d_m >= 1.0 OR is_f_cero_neg == 1
    - r3 (Tensión Monetaria f): ratio_monto_f > p75 AND is_o_missing == 0
    - r4 (Categoría Crítica): te_j_bayes > p75 AND is_o_missing == 0
    - r5 (Contexto/Semántica): es_madrugada == 1 OR score_texto_nb > p90
    - r6 (Antigüedad Baja l): log_l <= p25 AND is_o_missing == 0
    - r7 (Tensión Monetaria m): ratio_monto_m > p75 AND is_o_missing == 0
    - r8 (Ratio b Anómalo + Joven): b > p90 AND m <= p25
    - r9 (Canal Atípico): a != 4 AND is_o_missing == 0
    - r10 (Discrepancia Temporal): diff_m_d <= p10

    Restricción Anti-Leakage:
    Los percentiles (p10, p25, p75, p90) se calculan EXCLUSIVAMENTE sobre split == 'train'.

    Args:
        df_base: DataFrame base con 'split', 'p', 'o', 'b', 'm', 'a'.
        df_features: DataFrame transaccional con ratios y features ingenierizadas.
        include_individual_rules: Si es True, incluye las banderas binarias individuales (r1 a r10).

    Returns:
        pd.DataFrame: Tabla con 'transaction_id', scores agregados y reglas booleanas.
    """
    logger.info("Calculando panel ortogonal de 10 reglas CARs (WRAcc)...")
    feats = pd.DataFrame({"transaction_id": df_base["transaction_id"]})
    train_mask = df_base["split"] == "train"

    # Umbrales aprendidos estrictamente en train para prevenir leakage
    th_mf = float(df_features.loc[train_mask, "ratio_monto_f"].quantile(0.75))
    th_te = float(df_features.loc[train_mask, "te_j_bayes"].quantile(0.75))
    th_nb = float(df_features.loc[train_mask, "score_texto_nb"].quantile(0.90))
    th_ll = float(df_features.loc[train_mask, "log_l"].quantile(0.25))
    th_mm = float(df_features.loc[train_mask, "ratio_monto_m"].quantile(0.75))
    th_b = float(df_base.loc[train_mask, "b"].quantile(0.90))
    th_m = float(df_base.loc[train_mask, "m"].quantile(0.25))
    th_diff = float(df_features.loc[train_mask, "diff_m_d"].quantile(0.10))

    # Definición de las 10 reglas
    r1 = (df_base["p"] == "N") & (df_base["o"] == "N")
    r2 = (df_features["ratio_d_m"] >= 1.0) | (df_features["is_f_cero_neg"] == 1)
    r3 = (df_features["ratio_monto_f"] > th_mf) & (df_features["is_o_missing"] == 0)
    r4 = (df_features["te_j_bayes"] > th_te) & (df_features["is_o_missing"] == 0)
    r5 = (df_features["es_madrugada"] == 1) | (df_features["score_texto_nb"] > th_nb)
    r6 = (df_features["log_l"] <= th_ll) & (df_features["is_o_missing"] == 0)
    r7 = (df_features["ratio_monto_m"] > th_mm) & (df_features["is_o_missing"] == 0)
    r8 = (df_base["b"] > th_b) & (df_base["m"] <= th_m)
    r9 = (df_base["a"].astype(str) != "4.0") & (df_features["is_o_missing"] == 0)
    r10 = df_features["diff_m_d"] <= th_diff

    score_10 = (
        r1.astype(np.int8)
        + r2.astype(np.int8)
        + r3.astype(np.int8)
        + r4.astype(np.int8)
        + r5.astype(np.int8)
        + r6.astype(np.int8)
        + r7.astype(np.int8)
        + r8.astype(np.int8)
        + r9.astype(np.int8)
        + r10.astype(np.int8)
    )

    feats["score_heuristico_fraude_10"] = score_10.astype(np.int8)
    feats["pct_heuristico_fraude"] = (score_10 / 10.0).astype(np.float64)

    if include_individual_rules:
        feats["rule_car_1_seguridad_fallida"] = r1.astype(np.int8)
        feats["rule_car_2_inmediatez_sin_historial"] = r2.astype(np.int8)
        feats["rule_car_3_tension_monto_f"] = r3.astype(np.int8)
        feats["rule_car_4_categoria_critica"] = r4.astype(np.int8)
        feats["rule_car_5_madrugada_o_texto_nb"] = r5.astype(np.int8)
        feats["rule_car_6_antiguedad_baja_l"] = r6.astype(np.int8)
        feats["rule_car_7_tension_monto_m"] = r7.astype(np.int8)
        feats["rule_car_8_b_alto_joven"] = r8.astype(np.int8)
        feats["rule_car_9_canal_atipico"] = r9.astype(np.int8)
        feats["rule_car_10_discrepancia_m_d"] = r10.astype(np.int8)

    return feats


def build_rule_features_dataset(
    input_base_path: Path,
    input_features_path: Path,
    output_rules_path: Path,
) -> pd.DataFrame:
    """
    Función orquestadora que genera el dataset satélite de reglas de asociación.

    Args:
        input_base_path: Ruta a data/processed/dataset_base.parquet.
        input_features_path: Ruta a data/features/features_transaccionales.parquet.
        output_rules_path: Ruta destino para data/features/features_reglas_asociacion.parquet.

    Returns:
        pd.DataFrame: Tabla generada con reglas y transaction_id.
    """
    if not input_base_path.exists():
        raise FileNotFoundError(f"No se encontró el dataset base en: {input_base_path.resolve()}")
    if not input_features_path.exists():
        raise FileNotFoundError(f"No se encontró el dataset de features en: {input_features_path.resolve()}")

    logger.info(f"Cargando dataset base desde: {input_base_path}")
    df_base = pd.read_parquet(input_base_path)

    logger.info(f"Cargando dataset transaccional desde: {input_features_path}")
    df_features = pd.read_parquet(input_features_path)

    df_rules = compute_heuristic_risk_features(df_base, df_features, include_individual_rules=True)

    # Validaciones de integridad
    assert len(df_rules) == len(df_base), "Error: Se alteró el número de filas."
    assert df_rules["transaction_id"].is_unique, "Error: transaction_id duplicado."

    # Guardado atómico
    output_rules_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_rules_path.with_suffix(".parquet.tmp")
    df_rules.to_parquet(tmp_path, engine="pyarrow", index=False)
    os.replace(tmp_path, output_rules_path)

    size_mb = output_rules_path.stat().st_size / (1024 * 1024)
    logger.info(
        f"Dataset de reglas de asociación generado exitosamente en: {output_rules_path} "
        f"({len(df_rules):,} filas x {df_rules.shape[1]} columnas, {size_mb:.2f} MB)"
    )
    return df_rules


def main() -> None:
    """CLI para ejecución reproducible."""
    parser = argparse.ArgumentParser(
        description="Generador satélite de features basadas en reglas de asociación (CARs)."
    )
    parser.add_argument(
        "--input_base",
        type=Path,
        default=Path("data/processed/dataset_base.parquet"),
        help="Ruta al dataset base procesado.",
    )
    parser.add_argument(
        "--input_features",
        type=Path,
        default=Path("data/features/features_transaccionales.parquet"),
        help="Ruta al dataset de features transaccionales.",
    )
    parser.add_argument(
        "--output_rules",
        type=Path,
        default=Path("data/features/features_reglas_asociacion.parquet"),
        help="Ruta destino para guardar las features de reglas de asociación.",
    )
    args = parser.parse_args()

    build_rule_features_dataset(
        input_base_path=args.input_base,
        input_features_path=args.input_features,
        output_rules_path=args.output_rules,
    )


if __name__ == "__main__":
    main()
