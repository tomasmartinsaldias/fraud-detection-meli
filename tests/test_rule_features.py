"""
Tests unitarios para el generador satélite de reglas de asociación (CARs).
"""

from pathlib import Path
import pandas as pd
import pytest

from src.features.build_rule_features import compute_heuristic_risk_features, build_rule_features_dataset


def test_compute_heuristic_risk_features_contract():
    base_path = Path("data/processed/dataset_base.parquet")
    features_path = Path("data/features/features_transaccionales.parquet")

    if not base_path.exists() or not features_path.exists():
        pytest.skip("Datasets no disponibles para test de integración.")

    df_base = pd.read_parquet(base_path).head(100)
    df_feat = pd.read_parquet(features_path).head(100)

    res = compute_heuristic_risk_features(df_base, df_feat, include_individual_rules=True)

    assert "transaction_id" in res.columns
    assert "score_heuristico_fraude_10" in res.columns
    assert "pct_heuristico_fraude" in res.columns
    assert len(res) == len(df_base)
    assert res["transaction_id"].is_unique

    # Verificar que los scores estén en rangos válidos
    assert (res["score_heuristico_fraude_10"] >= 0).all()
    assert (res["score_heuristico_fraude_10"] <= 10).all()
    assert (res["pct_heuristico_fraude"] >= 0.0).all()
    assert (res["pct_heuristico_fraude"] <= 1.0).all()
