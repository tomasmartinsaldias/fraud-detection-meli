"""
Tests unitarios para el paquete src/models/production.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.production.generate_features import (
    EXPECTED_OUTPUT_COLUMNS,
    run_production_pipeline,
    validate_output_dataframe,
)
from tests.anomaly_detection.test_estimators import _make_dummy_df


class ProductionTests(unittest.TestCase):
    def test_validate_output_dataframe_success(self):
        df = pd.DataFrame({
            "transaction_id": ["tx_001", "tx_002"],
            "pca_q": [0.1, 0.9],
            "pca_magnitud": [0.2, 0.8],
            "pca_t2": [0.15, 0.85],
            "score_iforest_perpendicular": [0.3, 0.7],
            "score_iforest_diagonal": [0.25, 0.75],
            "score_lof_manhattan": [0.4, 0.6],
            "score_lof_euclidean": [0.35, 0.65],
            "score_ocsvm_rbf_nu_003": [0.2, 0.8],
            "score_ocsvm_rbf_nu_005": [0.22, 0.78],
            "score_gmm_k3": [0.18, 0.82],
            "score_gmm_k5": [0.19, 0.81],
            "score_anomalia_consolidado": [0.05, 0.92],
            "anomalia_x_monto": [0.15, 2.76],
        })
        validate_output_dataframe(df, expected_rows=2)

    def test_run_production_pipeline_end_to_end(self):
        df = _make_dummy_df(120)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            in_path = tmp / "base.parquet"
            mod_path = tmp / "bundle.joblib"
            out_path = tmp / "scores.parquet"

            df.to_parquet(in_path, index=False)

            res_df = run_production_pipeline(
                input_path=in_path,
                model_path=mod_path,
                output_path=out_path,
                seed=42,
            )

            self.assertTrue(mod_path.exists())
            self.assertTrue(out_path.exists())
            self.assertEqual(len(res_df), 120)
            self.assertEqual(list(res_df.columns), EXPECTED_OUTPUT_COLUMNS)


if __name__ == "__main__":
    unittest.main()
