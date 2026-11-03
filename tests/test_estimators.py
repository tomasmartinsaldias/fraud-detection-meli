"""
Tests unitarios para el paquete src/models/estimators.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.estimators.anomaly_bundle import AnomalyPipelineBundle
from src.models.estimators.preprocessing import (
    FrequencyPreprocessor,
    HybridIForestPreprocessor,
)


def _make_dummy_df(n_rows: int = 120, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    data = {
        "transaction_id": [f"tx_{i:06d}" for i in range(n_rows)],
        "split": ["train"] * (n_rows - 20) + ["val"] * 20,
        "fraude": rng.choice([0, 1], size=n_rows, p=[0.85, 0.15]),
        "score": rng.uniform(0, 100, size=n_rows),
        "monto": rng.exponential(scale=50.0, size=n_rows),
    }
    for col in ["a", "b", "d", "e", "f", "h", "l", "m", "n", "q", "r", "s"]:
        data[col] = rng.randn(n_rows)

    # Categóricas: g (3 categorías -> 2 OHE), o (2 categorías -> 1 OHE), p (2 categorías -> 1 OHE), j (alta card)
    data["g"] = rng.choice(["cat1", "cat2", "cat3"], size=n_rows)
    data["o"] = rng.choice(["Y", "N"], size=n_rows)
    data["p"] = rng.choice(["T", "F"], size=n_rows)
    data["j"] = [f"item_{i % 30}" for i in range(n_rows)]

    return pd.DataFrame(data)


class EstimatorsTests(unittest.TestCase):
    def test_hybrid_preprocessor_k_minus_1_columns(self):
        df = _make_dummy_df(100)
        legit = df.loc[df["fraude"] == 0]
        pre = HybridIForestPreprocessor().fit(legit)
        out = pre.transform(df)

        # 14 numéricas + (3-1 para g) + (2-1 para o) + (2-1 para p) + 1 frecuencia para j = 14 + 2 + 1 + 1 + 1 = 19
        self.assertEqual(out.shape, (100, 19))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_frequency_preprocessor_shape(self):
        df = _make_dummy_df(100)
        legit = df.loc[df["fraude"] == 0]
        pre = FrequencyPreprocessor().fit(legit)
        out = pre.transform(df)

        # 14 numéricas + 4 frecuencias = 18
        self.assertEqual(out.shape, (100, 18))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_bundle_fit_predict_save_load(self):
        df = _make_dummy_df(120)
        df_train = df.loc[df["split"] == "train"].copy()
        df_val = df.loc[df["split"] == "val"].copy()

        bundle = AnomalyPipelineBundle(
            pca_variance=0.95,
            iforest_n_estimators=20,
            iforest_max_samples=32,
            lof_sample_size=50,
            seed=42,
        )
        bundle.fit(df_train)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "bundle.joblib"
            bundle.save(model_path)
            loaded = AnomalyPipelineBundle.load(model_path)

        scores_df = loaded.predict_scores_df(df_val)
        expected_cols = [
            "transaction_id",
            "pca_q",
            "pca_magnitud",
            "pca_t2",
            "score_iforest_perpendicular",
            "score_iforest_diagonal",
            "score_lof_manhattan",
            "score_lof_euclidean",
            "score_ocsvm_rbf_nu_003",
            "score_ocsvm_rbf_nu_005",
            "score_gmm_k3",
            "score_gmm_k5",
            "score_anomalia_consolidado",
        ]
        self.assertEqual(list(scores_df.columns), expected_cols)
        self.assertEqual(len(scores_df), len(df_val))

        for col in expected_cols[1:]:
            vals = scores_df[col].to_numpy()
            self.assertTrue(np.all(vals >= 0.0) and np.all(vals <= 1.0))


if __name__ == "__main__":
    unittest.main()
