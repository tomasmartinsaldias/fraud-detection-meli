"""
Tests unitarios para el módulo de experimentos con Local Outlier Factor (LOF).
"""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from src.models.anomaly_experiments.lof_experiments import (
    frequency_encode,
    prepare_features,
    to_percentiles,
    run_lof_experiment,
)


class LOFExperimentsTests(unittest.TestCase):
    def test_frequency_encode(self):
        train = pd.Series(["A", "A", "B", None])
        val = pd.Series(["A", "B", "C", None])
        normal = np.array([True, True, True, False])

        tr_enc, v_enc = frequency_encode(train, val, normal)
        self.assertEqual(len(tr_enc), 4)
        self.assertEqual(len(v_enc), 4)
        # C en val no fue visto en legítimos -> 0.0
        self.assertEqual(v_enc[2], 0.0)

    def test_to_percentiles_range(self):
        rng = np.random.RandomState(42)
        raw_tr = rng.exponential(size=100)
        raw_val = rng.exponential(size=30)
        normal_tr = rng.choice([True, False], size=100, p=[0.9, 0.1])

        pct_tr, pct_val = to_percentiles(raw_tr, raw_val, normal_tr)
        self.assertTrue(np.all(pct_tr >= 0.0) and np.all(pct_tr <= 1.0))
        self.assertTrue(np.all(pct_val >= 0.0) and np.all(pct_val <= 1.0))

    def test_lof_experiment_execution(self):
        rng = np.random.RandomState(42)
        x_tr = rng.randn(100, 5)
        x_val = rng.randn(30, 5)
        y_tr = rng.choice([0, 1], size=100, p=[0.9, 0.1])
        y_val = rng.choice([0, 1], size=30, p=[0.9, 0.1])
        ref_indices = np.arange(50)

        res, pct_tr, pct_val = run_lof_experiment(
            x_train=x_tr,
            x_val=x_val,
            y_train=y_tr,
            y_val=y_val,
            ref_indices=ref_indices,
            config_name="test_lof",
            distance_name="Euclidiana",
            p_value=2,
            k_neighbors=5,
            leaf_size=30,
            seed=42,
        )

        self.assertEqual(res.config_name, "test_lof")
        self.assertTrue(0.0 <= res.lgb_univariate_pr_auc <= 1.0)
        self.assertTrue(0.0 <= res.lgb_univariate_roc_auc <= 1.0)


if __name__ == "__main__":
    unittest.main()
