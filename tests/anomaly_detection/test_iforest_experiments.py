"""
Tests unitarios para el módulo iforest_experiments.
"""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from src.models.iforest_experiments import (
    RotatedHyperplaneIsolationForest,
    frequency_encode,
    prepare_features,
)


class IForestExperimentsTests(unittest.TestCase):
    def test_rotated_hyperplane_isolation_forest(self):
        rng = np.random.RandomState(42)
        X_train = rng.randn(100, 5)
        X_test = rng.randn(20, 5)

        model = RotatedHyperplaneIsolationForest(
            n_estimators=20,
            n_rotations=2,
            max_samples=32,
            random_state=42,
        )
        model.fit(X_train)
        scores = model.score_samples(X_test)

        self.assertEqual(len(scores), 20)
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertTrue(np.all(scores <= 1.0))

    def test_frequency_encode(self):
        train_series = pd.Series(["A", "A", "B", "C", None])
        val_series = pd.Series(["A", "B", "D", None])
        normal_mask = np.array([True, True, True, False, True])  # A(2), B(1), None(1) en legítimos

        t_enc, v_enc, meta = frequency_encode(train_series, val_series, normal_mask)

        self.assertEqual(meta["categories"], 2)  # A y B
        self.assertEqual(meta["unknown_val"], 1)  # D
        self.assertEqual(meta["missing_val"], 1)  # None

        # En val: D debe valer 0.0
        self.assertEqual(v_enc[2], 0.0)
        # None debe valer la proporción de nulos en legítimos: 1/4 = 0.25
        self.assertEqual(v_enc[3], 0.25)


if __name__ == "__main__":
    unittest.main()
