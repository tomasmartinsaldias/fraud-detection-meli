"""Contract C regression tests. Synthetic data checks behavior, not model quality."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.models.supervised import (
    BASE_CATEGORICAL, BASE_NUMERIC, FrameEncoder, cost_curve, merge_features,
    policy_metrics, validation_blocks,
)
from src.models.train import build_full_dataset, develop, evaluate_test, load_data


class SupervisedTests(unittest.TestCase):
    def test_merge_rejects_missing_duplicate_and_conflicting_rows(self):
        base = pd.DataFrame({"transaction_id": ["a", "b"], "score": [1., 2.]})
        for features in [pd.DataFrame({"transaction_id": ["a"], "extra": [1]}),
                         pd.DataFrame({"transaction_id": ["a", "a"], "extra": [1, 2]}),
                         pd.DataFrame({"transaction_id": ["a", "b"], "score": [2., 1.]})]:
            with self.assertRaises(ValueError):
                merge_features(base, features)
        result = merge_features(base, pd.DataFrame({"transaction_id": ["b", "a"], "extra": [4., 3.]}))
        self.assertEqual(result.extra.tolist(), [3., 4.])

    def test_unseen_categories_do_not_enter_vocabulary(self):
        encoder = FrameEncoder().fit(pd.DataFrame({"o": ["Y", "N"], "score": [1, 2]}))
        encoded = encoder.transform(pd.DataFrame({"o": ["new"], "score": [3]}))
        self.assertTrue(encoded.o.isna().all())
        self.assertNotIn("new", encoder.categories["o"])

    def test_cost_curve_matches_brute_force_with_ties(self):
        y, p, amount = np.array([1, 0, 1, 0]), np.array([.5, .5, .9, .1]), np.array([20., 4., 5., 8.])
        curve = cost_curve(y, p, amount, 7., .8)
        for row in curve.itertuples():
            expected = policy_metrics(y, p, amount, row.threshold, 7., .8)
            self.assertAlmostEqual(row.cost, expected["cost"])
            self.assertEqual(row.fp, expected["fp"])
        self.assertEqual(len(curve), 4)
        self.assertEqual(curve.iloc[0].fp, 0)

    def test_assignment_fixed_costs_and_capacity(self):
        y, p, amounts = [1, 0, 1], [.9, .7, .1], [1., 100000., 2.]
        curve = cost_curve(y, p, amounts, 5., fn_cost=200.)
        constrained = curve.loc[curve.fp == 0]
        best = constrained.loc[constrained.cost.idxmin()]
        self.assertEqual(best.cost, 200.)
        self.assertEqual(best.fn, 1)
        self.assertEqual(best.threshold, .9)
        result = policy_metrics(y, p, amounts, .7, 5., 1., fn_cost=200.)
        self.assertEqual(result["cost"], 205.)
        self.assertEqual(result["recall"], .5)
        self.assertEqual(result["precision"], .5)

    def test_temporal_blocks_do_not_split_timestamps(self):
        frame = pd.DataFrame({"transaction_id": [str(i) for i in range(300)],
                              "fecha": np.repeat(pd.date_range("2025-01-01", periods=150), 2),
                              "fraude": np.arange(300) % 2})
        a, b, c = validation_blocks(frame)
        self.assertLess(a.fecha.max(), b.fecha.min())
        self.assertLess(b.fecha.max(), c.fecha.min())
        self.assertEqual(len(a) + len(b) + len(c), len(frame))

    def test_end_to_end_and_test_sealing(self):
        rng = np.random.default_rng(42)
        n = 1000
        frame = pd.DataFrame({c: rng.normal(size=n) for c in BASE_NUMERIC})
        for c in BASE_CATEGORICAL:
            frame[c] = np.where(np.arange(n) % 3, "Y", "N")
        frame["monto"] = rng.uniform(1, 100, n)
        frame["transaction_id"] = [f"tx_{i:06}" for i in range(n)]
        frame["fecha"] = pd.date_range("2025-01-01", periods=n, freq="h")
        frame["split"] = ["train"] * 500 + ["val"] * 300 + ["test"] * 200
        frame["fraude"] = (np.arange(n) % 4 == 0).astype(int)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.parquet"
            frame.to_parquet(base)
            args = argparse.Namespace(base=base, features=None, anomalies=None, output=root / "run",
                                      seed=42, trials=2, max_trees=5, jobs=1, fp_costs=[1., 5.], loss_fraction=1., fn_cost=200., max_fp=100,
                                      exclude_anomaly_cols=[])
            original_read = pd.read_parquet
            observed = []

            def guarded_read(path, **kwargs):
                if Path(path) == base:
                    observed.append(kwargs.get("filters"))
                    self.assertEqual(kwargs.get("filters"), [("split", "in", ["train", "val"])])
                return original_read(path, **kwargs)

            with patch("src.models.train.pd.read_parquet", side_effect=guarded_read):
                develop(args)
            self.assertTrue(observed)
            manifest = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["test_used"])
            self.assertEqual(len(manifest["comparison"]), 5)
            self.assertTrue((args.output / "trials_raw_cmaes.json").exists())
            amount = manifest["policy_validation"]["logistic_raw"]["amount_fn_val_only"]
            self.assertLessEqual(amount["fp"], 100)
            self.assertEqual(amount["fp_cost"], 1.)
            args.bundle = args.output / "bundle.joblib"
            evaluate_test(args)
            result = json.loads((args.output / "test_report.json").read_text(encoding="utf-8"))
            self.assertEqual(result["rows"], 200)
            self.assertEqual(result["winner_frozen_on_val"], manifest["winner"])
            with self.assertRaises(ValueError):
                evaluate_test(args)
            self.assertNotIn("monto", "".join(result["results"]["logistic_raw"]["scenarios"]))

    def test_build_full_dataset_and_anomaly_exclusion(self):
        ids = [f"tx_{i:06}" for i in range(6)]
        base = pd.DataFrame({"transaction_id": ids, "split": ["train"] * 2 + ["val"] * 2 + ["test"] * 2,
                             "fraude": [0, 1] * 3, "score": np.arange(6.)})
        feat = pd.DataFrame({"transaction_id": ids[::-1], "score": np.arange(6.)[::-1], "extra": np.ones(6)})
        anom = pd.DataFrame({"transaction_id": ids, "s1": np.linspace(0, 1, 6), "consolidado": np.zeros(6)})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, frame in (("b", base), ("f", feat), ("a", anom)):
                frame.to_parquet(root / f"{name}.parquet")
            full = build_full_dataset(root / "b.parquet", root / "f.parquet", root / "a.parquet",
                                      exclude_anomaly_cols=["consolidado"])
            self.assertEqual(len(full), 6)
            self.assertEqual(full.transaction_id.tolist(), ids)
            self.assertIn("s1", full)
            self.assertNotIn("consolidado", full)
            dev = build_full_dataset(root / "b.parquet", root / "f.parquet", None, splits=("train", "val"))
            self.assertEqual(set(dev.split), {"train", "val"})
            with self.assertRaises(ValueError):
                build_full_dataset(root / "b.parquet", None, root / "a.parquet", exclude_anomaly_cols=["nope"])


if __name__ == "__main__":
    unittest.main()
