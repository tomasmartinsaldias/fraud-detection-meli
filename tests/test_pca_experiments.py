"""Controles de geometría PCA y aislamiento de las particiones."""

import unittest
import contextlib
import io
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from sklearn.metrics import roc_auc_score

from src.models.pca_experiments import calculate_scores, frequency_encode, main, run_experiments, select_features


class PCAExperimentsTests(unittest.TestCase):
    def test_frequency_reference_excludes_fraud_and_handles_missing_and_unknown(self):
        train = pd.Series(["a", "a", "b", None, "fraud_only"])
        val = pd.Series(["a", "b", None, "fraud_only", "new"])
        encoded_train, encoded_val, info = frequency_encode(train, val, np.array([True, True, True, True, False]))
        np.testing.assert_allclose(encoded_train, [0.5, 0.5, 0.25, 0.25, 0])
        np.testing.assert_allclose(encoded_val, [0.5, 0.25, 0.25, 0, 0])
        self.assertEqual(info["unknown_val"], 2)

    def test_residual_and_distance_detect_different_directions(self):
        # Nube alargada en x: (4, 0) es extremo dentro del espacio;
        # (0, 2) se aparta de él, pero no de su centro proyectado.
        pca = PCA(n_components=1).fit(np.array([[-3., 0.], [-1., 0.], [1., 0.], [3., 0.]]))
        scores = calculate_scores(pca, np.array([[4., 0.], [0., 2.]]))
        self.assertGreater(scores["magnitud_ponderada_T2"][0], scores["magnitud_ponderada_T2"][1])
        self.assertLess(scores["reconstruccion_Q"][0], scores["reconstruccion_Q"][1])

    def test_pca_fit_uses_only_legitimate_train_and_ignores_test(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame(rng.normal(size=(90, 3)), columns=["a", "b", "c"])
        df["split"] = ["train"] * 50 + ["val"] * 20 + ["test"] * 20
        df["fraude"] = [0] * 40 + [1] * 10 + [0, 1] * 10 + [1] * 20
        original_fit = PCA.fit
        seen_sizes = []

        def record_fit(model, x, *args, **kwargs):
            seen_sizes.append(len(x))
            return original_fit(model, x, *args, **kwargs)

        with patch.object(PCA, "fit", record_fit):
            rows, _, probabilities = run_experiments(df, ["a", "b", "c"])
        self.assertEqual(seen_sizes, [40, 40, 40])
        self.assertEqual(len(rows), 12)
        self.assertEqual(sum(row["score"] == "tres_scores" for row in rows), 3)
        df.loc[df["split"] == "test", ["a", "b", "c", "fraude"]] = 100000
        self.assertEqual(probabilities.shape, (20, 12))
        self.assertAlmostEqual(rows[0]["roc_auc"], roc_auc_score(df.loc[df["split"] == "val", "fraude"], probabilities.iloc[:, 0]))
        changed_rows, _, changed_probabilities = run_experiments(df, ["a", "b", "c"])
        self.assertEqual(rows, changed_rows)
        pd.testing.assert_frame_equal(probabilities, changed_probabilities)

    def test_auto_selection_and_optional_export_preserve_validation_ids(self):
        rng = np.random.default_rng(5)
        df = pd.DataFrame(rng.normal(size=(90, 4)), columns=["a", "b", "c", "k"])
        df["i"] = "texto"
        df["g"] = np.where(df.index % 2, "A", "B")
        df.loc[0, "g"] = None
        df["split"] = ["train"] * 50 + ["val"] * 20 + ["test"] * 20
        df["fraude"] = [0] * 40 + [1] * 10 + [0, 1] * 10 + [1] * 20
        df["transaction_id"] = [f"tx_{n:06d}" for n in range(len(df))]
        self.assertEqual(select_features(df, ["c", "k", "i", "b"], None, None), (["a"], ["g"]))
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "input.parquet"
            output_path = tmp_path / "report.md"
            probs_path = tmp_path / "probs.parquet"
            df.to_parquet(input_path, index=False)
            argv = ["pca_experiments.py", "--input", str(input_path), "--output", str(output_path), "--probs-output", str(probs_path)]
            with patch("sys.argv", argv + ["--export-probs", "0"]), contextlib.redirect_stdout(io.StringIO()):
                main()
            self.assertFalse(probs_path.exists())
            with patch("sys.argv", argv + ["--export-probs", "1"]), contextlib.redirect_stdout(io.StringIO()):
                main()
            exported = pd.read_parquet(probs_path)
            self.assertEqual(exported["transaction_id"].tolist(), df.loc[df["split"] == "val", "transaction_id"].tolist())
            self.assertEqual(exported.shape, (20, 13))
            self.assertTrue(exported.iloc[:, 1:].apply(lambda col: col.between(0, 1).all()).all())


if __name__ == "__main__":
    unittest.main()
