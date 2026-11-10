import unittest
import numpy as np
import pandas as pd
from src.models.compare_team import WeightedModels, ScoreModel, choose_threshold, policy
from src.models.supervised import IdentityEncoder

class TeamComparisonTests(unittest.TestCase):
    def test_weighted_predictions_use_frozen_members(self):
        frame = pd.DataFrame({"a": [.1, .9], "b": [.5, .3]})
        members = [{"model": ScoreModel(c), "encoder": IdentityEncoder()} for c in ("a", "b")]
        model = WeightedModels(members, [.25, .75])
        np.testing.assert_allclose(model.predict_proba(frame)[:, 1], [.4, .45])

    def test_cost_threshold_respects_capacity_and_ties(self):
        frame = pd.DataFrame({"fraude": [1, 0, 1, 0], "monto": [1., 1000., 3., 5.]})
        probability = np.array([.9, .7, .7, .1])
        threshold = choose_threshold(frame, probability, 0)
        result = policy(frame, probability, threshold)
        self.assertEqual(result["fp"], 0)
        self.assertEqual(result["fn"], 1)
        self.assertEqual(result["cost"], 200.)
        self.assertEqual(threshold, .9)

    def test_serialized_ensemble_loads_in_fresh_process(self):
        import joblib, subprocess, sys, tempfile
        from pathlib import Path
        members = [{"model": ScoreModel("a"), "encoder": IdentityEncoder()}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ensemble.joblib"
            joblib.dump(WeightedModels(members, [1.]), path)
            result = subprocess.run([sys.executable, "-B", "-c",
                "import joblib,sys,pandas as pd; m=joblib.load(sys.argv[1]); print(m.predict_proba(pd.DataFrame({'a':[0.25]}))[0,1])",
                str(path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(float(result.stdout.strip()), .25)

if __name__ == "__main__":
    unittest.main()
