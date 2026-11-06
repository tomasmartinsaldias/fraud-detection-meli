"""
Tests unitarios para el paquete src/models/evaluation.
"""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

import numpy as np

from src.models.evaluation.evaluate_models import (
    compute_score_metrics,
    print_console_report,
)


class EvaluationTests(unittest.TestCase):
    def test_compute_score_metrics(self):
        y_true = np.array([0, 0, 0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.85, 0.95])

        row = compute_score_metrics(y_true, scores, "test_score")
        self.assertEqual(row.score_name, "test_score")
        self.assertAlmostEqual(row.pr_auc, 1.0, places=4)
        self.assertAlmostEqual(row.roc_auc, 1.0, places=4)
        self.assertAlmostEqual(row.best_f1, 1.0, places=4)
        self.assertAlmostEqual(row.precision_at_best_f1, 1.0, places=4)
        self.assertAlmostEqual(row.recall_at_best_f1, 1.0, places=4)

    def test_print_console_report_outputs_to_stdout(self):
        y_true = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.4, 0.8, 0.9])
        row = compute_score_metrics(y_true, scores, "score_anomalia_consolidado")

        buf = io.StringIO()
        with redirect_stdout(buf):
            print_console_report([row], n_val=4, n_frauds=2, prevalence=0.5)

        output = buf.getvalue()
        self.assertIn("REPORTE DE EVALUACIÓN DE MODELOS DE ANOMALÍAS", output)
        self.assertIn("score_anomalia_consolidado", output)


if __name__ == "__main__":
    unittest.main()
