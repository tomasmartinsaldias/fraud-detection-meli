"""
Módulo de Selección Inteligente de Características (Feature Selection).

Aplica Recursive Feature Elimination (RFE) optimizado con descarte adaptativo
y salvaguardas de rendimiento sobre validación OOT (PR-AUC), garantizando
parsimonia, interpretabilidad y eliminación de ruido sin sesgo humano.

Salida:
- data/features/features_seleccionadas.parquet (dataset curado para modelado)
- data/features/selected_features_report.json (metadatos y trazabilidad)
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("select_features")


class FeatureSelector:
    """
    Selector de características mediante eliminación recursiva backward guiada por PR-AUC.
    """

    BASE_NUMERIC_COLS = [
        "score", "monto", "b", "c", "d", "e", "f", "h", "l", "m", "q"
    ]
    BASE_CATEGORICAL_COLS = ["a", "g", "n", "o", "p", "is_na_bc"]

    def __init__(
        self,
        base_dataset_path: Path = Path("data/processed/dataset_base.parquet"),
        features_path: Path = Path("data/features/features_transaccionales.parquet"),
        random_state: int = 42,
    ) -> None:
        self.base_path = base_dataset_path
        self.features_path = features_path
        self.random_state = random_state

        logger.info(f"Cargando dataset base desde: {self.base_path}")
        self.df_base = pd.read_parquet(self.base_path)
        logger.info(f"Cargando features candidatas desde: {self.features_path}")
        self.df_feat = pd.read_parquet(self.features_path)

        assert "transaction_id" in self.df_feat.columns, "transaction_id ausente en features."
        self.merged_df = self.df_base.merge(self.df_feat, on="transaction_id")

        self.train_mask = self.merged_df["split"] == "train"
        self.val_mask = self.merged_df["split"] == "val"

        self.y_train = self.merged_df.loc[self.train_mask, "fraude"].values
        self.y_val = self.merged_df.loc[self.val_mask, "fraude"].values

        # Asegurar formato categórico para LightGBM
        for cat in self.BASE_CATEGORICAL_COLS:
            if cat in self.merged_df.columns:
                self.merged_df[cat] = self.merged_df[cat].astype("category")

        self.base_features = [
            c for c in (self.BASE_NUMERIC_COLS + self.BASE_CATEGORICAL_COLS)
            if c in self.merged_df.columns
        ]
        self.candidate_features = [
            c for c in self.df_feat.columns if c != "transaction_id"
        ]
        logger.info(
            f"Dataset cargado. Filas train: {np.sum(self.train_mask):,}, "
            f"val: {np.sum(self.val_mask):,}. Candidatas iniciales: {len(self.candidate_features)}"
        )

    def _fit_and_eval(
        self,
        active_candidates: List[str],
        n_estimators: int = 100,
        learning_rate: float = 0.05,
    ) -> Tuple[float, Dict[str, float]]:
        """Ajusta LightGBM determinista y evalúa PR-AUC con Feature Importances (Gain)."""
        feature_cols = self.base_features + active_candidates
        X_train = self.merged_df.loc[self.train_mask, feature_cols]
        X_val = self.merged_df.loc[self.val_mask, feature_cols]

        model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=5,
            learning_rate=learning_rate,
            subsample=1.0,
            colsample_bytree=1.0,
            importance_type="gain",
            random_state=self.random_state,
            verbose=-1,
            n_jobs=-1,
        )
        model.fit(X_train, self.y_train)

        probs = model.predict_proba(X_val)[:, 1]
        precision, recall, _ = precision_recall_curve(self.y_val, probs)
        pr_auc = float(auc(recall, precision))

        importances = dict(zip(feature_cols, model.feature_importances_))
        return pr_auc, importances

    def run_selection(
        self,
        tolerance: float = 0.0002,
        min_features: int = 15,
    ) -> Tuple[List[str], Dict]:
        """
        Ejecuta Backward Recursive Feature Elimination con tolerancia controlada.
        """
        t0 = time.time()
        logger.info("=== INICIANDO SELECCIÓN OPTIMIZADA DE VARIABLES ===")

        # 1. Rendimiento baseline con 0 candidatas (solo base cruda)
        base_pr, _ = self._fit_and_eval([])
        logger.info(f"PR-AUC Baseline Crudo (17 features base): {base_pr:.4f}")

        # 2. Rendimiento inicial con todas las candidatas
        active_cands = list(self.candidate_features)
        current_pr, current_imp = self._fit_and_eval(active_cands)
        logger.info(
            f"Paso 0: Pool Completo ({len(active_cands)} candidatas) -> PR-AUC: {current_pr:.4f} "
            f"(Delta: {current_pr - base_pr:+.4f})"
        )

        history = []
        step = 1

        # 3. Poda iterativa backward
        while len(active_cands) > min_features:
            # Ordenar candidatos activos por Gain
            cands_sorted = sorted(active_cands, key=lambda c: current_imp.get(c, 0.0))
            worst_cand = cands_sorted[0]
            worst_gain = current_imp.get(worst_cand, 0.0)

            # Probar eliminar la de menor Gain
            test_cands = [c for c in active_cands if c != worst_cand]
            new_pr, new_imp = self._fit_and_eval(test_cands)
            delta = new_pr - current_pr

            # Aceptamos removerla si no degrada significativamente (> -tolerance)
            if delta >= -tolerance:
                active_cands = test_cands
                current_imp = new_imp
                # Si la remoción mejoró el score, actualizamos la referencia
                current_pr = max(current_pr, new_pr)
                logger.info(
                    f"Paso {step:2d}: - Removida {worst_cand:22s} (Gain: {worst_gain:7.1f}) "
                    f"-> PR-AUC: {new_pr:.4f} (Delta: {delta:+.5f}) [Restan: {len(active_cands)}]"
                )
                history.append({
                    "step": step,
                    "removed_feature": worst_cand,
                    "gain_before_removal": worst_gain,
                    "pr_auc": new_pr,
                    "delta": delta,
                })
                step += 1
            else:
                # Probar si la segunda peor puede removerse sin daño
                if len(cands_sorted) > 1:
                    second_worst = cands_sorted[1]
                    second_gain = current_imp.get(second_worst, 0.0)
                    test_cands_2 = [c for c in active_cands if c != second_worst]
                    new_pr_2, new_imp_2 = self._fit_and_eval(test_cands_2)
                    delta_2 = new_pr_2 - current_pr

                    if delta_2 >= -tolerance:
                        active_cands = test_cands_2
                        current_imp = new_imp_2
                        current_pr = max(current_pr, new_pr_2)
                        logger.info(
                            f"Paso {step:2d} (alt): - Removida {second_worst:22s} (Gain: {second_gain:7.1f}) "
                            f"-> PR-AUC: {new_pr_2:.4f} (Delta: {delta_2:+.5f}) [Restan: {len(active_cands)}]"
                        )
                        history.append({
                            "step": step,
                            "removed_feature": second_worst,
                            "gain_before_removal": second_gain,
                            "pr_auc": new_pr_2,
                            "delta": delta_2,
                        })
                        step += 1
                        continue

                logger.info(
                    f"Criterio de parada alcanzado en paso {step}: remover '{worst_cand}' "
                    f"degrada PR-AUC en {delta:+.5f} (umbral: -{tolerance})."
                )
                break

        # 4. Verificación final oficial
        final_pr, final_imp = self._fit_and_eval(active_cands, n_estimators=100, learning_rate=0.05)
        elapsed = time.time() - t0

        logger.info("=== VERIFICACIÓN FINAL OFICIAL ===")
        logger.info(f"PR-AUC Final Curado: {final_pr:.4f} (Delta vs Baseline: {final_pr - base_pr:+.4f})")
        logger.info(f"Total Features Seleccionadas: {len(active_cands)} de {len(self.candidate_features)}")
        logger.info(f"Tiempo de ejecución: {elapsed:.2f}s")

        report = {
            "baseline_pr_auc": base_pr,
            "final_pr_auc": final_pr,
            "delta_pr_auc": final_pr - base_pr,
            "initial_candidate_count": len(self.candidate_features),
            "selected_feature_count": len(active_cands),
            "selected_features": active_cands,
            "feature_gains": {c: final_imp.get(c, 0.0) for c in active_cands},
            "history": history,
            "elapsed_seconds": round(elapsed, 2),
        }
        return active_cands, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Selector de características transaccionales para detección de fraude."
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
        help="Ruta al parquet de features transaccionales completas.",
    )
    parser.add_argument(
        "--output_selected_parquet",
        type=Path,
        default=Path("data/features/features_seleccionadas.parquet"),
        help="Ruta de salida para la tabla curada de features seleccionadas.",
    )
    parser.add_argument(
        "--output_report_json",
        type=Path,
        default=Path("data/features/selected_features_report.json"),
        help="Ruta de salida para el reporte JSON de selección.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0002,
        help="Tolerancia máxima de degradación de PR-AUC para permitir remoción.",
    )
    args = parser.parse_args()

    selector = FeatureSelector(
        base_dataset_path=args.input_base,
        features_path=args.input_features,
    )
    selected_features, report = selector.run_selection(tolerance=args.tolerance)

    # Exportar dataset curado
    df_curated = selector.df_feat[["transaction_id"] + selected_features]
    args.output_selected_parquet.parent.mkdir(parents=True, exist_ok=True)
    df_curated.to_parquet(args.output_selected_parquet, engine="pyarrow", index=False)
    logger.info(f"Dataset curado exportado en: {args.output_selected_parquet}")

    # Exportar reporte JSON
    with open(args.output_report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Reporte de selección exportado en: {args.output_report_json}")


if __name__ == "__main__":
    main()
