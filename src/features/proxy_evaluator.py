"""
Módulo de evaluación rápida y agnóstica de features (Modelo Proxy).

Permite cuantificar la ganancia marginal objetiva (Delta PR-AUC) de cualquier
conjunto de variables candidatas sobre el conjunto de validación OOT,
antes de incorporarlas al dataset definitivo de features.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("proxy_evaluator")


@dataclass
class EvaluationReport:
    """Contenedor de resultados de evaluación de un experimento de features."""
    experiment_name: str
    baseline_pr_auc: float
    candidate_pr_auc: float
    delta_pr_auc: float
    baseline_roc_auc: float
    candidate_roc_auc: float
    delta_roc_auc: float
    evaluated_features: List[str]
    feature_importances: Dict[str, float]
    verdict: str  # "APORTA POSITIVO", "NEUTRO / INSIGNIFICANTE", "DEGRADA / RUIDO"


class FeatureProxyEvaluator:
    """
    Evaluador agnóstico de features mediante un modelo proxy determinista y ultrarrápido (LightGBM).
    
    Ajusta estrictamente sobre la partición 'train' y evalúa en 'val' para evitar cualquier
    forma de data leakage.
    """

    DEFAULT_BASE_NUMERIC_COLS = [
        "score", "monto", "a", "b", "d", "e", "f", "h", "l", "m", "n", "q", "r", "s"
    ]
    DEFAULT_BASE_CATEGORICAL_COLS = ["g", "o", "p"]

    def __init__(
        self,
        base_dataset_path: Path = Path("data/processed/dataset_base.parquet"),
        target_col: str = "fraude",
        split_col: str = "split",
        random_state: int = 42,
    ) -> None:
        """
        Inicializa el evaluador cargando el dataset base oficial y calculando el baseline.

        Args:
            base_dataset_path: Ruta a 'data/processed/dataset_base.parquet'.
            target_col: Nombre de la variable objetivo binaria.
            split_col: Nombre de la columna de partición temporal ('train', 'val', 'test').
            random_state: Semilla para reproducibilidad estricta.
        """
        if not base_dataset_path.exists():
            raise FileNotFoundError(f"No se encontró el dataset base en: {base_dataset_path.resolve()}")

        self.df_base = pd.read_parquet(base_dataset_path)
        self.target_col = target_col
        self.split_col = split_col
        self.random_state = random_state

        # Máscaras de partición OOT
        self.train_mask = self.df_base[self.split_col] == "train"
        self.val_mask = self.df_base[self.split_col] == "val"

        self.y_train = self.df_base.loc[self.train_mask, self.target_col].values
        self.y_val = self.df_base.loc[self.val_mask, self.target_col].values

        # Filtrar columnas base existentes
        self.base_numeric = [c for c in self.DEFAULT_BASE_NUMERIC_COLS if c in self.df_base.columns]
        self.base_categorical = [c for c in self.DEFAULT_BASE_CATEGORICAL_COLS if c in self.df_base.columns]
        self.base_features = self.base_numeric + self.base_categorical

        # Cálculo inmediato del rendimiento Baseline de referencia
        self.baseline_pr_auc, self.baseline_roc_auc, _ = self._fit_and_score(
            X_df=self.df_base,
            feature_cols=self.base_features,
            categorical_cols=self.base_categorical,
        )
        logger.info(
            f"Evaluador inicializado. Baseline PR-AUC: {self.baseline_pr_auc:.4f} | "
            f"ROC-AUC: {self.baseline_roc_auc:.4f} (en {np.sum(self.val_mask):,} filas de val)"
        )

    def _get_proxy_model(self) -> lgb.LGBMClassifier:
        """Configuración fija y liviana del modelo proxy LightGBM."""
        return lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            verbose=-1,
            n_jobs=-1,
        )

    def _fit_and_score(
        self,
        X_df: pd.DataFrame,
        feature_cols: List[str],
        categorical_cols: List[str],
    ) -> Tuple[float, float, Optional[Dict[str, float]]]:
        """Entrena en train y evalúa métricas en val."""
        X_train = X_df.loc[self.train_mask, feature_cols].copy()
        X_val = X_df.loc[self.val_mask, feature_cols].copy()

        # Coaccionar categóricas a tipo category de pandas para soporte nativo de LightGBM
        for cat in categorical_cols:
            if cat in X_train.columns:
                X_train[cat] = X_train[cat].astype("category")
                X_val[cat] = X_val[cat].astype("category")

        model = self._get_proxy_model()
        model.fit(X_train, self.y_train)

        val_probs = model.predict_proba(X_val)[:, 1]

        # Métricas de validación
        precision, recall, _ = precision_recall_curve(self.y_val, val_probs)
        pr_auc = float(auc(recall, precision))
        roc_auc = float(roc_auc_score(self.y_val, val_probs))

        # Importancia relativa de variables (gain)
        importances = dict(zip(feature_cols, model.feature_importances_))
        return pr_auc, roc_auc, importances

    def evaluate(
        self,
        candidate_features_df: pd.DataFrame,
        candidate_cols: List[str],
        experiment_name: str = "experimento_candidato",
        min_improvement_threshold: float = 0.001,
    ) -> EvaluationReport:
        """
        Evalúa si un conjunto arbitrario de columnas candidatas aporta señal real.

        Args:
            candidate_features_df: DataFrame con las features candidatas (debe contener
                                   'transaction_id' para merge determinista o coincidir en índice).
            candidate_cols: Lista de nombres de las nuevas columnas a evaluar.
            experiment_name: Etiqueta descriptiva del experimento.
            min_improvement_threshold: Mínimo incremento de PR-AUC para considerar veredicto positivo.

        Returns:
            EvaluationReport: Resumen completo de métricas, deltas y veredicto.
        """
        logger.info(f"Evaluando experimento '{experiment_name}' con {len(candidate_cols)} feature(s)...")

        # Integrar features candidatas con dataset base por transaction_id si está presente
        if "transaction_id" in candidate_features_df.columns:
            merged_df = self.df_base.merge(
                candidate_features_df[["transaction_id"] + candidate_cols],
                on="transaction_id",
                how="left",
            )
        else:
            merged_df = pd.concat([self.df_base, candidate_features_df[candidate_cols]], axis=1)

        full_features = self.base_features + candidate_cols
        cand_pr_auc, cand_roc_auc, importances = self._fit_and_score(
            X_df=merged_df,
            feature_cols=full_features,
            categorical_cols=self.base_categorical,
        )

        delta_pr = cand_pr_auc - self.baseline_pr_auc
        delta_roc = cand_roc_auc - self.baseline_roc_auc

        # Veredicto cualitativo
        if delta_pr >= min_improvement_threshold:
            verdict = "APORTA POSITIVO (Recomendado para producción)"
        elif delta_pr <= -min_improvement_threshold:
            verdict = "DEGRADA / RUIDO (Desaconsejado)"
        else:
            verdict = "NEUTRO / INSIGNIFICANTE"

        # Filtrar importancias solo de las variables candidatas evaluadas
        cand_importances = {col: importances.get(col, 0.0) for col in candidate_cols}

        report = EvaluationReport(
            experiment_name=experiment_name,
            baseline_pr_auc=self.baseline_pr_auc,
            candidate_pr_auc=cand_pr_auc,
            delta_pr_auc=delta_pr,
            baseline_roc_auc=self.baseline_roc_auc,
            candidate_roc_auc=cand_roc_auc,
            delta_roc_auc=delta_roc,
            evaluated_features=candidate_cols,
            feature_importances=cand_importances,
            verdict=verdict,
        )
        self.print_report(report)
        return report

    @staticmethod
    def print_report(rep: EvaluationReport) -> None:
        """Imprime un reporte en consola formateado para diagnóstico rápido."""
        sep = "=" * 78
        print("\n" + sep)
        print(f"       RESULTADO DEL MODELO PROXY: {rep.experiment_name.upper()}")
        print(sep)
        print(f"  • PR-AUC Baseline:    {rep.baseline_pr_auc:.4f}")
        print(f"  • PR-AUC Candidato:   {rep.candidate_pr_auc:.4f}")
        sign = "+" if rep.delta_pr_auc >= 0 else ""
        print(f"  • Δ PR-AUC:           {sign}{rep.delta_pr_auc:.4f}")
        print(f"  • ROC-AUC Baseline:   {rep.baseline_roc_auc:.4f}")
        print(f"  • ROC-AUC Candidato:  {rep.candidate_roc_auc:.4f}")
        print(f"  • Veredicto:          [{rep.verdict}]")
        print("\n  • Importancia de features candidatas (split gain / splits count):")
        for feat, imp in rep.feature_importances.items():
            print(f"      - {feat:<28}: {imp:>8.1f}")
        print(sep + "\n")


def main() -> None:
    """CLI para ejecutar la evaluación rápida de un archivo Parquet con features candidatas."""
    parser = argparse.ArgumentParser(
        description="Modelo Proxy para evaluación rápida y agnóstica de features en validación OOT."
    )
    parser.add_argument(
        "--candidate_parquet",
        type=Path,
        default=None,
        help="Ruta al archivo Parquet que contiene las features a testear (debe contener transaction_id).",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=None,
        help="Nombres específicos de columnas a testear dentro del parquet candidato. Si se omite, testea todas menos transaction_id.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="experimento_cli",
        help="Nombre descriptivo para la corrida de evaluación.",
    )
    args = parser.parse_args()

    evaluator = FeatureProxyEvaluator()

    if args.candidate_parquet and args.candidate_parquet.exists():
        df_cand = pd.read_parquet(args.candidate_parquet)
        if args.features:
            test_cols = args.features
        else:
            test_cols = [c for c in df_cand.columns if c not in ("transaction_id", "split", "fraude")]

        evaluator.evaluate(
            candidate_features_df=df_cand,
            candidate_cols=test_cols,
            experiment_name=args.experiment_name,
        )
    else:
        logger.info("Modo de verificación del evaluador proxy: baseline calculado correctamente.")


if __name__ == "__main__":
    main()
