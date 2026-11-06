"""
Contenedor unificado de estimadores de anomalías y su serialización binaria (.joblib).

Incluye:
- PCA (95% varianza acumulada): pca_q, pca_magnitud, pca_t2
- Isolation Forest Perpendicular (ortogonal) y Diagonal (hiperplanos rotados)
- Local Outlier Factor (20k muestra): Manhattan (k=20) y Euclidiana (k=60)
- LightGBM Ensamble Sencillo: score_anomalia_consolidado
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor

from src.models.estimators.preprocessing import (
    FrequencyPreprocessor,
    HybridIForestPreprocessor,
)

logger = logging.getLogger("anomaly_bundle")


class RotatedHyperplaneIsolationForest(BaseEstimator):
    """
    Isolation Forest con cortes diagonales mediante rotaciones ortogonales uniformes R ~ O(d).
    """

    def __init__(
        self,
        n_estimators: int = 150,
        n_rotations: int = 5,
        max_samples: int = 256,
        max_features: float = 0.8,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        self.n_estimators = n_estimators
        self.n_rotations = max(1, n_rotations)
        self.max_samples = max_samples
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.forests_: List[IsolationForest] = []
        self.rotations_: List[np.ndarray] = []

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "RotatedHyperplaneIsolationForest":
        rng = np.random.RandomState(self.random_state)
        n_features = X.shape[1]
        trees_per_rotation = max(1, self.n_estimators // self.n_rotations)

        self.forests_ = []
        self.rotations_ = []

        for _ in range(self.n_rotations):
            gaussian_matrix = rng.randn(n_features, n_features)
            q, r = np.linalg.qr(gaussian_matrix)
            d = np.diagonal(r)
            ph = d / np.abs(d)
            rotation = q * ph
            X_rot = X @ rotation

            sub_seed = rng.randint(0, 1_000_000)
            forest = IsolationForest(
                n_estimators=trees_per_rotation,
                max_samples=self.max_samples,
                max_features=self.max_features,
                random_state=sub_seed,
                n_jobs=self.n_jobs,
            )
            forest.fit(X_rot)
            self.forests_.append(forest)
            self.rotations_.append(rotation)

        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        scores = np.zeros(X.shape[0], dtype=float)
        for forest, rotation in zip(self.forests_, self.rotations_):
            scores += forest.score_samples(X @ rotation)
        return scores / len(self.forests_)


class AnomalyPipelineBundle:
    """
    Orquestador que encapsula preprocesadores, estimadores one-class,
    vectores de referencia de percentiles y el modelo LightGBM consolidador.
    """

    INDIVIDUAL_SCORE_COLS = [
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
    ]

    def __init__(
        self,
        pca_variance: float = 0.95,
        iforest_n_estimators: int = 150,
        iforest_max_samples: int = 256,
        lof_sample_size: int = 20000,
        ocsvm_n_components: int = 300,
        seed: int = 42,
    ) -> None:
        self.pca_variance = pca_variance
        self.iforest_n_estimators = iforest_n_estimators
        self.iforest_max_samples = iforest_max_samples
        self.lof_sample_size = lof_sample_size
        self.ocsvm_n_components = ocsvm_n_components
        self.seed = seed

        self.freq_preprocessor_: Optional[FrequencyPreprocessor] = None
        self.hybrid_preprocessor_: Optional[HybridIForestPreprocessor] = None

        self.pca_: Optional[PCA] = None
        self.iforest_perp_: Optional[IsolationForest] = None
        self.iforest_diag_: Optional[RotatedHyperplaneIsolationForest] = None
        self.lof_manhattan_: Optional[LocalOutlierFactor] = None
        self.lof_euclidean_: Optional[LocalOutlierFactor] = None
        self.nystroem_: Optional[Nystroem] = None
        self.ocsvm_nu_003_: Optional[SGDOneClassSVM] = None
        self.ocsvm_nu_005_: Optional[SGDOneClassSVM] = None
        self.gmm_k3_: Optional[GaussianMixture] = None
        self.gmm_k5_: Optional[GaussianMixture] = None

        self.ref_sorted_scores_: Dict[str, np.ndarray] = {}
        self.lgbm_consolidator_: Optional[lgb.LGBMClassifier] = None

    @staticmethod
    def _to_percentile(raw_scores: np.ndarray, sorted_ref: np.ndarray) -> np.ndarray:
        n_ref = len(sorted_ref)
        pct = np.searchsorted(sorted_ref, raw_scores) / n_ref
        return np.clip(pct, 0.0, 1.0)

    def _compute_raw_scores(
        self,
        x_freq: np.ndarray,
        x_hybrid: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        # 1. PCA (3 indicadores)
        coords = self.pca_.transform(x_freq)
        residual = x_freq - self.pca_.inverse_transform(coords)
        eigenvalues = self.pca_.explained_variance_

        raw_pca_q = np.sum(residual ** 2, axis=1)
        raw_pca_mag = np.sum(coords ** 2, axis=1)
        raw_pca_t2 = np.sum(coords ** 2 / eigenvalues, axis=1)

        # 2. Isolation Forest Perpendicular y Diagonal
        raw_if_perp = -self.iforest_perp_.score_samples(x_hybrid)
        raw_if_diag = -self.iforest_diag_.score_samples(x_hybrid)

        # 3. LOF Manhattan y Euclidiana
        raw_lof_man = -self.lof_manhattan_.score_samples(x_freq)
        raw_lof_euc = -self.lof_euclidean_.score_samples(x_freq)

        # 4. OCSVM Nystroem Kernel RBF (nu=0.03 y nu=0.05)
        x_nystroem = self.nystroem_.transform(x_freq)
        raw_ocsvm_003 = -self.ocsvm_nu_003_.decision_function(x_nystroem)
        raw_ocsvm_005 = -self.ocsvm_nu_005_.decision_function(x_nystroem)

        # 5. Gaussian Mixture Models (k=3 y k=5)
        raw_gmm_k3 = -self.gmm_k3_.score_samples(x_freq)
        raw_gmm_k5 = -self.gmm_k5_.score_samples(x_freq)

        return {
            "pca_q": raw_pca_q,
            "pca_magnitud": raw_pca_mag,
            "pca_t2": raw_pca_t2,
            "score_iforest_perpendicular": raw_if_perp,
            "score_iforest_diagonal": raw_if_diag,
            "score_lof_manhattan": raw_lof_man,
            "score_lof_euclidean": raw_lof_euc,
            "score_ocsvm_rbf_nu_003": raw_ocsvm_003,
            "score_ocsvm_rbf_nu_005": raw_ocsvm_005,
            "score_gmm_k3": raw_gmm_k3,
            "score_gmm_k5": raw_gmm_k5,
        }

    def fit(self, df_train: pd.DataFrame) -> "AnomalyPipelineBundle":
        """
        Ajusta los preprocesadores y modelos one-class estrictamente sobre legítimos de train,
        y entrena el consolidador LightGBM sobre los percentiles resultantes de train.
        """
        if "fraude" not in df_train.columns:
            raise ValueError("df_train debe contener la columna 'fraude'.")

        legit_mask = (df_train["fraude"] == 0).to_numpy()
        df_legit = df_train.loc[legit_mask]
        y_train = df_train["fraude"].to_numpy()

        logger.info(f"Ajustando preprocesadores sobre {len(df_legit):,} transacciones legítimas de train...")
        self.freq_preprocessor_ = FrequencyPreprocessor().fit(df_legit)
        self.hybrid_preprocessor_ = HybridIForestPreprocessor().fit(df_legit)

        x_freq_legit = self.freq_preprocessor_.transform(df_legit)
        x_hybrid_legit = self.hybrid_preprocessor_.transform(df_legit)

        # 1. PCA (95% varianza)
        logger.info(f"Ajustando PCA ({self.pca_variance:.0%} varianza acumulada)...")
        self.pca_ = PCA(n_components=self.pca_variance, svd_solver="full", random_state=self.seed)
        self.pca_.fit(x_freq_legit)

        # 2. Isolation Forest Perpendicular (Ortogonal)
        logger.info("Ajustando Isolation Forest Perpendicular (ortogonal)...")
        self.iforest_perp_ = IsolationForest(
            n_estimators=self.iforest_n_estimators,
            max_samples=self.iforest_max_samples,
            max_features=0.8,
            random_state=self.seed,
            n_jobs=-1,
        )
        self.iforest_perp_.fit(x_hybrid_legit)

        # 3. Isolation Forest Diagonal (Rotated Hyperplanes)
        logger.info("Ajustando Isolation Forest Diagonal (cortes rotados)...")
        self.iforest_diag_ = RotatedHyperplaneIsolationForest(
            n_estimators=self.iforest_n_estimators,
            n_rotations=5,
            max_samples=self.iforest_max_samples,
            max_features=0.8,
            random_state=self.seed,
            n_jobs=-1,
        )
        self.iforest_diag_.fit(x_hybrid_legit)

        # 4. Muestra determinista para LOF
        rng = np.random.RandomState(self.seed)
        n_sample = min(self.lof_sample_size, len(x_freq_legit))
        ref_idx = rng.choice(len(x_freq_legit), size=n_sample, replace=False)
        x_lof_ref = x_freq_legit[ref_idx]

        logger.info(f"Ajustando LOF Manhattan (p=1, k=20) sobre {n_sample:,} legítimos...")
        self.lof_manhattan_ = LocalOutlierFactor(
            n_neighbors=20,
            metric="minkowski",
            p=1,
            algorithm="kd_tree",
            leaf_size=30,
            novelty=True,
            n_jobs=-1,
        )
        self.lof_manhattan_.fit(x_lof_ref)

        logger.info(f"Ajustando LOF Euclidiana (p=2, k=60) sobre {n_sample:,} legítimos...")
        self.lof_euclidean_ = LocalOutlierFactor(
            n_neighbors=60,
            metric="minkowski",
            p=2,
            algorithm="kd_tree",
            leaf_size=30,
            novelty=True,
            n_jobs=-1,
        )
        self.lof_euclidean_.fit(x_lof_ref)

        # 5. OCSVM con Nystroem Kernel RBF (nu=0.03 y nu=0.05)
        logger.info(f"Ajustando Nystroem ({self.ocsvm_n_components} componentes RBF)...")
        self.nystroem_ = Nystroem(
            kernel="rbf",
            gamma=None,
            n_components=self.ocsvm_n_components,
            random_state=self.seed,
            n_jobs=-1,
        )
        self.nystroem_.fit(x_freq_legit)
        x_nystroem_legit = self.nystroem_.transform(x_freq_legit)

        logger.info("Ajustando SGDOneClassSVM (nu=0.03, precision)...")
        self.ocsvm_nu_003_ = SGDOneClassSVM(
            nu=0.03,
            fit_intercept=True,
            max_iter=1000,
            tol=1e-3,
            random_state=self.seed,
        )
        self.ocsvm_nu_003_.fit(x_nystroem_legit)

        logger.info("Ajustando SGDOneClassSVM (nu=0.05, recall)...")
        self.ocsvm_nu_005_ = SGDOneClassSVM(
            nu=0.05,
            fit_intercept=True,
            max_iter=1000,
            tol=1e-3,
            random_state=self.seed,
        )
        self.ocsvm_nu_005_.fit(x_nystroem_legit)

        # 6. GMM con k=3 y k=5 (full covariance)
        logger.info("Ajustando Gaussian Mixture Model (k=3, full covariance)...")
        self.gmm_k3_ = GaussianMixture(
            n_components=3,
            covariance_type="full",
            max_iter=100,
            random_state=self.seed,
        )
        self.gmm_k3_.fit(x_freq_legit)

        logger.info("Ajustando Gaussian Mixture Model (k=5, full covariance)...")
        self.gmm_k5_ = GaussianMixture(
            n_components=5,
            covariance_type="full",
            max_iter=100,
            random_state=self.seed,
        )
        self.gmm_k5_.fit(x_freq_legit)

        # 7. Calcular scores en todo train y guardar distribuciones de referencia legítimas
        logger.info("Calculando distribuciones de referencia de percentiles en train...")
        x_freq_train = self.freq_preprocessor_.transform(df_train)
        x_hybrid_train = self.hybrid_preprocessor_.transform(df_train)
        raw_train_dict = self._compute_raw_scores(x_freq_train, x_hybrid_train)

        self.ref_sorted_scores_ = {}
        pct_train_dict = {}
        for col, raw_vals in raw_train_dict.items():
            sorted_legit = np.sort(raw_vals[legit_mask])
            self.ref_sorted_scores_[col] = sorted_legit
            pct_train_dict[col] = self._to_percentile(raw_vals, sorted_legit)

        # 6. Ajustar LightGBM consolidador sobre los 7 percentiles en train
        logger.info("Ajustando LightGBM simple para score_anomalia_consolidado...")
        X_lgb_train = pd.DataFrame(pct_train_dict)[self.INDIVIDUAL_SCORE_COLS]
        self.lgbm_consolidator_ = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=self.seed,
            verbose=-1,
            n_jobs=-1,
        )
        self.lgbm_consolidator_.fit(X_lgb_train, y_train)
        logger.info("Entrenamiento de todos los estimadores finalizado con éxito.")
        return self

    def predict_scores_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforma un DataFrame arbitrario y devuelve todas las columnas de anomalía
        normalizadas como percentiles en [0.0, 1.0] junto con el score consolidado.
        """
        x_freq = self.freq_preprocessor_.transform(df)
        x_hybrid = self.hybrid_preprocessor_.transform(df)
        raw_dict = self._compute_raw_scores(x_freq, x_hybrid)

        out_data = {}
        if "transaction_id" in df.columns:
            out_data["transaction_id"] = df["transaction_id"].astype("string").values

        for col in self.INDIVIDUAL_SCORE_COLS:
            out_data[col] = self._to_percentile(raw_dict[col], self.ref_sorted_scores_[col])

        out_df = pd.DataFrame(out_data)
        X_lgb = out_df[self.INDIVIDUAL_SCORE_COLS]
        out_df["score_anomalia_consolidado"] = self.lgbm_consolidator_.predict_proba(X_lgb)[:, 1]

        # Interacción oficial de exposición de riesgo anómalo
        if "monto" in df.columns:
            log_monto = np.log1p(np.maximum(df["monto"].to_numpy(dtype=float), 0.0))
            out_df["anomalia_x_monto"] = out_df["score_iforest_perpendicular"] * log_monto

        return out_df

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)
        logger.info(f"Bundle de modelos serializado en: {path}")

    @staticmethod
    def load(path: Path) -> "AnomalyPipelineBundle":
        if not path.exists():
            raise FileNotFoundError(f"No se encontró el archivo binario de modelos en: {path.resolve()}")
        return joblib.load(path)
