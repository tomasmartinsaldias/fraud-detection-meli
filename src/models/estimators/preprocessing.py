"""
Preprocesadores con anti-leakage estricto para modelos de detección de anomalías:
1. FrequencyPreprocessor: Para PCA y LOF (imputación mediana + Frequency Encoding en categóricas + StandardScaler).
2. HybridIForestPreprocessor: Para Isolation Forest (imputación mediana + One-Hot K-1 en baja cardinalidad + Frequency en alta cardinalidad + StandardScaler).
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DEFAULT_NUMERIC_COLS = [
    "score", "monto", "a", "b", "d", "e", "f", "h", "l", "m", "n", "q", "r", "s"
]
DEFAULT_CATEGORICAL_COLS = ["g", "j", "o", "p"]
LOW_CARDINALITY_COLS = ["g", "o", "p"]
HIGH_CARDINALITY_COLS = ["j"]


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Codificador por frecuencia relativa observada en la muestra de entrenamiento legítima.
    Los valores nulos reciben la frecuencia de nulos en legítimos; categorías no vistas reciben 0.0.
    """

    def __init__(self, cols: Optional[List[str]] = None) -> None:
        self.cols = cols or DEFAULT_CATEGORICAL_COLS
        self.freq_maps_: Dict[str, pd.Series] = {}
        self.null_freqs_: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame, y: Optional[np.ndarray] = None) -> "FrequencyEncoder":
        self.freq_maps_ = {}
        self.null_freqs_ = {}
        n_rows = len(df)
        for col in self.cols:
            series = df[col]
            counts = series.value_counts(dropna=True)
            self.freq_maps_[col] = counts / n_rows
            self.null_freqs_[col] = float(series.isna().mean())
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        blocks = []
        for col in self.cols:
            series = df[col]
            freq_map = self.freq_maps_[col]
            null_f = self.null_freqs_[col]

            missing = series.isna()
            unknown = ~missing & ~series.isin(freq_map.index)

            encoded = series.map(freq_map).astype(float)
            encoded.loc[missing] = null_f
            encoded.loc[unknown] = 0.0
            blocks.append(encoded.to_numpy()[:, None])
        return np.column_stack(blocks)


class FrequencyPreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocesador completo para PCA y LOF:
    - Imputación mediana de numéricas.
    - Frequency Encoding de todas las categóricas (g, j, o, p).
    - Estandarización conjunta con StandardScaler.
    """

    def __init__(
        self,
        numeric_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
    ) -> None:
        self.numeric_cols = numeric_cols or DEFAULT_NUMERIC_COLS
        self.categorical_cols = categorical_cols or DEFAULT_CATEGORICAL_COLS
        self.imputer_: Optional[SimpleImputer] = None
        self.freq_encoder_: Optional[FrequencyEncoder] = None
        self.scaler_: Optional[StandardScaler] = None

    def fit(self, df_legit: pd.DataFrame, y: Optional[np.ndarray] = None) -> "FrequencyPreprocessor":
        # The official cleaned dataset drops legacy r/s. Freeze the available
        # schema at fit; all other requested columns remain mandatory.
        self.numeric_cols_ = [c for c in self.numeric_cols
                              if c not in {'r', 's'} or c in df_legit.columns]
        x_num = df_legit[self.numeric_cols_].to_numpy(dtype=float, na_value=np.nan)
        self.imputer_ = SimpleImputer(strategy="median").fit(x_num)
        x_num_imp = self.imputer_.transform(x_num)

        self.freq_encoder_ = FrequencyEncoder(cols=self.categorical_cols).fit(df_legit)
        x_cat_freq = self.freq_encoder_.transform(df_legit)

        x_full = np.column_stack([x_num_imp, x_cat_freq])
        self.scaler_ = StandardScaler().fit(x_full)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        x_num = df[self.numeric_cols_].to_numpy(dtype=float, na_value=np.nan)
        x_num_imp = self.imputer_.transform(x_num)
        x_cat_freq = self.freq_encoder_.transform(df)
        x_full = np.column_stack([x_num_imp, x_cat_freq])
        return self.scaler_.transform(x_full)


class HybridIForestPreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocesador híbrido para Isolation Forest:
    - Imputación mediana de numéricas.
    - Baja cardinalidad (g, o, p): One-Hot Encoding con drop='first' (genera K - 1 columnas).
    - Alta cardinalidad (j): Frequency Encoding.
    - Estandarización conjunta con StandardScaler.
    """

    def __init__(
        self,
        numeric_cols: Optional[List[str]] = None,
        low_card_cols: Optional[List[str]] = None,
        high_card_cols: Optional[List[str]] = None,
    ) -> None:
        self.numeric_cols = numeric_cols or DEFAULT_NUMERIC_COLS
        self.low_card_cols = low_card_cols or LOW_CARDINALITY_COLS
        self.high_card_cols = high_card_cols or HIGH_CARDINALITY_COLS
        self.imputer_: Optional[SimpleImputer] = None
        self.ohe_: Optional[OneHotEncoder] = None
        self.freq_encoder_: Optional[FrequencyEncoder] = None
        self.scaler_: Optional[StandardScaler] = None

    def fit(self, df_legit: pd.DataFrame, y: Optional[np.ndarray] = None) -> "HybridIForestPreprocessor":
        # The official cleaned dataset drops legacy r/s. Freeze the available
        # schema at fit; all other requested columns remain mandatory.
        self.numeric_cols_ = [c for c in self.numeric_cols
                              if c not in {'r', 's'} or c in df_legit.columns]
        x_num = df_legit[self.numeric_cols_].to_numpy(dtype=float, na_value=np.nan)
        self.imputer_ = SimpleImputer(strategy="median").fit(x_num)
        x_num_imp = self.imputer_.transform(x_num)

        # One-Hot con drop='first' para generar K - 1 columnas por categoría
        self.ohe_ = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
        low_card_str = df_legit[self.low_card_cols].fillna("__MISSING__").astype(str)
        self.ohe_.fit(low_card_str)
        x_ohe = self.ohe_.transform(low_card_str)

        # Frequency Encoding para alta cardinalidad (j)
        self.freq_encoder_ = FrequencyEncoder(cols=self.high_card_cols).fit(df_legit)
        x_high_freq = self.freq_encoder_.transform(df_legit)

        x_full = np.column_stack([x_num_imp, x_ohe, x_high_freq])
        self.scaler_ = StandardScaler().fit(x_full)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        x_num = df[self.numeric_cols_].to_numpy(dtype=float, na_value=np.nan)
        x_num_imp = self.imputer_.transform(x_num)

        low_card_str = df[self.low_card_cols].fillna("__MISSING__").astype(str)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            x_ohe = self.ohe_.transform(low_card_str)

        x_high_freq = self.freq_encoder_.transform(df)
        x_full = np.column_stack([x_num_imp, x_ohe, x_high_freq])
        return self.scaler_.transform(x_full)
