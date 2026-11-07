"""
Pipeline modular de ingeniería de características transaccionales.

Genera variables analíticas validadas previamente mediante el modelo proxy,
respetando de forma estricta las restricciones anti-leakage temporal y el contrato de datos.

Salida oficial: data/features/features_transaccionales.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.naive_bayes import MultinomialNB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("build_features")


def compute_bayesian_monto_ratio(
    df: pd.DataFrame,
    m_weight: float = 30.0,
) -> pd.Series:
    """
    Calcula el ratio de monto normalizado por (a, g) con suavizado bayesiano hacia la media de 'a'.
    
    Restricción Anti-Leakage:
    Las medias a nivel 'a' y a nivel '(a, g)' se calculan EXCLUSIVAMENTE sobre split == 'train'.

    Args:
        df: DataFrame completo con 'transaction_id', 'split', 'monto', 'a' y 'g'.
        m_weight: Peso bayesiano para regularizar celdas con pocas observaciones.

    Returns:
        pd.Series: Serie con el ratio bayesiano alineada al índice de df.
    """
    logger.info("Calculando feature: ratio_monto_ag_bayes...")
    train_mask = df["split"] == "train"
    train_df = df[train_mask]

    # Medias de train
    mean_a = train_df.groupby("a")["monto"].mean().to_dict()
    stats_ag = train_df.groupby(["a", "g"])["monto"].agg(["count", "mean"]).reset_index()
    stats_ag["global_a"] = stats_ag["a"].map(mean_a)
    stats_ag["bayes_mean"] = (
        stats_ag["count"] * stats_ag["mean"] + m_weight * stats_ag["global_a"]
    ) / (stats_ag["count"] + m_weight)

    map_bayes = dict(zip(zip(stats_ag["a"], stats_ag["g"]), stats_ag["bayes_mean"]))

    # Mapeo sobre todo el dataset (train, val, test) usando fallback seguro a media de a
    ratios = [
        monto_val / map_bayes.get((row_a, row_g), mean_a.get(row_a, np.nan))
        for monto_val, row_a, row_g in zip(df["monto"], df["a"], df["g"])
    ]
    return pd.Series(ratios, index=df.index, name="ratio_monto_ag_bayes", dtype=np.float64)


def compute_profile_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula transformaciones no lineales e interacciones de las variables continuas de perfil b y c.
    
    Selección optimizada por el Modelo Proxy:
    - ratio_monto_b: Tensión entre el ticket monetario y la tasa continua b (+1700 gain).
    - prod_b_log_c: Interacción multiplicativa no lineal entre el factor b y log(c + 1) (+1800 gain).
    - ratio_monto_c: Proporción del monto sobre la escala cuantitativa de c (+1400 gain).

    Args:
        df: DataFrame base con 'monto', 'b' y 'c'.

    Returns:
        pd.DataFrame: DataFrame con las 3 features de interacción más potentes.
    """
    logger.info("Calculando transformaciones e interacciones de b y c...")
    feats = pd.DataFrame(index=df.index)

    log_c = np.log1p(df["c"].clip(lower=0)).astype(np.float64)

    # 1. ratio_monto_b: Tensión entre el ticket monetario y la tasa continua b
    feats["ratio_monto_b"] = (df["monto"] / (df["b"] + 0.001)).astype(np.float64)

    # 2. prod_b_log_c: Interacción multiplicativa no lineal entre el factor b y log(c + 1)
    feats["prod_b_log_c"] = (df["b"] * log_c).astype(np.float64)

    # 3. ratio_monto_c: Proporción del monto actual sobre la escala de c
    feats["ratio_monto_c"] = (df["monto"] / (df["c"] + 1.0)).astype(np.float64)

    return feats


def compute_dm_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula transformaciones no lineales e interacciones de las variables d y m.
    
    Selección optimizada y validada mediante el Modelo Proxy (Delta PR-AUC +0.0073 conjunto):
    - ratio_d_m: Proporción d / (m + 1.0). Captura la tensión entre ambas escalas relativas (+2500 a 4000 gain).
    - ratio_monto_m: Normalización monto / (m + 1.0). Relación entre el importe y la magnitud de m (+3100 gain).
    - diff_m_d: Diferencia directa m - d. Identifica discrepancias anómalas entre variables correlacionadas (+4400 gain).

    Args:
        df: DataFrame base con 'monto', 'd' y 'm'.

    Returns:
        pd.DataFrame: DataFrame con las 3 features calculadas.
    """
    logger.info("Calculando transformaciones e interacciones de d y m...")
    feats = pd.DataFrame(index=df.index)

    # 1. ratio_d_m: Escala relativa entre d y m (+1.0 evita división por cero cuando m == 0)
    feats["ratio_d_m"] = (df["d"] / (df["m"] + 1.0)).astype(np.float64)

    # 2. ratio_monto_m: Normalización del ticket respecto a m
    feats["ratio_monto_m"] = (df["monto"] / (df["m"] + 1.0)).astype(np.float64)

    # 3. diff_m_d: Discrepancia directa entre m y d
    feats["diff_m_d"] = (df["m"] - df["d"]).astype(np.float64)

    return feats


def compute_ef_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula transformaciones no lineales e indicadores de las variables e y f.

    Selección y tratamiento alineado a estándares de producción:
    - ratio_monto_f: Ticket normalizado por max(f, 0) + 1.0. Domina el outlier extremo
      (145.274) y penaliza transacciones sin historial (+4600 gain).
    - ratio_monto_e: Ticket normalizado por e + 0.01 (+760 gain).
    - log_f: Compresión logarítmica para mitigar la asimetría pesada de f.
    - is_f_cero_neg: Indicador booleano que aísla el segmento crítico con 12% de fraude.
    - is_e_cero: Indicador booleano de masa puntual en cero (44% de la muestra).

    Args:
        df: DataFrame base con 'monto', 'e' y 'f'.

    Returns:
        pd.DataFrame: DataFrame con las 5 features calculadas.
    """
    logger.info("Calculando transformaciones e interacciones de e y f...")
    feats = pd.DataFrame(index=df.index)

    f_clean = np.maximum(df["f"].fillna(0), 0)
    e_clean = np.maximum(df["e"].fillna(0), 0)

    # 1. ratio_monto_f: Tensión entre monto e historial acumulado f (denominador >= 1.0)
    feats["ratio_monto_f"] = (df["monto"] / (f_clean + 1.0)).astype(np.float64)

    # 2. ratio_monto_e: Tensión entre monto y escala e (denominador >= 0.01)
    feats["ratio_monto_e"] = (df["monto"] / (e_clean + 0.01)).astype(np.float64)

    # 3. log_f: Compresión logarítmica de la cola extrema
    feats["log_f"] = np.log1p(f_clean).astype(np.float64)

    # 4. is_f_cero_neg: Flag de zona de riesgo (f <= 0 o nulo)
    feats["is_f_cero_neg"] = ((df["f"].isna()) | (df["f"] <= 0)).astype(np.int8)

    # 5. is_e_cero: Flag de masa puntual en cero
    feats["is_e_cero"] = (df["e"] == 0).astype(np.int8)

    return feats


def compute_h_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula transformaciones no lineales e indicadores de la variable entera h.

    Selección y tratamiento alineado a estándares de producción:
    - ratio_monto_h: Ticket normalizado por h + 1.0. Tensión entre el importe y la madurez (+1375 gain).
    - is_h_cero: Indicador booleano que aísla el segmento de alto riesgo (h == 0 tiene 9.59% de fraude).
    - log_h: Compresión logarítmica de escala.

    Args:
        df: DataFrame base con 'monto' y 'h'.

    Returns:
        pd.DataFrame: DataFrame con las 3 features calculadas.
    """
    logger.info("Calculando transformaciones e interacciones de h...")
    feats = pd.DataFrame(index=df.index)

    h_val = df["h"].fillna(0)

    # 1. ratio_monto_h: Tensión ticket / escala de h (denominador >= 1.0)
    feats["ratio_monto_h"] = (df["monto"] / (h_val + 1.0)).astype(np.float64)

    # 2. is_h_cero: Flag de inicio de ciclo / sin historial (tasa de fraude 9.6%)
    feats["is_h_cero"] = (h_val == 0).astype(np.int8)

    # 3. log_h: Compresión logarítmica
    feats["log_h"] = np.log1p(np.maximum(h_val, 0)).astype(np.float64)

    return feats


def compute_text_features(
    df: pd.DataFrame,
    max_features: int = 5000,
    min_df: int = 30,
    alpha: float = 2.0,
    n_svd_components: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Computa representaciones estructurales, semánticas y bayesianas de los títulos 'i'.

    Restricciones Anti-Leakage y Estándares de Producción:
    - Capa Estructural: char_len, word_count, avg_word_len, digit_ratio (row-level, sin parámetros aprendidos).
    - Capa Semántica (LSA): TfidfVectorizer y TruncatedSVD ajustados estrictamente sobre split == 'train'
      con k=10 componentes principales optimizados según la regla del codo (+0.0134 PR-AUC).
    - Score Bayesiano (score_texto_nb): Generado en train mediante TimeSeriesSplit cronológico (OOF estricto)
      y proyectado sobre val/test con modelo ajustado sobre el 100% de train.
    - Regularización anti-ruido: min_df=30 (purga palabras con <30 apariciones) y Laplace alpha=2.0.

    Args:
        df: DataFrame base con 'i', 'split', 'fecha' y 'fraude'.

    Returns:
        pd.DataFrame: DataFrame con las 15 features de texto calculadas.
    """
    logger.info("Calculando features de texto sobre columna 'i'...")
    feats = pd.DataFrame(index=df.index)
    i_str = df["i"].fillna("").astype(str)

    # 1. Capa Estructural Objetiva
    feats["char_len"] = i_str.str.len().astype(np.float64)
    words = i_str.str.split()
    feats["word_count"] = words.str.len().astype(np.float64)
    feats["avg_word_len"] = (feats["char_len"] / np.maximum(feats["word_count"], 1.0)).astype(np.float64)
    digits = i_str.str.count(r"\d")
    feats["digit_ratio"] = (digits / np.maximum(feats["char_len"], 1.0)).astype(np.float64)

    # 2. Capa Semántica LSA (TF-IDF + TruncatedSVD)
    train_mask = df["split"] == "train"
    vec_svd = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=min_df,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    X_train_tfidf = vec_svd.fit_transform(i_str[train_mask])

    svd = TruncatedSVD(n_components=n_svd_components, random_state=random_state)
    svd.fit(X_train_tfidf)

    X_all_svd = svd.transform(vec_svd.transform(i_str))
    for j in range(n_svd_components):
        feats[f"text_svd_{j+1}"] = X_all_svd[:, j].astype(np.float64)

    # 3. Score Probabilístico Bayesiano con OOF Temporal Estricto en Train
    train_sub = df[train_mask].sort_values("fecha").copy()
    X_train_text_sorted = train_sub["i"].fillna("").astype(str)
    y_train_sorted = train_sub["fraude"].values

    tscv = TimeSeriesSplit(n_splits=5)
    oof_scores_temporal = np.full(len(train_sub), np.nan)

    for fold, (tr_idx, v_idx) in enumerate(tscv.split(train_sub)):
        vec_fold = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=min_df,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        X_tr = vec_fold.fit_transform(X_train_text_sorted.iloc[tr_idx])
        nb_fold = MultinomialNB(alpha=alpha)
        nb_fold.fit(X_tr, y_train_sorted[tr_idx])
        oof_scores_temporal[v_idx] = nb_fold.predict_proba(
            vec_fold.transform(X_train_text_sorted.iloc[v_idx])
        )[:, 1]

    # Imputar el primer bloque cronológico con el prior global de train
    oof_scores_temporal[np.isnan(oof_scores_temporal)] = y_train_sorted.mean()

    # Mapear scores de train alineados al índice original de df
    train_sub["score_texto_nb"] = oof_scores_temporal
    score_series = pd.Series(index=df.index, dtype=np.float64)
    score_series.loc[train_mask] = train_sub["score_texto_nb"]

    # Fitear modelo en TODO train y predecir en val y test (maneja OOV naturalmente)
    vec_full = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=min_df,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    X_train_full = vec_full.fit_transform(i_str[train_mask])
    nb_full = MultinomialNB(alpha=alpha)
    nb_full.fit(X_train_full, df.loc[train_mask, "fraude"].values)

    test_val_preds = nb_full.predict_proba(vec_full.transform(i_str[~train_mask]))[:, 1]
    score_series.loc[~train_mask] = test_val_preds

    feats["score_texto_nb"] = score_series.astype(np.float64)
    return feats


def compute_category_features(
    df: pd.DataFrame,
    m_te: float = 10.0,
    noise_std: float = 0.01,
    m_monto: float = 20.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Calcula variables analíticas sobre la columna categórica de alta cardinalidad 'j' (categorías de producto).

    Selección optimizada por el Modelo Proxy:
    - te_j_bayes: Target Encoding bayesiano con K-Fold OOF (k=5) y regularización m=10, con jitter gaussiano
      para prevenir memorización espuria en hojas de árboles de decisión (+0.0118 PR-AUC, Gain ~7900).
    - freq_j: Log-frecuencia observada de la categoría en train (log(1 + count)).
    - ratio_monto_j_bayes: Tensión del monto actual sobre la mediana bayesiana esperada para la categoría j.

    Restricción Anti-Leakage:
    - Todas las frecuencias, medianas y distribuciones de target se calculan EXCLUSIVAMENTE sobre split == 'train'.
    - El Target Encoding en train se calcula de forma Out-Of-Fold (OOF).
    - Para val y test, se aplican las estadísticas completas de train con fallback al prior global ante categorías no vistas.

    Args:
        df: DataFrame completo con 'j', 'monto', 'fraude' y 'split'.
        m_te: Peso del prior global en el target encoding bayesiano.
        noise_std: Desviación estándar del ruido gaussiano añadido a los OOF de train.
        m_monto: Peso bayesiano para el cálculo de la mediana esperada del monto en j.
        random_state: Semilla determinista para reproducibilidad estricta.

    Returns:
        pd.DataFrame: DataFrame con las 3 features de categoría ('te_j_bayes', 'freq_j', 'ratio_monto_j_bayes').
    """
    logger.info("Calculando features analíticas de categoría 'j'...")
    feats = pd.DataFrame(index=df.index)

    train_mask = df["split"] == "train"
    train_indices = df[train_mask].index.values
    non_train_indices = df[~train_mask].index.values

    global_fraud_rate = float(df.loc[train_mask, "fraude"].mean())
    global_monto_median = float(df.loc[train_mask, "monto"].median())

    # 1. te_j_bayes: Target Encoding Bayesiano con 5-Fold OOF y Jitter en train
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    te_series = pd.Series(index=df.index, dtype=np.float64)
    rng = np.random.RandomState(random_state)

    for tr_idx, va_idx in cv.split(train_indices):
        tr_df = df.iloc[train_indices[tr_idx]]
        va_df = df.iloc[train_indices[va_idx]]

        stats = tr_df.groupby("j")["fraude"].agg(["count", "mean"])
        smoothed = (stats["count"] * stats["mean"] + m_te * global_fraud_rate) / (stats["count"] + m_te)
        mapping = smoothed.to_dict()

        vals = va_df["j"].map(mapping).fillna(global_fraud_rate).values
        if noise_std > 0:
            vals = vals + rng.normal(0.0, noise_std, size=len(vals))
            vals = np.clip(vals, 0.0, 1.0)
        te_series.iloc[train_indices[va_idx]] = vals

    # Para val y test: aplicar estadísticas completas de train
    full_stats = df.loc[train_mask].groupby("j")["fraude"].agg(["count", "mean"])
    full_smoothed = (full_stats["count"] * full_stats["mean"] + m_te * global_fraud_rate) / (full_stats["count"] + m_te)
    full_mapping = full_smoothed.to_dict()
    te_series.iloc[non_train_indices] = (
        df.loc[non_train_indices, "j"].map(full_mapping).fillna(global_fraud_rate).values
    )
    feats["te_j_bayes"] = te_series.astype(np.float64)

    # 2. freq_j: Log-frecuencia en train
    counts = df.loc[train_mask, "j"].value_counts().to_dict()
    feats["freq_j"] = (
        df["j"].map(counts).fillna(0).apply(lambda x: np.log1p(x)).astype(np.float64)
    )

    # 3. ratio_monto_j_bayes: Monto dividido mediana bayesiana de la categoría
    j_stats = df[train_mask].groupby("j")["monto"].agg(["median", "count"])
    j_stats["bayes_median"] = (
        j_stats["count"] * j_stats["median"] + m_monto * global_monto_median
    ) / (j_stats["count"] + m_monto)
    bayes_median_dict = j_stats["bayes_median"].to_dict()

    feats["ratio_monto_j_bayes"] = [
        float(m / bayes_median_dict.get(j, global_monto_median))
        for m, j in zip(df["monto"], df["j"])
    ]
    feats["ratio_monto_j_bayes"] = feats["ratio_monto_j_bayes"].astype(np.float64)

    return feats


def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula variables temporales analíticas de ciclo continuo a partir de la columna 'fecha'.

    Diseño Continuo y Suave (sin discretización binaria arbitraria):
    - hora_cont: Tiempo continuo fraccional transcurrido del día (hora + min/60 + seg/3600), en [0.0, 24.0).
    - cos_hora: Proyección trigonométrica cos(2*pi*hora_cont / 24). Resuelve la discontinuidad de medianoche
      (23:59 y 00:01 adyacentes en ~ +1.0) y actúa como detector continuo noche/madrugada vs día.
    - sin_hora: Proyección trigonométrica sin(2*pi*hora_cont / 24). Captura el gradiente continuo mañana vs tarde.

    Validación Proxy:
    Aumenta el PR-AUC conjunto a 0.4547 (+0.0244 sobre baseline crudo), con altos índices de split gain
    (hora_cont: ~2226, cos_hora: ~1304, sin_hora: ~610).

    Args:
        df: DataFrame base que contiene la columna 'fecha' (timestamp).

    Returns:
        pd.DataFrame: DataFrame con las 3 variables temporales continuas ('hora_cont', 'cos_hora', 'sin_hora').
    """
    logger.info("Calculando features temporales continuas sobre columna 'fecha'...")
    feats = pd.DataFrame(index=df.index)

    fecha = pd.to_datetime(df["fecha"])
    hora_cont = (
        fecha.dt.hour + fecha.dt.minute / 60.0 + fecha.dt.second / 3600.0
    ).astype(np.float64)

    feats["hora_cont"] = hora_cont
    feats["cos_hora"] = np.cos(2.0 * np.pi * hora_cont / 24.0).astype(np.float64)
    feats["sin_hora"] = np.sin(2.0 * np.pi * hora_cont / 24.0).astype(np.float64)

    return feats


def compute_additional_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula variables candidatas complementarias para que el pool de selección sea exhaustivo:
    - log_l: Compresión logarítmica log1p de l.
    - log_m: Compresión logarítmica log1p de m.
    - ratio_monto_l: Monto normalizado por l (monto / (l + 1)).
    - score_x_monto: Interacción multiplicativa de exposición de riesgo (score * monto).
    - ratio_monto_q: Ticket normalizado por q (monto / (q + 0.01)).
    - flag_alto_riesgo_nop: Flag booleano de coincidencia de segmentos de alto riesgo (n=0, o='N', p='N').
    - is_o_missing: Flag indicador de ausencia en variable o (72% missing).
    - es_madrugada: Flag binario de ventana nocturna optimizada (hora >= 23 o hora <= 5).
    - dia_semana: Índice numérico del día de la semana (0=Lunes, 6=Domingo).
    - log_monto: Compresión logarítmica log1p de monto.
    - monto_centavos: Extracción de la parte decimal / centavos (0 a 99).
    - is_monto_alto: Flag booleano de ticket extremo perteneciente al percentil 99 superior (> 422).
    - log_delta_mismo_prod: Tiempo transcurrido en escala logarítmica log1p(delta_sec) desde la última venta del mismo producto i.
    - burst_mismo_prod_24h: Flag binario de ráfaga del mismo producto en menos de 24 horas.

    Args:
        df: DataFrame base con las columnas requeridas.

    Returns:
        pd.DataFrame: DataFrame con las 12 features candidatas adicionales.
    """
    logger.info("Calculando features candidatas complementarias (l, m, q, score, n, o, p y calendario)...")
    feats = pd.DataFrame(index=df.index)

    # 1. Transformaciones logarítmicas de l y m
    feats["log_l"] = np.log1p(df["l"].clip(lower=0)).astype(np.float64)
    feats["log_m"] = np.log1p(df["m"].clip(lower=0)).astype(np.float64)

    # 2. Interacciones de l, score y q
    feats["ratio_monto_l"] = (df["monto"] / (df["l"] + 1.0)).astype(np.float64)
    feats["score_x_monto"] = (df["score"] * df["monto"]).astype(np.float64)
    feats["ratio_monto_q"] = (df["monto"] / (df["q"].clip(lower=0) + 0.01)).astype(np.float64)

    # 3. Interacciones y flags categóricos de n, o, p
    o_str = df["o"].fillna("MISSING").astype(str)
    feats["flag_alto_riesgo_nop"] = (
        (df["n"] == 0) & (o_str == "N") & (df["p"] == "N")
    ).astype(np.int8)
    feats["is_o_missing"] = df["o"].isna().astype(np.int8)

    # 4. Variables de calendario
    fecha = pd.to_datetime(df["fecha"])
    feats["es_madrugada"] = ((fecha.dt.hour >= 23) | (fecha.dt.hour <= 5)).astype(np.int8)
    feats["dia_semana"] = fecha.dt.weekday.astype(np.float64)

    # 5. Transformaciones intrínsecas de monto
    monto_arr = np.maximum(df["monto"].to_numpy(), 0.0)
    feats["log_monto"] = np.log1p(monto_arr).astype(np.float64)
    feats["monto_centavos"] = np.round((monto_arr - np.floor(monto_arr)) * 100.0).astype(np.float64)
    feats["is_monto_alto"] = (monto_arr > 422.0).astype(np.int8)

    # 6. Variables de velocidad / ráfagas temporales sobre producto i (sin data leakage)
    prev_fecha_i = df.groupby("i")["fecha"].shift(1)
    delta_sec_i = (pd.to_datetime(df["fecha"]) - pd.to_datetime(prev_fecha_i)).dt.total_seconds()
    dias_30_sec = 30.0 * 86400.0
    feats["log_delta_mismo_prod"] = np.log1p(
        delta_sec_i.fillna(dias_30_sec).clip(lower=0.0)
    ).astype(np.float64)
    feats["burst_mismo_prod_24h"] = (delta_sec_i <= 86400.0).fillna(False).astype(np.int8)

    return feats


def compute_heuristic_risk_features(
    df_base: pd.DataFrame,
    df_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcula el Score Heurístico y Porcentaje de Violaciones de Fraude basado en un panel
    ortogonal de 10 reglas de asociación de alto WRAcc / impacto neto:
    - r1 (Perfil/Canal): p == 'N' AND o == 'N'
    - r2 (Inmediatez/Sin historial): ratio_d_m >= 1.0 OR is_f_cero_neg == 1
    - r3 (Tensión Monetaria f): ratio_monto_f > p75 AND is_o_missing == 0
    - r4 (Categoría Crítica): te_j_bayes > p75 AND is_o_missing == 0
    - r5 (Contexto/Semántica): es_madrugada == 1 OR score_texto_nb > p90
    - r6 (Antigüedad Baja l): log_l <= p25 AND is_o_missing == 0
    - r7 (Tensión Monetaria m): ratio_monto_m > p75 AND is_o_missing == 0
    - r8 (Ratio b Anómalo + Joven): b > p90 AND m <= p25
    - r9 (Canal Atípico): a != 4 AND is_o_missing == 0
    - r10 (Discrepancia Temporal): diff_m_d <= p10

    Restricción Anti-Leakage:
    Los percentiles (p10, p25, p75, p90) se calculan EXCLUSIVAMENTE sobre split == 'train'.

    Args:
        df_base: DataFrame base oficial.
        df_features: DataFrame con features previamente computadas.

    Returns:
        pd.DataFrame: DataFrame con 'score_heuristico_fraude_10' y 'pct_heuristico_fraude'.
    """
    logger.info("Calculando features de riesgo heurístico basadas en panel ortogonal de 10 reglas CARs (WRAcc)...")
    feats = pd.DataFrame(index=df_base.index)
    train_mask = df_base["split"] == "train"

    # Umbrales aprendidos estrictamente en train
    th_mf = float(df_features.loc[train_mask, "ratio_monto_f"].quantile(0.75))
    th_te = float(df_features.loc[train_mask, "te_j_bayes"].quantile(0.75))
    th_nb = float(df_features.loc[train_mask, "score_texto_nb"].quantile(0.90))
    th_ll = float(df_features.loc[train_mask, "log_l"].quantile(0.25))
    th_mm = float(df_features.loc[train_mask, "ratio_monto_m"].quantile(0.75))
    th_b = float(df_base.loc[train_mask, "b"].quantile(0.90))
    th_m = float(df_base.loc[train_mask, "m"].quantile(0.25))
    th_diff = float(df_features.loc[train_mask, "diff_m_d"].quantile(0.10))

    r1 = (df_base["p"] == "N") & (df_base["o"] == "N")
    r2 = (df_features["ratio_d_m"] >= 1.0) | (df_features["is_f_cero_neg"] == 1)
    r3 = (df_features["ratio_monto_f"] > th_mf) & (df_features["is_o_missing"] == 0)
    r4 = (df_features["te_j_bayes"] > th_te) & (df_features["is_o_missing"] == 0)
    r5 = (df_features["es_madrugada"] == 1) | (df_features["score_texto_nb"] > th_nb)
    r6 = (df_features["log_l"] <= th_ll) & (df_features["is_o_missing"] == 0)
    r7 = (df_features["ratio_monto_m"] > th_mm) & (df_features["is_o_missing"] == 0)
    r8 = (df_base["b"] > th_b) & (df_base["m"] <= th_m)
    r9 = (df_base["a"] != 4) & (df_features["is_o_missing"] == 0)
    r10 = df_features["diff_m_d"] <= th_diff

    score_10 = (
        r1.astype(np.int8)
        + r2.astype(np.int8)
        + r3.astype(np.int8)
        + r4.astype(np.int8)
        + r5.astype(np.int8)
        + r6.astype(np.int8)
        + r7.astype(np.int8)
        + r8.astype(np.int8)
        + r9.astype(np.int8)
        + r10.astype(np.int8)
    )

    feats["score_heuristico_fraude_10"] = score_10.astype(np.int8)
    feats["pct_heuristico_fraude"] = (score_10 / 10.0).astype(np.float64)

    return feats


def build_transactional_features(
    input_base_path: Path,
    output_features_path: Path,
) -> pd.DataFrame:
    """
    Función orquestadora que carga el dataset base oficial, computa las features
    validadas y las exporta a Parquet con transaction_id como clave primaria.

    Args:
        input_base_path: Ruta a data/processed/dataset_base.parquet.
        output_features_path: Ruta a data/features/features_transaccionales.parquet.

    Returns:
        pd.DataFrame: Tabla generada y exportada.
    """
    if not input_base_path.exists():
        raise FileNotFoundError(f"No se encontró el dataset base en: {input_base_path.resolve()}")

    logger.info(f"Cargando dataset base desde: {input_base_path}")
    df_base = pd.read_parquet(input_base_path)

    # DataFrame receptor con clave primaria obligatoria
    df_features = pd.DataFrame({"transaction_id": df_base["transaction_id"]})

    # 1. Feature de país y tipo de transacción (Suavizado Bayesiano)
    df_features["ratio_monto_ag_bayes"] = compute_bayesian_monto_ratio(df_base)

    # 2. Features de interacción de perfil (b y c)
    interactions = compute_profile_interactions(df_base)
    for col in interactions.columns:
        df_features[col] = interactions[col]

    # 3. Features de interacción matemática de d y m
    dm_interactions = compute_dm_interactions(df_base)
    for col in dm_interactions.columns:
        df_features[col] = dm_interactions[col]

    # 4. Features de interacción matemática de e y f
    ef_interactions = compute_ef_interactions(df_base)
    for col in ef_interactions.columns:
        df_features[col] = ef_interactions[col]

    # 5. Features de interacción matemática de h
    h_interactions = compute_h_interactions(df_base)
    for col in h_interactions.columns:
        df_features[col] = h_interactions[col]

    # 6. Features de texto de la columna i (estructurales, semánticas y bayesianas)
    text_features = compute_text_features(df_base)
    for col in text_features.columns:
        df_features[col] = text_features[col]

    # 7. Features analíticas de categoría j (Target Encoding bayesiano OOF, frecuencia y ratio de monto)
    category_features = compute_category_features(df_base)
    for col in category_features.columns:
        df_features[col] = category_features[col]

    # 8. Features temporales continuas sobre fecha (hora continua y proyecciones cíclicas trigonométricas)
    temporal_features = compute_temporal_features(df_base)
    for col in temporal_features.columns:
        df_features[col] = temporal_features[col]

    # 9. Features candidatas complementarias (l, m, q, score, n, o, p y calendario)
    additional_candidates = compute_additional_candidates(df_base)
    for col in additional_candidates.columns:
        df_features[col] = additional_candidates[col]

    # 10. Feature de riesgo heurístico basada en ensamble de reglas CARs (WRAcc)
    heuristic_feats = compute_heuristic_risk_features(df_base, df_features)
    for col in heuristic_feats.columns:
        df_features[col] = heuristic_feats[col]

    # Validaciones de integridad según Contrato A
    assert len(df_features) == len(df_base), "Error: Se alteró el número de filas."
    assert df_features["transaction_id"].is_unique, "Error: transaction_id duplicado."

    # Exportación atómica a Parquet
    output_features_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_features_path.with_suffix(".parquet.tmp")
    df_features.to_parquet(tmp_path, engine="pyarrow", index=False)
    import os
    os.replace(tmp_path, output_features_path)

    size_mb = output_features_path.stat().st_size / (1024 * 1024)
    logger.info(
        f"Features transaccionales generadas exitosamente en: {output_features_path} "
        f"({len(df_features):,} filas x {df_features.shape[1]} columnas, {size_mb:.2f} MB)"
    )
    return df_features


def main() -> None:
    """CLI para ejecución reproducible."""
    parser = argparse.ArgumentParser(
        description="Generador de features transaccionales para detección de fraude."
    )
    parser.add_argument(
        "--input_base",
        type=Path,
        default=Path("data/processed/dataset_base.parquet"),
        help="Ruta al dataset base procesado.",
    )
    parser.add_argument(
        "--output_features",
        type=Path,
        default=Path("data/features/features_transaccionales.parquet"),
        help="Ruta destino para guardar las features generadas.",
    )
    args = parser.parse_args()

    build_transactional_features(
        input_base_path=args.input_base,
        output_features_path=args.output_features,
    )


if __name__ == "__main__":
    main()
