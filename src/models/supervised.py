"""Reusable, train-only preprocessing and evaluation for Contract C."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc, average_precision_score, brier_score_loss, log_loss,
    precision_recall_curve, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_NUMERIC = ["score", "monto", "b", "c", "d", "e", "f", "h", "l", "m", "q"]
BASE_CATEGORICAL = ["a", "g", "n", "o", "p", "is_na_bc"]
METADATA = {"transaction_id", "split", "fraude", "fecha", "origen_bloque", "es_bloque_2"}


def check_ids(df):
    ids = df["transaction_id"]
    if ids.isna().any() or ids.duplicated().any() or not ids.map(lambda x: isinstance(x, str)).all():
        raise ValueError("transaction_id debe ser string, único y no nulo.")


def merge_features(base, features):
    """Require exact coverage and equality of overlapping raw features."""
    check_ids(base)
    check_ids(features)
    if set(base.transaction_id) != set(features.transaction_id):
        raise ValueError("Features con IDs faltantes o extra respecto de la población solicitada.")
    forbidden = (set(features) & METADATA) - {"transaction_id"}
    if forbidden:
        raise ValueError(f"Metadatos prohibidos como features: {sorted(forbidden)}")
    aligned = features.set_index("transaction_id").loc[base.transaction_id]
    for col in (set(features) & set(base)) - {"transaction_id"}:
        left, right = base[col].reset_index(drop=True), aligned[col].reset_index(drop=True)
        if not ((left == right) | (left.isna() & right.isna())).all():
            raise ValueError(f"Columna compartida inconsistente: {col}")
    new = [c for c in features if c not in base]
    # Natural missing values (e.g. b/c outage) are handled by train-only
    # preprocessing. Missing IDs are rejected above, independently of nulls.
    return base.merge(features[["transaction_id"] + new], on="transaction_id", how="left", validate="one_to_one", sort=False)


def validate_development(df):
    check_ids(df)
    if not set(df.split).issubset({"train", "val"}):
        raise ValueError("Desarrollo admite únicamente train y val.")
    for split in ("train", "val"):
        part = df.loc[df.split == split]
        if part.empty or set(part.fraude.unique()) != {0, 1}:
            raise ValueError(f"{split} debe tener ambas clases, sin etiquetas nulas.")
    dates = pd.to_datetime(df.fecha, utc=True, errors="raise")
    if dates.isna().any() or dates[df.split == "train"].max() > dates[df.split == "val"].min():
        raise ValueError("Particiones temporales inválidas.")
    amounts = df.monto.to_numpy(dtype=float)
    if not np.isfinite(amounts).all() or (amounts < 0).any():
        raise ValueError("monto debe ser finito y no negativo.")


def validation_blocks(val):
    """Disjoint chronological blocks: search 60%, calibration 20%, policy 20%."""
    val = val.sort_values(["fecha", "transaction_id"], kind="stable")
    dates = pd.to_datetime(val.fecha, utc=True)
    first, second = dates.iloc[int(len(val) * .6)], dates.iloc[int(len(val) * .8)]
    blocks = [val.loc[dates < first], val.loc[(dates >= first) & (dates < second)], val.loc[dates >= second]]
    if any(len(b) < 20 or set(b.fraude.unique()) != {0, 1} for b in blocks):
        raise ValueError("Val insuficiente para tres bloques temporales con ambas clases (mínimo 20 filas/bloque).")
    return blocks


class FrameEncoder:
    """Freeze category vocabularies from train; unseen categories become missing."""
    def fit(self, frame):
        self.columns = list(frame.columns)
        self.categories = {}
        for c in self.columns:
            if c in BASE_CATEGORICAL or not pd.api.types.is_numeric_dtype(frame[c]):
                self.categories[c] = sorted(frame[c].dropna().astype(str).unique())
        return self

    def transform(self, frame):
        result = frame[self.columns].copy()
        for c in self.columns:
            if c in self.categories:
                result[c] = pd.Categorical(result[c].astype("string"), categories=self.categories[c])
            else:
                result[c] = pd.to_numeric(result[c], errors="raise").astype(float).replace([np.inf, -np.inf], np.nan)
        return result


def logistic_model(x, seed):
    cat = list(x.select_dtypes(include="category").columns)
    num = [c for c in x if c not in cat]
    prep = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler())]), num),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)), ("encode", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])
    return Pipeline([("preprocess", prep), ("model", LogisticRegression(max_iter=3000, random_state=seed))])


def pr_auc(y, p):
    precision, recall, _ = precision_recall_curve(y, p)
    return float(auc(recall, precision))


def metrics(y, p):
    return {"pr_auc": pr_auc(y, p), "average_precision": float(average_precision_score(y, p)),
            "roc_auc": float(roc_auc_score(y, p)), "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p, labels=[0, 1]))}


def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p)).reshape(-1, 1)


def calibrate(p, calibrator):
    return p if calibrator is None else calibrator.predict_proba(logit(p))[:, 1]


def cost_curve(y, p, amounts, fp_cost, loss_fraction=1., fn_cost=None):
    """Exact O(n log n) threshold sweep, keeping tied probabilities together."""
    y, p, amounts = np.asarray(y), np.asarray(p), np.asarray(amounts, dtype=float)
    if not (np.isfinite(fp_cost) and fp_cost >= 0 and 0 <= loss_fraction <= 1):
        raise ValueError("Costo FP >= 0 y fracción de pérdida en [0,1].")
    if len(y) == 0 or not np.isfinite(p).all() or not np.isfinite(amounts).all() or (amounts < 0).any():
        raise ValueError("Probabilidades y montos deben ser finitos y montos no negativos.")
    order = np.argsort(-p, kind="stable")
    y, p, amounts = y[order], p[order], amounts[order]
    ends = np.r_[np.flatnonzero(np.diff(p) != 0), len(p) - 1]
    losses = y * (amounts * loss_fraction if fn_cost is None else fn_cost)
    fp = np.r_[0, np.cumsum(y == 0)[ends]]
    tp = np.r_[0, np.cumsum(y == 1)[ends]]
    missed = np.r_[losses.sum(), losses.sum() - np.cumsum(losses)[ends]]
    total = np.maximum(missed, 0) + fp_cost * fp
    return pd.DataFrame({"threshold": np.r_[np.nextafter(1., 2.), p[ends]], "fp": fp,
                         "tp": tp, "fn": int(y.sum()) - tp, "tn": int((y == 0).sum()) - fp,
                         "cost": total, "savings_vs_allow_all": losses.sum() - total})


def policy_metrics(y, p, amounts, threshold, fp_cost, loss_fraction, fn_cost=None):
    y, p, amounts = np.asarray(y), np.asarray(p), np.asarray(amounts)
    blocked = p >= threshold
    fp, fn = int(((y == 0) & blocked).sum()), int(((y == 1) & ~blocked).sum())
    losses = amounts * loss_fraction if fn_cost is None else np.full(len(y), fn_cost)
    cost = float(losses[(y == 1) & ~blocked].sum() + fp * fp_cost)
    tp = int(((y == 1) & blocked).sum())
    return {"threshold": threshold, "fp": fp, "fn": fn,
            "tp": tp, "tn": int(((y == 0) & ~blocked).sum()),
            "precision": tp / (tp + fp) if tp + fp else 0.,
            "recall": tp / (tp + fn) if tp + fn else 0.,
            "cost": cost, "savings_vs_allow_all": float(losses[y == 1].sum() - cost)}


class AveragedModels:
    """Equal-weight probability ensemble; members are already train-fitted."""
    def __init__(self, members):
        self.members = members

    def predict_proba(self, frame):
        p = np.mean([m["model"].predict_proba(m["encoder"].transform(frame))[:, 1]
                     for m in self.members], axis=0)
        return np.column_stack([1 - p, p])


class IdentityEncoder:
    def transform(self, frame):
        return frame
