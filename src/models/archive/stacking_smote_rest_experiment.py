"""Targeted SMOTE Experiment on Minority Countries (REST):
Doubles the fraud observations for minority countries (g not in ['BR', 'AR']) in train.
Evaluates the Global Tri-Model Architecture (LightGBM + XGBoost + Random Forest + Meta-LR L2).
Slice analysis on val_search:
  - Overall PR-AUC
  - Brasil PR-AUC (g == 'BR')
  - Argentina PR-AUC (g == 'AR')
  - Rest of Countries PR-AUC (g not in ['BR', 'AR'])
Comparison against H4 Champion (without SMOTE).
"""
from __future__ import annotations

import json
from pathlib import Path
import os
import random
import sys
import time
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import xgboost as xgb

from src.models.supervised import (
    calibrate, cost_curve, logit, metrics, policy_metrics, pr_auc,
    validate_development, validation_blocks,
)
from src.models.train import load_data, lgb_model

BASE_CATEGORICAL_COLS = ["a", "g", "n", "o", "p"]
BASE_NUMERIC_COLS = ["score", "monto", "b", "c", "d", "e", "f", "h", "l", "m", "q"]


def generate_synthetic_rest_frauds(
    train_df: pd.DataFrame,
    numeric_cols: List[str],
    cat_cols: List[str],
    n_synthetic: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generates synthetic fraud observations targeted strictly at REST countries."""
    rng = np.random.default_rng(seed)
    rest_mask = (~train_df["g"].isin(["BR", "AR"])) & (train_df["fraude"] == 1)
    fraud_rest = train_df.loc[rest_mask].copy().reset_index(drop=True)

    k_samples = len(fraud_rest)
    if k_samples < 5:
        return pd.DataFrame()

    if n_synthetic is None:
        n_synthetic = k_samples  # Duplicate: add exactly k_samples

    # Escalar numéricos para distancias euclidianas coherentes
    X_num = fraud_rest[numeric_cols].fillna(fraud_rest[numeric_cols].median()).to_numpy(dtype=float)
    scaler = StandardScaler().fit(X_num)
    X_scaled = scaler.transform(X_num)

    k_neighbors = min(5, k_samples - 1)
    nn = NearestNeighbors(n_neighbors=k_neighbors + 1, metric="euclidean").fit(X_scaled)
    distances, indices = nn.kneighbors(X_scaled)

    synth_rows = []
    for syn_idx in range(n_synthetic):
        i = rng.integers(0, k_samples)
        # Elegir un vecino distinto a sí mismo
        neighbor_col = rng.integers(1, k_neighbors + 1)
        j = indices[i, neighbor_col]

        lam = rng.uniform(0.0, 1.0)

        # Interpolar numéricos
        synth_dict = {}
        for c in numeric_cols:
            val_i = fraud_rest.at[i, c]
            val_j = fraud_rest.at[j, c]
            if pd.isna(val_i) and pd.isna(val_j):
                synth_dict[c] = np.nan
            elif pd.isna(val_i):
                synth_dict[c] = val_j
            elif pd.isna(val_j):
                synth_dict[c] = val_i
            else:
                interp = val_i + lam * (val_j - val_i)
                # Si la columna es un flag binario (0 o 1)
                if c.startswith("is_") or c.startswith("flag_") or c == "es_madrugada":
                    interp = float(round(interp))
                synth_dict[c] = interp

        # Categóricos: heredar de uno de los dos padres
        for c in cat_cols:
            synth_dict[c] = fraud_rest.at[i, c] if rng.random() < 0.5 else fraud_rest.at[j, c]

        synth_dict["transaction_id"] = f"tx_synth_rest_{syn_idx:05d}"
        synth_dict["fraude"] = 1
        synth_dict["split"] = "train"
        synth_rows.append(synth_dict)

    synth_df = pd.DataFrame(synth_rows)
    # Asegurar tipos idénticos
    for c in cat_cols:
        if c in synth_df.columns and c in train_df.columns:
            synth_df[c] = synth_df[c].astype(train_df[c].dtype)
    return synth_df


def run_smote_experiment():
    seed = 42
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    output_dir = ROOT / "models/experiments/exp_smote_rest_minority"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print("EXPERIMENTO: SMOTE DIRIGIDO A PAÍSES MINORITARIOS (REST)")
    print("=" * 85)

    # 1. Cargar datos
    print("\n[1/6] Cargando datos y particiones de desarrollo...")
    base_df = pd.read_parquet(ROOT / "data/processed/dataset_base.parquet", filters=[("split", "in", ["train", "val"])])
    feat_df = pd.read_parquet(ROOT / "data/features/features_transaccionales.parquet")
    anom_df = pd.read_parquet(ROOT / "data/features/scores_anomalias.parquet")

    exclude_anom = ["score_anomalia_consolidado", "anomalia_x_monto", "extended_iforest_x_monto"]
    anom_cols = [c for c in anom_df.columns if c not in ["transaction_id"] + exclude_anom]

    if "score_heuristico_fraude_10" in feat_df.columns:
        feat_df = feat_df.drop(columns=["score_heuristico_fraude_10"])

    df = base_df.merge(feat_df, on="transaction_id").merge(anom_df[["transaction_id"] + anom_cols], on="transaction_id")
    validate_development(df)

    train = df.loc[df.split == "train"].copy().reset_index(drop=True)
    tune, cal, policy = validation_blocks(df.loc[df.split == "val"])

    y_train = train.fraude.to_numpy()
    y_tune = tune.fraude.to_numpy()
    y_cal = cal.fraude.to_numpy()
    y_policy = policy.fraude.to_numpy()

    # Columnas y especificaciones
    lgb_cols = ["score", "o", "g", "a", "es_madrugada", "ratio_d_m", "ratio_monto_l", "prod_b_log_c", "score_texto_nb", "ratio_monto_e", "is_h_cero"]
    best_params_lgb = {
        "learning_rate": 0.03486992722440761,
        "num_leaves": 28,
        "min_child_samples": 100,
        "colsample_bytree": 0.8140488199767426,
        "subsample": 0.7573516799506432,
        "reg_alpha": 0.013018028720806228,
        "reg_lambda": 0.5270037815983998,
        "n_estimators": 253,
    }

    xgb_cols = ["score", "o", "te_j_bayes", "es_madrugada", "ratio_monto_l", "q", "score_x_monto", "score_texto_nb", "c", "diff_m_d", "g", "flag_alto_riesgo_nop"]
    xgb_params = {
        "learning_rate": 0.03880453457276846,
        "max_depth": 5,
        "min_child_weight": 25,
        "colsample_bytree": 0.8896276959161779,
        "subsample": 0.807191755093485,
        "reg_alpha": 0.012021842230622207,
        "reg_lambda": 0.2515850109156839,
        "n_estimators": 180,
        "tree_method": "hist",
        "enable_categorical": True,
        "random_state": seed,
        "n_jobs": 4,
        "verbosity": 0,
    }

    rf_cols = ["score", "o", "te_j_bayes", "es_madrugada", "q", "ratio_monto_c", "ratio_monto_l", "score_texto_nb", "hora_cont", "dia_semana", "n", "is_e_cero", "g", "is_o_missing"]
    rf_params = {
        "max_depth": 11,
        "min_samples_leaf": 35,
        "max_features": 0.5647846214083654,
        "n_estimators": 66,
        "random_state": seed,
        "n_jobs": 4,
    }

    # Unión exacta de variables requeridas por los modelos
    model_features_union = list(dict.fromkeys(lgb_cols + xgb_cols + rf_cols + anom_cols + ["pct_heuristico_fraude"]))
    cat_cols_all = [c for c in BASE_CATEGORICAL_COLS if c in model_features_union]
    num_cols_all = [c for c in model_features_union if c not in cat_cols_all]

    # Contar fraudes en REST en train
    rest_mask_train = (~train["g"].isin(["BR", "AR"])) & (train["fraude"] == 1)
    n_rest_fraud = rest_mask_train.sum()
    print(f"Fraudes reales en REST en train: {n_rest_fraud}")
    print(f"Generando {n_rest_fraud} fraudes sintéticos adicionales para duplicar la masa de fraude de REST a {n_rest_fraud * 2}...")

    # 2. Generar SMOTE sobre train completo para los modelos finales
    synth_train = generate_synthetic_rest_frauds(train, num_cols_all, cat_cols_all, n_synthetic=n_rest_fraud, seed=seed)
    train_aug = pd.concat([train, synth_train], ignore_index=True)
    print(f"Train original: {len(train)} filas -> Train aumentado con SMOTE: {len(train_aug)} filas")

    # Vocabularios categóricos fijos congelados desde train
    cat_dtypes = {}
    for cat in BASE_CATEGORICAL_COLS:
        if cat in df.columns:
            cats = sorted(train[cat].dropna().astype(str).unique())
            cat_dtypes[cat] = pd.CategoricalDtype(categories=cats)

    ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    cat_present = [c for c in BASE_CATEGORICAL_COLS if c in df.columns]
    ord_enc.fit(train[cat_present].astype(str))

    # Formatos categóricos para XGB y RF
    def format_frame(in_df):
        out_xgb = in_df.copy()
        for cat, dtype in cat_dtypes.items():
            if cat in out_xgb.columns:
                out_xgb[cat] = out_xgb[cat].astype(str).astype(dtype)

        out_rf = in_df.copy()
        if cat_present:
            out_rf[cat_present] = ord_enc.transform(out_rf[cat_present].astype(str))

        for c in rf_cols:
            if c in out_rf.columns:
                if out_rf[c].isna().any():
                    med = train[c].median()
                    out_rf[c] = out_rf[c].fillna(med)

        return out_xgb, out_rf

    train_xgb_orig, train_rf_orig = format_frame(train)
    train_xgb_aug, train_rf_aug = format_frame(train_aug)
    tune_xgb, tune_rf = format_frame(tune)
    cal_xgb, cal_rf = format_frame(cal)
    policy_xgb, policy_rf = format_frame(policy)

    # 3. Generar probabilidades Out-Of-Fold (OOF) con CV estricto sin Data Leakage
    print("\n[3/6] Generando OOF con 5-Fold CV aplicando SMOTE estrictamente dentro de cada fold de train...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    p_lgb_oof = np.zeros(len(train), dtype=float)
    p_xgb_oof = np.zeros(len(train), dtype=float)
    p_rf_oof = np.zeros(len(train), dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y_train), 1):
        fold_train_raw = train.iloc[tr_idx].copy().reset_index(drop=True)
        # SMOTE solo sobre el train del fold
        synth_fold = generate_synthetic_rest_frauds(fold_train_raw, num_cols_all, cat_cols_all, seed=seed + fold)
        fold_train_aug = pd.concat([fold_train_raw, synth_fold], ignore_index=True)

        tr_xgb_f_aug, tr_rf_f_aug = format_frame(fold_train_aug)
        va_xgb_f, va_rf_f = format_frame(train.iloc[va_idx])

        y_tr_aug = fold_train_aug.fraude.to_numpy()

        # LGB
        m_lgb_f = lgb_model(best_params_lgb, seed, 4).fit(tr_xgb_f_aug[lgb_cols], y_tr_aug)
        p_lgb_oof[va_idx] = m_lgb_f.predict_proba(va_xgb_f[lgb_cols])[:, 1]

        # XGB
        m_xgb_f = xgb.XGBClassifier(**xgb_params).fit(tr_xgb_f_aug[xgb_cols], y_tr_aug)
        p_xgb_oof[va_idx] = m_xgb_f.predict_proba(va_xgb_f[xgb_cols])[:, 1]

        # RF
        m_rf_f = RandomForestClassifier(**rf_params).fit(tr_rf_f_aug[rf_cols], y_tr_aug)
        p_rf_oof[va_idx] = m_rf_f.predict_proba(va_rf_f[rf_cols])[:, 1]

    print(f"OOF PR-AUC -> LGB: {pr_auc(y_train, p_lgb_oof):.5f} | XGB: {pr_auc(y_train, p_xgb_oof):.5f} | RF: {pr_auc(y_train, p_rf_oof):.5f}")

    # 4. Ajustar Modelos Finales de Nivel 1 en todo el train aumentado con SMOTE
    print("\n[4/6] Entrenando Modelos Finales de Nivel 1 con SMOTE...")
    y_train_aug = train_aug.fraude.to_numpy()
    m_lgb_final = lgb_model(best_params_lgb, seed, 4).fit(train_xgb_aug[lgb_cols], y_train_aug)
    m_xgb_final = xgb.XGBClassifier(**xgb_params).fit(train_xgb_aug[xgb_cols], y_train_aug)
    m_rf_final = RandomForestClassifier(**rf_params).fit(train_rf_aug[rf_cols], y_train_aug)

    p_lgb_tune = m_lgb_final.predict_proba(tune_xgb[lgb_cols])[:, 1]
    p_xgb_tune = m_xgb_final.predict_proba(tune_xgb[xgb_cols])[:, 1]
    p_rf_tune = m_rf_final.predict_proba(tune_rf[rf_cols])[:, 1]

    p_lgb_cal = m_lgb_final.predict_proba(cal_xgb[lgb_cols])[:, 1]
    p_xgb_cal = m_xgb_final.predict_proba(cal_xgb[xgb_cols])[:, 1]
    p_rf_cal = m_rf_final.predict_proba(cal_rf[rf_cols])[:, 1]

    p_lgb_policy = m_lgb_final.predict_proba(policy_xgb[lgb_cols])[:, 1]
    p_xgb_policy = m_xgb_final.predict_proba(policy_xgb[xgb_cols])[:, 1]
    p_rf_policy = m_rf_final.predict_proba(policy_rf[rf_cols])[:, 1]

    # 5. Construcción de Meta-Dataset y Ajuste de Metamodelo L2
    print("\n[5/6] Construyendo Meta-Dataset de 18 columnas y entrenando Metamodelo L2...")
    eps = 1e-7

    def build_meta(p_lgb_v, p_xgb_v, p_rf_v, part_df):
        d = {
            "p_lgb": p_lgb_v,
            "logit_lgb": np.log(np.clip(p_lgb_v, eps, 1 - eps) / (1 - np.clip(p_lgb_v, eps, 1 - eps))),
            "p_xgb": p_xgb_v,
            "logit_xgb": np.log(np.clip(p_xgb_v, eps, 1 - eps) / (1 - np.clip(p_xgb_v, eps, 1 - eps))),
            "p_rf": p_rf_v,
            "logit_rf": np.log(np.clip(p_rf_v, eps, 1 - eps) / (1 - np.clip(p_rf_v, eps, 1 - eps))),
        }
        for col in anom_cols:
            d[col] = part_df[col].to_numpy(dtype=float)
        d["pct_heuristico_fraude"] = part_df["pct_heuristico_fraude"].to_numpy(dtype=float)
        return pd.DataFrame(d)

    X_meta_train = build_meta(p_lgb_oof, p_xgb_oof, p_rf_oof, train)
    X_meta_tune = build_meta(p_lgb_tune, p_xgb_tune, p_rf_tune, tune)
    X_meta_cal = build_meta(p_lgb_cal, p_xgb_cal, p_rf_cal, cal)
    X_meta_policy = build_meta(p_lgb_policy, p_xgb_policy, p_rf_policy, policy)

    scaler = StandardScaler().fit(X_meta_train)
    m_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=seed).fit(
        scaler.transform(X_meta_train), y_train
    )

    p_meta_tune = m_lr.predict_proba(scaler.transform(X_meta_tune))[:, 1]
    p_meta_cal = m_lr.predict_proba(scaler.transform(X_meta_cal))[:, 1]
    p_meta_policy = m_lr.predict_proba(scaler.transform(X_meta_policy))[:, 1]

    # 6. Evaluación Comparativa Global y Desglose por Sector Geográfico
    print("\n[6/6] Evaluando métricas globales y desglose por sector geográfico en val_search...")
    br_mask = (tune["g"] == "BR").to_numpy()
    ar_mask = (tune["g"] == "AR").to_numpy()
    rest_mask = (~tune["g"].isin(["BR", "AR"])).to_numpy()

    # Métricas Globales
    m_glob = metrics(y_tune, p_meta_tune)
    pr_br = pr_auc(y_tune[br_mask], p_meta_tune[br_mask])
    pr_ar = pr_auc(y_tune[ar_mask], p_meta_tune[ar_mask])
    pr_rest = pr_auc(y_tune[rest_mask], p_meta_tune[rest_mask])

    # Calibración en val_policy
    calibrator = LogisticRegression(C=1e6, max_iter=1000, random_state=seed).fit(
        logit(p_meta_cal), y_cal
    )
    p_meta_po_cal = calibrate(p_meta_policy, calibrator)
    m_policy_cal = metrics(y_policy, p_meta_po_cal)

    c5 = cost_curve(y_policy, p_meta_po_cal, policy.monto, fp_cost=5.0, loss_fraction=1.0, fn_cost=200.0)
    b5 = c5.loc[c5["fp"] <= 100].iloc[-1] if not c5.loc[c5["fp"] <= 100].empty else c5.iloc[0]

    cm = cost_curve(y_policy, p_meta_po_cal, policy.monto, fp_cost=5.0, loss_fraction=1.0, fn_cost=None)
    bm = cm.loc[cm["fp"] <= 100].iloc[-1] if not cm.loc[cm["fp"] <= 100].empty else cm.iloc[0]

    # Línea base H4 para comparación
    baseline_h4 = {
        "pr_auc_global": 0.48887,
        "pr_auc_br": 0.49819,
        "pr_auc_ar": 0.47274,
        "pr_auc_rest": 0.38776,
        "cost_5_200": 20500.0,
        "cost_monto": 4432.0,
    }

    print("\n" + "=" * 90)
    print("RESULTADOS COMPARATIVOS: H4 ORIGINAL vs H7 (CON SMOTE EN PAÍSES MINORITARIOS)")
    print("=" * 90)
    print(f"* PR-AUC GLOBAL (val_search):")
    print(f"  H4 Original (sin SMOTE) : {baseline_h4['pr_auc_global']:.5f}")
    print(f"  H7 con SMOTE en REST    : {m_glob['pr_auc']:.5f} (Delta: {m_glob['pr_auc'] - baseline_h4['pr_auc_global']:+.5f})")

    print(f"\n* DESGLOSE POR SECTOR GEOGRÁFICO EN VAL_SEARCH:")
    print(f"  Brasil (`BR`)   -> H7 SMOTE: {pr_br:.5f} vs H4 Original: {baseline_h4['pr_auc_br']:.5f} (Delta: {pr_br - baseline_h4['pr_auc_br']:+.5f})")
    print(f"  Argentina (`AR`)-> H7 SMOTE: {pr_ar:.5f} vs H4 Original: {baseline_h4['pr_auc_ar']:.5f} (Delta: {pr_ar - baseline_h4['pr_auc_ar']:+.5f})")
    print(f"  Resto (`REST`)  -> H7 SMOTE: {pr_rest:.5f} vs H4 Original: {baseline_h4['pr_auc_rest']:.5f} (Delta: {pr_rest - baseline_h4['pr_auc_rest']:+.5f})")

    print(f"\n* POLÍTICAS OPERATIVAS EN VAL_POLICY:")
    print(f"  Costo Fijo ($5FP+200FN) : ${b5['cost']:.0f} (H4: ${baseline_h4['cost_5_200']:.0f}) | FP:{b5['fp']}, TP:{b5['tp']}")
    print(f"  Costo Monto             : ${bm['cost']:.0f} (H4: ${baseline_h4['cost_monto']:.0f})")

    # Guardar resumen en JSON
    summary_data = {
        "global_metrics": {
            "pr_auc_val_search": m_glob["pr_auc"],
            "ap_val_search": m_glob["average_precision"],
            "roc_auc_val_search": m_glob["roc_auc"],
            "pr_auc_val_policy_cal": m_policy_cal["pr_auc"],
            "cost_5_200": float(b5["cost"]),
            "cost_monto": float(bm["cost"]),
        },
        "slice_metrics": {
            "BR": {"pr_auc_smote": float(pr_br), "pr_auc_h4": baseline_h4["pr_auc_br"], "delta": float(pr_br - baseline_h4["pr_auc_br"])},
            "AR": {"pr_auc_smote": float(pr_ar), "pr_auc_h4": baseline_h4["pr_auc_ar"], "delta": float(pr_ar - baseline_h4["pr_auc_ar"])},
            "REST": {"pr_auc_smote": float(pr_rest), "pr_auc_h4": baseline_h4["pr_auc_rest"], "delta": float(pr_rest - baseline_h4["pr_auc_rest"])},
        },
        "synthetic_samples_generated": int(n_rest_fraud),
    }
    with open(output_dir / "smote_rest_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    print(f"\nResultados y reporte guardados en: {output_dir}")
    return summary_data


if __name__ == "__main__":
    run_smote_experiment()
