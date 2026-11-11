"""Country-Segmented Tri-Model Stacking Experiment:
Trains independent Tri-Model ensembles (LightGBM + XGBoost + Random Forest + Meta-LR L2)
specifically for each country sector:
  - Sector 1: Brasil (BR)
  - Sector 2: Argentina (AR)
  - Sector 3: Rest of Countries (REST)
Compares:
  - Global Model (trained on all 105k) vs Segmented Models (specialized per country).
  - Evaluated on val_search overall, and sliced per country (BR, AR, REST).
  - Policy metrics on val_policy.
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
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import xgboost as xgb

from src.models.supervised import (
    calibrate, cost_curve, logit, metrics, policy_metrics, pr_auc,
    validate_development, validation_blocks,
)
from src.models.train import load_data, lgb_model

BASE_CATEGORICAL_COLS = ["a", "g", "n", "o", "p"]
BASE_NUMERIC_COLS = ["score", "monto", "b", "c", "d", "e", "f", "h", "l", "m", "q"]


def assign_country_sector(val):
    if val == "BR":
        return "BR"
    elif val == "AR":
        return "AR"
    else:
        return "REST"


def run_country_segmented_experiment():
    seed = 42
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    output_dir = ROOT / "models/experiments/exp_country_segmented"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print("EXPERIMENTO DE STACKING CONDICIONADO POR PAÍS (BR, AR, REST)")
    print("=" * 85)

    # 1. Cargar datos
    print("\n[1/5] Cargando dataset y configurando sectores geográficos...")
    base_df = pd.read_parquet(ROOT / "data/processed/dataset_base.parquet", filters=[("split", "in", ["train", "val"])])
    feat_df = pd.read_parquet(ROOT / "data/features/features_transaccionales.parquet")
    anom_df = pd.read_parquet(ROOT / "data/features/scores_anomalias.parquet")

    exclude_anom = ["score_anomalia_consolidado", "anomalia_x_monto", "extended_iforest_x_monto"]
    anom_cols = [c for c in anom_df.columns if c not in ["transaction_id"] + exclude_anom]

    if "score_heuristico_fraude_10" in feat_df.columns:
        feat_df = feat_df.drop(columns=["score_heuristico_fraude_10"])

    df = base_df.merge(feat_df, on="transaction_id").merge(anom_df[["transaction_id"] + anom_cols], on="transaction_id")
    validate_development(df)

    # Crear sector
    df["country_sector"] = df["g"].apply(assign_country_sector)

    train = df.loc[df.split == "train"].copy()
    tune, cal, policy = validation_blocks(df.loc[df.split == "val"])

    y_train = train.fraude.to_numpy()
    y_tune = tune.fraude.to_numpy()
    y_cal = cal.fraude.to_numpy()
    y_policy = policy.fraude.to_numpy()

    # Preprocesamiento de formatos para XGBoost y Random Forest
    df_xgb = df.copy()
    for cat in BASE_CATEGORICAL_COLS:
        if cat in df_xgb.columns:
            df_xgb[cat] = df_xgb[cat].astype("category")

    ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df_rf = df.copy()
    cat_present = [c for c in BASE_CATEGORICAL_COLS if c in df_rf.columns]
    df_rf[cat_present] = ord_enc.fit_transform(df_rf[cat_present])

    cand_cols = [c for c in feat_df.columns if c not in ["transaction_id", "pct_heuristico_fraude"]]
    for c in cand_cols + [c for c in BASE_NUMERIC_COLS if c in df_rf.columns]:
        if df_rf[c].isna().any():
            med = df_rf.loc[df_rf.split == "train", c].median()
            df_rf[c] = df_rf[c].fillna(med)

    train_xgb = df_xgb.loc[df.split == "train"]
    tune_xgb = df_xgb.loc[tune.index]
    cal_xgb = df_xgb.loc[cal.index]
    policy_xgb = df_xgb.loc[policy.index]

    train_rf = df_rf.loc[df.split == "train"]
    tune_rf = df_rf.loc[tune.index]
    cal_rf = df_rf.loc[cal.index]
    policy_rf = df_rf.loc[policy.index]

    # Columnas y parámetros oficiales de cada modelo base
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

    # =========================================================================
    # ENFOQUE 1: MODELO GLOBAL CAMPEÓN (H4)
    # =========================================================================
    print("\n[2/5] Ajustando Modelo Global (LGB + XGB + RF + Meta-LR L2)...")
    t0 = time.time()
    # LGB Global
    m_lgb_glob = lgb_model(best_params_lgb, seed, 4).fit(train_xgb[lgb_cols], y_train)
    p_lgb_tu_glob = m_lgb_glob.predict_proba(tune_xgb[lgb_cols])[:, 1]
    p_lgb_ca_glob = m_lgb_glob.predict_proba(cal_xgb[lgb_cols])[:, 1]
    p_lgb_po_glob = m_lgb_glob.predict_proba(policy_xgb[lgb_cols])[:, 1]

    # XGB Global
    m_xgb_glob = xgb.XGBClassifier(**xgb_params).fit(train_xgb[xgb_cols], y_train)
    p_xgb_tu_glob = m_xgb_glob.predict_proba(tune_xgb[xgb_cols])[:, 1]
    p_xgb_ca_glob = m_xgb_glob.predict_proba(cal_xgb[xgb_cols])[:, 1]
    p_xgb_po_glob = m_xgb_glob.predict_proba(policy_xgb[xgb_cols])[:, 1]

    # RF Global
    m_rf_glob = RandomForestClassifier(**rf_params).fit(train_rf[rf_cols], y_train)
    p_rf_tu_glob = m_rf_glob.predict_proba(tune_rf[rf_cols])[:, 1]
    p_rf_ca_glob = m_rf_glob.predict_proba(cal_rf[rf_cols])[:, 1]
    p_rf_po_glob = m_rf_glob.predict_proba(policy_rf[rf_cols])[:, 1]

    # OOF Global
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    p_lgb_oof_glob = np.zeros(len(train), dtype=float)
    p_xgb_oof_glob = np.zeros(len(train), dtype=float)
    p_rf_oof_glob = np.zeros(len(train), dtype=float)

    for tr_idx, va_idx in skf.split(train, y_train):
        m_l = lgb_model(best_params_lgb, seed, 4).fit(train_xgb[lgb_cols].iloc[tr_idx], y_train[tr_idx])
        p_lgb_oof_glob[va_idx] = m_l.predict_proba(train_xgb[lgb_cols].iloc[va_idx])[:, 1]

        m_x = xgb.XGBClassifier(**xgb_params).fit(train_xgb[xgb_cols].iloc[tr_idx], y_train[tr_idx])
        p_xgb_oof_glob[va_idx] = m_x.predict_proba(train_xgb[xgb_cols].iloc[va_idx])[:, 1]

        m_r = RandomForestClassifier(**rf_params).fit(train_rf[rf_cols].iloc[tr_idx], y_train[tr_idx])
        p_rf_oof_glob[va_idx] = m_r.predict_proba(train_rf[rf_cols].iloc[va_idx])[:, 1]

    X_meta_tr_glob = build_meta(p_lgb_oof_glob, p_xgb_oof_glob, p_rf_oof_glob, train)
    X_meta_tu_glob = build_meta(p_lgb_tu_glob, p_xgb_tu_glob, p_rf_tu_glob, tune)
    X_meta_ca_glob = build_meta(p_lgb_ca_glob, p_xgb_ca_glob, p_rf_ca_glob, cal)
    X_meta_po_glob = build_meta(p_lgb_po_glob, p_xgb_po_glob, p_rf_po_glob, policy)

    scaler_glob = StandardScaler().fit(X_meta_tr_glob)
    m_lr_glob = LogisticRegression(C=1.0, max_iter=1000, random_state=seed).fit(
        scaler_glob.transform(X_meta_tr_glob), y_train
    )

    p_meta_tu_glob = m_lr_glob.predict_proba(scaler_glob.transform(X_meta_tu_glob))[:, 1]
    p_meta_ca_glob = m_lr_glob.predict_proba(scaler_glob.transform(X_meta_ca_glob))[:, 1]
    p_meta_po_glob = m_lr_glob.predict_proba(scaler_glob.transform(X_meta_po_glob))[:, 1]
    print(f"Modelo Global entrenado en {time.time() - t0:.1f}s | PR-AUC val_search: {pr_auc(y_tune, p_meta_tu_glob):.5f}")

    # =========================================================================
    # ENFOQUE 2: MODELOS SEGMENTADOS POR PAÍS (BR, AR, REST)
    # =========================================================================
    print("\n[3/5] Entrenando Modelos Especializados por País (BR, AR, REST)...")
    sectors = ["BR", "AR", "REST"]
    sector_models = {}

    p_meta_tu_seg = np.zeros(len(tune), dtype=float)
    p_meta_ca_seg = np.zeros(len(cal), dtype=float)
    p_meta_po_seg = np.zeros(len(policy), dtype=float)

    for sec in sectors:
        print(f"\n--- Ajustando Pipeline Especializado para Sector: {sec} ---")
        tr_mask = (train["country_sector"] == sec).to_numpy()
        tu_mask = (tune["country_sector"] == sec).to_numpy()
        ca_mask = (cal["country_sector"] == sec).to_numpy()
        po_mask = (policy["country_sector"] == sec).to_numpy()

        n_tr = tr_mask.sum()
        n_fraud = y_train[tr_mask].sum()
        print(f"Sector {sec}: {n_tr} filas en train, {n_fraud} fraudes ({n_fraud/n_tr:.2%})")

        # Subsets train
        tr_xgb_s = train_xgb.loc[tr_mask]
        tr_rf_s = train_rf.loc[tr_mask]
        y_tr_s = y_train[tr_mask]

        # Ajuste de modelos base en sector
        # Si un modelo como RF o LGBM requiere min_child_samples menor en REST:
        params_lgb_s = dict(best_params_lgb)
        if sec == "REST":
            params_lgb_s["min_child_samples"] = 20

        m_lgb_s = lgb_model(params_lgb_s, seed, 4).fit(tr_xgb_s[lgb_cols], y_tr_s)
        m_xgb_s = xgb.XGBClassifier(**xgb_params).fit(tr_xgb_s[xgb_cols], y_tr_s)
        m_rf_s = RandomForestClassifier(**rf_params).fit(tr_rf_s[rf_cols], y_tr_s)

        # OOF 5-fold en sector
        skf_s = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        p_lgb_oof_s = np.zeros(n_tr, dtype=float)
        p_xgb_oof_s = np.zeros(n_tr, dtype=float)
        p_rf_oof_s = np.zeros(n_tr, dtype=float)

        for s_tr_idx, s_va_idx in skf_s.split(tr_xgb_s, y_tr_s):
            m_l = lgb_model(params_lgb_s, seed, 4).fit(tr_xgb_s[lgb_cols].iloc[s_tr_idx], y_tr_s[s_tr_idx])
            p_lgb_oof_s[s_va_idx] = m_l.predict_proba(tr_xgb_s[lgb_cols].iloc[s_va_idx])[:, 1]

            m_x = xgb.XGBClassifier(**xgb_params).fit(tr_xgb_s[xgb_cols].iloc[s_tr_idx], y_tr_s[s_tr_idx])
            p_xgb_oof_s[s_va_idx] = m_x.predict_proba(tr_xgb_s[xgb_cols].iloc[s_va_idx])[:, 1]

            m_r = RandomForestClassifier(**rf_params).fit(tr_rf_s[rf_cols].iloc[s_tr_idx], y_tr_s[s_tr_idx])
            p_rf_oof_s[s_va_idx] = m_r.predict_proba(tr_rf_s[rf_cols].iloc[s_va_idx])[:, 1]

        # Inferencia en validación del sector
        tu_xgb_s = tune_xgb.loc[tu_mask]
        tu_rf_s = tune_rf.loc[tu_mask]
        p_lgb_tu_s = m_lgb_s.predict_proba(tu_xgb_s[lgb_cols])[:, 1]
        p_xgb_tu_s = m_xgb_s.predict_proba(tu_xgb_s[xgb_cols])[:, 1]
        p_rf_tu_s = m_rf_s.predict_proba(tu_rf_s[rf_cols])[:, 1]

        ca_xgb_s = cal_xgb.loc[ca_mask]
        ca_rf_s = cal_rf.loc[ca_mask]
        p_lgb_ca_s = m_lgb_s.predict_proba(ca_xgb_s[lgb_cols])[:, 1]
        p_xgb_ca_s = m_xgb_s.predict_proba(ca_xgb_s[xgb_cols])[:, 1]
        p_rf_ca_s = m_rf_s.predict_proba(ca_rf_s[rf_cols])[:, 1]

        po_xgb_s = policy_xgb.loc[po_mask]
        po_rf_s = policy_rf.loc[po_mask]
        p_lgb_po_s = m_lgb_s.predict_proba(po_xgb_s[lgb_cols])[:, 1]
        p_xgb_po_s = m_xgb_s.predict_proba(po_xgb_s[xgb_cols])[:, 1]
        p_rf_po_s = m_rf_s.predict_proba(po_rf_s[rf_cols])[:, 1]

        # Meta-Dataset del sector
        X_meta_tr_s = build_meta(p_lgb_oof_s, p_xgb_oof_s, p_rf_oof_s, train.loc[tr_mask])
        X_meta_tu_s = build_meta(p_lgb_tu_s, p_xgb_tu_s, p_rf_tu_s, tune.loc[tu_mask])
        X_meta_ca_s = build_meta(p_lgb_ca_s, p_xgb_ca_s, p_rf_ca_s, cal.loc[ca_mask])
        X_meta_po_s = build_meta(p_lgb_po_s, p_xgb_po_s, p_rf_po_s, policy.loc[po_mask])

        # Metamodelo LR del sector
        scaler_s = StandardScaler().fit(X_meta_tr_s)
        m_lr_s = LogisticRegression(C=1.0, max_iter=1000, random_state=seed).fit(
            scaler_s.transform(X_meta_tr_s), y_tr_s
        )

        p_meta_tu_seg[tu_mask] = m_lr_s.predict_proba(scaler_s.transform(X_meta_tu_s))[:, 1]
        p_meta_ca_seg[ca_mask] = m_lr_s.predict_proba(scaler_s.transform(X_meta_ca_s))[:, 1]
        p_meta_po_seg[po_mask] = m_lr_s.predict_proba(scaler_s.transform(X_meta_po_s))[:, 1]

        sec_pr = pr_auc(y_tune[tu_mask], p_meta_tu_seg[tu_mask])
        glob_pr_sec = pr_auc(y_tune[tu_mask], p_meta_tu_glob[tu_mask])
        print(f"Sector {sec} -> PR-AUC Modelo Especializado: {sec_pr:.5f} vs Modelo Global: {glob_pr_sec:.5f} (Delta: {sec_pr - glob_pr_sec:+.5f})")

        sector_models[sec] = {
            "n_train": int(n_tr),
            "n_fraud_train": int(n_fraud),
            "pr_auc_specialized": float(sec_pr),
            "pr_auc_global": float(glob_pr_sec),
            "delta": float(sec_pr - glob_pr_sec),
        }

    # =========================================================================
    # EVALUACIÓN GLOBAL Y POLÍTICAS
    # =========================================================================
    print("\n[4/5] Evaluando métricas globales y políticas en validación...")

    # Calibración global de ambos enfoques
    calibrator_glob = LogisticRegression(C=1e6, max_iter=1000, random_state=seed).fit(
        logit(p_meta_ca_glob), y_cal
    )
    p_meta_po_glob_cal = calibrate(p_meta_po_glob, calibrator_glob)

    calibrator_seg = LogisticRegression(C=1e6, max_iter=1000, random_state=seed).fit(
        logit(p_meta_ca_seg), y_cal
    )
    p_meta_po_seg_cal = calibrate(p_meta_po_seg, calibrator_seg)

    m_tu_glob = metrics(y_tune, p_meta_tu_glob)
    m_tu_seg = metrics(y_tune, p_meta_tu_seg)

    m_po_glob = metrics(y_policy, p_meta_po_glob_cal)
    m_po_seg = metrics(y_policy, p_meta_po_seg_cal)

    # Costos en val_policy (FP <= 100)
    c5_glob = cost_curve(y_policy, p_meta_po_glob_cal, policy.monto, fp_cost=5.0, loss_fraction=1.0, fn_cost=200.0)
    b5_glob = c5_glob.loc[c5_glob["fp"] <= 100].iloc[-1] if not c5_glob.loc[c5_glob["fp"] <= 100].empty else c5_glob.iloc[0]

    c5_seg = cost_curve(y_policy, p_meta_po_seg_cal, policy.monto, fp_cost=5.0, loss_fraction=1.0, fn_cost=200.0)
    b5_seg = c5_seg.loc[c5_seg["fp"] <= 100].iloc[-1] if not c5_seg.loc[c5_seg["fp"] <= 100].empty else c5_seg.iloc[0]

    cm_glob = cost_curve(y_policy, p_meta_po_glob_cal, policy.monto, fp_cost=5.0, loss_fraction=1.0, fn_cost=None)
    bm_glob = cm_glob.loc[cm_glob["fp"] <= 100].iloc[-1] if not cm_glob.loc[cm_glob["fp"] <= 100].empty else cm_glob.iloc[0]

    cm_seg = cost_curve(y_policy, p_meta_po_seg_cal, policy.monto, fp_cost=5.0, loss_fraction=1.0, fn_cost=None)
    bm_seg = cm_seg.loc[cm_seg["fp"] <= 100].iloc[-1] if not cm_seg.loc[cm_seg["fp"] <= 100].empty else cm_seg.iloc[0]

    # =========================================================================
    # REPORTE FINAL Y EXPORTACIÓN
    # =========================================================================
    print("\n" + "=" * 85)
    print("COMPARACIÓN FINAL: MODELO GLOBAL vs MODELOS SEGMENTADOS POR PAÍS")
    print("=" * 85)
    print(f"* MODELO GLOBAL (Tri-Modelo H4):")
    print(f"  val_search -> PR-AUC Global: {m_tu_glob['pr_auc']:.5f} | AP: {m_tu_glob['average_precision']:.5f} | ROC-AUC: {m_tu_glob['roc_auc']:.5f}")
    print(f"  val_policy -> PR-AUC Cal: {m_po_glob['pr_auc']:.5f} | Costo ($5FP+200FN): ${b5_glob['cost']:.0f} (FP:{b5_glob['fp']}, TP:{b5_glob['tp']}) | Costo Monto: ${bm_glob['cost']:.0f}")

    print(f"\n* MODELOS SEGMENTADOS POR PAÍS (Especializados BR, AR, REST):")
    print(f"  val_search -> PR-AUC Global: {m_tu_seg['pr_auc']:.5f} | AP: {m_tu_seg['average_precision']:.5f} | ROC-AUC: {m_tu_seg['roc_auc']:.5f}")
    print(f"  val_policy -> PR-AUC Cal: {m_po_seg['pr_auc']:.5f} | Costo ($5FP+200FN): ${b5_seg['cost']:.0f} (FP:{b5_seg['fp']}, TP:{b5_seg['tp']}) | Costo Monto: ${bm_seg['cost']:.0f}")

    print("\n--- DESGLOSE POR SECTOR GEOGRÁFICO EN VAL_SEARCH ---")
    for sec, dat in sector_models.items():
        print(f"  Sector {sec:<5s}: PR-AUC Especializado = {dat['pr_auc_specialized']:.5f} | PR-AUC Modelo Global = {dat['pr_auc_global']:.5f} | Delta = {dat['delta']:+.5f}")

    # Guardar resumen JSON
    summary = {
        "global_model": {
            "pr_auc_val_search": m_tu_glob["pr_auc"],
            "ap_val_search": m_tu_glob["average_precision"],
            "roc_auc_val_search": m_tu_glob["roc_auc"],
            "pr_auc_val_policy_cal": m_po_glob["pr_auc"],
            "cost_5_200": float(b5_glob["cost"]),
            "cost_monto": float(bm_glob["cost"]),
        },
        "segmented_models": {
            "pr_auc_val_search": m_tu_seg["pr_auc"],
            "ap_val_search": m_tu_seg["average_precision"],
            "roc_auc_val_search": m_tu_seg["roc_auc"],
            "pr_auc_val_policy_cal": m_po_seg["pr_auc"],
            "cost_5_200": float(b5_seg["cost"]),
            "cost_monto": float(bm_seg["cost"]),
        },
        "sectors_breakdown": sector_models,
    }
    with open(output_dir / "country_segmented_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nResultados guardados exitosamente en: {output_dir}")
    return summary


if __name__ == "__main__":
    run_country_segmented_experiment()
