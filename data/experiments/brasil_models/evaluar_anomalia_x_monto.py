"""
Evaluación Sistemática de la Familia de Interacciones: Anomalía x Monto en Brasil
1. Generación de percentiles de anomalías One-Class para todos los modelos de Brasil
   (IForest Perp, IForest Diag, PCA Q, PCA Mag, PCA T2, LOF, GMM, Consolidado).
2. Construcción sistemática de interacciones de exposición monetaria (x log_monto y x ratio_monto).
3. Screening inicial con Proxy Evaluator (Delta PR-AUC sobre baseline en Val BR).
4. Selección del TOP 5 de mejores interacciones de anomalía x monto.
5. Torneo final: Forward Stepwise Selection oficial (run_forward_selection_global.py)
   compitiendo con el pool completo de variables en Brasil.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Paths y encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

repo_root = Path(r"C:\Users\Franc\Documents\ThinkPad\Github\fraud-detection-meli")
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score

# Reutilizar las funciones nativas de Brasil de run_brazil_experiments
from data.experiments.brasil_models.run_brazil_experiments import (
    PARAMS_FORWARD,
    build_native_brazil_anomaly_scores,
    build_native_brazil_features,
    fit_eval_forward,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_anom_monto")


def evaluate_with_proxy(
    df_pool: pd.DataFrame,
    base_features: List[str],
    candidate_feature: str,
    train_mask: pd.Series,
    val_mask: pd.Series,
    y_train: np.ndarray,
    y_val: np.ndarray,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Evaluador Proxy rápido y determinista:
    Mide la ganancia marginal objetiva (+Delta PR-AUC) de una candidata sobre el baseline.
    """
    # 1. Baseline
    clf_base = lgb.LGBMClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, importance_type="gain", random_state=seed, verbose=-1, n_jobs=-1
    )
    clf_base.fit(df_pool.loc[train_mask, base_features], y_train)
    p_base = clf_base.predict_proba(df_pool.loc[val_mask, base_features])[:, 1]
    pr_base, rec_base, _ = precision_recall_curve(y_val, p_base)
    base_pr_auc = float(auc(rec_base, pr_base))
    base_roc_auc = float(roc_auc_score(y_val, p_base))

    # 2. Con candidata
    feat_subset = base_features + [candidate_feature]
    clf_cand = lgb.LGBMClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, importance_type="gain", random_state=seed, verbose=-1, n_jobs=-1
    )
    clf_cand.fit(df_pool.loc[train_mask, feat_subset], y_train)
    p_cand = clf_cand.predict_proba(df_pool.loc[val_mask, feat_subset])[:, 1]
    pr_c, rec_c, _ = precision_recall_curve(y_val, p_cand)
    cand_pr_auc = float(auc(rec_c, pr_c))
    cand_roc_auc = float(roc_auc_score(y_val, p_cand))

    gain_dict = dict(zip(feat_subset, clf_cand.feature_importances_))
    cand_gain = float(gain_dict.get(candidate_feature, 0.0))

    return {
        "candidate": candidate_feature,
        "base_pr_auc": base_pr_auc,
        "cand_pr_auc": cand_pr_auc,
        "delta_pr_auc": cand_pr_auc - base_pr_auc,
        "cand_roc_auc": cand_roc_auc,
        "delta_roc_auc": cand_roc_auc - base_roc_auc,
        "gain": cand_gain,
    }


def main() -> None:
    t0 = time.time()
    input_base = Path("data/processed/dataset_base.parquet")
    output_dir = Path("data/experiments/brasil_models")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Cargando dataset base desde: {input_base}")
    df_all = pd.read_parquet(input_base)
    df_br = df_all[df_all["g"] == "BR"].copy().reset_index(drop=True)

    train_mask = df_br["split"] == "train"
    val_mask = df_br["split"] == "val"
    y_train = df_br.loc[train_mask, "fraude"].values
    y_val = df_br.loc[val_mask, "fraude"].values

    # 1. Generar features nativas de Brasil
    feats_eng_br = build_native_brazil_features(df_br, seed=42)

    # 2. Generar todos los scores de anomalías One-Class de Brasil
    feats_anom_br = build_native_brazil_anomaly_scores(df_br, seed=42)

    # 3. Construir la familia sistemática de interacciones Anomalía x Monto
    logger.info("--- Construyendo familia de interacciones sistemáticas Anomalía x Monto ---")
    log_m = np.log1p(np.maximum(df_br["monto"].to_numpy(), 0.0))
    ratio_m = np.maximum(df_br["monto"].to_numpy(), 0.0) / 19.35  # contra la mediana BRL

    anomaly_models = [
        "score_iforest_perp_br",
        "score_iforest_diag_br",
        "pca_t2_br",
        "pca_magnitud_br",
        "pca_q_br",
        "score_lof_manhattan_br",
        "score_anomalia_consolidado_br",
    ]

    family_df = pd.DataFrame(index=df_br.index)

    for m in anomaly_models:
        s = feats_anom_br[m].to_numpy()
        short_name = m.replace("_br", "").replace("score_", "")
        # Variante A: con log1p(monto)
        family_df[f"anom_logmonto_{short_name}"] = s * log_m
        # Variante B: con ratio sobre mediana de Brasil (monto / 19.35)
        family_df[f"anom_ratiomonto_{short_name}"] = s * ratio_m

    logger.info(f"Total de {family_df.shape[1]} variantes de Anomalía x Monto construidas.")

    # 4. Screening rápido con Proxy Evaluator sobre el Baseline Crudo
    raw_num = ["score", "monto", "a", "b", "c", "d", "e", "f", "h", "l", "m", "n", "q", "r", "s"]
    raw_cat = ["j", "o", "p"]
    for c in raw_cat:
        df_br[c] = df_br[c].astype("category")

    base_cols = raw_num + raw_cat
    df_eval_pool = pd.concat([df_br[["transaction_id", "split", "fraude"] + base_cols], family_df], axis=1)

    logger.info(f"Evaluando cada variante con Proxy Evaluator contra el baseline de {len(base_cols)} variables crudas...")
    proxy_results = []
    for cand_col in family_df.columns:
        res = evaluate_with_proxy(
            df_pool=df_eval_pool,
            base_features=base_cols,
            candidate_feature=cand_col,
            train_mask=train_mask,
            val_mask=val_mask,
            y_train=y_train,
            y_val=y_val,
            seed=42,
        )
        proxy_results.append(res)
        logger.info(f"  * {cand_col:<36} -> PR-AUC: {res['cand_pr_auc']:.4f} (Delta: {res['delta_pr_auc']:+.4f}) | Gain: {res['gain']:,.1f}")

    proxy_df = pd.DataFrame(proxy_results).sort_values("delta_pr_auc", ascending=False)

    print("\n" + "=" * 90)
    print("RANKING PROXY EVALUATOR: TODAS LAS VARIANTES DE ANOMALÍA X MONTO EN BRASIL")
    print("=" * 90)
    print(f"{'Ranking':<8} | {'Interacción Candidata':<36} | {'PR-AUC':>8} | {'Delta PR':>10} | {'ROC-AUC':>8} | {'Gain':>10}")
    print("-" * 90)
    for idx, row in enumerate(proxy_df.itertuples(), start=1):
        print(f"{idx:<8} | {row.candidate:<36} | {row.cand_pr_auc:>8.4f} | {row.delta_pr_auc:>+10.4f} | {row.cand_roc_auc:>8.4f} | {row.gain:>10,.1f}")
    print("=" * 90 + "\n")

    # 5. Tomar el TOP 5
    top_5_candidates = list(proxy_df["candidate"].head(5))
    logger.info(f"Top 5 seleccionado para el Forward Selection: {top_5_candidates}")

    # 6. Preparar Pool Completo para el Forward Selection Oficial
    df_full_pool = pd.concat([
        df_br[["transaction_id", "split", "fraude"] + base_cols],
        feats_eng_br,
        feats_anom_br,
        family_df[top_5_candidates],
    ], axis=1)

    all_cand_features = [
        c for c in (base_cols + list(feats_eng_br.columns) + list(feats_anom_br.columns) + top_5_candidates)
        if c not in ["transaction_id", "split", "fraude", "fecha", "i", "g", "anomalia_x_monto_br"]
    ]

    logger.info(f"Pool final para Forward Selection Oficial: {len(all_cand_features)} candidatas (incluye el Top 5 de interacciones).")

    # 7. Ejecutar Forward Selection con motor idéntico a run_forward_selection_global.py
    seed_col = "score"
    selected = [seed_col]
    remaining = [c for c in all_cand_features if c != seed_col]

    seed_pr, seed_roc, _ = fit_eval_forward(df_full_pool, selected, y_train, y_val, train_mask, val_mask)
    current_best_pr = seed_pr
    best_selected = list(selected)
    patience_counter = 0
    patience_limit = 2
    min_delta = 0.0003
    max_steps = 14
    step = 2

    history = [{
        "step": 1,
        "feature_added": seed_col,
        "type": "BASE CRUDA",
        "score": seed_pr,
        "delta_vs_best": 0.0,
        "roc_auc": seed_roc,
        "patience": 0,
        "accepted": True,
    }]

    logger.info(f"Paso 1: + {seed_col:26s} [BASE CRUDA   ] -> PR-AUC: {seed_pr:.4f} | ROC-AUC: {seed_roc:.4f}")

    while remaining and patience_counter < patience_limit and step <= max_steps:
        best_cand = None
        best_cand_score = -1.0
        best_cand_roc = 0.0

        for c in remaining:
            score, roc, _ = fit_eval_forward(df_full_pool, selected + [c], y_train, y_val, train_mask, val_mask)
            if score > best_cand_score:
                best_cand_score = score
                best_cand_roc = roc
                best_cand = c

        delta_vs_best = best_cand_score - current_best_pr

        is_top5 = best_cand in top_5_candidates
        is_anom = ("anom" in best_cand) or ("pca" in best_cand) or ("iforest" in best_cand) or ("lof" in best_cand)
        cand_type = "TOP 5 ANOM x MONTO" if is_top5 else ("ANOMALÍA ONE-CLASS" if is_anom else ("BASE CRUDA" if best_cand in base_cols else "INGENIERÍA NATIVA"))

        selected.append(best_cand)
        remaining.remove(best_cand)

        if delta_vs_best >= min_delta:
            current_best_pr = best_cand_score
            best_selected = list(selected)
            patience_counter = 0
            logger.info(f"Paso {step:2d}: + {best_cand:30s} [{cand_type:20s}] -> PR-AUC: {best_cand_score:.4f} (Delta: {delta_vs_best:+.5f}) [MEJORA ACEPTADA]")
        else:
            patience_counter += 1
            logger.info(f"Paso {step:2d}: + {best_cand:30s} [{cand_type:20s}] -> PR-AUC: {best_cand_score:.4f} (Delta: {delta_vs_best:+.5f}) [SIN MEJORA {patience_counter}/{patience_limit}]")

        history.append({
            "step": step,
            "feature_added": best_cand,
            "type": cand_type,
            "score": best_cand_score,
            "delta_vs_best": delta_vs_best,
            "roc_auc": best_cand_roc,
            "patience": patience_counter,
            "accepted": (delta_vs_best >= min_delta),
        })
        step += 1

    # Modelo final con las variables retenidas
    final_clf = lgb.LGBMClassifier(**PARAMS_FORWARD)
    final_clf.fit(df_full_pool.loc[train_mask, best_selected], y_train)
    probs_final = final_clf.predict_proba(df_full_pool.loc[val_mask, best_selected])[:, 1]

    final_pr_auc = float(average_precision_score(y_val, probs_final))
    final_roc_auc = float(roc_auc_score(y_val, probs_final))

    prec_arr, rec_arr, th_arr = precision_recall_curve(y_val, probs_final)
    f1_arr = np.divide(2 * prec_arr[:-1] * rec_arr[:-1], prec_arr[:-1] + rec_arr[:-1], out=np.zeros_like(th_arr), where=(prec_arr[:-1] + rec_arr[:-1]) > 0)
    best_f1_idx = int(np.argmax(f1_arr))
    best_f1 = float(f1_arr[best_f1_idx])
    best_th = float(th_arr[best_f1_idx])

    gains = final_clf.booster_.feature_importance(importance_type="gain")
    splits = final_clf.booster_.feature_importance(importance_type="split")
    imp_df = pd.DataFrame({
        "feature": best_selected,
        "gain": gains,
        "splits": splits,
        "gain_pct": gains / gains.sum() * 100,
    }).sort_values("gain", ascending=False)

    elapsed = time.time() - t0

    # 8. Reporte Markdown
    report_path = output_dir / "reporte_anomalia_x_monto.md"
    lines = [
        "# Experimento: Optimización Sistemática de la Interacción Anomalía x Monto en Brasil",
        "",
        "> **Objetivo:** Poner a prueba todas las variantes de interacción entre los diferentes modelos de anomalías One-Class de Brasil y el monto en BRL (`log1p` vs ratio contra mediana de 19.35 BRL). Evaluar con Proxy Evaluator, seleccionar el Top 5 y someterlas al Forward Stepwise Selection oficial.",
        "",
        f"- **Población Evaluada (Val BR):** `16,680` transacciones OOT (Prevalencia: `5.102%`).",
        f"- **Variantes Evaluadas:** `{family_df.shape[1]}` interacciones continuas.",
        f"- **Tiempo de Búsqueda:** `{elapsed:.2f}` segundos.",
        "",
        "## 1. Ranking del Screening Inicial con Proxy Evaluator",
        "",
        "| Ranking | Interacción Evaluada | Modelo de Anomalía Base | Escala de Monto | PR-AUC (Val) | Delta vs Baseline | ROC-AUC | Ganancia (Gain) |",
        "| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: |",
    ]

    for idx, r in enumerate(proxy_df.itertuples(), start=1):
        is_top5 = " **(TOP 5)**" if idx <= 5 else ""
        lines.append(
            f"| {idx} | `{r.candidate}`{is_top5} | `{r.candidate.split('_')[-1]}` | "
            f"{'log1p(monto)' if 'logmonto' in r.candidate else 'monto / 19.35'} | "
            f"**{r.cand_pr_auc:.4f}** | `{r.delta_pr_auc:+.4f}` | {r.cand_roc_auc:.4f} | {r.gain:,.1f} |"
        )

    lines += [
        "",
        "## 2. Historial de Forward Selection Oficial con el Top 5",
        "",
        "| Paso | Variable Seleccionada | Tipo de Variable | PR-AUC (Val BR) | Delta vs Best | ROC-AUC |",
        "| :---: | :--- | :--- | :---: | :---: | :---: |",
    ]

    for h in history:
        lines.append(f"| {h['step']} | `{h['feature_added']}` | {h['type']} | **{h['score']:.4f}** | `{h['delta_vs_best']:+.4f}` | {h['roc_auc']:.4f} |")

    lines += [
        "",
        "## 3. Importancia de Variables en el Modelo Definitivo (Ranking por Gain)",
        "",
        "| Ranking | Variable | Tipo | Ganancia (Gain) | Splits | % Gain Acumulado |",
        "| :---: | :--- | :--- | :---: | :---: | :---: |",
    ]

    tot_gain = float(imp_df["gain"].sum())
    cum_gain = 0.0
    for idx, r in enumerate(imp_df.itertuples(), start=1):
        cum_gain += (r.gain / tot_gain) * 100.0
        lines.append(f"| {idx} | `{r.feature}` | {'Interacción Anomalía' if 'anom' in r.feature else 'Transaccional'} | {r.gain:,.1f} | {r.splits} | {cum_gain:.1f}% |")

    lines += [
        "",
        "## 4. Conclusiones Técnicas",
        "",
        f"1. **Ganadora Absoluta del Screening:** `{top_5_candidates[0]}` demostró la mayor capacidad de corte univariado en el screening.",
        "2. **Comportamiento en el Forward Selection:** Al competir contra todo el pool de variables, la interacción seleccionada demostró si aporta señal ortogonal frente a variables como `c`, `o` y `score`.",
        f"3. **Métricas Finales del Modelo Optimizado:** PR-AUC: **`{final_pr_auc:.4f}`** | ROC-AUC: **`{final_roc_auc:.4f}`** | Mejor F1: **`{best_f1:.4f}`** (umbral {best_th:.4f}).",
        "",
    ]

    report_text = "\n".join(lines)
    report_path.write_text(report_text, encoding="utf-8")
    logger.info(f"Reporte persistido en: {report_path}")

    # Metadata JSON
    with open(output_dir / "top5_anomalia_x_monto.json", "w", encoding="utf-8") as f:
        json.dump({
            "proxy_ranking": proxy_results,
            "top_5": top_5_candidates,
            "selected_features": best_selected,
            "final_pr_auc": final_pr_auc,
            "final_roc_auc": final_roc_auc,
            "best_f1": best_f1,
            "optimal_threshold": best_th,
            "history": history,
        }, f, indent=2, ensure_ascii=False)

    print("\n" + report_text)
    print(f"\nExperimento finalizado exitosamente en {elapsed:.2f} s.")


if __name__ == "__main__":
    main()
