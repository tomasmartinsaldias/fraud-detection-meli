"""Compare the team's official artifacts and ensembles using development data only.

The old test was already consumed. This experiment never reads test rows.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from src.models.supervised import (
    BASE_NUMERIC, BASE_CATEGORICAL, FrameEncoder, IdentityEncoder,
    merge_features, validate_development, validation_blocks, logistic_model,
    metrics, calibrate, logit, cost_curve, policy_metrics,
)
from src.models.train import lgb_model, search, save_json, export_curves
from src.models.report_supervised import chart


def predict(candidate, frame):
    return candidate["model"].predict_proba(candidate["encoder"].transform(frame))[:, 1]


class WeightedModels:
    def __init__(self, members, weights):
        self.members, self.weights = members, np.asarray(weights)
    def predict_proba(self, frame):
        p = np.average([predict(member, frame) for member in self.members], axis=0, weights=self.weights)
        return np.column_stack([1 - p, p])


class ScoreModel:
    def __init__(self, column):
        self.column = column
    def predict_proba(self, frame):
        p = frame[self.column].to_numpy()
        return np.column_stack([1 - p, p])


def choose_threshold(frame, probabilities, cap, fp_cost=5., fn_cost=200.):
    curve = cost_curve(frame.fraude, probabilities, frame.monto, fp_cost, fn_cost=fn_cost)
    allowed = curve.loc[curve.fp <= cap]
    return float(allowed.loc[allowed.cost.idxmin(), "threshold"])


def policy(frame, p, threshold, fp_cost=5.):
    return policy_metrics(frame.fraude, p, frame.monto, threshold, fp_cost, 1., fn_cost=200.)


def prepare_selected():
    path = Path("data/features/features_seleccionadas_end_to_end.parquet")
    if path.exists():
        return
    report = json.loads(Path("data/features/selected_features_end_to_end_report.json").read_text(encoding="utf-8"))
    columns = report["all_selected_features_order"]
    base_columns = ["transaction_id", "score", "o", "g", "a"]
    base = pd.read_parquet("data/processed/dataset_base.parquet", columns=base_columns)
    engineered = pd.read_parquet("data/features/features_transaccionales.parquet")
    result = merge_features(base, engineered)
    result[["transaction_id"] + columns].astype({"transaction_id": "string"}).to_parquet(path, index=False, compression="snappy")


def load_development():
    base = pd.read_parquet("data/processed/dataset_base.parquet", filters=[("split", "in", ["train", "val"])])
    features = pd.read_parquet("data/features/features_transaccionales.parquet")
    features = features.loc[features.transaction_id.isin(base.transaction_id)]
    df = merge_features(base, features)
    selected = json.loads(Path("data/features/selected_features_end_to_end_report.json").read_text(encoding="utf-8"))["all_selected_features_order"]
    # Preserve an exact official text column for a documented reproduction.
    df["score_texto_nb_official"] = df.score_texto_nb
    causal = pd.read_parquet("data/features/ian_features_selected.parquet", columns=["transaction_id", "score_texto_nb"])
    df["score_texto_nb"] = df.transaction_id.map(causal.set_index("transaction_id").score_texto_nb)
    scores = pd.read_parquet("data/features/scores_anomalias.parquet")
    scores = scores.loc[scores.transaction_id.isin(base.transaction_id)]
    df = merge_features(df, scores)
    score_columns = [c for c in scores if c not in {"transaction_id", "score_anomalia_consolidado"}]
    # Random KFold target encoding and velocity with timestamp ties are excluded
    # from the causal pool; official files are preserved for audit.
    excluded = {"transaction_id", "te_j_bayes", "log_delta_mismo_prod", "burst_mismo_prod_24h"}
    full = list(dict.fromkeys(BASE_NUMERIC + BASE_CATEGORICAL + [c for c in features if c not in excluded]))
    validate_development(df)
    return df, selected, full, score_columns


def write_report(args, report):
    chosen = report["chosen_before_check"]
    rows = report["validation_check"]
    baseline = rows["ian_v1_frozen"]
    new = rows[chosen]
    lines = ["# Comparación del equipo y ensambles", "",
        "**Solo validación. Test ya fue utilizado en la corrida anterior y no se consulta nuevamente.**", "",
        "## Datos y reproducibilidad", "",
        "CSV extraído del ZIP del curso a data/raw/fraud_dataset_v2.csv. La base coincide con make_dataset.py.",
        "Se ejecutaron build_features.py y anomaly_detection.py. Se reconstruyeron las once variables",
        "del reporte de selección guardado, sin repetir la selección sobre val.",
        "Se corrigió el preprocesador de Franco: r/s son opcionales porque la ingesta oficial las elimina.",
        "Los scores usan el mismo bundle original (PCA, IF, LOF, OCSVM y GMM), con ese ajuste de esquema.",
        "El bundle original incluye score como predictor: no sirve como prueba de descubrimiento de fraude sin scoring.",
        "", "## Protocolo", "",
        "Train: 105000 filas. Val_search: 13499; calibración y ajuste de política dividen las siguientes 4500;",
        "val_check: ultimas 4501. Los límites exactos y conteos se guardan en comparison.json.",
        "Hiperparámetros y miembros del ensamble se eligen con val_search. Se prueban pesos prefijados.",
        "Platt se ajusta en su bloque. Umbrales, pesos finales y candidato de negocio",
        "se congelan con policy_fit antes de evaluar val_check. El límite propuesto es 100 FP por 4501 transacciones;",
        "se escala al volumen de policy_fit. Es una hipótesis de capacidad, no una garantia en otras muestras.",
        "Val ya participo en trabajos anteriores; esta comparación es exploratoria y no sustituye un nuevo test.",
        "", "## Comparación predictiva en val_search", "",
        "| Modelo | PR-AUC | AP | ROC-AUC |", "|---|---:|---:|---:|"]
    for name, result in sorted(report["search_metrics"].items(), key=lambda pair: pair[1]["pr_auc"], reverse=True):
        lines.append(f"| {name} | {result['pr_auc']:.4f} | {result['average_precision']:.4f} | {result['roc_auc']:.4f} |")
    lines += ["", "## Decisión de negocio congelada antes de val_check", "",
        f"Candidato elegido: **{chosen}**. Menor costo con el límite de FP en policy_fit; desempate por PR-AUC.",
        "Función: 5 x FP + 200 x FN. Precision mide cuantas alertas son fraude; recall cuantos fraudes se detectan.",
        "Bloquear una compra legitima no equivale a perder un cliente: no hay etiquetas de churn para medirlo.",
        "", "## Evaluación diagnóstica en val_check", "",
        "| Modelo | PR-AUC | Umbral | FP | FN | Precision | Recall | Costo |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, result in rows.items():
        m, p = result["metrics"], result["policy"]
        lines.append(f"| {name} | {m['pr_auc']:.4f} | {p['threshold']:.5f} | {p['fp']} | {p['fn']} | {p['precision']:.3f} | {p['recall']:.3f} | {p['cost']:,.0f} |")
    delta_cost = baseline["policy"]["cost"] - new["policy"]["cost"]
    delta_fp = baseline["policy"]["fp"] - new["policy"]["fp"]
    lines += ["", "## Resultado e implicaciones", "",
        f"Frente a la política anterior: ahorro observado {delta_cost:,.0f}; reducción de falsos positivos {delta_fp}.",
        "Valores negativos significan empeoramiento. No se cambia el candidato congelado despues de ver val_check.",
        "No se promete eliminar todas las perdidas: subir la detección normalmente aumenta la fricción.",
        f"Intervalo exploratorio bootstrap pareado del ahorro (95%): {report['cost_difference_ci'][0]:,.0f} a {report['cost_difference_ci'][1]:,.0f}.",
        "El bootstrap de filas no modela dependencia temporal; no es una garantia de generalización.",
        "", "## Sensibilidad: pesos monetarios y presupuesto de FP", "",
        "| FP cost | Tope equivalente /4501 | Umbral fijado en policy_fit | FP val_check | FN val_check | Costo val_check |",
        "|---|---:|---:|---:|---:|---:|"]
    for row in report["sensitivity"]:
        p = row["validation_policy"]
        lines.append(f"| {row['fp_cost']} | {row['cap_per_4501']} | {p['threshold']:.5f} | {p['fp']} | {p['fn']} | {p['cost']:,.0f} |")
    lines += ["", "## Protecciones y límites", "",
        "- No se usa el consolidador supervisado de Franco como feature de entrenamiento in-sample.",
        "  Se evalúa directamente como modelo independiente y se permite en mezclas de predicciones de validación.",
        "- El pool causal excluye te_j_bayes (KFold aleatorio) y las dos variables de velocidad con empates temporales.",
        "  score_texto_nb se sustituye por la version temporal sin prior futuro de Ian.",
        "- tomas_selected_official reproduce las columnas originales para trazabilidad; no participa como candidato",
        "  de negocio final por el problema documentado del prior de texto. Los originales no se sobrescriben.",
        "- El modelo anterior usa su calibrador/umbral congelados; había ajustado ese umbral en val_check anteriormente.",
        "  Tiene una ventaja metodológica: no interpretar esta muestra como prueba independiente.",
        "- Los miembros del promedio se entrenan en train y se preseleccionan por val_search.",
        "  La elección final entre los pesos prefijados usa policy_fit, nunca val_check.",
        "- Mantener ian_v1 como referencia. Para reemplazarlo con evidencia independiente hace falta un período nuevo.",
        "- El costo FP de 5 cubre contacto; no incluye churn, reputacion ni margen perdido. Se explora subirlo a 10/20.",
        "- En producción, derivar riesgo intermedio a verificacion podria reducir fricción; falta medir su efectividad",
        "  y costo, por lo que no se contabiliza como ahorro demostrado.",
        "", "## Reproducir", "",
        "```powershell",
        ".\\run.bat src/features/build_features.py",
        ".\\run.bat src/models/anomaly_detection.py",
        ".\\run.bat src/models/compare_team.py --output models/supervised/team_v2 --trials 12 --seed 42",
        "```", "",
        "En esta máquina sigue sin estar disponible el launcher py/3.14; se ejecutó con .venv Python 3.12.14",
        "y las dependencias fijadas. No se modificó .python-version.",
        "", "Artefactos: comparison.json, trials_*.json, selected_bundle.joblib, predicciones val_check,",
        "curvas y políticas en la carpeta de corrida. No hay commits ni push."]
    # Plot comparison curves for three meaningful candidates on validation.
    plot_dir = args.report.parent / "figures_team"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_series = []
    import csv
    for name in dict.fromkeys(["ian_v1_frozen", chosen, "blend_top3"]):
        with (args.output / f"{name}_check_pr.csv").open() as stream:
            points = [(float(r["recall"]), float(r["precision"])) for r in csv.DictReader(stream)]
        plot_series.append((name, points))
    chart(plot_dir / "pr_validation.svg", plot_series, "Recall", "Precision", "Equipo: validación exploratoria")
    lines += ["", "![Comparación PR](figures_team/pr_validation.svg)"]
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/supervised/team_v2"))
    parser.add_argument("--report", type=Path, default=Path("docs/COMPARACION_EQUIPO.md"))
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--max-trees", type=int, default=600)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.trials < 1: parser.error("trials debe ser positivo")
    prepare_selected()
    df, selected, full, anomaly_cols = load_development()
    train = df.loc[df.split == "train"]
    tune, cal_policy, check = validation_blocks(df.loc[df.split == "val"])
    cutoff = cal_policy.fecha.iloc[len(cal_policy) // 2]
    calibration = cal_policy.loc[cal_policy.fecha < cutoff]
    policy_fit = cal_policy.loc[cal_policy.fecha >= cutoff]
    if any(set(part.fraude.unique()) != {0, 1} for part in (calibration, policy_fit)):
        raise ValueError("Los bloques necesitan ambas clases.")
    cap = max(0, int(100 * len(policy_fit) / 4501))
    args.output.mkdir(parents=True, exist_ok=False)
    previous = joblib.load("models/supervised/ian_v1/bundle.joblib")
    incumbent = previous["candidates"][previous["manifest"]["winner"]]
    candidates = {"ian_v1_model": {"model": incumbent["model"], "encoder": incumbent["encoder"]}}
    families = {
        "franco_pca": [c for c in anomaly_cols if c.startswith("pca_")],
        "franco_iforest": [c for c in anomaly_cols if "iforest" in c],
        "franco_lof": [c for c in anomaly_cols if "lof" in c],
        "franco_ocsvm": [c for c in anomaly_cols if "ocsvm" in c],
        "franco_gmm": [c for c in anomaly_cols if "gmm" in c],
        "franco_all_scores": anomaly_cols,
    }
    groups = {
        **families,
        "tomas_selected_official": ["score_texto_nb_official" if c == "score_texto_nb" else c for c in selected],
        "tomas_full_causal": full,
        "selected_plus_franco": selected + anomaly_cols,
        "full_plus_franco": full + anomaly_cols,
    }
    for name, cols in groups.items():
        print(f"Entrenando {name} ({len(cols)} variables)", flush=True)
        encoder = FrameEncoder().fit(train[cols])
        x_train, x_tune = encoder.transform(train), encoder.transform(tune)
        params = {"n_estimators": 100, "max_depth": 3 if name.startswith("franco_") else 5, "learning_rate": .05}
        candidates[name] = {"model": lgb_model(params, args.seed, args.jobs).fit(x_train, train.fraude), "encoder": encoder}
        if name in {"tomas_full_causal", "selected_plus_franco", "full_plus_franco"}:
            for sampler in ("tpe", "cmaes"):
                model, _ = search(x_train, train.fraude, x_tune, tune.fraude, sampler, args, args.output / f"trials_{name}_{sampler}.json")
                candidates[f"{name}_{sampler}"] = {"model": model, "encoder": encoder}
    candidates["franco_consolidator"] = {"model": ScoreModel("score_anomalia_consolidado"), "encoder": IdentityEncoder()}
    encoder = FrameEncoder().fit(train[selected + anomaly_cols])
    x_train = encoder.transform(train)
    forest = logistic_model(x_train, args.seed)
    forest.steps[-1] = ("model", RandomForestClassifier(n_estimators=200, max_depth=14, min_samples_leaf=30, max_features=.7, n_jobs=args.jobs, random_state=args.seed))
    candidates["random_forest"] = {"model": forest.fit(x_train, train.fraude), "encoder": encoder}
    candidates["hist_gradient"] = {"model": HistGradientBoostingClassifier(max_iter=250, max_leaf_nodes=15, learning_rate=.05, l2_regularization=10, early_stopping=False, random_state=args.seed).fit(x_train, train.fraude), "encoder": encoder}
    search_metrics = {name: metrics(tune.fraude, predict(candidate, tune)) for name, candidate in candidates.items()}
    eligible = [name for name in candidates if name != "tomas_selected_official"]
    top = sorted(eligible, key=lambda n: search_metrics[n]["pr_auc"], reverse=True)[:3]
    candidates["blend_top3"] = {"model": WeightedModels([candidates[n] for n in top], [1, 1, 1]), "encoder": IdentityEncoder()}
    for weight in (.25, .5, .75):
        candidates[f"blend_top2_{weight}"] = {"model": WeightedModels([candidates[n] for n in top[:2]], [weight, 1 - weight]), "encoder": IdentityEncoder()}
    search_metrics = {name: metrics(tune.fraude, predict(candidate, tune)) for name, candidate in candidates.items()}
    eligible += ["blend_top3", "blend_top2_0.25", "blend_top2_0.5", "blend_top2_0.75"]
    selection = {}
    for name, candidate in candidates.items():
        p_cal = predict(candidate, calibration)
        candidate["calibrator"] = LogisticRegression(C=1e6, max_iter=1000, random_state=args.seed).fit(logit(p_cal), calibration.fraude)
        p_fit = calibrate(predict(candidate, policy_fit), candidate["calibrator"])
        candidate["threshold"] = choose_threshold(policy_fit, p_fit, cap)
        selection[name] = policy(policy_fit, p_fit, candidate["threshold"])
    chosen = min(eligible, key=lambda n: (selection[n]["cost"], -search_metrics[n]["pr_auc"]))
    frozen = {"name": chosen, "candidate": candidates[chosen], "selection": selection, "seed": args.seed,
              "test_used": False, "top_ensemble_members": top, "cap_policy_fit": cap}
    joblib.dump(frozen, args.output / "selected_bundle.joblib")
    save_json(args.output / "selection_before_check.json", {"chosen": chosen, "selection": selection, "top": top, "seed": args.seed})
    # Everything above is frozen before evaluating this chronological diagnostic block.
    rows, predictions = {}, {}
    for name, candidate in candidates.items():
        p = calibrate(predict(candidate, check), candidate["calibrator"])
        predictions[name] = p
        rows[name] = {"metrics": metrics(check.fraude, p), "policy": policy(check, p, candidate["threshold"])}
        export_curves(args.output, f"{name}_check", check.fraude, p)
    previous_p = calibrate(predict(incumbent, check), incumbent["calibrator"])
    old_threshold = incumbent["policies"]["5.0"]["threshold"]
    predictions["ian_v1_frozen"] = previous_p
    rows["ian_v1_frozen"] = {"metrics": metrics(check.fraude, previous_p), "policy": policy(check, previous_p, old_threshold)}
    export_curves(args.output, "ian_v1_frozen_check", check.fraude, previous_p)
    sensitivity = []
    candidate = candidates[chosen]
    fit_p = calibrate(predict(candidate, policy_fit), candidate["calibrator"])
    for fp_cost in (5., 10., 20.):
        for budget in (50, 100, 200, 500):
            threshold = choose_threshold(policy_fit, fit_p, int(budget * len(policy_fit) / 4501), fp_cost)
            sensitivity.append({"fp_cost": fp_cost, "cap_per_4501": budget, "validation_policy": policy(check, predictions[chosen], threshold, fp_cost)})
    labels = check.fraude.to_numpy()
    old_errors = np.where(labels == 1, 200 * (previous_p < old_threshold), 5 * (previous_p >= old_threshold))
    new_errors = np.where(labels == 1, 200 * (predictions[chosen] < candidate["threshold"]), 5 * (predictions[chosen] >= candidate["threshold"]))
    difference = old_errors - new_errors
    rng = np.random.default_rng(args.seed)
    sampled = [float(difference[rng.integers(len(check), size=len(check))].sum()) for _ in range(1000)]
    report = {"chosen_before_check": chosen, "rows": {"train": len(train), "search": len(tune), "calibration": len(calibration), "policy_fit": len(policy_fit), "check": len(check)},
              "search_metrics": search_metrics, "selection": selection, "validation_check": rows,
              "sensitivity": sensitivity, "cost_difference_ci": np.quantile(sampled, [.025, .975]).tolist(),
              "top_ensemble_members": top, "test_used": False, "test_previously_consumed": True}
    save_json(args.output / "comparison.json", report)
    pd.DataFrame({"transaction_id": check.transaction_id, **predictions}).to_parquet(args.output / "validation_predictions.parquet", index=False)
    write_report(args, report)
    print(f"Candidato congelado: {chosen}. Informe: {args.report}", flush=True)


if __name__ == "__main__":
    from src.models.compare_team import main as entrypoint
    entrypoint()
