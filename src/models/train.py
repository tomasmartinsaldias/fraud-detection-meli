"""Contract C: reproducible supervised selection, calibration and business policy.

Development never reads test rows. Final evaluation is a separate explicit mode.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import os
import pandas as pd
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve

from src.models.supervised import (
    BASE_CATEGORICAL, BASE_NUMERIC, FrameEncoder, AveragedModels, IdentityEncoder, calibrate, check_ids,
    cost_curve, logistic_model, logit, merge_features, metrics,
    policy_metrics, pr_auc, validate_development, validation_blocks,
)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_data(base_path, feature_path, anomaly_path, splits, exclude_anomaly_cols=(), rules_path=None, rule_cols=None):
    if not base_path.is_file():
        raise FileNotFoundError(f"Falta {base_path}. Obtener el Parquet oficial o ejecutar src/make_dataset.py.")
    # Predicate pushdown: do not materialize test labels during development.
    base = pd.read_parquet(base_path, filters=[("split", "in", splits)])
    check_ids(base)
    if "is_na_bc" not in base.columns and "c" in base.columns:
        base["is_na_bc"] = base["c"].isna().astype(int)
    df, feature_cols, anomaly_cols = base, [], []
    for path, kind in ((feature_path, "features"), (anomaly_path, "anomalies")):
        if path is None:
            continue
        if not path.is_file():
            raise FileNotFoundError(f"Falta {path}; usar --{kind} none para omitir explícitamente.")
        table = pd.read_parquet(path)
        check_ids(table)
        table = table.loc[table.transaction_id.isin(base.transaction_id)].copy()
        if kind == "anomalies" and exclude_anomaly_cols:
            unknown = set(exclude_anomaly_cols) - set(table)
            if unknown:
                raise ValueError(f"Columnas a excluir inexistentes en {path}: {sorted(unknown)}")
            table = table.drop(columns=list(exclude_anomaly_cols))
        columns = [c for c in table if c != "transaction_id"]
        if not columns:
            raise ValueError(f"Archivo sin predictoras: {path}")
        if kind == "anomalies":
            if set(columns) & set(df):
                raise ValueError("Scores de anomalías con nombres duplicados.")
            values = table[columns].to_numpy(dtype=float)
            if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                raise ValueError("Scores de anomalías deben ser finitos en [0,1].")
        df = merge_features(df, table)
        if kind == "features":
            feature_cols = columns
        else:
            anomaly_cols = columns

    rules_cols = []
    if rules_path is not None:
        if not rules_path.is_file():
            raise FileNotFoundError(f"Falta {rules_path}; usar --rules none para omitir explícitamente.")
        table = pd.read_parquet(rules_path)
        check_ids(table)
        table = table.loc[table.transaction_id.isin(base.transaction_id)].copy()
        if rule_cols:
            unknown = set(rule_cols) - set(table)
            if unknown:
                raise ValueError(f"Columnas de reglas inexistentes en {rules_path}: {sorted(unknown)}")
            table = table[["transaction_id"] + list(rule_cols)]
        columns = [c for c in table if c != "transaction_id"]
        if not columns:
            raise ValueError(f"Archivo de reglas sin predictoras: {rules_path}")
        df = merge_features(df, table)
        rules_cols = columns

    return df, feature_cols, anomaly_cols, rules_cols


def build_full_dataset(
    base_path="data/processed/dataset_base.parquet",
    feat_path="data/features/features_seleccionadas_end_to_end.parquet",
    anom_path="data/features/scores_anomalias.parquet",
    splits=("train", "val", "test"),
    exclude_anomaly_cols=(),
    rules_path=None,
    rule_cols=(),
) -> pd.DataFrame:
    """Integración estándar de Contrato C: merge 1:1 por transaction_id.

    Por defecto integra las tres particiones, como el código del contrato.
    El entrenamiento (develop) la invoca solo con train/val para no leer test.
    """
    base_path, feat_path, anom_path, rules_path = (None if p is None else Path(p) for p in (base_path, feat_path, anom_path, rules_path))
    df, _, _, _ = load_data(base_path, feat_path, anom_path, list(splits), exclude_anomaly_cols, rules_path, rule_cols)
    n_base = len(pd.read_parquet(base_path, columns=["transaction_id"], filters=[("split", "in", list(splits))]))
    assert len(df) == n_base, "Error: El merge alteró la cantidad de filas."
    assert df["transaction_id"].nunique() == n_base, "Error: Hay transaction_id duplicados."
    return df


def lgb_model(params, seed, jobs):
    return lgb.LGBMClassifier(**params, objective="binary", metric="None", verbosity=-1,
                             random_state=seed, n_jobs=jobs, deterministic=True,
                             force_col_wise=True, subsample_freq=1, importance_type="gain")


def lgb_metric(y, prediction):
    return "pr_auc", pr_auc(y, prediction), True


def search(x_train, y_train, x_val, y_val, sampler_name, args, out):
    random.seed(args.seed)
    np.random.seed(args.seed)
    sampler = (optuna.samplers.TPESampler(seed=args.seed) if sampler_name == "tpe"
               else optuna.samplers.CmaEsSampler(seed=args.seed))
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=50, interval_steps=10)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", .015, .15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 30, 300),
            "colsample_bytree": trial.suggest_float("colsample_bytree", .65, 1.),
            "subsample": trial.suggest_float("subsample", .65, 1.),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 10., log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 30., log=True),
            "n_estimators": args.max_trees,
        }
        model = lgb_model(params, args.seed, args.jobs)

        def prune(env):
            score = next(v[2] for v in env.evaluation_result_list if v[1] == "pr_auc")
            trial.report(score, env.iteration)
            if trial.should_prune():
                raise optuna.TrialPruned()

        prune.order = 20
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], eval_metric=lgb_metric,
                  callbacks=[prune, lgb.early_stopping(60, first_metric_only=True, verbose=False)])
        trial.set_user_attr("best_iteration", int(model.best_iteration_))
        return pr_auc(y_val, model.predict_proba(x_val)[:, 1])

    started = time.perf_counter()
    study.optimize(objective, n_trials=args.trials, n_jobs=1)
    records = [{"number": t.number, "state": t.state.name, "value": t.value,
                "params": t.params, "distributions": {k: optuna.distributions.distribution_to_json(v) for k, v in t.distributions.items()},
                "intermediate_values": t.intermediate_values, "user_attrs": t.user_attrs,
                "seconds": t.duration.total_seconds() if t.duration else None} for t in study.trials]
    save_json(out, {"sampler": sampler_name, "seed": args.seed, "pruner": "MedianPruner(startup=5,warmup=50,interval=10)",
                    "trials_budget": args.trials, "seconds": time.perf_counter() - started,
                    "best_trial": study.best_trial.number, "best_pr_auc": study.best_value, "trials": records})
    params = dict(study.best_params, n_estimators=study.best_trial.user_attrs["best_iteration"])
    model = lgb_model(params, args.seed, args.jobs).fit(x_train, y_train)
    return model, params


def export_curves(out, name, y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)
    pd.DataFrame({"precision": precision, "recall": recall,
                  "threshold": np.r_[thresholds, np.nan]}).to_csv(out / f"{name}_pr.csv", index=False)
    fpr, tpr, thresholds = roc_curve(y, p)
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds}).to_csv(out / f"{name}_roc.csv", index=False)


def develop(args):
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    rules = getattr(args, "rules", None)
    rule_cols = getattr(args, "rule_cols", [])
    samplers = getattr(args, "samplers", ["tpe", "cmaes"])
    target_groups = getattr(args, "groups", None)

    df, feat, anom, rules_cols = load_data(
        args.base, args.features, args.anomalies, ["train", "val"], args.exclude_anomaly_cols, rules, rule_cols
    )
    validate_development(df)
    train = df.loc[df.split == "train"]
    tune, calibration, policy = validation_blocks(df.loc[df.split == "val"])
    raw = BASE_NUMERIC + BASE_CATEGORICAL
    missing = set(raw) - set(df)
    if missing:
        raise ValueError(f"Faltan variables base oficiales: {sorted(missing)}")
    groups = {"raw": raw}
    if feat:
        # Selected artifact includes raw columns; full engineered pool does not.
        groups["transactional"] = feat if "score" in feat else list(dict.fromkeys(raw + feat))
    if anom:
        groups["full"] = list(dict.fromkeys(groups.get("transactional", raw) + anom))
    if rules_cols:
        groups["full_plus_rules"] = list(dict.fromkeys(groups.get("full", groups.get("transactional", raw)) + rules_cols))

    if target_groups:
        groups = {k: v for k, v in groups.items() if k in target_groups}
        if not groups:
            raise ValueError(f"Ninguno de los grupos solicitados {target_groups} está disponible.")

    args.output.mkdir(parents=True, exist_ok=False)
    candidates, comparison = {}, []
    for name, columns in groups.items():
        encoder = FrameEncoder().fit(train[columns])
        x_train, x_val = encoder.transform(train), encoder.transform(tune)
        if name == "raw":
            model = logistic_model(x_train, args.seed).fit(x_train, train.fraude)
            candidates["logistic_raw"] = {"model": model, "encoder": encoder, "params": {"max_iter": 3000}}
        # Fixed reference helps interpret feature gains without different tuning budgets.
        params = {"n_estimators": 100, "max_depth": 5, "learning_rate": .05,
                  "subsample": .8, "colsample_bytree": .8}
        model = lgb_model(params, args.seed, args.jobs).fit(x_train, train.fraude)
        candidates[f"lgb_reference_{name}"] = {"model": model, "encoder": encoder, "params": params}
        for sampler in samplers:
            model, params = search(x_train, train.fraude, x_val, tune.fraude, sampler, args,
                                   args.output / f"trials_{name}_{sampler}.json")
            candidates[f"lgb_{name}_{sampler}"] = {"model": model, "encoder": encoder, "params": params}

    candidates["average_ensemble"] = {"model": AveragedModels(list(candidates.values())),
                                       "encoder": IdentityEncoder(), "params": {"weights": "equal"}}
    for name, candidate in candidates.items():
        p = candidate["model"].predict_proba(candidate["encoder"].transform(tune))[:, 1]
        comparison.append({"model": name, **metrics(tune.fraude, p)})
    winner = max(comparison, key=lambda row: row["pr_auc"])["model"]
    policies = {}
    for name, candidate in candidates.items():
        p_cal = candidate["model"].predict_proba(candidate["encoder"].transform(calibration))[:, 1]
        # Fixed Platt method: no selecting calibrator on the policy labels.
        calibrator = LogisticRegression(C=1e6, max_iter=1000, random_state=args.seed).fit(logit(p_cal), calibration.fraude)
        candidate["calibrator"] = calibrator
        p_raw = candidate["model"].predict_proba(candidate["encoder"].transform(policy))[:, 1]
        p = calibrate(p_raw, calibrator)
        candidate["policies"] = {}
        for fp_cost in args.fp_costs:
            curve = cost_curve(policy.fraude, p, policy.monto, fp_cost, args.loss_fraction, args.fn_cost)
            allowed = curve.loc[curve.fp <= args.max_fp] if args.max_fp is not None else curve
            best = allowed.loc[allowed.cost.idxmin()].to_dict()
            best.update(policy_metrics(policy.fraude, p, policy.monto, best["threshold"], fp_cost, args.loss_fraction, args.fn_cost))
            candidate["policies"][str(fp_cost)] = best
            curve.to_csv(args.output / f"{name}_cost_{fp_cost}.csv", index=False)
        # Sensibilidad del contrato (C_FN = monto defraudado), SOLO en val_policy.
        # No se guarda en candidate["policies"], así que evaluate_test nunca la usa.
        curve = cost_curve(policy.fraude, p, policy.monto, args.fp_costs[0], args.loss_fraction, None)
        allowed = curve.loc[curve.fp <= args.max_fp] if args.max_fp is not None else curve
        amount_policy = allowed.loc[allowed.cost.idxmin()].to_dict()
        amount_policy.update(policy_metrics(policy.fraude, p, policy.monto, amount_policy["threshold"],
                                            args.fp_costs[0], args.loss_fraction, None))
        amount_policy["fp_cost"] = args.fp_costs[0]
        curve.to_csv(args.output / f"{name}_cost_monto.csv", index=False)
        policies[name] = {"raw": metrics(policy.fraude, p_raw), "calibrated": metrics(policy.fraude, p),
                          "scenarios": candidate["policies"], "amount_fn_val_only": amount_policy}
        export_curves(args.output, f"{name}_val_policy", policy.fraude, p)
        pd.DataFrame({"transaction_id": policy.transaction_id, "probability": p}).to_parquet(
            args.output / f"{name}_val_predictions.parquet", index=False, compression="snappy")

        if hasattr(candidate["model"], "feature_importances_"):
            save_json(args.output / f"{name}_importance.json", dict(zip(candidate["encoder"].columns,
                      map(float, candidate["model"].feature_importances_))))

    paths = {k: str(v.resolve()) if v else None for k, v in
             (("base", args.base), ("features", args.features), ("anomalies", args.anomalies), ("rules", rules))}
    # Hash development rows only: test data is not inspected to create the model.
    data_hash = hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()
    manifest = {"winner": winner, "seed": args.seed, "paths": paths, "development_hash": data_hash,
                "exclude_anomaly_cols": list(args.exclude_anomaly_cols), "rule_cols": list(rule_cols) if rule_cols else [],
                "python": platform.python_version(),
                "versions": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scikit-learn", "lightgbm", "optuna", "cmaes")},
                "rows": {"train": len(train), "val_search": len(tune), "val_calibration": len(calibration), "val_policy": len(policy)},
                "comparison": comparison, "policy_validation": policies, "loss_fraction": args.loss_fraction,
                "train_end": str(pd.to_datetime(train.fecha, utc=True).max()),
                "val_end": str(pd.to_datetime(df.loc[df.split == "val", "fecha"], utc=True).max()),
                "fn_cost": args.fn_cost, "max_fp_validation": args.max_fp,
                "test_used": False, "costs_are_scenarios": args.fp_costs != [5.] or args.fn_cost != 200.}
    joblib.dump({"manifest": manifest, "candidates": candidates}, args.output / "bundle.joblib")
    manifest["bundle_sha256"] = fingerprint(args.output / "bundle.joblib")
    save_json(args.output / "manifest.json", manifest)
    lines = ["# Modelado supervisado — ejecución", "", f"Ganador en val_search: **{winner}**.",
             "Test permanece cerrado. Consigna: FP=5 y FN=200; otros costos son sensibilidad.", "",
             "| Modelo | PR-AUC (trapecios) | Average precision | ROC-AUC |", "|---|---:|---:|---:|"]
    for r in sorted(comparison, key=lambda row: row["pr_auc"], reverse=True):
        lines.append(f"| {r['model']} | {r['pr_auc']:.6f} | {r['average_precision']:.6f} | {r['roc_auc']:.6f} |")
    lines += ["", "Selección en primer 60% de val, Platt en siguiente 20%, umbrales en último 20%.",
              "Las métricas del último bloque no son independientes de la política, cuyos umbrales se ajustan allí.",
              "Las variables recibidas ya se seleccionaron sobre val; estas particiones no revierten esa selección previa.",
              "Consultar manifest.json para costos, calibración, versiones, filas y configuración; trials_*.json para las búsquedas."]
    (args.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Ganador: {winner}. Artefactos: {args.output}. Test no utilizado.")


def evaluate_test(args):
    if not args.bundle:
        raise ValueError("--evaluate-test requiere --bundle del modelo congelado.")
    report = args.bundle.parent / "test_report.json"
    lock = args.bundle.parent / "test_evaluation_started.json"
    if report.exists() or lock.exists():
        raise ValueError("Evaluación test ya iniciada/finalizada en esta ejecución; no se permite repetir automáticamente.")
    manifest = json.loads((args.bundle.parent / "manifest.json").read_text(encoding="utf-8"))
    if fingerprint(args.bundle) != manifest["bundle_sha256"]:
        raise ValueError("El bundle cambió desde su congelamiento.")
    bundle = joblib.load(args.bundle)
    paths = {k: Path(v) if v else None for k, v in manifest["paths"].items()}
    exclude = manifest.get("exclude_anomaly_cols", [])
    rule_cols = manifest.get("rule_cols", [])
    rules_path = paths.get("rules")
    dev, _, _, _ = load_data(paths["base"], paths["features"], paths["anomalies"], ["train", "val"], exclude, rules_path, rule_cols)
    current_hash = hashlib.sha256(pd.util.hash_pandas_object(dev, index=False).values.tobytes()).hexdigest()
    if current_hash != manifest["development_hash"]:
        raise ValueError("Los datos/features de desarrollo cambiaron desde el entrenamiento.")
    # Record consumption BEFORE reading test, also preventing concurrent evaluation.
    with lock.open("x", encoding="utf-8") as stream:
        json.dump({"bundle_sha256": manifest["bundle_sha256"]}, stream)
    df, _, _, _ = load_data(paths["base"], paths["features"], paths["anomalies"], ["test"], exclude, rules_path, rule_cols)
    dates = pd.to_datetime(df.fecha, utc=True, errors="raise")
    if df.empty or dates.isna().any() or dates.min() < pd.Timestamp(manifest["val_end"]):
        raise ValueError("Test vacío o no posterior a val.")
    if set(df.fraude.unique()) != {0, 1}:
        raise ValueError("Test debe contener ambas clases y etiquetas válidas.")
    results = {}
    for name, candidate in bundle["candidates"].items():
        raw = candidate["model"].predict_proba(candidate["encoder"].transform(df))[:, 1]
        p = calibrate(raw, candidate["calibrator"])
        results[name] = {"raw": metrics(df.fraude, raw), "calibrated": metrics(df.fraude, p), "scenarios": {}}
        for cost, policy in candidate["policies"].items():
            results[name]["scenarios"][cost] = policy_metrics(df.fraude, p, df.monto, policy["threshold"], float(cost), manifest["loss_fraction"], manifest["fn_cost"])
        export_curves(args.bundle.parent, f"{name}_test", df.fraude, p)
    save_json(report, {"winner_frozen_on_val": manifest["winner"], "rows": len(df), "results": results,
                       "note": "No se elige modelo ni umbral con estos resultados."})
    print(f"Evaluación final guardada en {report}")


def optional_path(value):
    return None if value.lower() == "none" else Path(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("data/processed/dataset_base.parquet"))
    parser.add_argument("--features", type=optional_path, default=Path("data/features/features_seleccionadas_end_to_end.parquet"))
    parser.add_argument("--anomalies", type=optional_path, default=Path("data/features/scores_anomalias.parquet"))
    parser.add_argument("--rules", type=optional_path, default=None,
                        help="Ruta al Parquet de reglas de asociación satélite (opcional).")
    parser.add_argument("--rule-cols", nargs="*", default=["pct_heuristico_fraude"],
                        help="Columnas de reglas a incorporar.")
    parser.add_argument("--samplers", nargs="+", default=["tpe", "cmaes"],
                        choices=["tpe", "cmaes"], help="Samplers de Optuna a ejecutar.")
    parser.add_argument("--groups", nargs="*", default=None,
                        help="Grupos de features a evaluar (p. ej. full full_plus_rules).")
    parser.add_argument("--output", type=Path, default=Path("models/supervised/first_run"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=20, help="Trials por sampler y conjunto de variables.")
    parser.add_argument("--max-trees", type=int, default=800)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--fp-costs", type=float, nargs="+", default=[5., 10., 20.])
    parser.add_argument("--fn-cost", type=float, default=200., help="Costo fijo por FN según consigna.")
    parser.add_argument("--max-fp", type=int, default=100, help="Tope absoluto de FP en val_policy; hipótesis operativa configurable.")
    parser.add_argument("--loss-fraction", type=float, default=1.)
    parser.add_argument("--exclude-anomaly-cols", nargs="*", default=[],
                        help="Columnas del Parquet de anomalías a no usar (p. ej. un consolidador supervisado in-sample).")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args()
    if args.trials < 1 or args.max_trees < 1 or args.jobs < 1:
        parser.error("trials, max-trees y jobs deben ser positivos.")
    if not 0 <= args.loss_fraction <= 1 or any(not np.isfinite(c) or c < 0 for c in args.fp_costs):
        parser.error("Costos deben ser finitos >= 0; loss-fraction en [0,1].")
    if not np.isfinite(args.fn_cost) or args.fn_cost < 0 or args.max_fp < 0:
        parser.error("fn-cost y max-fp deben ser no negativos y finitos.")
    if args.evaluate_test:
        evaluate_test(args)
    else:
        develop(args)


if __name__ == "__main__":
    main()
