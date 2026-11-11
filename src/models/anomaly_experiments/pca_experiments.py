"""Comparación exploratoria de tres scores PCA en train y validación temporal."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler


DEFAULT_EXCLUDED = ["c", "k", "i"]
PROTECTED_COLUMNS = {"transaction_id", "split", "fraude", "fecha"}


def frequency_encode(train: pd.Series, val: pd.Series, normal: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Frecuencias de legítimos de train; nulo es categoría y desconocidos valen 0."""
    reference = train.loc[normal]
    counts = reference.value_counts(dropna=True)
    frequencies = counts / len(reference)
    null_frequency = float(reference.isna().mean())

    def transform(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        missing = values.isna()
        unknown = ~missing & ~values.isin(counts.index)
        encoded = values.map(frequencies).astype(float)
        encoded.loc[missing] = null_frequency
        encoded.loc[unknown] = 0.0
        return encoded.to_numpy(), unknown.to_numpy()

    train_encoded, train_unknown = transform(train)
    val_encoded, val_unknown = transform(val)
    return train_encoded, val_encoded, {
        "categories": len(counts), "missing_train_legitimate": int(reference.isna().sum()),
        "unknown_train": int(train_unknown.sum()), "unknown_val": int(val_unknown.sum()),
        "missing_val": int(val.isna().sum()),
    }


def calculate_scores(pca: PCA, x: np.ndarray) -> dict[str, np.ndarray]:
    """Q residual, distancia en el espacio PCA y distancia ponderada por varianza."""
    coordinates = pca.transform(x)
    residual = x - pca.inverse_transform(coordinates)
    eigenvalues = pca.explained_variance_
    if np.any(eigenvalues <= np.finfo(float).eps * eigenvalues.max()):
        raise ValueError("PCA contiene varianzas nulas o numéricamente inestables para T².")
    return {
        "reconstruccion_Q": np.sum(residual ** 2, axis=1),
        "magnitud_cuadrada": np.sum(coordinates ** 2, axis=1),
        "magnitud_ponderada_T2": np.sum(coordinates ** 2 / eigenvalues, axis=1),
    }


def run_experiments(df: pd.DataFrame, features: list[str], seed: int = 42, categorical: list[str] | None = None) -> tuple[list[dict], dict, pd.DataFrame]:
    categorical = categorical or []
    all_features = features + categorical
    required = {"split", "fraude", *all_features}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")
    forbidden = {"fraude", "split", "transaction_id", "fecha"}
    if len(all_features) != len(set(all_features)) or forbidden.intersection(all_features) or not features:
        raise ValueError("Las variables deben ser únicas y excluir identificador, fecha, split y fraude.")

    # No consultar los valores ni etiquetas de test durante la exploración.
    train = df.loc[df["split"] == "train", all_features + ["fraude"]].copy()
    val = df.loc[df["split"] == "val", all_features + ["fraude"]].copy()
    for name, part in (("train", train), ("val", val)):
        if part.empty or part["fraude"].isna().any() or set(part["fraude"].unique()) != {0, 1}:
            raise ValueError(f"{name} debe contener etiquetas 0 y 1, sin faltantes.")
    for col in features:
        if not pd.api.types.is_numeric_dtype(train[col]):
            raise ValueError(f"La variable {col} debe ser numérica.")
    normal = train["fraude"].to_numpy() == 0
    x_train = train[features].to_numpy(dtype=float, na_value=np.nan)
    x_val = val[features].to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(x_train).any() or np.isinf(x_val).any():
        raise ValueError("Las variables contienen valores infinitos.")
    if np.isnan(x_train[normal]).all(axis=0).any():
        raise ValueError("Hay variables completamente vacías entre los legítimos de train.")

    # Todo el preprocesamiento que alimenta PCA se ajusta con train legítimo.
    imputer = SimpleImputer(strategy="median")
    imputer.fit(x_train[normal])
    x_train = imputer.transform(x_train)
    x_val = imputer.transform(x_val)
    categorical_info = {}
    for col in categorical:
        encoded_train, encoded_val, categorical_info[col] = frequency_encode(train[col], val[col], normal)
        x_train = np.column_stack((x_train, encoded_train))
        x_val = np.column_stack((x_val, encoded_val))
    scaler = StandardScaler().fit(x_train[normal])
    x_train = scaler.transform(x_train)
    x_val = scaler.transform(x_val)
    y_train, y_val = train["fraude"].to_numpy(), val["fraude"].to_numpy()
    rows = []
    predictions = pd.DataFrame(index=val.index)

    for fraction in (0.90, 0.95, 0.99):
        pca = PCA(n_components=fraction, svd_solver="full", random_state=seed)
        pca.fit(x_train[normal])
        train_scores = calculate_scores(pca, x_train)
        val_scores = calculate_scores(pca, x_val)
        score_names = list(train_scores)
        for selected in [[name] for name in score_names] + [score_names]:
            score_name = selected[0] if len(selected) == 1 else "tres_scores"
            score_train = np.column_stack([train_scores[name] for name in selected])
            score_val = np.column_stack([val_scores[name] for name in selected])
            if not np.isfinite(score_train).all() or not np.isfinite(score_val).all():
                raise ValueError(f"El score {score_name} contiene valores no finitos.")
            # Escala ajustada únicamente con legítimos, para una logística estable.
            score_scaler = StandardScaler().fit(score_train[normal])
            logistic = LogisticRegression(random_state=seed, max_iter=1000)
            logistic.fit(score_scaler.transform(score_train), y_train)
            probs = logistic.predict_proba(score_scaler.transform(score_val))[:, 1]
            predictions[f"pca_{int(fraction * 100)}_{score_name}"] = probs
            precision, recall, _ = precision_recall_curve(y_val, probs)
            rows.append({
                "objetivo": fraction,
                "componentes": int(pca.n_components_),
                "varianza": float(pca.explained_variance_ratio_.sum()),
                "score": score_name,
                "train_min": float(score_train.min()) if len(selected) == 1 else None,
                "train_max": float(score_train.max()) if len(selected) == 1 else None,
                "val_min": float(score_val.min()) if len(selected) == 1 else None,
                "val_max": float(score_val.max()) if len(selected) == 1 else None,
                "coeficientes": dict(zip(selected, [float(v) for v in logistic.coef_[0]])),
                "pr_auc": float(auc(recall, precision)),
                "average_precision": float(average_precision_score(y_val, probs)),
                "roc_auc": float(roc_auc_score(y_val, probs)),
            })
    metadata = {
        "features": features, "categorical": categorical, "categorical_info": categorical_info, "seed": seed, "train": len(train), "val": len(val),
        "legitimos_train": int(normal.sum()), "fraudes_train": int(y_train.sum()),
        "fraudes_val": int(y_val.sum()), "prevalencia_val": float(y_val.mean()),
    }
    return rows, metadata, predictions


def select_features(df: pd.DataFrame, excluded: list[str], features: list[str] | None, categorical: list[str] | None) -> tuple[list[str], list[str]]:
    """Selecciona por tipo o usa listas explícitas, siempre respetando exclusiones."""
    excluded_set = set(excluded) | PROTECTED_COLUMNS
    if features is not None or categorical is not None:
        numeric = features or []
        categories = categorical or []
        invalid = set(numeric + categories) & excluded_set
        if invalid:
            raise ValueError(f"Columnas seleccionadas también excluidas: {sorted(invalid)}")
        return numeric, categories
    numeric, categories, unsupported = [], [], []
    for col in df.columns:
        if col in excluded_set:
            continue
        series = df[col]
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            numeric.append(col)
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype) or pd.api.types.is_bool_dtype(series):
            categories.append(col)
        else:
            unsupported.append(col)
    if unsupported:
        raise ValueError(f"Tipos no admitidos; excluir o elegir variables explícitas: {unsupported}")
    return numeric, categories


def format_report(rows: list[dict], meta: dict, input_path: Path) -> str:
    lines = [
        "# Experimentos PCA — validación temporal", "",
        f"- Entrada: `{input_path}`; semilla: {meta['seed']}.",
        f"- Variables numéricas: {', '.join(meta['features'])}.",
        f"- Categóricas codificadas por frecuencia: {', '.join(meta['categorical']) or 'ninguna'}.",
        f"- Train: {meta['train']:,} transacciones; {meta['legitimos_train']:,} legítimas y {meta['fraudes_train']:,} fraudes.",
        f"- Validación: {meta['val']:,} transacciones; {meta['fraudes_val']:,} fraudes; prevalencia {meta['prevalencia_val']:.3%}.",
        "- Medianas, frecuencias, escalado y PCA ajustados solo con legítimos de train. La regresión logística usa ambas clases de train.",
        "- Para cada PCA se prueban tres logísticas individuales y otra conjunta; scores estandarizados con legítimos de train. Test queda reservado.", "",
        "| Varianza objetivo | Componentes | Varianza real | Indicadores | PR-AUC | Average precision | ROC-AUC |",
        "| ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(f"| {row['objetivo']:.0%} | {row['componentes']} | {row['varianza']:.2%} | {row['score']} | {row['pr_auc']:.4f} | {row['average_precision']:.4f} | {row['roc_auc']:.4f} |")
    lines += ["", "### Coeficientes de las regresiones", "", "| Objetivo | Modelo | Coeficientes por score estandarizado |", "| ---: | --- | --- |"]
    for row in rows:
        coefficients = ", ".join(f"{name}: {value:.4f}" for name, value in row["coeficientes"].items())
        lines.append(f"| {row['objetivo']:.0%} | {row['score']} | {coefficients} |")
    lines += ["", "### Rango de los scores sin estandarizar", "", "| Objetivo | Score | Train mínimo | Train máximo | Val mínimo | Val máximo |", "| ---: | --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        if row["score"] == "tres_scores":
            continue
        lines.append(f"| {row['objetivo']:.0%} | {row['score']} | {row['train_min']:.3g} | {row['train_max']:.3g} | {row['val_min']:.3g} | {row['val_max']:.3g} |")
    for row in rows:
        if row["score"] == "reconstruccion_Q" and max(row["train_max"], row["val_max"]) < 1e-20:
            lines += ["", f"**Advertencia:** en {row['objetivo']:.0%}, Q es casi cero en train y val; sus diferencias pueden ser errores numéricos, no una señal de anomalía interpretable."]
    if meta["categorical_info"]:
        lines += ["", "### Cobertura de categorías", "", "| Variable | Categorías conocidas | Nulos legítimos train | Desconocidos train | Desconocidos val | Nulos val |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for col, info in meta["categorical_info"].items():
            lines.append(f"| {col} | {info['categories']} | {info['missing_train_legitimate']} | {info['unknown_train']} | {info['unknown_val']} | {info['missing_val']} |")
        lines += ["", "Los nulos reciben su frecuencia propia en legítimos de train; categorías no vistas reciben 0. La frecuencia de train incluye cada legítimo en su propia referencia."]
    lines += [
        "", "## Lectura", "",
        "- Q es la suma de residuos cuadrados en el espacio estandarizado; la magnitud es la suma de coordenadas PCA al cuadrado; T² divide cada término por su varianza.",
        "- PR-AUC es el área trapezoidal bajo la curva precisión-recall; average precision usa otra integración. La prevalencia orienta la comparación con un ranking aleatorio.",
        "- El signo del coeficiente indica si la logística asoció scores mayores con más o menos fraude en train; un signo negativo invierte el ranking.",
        "- La logística es un evaluador supervisado, no parte del detector one-class. No mide el aporte incremental al modelo final y sus probabilidades no están necesariamente calibradas.",
        "- Los scores de train son calculados por un PCA ajustado sobre sus legítimos: pueden ser optimistas frente a observaciones nuevas. Las comparaciones son exploratorias y usan la misma validación.",
        "- Confirmar que todas las variables estén disponibles al momento de la transacción; no asumir causalidad por figurar en el baseline.", "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/dataset_base.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/experiments/pca_report_frequency_90_95_99.md"))
    parser.add_argument("--features", nargs="+", default=None, help="Selección explícita de variables numéricas.")
    parser.add_argument("--categorical", nargs="*", default=None, help="Selección explícita de categóricas para frequency encoding.")
    parser.add_argument("--exclude", nargs="*", default=DEFAULT_EXCLUDED, help="Columnas excluidas (por defecto: c k i). Sustituye la lista predeterminada.")
    parser.add_argument("--export-probs", type=int, choices=[0, 1], default=0, help="1 guarda probabilidades de validación en Parquet; 0 no guarda (predeterminado).")
    parser.add_argument("--probs-output", type=Path, default=Path("data/experiments/pca_probabilidades_val.parquet"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    started = time.perf_counter()
    print("Tiempo de referencia: ~3,2 s por ejecución completa en este equipo y dataset (promedio de 2 corridas previas); puede variar con los datos y el entorno.", flush=True)
    if args.input.resolve() == args.output.resolve() or (args.export_probs and (args.input.resolve() == args.probs_output.resolve() or args.output.resolve() == args.probs_output.resolve())):
        parser.error("Las rutas de entrada, reporte y probabilidades deben ser distintas.")
    # Filtrar en lectura: no cargar registros de test en el banco de pruebas.
    df = pd.read_parquet(args.input, filters=[("split", "in", ["train", "val"])])
    features, categorical = select_features(df, args.exclude, args.features, args.categorical)
    print(f"Columnas excluidas: {', '.join(args.exclude)}; además de transaction_id, split, fraude y fecha.", flush=True)
    if args.export_probs:
        if "transaction_id" not in df:
            raise ValueError("Se requiere transaction_id para exportar probabilidades.")
        ids = df.loc[df["split"] == "val", "transaction_id"]
        if ids.isna().any() or ids.duplicated().any() or not ids.map(lambda value: isinstance(value, str)).all():
            raise ValueError("Los transaction_id de val deben ser strings únicos y no nulos.")
    rows, meta, predictions = run_experiments(df, features, args.seed, categorical)
    report = format_report(rows, meta, args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)
    print(f"Reporte guardado en: {args.output}")
    if args.export_probs:
        predictions.insert(0, "transaction_id", ids.astype("string"))
        args.probs_output.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(args.probs_output, compression="snappy", index=False)
        print(f"Probabilidades de validación guardadas en: {args.probs_output}")
    print(f"Tiempo real del script: {time.perf_counter() - started:.2f} s (sin preparación del entorno por run.bat).")


if __name__ == "__main__":
    main()
