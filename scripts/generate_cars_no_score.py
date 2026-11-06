"""
Generación de Reglas de Asociación de Clase (CARs) sin utilizar 'score'.

Excluye 'score' para identificar patrones intrínsecos de fraude sobre montos,
comportamiento, canales, contadores y temporalidad.

Salida ordenada por impacto integral (WRAcc / Fraudes Netos Extra) incorporando F1-Score:
- WRAcc (Weighted Relative Accuracy): pondera el tamaño del subgrupo por el exceso de fraude.
- Fraudes Netos Extra: fraudes capturados por encima de lo esperado por puro azar.
- F1-Score: equilibrio armónico entre Confianza (Precisión) y Recall (Cobertura de fraude).
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd


def load_datasets(parquet_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(parquet_path)
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    return train_df, val_df


def get_percentile(train_series: pd.Series, threshold: float) -> int:
    clean = train_series.dropna()
    pct = (clean <= threshold).mean() * 100.0
    return int(round(pct))


def build_binary_items(df: pd.DataFrame, train_ref: pd.DataFrame) -> pd.DataFrame:
    items = pd.DataFrame(index=df.index)
    items["fraude"] = (df["fraude"] == 1)

    # 1. Variables discretas y categóricas
    items["n == 0"] = (df["n"] <= 0)
    items["s == 0"] = (df["s"] <= 0)
    items["a != 4"] = (df["a"] != 4)
    items["hora < 6"] = (df["fecha"].dt.hour < 6)

    items["g == 'BR'"] = (df["g"] == "BR")
    items["g == 'AR'"] = (df["g"] == "AR")
    items["g == 'UY'"] = (df["g"] == "UY")

    items["o == 'N'"] = (df["o"] == "N")
    items["o == 'Y'"] = (df["o"] == "Y")
    items["o is null"] = df["o"].isna()

    # 2. Variables continuas con percentiles calculados en Train
    # Monto
    m_p90 = float(train_ref["monto"].quantile(0.90))
    m_p75 = float(train_ref["monto"].quantile(0.75))
    m_p25 = float(train_ref["monto"].quantile(0.25))
    items[f"monto > {m_p90:.2f} (p90)"] = (df["monto"] > m_p90)
    items[f"monto > {m_p75:.2f} (p75)"] = (df["monto"] > m_p75)
    items[f"monto <= {m_p25:.2f} (p25)"] = (df["monto"] <= m_p25)

    # b
    b_p95 = float(train_ref["b"].quantile(0.95))
    b_p90 = float(train_ref["b"].quantile(0.90))
    b_p25 = float(train_ref["b"].quantile(0.25))
    items[f"b > {b_p95:.2f} (p95)"] = (df["b"] > b_p95)
    items[f"b > {b_p90:.2f} (p90)"] = (df["b"] > b_p90)
    items[f"b <= {b_p25:.2f} (p25)"] = (df["b"] <= b_p25)

    # m
    m_p10 = float(train_ref["m"].quantile(0.10))
    m_p25 = float(train_ref["m"].quantile(0.25))
    pct_m10 = get_percentile(train_ref["m"], m_p10)
    pct_m25 = get_percentile(train_ref["m"], m_p25)
    items[f"m <= {m_p10:.0f} (p{pct_m10})"] = (df["m"] <= m_p10)
    items[f"m <= {m_p25:.0f} (p{pct_m25})"] = (df["m"] <= m_p25)

    # l
    l_p10 = float(train_ref["l"].quantile(0.10))
    l_p25 = float(train_ref["l"].quantile(0.25))
    pct_l10 = get_percentile(train_ref["l"], l_p10)
    pct_l25 = get_percentile(train_ref["l"], l_p25)
    items[f"l <= {l_p10:.0f} (p{pct_l10})"] = (df["l"] <= l_p10)
    items[f"l <= {l_p25:.0f} (p{pct_l25})"] = (df["l"] <= l_p25)

    # c
    c_p90 = float(train_ref["c"].quantile(0.90))
    c_p25 = float(train_ref["c"].quantile(0.25))
    pct_c90 = get_percentile(train_ref["c"], c_p90)
    pct_c25 = get_percentile(train_ref["c"], c_p25)
    items[f"c > {c_p90:.0f} (p{pct_c90})"] = (df["c"] > c_p90)
    items[f"c <= {c_p25:.0f} (p{pct_c25})"] = (df["c"] <= c_p25)

    # d
    pct_d1 = get_percentile(train_ref["d"], 1.0)
    pct_d5 = get_percentile(train_ref["d"], 5.0)
    items[f"d == 1 (p{pct_d1})"] = (df["d"] == 1.0)
    items[f"d <= 5 (p{pct_d5})"] = (df["d"] <= 5.0)

    # f
    f_p25 = float(train_ref["f"].quantile(0.25))
    f_p50 = float(train_ref["f"].quantile(0.50))
    pct_f25 = get_percentile(train_ref["f"], f_p25)
    pct_f50 = get_percentile(train_ref["f"], f_p50)
    items[f"f <= {f_p25:.0f} (p{pct_f25})"] = (df["f"] <= f_p25)
    items[f"f <= {f_p50:.0f} (p{pct_f50})"] = (df["f"] <= f_p50)

    # h
    pct_h0 = get_percentile(train_ref["h"], 0.0)
    h_p25 = float(train_ref["h"].quantile(0.25))
    pct_h25 = get_percentile(train_ref["h"], h_p25)
    items[f"h == 0 (p{pct_h0})"] = (df["h"] == 0.0)
    items[f"h <= {h_p25:.0f} (p{pct_h25})"] = (df["h"] <= h_p25)

    # q
    q_p25 = float(train_ref["q"].quantile(0.25))
    pct_q25 = get_percentile(train_ref["q"], q_p25)
    pct_q35 = get_percentile(train_ref["q"], 0.10)
    items[f"q <= {q_p25:.2f} (p{pct_q25})"] = (df["q"] <= q_p25)
    items[f"q <= 0.10 (p{pct_q35})"] = (df["q"] <= 0.10)

    return items


def is_incompatible_combo(combo: tuple[str, ...]) -> bool:
    variables = [item.split()[0] for item in combo]
    return len(variables) != len(set(variables))


def mine_cars_no_score(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    min_support: float = 0.003,      # Mínimo 0.3% (~315 transacciones en train)
    min_lift: float = 1.50,          # Umbral de lift
    max_length: int = 3,
) -> pd.DataFrame:
    n_train_total = len(train_df)
    n_val_total = len(val_df)

    total_frauds_tr = int(train_df["fraude"].sum())
    total_frauds_val = int(val_df["fraude"].sum())

    base_rate_tr = total_frauds_tr / n_train_total
    base_rate_val = total_frauds_val / n_val_total

    tr_items = build_binary_items(train_df, train_df)
    val_items = build_binary_items(val_df, train_df)

    item_cols = [c for c in tr_items.columns if c != "fraude"]
    print(f"Total de condiciones analizadas: {len(item_cols)}")

    raw_rules = []

    for k in range(1, max_length + 1):
        print(f"Evaluando combinaciones de longitud {k}...")
        for combo in combinations(item_cols, k):
            if is_incompatible_combo(combo):
                continue

            mask_tr = tr_items[list(combo)].all(axis=1)
            supp_tr = float(mask_tr.mean())
            if supp_tr < min_support:
                continue

            n_tr = int(mask_tr.sum())
            fraudes_tr = int(tr_items.loc[mask_tr, "fraude"].sum())
            conf_tr = fraudes_tr / n_tr if n_tr > 0 else 0.0
            lift_tr = conf_tr / base_rate_tr if base_rate_tr > 0 else 0.0

            if lift_tr < min_lift:
                continue

            # Métricas Train
            recall_tr = fraudes_tr / total_frauds_tr if total_frauds_tr > 0 else 0.0
            f1_tr = (2 * conf_tr * recall_tr) / (conf_tr + recall_tr + 1e-12)
            wracc_tr = (n_tr / n_train_total) * (conf_tr - base_rate_tr)
            fraudes_netos_tr = fraudes_tr - (n_tr * base_rate_tr)

            # Validación Out-of-Time
            mask_val = val_items[list(combo)].all(axis=1)
            supp_val = float(mask_val.mean())
            n_val = int(mask_val.sum())
            fraudes_val = int(val_items.loc[mask_val, "fraude"].sum()) if n_val > 0 else 0
            conf_val = fraudes_val / n_val if n_val > 0 else 0.0
            lift_val = conf_val / base_rate_val if base_rate_val > 0 else 0.0

            recall_val = fraudes_val / total_frauds_val if total_frauds_val > 0 else 0.0
            f1_val = (2 * conf_val * recall_val) / (conf_val + recall_val + 1e-12) if (conf_val + recall_val) > 0 else 0.0
            wracc_val = (n_val / n_val_total) * (conf_val - base_rate_val) if n_val_total > 0 else 0.0
            fraudes_netos_val = fraudes_val - (n_val * base_rate_val)

            ratio_estabilidad = (conf_val / conf_tr) if conf_tr > 0 else 0.0

            raw_rules.append({
                "longitud": k,
                "regla": " AND ".join(combo),
                "wracc_train": round(wracc_tr, 5),
                "fraudes_netos_extra_train": round(fraudes_netos_tr, 1),
                "f1_score_train": round(f1_tr, 4),
                "recall_train_pct": round(recall_tr * 100, 2),
                "confianza_train_pct": round(conf_tr * 100, 2),
                "lift_train": round(lift_tr, 2),
                "casos_train": n_tr,
                "fraudes_train": fraudes_tr,
                "soporte_train_pct": round(supp_tr * 100, 3),
                "wracc_val_oot": round(wracc_val, 5),
                "fraudes_netos_extra_val": round(fraudes_netos_val, 1),
                "f1_score_val_oot": round(f1_val, 4),
                "recall_val_pct": round(recall_val * 100, 2),
                "confianza_val_pct": round(conf_val * 100, 2),
                "lift_val_oot": round(lift_val, 2),
                "casos_val_oot": n_val,
                "fraudes_val_oot": fraudes_val,
                "soporte_val_pct": round(supp_val * 100, 3),
                "ratio_estabilidad_oot": round(ratio_estabilidad, 2),
            })

    res_df = pd.DataFrame(raw_rules)
    if res_df.empty:
        return res_df

    # Ordenar por WRAcc descendente (mayor impacto neto), desempatando por F1-Score y Lift
    res_df = res_df.sort_values(
        by=["wracc_train", "f1_score_train", "lift_train"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    res_df.insert(0, "regla_id", [f"CAR_{i+1:04d}" for i in range(len(res_df))])
    return res_df


def main():
    parser = argparse.ArgumentParser(description="Generar CSV de CARs ordenado por WRAcc.")
    parser.add_argument("--data_path", default="data/processed/dataset_base.parquet", help="Ruta a dataset_base.parquet")
    parser.add_argument("--output_csv", default="data/experiments/reglas_cars_sin_score.csv", help="Ruta del CSV completo")
    parser.add_argument("--output_top_csv", default="data/experiments/reglas_cars_sin_score_top100.csv", help="Ruta del CSV top 100")
    parser.add_argument("--min_support", type=float, default=0.003, help="Soporte minimo en train (default: 0.003 = 0.3 pct)")
    parser.add_argument("--min_lift", type=float, default=1.50, help="Lift minimo en train (default: 1.50x)")
    args = parser.parse_args()

    data_file = Path(args.data_path)
    out_file = Path(args.output_csv)
    top_file = Path(args.output_top_csv)

    print(f"Cargando dataset desde: {data_file}")
    train_df, val_df = load_datasets(data_file)
    print(f"Train: {len(train_df):,} filas (Fraude: {train_df['fraude'].mean()*100:.2f}%)")
    print(f"Val:   {len(val_df):,} filas (Fraude: {val_df['fraude'].mean()*100:.2f}%)\n")

    cars = mine_cars_no_score(train_df, val_df, min_support=args.min_support, min_lift=args.min_lift, max_length=3)
    print(f"\nReglas generadas exitosamente: {len(cars):,}")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    cars.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"CSV Completo ordenado por WRAcc guardado en: {out_file} ({out_file.stat().st_size / 1024:.1f} KB)")

    top100 = cars.head(100)
    top100.to_csv(top_file, index=False, encoding="utf-8-sig")
    print(f"CSV Top 100 guardado en: {top_file} ({top_file.stat().st_size / 1024:.1f} KB)")

    print("\n--- TOP 10 REGLAS POR WRACC (IMPACTO NETO) ---")
    for _, r in cars.head(10).iterrows():
        print(f"[{r.regla_id}] {r.regla}")
        print(f"         Train: WRAcc = {r.wracc_train:.4f} (+{r.fraudes_netos_extra_train:.0f} fraudes netos) | F1 = {r.f1_score_train:.3f}")
        print(f"                Conf = {r.confianza_train_pct}% | Recall = {r.recall_train_pct}% | Lift = {r.lift_train}x | N = {r.casos_train}")
        print(f"         Val:   WRAcc = {r.wracc_val_oot:.4f} (+{r.fraudes_netos_extra_val:.0f} fraudes netos) | F1 = {r.f1_score_val_oot:.3f} | Ratio = {r.ratio_estabilidad_oot}x\n")


if __name__ == "__main__":
    main()
