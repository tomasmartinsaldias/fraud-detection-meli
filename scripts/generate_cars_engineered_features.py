"""
Generación de Reglas de Asociación de Clase (CARs) con variables de ingeniería transaccional
ordenadas por impacto integral (WRAcc / Fraudes Netos Extra) incorporando F1-Score.

Excluye scores de anomalía y el score base de MELI.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd


def load_datasets(base_path: Path, features_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not base_path.exists():
        raise FileNotFoundError(f"No existe: {base_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"No existe: {features_path}")

    base_df = pd.read_parquet(base_path, columns=["transaction_id", "split", "fraude"])
    feats_df = pd.read_parquet(features_path)

    merged = base_df.merge(feats_df, on="transaction_id")
    train_df = merged[merged["split"] == "train"].copy()
    val_df = merged[merged["split"] == "val"].copy()

    return train_df, val_df


def get_percentile(train_series: pd.Series, threshold: float) -> int:
    clean = train_series.dropna()
    pct = (clean <= threshold).mean() * 100.0
    return int(round(pct))


def build_engineered_items(df: pd.DataFrame, train_ref: pd.DataFrame) -> pd.DataFrame:
    items = pd.DataFrame(index=df.index)
    items["fraude"] = (df["fraude"] == 1)

    # 1. Flags lógicos e indicadores de zona de riesgo generados por build_features.py
    items["flag_alto_riesgo_nop == 1"] = (df["flag_alto_riesgo_nop"] == 1)
    items["is_f_cero_neg == 1"] = (df["is_f_cero_neg"] == 1)
    items["is_h_cero == 1"] = (df["is_h_cero"] == 1)
    items["is_e_cero == 1"] = (df["is_e_cero"] == 1)
    items["is_o_missing == 0"] = (df["is_o_missing"] == 0)
    items["es_madrugada == 1"] = (df["es_madrugada"] == 1)
    items["is_monto_alto == 1"] = (df["is_monto_alto"] == 1)

    def add_cut(col_name: str, op: str, q: float, is_int: bool = False, decimals: int = 2):
        val = float(train_ref[col_name].quantile(q))
        pct = get_percentile(train_ref[col_name], val)
        val_str = f"{val:.0f}" if is_int else f"{val:.{decimals}f}"
        item_key = f"{col_name} {op} {val_str} (p{pct})"
        if op == ">":
            items[item_key] = (df[col_name] > val)
        elif op == "<=":
            items[item_key] = (df[col_name] <= val)
        elif op == ">=":
            items[item_key] = (df[col_name] >= val)

    # 2. Ratios de tensión monetaria
    add_cut("ratio_monto_m", ">", 0.90, decimals=2)
    add_cut("ratio_monto_m", ">", 0.75, decimals=2)

    add_cut("ratio_monto_f", ">", 0.90, decimals=2)
    add_cut("ratio_monto_f", ">", 0.75, decimals=2)

    add_cut("ratio_monto_ag_bayes", ">", 0.90, decimals=2)
    add_cut("ratio_monto_ag_bayes", ">", 0.75, decimals=2)

    add_cut("ratio_monto_b", ">", 0.90, decimals=2)
    add_cut("ratio_monto_h", ">", 0.90, decimals=2)
    add_cut("ratio_monto_l", ">", 0.90, decimals=2)
    add_cut("ratio_monto_q", ">", 0.90, decimals=2)

    # 3. Interacciones entre variables (d y m)
    pct_rdm_1 = get_percentile(train_ref["ratio_d_m"], 1.0)
    pct_rdm_05 = get_percentile(train_ref["ratio_d_m"], 0.5)
    items[f"ratio_d_m >= 1.00 (p{pct_rdm_1})"] = (df["ratio_d_m"] >= 1.0)
    items[f"ratio_d_m >= 0.50 (p{pct_rdm_05})"] = (df["ratio_d_m"] >= 0.5)

    add_cut("diff_m_d", "<=", 0.10, is_int=True)
    add_cut("prod_b_log_c", ">", 0.90, decimals=2)

    # 4. Features bayesianas y semánticas (Categoría j y Texto i)
    add_cut("te_j_bayes", ">", 0.90, decimals=4)
    add_cut("te_j_bayes", ">", 0.75, decimals=4)

    add_cut("score_texto_nb", ">", 0.90, decimals=4)
    add_cut("score_texto_nb", ">", 0.75, decimals=4)

    add_cut("ratio_monto_j_bayes", ">", 0.90, decimals=2)
    add_cut("freq_j", "<=", 0.25, decimals=2)

    # 5. Componentes semánticas LSA del título
    add_cut("text_svd_4", ">", 0.90, decimals=3)
    add_cut("text_svd_7", ">", 0.90, decimals=3)

    # 6. Compresiones logarítmicas de antigüedad/acumulación (log_l, log_m, log_f)
    add_cut("log_l", "<=", 0.10, decimals=2)
    add_cut("log_l", "<=", 0.25, decimals=2)

    add_cut("log_m", "<=", 0.10, decimals=2)
    add_cut("log_m", "<=", 0.25, decimals=2)

    add_cut("log_f", "<=", 0.25, decimals=2)

    # 7. Ciclo temporal continuo
    add_cut("cos_hora", ">", 0.75, decimals=2)

    return items


def is_incompatible_combo(combo: tuple[str, ...]) -> bool:
    variables = [item.split()[0] for item in combo]
    return len(variables) != len(set(variables))


def mine_cars(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    min_support: float = 0.003,      # Mínimo 0.3% (~315 casos en train)
    min_lift: float = 1.50,          # Al menos 1.5x de lift
    max_length: int = 3,
) -> pd.DataFrame:
    n_train_total = len(train_df)
    n_val_total = len(val_df)

    total_frauds_tr = int(train_df["fraude"].sum())
    total_frauds_val = int(val_df["fraude"].sum())

    base_rate_tr = total_frauds_tr / n_train_total
    base_rate_val = total_frauds_val / n_val_total

    tr_items = build_engineered_items(train_df, train_df)
    val_items = build_engineered_items(val_df, train_df)

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

    res_df = res_df.sort_values(
        by=["wracc_train", "f1_score_train", "lift_train"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    res_df.insert(0, "regla_id", [f"ENG_CAR_{i+1:04d}" for i in range(len(res_df))])
    return res_df


def main():
    parser = argparse.ArgumentParser(description="Generar CSV de Reglas CARs ordenado por WRAcc.")
    parser.add_argument("--base_path", default="data/processed/dataset_base.parquet", help="Ruta a dataset_base.parquet")
    parser.add_argument("--features_path", default="data/features/features_transaccionales.parquet", help="Ruta a features_transaccionales.parquet")
    parser.add_argument("--output_csv", default="data/experiments/reglas_cars_features_ingenieria.csv", help="Ruta del CSV completo")
    parser.add_argument("--output_top_csv", default="data/experiments/reglas_cars_features_ingenieria_top100.csv", help="Ruta del CSV top 100")
    parser.add_argument("--min_support", type=float, default=0.003, help="Soporte minimo en train (default: 0.003 = 0.3 pct)")
    parser.add_argument("--min_lift", type=float, default=1.50, help="Lift minimo en train (default: 1.50x)")
    args = parser.parse_args()

    base_f = Path(args.base_path)
    feats_f = Path(args.features_path)
    out_f = Path(args.output_csv)
    top_f = Path(args.output_top_csv)

    print(f"Cargando datasets:\n - Base: {base_f}\n - Features: {feats_f}")
    train_df, val_df = load_datasets(base_f, feats_f)
    print(f"Train: {len(train_df):,} filas (Fraude: {train_df['fraude'].mean()*100:.2f}%)")
    print(f"Val:   {len(val_df):,} filas (Fraude: {val_df['fraude'].mean()*100:.2f}%)\n")

    cars = mine_cars(train_df, val_df, min_support=args.min_support, min_lift=args.min_lift, max_length=3)
    print(f"\nReglas generadas exitosamente: {len(cars):,}")

    out_f.parent.mkdir(parents=True, exist_ok=True)
    cars.to_csv(out_f, index=False, encoding="utf-8-sig")
    print(f"CSV Completo guardado en: {out_f} ({out_f.stat().st_size / 1024:.1f} KB)")

    top100 = cars.head(100)
    top100.to_csv(top_f, index=False, encoding="utf-8-sig")
    print(f"CSV Top 100 guardado en: {top_f} ({top_f.stat().st_size / 1024:.1f} KB)")

    print("\n--- TOP 10 REGLAS POR WRACC (IMPACTO NETO) ---")
    for _, r in cars.head(10).iterrows():
        print(f"[{r.regla_id}] {r.regla}")
        print(f"         Train: WRAcc = {r.wracc_train:.4f} (+{r.fraudes_netos_extra_train:.0f} fraudes netos) | F1 = {r.f1_score_train:.3f}")
        print(f"                Conf = {r.confianza_train_pct}% | Recall = {r.recall_train_pct}% | Lift = {r.lift_train}x | N = {r.casos_train}")
        print(f"         Val:   WRAcc = {r.wracc_val_oot:.4f} (+{r.fraudes_netos_extra_val:.0f} fraudes netos) | F1 = {r.f1_score_val_oot:.3f} | Ratio = {r.ratio_estabilidad_oot}x\n")


if __name__ == "__main__":
    main()
