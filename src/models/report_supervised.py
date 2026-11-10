"""Render frozen experiment results as Markdown and standalone SVG curves."""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path


def chart(path, series, xlabel, ylabel, title):
    colors = ["#2563eb", "#ea580c", "#059669", "#7c3aed"]
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="800" height="530" viewBox="0 0 800 530">',
             '<rect width="800" height="530" fill="white"/>',
             f'<text x="60" y="30" font-family="sans-serif" font-size="20">{html.escape(title)}</text>']
    for tick in range(6):
        value = tick / 5
        x, y = 70 + value * 660, 390 - value * 330
        parts += [f'<path d="M {x} 60 V 390 M 70 {y} H 730" stroke="#e2e8f0" fill="none"/>',
                  f'<text x="{x}" y="412" font-size="12" text-anchor="middle">{value:.1f}</text>',
                  f'<text x="55" y="{y+4}" font-size="12" text-anchor="end">{value:.1f}</text>']
    for index, (name, points) in enumerate(series):
        # Limit exported SVG size while preserving endpoints.
        step = max(1, len(points) // 1500)
        display = points[::step] + points[-1:]
        coords = " ".join(f"{70 + x * 660:.2f},{390 - y * 330:.2f}" for x, y in display)
        color = colors[index % len(colors)]
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="1.8"/>')
        parts.append(f'<text x="70" y="{455 + index*18}" font-family="sans-serif" font-size="12" fill="{color}">{html.escape(name)}</text>')
    parts += [f'<text x="400" y="435" text-anchor="middle" font-family="sans-serif" font-size="14">{xlabel}</text>',
              f'<text transform="translate(20 230) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="14">{ylabel}</text>', '</svg>']
    path.write_text("\n".join(parts), encoding="utf-8")


def best_of(manifest, prefix):
    rows = [r for r in manifest["comparison"] if r["model"].startswith(prefix)]
    return max(rows, key=lambda r: r["pr_auc"])["model"] if rows else None


def anomaly_note(manifest):
    anomalies = manifest["paths"].get("anomalies") or ""
    excluded = manifest.get("exclude_anomaly_cols", [])
    if not anomalies:
        return "Corrida sin scores de anomalía."
    if "ian_scores" in anomalies:
        return "Los scores de anomalía son la alternativa IF/PCA/kNN de Ian, no la entrega completa de Franco."
    note = f"Scores de anomalía: entrega oficial de Franco (`{Path(anomalies).name}`)"
    if excluded:
        names = ", ".join(f"`{c}`" for c in excluded)
        note += f", excluyendo {names} (consolidador supervisado ajustado in-sample sobre train)"
    return note + "."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/RESULTADOS_IAN.md"))
    parser.add_argument("--figures", default="figures_ian", help="Subcarpeta de figuras junto al Markdown.")
    parser.add_argument("--title", default="Resultados de modelado — Ian")
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text(encoding="utf-8"))
    test_path = args.run / "test_report.json"
    test = json.loads(test_path.read_text(encoding="utf-8")) if test_path.exists() else None
    winner = manifest["winner"]
    features = Path(manifest["paths"].get("features") or "sin features").name
    lines = [f"# {args.title}", "", f"Corrida: `{args.run}`. Semilla: {manifest['seed']}. Features: `{features}`.",
             f"Modelo elegido **solo con val_search**: `{winner}`.", "",
             "## Comparación de búsqueda (val_search)", "",
             "| Modelo | PR-AUC | AP | ROC-AUC |", "|---|---:|---:|---:|"]
    for row in sorted(manifest["comparison"], key=lambda r: r["pr_auc"], reverse=True):
        lines.append(f"| {row['model']} | {row['pr_auc']:.4f} | {row['average_precision']:.4f} | {row['roc_auc']:.4f} |")
    lines += ["", "## Búsquedas (TPE + pruning vs. CMA-ES, mismo presupuesto)", "",
              "| Estudio | Trials | Podados | Mejor PR-AUC | Segundos |", "|---|---:|---:|---:|---:|"]
    for path in sorted(args.run.glob("trials_*.json")):
        study = json.loads(path.read_text(encoding="utf-8"))
        lines.append(f"| {path.stem} | {len(study['trials'])} | {sum(t['state'] == 'PRUNED' for t in study['trials'])} | {study['best_pr_auc']:.4f} | {study['seconds']:.1f} |")

    contract = {"Baseline (logística, crudas)": "logistic_raw",
                "Con features transaccionales": best_of(manifest, "lgb_transactional_"),
                "Completo (+ scores de anomalías)": best_of(manifest, "lgb_full_")}
    contract = {k: v for k, v in contract.items() if v}
    header = "| Nivel | Modelo | PR-AUC val_policy | ROC-AUC val_policy |"
    rule = "|---|---|---:|---:|"
    if test:
        header += " PR-AUC test | ROC-AUC test |"
        rule += "---:|---:|"
    lines += ["", "## Comparación del contrato (probabilidades calibradas)", "",
              "Para cada nivel se toma el mejor LightGBM de búsqueda según val_search (sin mirar test).", "", header, rule]
    for level, name in contract.items():
        v = manifest["policy_validation"][name]["calibrated"]
        row = f"| {level} | `{name}` | {v['pr_auc']:.4f} | {v['roc_auc']:.4f} |"
        if test:
            t = test["results"][name]["calibrated"]
            row += f" {t['pr_auc']:.4f} | {t['roc_auc']:.4f} |"
        lines.append(row)

    results = test["results"] if test else manifest["policy_validation"]
    label = "test final" if test else "val_policy (umbrales ajustados en este bloque)"
    lines += ["", f"## Negocio — {label}", "",
              f"Costo principal: FP=5, FN=200. Umbrales elegidos en val_policy, máximo {manifest['max_fp_validation']} FP en ese bloque.",
              "", "| Modelo | PR-AUC calibrado | Umbral | Precision | Recall | FP | FN | Costo | Ahorro vs permitir todo |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, result in results.items():
        policy = result["scenarios"]["5.0"]
        lines.append(f"| {name} | {result['calibrated']['pr_auc']:.4f} | {policy['threshold']:.5f} | {policy['precision']:.3f} | {policy['recall']:.3f} | {int(policy['fp'])} | {int(policy['fn'])} | {policy['cost']:,.0f} | {policy['savings_vs_allow_all']:,.0f} |")
    selected = results[winner]
    baseline = results["logistic_raw"]["scenarios"]["5.0"]
    principal = selected["scenarios"]["5.0"]
    lines += ["", "## Interpretación", "",
              f"El modelo elegido detecta {principal['recall']:.1%} de los fraudes con precision {principal['precision']:.1%} en {label}.",
              f"Su costo es {principal['cost']:,.0f}, frente a {baseline['cost']:,.0f} de la regresión logística bajo las políticas congeladas.",
              f"La diferencia de costo frente a logística es {baseline['cost'] - principal['cost']:,.0f}; un valor negativo significa que el ganador por PR-AUC cuesta más.",
              "El ganador por PR-AUC no necesariamente minimiza el costo con una restricción operativa. No se cambia de ganador por observar test.",
              f"Calibración del ganador: Brier sin calibrar {selected['raw']['brier']:.6f}, con Platt {selected['calibrated']['brier']:.6f}.",
              "", "### Sensibilidad del ganador", "", "| Costo FP | Umbral congelado | FP | FN | Costo total |", "|---|---:|---:|---:|---:|"]
    for cost, policy in selected["scenarios"].items():
        lines.append(f"| {cost} | {policy['threshold']:.5f} | {int(policy['fp'])} | {int(policy['fn'])} | {policy['cost']:,.0f} |")

    if "amount_fn_val_only" in manifest["policy_validation"][winner]:
        lines += ["", "### Sensibilidad del contrato: C_FN = monto defraudado (solo val_policy)", "",
                  "Escenario de CONTRATOS.md: C_FP=5 y C_FN = monto de cada fraude no detectado. El umbral se elige y se mide",
                  "en val_policy, por lo que es optimista, y **no se evaluó en test** (decisión explícita para no reabrirlo).", "",
                  "| Modelo | Umbral | FP | FN | Precision | Recall | Pérdida + fricción | Ahorro vs permitir todo |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name, result in manifest["policy_validation"].items():
            a = result["amount_fn_val_only"]
            lines.append(f"| {name} | {a['threshold']:.5f} | {int(a['fp'])} | {int(a['fn'])} | {a['precision']:.3f} | {a['recall']:.3f} | {a['cost']:,.0f} | {a['savings_vs_allow_all']:,.0f} |")

    importance_path = args.run / f"{winner}_importance.json"
    if importance_path.exists():
        importance = json.loads(importance_path.read_text(encoding="utf-8"))
        lines += ["", "### Variables principales (gain, no causalidad)", ""]
        for name, value in sorted(importance.items(), key=lambda pair: pair[1], reverse=True)[:8]:
            lines.append(f"- `{name}`: {value:,.1f}.")
    lines += ["", "## Límites y próximos pasos", "",
              "La selección previa de variables ya había consultado val; las cifras de búsqueda son exploratorias.",
              f"Entorno de la corrida: Python {manifest.get('python', '3.12.14')} con las dependencias fijadas en requirements.txt.",
              anomaly_note(manifest),
              "Costos del período evaluado, sin extrapolación mensual. El tope de FP de validación no es una garantía para otro período.",
              "Antes de producción: acordar capacidad real de revisión, monitorear calibración y deriva, medir costos de contacto y pérdida de clientes.",
              "Datos adicionales deseables: identidad de usuario/dispositivo, historial causal, chargebacks confirmados y fecha de confirmación de etiqueta.",
              "No usar resultados finales para una nueva ronda de ajuste: se necesitaría otro período fuera de muestra."]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plots = args.output.parent / args.figures
    plots.mkdir(exist_ok=True)
    names = list(dict.fromkeys(list(contract.values()) + [winner]))
    blocks = [("val_policy", "val_policy")] + ([("test", "test final")] if test else [])
    kinds = (("pr", "recall", "precision", "Recall", "Precision"),
             ("roc", "fpr", "tpr", "Tasa de falsos positivos", "Tasa de verdaderos positivos"))
    lines += ["", "## Curvas"]
    for suffix, title in blocks:
        for kind, xkey, ykey, xlabel, ylabel in kinds:
            series = []
            for name in names:
                with (args.run / f"{name}_{suffix}_{kind}.csv").open() as stream:
                    points = [(float(r[xkey]), float(r[ykey])) for r in csv.DictReader(stream)]
                series.append((name, points))
            path = plots / f"{kind}_{suffix}.svg"
            chart(path, series, xlabel, ylabel, f"Curva {kind.upper()} — {title}")
            lines += ["", f"![Curva {kind.upper()} {title}]({args.figures}/{path.name})"]
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
