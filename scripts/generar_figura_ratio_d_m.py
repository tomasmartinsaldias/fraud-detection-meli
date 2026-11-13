import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

NAVY = '#2D3277'
MELI_YELLOW = '#FFD100'
GRAY_MUTED = '#bdc1c6'
GRAY_TEXT = '#5f6368'

out_dir = Path(__file__).resolve().parent.parent / "presentacion" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(Path(__file__).resolve().parent.parent / "data" / "processed" / "dataset_base.parquet")
corr, _ = spearmanr(df['d'].dropna(), df['m'].dropna())

fig, ax = plt.subplots(figsize=(6.8, 3.8), dpi=150)
fig.patch.set_facecolor('none')
ax.set_facecolor('none')

categories = ['Normal (d ≈ m)\nratio ≤ 1.0', 'Atípico (d > m)\n1.0 < ratio ≤ 2.0', 'Quiebre Extremo (d >> m)\nratio > 2.0']
fraud_rates = [4.20, 11.04, 28.75]
bar_colors = [GRAY_MUTED, NAVY, MELI_YELLOW]
edge_colors = ['none', 'none', NAVY]

bars = ax.bar(categories, fraud_rates, color=bar_colors, edgecolor=edge_colors, lw=2, width=0.48)

ax.text(0, 4.20 + 0.8, "4.2%", ha='center', va='bottom', fontsize=11, fontweight='600', color=GRAY_TEXT)
ax.text(1, 11.04 + 0.8, "11.0% (2.6x)", ha='center', va='bottom', fontsize=11.5, fontweight='700', color=NAVY)
ax.text(2, 28.75 + 0.8, "28.8% (Casi 7x)", ha='center', va='bottom', fontsize=13, fontweight='800', color=NAVY)

ax.set_ylim(0, 36)
ax.set_ylabel("Tasa de Fraude (%)", fontsize=10.5, fontweight='600', color=GRAY_TEXT)
ax.set_title("Quiebre de Colinealidad d vs m: Salto de Fraude x5 a x7", fontsize=12, fontweight='700', loc='left', color=NAVY, pad=12)

ax.annotate('Multiplica el fraude hasta casi 7x\n(+132 bps en el modelo)', 
             xy=(2, 29), xytext=(1.05, 32.5),
             arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.5, connectionstyle="arc3,rad=-0.12"),
             fontsize=9.5, fontweight='700', color=NAVY)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#dadce0')
ax.yaxis.set_visible(False)

plt.tight_layout()
out_path = out_dir / "fig_ratio_d_m.svg"
plt.savefig(out_path, format='svg', bbox_inches='tight', transparent=True)
plt.close()
print(f"Figura simplificada guardada en: {out_path}")
