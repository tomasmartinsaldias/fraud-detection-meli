import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

NAVY = '#2D3277'
MELI_YELLOW = '#FFD100'
GRAY_MUTED = '#bdc1c6'
GRAY_TEXT = '#5f6368'
CORAL = '#d93025'

out_dir = Path(__file__).resolve().parent.parent / "presentacion" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(Path(__file__).resolve().parent.parent / "data" / "processed" / "dataset_base.parquet")
corr, _ = spearmanr(df['d'].dropna(), df['m'].dropna())

df['ratio_d_m'] = (df['d'] + 1) / (df['m'] + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=150)
fig.patch.set_facecolor('none')
ax1.set_facecolor('none')
ax2.set_facecolor('none')

# -------------------------------------------------------------
# Panel 1: Colinealidad en EDA entre d y m (Spearman 0.735)
# -------------------------------------------------------------
# Subsample for clear visualization
df_sample = df.sample(n=3500, random_state=42)

# Normal points (ratio <= 1.0)
norm_mask = df_sample['ratio_d_m'] <= 1.0
ax1.scatter(df_sample.loc[norm_mask, 'm'], df_sample.loc[norm_mask, 'd'], 
            alpha=0.25, s=18, color=GRAY_MUTED, label='Normal (r ≤ 1.0, 4.2% fraude)')

# Atipicos / Quiebre (ratio > 1.0)
break_mask = df_sample['ratio_d_m'] > 1.0
ax1.scatter(df_sample.loc[break_mask, 'm'], df_sample.loc[break_mask, 'd'], 
            alpha=0.7, s=28, color=NAVY, label='Quiebre de correlación (r > 1.0)')

# Diagonal reference line
max_val = min(df_sample['m'].quantile(0.98), df_sample['d'].quantile(0.98))
ax1.plot([0, max_val], [0, max_val], '--', color=GRAY_TEXT, lw=1.5, alpha=0.7)

ax1.set_xlim(-5, max_val * 1.05)
ax1.set_ylim(-5, max_val * 1.05)
ax1.set_xlabel("Variable nativa 'm'", fontsize=10.5, fontweight='600', color=GRAY_TEXT)
ax1.set_ylabel("Variable nativa 'd'", fontsize=10.5, fontweight='600', color=GRAY_TEXT)
ax1.set_title(f"Colinealidad en EDA: Spearman r = {corr:.3f}", fontsize=11.5, fontweight='700', loc='left', color=NAVY)

ax1.annotate('Zona de Quiebre Atípico\n(d >> m: alta densidad de fraude)', 
             xy=(max_val*0.2, max_val*0.65), xytext=(max_val*0.1, max_val*0.85),
             arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.5),
             fontsize=9.5, fontweight='700', color=NAVY)

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_color('#dadce0')
ax1.spines['left'].set_color('#dadce0')
ax1.legend(loc='lower right', frameon=False, fontsize=8.5)

# -------------------------------------------------------------
# Panel 2: Tasa de Fraude según ratio_d_m
# -------------------------------------------------------------
categories = ['Normal (d ≈ m)\nratio ≤ 1.0', 'Atípico (d > m)\n1.0 < ratio ≤ 2.0', 'Quiebre Extremo\nratio > 2.0']
fraud_rates = [4.20, 11.04, 28.75]
bar_colors = [GRAY_MUTED, NAVY, MELI_YELLOW]
edge_colors = ['none', 'none', NAVY]

bars = ax2.bar(categories, fraud_rates, color=bar_colors, edgecolor=edge_colors, lw=2, width=0.52)

ax2.text(0, 4.20 + 0.8, "4.2%", ha='center', va='bottom', fontsize=11, fontweight='600', color=GRAY_TEXT)
ax2.text(1, 11.04 + 0.8, "11.0% (2.6x)", ha='center', va='bottom', fontsize=11.5, fontweight='700', color=NAVY)
ax2.text(2, 28.75 + 0.8, "28.8% (Casi 7x)", ha='center', va='bottom', fontsize=12.5, fontweight='800', color=NAVY)

ax2.set_ylim(0, 36)
ax2.set_ylabel("Tasa de Fraude Observada (%)", fontsize=10.5, fontweight='600', color=GRAY_TEXT)
ax2.set_title("Explosión de Fraude al Romperse la Relación", fontsize=11.5, fontweight='700', loc='left', color=NAVY)

ax2.annotate('Multiplica la tasa de fraude x5-x7\n(+132 bps en el modelo)', 
             xy=(2, 29), xytext=(1.05, 33),
             arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.5, connectionstyle="arc3,rad=-0.1"),
             fontsize=9.5, fontweight='700', color=NAVY)

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['bottom'].set_color('#dadce0')
ax2.yaxis.set_visible(False)

plt.tight_layout()
out_path = out_dir / "fig_ratio_d_m.svg"
plt.savefig(out_path, format='svg', bbox_inches='tight', transparent=True)
plt.close()
print(f"Figura guardada exitosamente en: {out_path}")
