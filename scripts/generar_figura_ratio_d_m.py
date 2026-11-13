import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

NAVY = '#2D3277'
MELI_YELLOW = '#FFD100'
GRAY_MUTED = '#bdc1c6'
GRAY_TEXT = '#5f6368'

out_dir = Path(__file__).resolve().parent.parent / "presentacion" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
fig.patch.set_facecolor('none')
ax.set_facecolor('none')

categories = [
    'Normal (d ≈ m)\nratio ≤ 1.0\nN = 133.621',
    'Atípico (d > m)\n1.0 < ratio ≤ 2.0\nN = 15.468',
    'Quiebre Extremo (d >> m)\nratio > 2.0\nN = 546'
]
fraud_rates = [4.20, 11.04, 28.75]
bar_colors = [GRAY_MUTED, NAVY, MELI_YELLOW]
edge_colors = ['none', 'none', NAVY]

bars = ax.bar(categories, fraud_rates, color=bar_colors, edgecolor=edge_colors, lw=2, width=0.48)

# Direct labels with generous vertical spacing
# Bar 0: Normal
ax.text(0, 4.20 + 0.9, "4.2%", ha='center', va='bottom', fontsize=12.5, fontweight='600', color=GRAY_TEXT)
ax.text(0, 4.20 + 3.6, "Línea base", ha='center', va='bottom', fontsize=9.0, color=GRAY_MUTED)

# Bar 1: Atipico
ax.text(1, 11.04 + 0.9, "11.0%", ha='center', va='bottom', fontsize=13, fontweight='700', color=NAVY)
ax.text(1, 11.04 + 3.6, "2.6x prevalencia", ha='center', va='bottom', fontsize=9.5, fontweight='600', color=NAVY)

# Bar 2: Quiebre Extremo (separating 28.8% and the badge clearly)
ax.text(2, 28.75 + 0.9, "28.8%", ha='center', va='bottom', fontsize=15, fontweight='800', color=NAVY)
ax.text(2, 28.75 + 5.6, "Fraude multiplicado x5+ (+132 bps)", ha='center', va='bottom', fontsize=9.5, fontweight='700', color=NAVY,
        bbox=dict(boxstyle='round,pad=0.32', facecolor='#FFF8D6', edgecolor=NAVY, lw=1))

ax.set_ylim(0, 42)
ax.set_ylabel("Tasa de Fraude (%)", fontsize=10.5, fontweight='600', color=GRAY_TEXT)
ax.set_title("Quiebre de Colinealidad d vs m (Spearman 0.735)", fontsize=12.5, fontweight='700', loc='left', color=NAVY, pad=14)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#dadce0')
ax.yaxis.set_visible(False)

plt.tight_layout()
out_path = out_dir / "fig_ratio_d_m.svg"
plt.savefig(out_path, format='svg', bbox_inches='tight', transparent=True)
plt.close()
print(f"Figura corregida con espaciado vertical guardada en: {out_path}")
