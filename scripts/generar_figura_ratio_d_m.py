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

fig, ax = plt.subplots(figsize=(6.8, 3.8), dpi=150)
fig.patch.set_facecolor('none')
ax.set_facecolor('none')

categories = ['Normal (d ≈ m)\nratio ≤ 1.0', 'Atípico (d > m)\n1.0 < ratio ≤ 2.0', 'Quiebre Extremo (d >> m)\nratio > 2.0']
fraud_rates = [4.20, 11.04, 28.75]
bar_colors = [GRAY_MUTED, NAVY, MELI_YELLOW]
edge_colors = ['none', 'none', NAVY]

bars = ax.bar(categories, fraud_rates, color=bar_colors, edgecolor=edge_colors, lw=2, width=0.48)

# Direct labels without arrows crossing text
# Bar 0: Normal
ax.text(0, 4.20 + 1.0, "4.2%", ha='center', va='bottom', fontsize=12, fontweight='600', color=GRAY_TEXT)
ax.text(0, 4.20 + 3.4, "Línea base", ha='center', va='bottom', fontsize=8.5, color=GRAY_MUTED)

# Bar 1: Atipico
ax.text(1, 11.04 + 1.0, "11.0%", ha='center', va='bottom', fontsize=12.5, fontweight='700', color=NAVY)
ax.text(1, 11.04 + 3.4, "2.6x lift", ha='center', va='bottom', fontsize=9.0, fontweight='600', color=NAVY)

# Bar 2: Quiebre Extremo (Hero)
ax.text(2, 28.75 + 1.0, "28.8%", ha='center', va='bottom', fontsize=14, fontweight='800', color=NAVY)
ax.text(2, 28.75 + 3.7, "¡Casi 7x lift! (+132 bps)", ha='center', va='bottom', fontsize=9.5, fontweight='700', color=NAVY,
        bbox=dict(boxstyle='round,pad=0.25', facecolor='#FFF8D6', edgecolor=NAVY, lw=1))

ax.set_ylim(0, 38)
ax.set_ylabel("Tasa de Fraude (%)", fontsize=10.5, fontweight='600', color=GRAY_TEXT)
ax.set_title("Quiebre de Colinealidad d vs m: Salto de Fraude x5 a x7", fontsize=12, fontweight='700', loc='left', color=NAVY, pad=12)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#dadce0')
ax.yaxis.set_visible(False)

plt.tight_layout()
out_path = out_dir / "fig_ratio_d_m.svg"
plt.savefig(out_path, format='svg', bbox_inches='tight', transparent=True)
plt.close()
print(f"Figura corregida sin flecha guardada en: {out_path}")
