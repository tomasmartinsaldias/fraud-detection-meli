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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=150)
fig.patch.set_facecolor('none')
ax1.set_facecolor('none')
ax2.set_facecolor('none')

# -------------------------------------------------------------
# Panel 1: Patrón Circadiano es_madrugada (24 horas UTC)
# -------------------------------------------------------------
horas = list(range(24))
tasas = [
    12.4, 15.1, 17.2, 14.8, 11.2, 8.5,
    4.1, 3.8, 3.5, 3.2, 3.6, 4.0,
    4.2, 4.5, 4.3, 4.1, 3.9, 4.2,
    4.4, 4.8, 5.2, 6.1, 8.3, 11.5
]
colors_hora = [MELI_YELLOW if (h <= 5 or h >= 23) else GRAY_MUTED for h in horas]
edges_hora = [NAVY if h == 2 else 'none' for h in horas]
lws_hora = [2.2 if h == 2 else 0 for h in horas]

bars1 = ax1.bar(horas, tasas, color=colors_hora, edgecolor=edges_hora, linewidth=lws_hora, width=0.75)
ax1.axvspan(-0.5, 5.5, color='#2D3277', alpha=0.06)
ax1.axvspan(22.5, 23.5, color='#2D3277', alpha=0.06)

ax1.annotate('Pico 17.2% (02:00 UTC)\nVentana es_madrugada\n(3.8x sobre el día)', 
             xy=(2, 17.2), xytext=(4.5, 14.2),
             arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.5),
             fontsize=9.5, fontweight='700', color=NAVY)

ax1.set_xlim(-0.8, 23.8)
ax1.set_ylim(0, 20)
ax1.set_xlabel('Hora del Día (UTC)', fontsize=10.5, color=NAVY, fontweight='600')
ax1.set_ylabel('Incidencia de Fraude (%)', fontsize=10.5, color=NAVY, fontweight='600')
ax1.set_title('1. Ventana Circadiana (es_madrugada)', fontsize=11.5, fontweight='700', loc='left', color=NAVY)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_color('#dadce0')
ax1.spines['left'].set_color('#dadce0')

# -------------------------------------------------------------
# Panel 2: Tasa de Fraude y Lift de las 3 Variables Complejas
# -------------------------------------------------------------
segments = [
    'Base Plataforma\n(Día habitual)',
    'prod_b_log_c\n(b > 0.8 & c < 1000)',
    'es_madrugada\n(Pico 02:00 UTC)',
    'score_texto_nb\n(Top 1% Títulos)'
]
fraud_rates = [4.10, 12.82, 17.20, 24.47]
bar_colors = [GRAY_MUTED, NAVY, NAVY, MELI_YELLOW]
edge_colors = ['none', 'none', 'none', NAVY]

bars2 = ax2.bar(segments, fraud_rates, color=bar_colors, edgecolor=edge_colors, lw=2, width=0.52)

# Direct labels
ax2.text(0, 4.10 + 0.7, "4.1%", ha='center', va='bottom', fontsize=11, fontweight='600', color=GRAY_TEXT)
ax2.text(0, 4.10 + 2.5, "1.0x (Base)", ha='center', va='bottom', fontsize=8.5, color=GRAY_MUTED)

ax2.text(1, 12.82 + 0.7, "12.8%", ha='center', va='bottom', fontsize=11.5, fontweight='700', color=NAVY)
ax2.text(1, 12.82 + 2.5, "3.1x lift", ha='center', va='bottom', fontsize=9.0, fontweight='600', color=NAVY)

ax2.text(2, 17.20 + 0.7, "17.2%", ha='center', va='bottom', fontsize=11.5, fontweight='700', color=NAVY)
ax2.text(2, 17.20 + 2.5, "3.8x lift", ha='center', va='bottom', fontsize=9.0, fontweight='600', color=NAVY)

ax2.text(3, 24.47 + 0.7, "24.5%", ha='center', va='bottom', fontsize=13, fontweight='800', color=NAVY)
ax2.text(3, 24.47 + 2.5, "5.4x lift (Dirichlet)", ha='center', va='bottom', fontsize=9.5, fontweight='700', color=NAVY,
         bbox=dict(boxstyle='round,pad=0.25', facecolor='#FFF8D6', edgecolor=NAVY, lw=1))

ax2.set_ylim(0, 31)
ax2.set_ylabel('Tasa de Fraude Observada (%)', fontsize=10.5, color=NAVY, fontweight='600')
ax2.set_title('2. Capacidad de Aislamiento de Riesgo (Lift)', fontsize=11.5, fontweight='700', loc='left', color=NAVY)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['bottom'].set_color('#dadce0')
ax2.yaxis.set_visible(False)

plt.tight_layout()
out_path = out_dir / "fig_fe_complejidad.svg"
plt.savefig(out_path, format='svg', bbox_inches='tight', transparent=True)
plt.close()
print(f"Figura actualizada guardada en: {out_path}")
