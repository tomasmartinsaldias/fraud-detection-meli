import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

NAVY = '#2D3277'
MELI_YELLOW = '#FFD100'
GRAY_MUTED = '#bdc1c6'
GRAY_TEXT = '#5f6368'

out_dir = Path(__file__).resolve().parent.parent / "presentacion" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)

fig = plt.figure(figsize=(11.5, 4.5), dpi=150)
fig.patch.set_facecolor('none')

# -------------------------------------------------------------
# Panel 1: Reloj Polar Circadiano (24 horas UTC)
# -------------------------------------------------------------
ax1 = plt.subplot(1, 2, 1, projection='polar')
ax1.set_facecolor('none')
ax1.set_theta_offset(np.pi / 2)  # 00:00 arriba (las 12 del reloj)
ax1.set_theta_direction(-1)      # Sentido horario como un reloj analógico

horas = np.arange(24)
theta = horas * (2 * np.pi / 24)
tasas = np.array([
    12.4, 15.1, 17.2, 14.8, 11.2, 8.5,
    4.1, 3.8, 3.5, 3.2, 3.6, 4.0,
    4.2, 4.5, 4.3, 4.1, 3.9, 4.2,
    4.4, 4.8, 5.2, 6.1, 8.3, 11.5
])

width = (2 * np.pi / 24) * 0.86
# Horas de la ventana es_madrugada (01:00 a 05:00 UTC) en amarillo Mercado Libre
colors = [MELI_YELLOW if (1 <= h <= 5) else ('#FFE580' if (h == 0 or h == 23) else '#cfd4dc') for h in horas]
edgecolors = [NAVY if h == 2 else 'none' for h in horas]
lws = [2.2 if h == 2 else 0 for h in horas]

bottom_val = 3.5
bars = ax1.bar(theta, tasas, width=width, bottom=bottom_val, color=colors, edgecolor=edgecolors, linewidth=lws)

# Centro del reloj (esfera interior)
ax1.text(0, 0, '24h UTC\nReloj', ha='center', va='center', fontsize=8.5, fontweight='700', color=NAVY)

# Arco exterior delimitando la ventana es_madrugada (01:00 a 05:00 UTC)
theta_arc = np.linspace(0.5 * (2*np.pi/24), 5.5 * (2*np.pi/24), 50)
r_arc = np.full_like(theta_arc, 22.8)
ax1.plot(theta_arc, r_arc, color=NAVY, lw=2.2)
ax1.plot([0.5 * (2*np.pi/24), 0.5 * (2*np.pi/24)], [21.8, 23.8], color=NAVY, lw=1.8)
ax1.plot([5.5 * (2*np.pi/24), 5.5 * (2*np.pi/24)], [21.8, 23.8], color=NAVY, lw=1.8)

# Ticks angulares del reloj (omitimos 03:00 para no chocar con la etiqueta del arco)
tick_hours = [0, 6, 9, 12, 15, 18, 21]
ax1.set_xticks(np.array(tick_hours) * (2 * np.pi / 24))
ax1.set_xticklabels(['00:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00'], 
                    fontsize=8.5, fontweight='600', color='#3c4043')

# Etiqueta de la ventana de madrugada colocada sobre el arco
ax1.text(3.0 * (2*np.pi/24), 25.5, 'Ventana 01:00 a 05:00 UTC', 
         ha='center', va='bottom', fontsize=8.2, fontweight='700', color=NAVY,
         bbox=dict(boxstyle='round,pad=0.25', facecolor='#FFF8D6', edgecolor=NAVY, lw=0.8))

# Anotación destacada del pico de las 02:00 UTC (17.2%)
theta_2 = 2 * (2 * np.pi / 24)
ax1.annotate('Pico 17.2%\n(02:00 UTC)', 
             xy=(theta_2, 17.2 + bottom_val), xytext=(theta_2 - 0.42, 19.5),
             arrowprops=dict(arrowstyle='->', color=NAVY, lw=1.4),
             fontsize=8.0, fontweight='700', color=NAVY)

ax1.set_ylim(0, 28)
ax1.set_rlabel_position(180)  # Labels en las 12:00 (abajo)
ax1.set_yticks([5, 10, 15, 20])
ax1.set_yticklabels(['5%', '10%', '15%', '20%'], fontsize=7.5, color=GRAY_TEXT)
ax1.grid(color='#e8eaed', linestyle='--', linewidth=0.8)
ax1.spines['polar'].set_color('#dadce0')

ax1.set_title('1. Reloj Circadiano de Fraude (24h UTC)', fontsize=11, fontweight='700', color=NAVY, pad=18)

# -------------------------------------------------------------
# Panel 2: Tasa de Fraude y Lift de las 3 Variables Complejas
# -------------------------------------------------------------
ax2 = plt.subplot(1, 2, 2)
ax2.set_facecolor('none')

segments = [
    'Base Diurna\n(Habitual)',
    'prod_b_log_c\n(b > 0.8 & c < 1000)',
    'es_madrugada\n(Pico 02:00 UTC)',
    'score_texto_nb\n(Top 1% Títulos)'
]
fraud_rates = [4.10, 12.82, 17.20, 24.47]
bar_colors = [GRAY_MUTED, NAVY, NAVY, MELI_YELLOW]
edge_colors = ['none', 'none', 'none', NAVY]

bars2 = ax2.bar(segments, fraud_rates, color=bar_colors, edgecolor=edge_colors, lw=2, width=0.52)

# Direct labels con espaciado vertical generoso (sin solapamiento)
ax2.text(0, 4.10 + 0.9, '4.1%', ha='center', va='bottom', fontsize=11, fontweight='600', color=GRAY_TEXT)
ax2.text(0, 4.10 + 4.2, '1.0x (Base)', ha='center', va='bottom', fontsize=8.5, color=GRAY_MUTED)

ax2.text(1, 12.82 + 0.9, '12.8%', ha='center', va='bottom', fontsize=11.5, fontweight='700', color=NAVY)
ax2.text(1, 12.82 + 4.2, '3.1x lift', ha='center', va='bottom', fontsize=9.0, fontweight='600', color=NAVY)

ax2.text(2, 17.20 + 0.9, '17.2%', ha='center', va='bottom', fontsize=11.5, fontweight='700', color=NAVY)
ax2.text(2, 17.20 + 4.2, '3.8x lift', ha='center', va='bottom', fontsize=9.0, fontweight='600', color=NAVY)

# Barra 4: 24.5% y badge 5.4x lift (sin palabra Dirichlet)
ax2.text(3, 24.47 + 0.9, '24.5%', ha='center', va='bottom', fontsize=13, fontweight='800', color=NAVY)
ax2.text(3, 24.47 + 5.2, '5.4x lift', ha='center', va='bottom', fontsize=9.5, fontweight='700', color=NAVY,
         bbox=dict(boxstyle='round,pad=0.25', facecolor='#FFF8D6', edgecolor=NAVY, lw=1))

ax2.set_ylim(0, 36)
ax2.set_ylabel('Tasa de Fraude Observada (%)', fontsize=10.5, color=NAVY, fontweight='600')
ax2.set_title('2. Capacidad de Aislamiento de Riesgo (Lift)', fontsize=11, fontweight='700', loc='left', color=NAVY, pad=18)
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
