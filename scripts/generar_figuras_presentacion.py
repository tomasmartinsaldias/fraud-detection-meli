import matplotlib.pyplot as plt
from pathlib import Path

# Configuración Knaflic pura y paleta Mercado Libre
NAVY = '#2D3277'
MELI_YELLOW = '#FFD100'
GRAY_MUTED = '#bdc1c6'
GRAY_TEXT = '#5f6368'

out_dir = Path(__file__).resolve().parent.parent / "presentacion" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)

# =============================================================
# FIGURA 1: Embudo de Selección (51 a 11 variables)
# =============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0), dpi=150)
fig.patch.set_facecolor('none')
for a in (ax1, ax2): a.set_facecolor('none')

# Panel 1: De 51 a 11
bars1 = ax1.bar(['Pool Candidato\n(51 vars)', 'Núcleo Seleccionado\n(11 vars)'], 
                [51, 11], 
                color=[GRAY_MUTED, MELI_YELLOW], 
                edgecolor=['none', NAVY], lw=2, width=0.45)

ax1.text(0, 51 + 1.2, "51", ha='center', va='bottom', fontsize=12, fontweight='600', color=GRAY_TEXT)
ax1.text(1, 11 + 1.2, "11", ha='center', va='bottom', fontsize=13, fontweight='700', color=NAVY)

ax1.annotate('−78% de ruido podado\npor Forward Selection', 
             xy=(1.1, 15), xytext=(1.25, 32),
             arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.5, connectionstyle="arc3,rad=-0.15"),
             fontsize=9.5, fontweight='700', color=NAVY)

ax1.set_ylim(0, 62)
ax1.set_title("Poda de Dimensionalidad", fontsize=11.5, fontweight='700', loc='left', color=NAVY)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_color('#dadce0')
ax1.spines['left'].set_visible(False)
ax1.yaxis.set_visible(False)

# Panel 2: Salto PR-AUC
bars2 = ax2.bar(['Árbol Crudo', 'LightGBM\n11 vars'], 
                [0.43135, 0.44435], 
                color=[GRAY_MUTED, NAVY], width=0.45)

ax2.text(0, 0.43135 + 0.0006, "0.4313", ha='center', va='bottom', fontsize=11, fontweight='600', color=GRAY_TEXT)
ax2.text(1, 0.44435 + 0.0006, "0.4443", ha='center', va='bottom', fontsize=12, fontweight='700', color=NAVY)
ax2.text(1, 0.4485, "+130.0 bps en Test", ha='center', va='bottom', fontsize=11, fontweight='700', color=NAVY)

ax2.set_ylim(0.426, 0.453)
ax2.set_title("Ganancia de Generalización (OOT)", fontsize=11.5, fontweight='700', loc='left', color=NAVY)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_color('#dadce0')
ax2.spines['left'].set_visible(False)
ax2.yaxis.set_visible(False)

plt.tight_layout()
plt.savefig(out_dir / "fig1_embudo_seleccion.svg", format='svg', bbox_inches='tight', transparent=True)
plt.close()

# =============================================================
# FIGURA 2: Detección de Anomalías (Univariado Isolation Forest)
# =============================================================
fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=150)
fig.patch.set_facecolor('none')
ax.set_facecolor('none')

models = [
    'Azar / Prevalencia Base (4.5%)',
    'IF Ortogonal Numérico',
    'IF Ortogonal Frecuencia',
    'IF Diagonal + Preprocesamiento Híbrido'
]
prauc = [0.0453, 0.0871, 0.1016, 0.1557]
colors_if = [GRAY_MUTED, GRAY_MUTED, '#7277b5', MELI_YELLOW]
edges_if = ['none', 'none', 'none', NAVY]

bars = ax.barh(models, prauc, color=colors_if, edgecolor=edges_if, lw=2, height=0.48)

for b, val in zip(bars, prauc):
    c = NAVY if val == max(prauc) else GRAY_TEXT
    w = '700' if val == max(prauc) else '600'
    ax.text(val + 0.0035, b.get_y() + b.get_height()/2, f'{val:.4f}', 
            va='center', fontsize=10.5, fontweight=w, color=c)

ax.annotate('Triplica el azar\n(Señal no supervisada pura)', 
            xy=(0.1557, 3), xytext=(0.11, 2.1),
            arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.4),
            fontsize=9.5, fontweight='700', color=NAVY)

ax.set_xlim(0, 0.19)
ax.set_xlabel('LightGBM PR-AUC Univariado (Val OOT)', fontsize=10, color=NAVY)
ax.set_title('Capacidad Aislada de Detección de Anomalías (One-Class)', 
             fontsize=11.5, fontweight='700', loc='left', color=NAVY, pad=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('#dadce0')
ax.spines['left'].set_color('#dadce0')

plt.tight_layout()
plt.savefig(out_dir / 'fig_anomalias_comparativa.svg', format='svg', bbox_inches='tight', transparent=True)
plt.close()

# =============================================================
# FIGURA 3: Residuos Fraude Hormiga (Hero Chart)
# =============================================================
fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=150)
fig.patch.set_facecolor('none')
ax.set_facecolor('none')

cats = [
    'Fraudes Detectados\n(Verdaderos Positivos)',
    'Fraudes que Evadían el Modelo\n(Punto Ciego: Falsos Negativos)'
]
montos = [74.94, 22.00]
colors = [NAVY, MELI_YELLOW]
edges = ['none', NAVY]

bars = ax.bar(cats, montos, color=colors, edgecolor=edges, lw=2.2, width=0.42)

ax.text(0, 74.94 + 1.8, "$74.94", ha='center', va='bottom', fontsize=13, fontweight='700', color=NAVY)
ax.text(1, 22.00 + 1.8, "$22.00", ha='center', va='bottom', fontsize=13, fontweight='700', color=NAVY)

ax.annotate('Mediana 3.4x menor:\nlos fraudes chicos se mimetizaban\nen el tráfico diurno', 
            xy=(1, 25), xytext=(0.55, 52),
            arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.5),
            fontsize=10, fontweight='700', color=NAVY)

ax.set_ylim(0, 95)
ax.set_ylabel("Ticket Típico de la Compra ($)", fontsize=10.5, color=NAVY)
ax.set_title("La Asimetría de Montos en los Residuos del Modelo Base", 
             fontsize=12, fontweight='700', loc='left', pad=12, color=NAVY)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('#dadce0')
ax.spines['left'].set_color('#dadce0')

plt.tight_layout()
plt.savefig(out_dir / "fig2_residuos_fraude_hormiga.svg", format='svg', bbox_inches='tight', transparent=True)
plt.close()

# =============================================================
# FIGURA 4: Complejidad Técnica del Feature Engineering
# =============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0), dpi=150)
fig.patch.set_facecolor('none')
for a in (ax1, ax2): a.set_facecolor('none')

# Panel 1: Patrón Circadiano es_madrugada
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

bars = ax1.bar(horas, tasas, color=colors_hora, edgecolor=edges_hora, linewidth=lws_hora, width=0.75)
ax1.axvspan(-0.5, 5.5, color='#2D3277', alpha=0.06)
ax1.axvspan(22.5, 23.5, color='#2D3277', alpha=0.06)

ax1.annotate('Pico 17.2% (02:00 UTC)\nVentana es_madrugada\n(3.8x sobre el día)', 
             xy=(2, 17.2), xytext=(4.5, 14.2),
             arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.5),
             fontsize=9.5, fontweight='700', color=NAVY)

ax1.set_xlim(-0.8, 23.8)
ax1.set_ylim(0, 20)
ax1.set_xlabel('Hora del Día (UTC)', fontsize=10, color=NAVY)
ax1.set_ylabel('Incidencia de Fraude (%)', fontsize=10, color=NAVY)
ax1.set_title('1. Modelado Circadiano (es_madrugada)', fontsize=11.5, fontweight='700', loc='left', color=NAVY)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_color('#dadce0')
ax1.spines['left'].set_color('#dadce0')

# Panel 2: Ganancia de Formulaciones No Lineales y NLP
features = [
    'prod_b_log_c\n(Perfil x Magnitud)',
    'ratio_d_m\n(Tensión Histórica)',
    'ratio_monto_l\n(Ticket s/ Antigüedad)',
    'score_texto_nb\n(Naive Bayes Títulos)'
]
gain_relative = [1880, 2150, 2640, 3120]
colors_feat = [GRAY_MUTED, GRAY_MUTED, '#7277b5', NAVY]

bars2 = ax2.barh(features, gain_relative, color=colors_feat, height=0.52)
for b, g in zip(bars2, gain_relative):
    c = NAVY if g == max(gain_relative) else GRAY_TEXT
    w = '700' if g == max(gain_relative) else '600'
    ax2.text(g + 50, b.get_y() + b.get_height()/2, f'Gain: {g:,}', va='center', fontsize=9.5, fontweight=w, color=c)

ax2.annotate('Priors Dirichlet en Train\n(Cero Data Leakage)', 
             xy=(3120, 3), xytext=(1700, 2.3),
             arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.4),
             fontsize=9.5, fontweight='700', color=NAVY)

ax2.set_xlim(0, 4200)
ax2.set_xlabel('Importancia Relativa en LightGBM (Gain)', fontsize=10, color=NAVY)
ax2.set_title('2. Formulaciones No Lineales y NLP', fontsize=11.5, fontweight='700', loc='left', color=NAVY)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_color('#dadce0')
ax2.spines['left'].set_color('#dadce0')

plt.tight_layout()
plt.savefig(out_dir / "fig_fe_complejidad.svg", format='svg', bbox_inches='tight', transparent=True)
plt.close()

print("Figuras de presentación generadas exitosamente en presentacion/figures/.")
