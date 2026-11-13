import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

NAVY = '#2D3277'
MELI_YELLOW = '#FFD100'
GRAY_TEXT = '#5f6368'

out_dir = Path(__file__).resolve().parent.parent / "presentacion" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=150)
fig.patch.set_facecolor('none')
ax.set_facecolor('none')

m = np.linspace(0, 200, 300)
M0 = 23.02
sigma = 35.0
tau_max = 0.40
tau_min = 0.18
tau_static = 0.2550

# Función Curva S
tau_s = tau_min + (tau_max - tau_min) / (1.0 + np.exp((m - M0) / sigma))

# Zonas sombreadas limpias
# Zona Verde (Aprobación directa)
ax.fill_between(m, 0, tau_s, color='#E6F4EA', alpha=0.9)
# Zona Roja (Bloqueo duro)
ax.fill_between(m, tau_s, 0.50, color='#FCE8E6', alpha=0.85)

# Líneas principales
ax.plot(m, tau_s, color=NAVY, lw=2.5)
ax.axhline(tau_static, color=GRAY_TEXT, linestyle='--', lw=1.6, alpha=0.85)

# Rótulos limpios y directos (sin cajas flotantes ni párrafos)
ax.text(100, 0.08, 'APROBACIÓN DIRECTA', fontsize=11, fontweight='700', color='#137333', ha='center', va='center')
ax.text(100, 0.38, 'BLOQUEO DURO', fontsize=11, fontweight='700', color='#c5221f', ha='center', va='center')

# Rótulos de las líneas
ax.text(195, tau_static + 0.012, 'Umbral plano estático (0.2550)', 
        fontsize=8.5, fontweight='600', color=GRAY_TEXT, ha='right')
ax.text(195, tau_s[-1] - 0.022, r'Curva S $\tau_S(\mathrm{monto})$', 
        fontsize=9.0, fontweight='700', color=NAVY, ha='right')

# Punto de inflexión sobrio (sin caja amarilla grande)
tau_inflexion = (tau_max + tau_min) / 2.0
ax.scatter([M0], [tau_inflexion], color=MELI_YELLOW, s=50, zorder=5, edgecolor=NAVY, lw=1.5)
ax.annotate(r'$M_0 = \$23$', 
            xy=(M0, tau_inflexion), xytext=(M0 + 10, tau_inflexion + 0.035),
            arrowprops=dict(arrowstyle='->', color=NAVY, lw=1.1),
            fontsize=8.5, fontweight='700', color=NAVY)

# Rótulos en los ejes
ax.set_xlim(0, 200)
ax.set_ylim(0, 0.50)
ax.set_xlabel('Monto de la Transacción ($)', fontsize=10, fontweight='600', color=NAVY)
ax.set_ylabel('Probabilidad de Fraude (Score p)', fontsize=10, fontweight='600', color=NAVY)
ax.set_title('Curva S Sigmoidea: Umbral Elástico Adaptado al Ticket', 
             fontsize=11.5, fontweight='700', loc='left', color=NAVY, pad=12)

# Quitar spines innecesarios
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('#dadce0')
ax.spines['left'].set_color('#dadce0')
ax.grid(color='#e8eaed', linestyle=':', linewidth=0.7)

plt.tight_layout()
out_file = out_dir / "fig3_curva_s_umbral.svg"
plt.savefig(out_file, format='svg', bbox_inches='tight', transparent=True)
plt.close()
print(f"Gráfico de Curva S simplificado guardado en: {out_file}")
