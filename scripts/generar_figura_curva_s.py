import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

NAVY = '#2D3277'
MELI_YELLOW = '#FFD100'
GRAY_MUTED = '#bdc1c6'
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

# 2 Zonas sombreadas con verde suave y rojo suave (continuidad con Semáforo 2FA)
# Zona Verde (Aprobación: 0 a tau_s)
ax.fill_between(m, 0, tau_s, color='#E6F4EA', alpha=0.9, label='Carril Verde: Aprobación directa')
# Zona Roja (Bloqueo: tau_s a 0.52)
ax.fill_between(m, tau_s, 0.52, color='#FCE8E6', alpha=0.85, label='Carril Rojo: Bloqueo duro')

# Línea de la Curva S
ax.plot(m, tau_s, color=NAVY, lw=2.5, label=r'Curva S Elástica $\tau_S(\mathrm{monto})$')

# Línea del Umbral Estático Plano
ax.axhline(tau_static, color=GRAY_TEXT, linestyle='--', lw=1.6, alpha=0.85, label=r'Umbral Estático Plano $\tau = 0.2550$')

# Rótulo del umbral estático
ax.text(196, tau_static + 0.012, r'Umbral Plano Estático $\tau = 0.2550$', 
        fontsize=8.5, fontweight='600', color=GRAY_TEXT, ha='right')

# Rótulo de la función Curva S
ax.text(196, tau_s[-1] - 0.022, r'$\tau_S(\mathrm{monto}) = 0.18 + \frac{0.22}{1 + e^{(m - 23.02)/35}}$', 
        fontsize=8.5, fontweight='700', color=NAVY, ha='right')

# Rótulos directos dentro de las zonas
ax.text(105, 0.065, 'ZONA DE APROBACIÓN (Fricción cero)\nCompras con score inferior a la curva pasan directo', 
        fontsize=8.5, fontweight='700', color='#137333', ha='center', va='center')
ax.text(105, 0.38, 'ZONA DE BLOQUEO DURO (Cerrojo patrimonial)\nCompras con score superior a la curva se rechazan', 
        fontsize=8.5, fontweight='700', color='#c5221f', ha='center', va='center')

# Punto de inflexión M0
tau_inflexion = (tau_max + tau_min) / 2.0
ax.scatter([M0], [tau_inflexion], color=MELI_YELLOW, s=60, zorder=5, edgecolor=NAVY, lw=1.8)
ax.annotate(r'Inflexión $M_0 = \$23.02$' + '\n' + r'($\tau = 0.29$ | Mediana de fraude)', 
            xy=(M0, tau_inflexion), xytext=(M0 + 12, tau_inflexion + 0.05),
            arrowprops=dict(arrowstyle='->', color=NAVY, lw=1.4),
            fontsize=8.2, fontweight='700', color=NAVY,
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#FFF8D6', edgecolor=NAVY, lw=0.8))

# Anotación en tickets altos
ax.annotate('Tickets Altos (> $100):\n' + r'$\tau \to 0.18$ (Cerrojo estricto)', 
            xy=(150, tau_s[225]), xytext=(120, 0.23),
            arrowprops=dict(arrowstyle='->', color=NAVY, lw=1.3),
            fontsize=8.0, fontweight='700', color=NAVY,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#ffffff', edgecolor='#dadce0', lw=0.8))

# Anotación en tickets bajos
ax.annotate('Tickets Bajos (< $15):\n' + r'$\tau \to 0.40$ (No quemar FPs)', 
            xy=(10, tau_s[15]), xytext=(22, 0.44),
            arrowprops=dict(arrowstyle='->', color=NAVY, lw=1.3),
            fontsize=8.0, fontweight='700', color=NAVY,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#ffffff', edgecolor='#dadce0', lw=0.8))

# Rótulos en los ejes
ax.set_xlim(0, 200)
ax.set_ylim(0, 0.52)
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
print(f"Gráfico de Curva S con zonas verde y roja guardado en: {out_file}")
