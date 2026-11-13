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
beta = 0.40

# Fronteras de decisión
tau_block = tau_min + (tau_max - tau_min) / (1.0 + np.exp((m - M0) / sigma))
tau_chal = beta * tau_block

# 3 Zonas sombreadas
# Zona Verde (0 a tau_chal)
ax.fill_between(m, 0, tau_chal, color='#E6F4EA', alpha=0.9, label='Carril Verde (94.2%): Aprobación directa')
# Zona Amarilla (tau_chal a tau_block)
ax.fill_between(m, tau_chal, tau_block, color='#FEF7E0', alpha=0.95, label='Carril Amarillo (4.8%): Reto 2FA WhatsApp')
# Zona Roja (tau_block a 0.55)
ax.fill_between(m, tau_block, 0.55, color='#FCE8E6', alpha=0.85, label='Carril Rojo (1.0%): Bloqueo duro')

# Líneas de frontera
ax.plot(m, tau_block, color='#d93025', lw=2.2)
ax.plot(m, tau_chal, color='#b06000', lw=2.0, linestyle='--')

# Fórmulas explícitas de cálculo sobre las fronteras
ax.text(196, tau_block[-1] + 0.012, r'$\tau_{\mathrm{block}}(\mathrm{monto}) = \tau_S - \alpha \cdot \mathrm{déficit}$', 
        fontsize=8.0, fontweight='700', color='#d93025', ha='right')
ax.text(196, tau_chal[-1] - 0.018, r'$\tau_{\mathrm{challenge}} = 0.40 \cdot \tau_{\mathrm{block}}$', 
        fontsize=8.0, fontweight='700', color='#b06000', ha='right')

# Rótulos directos dentro de las zonas
ax.text(105, 0.035, 'CARRIL VERDE (94.2%)\nPaso limpio en 200 ms (Fricción cero)', 
        fontsize=8.5, fontweight='700', color='#137333', ha='center', va='center')
ax.text(105, 0.138, 'CARRIL AMARILLO (4.8%)\nReto WhatsApp (~$0.05) | Costo FP: 1.302 × $0.05 = $65', 
        fontsize=8.0, fontweight='700', color='#b06000', ha='center', va='center')
ax.text(105, 0.35, 'CARRIL ROJO (1.0%)\nBloqueo duro patrimonial ($25)', 
        fontsize=8.5, fontweight='700', color='#c5221f', ha='center', va='center')

# Punto emblemático de fraude chico rescatado por 2FA
ax.scatter([15], [0.22], color='#d93025', s=50, zorder=5, edgecolor=NAVY, lw=1.5)
ax.annotate('Fraude chico ($15, p=0.22)\nCurva S lo dejaba pasar; 2FA lo frena', 
            xy=(15, 0.22), xytext=(24, 0.25),
            arrowprops=dict(arrowstyle='->', color=NAVY, lw=1.4),
            fontsize=8.0, fontweight='700', color=NAVY,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFF8D6', edgecolor=NAVY, lw=0.8))

# Rótulos en los ejes
ax.set_xlim(0, 200)
ax.set_ylim(0, 0.52)
ax.set_xlabel('Monto de la Transacción ($)', fontsize=10, fontweight='600', color=NAVY)
ax.set_ylabel('Probabilidad de Fraude (Score p)', fontsize=10, fontweight='600', color=NAVY)
ax.set_title('Espacio 2D de Decisión: Semáforo Tripartito (Curva S + 2FA)', 
             fontsize=11, fontweight='700', loc='left', color=NAVY, pad=12)

# Quitar spines innecesarios
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('#dadce0')
ax.spines['left'].set_color('#dadce0')
ax.grid(color='#e8eaed', linestyle=':', linewidth=0.7)

plt.tight_layout()
out_file = out_dir / "fig4_semaforo_3zonas.svg"
plt.savefig(out_file, format='svg', bbox_inches='tight', transparent=True)
plt.close()
print(f"Gráfico 2D con fórmulas explícitas guardado en: {out_file}")
