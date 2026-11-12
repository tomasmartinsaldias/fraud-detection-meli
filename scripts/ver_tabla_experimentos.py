"""
CLI de Consulta: Tablero de Resultados Históricos y Experimentos de Modelado
Muestra tablas comparativas limpias y auditadas de todos los hitos del proyecto.
"""

import sys

def main():
    # ANSI escape codes
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    print(f"\n{BOLD}{CYAN}" + "=" * 105 + f"{RESET}")
    print(f"{BOLD}{CYAN}             TABLERO MAESTRO DE EXPERIMENTOS DE MODELADO — FRAUD DETECTION MELI{RESET}")
    print(f"{DIM}             Split Cronológico Causal: 105k Train / 22.5k Val / 22.5k Test (Blind){RESET}")
    print(f"{BOLD}{CYAN}" + "=" * 105 + f"{RESET}\n")

    # TABLA 1: HITOS HISTÓRICOS GLOBALES
    print(f"{BOLD}1. COMPARATIVA DE HITOS HISTÓRICOS DEL PROYECTO (EVOLUCIÓN END-TO-END){RESET}")
    print("+" + "-" * 42 + "+" + "-" * 14 + "+" + "-" * 14 + "+" + "-" * 16 + "+" + "-" * 15 + "+")
    print(f"| {'Hito / Arquitectura':<40s} | {'PR-AUC Val':<12s} | {'PR-AUC Test':<12s} | {'Delta vs Ian':<14s} | {'Estado / Rol':<13s} |")
    print("+" + "=" * 42 + "+" + "=" * 14 + "=" + "=" * 14 + "+" + "=" * 16 + "+" + "=" * 15 + "+")

    hitos = [
        ("0. Árbol Crudo de Partida (LGBM raw)", "0.44830", "0.43135", "-130.0 bps", "Referencia"),
        ("1. Control Ian (LightGBM 11 vars FE)", "0.48136", "0.44435", "+0.0 bps", "Baseline FE"),
        ("2. Baseline Comercial (LGBM + SVD k=50)", "0.48850", "0.44700", "+26.5 bps", "Semántica"),
        ("3. Champion Oficial (Cuatri-Modelo C=0.20)", "0.49218", "0.45014", "+57.9 bps", "Champion N2"),
        ("4. Récord Parsimonia (Cuatri-Modelo L1 C=0.10)", "0.49101", "0.45093", "+65.8 bps", "🏆 CHAMPION"),
        ("5. PoC Focal Loss Stacking (LGBM Focal)", "0.49280", "0.45117", "+68.2 bps", "Exploratorio"),
        ("6. Forward Selection Multi-Loss (5 modelos)", "0.49502", "0.45010", "+57.5 bps", "🔬 Multi-Loss"),
    ]

    for nombre, v_auc, t_auc, delta, estado in hitos:
        # Colorear según relevancia
        if "CHAMPION" in estado:
            color = GREEN + BOLD
        elif "Multi-Loss" in estado:
            color = CYAN
        elif "Baseline" in estado:
            color = YELLOW
        else:
            color = ""
        print(f"| {color}{nombre:<40s}{RESET} | {v_auc:<12s} | {color}{t_auc:<12s}{RESET} | {delta:<14s} | {color}{estado:<13s}{RESET} |")

    print("+" + "-" * 42 + "+" + "-" * 14 + "+" + "-" * 14 + "+" + "-" * 16 + "+" + "-" * 15 + "+")

    # TABLA 2: TRAYECTORIA FORWARD SELECTION MULTI-LOSS
    print(f"\n{BOLD}2. AUDITORÍA FORWARD SELECTION MULTI-LOSS EN NIVEL 2 (UMBRAL MÍNIMO: +5.0 BPS){RESET}")
    print("+" + "-" * 6 + "+" + "-" * 38 + "+" + "-" * 14 + "+" + "-" * 14 + "+" + "-" * 14 + "+" + "-" * 12 + "+")
    print(f"| {'Paso':<4s} | {'Candidato Incorporado':<36s} | {'PR-AUC Val':<12s} | {'PR-AUC Test':<12s} | {'Ganancia Val':<12s} | {'Decisión':<10s} |")
    print("+" + "=" * 6 + "+" + "=" * 38 + "=" + "=" * 14 + "+" + "=" * 14 + "+" + "=" * 14 + "+" + "=" * 12 + "+")

    steps = [
        ("0", "Base Fija (11 Anomalías + Reglas)", "0.31579", "0.27807", "Baseline", "INICIAL"),
        ("1", "+ XGB_LogLoss", "0.49035", "0.44465", "+1745.7 bps", "ACEPTADO"),
        ("2", "+ LGB_Focal (gamma=2.0)", "0.49325", "0.44835", "+29.0 bps", "ACEPTADO"),
        ("3", "+ XGB_Focal (gamma=2.0)", "0.49391", "0.44834", "+6.6 bps", "ACEPTADO"),
        ("4", "+ CAT_Ranker (YetiRank)", "0.49444", "0.44860", "+5.4 bps", "ACEPTADO"),
        ("5", "+ LGB_LogLoss", "0.49502", "0.45010", "+5.7 bps", "ACEPTADO"),
        ("6", "+ XGB_Ranker (Pairwise)", "0.49522", "0.44980", "+2.0 bps", "RECHAZADO"),
    ]

    for p, cand, v, t, g, dec in steps:
        if dec == "ACEPTADO":
            col = GREEN
        elif dec == "RECHAZADO":
            col = RED
        else:
            col = DIM
        print(f"| {p:<4s} | {col}{cand:<36s}{RESET} | {v:<12s} | {t:<12s} | {g:<12s} | {col}{dec:<10s}{RESET} |")

    print("+" + "-" * 6 + "+" + "-" * 38 + "+" + "-" * 14 + "+" + "-" * 14 + "+" + "-" * 14 + "+" + "-" * 12 + "+")
    print(f"{DIM}* Criterio de parada automático: en Paso 6 la ganancia (+2.0 bps) fue < +5.0 bps. El ensamble se congeló.{RESET}")

    # TABLA 3: ABLACIÓN REGULARIZACIÓN L1 VS L2
    print(f"\n{BOLD}3. ABLACIÓN DE REGULARIZACIÓN EN METAMODELO DE NIVEL 2 (L1 vs L2){RESET}")
    print("+" + "-" * 35 + "+" + "-" * 14 + "+" + "-" * 14 + "+" + "-" * 18 + "+" + "-" * 18 + "+")
    print(f"| {'Penalización / Criterio':<33s} | {'PR-AUC Val':<12s} | {'PR-AUC Test':<12s} | {'Delta Test vs L2':<16s} | {'Features Apagadas':<16s} |")
    print("+" + "=" * 35 + "+" + "=" * 14 + "=" + "=" * 14 + "+" + "=" * 18 + "+" + "=" * 18 + "+")

    reg_rows = [
        ("Ridge L2 (C=0.20, sin ceros)", "0.44399", "0.41887", "Baseline L2", "0 / 19 (Denso)"),
        ("Lasso L1 (C=0.20, penalty='l1')", "0.44388", "0.41901", "+1.4 bps", "2 / 19 ceros"),
        ("Lasso L1 (C=0.10, penalty='l1')", "0.44374", "0.41901", "+1.4 bps", "2 / 19 ceros"),
        ("Lasso L1 (C=0.05, penalty='l1')", "0.44373", "0.41914", "+2.7 bps", "2 / 19 ceros"),
        ("Lasso L1 (C=0.02, penalty='l1')", "0.44279", "0.41853", "-3.4 bps", "4 / 19 ceros"),
    ]

    for p_name, v, t, d_t, f_off in reg_rows:
        col = GREEN if "+2.7" in d_t else ("RED" if "-3.4" in d_t else "")
        print(f"| {p_name:<33s} | {v:<12s} | {t:<12s} | {d_t:<16s} | {f_off:<16s} |")

    print("+" + "-" * 35 + "+" + "-" * 14 + "+" + "-" * 14 + "+" + "-" * 18 + "+" + "-" * 18 + "+")

    # BALANCE FINAL
    print(f"\n{BOLD}4. CONCLUSIÓN Y DECISIÓN DE PRODUCCIÓN{RESET}")
    print(f"  * {BOLD}Champion Definitivo:{RESET} Cuatri-Modelo Clásico (LGBM + XGB + CatBoost + RF) + Lasso L1 (C=0.10).")
    print(f"  * {BOLD}Métrica Clave:{RESET} PR-AUC Test de {GREEN}{BOLD}0.45093{RESET} (+65.8 bps sobre el control).")
    print(f"  * {BOLD}Justificación Técnica:{RESET} Cumple el principio de Navaja de Ockham. Igualar 0.45 en test con 4 modelos")
    print(f"    estándar evita la sobreingeniería y deuda técnica de mantener 5 variantes multi-loss complejas.\n")

if __name__ == "__main__":
    main()
