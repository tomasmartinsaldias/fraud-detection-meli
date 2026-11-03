"""
Punto de entrada productivo oficial (Contrato B - Detección de Anomalías No Supervisadas).
Delega la ejecución al pipeline productivo en src.models.production.generate_features.
"""

from __future__ import annotations

from src.models.production.generate_features import main

if __name__ == "__main__":
    main()
