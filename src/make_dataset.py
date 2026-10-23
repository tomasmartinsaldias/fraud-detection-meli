"""
Pipeline modular de ingesta, trazabilidad y partición temporal (OOT)
para dataset de detección de fraude transaccional.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Configuración de logging para observabilidad en consola
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("make_dataset")


def load_raw_data(file_path: Path) -> pd.DataFrame:
    """
    Carga el dataset transaccional crudo preservando la columna de índice original.

    Args:
        file_path: Ruta al archivo CSV original.

    Returns:
        pd.DataFrame: DataFrame con los datos crudos y columna '...1' normalizada.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de entrada: {file_path.resolve()}")

    logger.info(f"Cargando dataset crudo desde: {file_path}")
    df = pd.read_csv(file_path)

    # El índice de origen sin encabezado en CSV puede ser leído como 'Unnamed: 0' por pandas
    # o como '...1' por herramientas como readr en R. Aseguramos la denominación '...1'.
    if "Unnamed: 0" in df.columns and "...1" not in df.columns:
        df = df.rename(columns={"Unnamed: 0": "...1"})
    elif "...1" not in df.columns:
        logger.warning("No se encontró la columna '...1' ni 'Unnamed: 0' en el dataset original.")

    logger.info(f"Dataset cargado exitosamente. Dimensiones: {df.shape[0]:,} filas x {df.shape[1]} columnas.")
    return df


def add_traceability_features(
    df: pd.DataFrame,
    block_1_size: int = 150_000,
) -> pd.DataFrame:
    """
    Genera llaves de trazabilidad globales e identifica el bloque de origen de cada registro.
    
    Debe aplicarse antes de cualquier ordenamiento o filtrado para preservar el orden
    físico original.

    Args:
        df: DataFrame crudo original.
        block_1_size: Cantidad de filas que corresponden al Bloque 1 (default: 150.000).

    Returns:
        pd.DataFrame: DataFrame con columnas 'transaction_id', 'origen_bloque' y 'es_bloque_2'.
    """
    logger.info("Generando identificadores de trazabilidad e identificación de bloques...")
    df = df.copy()

    total_rows = len(df)
    # Formateo vectorizado de transaction_id único e inmutable (ej: tx_000000 a tx_249999)
    # Usamos zfill a 6 posiciones para garantizar orden léxico y consistencia
    digits = max(6, len(str(total_rows - 1)))
    indices = np.arange(total_rows)
    df["transaction_id"] = [f"tx_{i:0{digits}d}" for i in indices]

    # Identificación estricta del bloque basada en la posición física inicial
    is_block_2 = indices >= block_1_size
    df["es_bloque_2"] = is_block_2
    df["origen_bloque"] = np.where(is_block_2, "bloque_2", "bloque_1")

    logger.info(
        f"Trazabilidad lista: {np.sum(~is_block_2):,} en Bloque 1 | {np.sum(is_block_2):,} en Bloque 2."
    )
    return df


def filter_population(
    df: pd.DataFrame,
    drop_second_block: bool = False,
) -> Tuple[pd.DataFrame, int]:
    """
    Aplica la capa de exclusión parametrizada sobre el segundo bloque.

    Args:
        df: DataFrame con trazabilidad asignada.
        drop_second_block: Si es True, descarta las filas del segundo bloque.

    Returns:
        Tuple[pd.DataFrame, int]: DataFrame filtrado y conteo de filas descartadas.
    """
    initial_rows = len(df)
    if drop_second_block:
        logger.info(
            "Aplicando filtro de exclusión del Bloque 2 (filas 150.001 a 250.000). "
            "Motivo: Índices reiniciados (0 a 99.999) con fraude=0 artificial detectados en EDA."
        )
        df_filtered = df[~df["es_bloque_2"]].copy()
        discarded_count = initial_rows - len(df_filtered)
        logger.info(f"Filas descartadas: {discarded_count:,}. Filas limpias activas: {len(df_filtered):,}.")
        return df_filtered, discarded_count

    logger.info("Filtro no aplicado: conservando ambas poblaciones (250k filas activas).")
    return df.copy(), 0


def apply_temporal_split(
    df: pd.DataFrame,
    date_col: str = "fecha",
    val_date: Optional[str] = None,
    test_date: Optional[str] = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> pd.DataFrame:
    """
    Parsea la fecha a datetime estricto, ordena cronológicamente y asigna la partición OOT.

    Args:
        df: DataFrame de la población activa.
        date_col: Nombre de la columna temporal.
        val_date: Fecha de corte inicial de validación (opcional, ej. '2020-04-05').
        test_date: Fecha de corte inicial de test (opcional, ej. '2020-04-15').
        train_ratio: Proporción para train cuando no se especifican fechas (default: 0.70).
        val_ratio: Proporción para val cuando no se especifican fechas (default: 0.15).
        test_ratio: Proporción para test cuando no se especifican fechas (default: 0.15).

    Returns:
        pd.DataFrame: DataFrame ordenado con columna categórica 'split'.
    """
    logger.info(f"Parseando columna temporal '{date_col}' a datetime estricto...")
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="raise")

    logger.info("Ordenando cronológicamente la población activa...")
    df = df.sort_values(by=date_col, ascending=True).reset_index(drop=True)

    n_rows = len(df)
    split_series = pd.Series(index=df.index, dtype=object)

    if val_date is not None and test_date is not None:
        logger.info(f"Generando split OOT por fechas de corte: val_date={val_date}, test_date={test_date}")
        val_dt = pd.to_datetime(val_date)
        test_dt = pd.to_datetime(test_date)

        if val_dt >= test_dt:
            raise ValueError(f"val_date ({val_date}) debe ser estrictamente menor que test_date ({test_date}).")

        split_series[df[date_col] < val_dt] = "train"
        split_series[(df[date_col] >= val_dt) & (df[date_col] < test_dt)] = "val"
        split_series[df[date_col] >= test_dt] = "test"
    else:
        logger.info(
            f"Generando split OOT por cortes porcentuales cronológicos: "
            f"Train={train_ratio*100:.1f}%, Val={val_ratio*100:.1f}%, Test={test_ratio*100:.1f}%"
        )
        total_ratio = train_ratio + val_ratio + test_ratio
        if not np.isclose(total_ratio, 1.0, atol=1e-3):
            raise ValueError(f"Los ratios deben sumar 1.0 (suma actual: {total_ratio:.4f}).")

        train_cutoff = int(n_rows * train_ratio)
        val_cutoff = int(n_rows * (train_ratio + val_ratio))

        split_series.iloc[:train_cutoff] = "train"
        split_series.iloc[train_cutoff:val_cutoff] = "val"
        split_series.iloc[val_cutoff:] = "test"

    df["split"] = pd.Categorical(split_series, categories=["train", "val", "test"], ordered=True)
    return df


def clean_final_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina columnas de índice residual (...1) y auxiliares de trazabilidad
    (es_bloque_2, origen_bloque) que quedan obsoletas tras el descarte del Bloque 2.
    Reorganiza 'transaction_id' como clave primaria al inicio.

    Args:
        df: DataFrame ordenado con split asignado.

    Returns:
        pd.DataFrame: DataFrame limpio sin columnas redundantes.
    """
    cols_to_drop = [
        c for c in ["...1", "Unnamed: 0", "es_bloque_2", "origen_bloque"] if c in df.columns
    ]
    if cols_to_drop:
        logger.info(f"Removiendo columnas auxiliares y de índice residual: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    if "transaction_id" in df.columns:
        cols_order = ["transaction_id"] + [c for c in df.columns if c != "transaction_id"]
        df = df[cols_order]

    return df


def save_processed_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """
    Exporta el dataset procesado a formato Parquet preservando tipos nativos.
    Utiliza escritura atómica con archivo temporal para prevenir bloqueos por memory-mapping
    (WinError 1224 cuando está abierto por librerías como arrow en R).

    Args:
        df: DataFrame final procesado.
        output_path: Ruta destino del archivo Parquet.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Guardando dataset procesado en: {output_path.resolve()}")
    
    # Escritura atómica segura
    tmp_path = output_path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp_path, engine="pyarrow", index=False)
    import os
    os.replace(tmp_path, output_path)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Dataset exportado exitosamente. Tamaño del archivo: {file_size_mb:.2f} MB.")


def print_diagnostic_report(
    raw_df_with_trace: pd.DataFrame,
    processed_df: pd.DataFrame,
    discarded_rows: int,
) -> None:
    """
    Imprime en consola un reporte de diagnóstico y calidad de datos detallado.

    Args:
        raw_df_with_trace: DataFrame completo con trazabilidad de bloques.
        processed_df: DataFrame resultante final luego de filtros y splits.
        discarded_rows: Número de registros excluidos.
    """
    sep = "=" * 80
    sub_sep = "-" * 80

    print("\n" + sep)
    print("                      REPORTE DE DIAGNÓSTICO Y CALIDAD")
    print(sep)

    # 1. Resumen de Población
    print("\n1. VOLUMETRÍA Y EXCLUSIÓN DE DATOS:")
    print(sub_sep)
    print(f"  • Total registros originales: {len(raw_df_with_trace):>10,}")
    print(f"  • Registros descartados:      {discarded_rows:>10,}")
    print(f"  • Registros conservados:      {len(processed_df):>10,}")

    # 2. Rango de Fechas por Bloque de Origen
    print("\n2. COBERTURA TEMPORAL POR BLOQUE DE ORIGEN (Dataset Completo):")
    print(sub_sep)
    temp_raw = raw_df_with_trace.copy()
    temp_raw["fecha_dt"] = pd.to_datetime(temp_raw["fecha"], errors="coerce")
    block_dates = (
        temp_raw.groupby("origen_bloque")["fecha_dt"]
        .agg(["count", "min", "max"])
        .rename(columns={"count": "Registros", "min": "Fecha Mínima", "max": "Fecha Máxima"})
    )
    for block_name, row in block_dates.iterrows():
        print(f"  [{block_name.upper()}]:")
        print(f"    - N° Registros:  {int(row['Registros']):,}")
        print(f"    - Fecha Mínima:  {row['Fecha Mínima']}")
        print(f"    - Fecha Máxima:  {row['Fecha Máxima']}")

    # 3. Diagnóstico por Split OOT
    print("\n3. PARTICIÓN TEMPORAL (OUT-OF-TIME) Y CALIDAD POR SPLIT:")
    print(sub_sep)
    template_header = "{:<8} | {:>9} | {:^19} | {:^19} | {:>12} | {:>10} | {:>9}"
    print(
        template_header.format(
            "Split", "Filas", "Fecha Mínima", "Fecha Máxima", "Tasa Fraude", "NaN Fraude", "NaN Score"
        )
    )
    print("-" * 96)

    for split_val in ["train", "val", "test"]:
        split_data = processed_df[processed_df["split"] == split_val]
        count = len(split_data)
        if count > 0:
            min_date = str(split_data["fecha"].min())
            max_date = str(split_data["fecha"].max())

            nan_fraude = split_data["fraude"].isna().sum()
            nan_score = split_data["score"].isna().sum()

            valid_fraude = split_data["fraude"].dropna()
            if len(valid_fraude) > 0:
                positives = (valid_fraude == 1).sum()
                fraud_rate_str = f"{(positives / len(valid_fraude)) * 100:.2f}% ({positives:,})"
            else:
                fraud_rate_str = "N/A"
        else:
            min_date, max_date = "N/A", "N/A"
            nan_fraude, nan_score = 0, 0
            fraud_rate_str = "0.00% (0)"

        print(
            "{:<8} | {:>9,d} | {:^19} | {:^19} | {:>12} | {:>10,d} | {:>9,d}".format(
                split_val.upper(),
                count,
                min_date[:19],
                max_date[:19],
                fraud_rate_str,
                nan_fraude,
                nan_score,
            )
        )

    print(sep + "\n")


def build_parser() -> argparse.ArgumentParser:
    """Configura los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Pipeline modular de ingesta, trazabilidad y partición temporal OOT para detección de fraude."
    )
    # Por defecto busca en la ruta raíz del repo o relativa
    default_input = Path("data/raw/fraud_dataset_v2.csv")
    if not default_input.exists():
        default_input = Path("../data/raw/fraud_dataset_v2.csv")

    parser.add_argument(
        "--input_path",
        type=Path,
        default=default_input,
        help="Ruta al archivo CSV crudo de entrada.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("data/processed/dataset_base.parquet"),
        help="Ruta destino para guardar el dataset en formato Parquet.",
    )
    parser.add_argument(
        "--drop_second_block",
        action="store_true",
        default=True,
        help="Descarta las 100k filas del segundo bloque (True por defecto tras corroborar en EDA índices duplicados y fraude=0 artificial).",
    )
    parser.add_argument(
        "--keep_second_block",
        action="store_true",
        default=False,
        help="Fuerza conservar las 250k filas originales desactivando el descarte del Bloque 2.",
    )
    parser.add_argument(
        "--val_date",
        type=str,
        default=None,
        help="Fecha de corte para inicio de validación (ej: '2020-04-05' o '2020-04-05 00:00:00').",
    )
    parser.add_argument(
        "--test_date",
        type=str,
        default=None,
        help="Fecha de corte para inicio de test (ej: '2020-04-15' o '2020-04-15 00:00:00').",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.70,
        help="Proporción cronológica de train si no se especifican fechas de corte (default: 0.70).",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Proporción cronológica de val si no se especifican fechas de corte (default: 0.15).",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Proporción cronológica de test si no se especifican fechas de corte (default: 0.15).",
    )
    return parser


def main() -> None:
    """Función principal de ejecución del pipeline."""
    parser = build_parser()
    args = parser.parse_args()

    # 1. Ingesta y normalización inicial
    raw_df = load_raw_data(args.input_path)

    # 2. Trazabilidad global e identificación de bloques
    trace_df = add_traceability_features(raw_df, block_1_size=150_000)

    # 3. Capa de exclusión parametrizada (por defecto excluye bloque 2 salvo --keep_second_block)
    should_drop = args.drop_second_block and not args.keep_second_block
    active_df, discarded_count = filter_population(
        trace_df, drop_second_block=should_drop
    )

    # 4. Partición temporal OOT (Out-Of-Time)
    processed_df = apply_temporal_split(
        active_df,
        date_col="fecha",
        val_date=args.val_date,
        test_date=args.test_date,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    # 5. Limpieza de columnas residuales (...1, es_bloque_2, origen_bloque)
    final_df = clean_final_columns(processed_df)

    # 6. Exportación a Parquet
    save_processed_dataset(final_df, args.output_path)

    # 7. Diagnóstico y reporte por consola
    print_diagnostic_report(
        raw_df_with_trace=trace_df,
        processed_df=final_df,
        discarded_rows=discarded_count,
    )


if __name__ == "__main__":
    main()
