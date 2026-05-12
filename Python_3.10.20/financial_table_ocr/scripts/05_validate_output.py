from pathlib import Path
import pandas as pd
import numpy as np
import re

# ============================================
# PATHS
# ============================================

BASE_DIR = Path(
    "/home/alvaro/Python/Curso_Python/Python_3.10.20/financial_table_ocr"
)

# ============================================
# INPUT
# ============================================

FINAL_DIR = (
    BASE_DIR /
    "outputs" /
    "final_output"
)

# ============================================
# LOGS
# ============================================

LOG_DIR = (
    BASE_DIR /
    "outputs" /
    "logs"
)

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================
# FILES
# ============================================

FINAL_XLSX = (
    FINAL_DIR /
    "Restricted_Final.xlsx"
)

REPORT_FILE = (
    LOG_DIR /
    "validation_report.txt"
)

# ============================================
# CONFIG
# ============================================

MAX_NA_RATIO_WARNING = 0.50

# ============================================
# VALIDADORES
# ============================================

financial_pattern = re.compile(
    r"^-?[\d,\.]+\s*[BMK%]$",
    re.IGNORECASE
)

# ============================================
# LOGGING
# ============================================

report_lines = []

def log(message):

    print(message)

    report_lines.append(message)

# ============================================
# VALIDAR ARCHIVO
# ============================================

log("\n===================================")
log("VALIDACIÓN OCR FINANCIERA")
log("===================================\n")

log(f"Archivo esperado:")
log(str(FINAL_XLSX))
log("")

# ============================================

if not FINAL_XLSX.exists():

    raise Exception(
        f"No existe archivo final:\n{FINAL_XLSX}"
    )

# ============================================
# CARGAR DATAFRAME
# ============================================

df = pd.read_excel(FINAL_XLSX)

# ============================================
# VALIDAR VACÍO
# ============================================

if df.empty:

    raise Exception(
        "El DataFrame final está vacío"
    )

# ============================================
# INFO GENERAL
# ============================================

log("===================================")
log("INFO GENERAL")
log("===================================\n")

log(f"Filas: {df.shape[0]}")
log(f"Columnas: {df.shape[1]}")
log("")

# ============================================
# VALIDAR HEADERS DUPLICADOS
# ============================================

log("===================================")
log("HEADERS DUPLICADOS")
log("===================================\n")

duplicated_headers = df.columns[
    df.columns.duplicated()
]

if len(duplicated_headers) == 0:

    log("OK -> No hay headers duplicados\n")

else:

    for h in duplicated_headers:

        log(f"DUPLICADO: {h}")

# ============================================
# VALIDAR COLUMNAS VACÍAS
# ============================================

log("\n===================================")
log("COLUMNAS COMPLETAMENTE VACÍAS")
log("===================================\n")

empty_columns = []

for col in df.columns:

    na_ratio = (
        df[col]
        .astype(str)
        .isin(["NA", "nan"])
        .mean()
    )

    if na_ratio == 1.0:

        empty_columns.append(col)

if len(empty_columns) == 0:

    log("OK -> No hay columnas vacías\n")

else:

    for col in empty_columns:

        log(f"VACÍA: {col}")

# ============================================
# VALIDAR COLUMNAS SOSPECHOSAS
# ============================================

log("\n===================================")
log("COLUMNAS CON MUCHOS NA")
log("===================================\n")

suspicious_columns = []

for col in df.columns:

    na_ratio = (
        df[col]
        .astype(str)
        .isin(["NA", "nan"])
        .mean()
    )

    if na_ratio >= MAX_NA_RATIO_WARNING:

        suspicious_columns.append(
            (col, na_ratio)
        )

if len(suspicious_columns) == 0:

    log("OK -> No hay columnas sospechosas\n")

else:

    for col, ratio in suspicious_columns:

        log(
            f"{col} -> "
            f"{ratio:.2%} NA"
        )

# ============================================
# VALIDAR VALORES FINANCIEROS
# SIN NORMALIZAR
# ============================================

log("\n===================================")
log("VALORES FINANCIEROS NO NORMALIZADOS")
log("===================================\n")

financial_issues = []

for col in df.columns:

    for idx, value in enumerate(df[col]):

        if pd.isna(value):

            continue

        value_str = str(value).strip()

        if financial_pattern.match(value_str):

            financial_issues.append(

                (
                    idx,
                    col,
                    value_str
                )

            )

if len(financial_issues) == 0:

    log("OK -> No se detectaron valores financieros sin convertir\n")

else:

    for row, col, val in financial_issues:

        log(
            f"Fila {row} | "
            f"Columna: {col} | "
            f"Valor: {val}"
        )

# ============================================
# VALIDAR TIPOS NUMÉRICOS
# ============================================

log("\n===================================")
log("VALIDACIÓN COLUMNAS NUMÉRICAS")
log("===================================\n")

numeric_warnings = []

for col in df.columns:

    # ========================================
    # IGNORAR TICKERS
    # ========================================

    if "Ticker" in str(col):

        continue

    series = df[col]

    # ========================================
    # CONTAR NUMÉRICOS
    # ========================================

    numeric_ratio = pd.to_numeric(
        series,
        errors="coerce"
    ).notna().mean()

    # ========================================
    # SI PARECE NUMÉRICA
    # ========================================

    if numeric_ratio >= 0.60:

        # ====================================
        # BUSCAR STRINGS SOSPECHOSOS
        # ====================================

        for idx, value in enumerate(series):

            if pd.isna(value):

                continue

            try:

                float(value)

            except:

                if str(value) != "NA":

                    numeric_warnings.append(

                        (
                            idx,
                            col,
                            value
                        )

                    )

if len(numeric_warnings) == 0:

    log("OK -> No se detectaron strings sospechosos en columnas numéricas\n")

else:

    for row, col, val in numeric_warnings:

        log(
            f"Fila {row} | "
            f"Columna: {col} | "
            f"Valor sospechoso: {val}"
        )

# ============================================
# VALIDAR FILAS MUY VACÍAS
# ============================================

log("\n===================================")
log("FILAS SOSPECHOSAS")
log("===================================\n")

suspicious_rows = []

for idx in range(len(df)):

    row = df.iloc[idx]

    na_ratio = (
        row.astype(str)
        .isin(["NA", "nan"])
        .mean()
    )

    if na_ratio >= 0.70:

        suspicious_rows.append(
            (idx, na_ratio)
        )

if len(suspicious_rows) == 0:

    log("OK -> No se detectaron filas sospechosas\n")

else:

    for idx, ratio in suspicious_rows:

        log(
            f"Fila {idx} -> "
            f"{ratio:.2%} NA"
        )

# ============================================
# EXPORTAR REPORTE
# ============================================

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    for line in report_lines:

        f.write(line + "\n")

# ============================================
# FINAL
# ============================================

log("\n===================================")
log("VALIDACIÓN COMPLETADA")
log("===================================\n")

log(f"Reporte generado:")
log(str(REPORT_FILE))