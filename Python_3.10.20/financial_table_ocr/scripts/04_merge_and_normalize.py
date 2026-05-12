from pathlib import Path
import pandas as pd
import numpy as np
import re
import unicodedata

# ============================================
# PATHS
# ============================================

BASE_DIR = Path(
    "/home/alvaro/Python/Curso_Python/Python_3.10.20/financial_table_ocr"
)

INPUT_DIR = BASE_DIR / "outputs" / "column_inferred_tables"
OUTPUT_DIR = BASE_DIR / "outputs" / "final_output"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================
# HEADER MAPPING
# ============================================

HEADER_MAPPING = {
    "Revenue Forecast Count of":
        "Revenue Forecast Count of Estimates",

    "Change In Accounts":
        "Change In Accounts Receivable",

    "Net Income to Common Excl":
        "Net Income to Common Excl Extra Items",

    "Merger & Related Restructuring":
        "Merger & Related Restructuring Charges"
}

# ============================================
# FUNCIONES
# ============================================

def clean_header(header):
    """
    Limpia encabezados OCR para hacerlos legibles
    y comparables con diccionarios externos.
    """

    header = str(header)

    header = unicodedata.normalize(
        "NFKC",
        header
    )

    header = header.replace("\n", " ")
    header = header.replace("\r", " ")

    header = " ".join(header.split())

    return header.strip()


def rename_headers(df):
    """
    Limpia y renombra headers específicos.
    """

    new_columns = []

    for col in df.columns:

        col_clean = clean_header(col)

        if col_clean in HEADER_MAPPING:

            new_columns.append(
                HEADER_MAPPING[col_clean]
            )

        else:

            new_columns.append(col_clean)

    df.columns = new_columns

    df = df.loc[
        :,
        ~df.columns.duplicated()
    ]

    return df


def is_null_like(value):
    """
    Detecta vacíos, guiones y nulos OCR.
    """

    if pd.isna(value):
        return True

    value = str(value).strip()

    return value in [
        "",
        "-",
        "—",
        "–",
        "nan",
        "NaN",
        "None"
    ]


def is_numeric_candidate(value):
    """
    Detecta posibles valores financieros:
    2.98B, 1.52M, 812K, 23%, (2.5B), -1.2M.
    """

    if is_null_like(value):
        return True

    value = str(value).strip()
    value = value.replace(",", "")

    pattern = r"^\(?-?\d+(\.\d+)?\s*[BMK%]?\)?$"

    return bool(
        re.match(
            pattern,
            value,
            re.IGNORECASE
        )
    )


def convert_financial_value(value):
    """
    Convierte:
    2.54B -> 2540000000
    3.2M -> 3200000
    812K -> 812000
    23% -> 0.23
    (2.5B) -> -2500000000
    - -> pd.NA
    vacío -> pd.NA
    """

    if is_null_like(value):

        return pd.NA

    value = str(value).strip()
    value = value.replace(",", "")

    negative = False

    if value.startswith("(") and value.endswith(")"):

        negative = True
        value = value[1:-1].strip()

    multiplier = 1

    if value.upper().endswith("B"):

        multiplier = 1_000_000_000
        value = value[:-1].strip()

    elif value.upper().endswith("M"):

        multiplier = 1_000_000
        value = value[:-1].strip()

    elif value.upper().endswith("K"):

        multiplier = 1_000
        value = value[:-1].strip()

    elif value.endswith("%"):

        multiplier = 0.01
        value = value[:-1].strip()

    try:

        result = float(value) * multiplier

        if negative:
            result = -result

        return result

    except:

        return str(value).strip()


def normalize_dataframe(df):
    """
    Limpieza y normalización financiera.
    """

    df = rename_headers(df)

    df = df.replace(
        ["", "-", "—", "–"],
        pd.NA
    )

    for col in df.columns:

        col_name = str(col)

        if "ticker" in col_name.lower():

            df[col] = (
                df[col]
                .astype("string")
                .str.upper()
            )

            df[col] = df[col].fillna("NA")

            continue

        numeric_ratio = (
            df[col]
            .apply(is_numeric_candidate)
            .mean()
        )

        if numeric_ratio >= 0.40:

            df[col] = df[col].apply(
                convert_financial_value
            )

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    df = df.replace(
        ["", "-", "—", "–"],
        pd.NA
    )

    df = df.loc[
        :,
        ~(df.isna().all())
    ]

    return df


def make_unique_columns(columns):
    """
    Evita columnas duplicadas después del merge.
    """

    seen = {}
    result = []

    for col in columns:

        col = clean_header(col)

        if col not in seen:

            seen[col] = 0
            result.append(col)

        else:

            seen[col] += 1
            result.append(f"{col}_{seen[col]}")

    return result


# ============================================
# LEER ARCHIVOS
# ============================================

xlsx_files = sorted(
    INPUT_DIR.glob("*_columns.xlsx")
)

print("\n===================================")
print("ARCHIVOS DETECTADOS")
print("===================================")

for f in xlsx_files:
    print(f.name)

if len(xlsx_files) == 0:
    raise Exception("No se encontraron archivos XLSX")

# ============================================
# CARGAR DATAFRAMES
# ============================================

dfs = []

for file in xlsx_files:

    print(f"\nProcesando: {file.name}")

    df = pd.read_excel(file)

    if df.empty:

        print(f"Archivo vacío: {file.name}")
        continue

    df = normalize_dataframe(df)
    df = df.reset_index(drop=True)

    dfs.append(df)

if len(dfs) == 0:
    raise Exception("No se generaron DataFrames válidos")

# ============================================
# MERGE HORIZONTAL POR ORDEN DE FILA
# ============================================

master_df = dfs[0]

for df in dfs[1:]:

    duplicate_cols = [
        c for c in df.columns
        if c in master_df.columns
    ]

    df = df.drop(
        columns=duplicate_cols,
        errors="ignore"
    )

    master_df = pd.concat(
        [
            master_df.reset_index(drop=True),
            df.reset_index(drop=True)
        ],
        axis=1
    )

master_df.columns = make_unique_columns(
    master_df.columns
)

# ============================================
# PREPARAR EXPORT
# ============================================

export_df = master_df.copy()

export_df = export_df.replace(
    ["", "-", "—", "–"],
    pd.NA
)

export_df = export_df.fillna("NA")

# ============================================
# EXPORT FINAL
# ============================================

final_csv = OUTPUT_DIR / "Restricted_Final.csv"
final_xlsx = OUTPUT_DIR / "Restricted_Final.xlsx"

export_df.to_csv(
    final_csv,
    index=False,
    encoding="utf-8-sig"
)

export_df.to_excel(
    final_xlsx,
    index=False
)

print("\n===================================")
print("TABLA FINAL GENERADA")
print("===================================")
print(f"CSV: {final_csv}")
print(f"XLSX: {final_xlsx}")

print("\n===================================")
print("NORMALIZACIÓN COMPLETADA")
print("===================================")