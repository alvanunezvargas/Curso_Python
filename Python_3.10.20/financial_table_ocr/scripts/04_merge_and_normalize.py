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

INPUT_DIR = (
    BASE_DIR /
    "outputs" /
    "column_inferred_tables"
)

OUTPUT_DIR = (
    BASE_DIR /
    "outputs" /
    "final_output"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================
# HEADERS MANUALES EN ORDEN GLOBAL
# ============================================

MANUAL_HEADERS = [
    "Full Ticker",
    "Revenue Forecast Count of Estimates",
    "Operating Income",
    "Gross Profit",
    "Cash from Operations",
    "Capital Expenditures",
    "Cash And Equivalents",
    "Short Term Investments",
    "Cash And Short Term Investments",
    "Total Current Assets",
    "Total Current Liabilities",
    "Net Income to Stockholders",
    "Total Debt",
    "Net Debt",
    "Accounts Receivable, Net",
    "Change In Accounts Receivable",
    "Inventory",
    "Change In Inventories",
    "Net Income to Common Excl Extra Items",
    "EBT, Incl. Unusual Items",
    "EBT Excl. Unusual Items",
    "Restructuring Charges",
    "Merger & Related Restructuring Charges",
    "Impairment of Goodwill",
    "Asset Writedown",
    "Other Unusual Items",
    "Effective Tax Rate",
]

# ============================================
# COLUMNA ESTRUCTURAL QUE PUEDE NO SER DETECTADA
# ============================================

OPTIONAL_MISSING_HEADER = "Impairment of Goodwill"

# ============================================
# FUNCIONES
# ============================================

def natural_sort_key(path):
    """
    Ordena Imagen 1, Imagen 2, Imagen 3 correctamente.
    """

    text = path.name

    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    ]


def clean_header(header):
    """
    Limpia headers para que sean legibles y comparables.
    """

    header = str(header)

    header = unicodedata.normalize(
        "NFKC",
        header
    )

    header = header.replace("\n", " ")
    header = header.replace("\r", " ")

    header = " ".join(
        header.split()
    )

    return header.strip()


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
        "None",
        "NONE",
        "null",
        "NULL",
    ]


def convert_financial_value(value):
    """
    Convierte:
    2.54B -> 2540000000
    3.2M -> 3200000
    812K -> 812000
    23% -> 0.23
    (2.5B) -> -2500000000
    0K / 0M / 0B -> 0
    - -> pd.NA
    vacío -> pd.NA
    """

    if is_null_like(value):

        return pd.NA

    value = str(value).strip()
    value = value.replace(",", "")
    value = value.replace("％", "%")

    # Casos explícitos de cero con sufijo
    if value.upper() in ["0K", "0M", "0B", "0T"]:

        return 0

    # ========================================
    # NEGATIVOS CON PARÉNTESIS
    # ========================================

    negative = False

    if value.startswith("(") and value.endswith(")"):

        negative = True
        value = value[1:-1].strip()

    # ========================================
    # MULTIPLICADORES
    # ========================================

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

    elif value.upper().endswith("T"):

        multiplier = 1_000_000_000_000
        value = value[:-1].strip()

    elif value.endswith("%"):

        multiplier = 0.01
        value = value[:-1].strip()

    # ========================================
    # CONVERTIR
    # ========================================

    try:

        result = float(value) * multiplier

        if negative:
            result = -result

        return result

    except:

        return str(value).strip()


def insert_missing_structural_column(
    df,
    missing_header,
    local_position,
    source_name
):
    """
    Inserta una columna estructural faltante con NA.
    Se usa cuando PaddleOCR no detecta una columna completa
    porque todos sus valores eran guiones.
    """

    print("\n===================================")
    print("INSERCIÓN DE COLUMNA ESTRUCTURAL")
    print("===================================")
    print(f"Archivo: {source_name}")
    print(f"Columna insertada: {missing_header}")
    print(f"Posición local: {local_position}")
    print("Valor asignado: NA")

    df.insert(
        local_position,
        missing_header,
        pd.NA
    )

    return df


def assign_header_slice(df, headers_slice, source_name):
    """
    Asigna a cada archivo body solo el bloque de headers que le corresponde.
    """

    actual_cols = df.shape[1]
    expected_cols = len(headers_slice)

    print(f"\nValidando columnas para {source_name}")
    print(f"Columnas detectadas: {actual_cols}")
    print(f"Headers asignados: {expected_cols}")
    print("Headers:")
    for h in headers_slice:
        print(f" - {h}")

    if actual_cols != expected_cols:

        preview_path = (
            OUTPUT_DIR /
            f"ERROR_column_mismatch_{source_name}.csv"
        )

        df.to_csv(
            preview_path,
            index=False,
            header=False,
            encoding="utf-8-sig"
        )

        raise ValueError(
            "\n❌ ERROR DE ESTRUCTURA\n"
            f"Archivo: {source_name}\n"
            f"Columnas detectadas: {actual_cols}\n"
            f"Headers asignados: {expected_cols}\n"
            f"Se guardó una copia de diagnóstico en:\n{preview_path}\n"
            "Revisa si el OCR perdió, fusionó o creó una columna extra."
        )

    df.columns = headers_slice

    return df


def normalize_dataframe(df):
    """
    Limpieza y normalización financiera.
    """

    df.columns = [
        clean_header(c)
        for c in df.columns
    ]

    # ========================================
    # NULOS BÁSICOS
    # ========================================

    df = df.replace(
        ["", "-", "—", "–"],
        pd.NA
    )

    # ========================================
    # FULL TICKER, SI EXISTE EN ESTE BLOQUE
    # ========================================

    if "Full Ticker" in df.columns:

        df["Full Ticker"] = (
            df["Full Ticker"]
            .astype("string")
            .str.upper()
            .str.strip()
        )

    # ========================================
    # NORMALIZAR COLUMNAS NUMÉRICAS
    # ========================================

    for col in df.columns:

        if col == "Full Ticker":
            continue

        df[col] = df[col].apply(
            convert_financial_value
        )

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # ========================================
    # ELIMINAR FILAS COMPLETAMENTE VACÍAS
    # ========================================

    df = df.dropna(
        how="all"
    )

    df.reset_index(
        drop=True,
        inplace=True
    )

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
# LEER ARCHIVOS BODY
# ============================================

body_files = sorted(
    INPUT_DIR.glob("*_body.xlsx"),
    key=natural_sort_key
)

print("\n===================================")
print("ARCHIVOS BODY DETECTADOS")
print("===================================")

for f in body_files:
    print(f.name)

if len(body_files) == 0:

    raise Exception(
        f"No se encontraron archivos *_body.xlsx en:\n{INPUT_DIR}"
    )

# ============================================
# VALIDAR TOTAL DE COLUMNAS
# ============================================

file_shapes = []
total_detected_columns = 0

for file in body_files:

    temp_df = pd.read_excel(
        file,
        header=None,
        engine="openpyxl"
    )

    cols = temp_df.shape[1]

    file_shapes.append(
        {
            "file": file,
            "columns": cols
        }
    )

    total_detected_columns += cols

expected_total_columns = len(MANUAL_HEADERS)
missing_header_index = MANUAL_HEADERS.index(
    OPTIONAL_MISSING_HEADER
)

print("\n===================================")
print("VALIDACIÓN GLOBAL DE COLUMNAS")
print("===================================")

for item in file_shapes:

    print(
        f"{item['file'].name}: "
        f"{item['columns']} columnas"
    )

print(f"\nTotal detectado: {total_detected_columns}")
print(f"Total esperado: {expected_total_columns}")

# ============================================
# DETERMINAR SI SE DEBE INSERTAR COLUMNA FALTANTE
# ============================================

insert_optional_missing_column = False

if total_detected_columns == expected_total_columns:

    print("\nOK -> Se detectaron todas las columnas esperadas.")

elif total_detected_columns == expected_total_columns - 1:

    insert_optional_missing_column = True

    print("\nADVERTENCIA:")
    print(
        f"Se detectó una columna menos. "
        f"Se asumirá que la columna faltante es: "
        f"{OPTIONAL_MISSING_HEADER}"
    )

else:

    raise ValueError(
        "\n❌ ERROR GLOBAL DE COLUMNAS\n"
        f"Total detectado: {total_detected_columns}\n"
        f"Total esperado: {expected_total_columns}\n"
        "Solo se permite una columna faltante si corresponde a "
        f"'{OPTIONAL_MISSING_HEADER}'.\n"
        "Revisa si alguna imagen perdió, fusionó o creó columnas extra."
    )

# ============================================
# CARGAR, INSERTAR COLUMNA FALTANTE SI APLICA,
# ASIGNAR HEADERS Y NORMALIZAR
# ============================================

dfs = []

header_start = 0
missing_inserted = False

for item in file_shapes:

    file = item["file"]
    detected_col_count = item["columns"]

    print(f"\nProcesando: {file.name}")

    df = pd.read_excel(
        file,
        header=None,
        engine="openpyxl"
    )

    if df.empty:

        print(f"Archivo vacío: {file.name}")
        continue

    # ========================================
    # SI FALTA IMPAIRMENT OF GOODWILL
    # ========================================

    col_count_for_header_slice = detected_col_count

    if insert_optional_missing_column and not missing_inserted:

        # La columna faltante está en este bloque si su índice global
        # cae entre el inicio esperado de este bloque y el final esperado
        # considerando que esta imagen podría estar perdiendo una columna.

        expected_block_start = header_start
        expected_block_end_if_missing_here = (
            header_start +
            detected_col_count
        )

        if (
            missing_header_index >= expected_block_start
            and
            missing_header_index <= expected_block_end_if_missing_here
        ):

            local_position = (
                missing_header_index -
                expected_block_start
            )

            df = insert_missing_structural_column(
                df=df,
                missing_header=OPTIONAL_MISSING_HEADER,
                local_position=local_position,
                source_name=file.stem
            )

            col_count_for_header_slice = (
                detected_col_count + 1
            )

            missing_inserted = True

    header_end = header_start + col_count_for_header_slice

    headers_slice = MANUAL_HEADERS[
        header_start:header_end
    ]

    df = assign_header_slice(
        df,
        headers_slice,
        file.stem
    )

    df = normalize_dataframe(df)

    dfs.append(
        df.reset_index(drop=True)
    )

    header_start = header_end

# ============================================
# VALIDAR QUE LA COLUMNA FALTANTE FUE INSERTADA
# ============================================

if insert_optional_missing_column and not missing_inserted:

    raise ValueError(
        "\n❌ ERROR\n"
        f"Se esperaba insertar la columna '{OPTIONAL_MISSING_HEADER}', "
        "pero no se pudo determinar en qué archivo correspondía."
    )

if len(dfs) == 0:

    raise Exception(
        "No se generaron DataFrames válidos."
    )

# ============================================
# MERGE HORIZONTAL POR ORDEN DE FILA
# ============================================

master_df = dfs[0]

for df in dfs[1:]:

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
# VALIDAR FULL TICKER FINAL
# ============================================

if "Full Ticker" not in master_df.columns:

    raise ValueError(
        '❌ El resultado final no contiene columna "Full Ticker".'
    )

master_df = master_df[
    ~(
        master_df["Full Ticker"].isna()
        |
        (master_df["Full Ticker"].astype(str).str.strip() == "")
        |
        (master_df["Full Ticker"].astype(str).str.upper() == "NA")
    )
].copy()

master_df.reset_index(
    drop=True,
    inplace=True
)

# ============================================
# PREPARAR EXPORT
# ============================================

export_df = master_df.copy()

export_df = export_df.replace(
    ["", "-", "—", "–"],
    pd.NA
)

# Mantener números como números en pandas.
# Solo al exportar representamos faltantes como NA.
export_df = export_df.fillna("NA")

# ============================================
# EXPORT FINAL
# ============================================

final_csv = (
    OUTPUT_DIR /
    "Restricted_Final.csv"
)

final_xlsx = (
    OUTPUT_DIR /
    "Restricted_Final.xlsx"
)

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