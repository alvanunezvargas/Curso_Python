from pathlib import Path
import pandas as pd
import numpy as np

# ============================================
# PATHS
# ============================================

BASE_DIR = Path(
    "/home/alvaro/Python/Curso_Python/Python_3.10.20/financial_table_ocr"
)

# ============================================
# INPUT:
# OCR CSV files
# ============================================

OCR_DIR = (
    BASE_DIR /
    "outputs" /
    "ocr_coordinates"
)

# ============================================
# OUTPUT:
# reconstructed tables
# ============================================

OUTPUT_DIR = (
    BASE_DIR /
    "outputs" /
    "column_inferred_tables"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================
# CONFIG
# ============================================

ROW_Y_THRESHOLD = 22

COLUMN_X_THRESHOLD = 45

# ============================================
# FUNCIONES
# ============================================

def group_rows(df):

    """
    Agrupa OCR blocks por coordenada Y
    """

    df = df.sort_values(
        ["y", "x"]
    ).reset_index(drop=True)

    rows = []

    current_row = []

    current_y = None

    for _, item in df.iterrows():

        y = item["y"]

        # ====================================
        # PRIMER ELEMENTO
        # ====================================

        if current_y is None:

            current_y = y

            current_row.append(item)

            continue

        # ====================================
        # MISMA FILA
        # ====================================

        if abs(y - current_y) <= ROW_Y_THRESHOLD:

            current_row.append(item)

        else:

            rows.append(current_row)

            current_row = [item]

            current_y = y

    # ========================================
    # AGREGAR ÚLTIMA FILA
    # ========================================

    if len(current_row) > 0:

        rows.append(current_row)

    return rows


def infer_column_anchors(df):

    """
    Detecta columnas globales usando X
    """

    x_values = sorted(
        df["x"].astype(int).tolist()
    )

    anchors = []

    for x in x_values:

        if not anchors:

            anchors.append(x)

            continue

        # ====================================
        # MISMA COLUMNA
        # ====================================

        if abs(x - anchors[-1]) <= COLUMN_X_THRESHOLD:

            anchors[-1] = int(
                (anchors[-1] + x) / 2
            )

        else:

            anchors.append(x)

    return anchors


def nearest_column(x, anchors):

    """
    Encuentra anchor más cercano
    """

    distances = [
        abs(x - a)
        for a in anchors
    ]

    return int(np.argmin(distances))


def reconstruct_table(rows, anchors):

    """
    Reconstruye tabla usando columnas inferidas
    """

    table = []

    for row in rows:

        row_cells = [""] * len(anchors)

        row = sorted(
            row,
            key=lambda r: r["x"]
        )

        for item in row:

            text = str(
                item["text"]
            ).strip()

            x = int(item["x"])

            col_idx = nearest_column(
                x,
                anchors
            )

            # ================================
            # CONCATENAR TEXTO
            # ================================

            if row_cells[col_idx] == "":

                row_cells[col_idx] = text

            else:

                row_cells[col_idx] += " " + text

        table.append(row_cells)

    return table


def make_unique_headers(headers):

    """
    Evita headers duplicados
    """

    seen = {}

    unique = []

    for h in headers:

        if h not in seen:

            seen[h] = 0

            unique.append(h)

        else:

            seen[h] += 1

            unique.append(
                f"{h}_{seen[h]}"
            )

    return unique


def merge_header_rows(df):

    """
    Fusiona fila 0 y 1
    para construir headers reales
    """

    # ========================================
    # VALIDAR
    # ========================================

    if len(df) < 2:

        return df

    row1 = (
        df.iloc[0]
        .fillna("")
        .astype(str)
    )

    row2 = (
        df.iloc[1]
        .fillna("")
        .astype(str)
    )

    headers = []

    for a, b in zip(row1, row2):

        header = f"{a} {b}".strip()

        # ====================================
        # LIMPIAR ESPACIOS
        # ====================================

        header = " ".join(
            header.split()
        )

        # ====================================
        # HEADER VACÍO
        # ====================================

        if header == "":

            header = "NA"

        headers.append(header)

    # ========================================
    # HEADERS ÚNICOS
    # ========================================

    headers = make_unique_headers(
        headers
    )

    # ========================================
    # NUEVO DATAFRAME
    # ========================================

    new_df = df.iloc[2:].copy()

    new_df.columns = headers

    return new_df


def clean_dataframe(df):

    """
    Limpieza general
    """

    # ========================================
    # VACÍOS -> NA
    # ========================================

    df = df.replace("", np.nan)

    # ========================================
    # ELIMINAR FILAS VACÍAS
    # ========================================

    df = df.dropna(
        how="all"
    )

    # ========================================
    # RELLENAR
    # ========================================

    df = df.fillna("NA")

    # ========================================
    # ELIMINAR COLUMNAS VACÍAS
    # ========================================

    df = df.loc[
        :,
        ~(df == "NA").all()
    ]

    # ========================================
    # FULL TICKER MAYÚSCULAS
    # ========================================

    for col in df.columns:

        if "Ticker" in str(col):

            df[col] = (
                df[col]
                .astype(str)
                .str.upper()
            )

    return df

# ============================================
# MAIN
# ============================================

csv_files = sorted(
    OCR_DIR.glob("*_ocr.csv")
)

print("\n===================================")
print("CSV OCR DETECTADOS")
print("===================================")

for f in csv_files:

    print(f.name)

# ============================================

for csv_file in csv_files:

    print(f"\nProcesando: {csv_file.name}")

    # ========================================
    # LEER CSV
    # ========================================

    df = pd.read_csv(csv_file)

    # ========================================
    # VALIDAR VACÍO
    # ========================================

    if df.empty:

        print(f"CSV vacío: {csv_file.name}")

        continue

    # ========================================
    # TEXTO
    # ========================================

    df["text"] = (
        df["text"]
        .astype(str)
    )

    # ========================================
    # FILTRAR BASURA OCR
    # ========================================

    df = df[

        (df["text"].str.strip() != "") &

        (df["text"].str.len() > 1)

    ].copy()

    # ========================================
    # GROUP ROWS
    # ========================================

    rows = group_rows(df)

    print(f"Filas detectadas: {len(rows)}")

    # ========================================
    # INFER COLUMNS
    # ========================================

    anchors = infer_column_anchors(df)

    print(f"Columnas inferidas: {len(anchors)}")

    # ========================================
    # RECONSTRUCT TABLE
    # ========================================

    table = reconstruct_table(
        rows,
        anchors
    )

    reconstructed_df = pd.DataFrame(table)

    # ========================================
    # MERGE HEADERS
    # ========================================

    reconstructed_df = merge_header_rows(
        reconstructed_df
    )

    # ========================================
    # CLEAN
    # ========================================

    reconstructed_df = clean_dataframe(
        reconstructed_df
    )

    # ========================================
    # EXPORT
    # ========================================

    name = csv_file.stem.replace(
        "_ocr",
        ""
    )

    output_csv = (
        OUTPUT_DIR /
        f"{name}_columns.csv"
    )

    output_xlsx = (
        OUTPUT_DIR /
        f"{name}_columns.xlsx"
    )

    # ========================================
    # CSV UTF-8
    # ========================================

    reconstructed_df.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================
    # EXCEL
    # ========================================

    reconstructed_df.to_excel(
        output_xlsx,
        index=False
    )

    print(f"CSV guardado: {output_csv}")

    print(f"Excel guardado: {output_xlsx}")

print("\n===================================")
print("INFERENCIA DE COLUMNAS COMPLETADA")
print("===================================")