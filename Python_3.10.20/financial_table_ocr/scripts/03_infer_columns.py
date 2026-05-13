from pathlib import Path
import pandas as pd
import numpy as np

# ============================================
# PATHS
# ============================================

BASE_DIR = Path(
    "/home/alvaro/Python/Curso_Python/Python_3.10.20/financial_table_ocr"
)

OCR_DIR = BASE_DIR / "outputs" / "ocr_coordinates"

OUTPUT_DIR = BASE_DIR / "outputs" / "column_inferred_tables"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================
# CONFIG
# ============================================

ROW_Y_THRESHOLD = 22
COLUMN_X_THRESHOLD = 45
MIN_TEXT_LENGTH = 1

# ============================================
# FUNCIONES
# ============================================

def group_rows(df):
    df = df.sort_values(["y", "x"]).reset_index(drop=True)

    rows = []
    current_row = []
    current_y = None

    for _, item in df.iterrows():
        y = int(item["y"])

        if current_y is None:
            current_y = y
            current_row.append(item)
            continue

        if abs(y - current_y) <= ROW_Y_THRESHOLD:
            current_row.append(item)
        else:
            rows.append(current_row)
            current_row = [item]
            current_y = y

    if len(current_row) > 0:
        rows.append(current_row)

    return rows


def infer_column_anchors(df):
    """
    Detecta columnas globales usando el borde izquierdo X.
    Esto funcionó mejor en tu tabla que usar el centro del cuadro.
    """

    x_values = sorted(
        df["x"].astype(int).tolist()
    )

    anchors = []

    for x in x_values:
        if not anchors:
            anchors.append(x)
            continue

        if abs(x - anchors[-1]) <= COLUMN_X_THRESHOLD:
            anchors[-1] = int((anchors[-1] + x) / 2)
        else:
            anchors.append(x)

    return anchors


def nearest_column(x, anchors):
    distances = [
        abs(x - a)
        for a in anchors
    ]

    return int(np.argmin(distances))


def reconstruct_table(rows, anchors):
    table = []

    for row in rows:
        row_cells = [""] * len(anchors)

        row = sorted(
            row,
            key=lambda r: r["x"]
        )

        for item in row:
            text = str(item["text"]).strip()

            if text == "":
                continue

            x = int(item["x"])

            col_idx = nearest_column(
                x,
                anchors
            )

            if row_cells[col_idx] == "":
                row_cells[col_idx] = text
            else:
                if text not in row_cells[col_idx]:
                    row_cells[col_idx] += " " + text

        table.append(row_cells)

    return table


def clean_dataframe(df):
    df = df.replace("", np.nan)

    df = df.dropna(
        how="all"
    )

    df = df.fillna("NA")

    df = df.loc[
        :,
        ~(df == "NA").all()
    ]

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

if len(csv_files) == 0:
    raise Exception(
        f"No se encontraron archivos OCR CSV en: {OCR_DIR}"
    )

for csv_file in csv_files:
    print(f"\nProcesando: {csv_file.name}")

    df = pd.read_csv(csv_file)

    if df.empty:
        print(f"CSV vacío: {csv_file.name}")
        continue

    df["text"] = df["text"].astype(str)

    # IMPORTANTE:
    # No eliminar texto de 1 carácter.
    # Esto preserva valores como 7, 8, 9.
    df = df[
        (df["text"].str.strip() != "") &
        (df["text"].str.len() >= MIN_TEXT_LENGTH)
    ].copy()

    rows = group_rows(df)

    print(f"Filas detectadas: {len(rows)}")

    anchors = infer_column_anchors(df)

    print(f"Columnas inferidas: {len(anchors)}")
    print(f"Anchors X: {anchors}")

    table = reconstruct_table(
        rows,
        anchors
    )

    reconstructed_df = pd.DataFrame(table)

    reconstructed_df = clean_dataframe(
        reconstructed_df
    )

    name = csv_file.stem.replace(
        "_ocr",
        ""
    )

    output_csv = OUTPUT_DIR / f"{name}_body.csv"
    output_xlsx = OUTPUT_DIR / f"{name}_body.xlsx"

    reconstructed_df.to_csv(
        output_csv,
        index=False,
        header=False,
        encoding="utf-8-sig"
    )

    reconstructed_df.to_excel(
        output_xlsx,
        index=False,
        header=False
    )

    print(f"CSV guardado: {output_csv}")
    print(f"Excel guardado: {output_xlsx}")

print("\n===================================")
print("INFERENCIA DE COLUMNAS COMPLETADA")
print("===================================")