from paddleocr import PaddleOCR
from pathlib import Path
import cv2
import pandas as pd
import numpy as np

# ============================================
# PATHS
# ============================================

BASE_DIR = Path(
    "/home/alvaro/Python/Curso_Python/Python_3.10.20/financial_table_ocr"
)

# ============================================
# INPUT IMAGES
# ============================================

IMAGE_DIR = (
    BASE_DIR /
    "images"
)

# ============================================
# OUTPUT OCR
# ============================================

OUTPUT_DIR = (
    BASE_DIR /
    "outputs" /
    "ocr_coordinates"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================
# DEBUG PATHS
# ============================================

print("\n===================================")
print("BASE_DIR")
print(BASE_DIR)

print("\nIMAGE_DIR")
print(IMAGE_DIR)

print("\nOUTPUT_DIR")
print(OUTPUT_DIR)
print("===================================\n")

# ============================================
# OCR ENGINE
# ============================================

print("Inicializando PaddleOCR...\n")

ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    show_log=False
)

print("PaddleOCR inicializado correctamente.\n")

# ============================================
# UPSCALE
# ============================================

def upscale(image, scale=2):

    return cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )

# ============================================
# DETECTAR IMÁGENES
# ============================================

images = sorted(

    list(IMAGE_DIR.glob("*.jpg")) +

    list(IMAGE_DIR.glob("*.JPG")) +

    list(IMAGE_DIR.glob("*.png")) +

    list(IMAGE_DIR.glob("*.jpeg")) +

    list(IMAGE_DIR.glob("*.JPEG"))

)

# ============================================
# DEBUG IMÁGENES
# ============================================

print("===================================")
print("IMÁGENES DETECTADAS")
print("===================================")

print(f"Cantidad: {len(images)}\n")

for img in images:

    print(img.name)

print("\n===================================\n")

# ============================================
# VALIDAR
# ============================================

if len(images) == 0:

    raise Exception(
        f"No se encontraron imágenes en: {IMAGE_DIR}"
    )

# ============================================
# OCR LOOP
# ============================================

for img_path in images:

    print(f"\nProcesando: {img_path.name}")

    # ========================================
    # LEER IMAGEN
    # ========================================

    image = cv2.imread(str(img_path))

    if image is None:

        print(f"ERROR leyendo imagen: {img_path}")

        continue

    print(f"Imagen cargada: {image.shape}")

    # ========================================
    # UPSCALE
    # ========================================

    image = upscale(
        image,
        scale=2
    )

    print(f"Upscale aplicado: {image.shape}")

    output_image = image.copy()

    # ========================================
    # OCR
    # ========================================

    print("Ejecutando OCR...")

    try:

        result = ocr.ocr(
            image,
            cls=True
        )

    except Exception as e:

        print(f"\nERROR OCR en {img_path.name}")
        print(e)

        continue

    # ========================================
    # EXTRAER DATOS
    # ========================================

    rows = []

    count = 0

    # ========================================
    # VALIDAR RESULTADOS
    # ========================================

    if result and result[0]:

        for line in result[0]:

            try:

                box = line[0]

                text = line[1][0]

                score = line[1][1]

                # ================================
                # COORDENADAS
                # ================================

                x_coords = [p[0] for p in box]

                y_coords = [p[1] for p in box]

                x = int(min(x_coords))
                y = int(min(y_coords))

                w = int(
                    max(x_coords) -
                    min(x_coords)
                )

                h = int(
                    max(y_coords) -
                    min(y_coords)
                )

                # ================================
                # GUARDAR FILA
                # ================================

                rows.append({

                    "text": text,

                    "x": x,

                    "y": y,

                    "w": w,

                    "h": h,

                    "score": score

                })

                count += 1

                # ================================
                # DIBUJAR OCR
                # ================================

                pts = np.array(
                    box,
                    np.int32
                )

                cv2.polylines(

                    output_image,

                    [pts],

                    True,

                    (0,255,0),

                    2
                )

                cv2.putText(

                    output_image,

                    str(count),

                    (x, y - 5),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.5,

                    (0,0,255),

                    1,

                    cv2.LINE_AA
                )

            except Exception as e:

                print(f"Error procesando bloque OCR:")
                print(e)

    # ========================================
    # RESULTADOS OCR
    # ========================================

    print(f"Bloques OCR detectados: {count}")

    # ========================================
    # VALIDAR
    # ========================================

    if count == 0:

        print("WARNING: OCR no detectó texto.")

        continue

    # ========================================
    # DATAFRAME
    # ========================================

    df = pd.DataFrame(rows)

    # ========================================
    # ORDENAR
    # ========================================

    df = df.sort_values(
        by=["y", "x"]
    )

    # ========================================
    # EXPORT CSV
    # ========================================

    name = img_path.stem

    csv_output = (
        OUTPUT_DIR /
        f"{name}_ocr.csv"
    )

    df.to_csv(

        csv_output,

        index=False,

        encoding="utf-8-sig"
    )

    print(f"CSV guardado:")
    print(csv_output)

    # ========================================
    # EXPORT IMAGEN OCR
    # ========================================

    image_output = (
        OUTPUT_DIR /
        f"{name}_ocr.jpg"
    )

    cv2.imwrite(
        str(image_output),
        output_image
    )

    print(f"Imagen OCR guardada:")
    print(image_output)

# ============================================
# FINAL
# ============================================

print("\n===================================")
print("OCR COORDINATES COMPLETADO")
print("===================================")