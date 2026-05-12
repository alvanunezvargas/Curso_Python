import cv2
import numpy as np
from pathlib import Path

# ============================================
# PATHS
# ============================================

BASE_DIR = Path("/home/alvaro/Python/Curso_Python/Python_3.10.20/financial_table_ocr")

IMAGE_DIR = BASE_DIR / "images"

OUTPUT_DIR = BASE_DIR / "outputs" / "preprocess"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================
# FUNCIONES
# ============================================

def save_image(name, image, folder):

    output_path = folder / name

    cv2.imwrite(str(output_path), image)

    print(f"Guardado: {output_path}")


def upscale(image, scale=2):

    return cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )


def sharpen(image):

    kernel = np.array([
        [0, -1, 0],
        [-1, 5,-1],
        [0, -1, 0]
    ])

    return cv2.filter2D(image, -1, kernel)


def apply_clahe(gray):

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )

    return clahe.apply(gray)


# ============================================
# PROCESAMIENTO
# ============================================

images = sorted(IMAGE_DIR.glob("*"))

print("\n===================================")
print("IMÁGENES DETECTADAS")
print("===================================")

for img_path in images:
    print(img_path.name)

# ============================================

for img_path in images:

    print(f"\nProcesando: {img_path.name}")

    image = cv2.imread(str(img_path))

    if image is None:
        print(f"ERROR leyendo {img_path}")
        continue

    name = img_path.stem

    # ========================================
    # Crear carpeta específica
    # ========================================

    img_output = OUTPUT_DIR / name

    img_output.mkdir(exist_ok=True)

    # ========================================
    # ORIGINAL
    # ========================================

    save_image(
        "00_original.jpg",
        image,
        img_output
    )

    # ========================================
    # GRAYSCALE
    # ========================================

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    save_image(
        "01_grayscale.jpg",
        gray,
        img_output
    )

    # ========================================
    # OTSU
    # ========================================

    otsu = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    save_image(
        "02_otsu.jpg",
        otsu,
        img_output
    )

    # ========================================
    # ADAPTIVE GAUSSIAN
    # ========================================

    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    save_image(
        "03_adaptive_gaussian.jpg",
        adaptive,
        img_output
    )

    # ========================================
    # CLAHE
    # ========================================

    clahe = apply_clahe(gray)

    save_image(
        "04_clahe.jpg",
        clahe,
        img_output
    )

    # ========================================
    # SHARPEN
    # ========================================

    sharp = sharpen(gray)

    save_image(
        "05_sharpen.jpg",
        sharp,
        img_output
    )

    # ========================================
    # BILATERAL FILTER
    # ========================================

    bilateral = cv2.bilateralFilter(
        gray,
        9,
        75,
        75
    )

    save_image(
        "06_bilateral.jpg",
        bilateral,
        img_output
    )

    # ========================================
    # UPSCALE 2X
    # ========================================

    upscale_2x = upscale(gray, 2)

    save_image(
        "07_upscale_2x.jpg",
        upscale_2x,
        img_output
    )

    # ========================================
    # MORPH CLOSE
    # ========================================

    kernel = np.ones((2,2), np.uint8)

    morph = cv2.morphologyEx(
        gray,
        cv2.MORPH_CLOSE,
        kernel
    )

    save_image(
        "08_morph_close.jpg",
        morph,
        img_output
    )

    # ========================================
    # CANNY EDGES
    # ========================================

    edges = cv2.Canny(
        gray,
        100,
        200
    )

    save_image(
        "09_canny_edges.jpg",
        edges,
        img_output
    )

    # ========================================
    # COMBINED PIPELINE
    # ========================================

    combo = upscale(gray, 2)

    combo = apply_clahe(combo)

    combo = sharpen(combo)

    combo = cv2.adaptiveThreshold(
        combo,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    save_image(
        "10_combined_enhanced.jpg",
        combo,
        img_output
    )

print("\n===================================")
print("PREPROCESSING COMPLETADO")
print("===================================")