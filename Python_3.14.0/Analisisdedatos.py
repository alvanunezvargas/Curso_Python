# -*- coding: utf-8 -*-
"""
Desbloquea un PDF protegido con clave y guarda una copia sin contraseña.
Requiere: pip install pypdf
Compatible: Python 3.13.3, pypdf 6.1.2
"""

from pathlib import Path
from io import BytesIO
from pypdf import PdfReader, PdfWriter

PASSWORD = "18391106"

INPUT_PATH = Path(
    r"C:\Users\Alvaro\OneDrive\Documentos\Cuentas familia Nuñez-Osorio\Sebastian Nuñez Osorio\Nueva Zelanda\Solvencia Economica\CertificadoRentasTemporales.pdf"
)

OUTPUT_PATH = Path(
    r"C:\Users\Alvaro\OneDrive\Documentos\Cuentas familia Nuñez-Osorio\Sebastian Nuñez Osorio\Nueva Zelanda\Solvencia Economica\CertificadoRentasTemporalesSIN.pdf"
)

def decrypt_to_memory(pdf_path: Path, password: str) -> BytesIO:
    """
    Abre un PDF (encriptado o no) y devuelve una versión sin contraseña en memoria (BytesIO).
    Lanza RuntimeError si la clave no es válida.
    """
    reader = PdfReader(str(pdf_path))
    if reader.is_encrypted:
        ok = reader.decrypt(password)
        # decrypt() devuelve 0 si falla; 1 o 2 si tuvo éxito
        if ok == 0:
            raise RuntimeError(f"No se pudo desencriptar el archivo con la clave proporcionada: {pdf_path}")

    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)

    buffer = BytesIO()
    writer.write(buffer)
    buffer.seek(0)
    return buffer

def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"No se encontró el archivo de entrada: {INPUT_PATH}")

    # Asegurar que la carpeta de salida exista
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Generar versión sin contraseña en memoria y guardarla en disco
    buf = decrypt_to_memory(INPUT_PATH, PASSWORD)
    with open(OUTPUT_PATH, "wb") as f_out:
        f_out.write(buf.read())

    print("✅ Listo")
    print(f"Archivo sin contraseña guardado en:\n{OUTPUT_PATH}")

if __name__ == "__main__":
    main()




