import os
from PyPDF2 import PdfReader, PdfWriter

# Contraseña de los PDFs
password = "18391106"

# Rutas de los archivos PDF
archivos_pdf = [
    r"C:\Users\Alvaro\Downloads\1 Certificado CDT.pdf",
    r"C:\Users\Alvaro\Downloads\2 Certificado CDT.pdf",
    r"C:\Users\Alvaro\Downloads\3 Certificado CDT.pdf",
    r"C:\Users\Alvaro\Downloads\4 Certificado CDT.pdf",
    r"C:\Users\Alvaro\Downloads\5 Certificado CDT.pdf"
]

# Crear un escritor PDF
pdf_final = PdfWriter()

for ruta in archivos_pdf:
    with open(ruta, "rb") as archivo:
        lector = PdfReader(archivo)
        if lector.is_encrypted:
            lector.decrypt(password)
        for pagina in lector.pages:
            pdf_final.add_page(pagina)

# Guardar el archivo combinado
salida = r"C:\Users\Alvaro\Downloads\Certificados_Unidos.pdf"
with open(salida, "wb") as archivo_salida:
    pdf_final.write(archivo_salida)

print("PDFs unidos exitosamente en:", salida)