import pdfplumber

PDF_ENTRADA = r"C:\Users\Alvaro\Downloads\Sherwin Willians Contractor Facility Maintenance Catalog.pdf"
SALIDA = r"C:\Users\Alvaro\Downloads\salida_organizada.md"

with pdfplumber.open(PDF_ENTRADA) as pdf:
    texto = ""
    for i, pagina in enumerate(pdf.pages, start=1):
        texto_pagina = pagina.extract_text() or ""
        texto += f"\n--- Página {i} ---\n{texto_pagina}\n"

with open(SALIDA, "w", encoding="utf-8") as archivo_salida:
    archivo_salida.write(texto)