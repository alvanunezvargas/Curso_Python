
"""
Este código esta diseñado para analizar un documento con palabras para una sopa de letras, ayudando a identificar palabras que se repiten (posibles errores o duplicados no deseados) y extraer la estructura temática del documento.

Este código procesa un archivo PDF que contiene palabras clave organizadas por temas y realiza dos tareas principales:

1. Detecta palabras duplicadas:
Extrae todas las palabras de al menos 3 caracteres del PDF
Las normaliza removiendo tildes y convirtiendo a minúsculas para comparar sin diferencias de acentuación
Cuenta cuántas veces aparece cada palabra normalizada
Muestra solo las palabras que se repiten más de una vez, junto con su número de apariciones

2. Identifica los temas:
Busca líneas que siguen el formato de encabezado: número + punto + texto en mayúsculas (ej: "1. ANIMALES")
Extrae y lista todos estos encabezados de temas encontrados en el documento
Excluye estas líneas de encabezado del análisis de palabras duplicadas
"""

import PyPDF2
import unicodedata
import re
from collections import defaultdict, Counter

# Ruta al PDF con las palabras clave
file_path = 'C:/Users/Alvaro/Downloads/Palabras Sopa de Letras.pdf'

# Función para normalizar cadenas (remover tildes y pasar a minúsculas)
def normalize(word: str) -> str:
    nfkd = unicodedata.normalize('NFKD', word)
    return ''.join([c for c in nfkd if not unicodedata.combining(c)]).lower()

# Leer y extraer todo el texto del PDF
reader = PyPDF2.PdfReader(file_path)
full_text = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        full_text += text + "\n"

# Diccionarios para conteo y para mapear normalizado -> lista de formas originales
count_norm = defaultdict(int)
original_forms = defaultdict(list)

# Procesar líneas
for line in full_text.splitlines():
    # Ignorar encabezados de tema
    if re.match(r'^\d+\.\s*[A-ZÁÉÍÓÚÑÜ\s]+$', line):
        continue

    # Extraer palabras de al menos 3 caracteres
    palabras = re.findall(r'\b\w{3,}\b', line)

    for palabra in palabras:
        palabra_norm = normalize(palabra)
        count_norm[palabra_norm] += 1
        original_forms[palabra_norm].append(palabra)

# Filtrar solo las palabras duplicadas (ignorando tildes y mayúsculas)
duplicates = {word: count for word, count in count_norm.items() if count > 1}

# Mostrar resultados con las formas originales encontradas
print("Palabras duplicadas (forma original tal como aparece en el PDF):")
for word_norm, count in sorted(duplicates.items()):
    formas_originales = original_forms[word_norm]
    # Contar cuántas veces aparece cada forma exacta
    formas_counter = Counter(formas_originales)
    # Construir una cadena para mostrar variantes exactas y su conteo
    formas_str = ', '.join([f"{forma}({cnt})" for forma, cnt in formas_counter.items()])
    print(f"{word_norm} => {formas_str} [total: {count}]")

# Listar los temas: líneas que empiezan con número, punto y mayúsculas
temas = []
for line in full_text.splitlines():
    match = re.match(r'^(\d+\.\s*[A-ZÁÉÍÓÚÑÜ\s]+)$', line)
    if match:
        temas.append(match.group(1).strip())

print("\nTemas encontrados en el archivo:")
for tema in temas:
    print(tema)



"""
Este script tiene una función de control de calidad para verificar la integridad de los datos procesados:

Objetivo: Detectar palabras duplicadas en un archivo CSV ya filtrado.
Proceso:
Carga el archivo CSV "neto" (resultado del script anterior)
Limpia las palabras eliminando espacios extra por seguridad
Analiza duplicados contando las ocurrencias de cada palabra

Reporta el resultado:

✅ Mensaje de confirmación si no hay duplicados
⚠️ Lista de palabras duplicadas con su frecuencia si las encuentra


Características:

Incluye manejo de errores para problemas de lectura del archivo
Normaliza los datos (convierte a string y elimina espacios)
Proporciona retroalimentación clara sobre el estado de los datos

Este script actúa como una verificación final que no quedaron palabras repetidas en el archivo final, lo cual es crucial para la calidad de una sopa de letras.
"""

import pandas as pd

# === 1️⃣ RUTA DEL ARCHIVO ===
ruta_archivo = r"C:\Users\Alvaro\Downloads\Palabras Sopa de Letras Neto.csv"

# === 2️⃣ LEER CSV ===
try:
    df = pd.read_csv(ruta_archivo, sep=';', encoding='utf-8')
except Exception as e:
    print(f"Error al leer el archivo: {e}")
    exit()

# === 3️⃣ ANALIZAR DUPLICADOS ===
# Limpia espacios en blanco alrededor de cada palabra (por seguridad)
df['Palabra'] = df['Palabra'].astype(str).str.strip()

# Contar ocurrencias de cada palabra
conteo = df['Palabra'].value_counts()

# Filtrar las palabras que aparecen más de una vez
duplicados = conteo[conteo > 1]

# === 4️⃣ MOSTRAR RESULTADO ===
if duplicados.empty:
    print("✅ No se encontraron palabras duplicadas en la columna 'Palabra'.")
else:
    print("⚠️ Palabras duplicadas encontradas:")
    print(duplicados)