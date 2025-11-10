"""
Este script procesa un archivo CSV de palabras para crear sopas de letras temáticas. Su función principal es:
Propósito: Filtrar y organizar palabras por tema, eliminando duplicados y limitando cada tema a máximo 20 palabras únicas.
Proceso:

Carga un archivo CSV con palabras organizadas por temas
Limpia los datos eliminando espacios en blanco de las columnas
Procesa cada tema individualmente:

Extrae las palabras del tema
Elimina duplicados (tanto dentro del tema como globalmente)
Convierte a minúsculas para uniformidad
Limita a 20 palabras por tema


Genera dos archivos de salida:

Un CSV "neto" con las palabras filtradas
Un resumen con el conteo de palabras por tema



Características destacadas:

Evita que la misma palabra aparezca en múltiples temas
Alerta cuando un tema tiene menos de 20 palabras únicas
Mantiene un registro global de palabras ya utilizadas

Es útil para preparar contenido limpio y organizado para generar sopas de letras, asegurando variedad y evitando repeticiones entre diferentes temas.
"""

import pandas as pd

# =============================
# 1️⃣ CARGAR EL ARCHIVO CSV
# =============================
df = pd.read_csv("C:/Users/Alvaro/Downloads/Palabras Sopa de Letras Bruto1.csv", sep=';')

# =============================
# 2️⃣ CONFIGURAR ESTRUCTURAS
# =============================
df.columns = df.columns.str.strip()
temas = df['Tema'].unique()
palabras_usadas = set()
resultados = []

# =============================
# 3️⃣ PROCESAR CADA TEMA
# =============================
for tema in temas:
    lista = df[df['Tema'] == tema]['Palabra'].tolist()
    palabras_unicas = []

    for palabra in lista:
        palabra_clean = palabra.strip().lower()
        if palabra_clean not in palabras_usadas and palabra_clean not in palabras_unicas:
            palabras_unicas.append(palabra_clean)
        if len(palabras_unicas) == 20:
            break

    if len(palabras_unicas) < 20:
        print(f"⚠️ Tema '{tema}' tiene solo {len(palabras_unicas)} palabras únicas. Revisa o añade más palabras.")
    
    palabras_usadas.update(palabras_unicas)

    for palabra in palabras_unicas:
        resultados.append({
            'Tema': tema,
            'Palabra': palabra
        })

# =============================
# 4️⃣ EXPORTAR RESULTADO FINAL
# =============================
final_df = pd.DataFrame(resultados)
final_df.to_csv("C:/Users/Alvaro/Downloads/Palabras Sopa de Letras Neto1.csv", index=False)

# =============================
# 5️⃣ GENERAR RESUMEN POR TEMA
# =============================
resumen = final_df.groupby('Tema').size().reset_index(name='Total_Palabras')
resumen.to_csv("C:/Users/Alvaro/Downloads/Resumen Palabras Por Tema.csv", index=False)

print("✅ Listo: archivos 'Palabras Sopa de Letras Neto1.csv' y 'Resumen Palabras Por Tema.csv' generados.")