"""
Qué es el Web Scrapping
Internet está lleno de una enorme cantidad de datos que pueden ser utilizados para diferentes propósitos.Para recopilar estos datos necesitamos saber cómo hacer scraping de datos de un sitio web.

Web scraping es el proceso de extraer y recopilar datos de sitios web y almacenarlos en una máquina local o en una base de datos.

En esta sección, utilizaremos beautifulsoup y el paquete requests para scrapear datos. La versión del paquete que estamos utilizando es beautifulsoup4

Para empezar a scrapear sitios web necesitas requests, beautifoulSoup4 y un sitio web.

pip install requests
pip install beautifulsoup4

Para extraer datos de sitios web, se necesitan conocimientos básicos de etiquetas HTML y selectores CSS. Nos centramos en el contenido de un sitio web utilizando etiquetas HTML, clases y/o ids. Importemos el módulo requests y BeautifulSoup
"""
import requests
from bs4 import BeautifulSoup
url = 'https://archive.ics.uci.edu/datasets'

response = requests.get(url)
status = response.status_code
print(status) # 200

# Uso de beautifulSoup para analizar el contenido de la página web.

"""
Si ejecuta este código, puede ver que la extracción está a medio hacer. Puedes continuar haciéndolo porque es parte del ejercicio 1. Como referencia consulta la documentación de beautifulsoup
"""

import requests
from bs4 import BeautifulSoup
url = 'https://archive.ics.uci.edu/datasets'

import requests
from bs4 import BeautifulSoup

url = 'https://archive.ics.uci.edu/datasets'

response = requests.get(url)
content = response.content
soup = BeautifulSoup(content, 'html.parser')
print(soup.title.get_text())  # UCI Machine Learning Repository

tables = soup.find_all('table', {'cellpadding': '3'})
for table in tables:
    rows = table.find_all('tr')
    for row in rows:
        cols = row.find_all('td')
        for col in cols:
            print(col.text)  

"""
Este código realiza una solicitud HTTP a la página web "https://archive.ics.uci.edu/datasets" utilizando la biblioteca `requests`. Luego, utiliza la biblioteca `BeautifulSoup` para analizar el contenido HTML de la página web y extraer información de ella.

Primero, el código imprime el título de la página web utilizando el método `get_text()` del objeto `title` de `soup`.

Luego, el código utiliza el método `find_all()` de `soup` para encontrar todas las tablas en la página web que tienen un atributo `cellpadding` con un valor de `3`. Estas tablas contienen información sobre conjuntos de datos.

Para cada tabla encontrada, el código utiliza el método `find_all()` de `table` para encontrar todas las filas (`tr`) en la tabla. Luego, para cada fila encontrada, el código utiliza el método `find_all()` de `row` para encontrar todas las columnas (`td`) en la fila. Finalmente, para cada columna encontrada, el código imprime el texto de la columna utilizando el método `text` de `col`.

En resumen, este código extrae información de las tablas de conjuntos de datos en la página web "https://archive.ics.uci.edu/datasets" e imprime el texto de cada columna en la consola.
"""

# Ejercicios

"""
 Descargar una copia del set iris disponible en el UCI Machine Learning Repository en la dirección https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data. Guardar el archivo en el directorio actual con el nombre iris.data. y convertirlo a un archivo Excel (.xlsx) con el nombre iris.xlsx.
"""

import requests

# URL del archivo a descargar
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Nombre del archivo de destino
destfile = "iris.data"

# Realizar la solicitud GET para descargar el archivo
response = requests.get(url)

# Verificar si la descarga fue exitosa (código de respuesta 200)
if response.status_code == 200:
    # Guardar el contenido descargado en un archivo local
    with open(destfile, 'wb') as file:
        file.write(response.content)
    print(f"Archivo '{destfile}' descargado exitosamente.")
else:
    print(f"Error al descargar el archivo. Código de respuesta: {response.status_code}")
print(destfile)

import pandas as pd

# Cargar los datos desde el archivo iris.data (o tu archivo de datos)
data = pd.read_csv("iris.data", header=None, names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])

# Escribir los datos en un archivo Excel (.xlsx)
data.to_excel("iris.xlsx", index=False)

print("Datos convertidos y guardados en iris.xlsx")

# Sin embargo es mucho mas sencillo descargar "pip install ucimlrepo" y utilizarlo para descargar los datos de UCI Machine Learning Repository. El codigo mas simplificado nos quedaria asi:

import pandas as pd
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
iris = fetch_ucirepo(id=53) 
  
# data (as pandas dataframes) 
X = iris.data.features 
y = iris.data.targets 
  
# Convertir los datos a un DataFrame de pandas
df = pd.DataFrame(X, columns=iris.data.features_names)
df['target'] = y

# Escribir los datos en un archivo Excel (.xlsx)
df.to_excel('iris.xlsx', sheet_name='iris', index=False)

print("Datos convertidos y guardados en iris.xlsx") # Datos convertidos y guardados en iris.xlsx

"""
 Descargar una copia del set Heart Disease disponible en el UCI Machine Learning Repository en la dirección http://archive.ics.uci.edu/dataset/45/heart+disease. Guardar el archivo en el directorio actual con el nombre heart_desease.data. y convertirlo a un archivo Excel (.xlsx) con el nombre heart_desease.xlsx. Utilice el paquete ucimlrepo
"""

import pandas as pd
from ucimlrepo import fetch_ucirepo

# Importar conjunto de datos
heart_disease = fetch_ucirepo(id=45)

# Acceder a los datos
X = heart_disease.data.features
y = heart_disease.data.targets

# Convertir los datos a un DataFrame de pandas
df = pd.DataFrame(X, columns=heart_disease.data.feature_names)
df['target'] = y

# Escribir los datos en un archivo Excel (.xlsx)
df.to_excel('heart_disease.xlsx', sheet_name='heart_disease', index=False)

print("Datos convertidos y guardados en heart_disease.xlsx") # Datos convertidos y guardados en heart_disease.xlsx


"""
Rastree el siguiente sitio web y almacene los datos como archivo json(url = 'http://www.bu.edu/president/boston-university-facts-stats/').
"""

import requests
from bs4 import BeautifulSoup
import json

# URL del sitio web a rastrear
url = 'http://www.bu.edu/president/boston-university-facts-stats/'

# Realizar una solicitud GET para obtener el contenido del sitio web
response = requests.get(url)

# Verificar si la solicitud fue exitosa
if response.status_code == 200:
    # Parsear el contenido HTML usando BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Aquí debes escribir el código para extraer los datos que deseas del sitio web
    # Supongamos que extraemos datos ficticios para este ejemplo
    data = {
        "titulo": soup.title.string,
        "parrafos": [p.get_text() for p in soup.find_all('p')]
    }
    
    # Nombre del archivo JSON de destino
    output_file = 'web_data.json'

    # Guardar los datos en un archivo JSON
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)
    
    print(f'Datos del sitio web guardados en "{output_file}"')  # Datos del sitio web guardados en "web_data.json"
else:
    print(f'Error al obtener el contenido del sitio web. Código de respuesta: {response.status_code}')
    
# Otra forma:

import requests
from bs4 import BeautifulSoup
import json

# URL del sitio web a rastrear
url = 'http://www.bu.edu/president/boston-university-facts-stats/'

# Realizar una solicitud GET para obtener el contenido del sitio web
response = requests.get(url)

# Verificar si la solicitud fue exitosa
if response.status_code == 200:
    # Parsear el contenido HTML usando BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Aquí debes escribir el código para extraer los datos que deseas del sitio web
    # Supongamos que extraemos datos ficticios para este ejemplo
    data = {
        "titulo": soup.title.string,
        "parrafos": soup.get_text().split('\n\n')
    }
    
    # Nombre del archivo JSON de destino
    output_file = 'web_data.json'

    # Guardar los datos en un archivo JSON
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)
    
    print(f'Datos del sitio web guardados en "{output_file}"')  # Datos del sitio web guardados en "web_data.json"
else:
    print(f'Error al obtener el contenido del sitio web. Código de respuesta: {response.status_code}')
    
"""
Extrae la tabla de esta url (https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data) y cámbiala a un archivo json
"""

import requests
import pandas as pd
import json

# URL del archivo de datos
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Realizar una solicitud GET para obtener el contenido del archivo
response = requests.get(url)

# Verificar si la solicitud fue exitosa (código de respuesta 200)
if response.status_code == 200:
    # Leer los datos de la tabla usando pandas
    # Suponemos que los datos son de tipo CSV con comas como separadores
    df = pd.read_csv(url, header=None, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
    
    # Convertir el DataFrame a un diccionario
    data_dict = df.to_dict(orient='records')
    
    # Nombre del archivo JSON de destino
    output_file = 'iris_data.json'

    # Guardar los datos en un archivo JSON
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(data_dict, json_file, ensure_ascii=False, indent=4)
    
    print(f'Datos de la tabla guardados en "{output_file}" como JSON.')  # Datos de la tabla guardados en "iris_data.json" como JSON.
else:
    print(f'Error al obtener el contenido del archivo. Código de respuesta: {response.status_code}')
    
# Otra forma

import pandas as pd
import json

# URL del archivo de datos
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Leer los datos de la tabla usando pandas
# Suponemos que los datos son de tipo CSV con comas como separadores
df = pd.read_csv(url, header=None, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])

# Convertir el DataFrame a un diccionario
data_dict = df.to_dict(orient='records')

# Nombre del archivo JSON de destino
output_file = 'iris_data.json'

# Guardar los datos en un archivo JSON
with open(output_file, 'w', encoding='utf-8') as json_file:
    json.dump(data_dict, json_file, ensure_ascii=False, indent=4)

print(f'Datos de la tabla guardados en "{output_file}" como JSON.')  # Datos de la tabla guardados en "iris_data.json" como JSON.

# Otra forma

import pandas as pd
from ucimlrepo import fetch_ucirepo 
import json

# fetch dataset 
iris = fetch_ucirepo(id=53) 

# data (as pandas dataframes) 
X = iris.data.features 
y = iris.data.targets 

# Convertir los datos a un diccionario
data_dict = {
    "features": X.to_numpy().tolist(),
    "targets": y.to_numpy().tolist()
}

# Escribir los datos en un archivo JSON
with open('iris_data.json', 'w') as f:
    json.dump(data_dict, f)

print("Datos convertidos y guardados en iris_data.json")  # Datos convertidos y guardados en iris_data.json

"""
Explore la tabla de presidentes de la direcion URL 'https://en.wikipedia.org/wiki/List_of_presidents_of_the_United_States'  y almacene los datos como un archivo json(). 
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
from io import StringIO

# URL de la página de Wikipedia
url = 'https://en.wikipedia.org/wiki/List_of_presidents_of_the_United_States'

# Realizar una solicitud GET para obtener el contenido de la página
response = requests.get(url)

# Verificar si la solicitud fue exitosa
if response.status_code == 200:
    # Utilizar BeautifulSoup para analizar el contenido HTML
    soup = BeautifulSoup(response.text, 'html.parser')

    # Encontrar la tabla que contiene los datos de los presidentes
    table = soup.find('table', {'class': 'wikitable'})

    # Leer la tabla en un DataFrame de pandas
    html_string = str(table)
    df = pd.read_html(StringIO(html_string))[0]

    # Convertir el DataFrame a una lista de diccionarios
    president_data = df.to_dict(orient='records')

    # Nombre del archivo JSON de destino
    output_file = 'presidents.json'

    # Guardar los datos en un archivo JSON
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(president_data, json_file, ensure_ascii=False, indent=4)

    print(f'Datos de la tabla de presidentes guardados en {output_file} como JSON.')  # Datos de la tabla de presidentes guardados en presidents.json como JSON.
else:
    print(f'Error al obtener la página. Código de respuesta: {response.status_code}')

"""
Explore la tabla de presidentes de la direcion URL 'https://en.wikipedia.org/wiki/List_of_presidents_of_the_United_States'  y almacene los datos en un archivo excel xlsx
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from io import StringIO

# URL de la página de Wikipedia
url = 'https://en.wikipedia.org/wiki/List_of_presidents_of_the_United_States'

# Realizar una solicitud GET para obtener el contenido de la página
response = requests.get(url)

# Verificar si la solicitud fue exitosa
if response.status_code == 200:
    # Utilizar BeautifulSoup para analizar el contenido HTML
    soup = BeautifulSoup(response.text, 'html.parser')

    # Encontrar la tabla que contiene los datos de los presidentes
    table = soup.find('table', {'class': 'wikitable'})

    # Leer la tabla en un DataFrame de pandas
    html_string = str(table)
    df = pd.read_html(StringIO(html_string))[0]

    # Nombre de la hoja de trabajo de destino
    sheet_name = 'presidents'

    # Crear un libro de trabajo de Excel
    wb = Workbook()

    # Seleccionar la hoja de trabajo activa
    ws = wb.active

    # Escribir los datos en la hoja de trabajo
    for r in dataframe_to_rows(df, index=False, header=True):
        ws.append(r)

    # Guardar el libro de trabajo de Excel
    output_file = 'presidents.xlsx'
    wb.save(output_file)

    print(f'Datos de la tabla de presidentes guardados en {output_file} como Excel.') # Datos de la tabla de presidentes guardados en presidents.xlsx como Excel.
else:
    print(f'Error al obtener la página. Código de respuesta: {response.status_code}')
    
# Otra forma

import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO

# URL de la página de Wikipedia
url = 'https://en.wikipedia.org/wiki/List_of_presidents_of_the_United_States'

# Realizar una solicitud GET para obtener el contenido de la página
response = requests.get(url)

# Verificar si la solicitud fue exitosa
if response.status_code == 200:
    # Utilizar BeautifulSoup para analizar el contenido HTML
    soup = BeautifulSoup(response.text, 'html.parser')

    # Encontrar la tabla que contiene los datos de los presidentes
    table = soup.find('table', {'class': 'wikitable'})

    # Leer la tabla en un DataFrame de pandas
    html_string = str(table)
    df = pd.read_html(StringIO(html_string))[0]

    # Nombre del archivo Excel de destino
    output_file = 'presidents.xlsx'

    # Guardar los datos en un archivo Excel (.xlsx)
    df.to_excel(output_file, index=False, engine='openpyxl')

    print(f'Datos de la tabla de presidentes guardados en {output_file} como Excel (.xlsx).') # Datos de la tabla de presidentes guardados en presidents.xlsx como Excel (.xlsx).
else:
    print(f'Error al obtener la página. Código de respuesta: {response.status_code}')