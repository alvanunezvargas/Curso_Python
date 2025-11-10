"""
Python PIP - Gestor de paquetes de Python
¿Qué es PIP?
PIP son las siglas de Preferred installer program. Usamos pip para instalar diferentes paquetes de Python. Un paquete es un módulo de Python que puede contener uno o más módulos u otros paquetes. Un módulo o módulos que podemos instalar en nuestra aplicación es un paquete. En programación, no tenemos que escribir cada programa de utilidad, en su lugar instalamos paquetes y los importamos a nuestras aplicaciones.

Instalando PIP
Vamos a terminal o símbolo del sistema y copie y pegue lo siguiente:
"""
# pip install pip

# Chequee la version de pip:

# pip --version

"""
Instalando paquetes usando pip:

Instalemos numpy, llamado numeric python. Es uno de los paquetes más populares en la comunidad de aprendizaje automático y ciencia de datos.

NumPy es el paquete fundamental para la computación científica con Python. Contiene entre otras cosas
-  un potente objeto array N-dimensional
-  sofisticadas funciones (de emisión)
-  herramientas para integrar código C/C++ y Fortran
-  funciones utiles de álgebra lineal, transformada de Fourier y números aleatorios
"""
# Empecemos usando numpy. Abra su shell interactivo python, escriba python y luego importar numpy de la siguiente manera:

import numpy

lst = [1, 2, 3, 4, 5]
np_arr = numpy.array(lst)
print(np_arr)  # [1 2 3 4 5]
print(len(np_arr))  # 5
print(np_arr * 2)  # [ 2  4  6  8 10]
print(np_arr + 2)  # [3 4 5 6 7]

# Pandas es una biblioteca de código abierto con licencia BSD que proporciona estructuras de datos y herramientas de análisis de datos de alto rendimiento y fáciles de usar para el lenguaje de programación Python. Intalemosla desde la terminal.

# pip install pandas

"""
Vamos a importar un módulo de navegador web, que nos puede ayudar a abrir cualquier sitio web. No necesitamos instalar este módulo, ya viene instalado por defecto con Python 3. Por ejemplo, si nos gusta abrir cualquier número de sitios web en cualquier momento o si quiere programar algo, este módulo webbrowser se puede utilizar.
"""

import webbrowser # (Módulo de navegador web para abrir sitios web)

# list of urls: python
url_lists = [
    'http://www.python.org',
    'https://www.linkedin.com/in/asabeneh/',
    'https://github.com/Asabeneh',
    'https://twitter.com/Asabeneh',
]

# (Abre la lista de sitios web anterior en otra pestaña)
for url in url_lists:
    webbrowser.open_new_tab(url)

"""
Desinstalación de paquetes: Si no desea conservar los paquetes instalados, puede eliminarlos mediante el siguiente comando.

pip uninstall nombrepaquete

Lista de paquetes: Para ver los paquetes instalados en nuestra máquina. Podemos usar pip seguido de list. Veamos el comando.

pip list

Mostrar informacion de un paquete: Para mostrar información sobre un paquete. Veamos el comando.

pip show nombrepaquete

Si queremos aún más detalles, basta con añadir --verbose. Veamos el comando.

pip show --verbose pandas

PIP Freeze: Genera paquetes Python instalados con su versión y la salida es adecuada para usarla en un fichero de requisitos. Un archivo requirements.txt es un archivo que debe contener todos los paquetes Python instalados en un proyecto Python. Veamos el comando.

pip freeze

Lectura desde URL
Por ahora estás familiarizado con cómo leer o escribir en un archivo ubicado en tu máquina local. A veces, nos gustaría leer desde un sitio web usando una url o desde una API. API significa Application Program Interface. Es un medio para intercambiar datos estructurados entre servidores, principalmente como datos json. Para abrir una conexión de red, necesitamos un paquete llamado requests - permite abrir una conexión de red e implementar operaciones CRUD (crear, leer, actualizar y borrar). En esta sección, sólo cubriremos la parte de lectura u obtención de un CRUD. Verifique en el terminal que tiene instalado request con el comando pip list

Veremos los métodos get, status_code, headers, text y json en el módulo requests:

- get(): para abrir una red y obtener datos de la url - devuelve un objeto de respuesta
- status_code: Después de obtener los datos, podemos comprobar el estado de la operación (éxito, error, etc)
- headers: Para comprobar los tipos de cabecera
- text: para extraer el texto del objeto de respuesta obtenido
- json: para extraer datos json Leamos un archivo txt de este sitio web, https://www.w3.org/TR/PNG/iso_8859-1.txt.
"""
import requests # (Importación del módulo requests)

url = 'https://www.w3.org/TR/PNG/iso_8859-1.txt' # (Dirección URL del archivo de texto que se desea obtener)

response = requests.get(url) # (Realizar una solicitud HTTP GET a la URL especificada)
print(response)  # (Imprime el objeto de respuesta completo)
print(response.status_code) # (Imprime el código de estado de la respuesta (por ejemplo, 200 para una respuesta exitosa o 404 para una respuesta no encontrada)
print(response.headers) # (Imprime las cabeceras de la respuesta)
print(response.text) # (Se imprime el contenido del archivo de texto)

"""
Leamos desde una API. API significa Application Program Interface. Es un medio para intercambiar datos de estructura entre servidores, principalmente datos json. Un ejemplo de una API:https://raw.githubusercontent.com/Asabeneh/30-Days-Of-Python/master/data/countries.py. Leamos esta API utilizando el módulo requests.
"""

import requests

def download_data(url, output_file):
    try:
        # Realizar una solicitud GET para obtener los datos de la URL
        response = requests.get(url)
        response.raise_for_status()
        
        # Escribir los datos en un archivo de texto
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(response.text)
        
        print(f"Los datos de la URL se han convertido y guardado en el archivo '{output_file}'.")
    except requests.exceptions.HTTPError as e:
        print(f"La solicitud a la URL no fue exitosa: {str(e)}")
    except Exception as e:
        print(f"Se produjo un error: {str(e)}")

# URL de la fuente de datos
url = 'https://raw.githubusercontent.com/Asabeneh/30-Days-Of-Python/master/data/countries.py'

# Nombre del archivo de destino
output_file1 = 'C:/Users/alvan/Downloads/countries.txt'

# Descargar los datos de la URL y guardarlos en un archivo de texto
download_data(url, output_file1)
# Los datos de la URL se han convertido y guardado en el archivo 'C:/Users/alvan/Downloads/countries.txt'.

import json

countries = open(output_file1)
txt = countries.read()
output_file2 = 'C:/Users/alvan/Downloads/countries.json'
with open(output_file2, 'w') as json_file:
    json.dump(txt, json_file)
print(f"La lista de países se ha convertido y guardado en el archivo ({output_file2})")
# La lista de países se ha convertido y guardado en el archivo (C:/Users/alvan/Downloads/countries.json)

# Abrir el archivo JSON y cargar su contenido en una variable
with open('C:/Users/alvan/Downloads/countries.json', 'r') as json_file:
    countries = json.load(json_file)

print(countries[:20])
print(countries)

# Lo mismo podemos hacerlo con este codigo:

import requests
import json

def download_data(url, output_file):
    try:
        # Realizar una solicitud GET para obtener los datos de la URL
        response = requests.get(url)
        response.raise_for_status()
        
        # Escribir los datos en un archivo de texto
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(response.text)
        
        print(f"Los datos de la URL se han convertido y guardado en el archivo '{output_file}'.")
    except requests.exceptions.HTTPError as e:
        print(f"La solicitud a la URL no fue exitosa: {str(e)}")
    except Exception as e:
        print(f"Se produjo un error: {str(e)}")

# URL de la fuente de datos
url = 'https://raw.githubusercontent.com/Asabeneh/30-Days-Of-Python/master/data/countries.py'

# Nombre del archivo de destino
output_file1 = 'C:/Users/alvan/Downloads/countries.txt'
output_file2 = 'C:/Users/alvan/Downloads/countries.json'

# Descargar los datos de la URL y guardarlos en un archivo de texto
download_data(url, output_file1)

# Convertir el archivo de texto en un archivo JSON
with open(output_file2, 'r', encoding='utf-8') as file:
    data = file.read()
    countries = json.loads(data)

with open(output_file2, 'w', encoding='utf-8') as file:
    json.dump(countries, file)

print(f"La lista de países se ha convertido y guardado en el archivo '{output_file2}'.")
print(countries[:20])
print(countries)

# Imprime el código de estado de la respuesta, 200 para una respuesta exitosa o 404 para una respuesta no encontrada

import requests
url = 'https://raw.githubusercontent.com/Asabeneh/30-Days-Of-Python/master/data/countries.py'
response = requests.get(url)  
print(response.status_code)  # 200

"""
Crear un paquete
Organizamos un gran número de archivos en diferentes carpetas y subcarpetas basándonos en algunos criterios, de forma que podamos encontrarlos y gestionarlos fácilmente. Un módulo puede contener múltiples objetos, como clases, funciones, etc. Un paquete puede contener uno o más módulos relevantes. Un paquete es en realidad una carpeta que contiene uno o más archivos de módulo. Creemos un paquete llamado mypackage, siguiendo los siguientes pasos:

Crear una nueva carpeta llamada mypacakge dentro de la carpeta 30DaysOfPython Crear un archivo init.py vacío en la carpeta mypackage. Crear los módulos arithmetic.py y greet.py con el siguiente código:
"""

# mypackage/arithmetics.py
# arithmetics.py
def add_numbers(*args):
    total = 0
    for num in args:
        total += num
    return total


def subtract(a, b):
    return (a - b)


def multiple(a, b):
    return a * b


def division(a, b):
    return a / b


def remainder(a, b):
    return a % b


def power(a, b):
    return a ** b
print(add_numbers(1, 2, 3, 4, 5))

# Y

# mypackage/greet.py
# greet.py
def greet_person(firstname, lastname):
    return f'{firstname} {lastname}, welcome to 30DaysOfPython Challenge!'

"""
El archivo __init__.py en Python es un archivo especial que se utiliza para indicar que un directorio debe ser considerado como un paquete o un módulo de Python. Este archivo es opcional en la mayoría de los casos, pero es necesario cuando se desea crear un paquete de Python.

Aquí hay algunas cosas importantes que debes saber sobre el archivo __init__.py:

Marca de Paquete: Cuando Python encuentra un archivo __init__.py dentro de un directorio, trata ese directorio como un paquete o un subpaquete de un paquete más grande. Esto permite una organización jerárquica de módulos en paquetes.

Inicialización: Aunque el nombre sugiere que su propósito es la inicialización, el archivo __init__.py no se utiliza para inicializar el paquete en sí. En cambio, se usa principalmente para configuraciones iniciales, importaciones y para exponer variables, funciones o clases que deben estar disponibles cuando se importa el paquete.

Estructura de Paquetes: Los paquetes pueden contener otros paquetes y módulos. Esto permite una estructura de organización más limpia para el código. Por ejemplo, puedes tener una estructura como esta:

mypackage/
├── __init__.py
├── arimethic.py
├── geet.py

Importaciones Relativas: El archivo __init__.py puede contener importaciones que son relevantes para el paquete o el subpaquete en sí. Estas importaciones pueden ser importaciones absolutas o importaciones relativas utilizando el punto (.) para referirse a módulos dentro del mismo paquete.

Python 3.3 y Posteriores: A partir de Python 3.3, los archivos __init__.py ya no son estrictamente necesarios para considerar un directorio como un paquete. Sin embargo, todavía son una buena práctica y se utilizan ampliamente en la comunidad de Python para mantener una estructura organizada y para proporcionar una API limpia y clara para los módulos y paquetes.

En resumen, el archivo __init__.py en Python es un componente importante para organizar tu código en paquetes y subpaquetes, y puede contener configuraciones, importaciones y definiciones que son relevantes para el paquete en el que se encuentra.


Más información sobre los paquetes

Base de datos
SQLAlchemy o SQLObject - Acceso orientado a objetos a diferentes sistemas de bases de datos
pip install SQLAlchemy

Desarrollo web
Django - Framework web de alto nivel.
pip install django
Flask - Micro framework para Python basado en Werkzeug, Jinja 2. (Tiene licencia BSD)
pip install flask

Parser HTML
Beautiful Soup - Parser HTML/XML diseñado para proyectos rápidos como screen-scraping, acepta mal marcado.
pip install beautifulsoup4
PyQuery - implementa jQuery en Python; aparentemente más rápido que BeautifulSoup.

Procesamiento XML
ElementTree - El tipo Element es un objeto contenedor simple pero flexible, diseñado para almacenar estructuras de datos jerárquicas, como infosets XML simplificados, en memoria. --Nota: Python 2.5 y superiores tienen ElementTree en la Librería Estándar.

GUI
PyQt - Enlaces para el framework multiplataforma Qt.
TkInter - El tradicional conjunto de herramientas de interfaz de usuario de Python.

Análisis de datos, ciencia de datos y aprendizaje automático

Numpy: Numpy(numeric python) es conocida como una de las librerías más populares de aprendizaje automático en Python.
Pandas: es una librería de análisis de datos, ciencia de datos y aprendizaje automático en Python que proporciona estructuras de datos de alto nivel y una amplia variedad de herramientas para el análisis.
SciPy: SciPy es una biblioteca de aprendizaje automático para desarrolladores de aplicaciones e ingenieros. La biblioteca SciPy contiene módulos de optimización, álgebra lineal, integración, procesamiento de imágenes y estadística.
Scikit-Learn: Es NumPy y SciPy. Está considerada como una de las mejores librerías para trabajar con datos complejos.
TensorFlow: es una librería de machine learning construida por Google.
Keras: está considerada como una de las mejores librerías de aprendizaje automático en Python. Proporciona un mecanismo más sencillo para expresar redes neuronales. Keras también proporciona algunas de las mejores utilidades para compilar modelos, procesar conjuntos de datos, visualización de gráficos y mucho más.

Network:

requests: es un paquete que podemos utilizar para enviar peticiones a un servidor(GET, POST, DELETE, PUT)
pip install peticiones
"""

# Ejercicios

# Lee esta url y encuentra las 10 palabras más frecuentes. 'http://www.gutenberg.org/files/1112/1112.txt'

import requests
from collections import Counter
import re

# Descargar el contenido del texto desde la URL
url = 'http://www.gutenberg.org/files/1112/1112.txt'
response = requests.get(url)

# Verificar si la solicitud fue exitosa
if response.status_code == 200:
    # Obtener el contenido del texto
    text = response.text

    # Limpiar el texto de caracteres especiales y dividirlo en palabras
    words = re.findall(r'\b\w+\b', text.lower())

    # Calcular las 10 palabras más frecuentes
    word_counts = Counter(words)
    most_common_words = word_counts.most_common(10)

    # Imprimir las 10 palabras más frecuentes
    print("Las 10 palabras más frecuentes en el texto son:")
    for word, count in most_common_words:
        print(f"{word}: {count}")

else:
    print("No se pudo obtener el contenido del texto desde la URL.")
"""
Las 10 palabras más frecuentes en el texto son:
the: 868
and: 800
to: 661
i: 658
of: 535
a: 530
is: 381
in: 378
that: 371
you: 367
"""
# Otra froma:

import requests
from collections import Counter
import re

def get_most_common_words(url):
    try:
        # Realizar una solicitud GET para obtener el contenido del texto
        response = requests.get(url)
        
        # Verificar si la solicitud fue exitosa
        if response.ok:
            # Obtener el contenido del texto
            text = response.text

            # Limpiar el texto de caracteres especiales y dividirlo en palabras
            words = re.findall(r'\b\w+\b', text.lower())

            # Calcular las 10 palabras más frecuentes
            word_counts = Counter(words)
            most_common_words = word_counts.most_common(10)

            # Imprimir las 10 palabras más frecuentes
            print("Las 10 palabras más frecuentes en el texto son:")
            for word, count in most_common_words:
                print(f"{word}: {count}")
        else:
            print("No se pudo obtener el contenido del texto desde la URL.")
    except requests.exceptions.RequestException as e:
        print(f"Se produjo un error al realizar la solicitud: {str(e)}")

# URL del texto
url = 'http://www.gutenberg.org/files/1112/1112.txt'

# Obtener las 10 palabras más frecuentes en el texto
get_most_common_words(url)
"""
Las 10 palabras más frecuentes en el texto son:
the: 868
and: 800
to: 661
i: 658
of: 535
a: 530
is: 381
in: 378
that: 371
you: 367
"""

"""
Lea la API de gatos en 'https://api.thecatapi.com/v1/breeds' y encuentre :
1-  la desviación mínima, máxima, media, mediana y estándar del peso de los gatos en unidades métricas.
2-  la desviación mínima, máxima, media, mediana y estándar de la esperanza de vida de los gatos en años.
3-  Crear una tabla de frecuencias del país y la raza de los gatos
"""

import requests
import pandas as pd

# Realizar una solicitud a la API para obtener datos de gatos
url = 'https://api.thecatapi.com/v1/breeds'
response = requests.get(url)
response.raise_for_status()

# Crear un DataFrame de pandas a partir de los datos JSON
df = pd.DataFrame.from_records(response.json())

# 1. Estadísticas del peso de los gatos en unidades métricas
weight_metrics = df['weight'].apply(lambda x: x['metric']).str.extract(r'(\d+)').astype(float)
weight_stats = weight_metrics.describe()

print("Estadísticas del peso de los gatos en unidades métricas:")
print(weight_stats)

# 2. Estadísticas de la esperanza de vida de los gatos en años
life_span_metrics = df['life_span'].str.extract(r'(\d+)').astype(float)
life_span_stats = life_span_metrics.describe()

print("\nEstadísticas de la esperanza de vida de los gatos en años:")
print(life_span_stats)

# 3. Tabla de frecuencias del país y la raza de los gatos
country_breed_counts = pd.crosstab(df['origin'], df['name'])

print("\nTabla de frecuencias del país y la raza de los gatos:")
print(country_breed_counts)

"""
De la siguiente direccion URL "https://raw.githubusercontent.com/Asabeneh/30-Days-Of-Python/master/data/countries-data.py"
1- Encuentra los 10 idiomas más hablados
2- Encuentra los 10 países más poblados del mundo
"""
# Los 10 idiomas más hablados

import requests
from collections import Counter

# Realizar una solicitud a la API para obtener datos de países
url = "https://raw.githubusercontent.com/Asabeneh/30-Days-Of-Python/master/data/countries-data.py"
response = requests.get(url)
response.raise_for_status()

# Convertir los datos JSON a una lista de diccionarios
data = eval(response.content)

# Contar el número de lenguas en todos los países
languages_count = Counter(language for country in data for language in country['languages'])

# Ordenar el conteo de lenguas y obtener las diez lenguas más habladas
sorted_languages = languages_count.most_common(10)

# Imprimir las diez lenguas más habladas
print("Las diez lenguas más habladas en el mundo son:")
for language, count in sorted_languages:
    print(f"{language} ({count} países)")

"""
Las diez lenguas más habladas en el mundo son:
English (91 países)
French (45 países)
Arabic (25 países)
Spanish (24 países)
Portuguese (9 países)
Russian (9 países)
Dutch (8 países)
German (7 países)
Chinese (5 países)
Serbian (4 países)
"""

# Los 10 países más poblados del mundo

import requests

# Realizar una solicitud a la API para obtener datos de países
url = 'https://raw.githubusercontent.com/Asabeneh/30-Days-Of-Python/master/data/countries-data.py'
response = requests.get(url)
response.raise_for_status()

# Convertir los datos JSON a una lista de diccionarios
data = response.json()

# Crear una lista de tuplas de cada país y su población
populations = [(country['name'], country['population']) for country in data]

# Ordenar la lista de tuplas por la población en orden descendente
sorted_populations = sorted(populations, key=lambda x: x[1], reverse=True)

# Imprimir los primeros 10 países de la lista ordenada
print('Los 10 países más poblados del mundo son:')
for i, (name, population) in enumerate(sorted_populations[:10]):
    print(f'{i+1}. {name} - {population}')
"""
Los 10 países más poblados del mundo son:
1. China - 1377422166
2. India - 1295210000
3. United States of America - 323947000
4. Indonesia - 258705000
5. Brazil - 206135893
6. Pakistan - 194125062
7. Nigeria - 186988000
8. Bangladesh - 161006790
9. Russian Federation - 146599183
10. Japan - 126960000
"""