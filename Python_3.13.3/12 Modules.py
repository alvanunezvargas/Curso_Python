# Qué es un módulo: Un módulo es un archivo que contiene un conjunto de códigos o un conjunto de funciones que pueden incluirse en una aplicación.
# Un módulo puede ser un fichero que contenga una única variable, una función o una gran base de código.

# Crear un módulo: Para crear un módulo escribimos nuestros códigos en un script de python y lo guardamos como un archivo .py. Crea un archivo
# llamado mymodule.py dentro de la carpeta de tu proyecto. Escribamos algo de código en este archivo.

# En una nueva carpeta generamos el archivo mymodule.py
# dentro de la misma carpeta, generamos el archivo main.py
# En el archivo mymodule.py generamos el siguiente codigo:

def generate_full_name(firstname, lastname):
    return firstname + ' ' + lastname
def sum_two_nums(num1, num2):
    return num1 + num2
gravity = 9.8
person = {
    'firstname': 'Alvaro',
    'lastname': 'Nunez',
    'age': 55,
}

# En el archivo main.py generamos el siguiente codigo:

from mymodule import generate_full_name, sum_two_nums, person, gravity
print(generate_full_name('Alvaro','Nunez'))  # Alvaro Nunez
print(sum_two_nums(1,9))   # 10
mass = 100;
weight = mass * gravity
print(weight)                   # 980.0000000000001
print(person['firstname'])     # Alvaro

# Nota, debemos correr (run) el codigo  que escribimos en el archivo mymodule.py, antes de correr el codigo que escribimos en el archivo main.py
# sino cuando corramos este ultimo, nos dara un error.

# Importar funciones de un módulo y cambiar el nombre: Durante la importación podemos renombrar el nombre del módulo.
# En el archivo main.py generamos el siguiente codigo:

from mymodule import generate_full_name as fullname, sum_two_nums as total, person as p, gravity as g
print(fullname('Alvaro','Nunez'))  # Alvaro Nunez
print(total(1, 9))   # 10
mass = 100;
weight = mass * g
print(weight)       # 980.0000000000001"
print(p)            # {'firstname': 'Alvaro', 'lastname': 'Nunez', 'age': 55}
print(p['firstname'])  # Alvaro

# Importar módulos incorporados: Al igual que otros lenguajes de programación también podemos importar módulos importando el archivo/función usando
# la palabra clave "import". Vamos a importar el módulo común que utilizaremos la mayor parte del tiempo. Algunos de los módulos comunes incorporados:
# math, datetime, os,sys, random, statistics, collections, json,re

# Módulo OS: Usando el módulo OS de Python es posible realizar automáticamente muchas tareas del sistema operativo. El módulo OS de Python proporciona
# funciones para crear, cambiar el directorio de trabajo actual y eliminar un directorio (carpeta), obtener su contenido, cambiar e identificar el
# directorio actual.

import os
os.mkdir('nombre_archivo')  # (Crear archivo 'nombre_archivo')
os.rmdir('nombre_archivo')    # (Eliminar archivo 'nombre_archivo' )
os.getcwd()    # (Obtener la ruta de directorio de trabajo actual)
os.chdir('ruta')    # (Cambiar la ruta de la carpeta actual por 'ruta')

# Módulo Sys: El módulo sys proporciona funciones y variables utilizadas para manipular diferentes partes del entorno de ejecución de Python. La función
# sys.argv devuelve una lista de argumentos de línea de comandos pasados a un script de Python. El elemento en el índice 0 de esta lista es siempre el
# nombre del script, en el índice 1 está el argumento pasado desde la línea de comandos.

# Ejemplo de un archivo script.py:

# Crea el siguiente codigo en el archivo Prueba.py:

import sys
print('Welcome {}. Enjoy  {} challenge!'.format(sys.argv[1], sys.argv[2]))  # (Esta línea imprimiría: filename argument1 argument2)

# Luego vaya al terminal o linea de comando y escriba los siguiente, "py Prueba.py Alvaro 30DaysOfPython" luego de enter.  
# se imprimira "Welcome Alvaro. Enjoy  30DaysOfPython challenge!"

# Otros comandos utiles de sys son:

# sys.exit(): Para salir de sys

import sys
def divide(a, b):
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        print("Error: No es posible dividir entre cero.")  # Error: No es posible dividir entre cero.
        sys.exit()  # Termina la ejecución del programa
num1 = 10
num2 = 0
resultado = divide(num1, num2)
print("El resultado de la división es:", resultado)

# cuando el dividendo no sea cero:

import sys
def divide(a, b):
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        print("Error: No es posible dividir entre cero.")  
        sys.exit()  
num1 = 10
num2 = 2
resultado = divide(num1, num2)
print("El resultado de la división es:", resultado)  # El resultado de la división es: 5.0

# sys.maxsize: Devuelve el valor máximo permitido para un entero en la plataforma actual. Este valor depende de si estás utilizando una arquitectura
# de 32 bits o 64 bits.

import sys
max_integer = sys.maxsize
print("El valor máximo para un entero es:", max_integer)  # El valor máximo para un entero es: 9223372036854775807
# Este valor corresponde al valor máximo para un entero de 64 bits en la plataforma actual

# sys.path:  Devuelve una lista de cadenas que representan las rutas de búsqueda del intérprete de Python. Estas rutas se utilizan para buscar
# módulos y paquetes durante la ejecución de un programa.

import sys
rutas = sys.path
print("Rutas de búsqueda del intérprete de Python:")
for ruta in rutas:
    print(ruta)
# Rutas de búsqueda del intérprete de Python:
# c:\Users\alvan\OneDrive\Documentos\Python\Curso Python
# C:\Users\alvan\AppData\Local\Programs\Python\Python311\python311.zip
# C:\Users\alvan\AppData\Local\Programs\Python\Python311\DLLs
# C:\Users\alvan\AppData\Local\Programs\Python\Python311\Lib
# C:\Users\alvan\AppData\Local\Programs\Python\Python311
# C:\Users\alvan\AppData\Local\Programs\Python\Python311\Lib\site-packages

# sys.version:  Devuelve una cadena que representa la versión del intérprete de Python que se está utilizando. 

import sys
print(sys.version)  # 3.11.3 (tags/v3.11.3:f3909b8, Apr  4 2023, 23:49:59) [MSC v.1934 64 bit (AMD64)]

# Módulo Estadística: El módulo de estadística proporciona funciones para la estadística matemática de datos numéricos. Las funciones estadísticas
# más populares que se definen en este módulo son: media, mediana, moda, stdev, etc.

import statistics  # (Este comando importa el módulo statistics completo en el espacio de nombres actual, pero no importa directamente los elementos individuales del módulo. )
ages = [20, 20, 4, 24, 25, 22, 26, 20, 23, 22, 26]
mean = statistics.mean(ages)
median = statistics.median(ages)
mode = statistics.mode(ages)
stdev = statistics.stdev(ages)
print('la media es: ', mean)      # la media es:  21.09090909090909
print('la mediana es: ', median)  # la mediana es:  22
print('la media es: ', mean)      # la media es:  21.09090909090909
print('la moda es: ', mode)       # la moda es:  20
print('la desviacion estandar es: ', stdev)  # la desviacion estandar es:  6.106628291529549

#Otra forma

from statistics import * # (Este comando importa todos los elementos (funciones, clases, variables) del módulo statistics )
ages = [20, 20, 4, 24, 25, 22, 26, 20, 23, 22, 26]
print(mean(ages))       # 21.09090909090909
print(median(ages))     # 22
print(mode(ages))       # 20
print(stdev(ages))      # 6.106628291529549

# Módulo Matemáticas: Módulo que contiene muchas operaciones matemáticas y constantes.

import math
print(math.pi)           # 3.141592653589793 (constante pi)
print(math.sqrt(2))      # 1.4142135623730951 (raíz cuadrada)
print(math.pow(2, 3))    # 8.0 (función exponencial)
print(math.floor(9.81))  # 9 (redondeo al mínimo)
print(math.ceil(9.81))   # 10 (redondeando al máximo)
print(math.log10(100))   # 2.0 (logaritmo con base 10)

# Si queremos importar todas las funciones del módulo math podemos usar * .

from math import *
print(pi)                  # 3.141592653589793 (constante pi)
print(sqrt(2))             # 1.4142135623730951 (raíz cuadrada)
print(pow(2, 3))           # 8.0 (función exponencial)
print(floor(9.81))         # 9  (redondeo al mínimo)
print(ceil(9.81))          # 10 (redondeando al máximo)
print(log10(100))          # 2.0 (logaritmo con base 10)

# Ahora, hemos importado el módulo math que contiene muchas funciones que nos pueden ayudar a realizar cálculos matemáticos. Para comprobar qué
# funciones tiene el módulo, podemos utilizar help(math), o dir(math). Esto mostrará las funciones disponibles en el módulo. 

import math
help(math)

# Otra forma de obtener ayuda

import math
print(dir(math))

# Si queremos importar sólo una función específica del módulo, la importamos de la siguiente manera:

from math import pi
print(pi)   # 3.141592653589793

# También es posible importar varias funciones a la vez

from math import pi, sqrt, pow, floor, ceil, log10
print(pi)                 # 3.141592653589793
print(sqrt(2))            # 1.4142135623730951
print(pow(2, 3))          # 8.0
print(floor(9.81))        # 9
print(ceil(9.81))         # 10
print(log10(100))         # 2.0

# Cuando importamos también podemos renombrar el nombre de la función.

from math import pi as  PI
print(PI) # 3.141592653589793

# Módulo String: Un módulo de cadena es un módulo útil para muchos propósitos. El siguiente ejemplo muestra algunos usos del módulo string.

import string
print(string.ascii_letters)   # abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ
print(string.digits)          # 0123456789
print(string.punctuation)     # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
print(string.ascii_uppercase) # ABCDEFGHIJKLMNOPQRSTUVWXYZ
print(string.hexdigits)       # 0123456789abcdefABCDEF
print(string.octdigits)       # 01234567
print(string.printable)       # 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ 

# Modulo aleatorio: Vamos a importar el módulo random que nos da un número aleatorio entre 0 y 0.9999.... El módulo random tiene muchas funciones
# pero en esta sección sólo usaremos random y randint.

from random import random, randint
print(random())         # 0.5553935029215396 (devuelve un valor entre 0 y 0.9999)
print(randint(5, 20))   # 15 (Devuelve un número entero aleatorio entre [5, 20])


# Ejercicios

# Escribe una función que genere un random_user_id de seis dígitos/caracteres

import random
import string
def generate_random_user_id():
    characters = string.ascii_letters + string.digits
    random_id = ''.join(random.choices(characters, k=6))
    return random_id
random_id = generate_random_user_id()
print(random_id)                        # hTITys

# Declare una función llamada user_id_gen_by_user. No toma ningún parámetro pero toma dos entradas usando input(). Una de las entradas es el número
# de caracteres y la segunda entrada es el número de IDs que se supone deben ser generados.

import string
import random

def user_id_gen_by_user():
    num_chars, num_ids = input("Ingrese el número de caracteres y el número de IDs separados por un espacio: ").split()
    num_chars = int(num_chars)  # (transforma el numero de caracteres en numero)
    num_ids = int(num_ids)      # (transforma el numero de id's en numero.)
    ids = []
    for _ in range(num_ids):
        user_id = ''.join(random.choices(string.ascii_letters + string.digits, k=num_chars))
        ids.append(user_id)
    return '\n'.join(ids) # (se devuelve una cadena que contiene los IDs generados, separados por saltos de línea ('\n').)
print(user_id_gen_by_user())
# r5pge
# bptiI
# p9pBe
# yHmMS
# zPnlq

# Escribe una función llamada rgb_color_gen. Generará colores rgb (3 valores que van de 0 a 255 cada uno)

import random
def rgb_color_gen():
    red = random.randint(0, 255)
    green = random.randint(0, 255)
    blue = random.randint(0, 255)
    return red, green, blue
print(rgb_color_gen())            # (134, 236, 84)

# Escriba una función lista_de_colores_hexa que devuelva cualquier número de colores hexadecimales en una matriz (seis números hexadecimales
# escritos después de #.

import random
def lista_de_colores_hexa(num_colores):
    colores = []
    for _ in range(num_colores):
        color = ''.join([random.choice('0123456789ABCDEF') for _ in range(6)])
        colores.append('#' + color)
    return colores
num_colores = int(input('Ingrese el numero de colores'))  # 6
colores = lista_de_colores_hexa(num_colores)
print(colores)               # ['#4D5B08', '#4366A3', '#4ED15E', '#E2E74C', '#127A52', '#F9C8B6']

# Escriba una función list_of_rgb_colors que devuelva cualquier número de colores RGB en una matriz.

import random

def list_of_rgb_colors(num_colors):
    colors = []
    for _ in range(num_colors):
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        colors.append(color)
    return colors
num_colors = int(input('Ingrese el numero de colores')) # 4
colors = list_of_rgb_colors(num_colors)
print(colors)          # [(138, 203, 217), (105, 74, 107), (209, 75, 130), (205, 28, 83)]

# Escribe una función generar_colores que pueda generar cualquier número de colores hexa o rgb.

import random
def generar_colores(num_colores, tipo):
    colores = []
    for _ in range(num_colores):
        if tipo == 'hexa':
            color = '#' + ''.join(random.choices('0123456789ABCDEF', k=6))
        elif tipo == 'rgb':
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        else:
            return []
        colores.append(color)
    return colores
tipo_colores, num_colores = input("Ingrese el tipo de color: hexa o rgb y el numero de colores separado por un espacio: ").split()  # rgb 5
tipo_colores = str(tipo_colores)  
num_colores = int(num_colores)     
colores = generar_colores(num_colores, tipo_colores)
print(colores)   # [(231, 34, 137), (139, 155, 229), (153, 162, 27), (28, 213, 104), (224, 238, 167)]

# Llame a su función shuffle_list, toma una lista como parámetro y devuelve una lista barajada

import random
def shuffle_list(lista):
    random.shuffle(lista)
    return lista
lista = [1, 2, 3, 4, 5, 6, 7]
print(shuffle_list(lista))    # [3, 7, 6, 4, 2, 1, 5]

# Escriba una función que devuelva una matriz de siete números aleatorios en un rango de 0-9. 

import random
def generate_random_matrix():
    matrix = []
    for _ in range(7):
        number = random.randint(0, 9)
        matrix.append(number)
    return matrix
print(generate_random_matrix())   # [9, 7, 5, 1, 3, 2, 6]