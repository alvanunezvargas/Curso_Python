# Bucles (Loops)
# Para manejar tareas repetitivas los lenguajes de programación utilizan bucles. El lenguaje de programación Python también proporciona los siguientes 
# dos bucles:
# bucle while
# bucle for

# Bucle while: Utilizamos la palabra reservada while para hacer un bucle while. Se utiliza para ejecutar un bloque de sentencias repetidamente hasta
# que se cumpla una condición dada. Cuando la condición se convierte en falsa, las líneas de código después del bucle continuarán ejecutándose.

# while:
#     el código va aquí

count = 0
while count < 5:
    print(count)  # 0 1 2 3 4
    count = count + 1
# Este es un código en Python que utiliza un bucle while para imprimir los números del 0 al 4. La variable count se inicializa en 0 y luego se verifica
# si es menor que 5. Si es así, se imprime el valor de count y luego se incrementa en 1. Este proceso se repite hasta que count sea igual a 5, momento
# en el que el bucle termina y el programa finaliza.

# En el bucle while anterior, la condición se convierte en falsa cuando la cuenta es 5. Es entonces cuando el bucle se detiene. Si estamos interesados
# en ejecutar el bloque de código una vez que la condición ya no es verdadera, podemos utilizar else.

# while:
#     el código va aquí
# else:
#    el código va aquí

count = 0
while count < 5:
    print(count)  # 0 1 2 3 4
    count = count + 1
else:
    print(count) # 5
# La condición del bucle anterior será falsa cuando la cuenta sea 5 y el bucle se detenga, y la ejecución comience con la sentencia else. Como resultado
# se imprimirá 5.

# Pausa y continuación - Parte 1
# Pausa: Utilizamos break cuando queremos salir del bucle o detenerlo.

# Pausa y continuación - Parte 1: Pausa: Utilizamos break cuando queremos salir del bucle o detenerlo.

# while:
#     el código va aquí
#     if otra_condición:
#        break

count = 0
while count < 5:
    print(count)
    count = count + 1
    if count == 3:
        break
# El bucle while anterior sólo imprime 0, 1, 2, pero cuando llega a 3 se detiene.

# Continuar: Con la sentencia "continue" podemos saltarnos la iteración actual, y continuar con la siguiente:

# condición while:
#    el código va aquí
#    if otra_condición:
#        continue

count = 0
while count < 5:
    if count == 3:
        count = count + 1
        continue
    print(count)
    count = count + 1
# Otra forma
count = 0
while count < 5:
    if count == 3:
        count += 1  # (Este codigo es equivalente a count = count + 1)
        continue
    print(count)
    count += 1
# El bucle while sólo imprime 0, 1, 2 y 4 (se salta 3). Este código comienza inicializando la variable "count" en 0. Luego, el bucle "while" 
# se ejecuta siempre que el valor de "count" sea menor que 5. # Dentro del bucle, la estructura "if" evalúa si el valor actual de "count" es igual a 3.
# Si es así, la variable "count" se incrementa en 1 y la palabra clave "continue" salta a la siguiente iteración del bucle sin imprimir el valor actual 
# de "count". Si el valor actual de "count" no es igual a 3, se imprime utilizando la función "print". Luego, la variable "count" se incrementa en 1 
# y el bucle continúa.

# Bucle for: La palabra clave for se utiliza para hacer un bucle for, similar a otros lenguajes de programación, pero con algunas diferencias sintácticas.
# El bucle se utiliza para iterar sobre una secuencia (que puede ser una lista, una tupla, un diccionario, un conjunto o una cadena).
# for iterador en lst:
#    el código va aquí
numeros = [0, 1, 2, 3, 4, 5]
for numero in numeros: # número es un nombre temporal para referirse a los elementos de la lista, válido sólo dentro de este bucle
    print(numero) # los números se imprimirán línea a línea, del 0 al 5

# Bucle for con cadena de caracteres
language = 'Python'
for letter in language:
    print(letter)  # (las letras se imprimirán línea a línea, de la "P" a la "n")

# Otra forma de obtener lo mismo:
    
for i in range(len(language)):
    print(language[i])  # las letras se imprimirán línea a línea, de la "P" a la "n"

# Bucle for con tupla:
# for iterador in tpl:
#    el código va aquí
numeros = (0, 1, 2, 3, 4, 5)
for numero in numeros: # número es un nombre temporal para referirse a los elementos de la lista, válido sólo dentro de este bucle
    print(numero) # los números se imprimirán línea a línea, del 0 al 5

# Bucle for con diccionario El bucle a través de un diccionario le proporciona la clave del diccionario.
# for iterador in dct:
#    el código va aquí
person = {
    'first_name':'Alvaro',
    'last_name':'Nunez',
    'age':55,
    'country':'Colombia',
    'is_marred':True,
    'skills':['MatLab', 'Quality Control', 'Fuel oils', 'volumetric correction factors', 'Python'],
    'address':{
        'street':'Avenue 19',
        'zipcode':'630004'
    }
    }
for key in person:
    print(key)    # (De esta forma obtendremos las claves)

for key, value in person.items():
    print(key, value) # (De esta forma obtendremos las claves y los valores)
    
# Bucles en set:  
# for iterator in st:
# el código va aquí
it_companies = {'Facebook', 'Google', 'Microsoft', 'Apple', 'IBM', 'Oracle', 'Amazon'}
for company in it_companies:
    print(company)
# Oracle
# IBM
# Facebook
# Microsoft
# Amazon
# Apple
# Google

# El resultado de este código sería imprimir en la consola cada uno de los elementos del conjunto "it_companies" en una línea separada, en el orden en 
# que aparecen en el conjunto (que puede variar en cada ejecución debido a que los conjuntos no están ordenados)

# Pausa y continuación - Parte 2
# Breve recordatorio: Break: Usamos break cuando queremos parar nuestro bucle antes de que se complete.
# for iterador en secuencia:
#    el código va aquí
#    if condición:
#        break
numbers = (0,1,2,3,4,5)
for number in numbers:
    print(number)
    if number == 3:
        break
# 0
# 1
# 2
# 3
# En el ejemplo anterior, el bucle se detiene cuando llega a 3.

# Continuar: Utilizamos continue cuando queremos saltarnos alguno de los pasos de la iteración del bucle.
# for iterador en secuencia:
#    el código va aquí
#    if condición:
#        continue
numbers = (0,1,2,3,4,5)
for number in numbers:
    print(number)
    if number == 3:
        continue
    print('El siguiente número debería ser ', number + 1) if number != 5 else print("fin de bucle") # para las condiciones abreviadas se necesitan declaraciones if y else
print('fuera del bucle')
# 0
# El siguiente número debería ser  1
# 1
# El siguiente número debería ser  2
# 2
# El siguiente número debería ser  3
# 3
# 4
# El siguiente número debería ser  5
# 5
# fin de bucle
# fuera del bucle
# En el ejemplo anterior, si el número es igual a 3, el paso después de la condición (pero dentro del bucle) se salta y la ejecución del bucle continúa si
# queda alguna iteración.

# Otra forma de escribir el codigo anterior es:
numbers = (0, 1, 2, 3, 4, 5)
for number in numbers:
    print(number)
    if number == 3:
        continue
    if number < 5:
        print('El siguiente número debería ser', number + 1)
    else:
        print('fin de bucle')
print('fuera del bucle')

# Creación de secuencias con range: La función range() utiliza una lista de números. La función range(inicio, fin, paso) toma tres parámetros: inicio, fin e incremento.
# Por defecto empieza en 0 y el incremento es 1. La secuencia range necesita al menos 1 argumento (end).
lst = list(range(11)) 
print(lst) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
st = set(range(1, 11))    # 2 argumentos indican el inicio y el final de la secuencia, paso fijado por defecto en 1
print(st) # {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

lst = list(range(11)) 
print(lst) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
st = set(range(0, 11))    # 2 argumentos indican el inicio y el final de la secuencia, paso fijado por defecto en 1
print(st) # {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

lst = list(range(11)) 
print(lst) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
st = set(range(1, 11, 2))    # 2 argumentos indican el inicio y el final de la secuencia, paso fijado por defecto en 1
print(st) # {1, 3, 5, 7, 9}

# for iterador en rango(inicio, fin, intervalo):
lst = list(range(0,11,2))    # ((0,11,2) (O: Digito de inicio; 11: Numero de digitos; 2: Intervalo de los digitos)
print(lst) # [0, 2, 4, 6, 8, 10]
st = set(range(0,11,2))
print(st) #  {0, 2, 4, 6, 8, 10}

# Bucle For anidado: Podemos escribir bucles dentro de un bucle.
person = {
    'first_name':'Alvaro',
    'last_name':'Nunez',
    'age':55,
    'country':'Colombia',
    'is_marred':True,
    'skills':['MatLab', 'Quality Control', 'Fuel oils', 'volumetric correction factors', 'Python'],
    'address':{
        'street':'Avenue 19',
        'zipcode':'630004'
    }
    }
for key in person:
    if key == 'skills':
        for skill in person['skills']:
            print(skill)
# MatLab
# Quality Control
# Fuel oils
# volumetric correction factors
# Python

# for else: Si queremos ejecutar algún mensaje cuando termine el bucle, usamos else.
# for iterador en rango(inicio, fin, intervalo):
#     hacer algo
# else:
#     print('El bucle ha terminado')
for numero in range(11):
    print(numero) # imprime 0 a 10, sin incluir 11
else:
    print('El bucle se detiene en', numero)
# 0
# 1
# 2
# 3
# 4
# 5
# 6
# 7
# 8
# 9
# 10
# El bucle se detiene en 10

# Pass: En python cuando se requiere una sentencia (después del punto y coma), pero no queremos ejecutar ningún código allí, podemos escribir la palabra
# pass para evitar errores. También podemos utilizarlo como un marcador de posición, para futuras declaraciones.
# Sea el siguiente comando:
i = 10
while i>-3:
    i-=1
    if i==0:
        continue
    print("1/"+str(i)+"="+str(1/i))
# Si borramos la linea "continue", y traemos el siguiente comando, nos va a dar error.
i = 10
while i>-3:
    i-=1
    if i==0:
        print("1/"+str(i)+"="+str(1/i))  # ZeroDivisionError: division by zero
# para evitar el error anterior, introducimos "pass" en la linea donde estaba "continue"
i = 10
while i>-3:
    i-=1
    if i==0:
        pass
    print("1/"+str(i)+"="+str(1/i))


# Ejercicios:

# Iterar de 0 a 10 usando el bucle for, hacer lo mismo usando el bucle while.
for i in range(11):
    print(i)

i = 0
while i <= 10:
    print(i)
    i += 1

# Iterar de 10 a 0 usando el bucle for, hacer lo mismo usando el bucle while.
for i in range(10, -1, -1):
    print(i)

i = 10
while i >= 0:
    print(i)
    i -= 1

# Escribe un bucle que haga siete llamadas a print(), de forma que obtengamos en la salida el siguiente triángulo:
  #
  ##
  ###
  ####
  #####
  ######
  #######

for i in range(1, 8):
    print("#" * i)

# Utiliza bucles anidados para crear lo siguiente:
# # # # # # # #
# # # # # # # #
# # # # # # # #
# # # # # # # #
# # # # # # # #
# # # # # # # #
# # # # # # # #
# # # # # # # #
for i in range(8):
    for j in range(8):
        print("# ", end="")
    print()
# En este código, el primer bucle for se encarga de imprimir cada fila, mientras que el segundo bucle for se encarga de imprimir cada columna dentro de la fila.
# El print() dentro del primer bucle for se utiliza para imprimir un salto de línea después de imprimir cada fila. El end="" en el print("# ", end="")
# se utiliza para imprimir el # y el espacio en la misma línea sin agregar un salto de línea adicional.

# Imprime el siguiente patrón:
# 0 x 0 = 0
# 1 x 1 = 1
# 2 x 2 = 4
# 3 x 3 = 9
# 4 x 4 = 16
# 5 x 5 = 25
# 6 x 6 = 36
# 7 x 7 = 49
# 8 x 8 = 64
# 9 x 9 = 81
#10 x 10 = 100
for i in range(11):
    print(str(i) + ' x ' + str(i) + ' = ' + str(i*i))

# Iterar a través de la lista, ['Python', 'Numpy','Pandas','Django', 'Flask'] utilizando un bucle for e imprimir los elementos.
lst = ['Python', 'Numpy', 'Pandas', 'Django', 'Flask']
for item in lst:
    print(item)

# Utiliza el bucle for para iterar de 0 a 100 e imprimir sólo los números pares
for i in range(0, 101, 2):
    print(i)

# Utiliza el bucle for para iterar de 0 a 100 e imprimir sólo los números impares
for i in range(1, 101, 2):
    print(i)

# Utiliza el bucle for para iterar de 0 a 100 e imprimir la suma de todos los números.
total = 0
for i in range(101):
    total += i
print(total)  # 5050

# Utilice el bucle for para iterar de 0 a 100 e imprimir la suma de todos los pares y la suma de todas las probabilidades.
suma_pares = 0
suma_impares = 0
for i in range(101):
    if i % 2 == 0:
        suma_pares += i
    else:
        suma_impares += i
print("Suma de los números pares: ", suma_pares)   # Suma de los números pares:  2550
print("Suma de los números impares: ", suma_impares)  # Suma de los números impares:  2500

# De la lista countries extraer todos los que contengan la palabra land.
import requests
url = 'https://raw.githubusercontent.com/Asabeneh/30-Days-Of-Python/master/data/countries.py'
response = requests.get(url)
data = response.text.split('\n')
land_countries = [country for country in data if 'land' in country.lower()]
cleaned_list = []
for item in land_countries:
    cleaned_item = item.strip().replace("'", "")
    cleaned_list.append(cleaned_item)
print(cleaned_list)
# ['Finland,', 'Iceland,', 'Ireland,', 'Marshall Islands,', 'Netherlands,', 'New Zealand,', 'Poland,', 'Solomon Islands,', 'Swaziland,', 'Switzerland,', 'Thailand,']

# Ve a la carpeta de datos y utiliza el archivo countries_data.py. en la pagina https://raw.githubusercontent.com/Asabeneh/30-Days-Of-Python/master/data/countries-data.py
# ¿Cuál es el número total de lenguas en los datos?

import urllib.request   #(Solicitudes HTTP:"urllib" es una biblioteca de Python integrada en la versión de Python, mas dificil que "requests")
import ast
url = "https://raw.githubusercontent.com/Asabeneh/30-Days-Of-Python/master/data/countries-data.py"
response = urllib.request.urlopen(url)
data = response.read().decode()
data_dict = ast.literal_eval(data)

total_languages = 0

for country in data_dict:
    total_languages += len(country["languages"])

print("El número total de lenguajes en los datos es:", total_languages) # El número total de lenguajes en los datos es: 368

# Encuentra los diez idiomas más hablados en los datos

import requests #(Solicitudes HTTP:"requests" es un módulo de terceros que se debe instalar por separado )
url = "https://raw.githubusercontent.com/Asabeneh/30-Days-Of-Python/master/data/countries-data.py"
response = requests.get(url)
data = eval(response.content)
languages_count = {}
# Recorrer cada país y contar el número de lenguas
for country in data:
    for language in country['languages']:
        if language in languages_count:
            languages_count[language] += 1
        else:
            languages_count[language] = 1
# Ordenar el diccionario por cuenta de lengua
sorted_languages = sorted(languages_count.items(), key=lambda x: x[1], reverse=True)
# Imprimir las diez lenguas más habladas
print("Las diez lenguas más habladas en el mundo son:")
for language, count in sorted_languages[:10]:
    print(f"{language} ({count} países)")
# Las diez lenguas más habladas en el mundo son:
# English (91 países)
# French (45 países)
# Arabic (25 países)
# Spanish (24 países)
# Portuguese (9 países)
# Russian (9 países)
# Dutch (8 países)
# German (7 países)
# Chinese (5 países)
# Serbian (4 países)

# Encuentra los 10 países más poblados del mundo

import requests
# Descarga el archivo de países
url = 'https://raw.githubusercontent.com/Asabeneh/30-Days-Of-Python/master/data/countries-data.py'
response = requests.get(url)
data = response.json()
# Crea una lista de tuplas de cada país y su población
populations = []
for country in data:
    name = country['name']
    population = country['population']
    populations.append((name, population))
# Ordena la lista de tuplas por la población en orden descendente
populations.sort(key=lambda x: x[1], reverse=True)
# Imprime los primeros 10 países de la lista ordenada
print('Los 10 países más poblados del mundo son:')
for i, (name, population) in enumerate(populations[:10]):
    print(f'{i+1}. {name} - {population}')
# Los 10 países más poblados del mundo son:
# 1. China - 1377422166
# 2. India - 1295210000
# 3. United States of America - 323947000
# 4. Indonesia - 258705000
# 5. Brazil - 206135893
# 6. Pakistan - 194125062
# 7. Nigeria - 186988000
# 8. Bangladesh - 161006790
# 9. Russian Federation - 146599183
# 10. Japan - 126960000