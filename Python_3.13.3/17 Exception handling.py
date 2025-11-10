"""
Manejo de Excepciones
El uso de try y except en Python nos permite manejar los errores de manera controlada y evitar que el programa se bloquee cuando se produce una excepción.
Esto se conoce como "manejo de errores gracioso" o "manejo de errores con gracia". La estructura básica de try y except es la siguiente: 

try:
    # Código donde se podría producir una excepción
    # ...
except TipoDeExcepcion:
    # Código a ejecutar si se produce la excepción
    # ...

Dentro del bloque try, se coloca el código que podría generar una excepción. Si ocurre alguna excepción dentro de ese bloque, el flujo del programa se desvía inmediatamente al bloque except correspondiente al tipo de excepción capturada.

El bloque except es donde se maneja la excepción. Aquí puedes escribir código para controlar la situación de error de manera adecuada, como mostrar un mensaje de error, registrar información relevante o tomar acciones correctivas.
Puedes tener varios bloques except para manejar diferentes tipos de excepciones y también puedes utilizar un bloque except sin especificar el tipo de excepción para capturar cualquier tipo de excepción no esperada.
Aquí hay un ejemplo para ilustrar el uso de try y except:
"""

try:
    dividend = 10
    divisor = 0
    result = dividend / divisor
    print("El resultado es:", result)
except ZeroDivisionError:
    print("Error: No se puede dividir entre cero")  # Error: No se puede dividir entre cero

# Otro ejemplo

try:
    print(10 + '5')
except:
    print('Algo salió mal')  # Algo salió mal
    
"""
En el ejemplo anterior el segundo operando es una cadena (string). Podríamos cambiarlo a float o int para sumarlo con el número y que funcionara.
Pero sin este cambio, el segundo bloque, except, se ejecutará.
"""

"""
En el siguiente ejemplo, el bloque de excepciones se ejecutará y no sabremos exactamente cuál es el problema. Para analizar el problema,
podemos utilizar los diferentes tipos de error con except.
"""
try:
    name = input('Enter your name:')
    year_born = input('Year you were born:')
    age = 2019 - year_born
    print(f'You are {name}. And your age is {age}.')
except:
    print('Something went wrong')

"""
Enter your name:Alvaro
Year you were born:1967
Something went wrong
"""

# En el siguiente ejemplo, manejará el error y también nos dirá el tipo de error planteado.

try:
    name = input('Enter your name:')
    year_born = input('Year you were born:')
    age = 2019 - year_born
    print(f'You are {name}. And your age is {age}.')
except TypeError:
    print('Type error occured')
except ValueError:
    print('Value error occured')
except ZeroDivisionError:
    print('zero division error occured')
    
"""
Enter your name:Alvaro
Year you were born:1967
Type error occured
"""

# En el código anterior el error es tipo "TypeError". Ahora, vamos a correjir la linea "age = 2019 - year_born":

try:
    name = input('Enter your name:')
    year_born = input('Year you born:')
    age = 2019 - int(year_born)
    print('You are {name}. And your age is {age}.')
except TypeError:
    print('Type error occur')
except ValueError:
    print('Value error occur')
except ZeroDivisionError:
    print('zero division error occur')
else:
    print('I usually run with the try block')
finally:
    print('I alway run.')

"""
Enter your name:Alvaro
Year you born:1967
You are {name}. And your age is {age}.
I usually run with the try block
I alway run.
"""

# El código anterior lo podemos reducir de la siguiente manera:

try:
    name = input('Enter your name:')
    year_born = input('Year you born:')
    age = 2019 - int(year_born)
    print('You are {name}. And your age is {age}.')
except Exception as e:
    print(e)
    
"""
Enter your name:Alvaro
Year you born:1967
You are {name}. And your age is {age}.
"""

"""
Empaquetar y desempaquetar argumentos en Python
 Para esto Utilizamos dos operadores:
* para tuplas
** para diccionarios
"""

def sum_of_five_nums(a, b, c, d, e):
    return a + b + c + d + e
lst = [1, 2, 3, 4, 5]
print(sum_of_five_nums(lst)) # TypeError: sum_of_five_nums() missing 4 required positional arguments: 'b', 'c', 'd', and 'e'

"""
El código que has proporcionado muestra un error de tipo (TypeError) debido a una incompatibilidad en la llamada a la función sum_of_five_nums().

La función sum_of_five_nums() espera cinco argumentos posicionales (a, b, c, d, e), pero al llamar a la función pasando lst, que es una lista, se produce
un error.

El error indica que faltan cuatro argumentos posicionales requeridos: 'b', 'c', 'd' y 'e'. Esto se debe a que la lista lst no se desempaquetó para pasar
sus elementos como argumentos individuales.

Para solucionar esto, debes desempaquetar la lista utilizando el operador * al llamar a la función. De esta manera, los elementos de la lista se pasarán
como argumentos individuales y coincidirán con los parámetros esperados por la función.
"""

def sum_of_five_nums(a, b, c, d, e):
    return a + b + c + d + e
lst = [1, 2, 3, 4, 5]
print(sum_of_five_nums(*lst))  # 15 (Desempaqueta la lista usando el operador *)

# También podemos usar el desempaquetado en la función incorporada range(), que espera un inicio y un final.

numbers = range(2, 7)  
print(list(numbers)) # [2, 3, 4, 5, 6]
args = [2, 7]
numbers = range(args) 
print(list(numbers))      # TypeError: 'list' object cannot be interpreted as an integer

"""
El código que has proporcionado muestra un error debido a un mal uso de la función range() en la segunda parte del código.

En la primera parte del código, se define la variable numbers utilizando la función range() con los argumentos 2 y 7. Esto crea una secuencia de
números desde 2 hasta 6 (el número 7 no está incluido). Luego, se convierte la secuencia en una lista utilizando la función list() y se imprime el
resultado: [2, 3, 4, 5, 6].

En la segunda parte del código, se define la lista args con los elementos 2 y 7. Luego, se intenta utilizar la lista args como argumento para la función
range(). Sin embargo, la función range() espera recibir argumentos enteros que representen el inicio, el final y el paso de la secuencia, no una lista.
Por lo tanto, se produce un error de tipo TypeError con el mensaje "TypeError: 'list' object cannot be interpreted as an integer".

Para solucionar este error, debes desempaquetar la lista args utilizando el operador * al llamar a la función range(), de la siguiente manera:
"""
numbers = range(2, 7)  
print(list(numbers)) # [2, 3, 4, 5, 6]
args = [2, 7]
numbers = range(*args)
print(list(numbers))   # [2, 3, 4, 5, 6]

# Una lista o tupla también se puede descomprimir de esta forma:

countries = ['Finland', 'Sweden', 'Norway', 'Denmark', 'Iceland']
fin, sw, nor, *rest = countries
print(fin, sw, nor, rest)   # Finland Sweden Norway ['Denmark', 'Iceland']
numbers = [1, 2, 3, 4, 5, 6, 7]
one, *middle, last = numbers
print(one, middle, last)      #  1 [2, 3, 4, 5, 6] 7

# Desempaquetar diccionarios

def unpacking_person_info(name, country, city, age):
    return f'{name} lives in {country}, {city}. He is {age} year old.'
dct = {'name':'Alvaro', 'country':'Colombia', 'city':'Armenia', 'age':55}
print(unpacking_person_info(**dct)) # Alvaro lives in Colombia, Armenia. He is 55 year old.

"""
Empaquetar
A veces nunca sabemos cuántos argumentos hay que pasar a una función python. Podemos utilizar el método de empaquetamiento para permitir que nuestra
función tome un número ilimitado o arbitrario de argumentos.
"""
# Empaquetando listas:

def sum_all(*args):
    s = 0
    for i in args:
        s += i
    return s
print(sum_all(1, 2, 3))             # 6
print(sum_all(1, 2, 3, 4, 5, 6, 7)) # 28

# Empaquetando diccionarios:

"""
En el siguiente código se define una función llamada packing_person_info que utiliza el operador **kwargs para permitir el paso de un número variable
de argumentos de palabras clave (keyword arguments) a la función.

Dentro de la función, se itera sobre los elementos del diccionario kwargs utilizando un bucle for, y se imprime cada clave y valor utilizando la 
sintaxis de f-string. Luego, se devuelve el diccionario kwargs.

Cuando se llama a la función packing_person_info con los argumentos de palabras clave name="Alvaro", country="Colombia", city="Armenia", age=55, se
imprimirá cada clave y valor en el diccionario, y el resultado se devolverá como un diccionario.
"""
def packing_person_info(**kwargs):
    for key in kwargs:
        print(f"{key} = {kwargs[key]}")
    return kwargs
print(packing_person_info(name="Alvaro", country="Colombia", city="Armenia", age=55))

"""
name = Alvaro
country = Colombia
city = Armenia
age = 55
{'name': 'Alvaro', 'country': 'Colombia', 'city': 'Armenia', 'age': 55}
"""

# Difundir en Python

lst_one = [1, 2, 3]
lst_two = [4, 5, 6, 7]
lst = [0, *lst_one, *lst_two]
print(lst)          # [0, 1, 2, 3, 4, 5, 6, 7]
country_lst_one = ['Finland', 'Sweden', 'Norway']
country_lst_two = ['Denmark', 'Iceland']
nordic_countries = [*country_lst_one, *country_lst_two]
print(nordic_countries)  # ['Finland', 'Sweden', 'Norway', 'Denmark', 'Iceland']


# Enumerar: Si estamos interesados en un índice de una lista, utilizamos la función incorporada "enumerate" para obtener el índice de cada elemento de la lista.

for index, item in enumerate([20, 30, 40]):
    print(index, item)
"""
0 20
1 30
2 40
"""

countries = ['Finland', 'Sweden', 'Norway', 'Denmark', 'Iceland']
for index, country in enumerate(countries):
    print('hi')
    if country == 'Finland':
        print(f"The country {country} has been found at index {index}")
"""
hi
The country Finland has been found at index 0
hi
hi
hi
hi
"""

countries = ['Finland', 'Sweden', 'Norway', 'Denmark', 'Iceland']
for index, country in enumerate(countries):
    if country == 'Finland':
        print(f"The country {country} has been found at index {index}")  # The country Finland has been found at index 0
        
# ZIP:  Se utiliza para combinar elementos de dos o más listas en pares ordenados durante un bucle. La función zip() toma varias secuencias como argumentos
# (puede ser listas, tuplas, etc.) y crea un nuevo iterador que produce tuplas con elementos correspondientes de las secuencias dadas.

names = ['Alice', 'Bob', 'Charlie']
ages = [25, 30, 22]
for name, age in zip(names, ages):
    print(f"{name} is {age} years old")
"""
Alice is 25 years old
Bob is 30 years old
Charlie is 22 years old
"""

fruits = ['banana', 'orange', 'mango', 'lemon', 'lime']                    
vegetables = ['Tomato', 'Potato', 'Cabbage','Onion', 'Carrot']
fruits_and_veges = []
for f, v in zip(fruits, vegetables):
    fruits_and_veges.append({'fruit':f, 'veg':v})
print(fruits_and_veges)
"""
[{'fruit': 'banana', 'veg': 'Tomato'}, {'fruit': 'orange', 'veg': 'Potato'}, {'fruit': 'mango', 'veg': 'Cabbage'}, {'fruit': 'lemon', 'veg': 'Onion'}, {'fruit': 'lime', 'veg': 'Carrot'}]
"""

# Ejercicios:

# names = ['Finlandia', 'Suecia', 'Noruega','Dinamarca','Islandia', 'Estonia','Rusia']. Descomprime los cinco primeros países y guárdalos en una variable
# paises_nordicos, Estonia en la variable es y Rusia en la variable ru

names = ['Finlandia', 'Suecia', 'Noruega', 'Dinamarca', 'Islandia', 'Estonia', 'Rusia']
paises_nordicos, es, ru = names[:5], names[-2:-1], names[-1:]
print("Países nórdicos:", paises_nordicos)
print("es:", es)
print("ru:", ru)
"""
Países nórdicos: ['Finlandia', 'Suecia', 'Noruega', 'Dinamarca', 'Islandia']
es: ['Estonia']
ru: ['Rusia']
"""

names = ['Finlandia', 'Suecia', 'Noruega', 'Dinamarca', 'Islandia', 'Estonia', 'Rusia']
paises_nordicos, *otros, es, ru = names[:-2], names[-2], names[-1]
print("Países nórdicos:", paises_nordicos)
print("es:", [es])
print("ru:", [ru])
"""
Países nórdicos: ['Finlandia', 'Suecia', 'Noruega', 'Dinamarca', 'Islandia']
es: ['Estonia']
ru: ['Rusia']
"""

names = ['Finlandia', 'Suecia', 'Noruega', 'Dinamarca', 'Islandia', 'Estonia', 'Rusia']
paises_nordicos = names[:5]
es = names[-2]
ru = names[-1]
print("Países nórdicos:", paises_nordicos)
print("es:", [es])
print("ru:", [ru])
"""
Países nórdicos: ['Finlandia', 'Suecia', 'Noruega', 'Dinamarca', 'Islandia']
es: ['Estonia']
ru: ['Rusia']
"""