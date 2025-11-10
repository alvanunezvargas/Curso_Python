# Comprensión de listas: La comprensión de listas en Python es una forma compacta de crear una lista a partir de una secuencia. Es una forma corta
# de crear una nueva lista. La comprensión de listas es considerablemente más rápida que procesar una lista utilizando el bucle for.
# [i for i in iterable if expression]

# Por ejemplo, si desea cambiar una cadena (string) a una lista de caracteres. Puedes utilizar un par de métodos. Veamos algunos de ellos:
# primera forma
language = 'Python'
lst = list(language) # (cambiar la cadena o string a lista)
print(type(lst))     # <class 'list'>
print(lst)           # ['P', 'y', 't', 'h', 'o', 'n']

# Segunda forma
language = 'Python'
lst = [i for i in language]
print(type(lst)) # <class 'list'>
print(lst)       # ['P', 'y', 't', 'h', 'o', 'n']

# Por ejemplo, si desea generar una lista de números

# Generar numeros
numbers = [i for i in range(11)]  # (Genera numeros de 0 a 10)
print(numbers)                    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Es posible realizar operaciones matemáticas durante la iteración
squares = [i * i for i in range(11)]
print(squares)                    # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100]

# También es posible hacer una lista de tuplas
numbers = [(i, i * i) for i in range(11)]
print(numbers)                             # [(0, 0), (1, 1), (2, 4), (3, 9), (4, 16), (5, 25)]

# La comprensión de listas puede combinarse con la expresión if

# Generar números pares
even_numbers = [i for i in range(21) if i % 2 == 0]  # (Generar una lista de números pares del 0 al 20)
print(even_numbers)                    # [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

# Generar números impares
odd_numbers = [i for i in range(21) if i % 2 != 0]  # (Generar una lista de números pares del 0 al 20)
print(odd_numbers)                      # [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]

# Filtrar números: vamos a filtrar los números pares positivos de la siguiente lista
numbers = [-8, -7, -3, -1, 0, 1, 3, 4, 5, 7, 6, 8, 10]
positive_even_numbers = [i for i in numbers if i % 2 == 0 and i > 0]
print(positive_even_numbers)                    # [4, 6, 8, 10]

# Aplanamiento de una matriz tridimensional
list_of_lists = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened_list = [number for row in list_of_lists for number in row]
print(flattened_list)    # [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Función lambda: La función lambda es una pequeña función anónima sin nombre. Puede tomar cualquier número de argumentos, pero sólo puede tener
# una expresión. La función lambda es similar a las funciones anónimas en JavaScript. La necesitamos cuando queremos escribir una función anónima
# dentro de otra función.

# Creación de una función lambda: Para crear una función lambda utilizamos la palabra clave lambda seguida de un parámetro(s), seguido de una 
# expresión.Vea la sintaxis y el ejemplo a continuación. La función lambda no utiliza return sino que devuelve explícitamente la expresión.
# x = lambda param1, param2, param3: param1 + param2 + param2
# print(x(arg1, arg2, arg3))

def add_two_nums(a, b):
    return a + b
print(add_two_nums(2, 3))     # 5

# Cambiemos la función anterior por una función lambda

add_two_nums = lambda a, b: a + b
print(add_two_nums(2,3))    # 5

#  Función lambda inmediata o función lambda anónima autoinvocable, es una función lambda

print((lambda a, b: a + b)(2,3))   # 5

square = lambda x : x ** 2
print(square(3))             # 9

cube = lambda x : x ** 3
print(cube(3))               # 27

# Variables múltiples

multiple_variable = lambda a, b, c: a ** 2 - 3 * b + 4 * c
print(multiple_variable(5, 5, 3))    # 22

# Uso de una función lambda dentro de otra función.

def power(x):
    return lambda n : x ** n
cube = power(2)(3)   # la función power necesita ahora 2 argumentos para ejecutarse, entre paréntesis redondeados separados
print(cube)          # 8
two_power_of_five = power(2)(5) 
print(two_power_of_five)  # 32


# Ejercicios

# Filtrar sólo negativo y cero en la lista utilizando la comprensión de la lista numeros = [-4, -3, -2, -1, 0, 2, 4, 6]

numeros = [-4, -3, -2, -1, 0, 2, 4, 6]
filtrados = [i for i in numeros if i <= 0]
print(filtrados)   # [-4, -3, -2, -1, 0]

# Aplanar la siguiente lista lista_de_listas a una lista unidimensional:: lista_de_listas =[[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]]]

lista_de_listas = [[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]]]
lista_aplanada = [num for sublist in lista_de_listas for subsublist in sublist for num in subsublist]
print(lista_aplanada)   # [1, 2, 3, 4, 5, 6, 7, 8, 9]

# lista_aplanada = [num indica que vamos a crear una nueva lista llamada lista_aplanada y cada elemento de esta lista será num.
# for sublist in lista_de_listas establece un bucle for para recorrer cada elemento sublist en la lista lista_de_listas. Cada sublist representa
# una sublista dentro de lista_de_listas.
# for subsublist in sublist establece otro bucle for anidado para recorrer cada elemento subsublist dentro de la sublista sublist. Cada subsublist
# representa una sub-sublista dentro de la sublista sublist.
# for num in subsublist establece otro bucle for anidado para recorrer cada elemento num dentro de la sub-sublista subsublist. Cada num representa
# un número dentro de la sub-sublista subsublist.

# Usando la comprensión de listas crea la siguiente lista de tuplas:
[(0, 1, 0, 0, 0, 0, 0),
(1, 1, 1, 1, 1, 1, 1),
(2, 1, 2, 4, 8, 16, 32),
(3, 1, 3, 9, 27, 81, 243),
(4, 1, 4, 16, 64, 256, 1024),
(5, 1, 5, 25, 125, 625, 3125),
(6, 1, 6, 36, 216, 1296, 7776),
(7, 1, 7, 49, 343, 2401, 16807),
(8, 1, 8, 64, 512, 4096, 32768),
(9, 1, 9, 81, 729, 6561, 59049),
(10, 1, 10, 100, 1000, 10000, 100000)]

lista_de_tuplas = [(i, 1, i, i**2, i**3, i**4, i**5) for i in range(11)]
print(lista_de_tuplas)

# Aplanar la lista [[('Finland', 'Helsinki')], [('Sweden', 'Stockholm')], [('Norway', 'Oslo')]] para obtener la siguiente lista
# [['FINLAND','FIN', 'HELSINKI'], ['SWEDEN', 'SWE', 'STOCKHOLM'], ['NORWAY', 'NOR', 'OSLO']]

lista = [[('Finland', 'Helsinki')], [('Sweden', 'Stockholm')], [('Norway', 'Oslo')]]
lista_aplanada = [[pais.upper(), pais[:3].upper(), ciudad.upper()] for sublist in lista for pais, ciudad in sublist]
print(lista_aplanada)

# Cambia la siguiente lista [[('Finland', 'Helsinki')], [('Sweden', 'Stockholm')], [('Norway', 'Oslo')]] por una lista de diccionarios 
# [{'country': 'FINLAND', 'city': 'HELSINKI'}, {'country': 'SWEDEN', 'city': 'STOCKHOLM'}, {'country': 'NORWAY', 'city': 'OSLO'}]

lista_tuplas = [[('Finland', 'Helsinki')], [('Sweden', 'Stockholm')], [('Norway', 'Oslo')]]
lista_diccionarios = [{'country': pais.upper(), 'city': ciudad.upper()} for sublist in lista_tuplas for pais, ciudad in sublist]
print(lista_diccionarios)

# Cambia la siguiente lista [[('Álvaro', 'Núñez')], [('Claudia', 'Osorio')], [('Nicolás', 'Núñez')], [('Sebastián', 'Núñez')]] por una lista de
# cadenas concatenadas ['Álvaro Núñez', 'Claudia Osorio', 'Nicolás Núñez', 'Sebastián Núñez'].

lista_tuplas = [[('Alvaro', 'Nunez')], [('Claudia', 'Osorio')], [('Nicolas', 'Nunez')], [('Sebastian', 'Nunez')]]
lista_cadenas = [' '.join(tupla[0]) for tupla in lista_tuplas]
print(lista_cadenas)

# Escribe una función lambda que pueda resolver una pendiente o intersección de funciones lineales.

funcion_lineal = lambda x, m, b: m * x + b
# Calcule el valor de y para x = 2, m = 3, b = 1
result = funcion_lineal(2, 3, 1)
print(result)  # 7
# Calculale el intercepto x = 0, m = -2, b = 5
print(funcion_lineal(0, -2, 5))  # 5

pendiente = lambda x1, x2, y1, y2: (y2-y1)/(x2-x1)
print(pendiente(5, 10, 4, 8))  # 0.8