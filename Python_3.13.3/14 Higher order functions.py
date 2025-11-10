# Funciones de orden superior: son aquellas que pueden tomar otras funciones como argumentos y/o devolver funciones como resultado. Esto permite
# una mayor flexibilidad y modularidad en el diseño de programas.

# Una función puede tomar una o más funciones como parámetros
# Una función puede ser devuelta como resultado de otra función
# Una función puede modificarse
# Una función puede asignarse a una variable

# En esta sección, cubriremos:

# Manejo de funciones como parámetros
# Devolver funciones como valor de retorno de otras funciones
# Uso de cierres y decoradores de Python

# Función como parámetro

def sum_numbers(nums):  # (Toma una lista de números (nums) como argumento y devuelve la suma de los números utilizando la función sum())
    return sum(nums)    
def higher_order_function(f, lst):  # (Es una función de orden superior que toma dos argumentos: f es una función y lst es una lista.)
    summation = f(lst)
    return summation  # (Dentro de la función, se llama a la función f pasando la lista lst como argumento, y el resultado se almacena en la variable summation)
result = higher_order_function(sum_numbers, [1, 2, 3, 4, 5]) # (Se llama a la función de orden superior pasando la función sum_numbers como argumento y una lista [1, 2, 3, 4, 5])
print(result)       # 15

# Función como valor de retorno

def square(x):          # (Funcion cuadratica)
    return x ** 2

def cube(x):           # (Funcion cubica)
    return x ** 3

def absolute(x):        # (Funcion valor absoluto)
    if x >= 0:
        return x
    else:
        return -(x)

def higher_order_function(type):  # (Función de orden superior que devuelve una función)
    if type == 'square':
        return square
    elif type == 'cube':
        return cube
    elif type == 'absolute':
        return absolute

result = higher_order_function('square')
print(result(3))       # 9
result = higher_order_function('cube')
print(result(3))       # 27
result = higher_order_function('absolute')
print(result(-3))      # 3

# Puede ver en el ejemplo anterior que la función de orden superior devuelve diferentes funciones dependiendo del parámetro pasado


# Cierres en Python: Python permite que una función anidada acceda al ámbito externo de la función que la encierra. Esto se conoce como Cierre.
# Veamos cómo funcionan los cierres en Python. En Python, un cierre se crea anidando una función dentro de otra función encapsuladora y devolviendo
# la función interna. Véase el ejemplo siguiente.

def add_ten():
    ten = 10
    def add(num):
        return num + ten
    return add

closure_result = add_ten()
print(closure_result(5))  # 15
print(closure_result(10))  # 20

# Explicacion codigo:
    
# 1- Se define la función add_ten() que inicializa una variable ten con el valor 10.
# 2- Se define la función interna add(num) que toma un argumento num y retorna la suma de num con ten.
# 3- Se devuelve la función interna add como resultado de add_ten().
# 4- Se asigna el resultado de add_ten() a la variable closure_result.
# 5- Se llama a closure_result(5), lo que ejecuta el cierre add con el argumento 5. El cierre suma 5 con el valor de ten (que es 10) y devuelve
# el resultado 15.
# 6- Se imprime el resultado de closure_result(5), que es 15.
# 7- Se llama a closure_result(10), lo que ejecuta nuevamente el cierre add con el argumento 10. El cierre suma 10 con el valor de ten (que es 10)
# y devuelve el resultado 20.

# Decoradores Python: Un decorador es un patrón de diseño en Python que permite al usuario añadir nuevas funcionalidades a un objeto existente
# sin modificar su estructura. Los decoradores suelen invocarse antes de la definición de una función que se desea decorar.

# Creación de decoradores: Para crear una función decoradora, necesitamos una función externa con una función envolvente interna.

def greeting():   # (Función normal)
    return 'Welcome to Python'  # (Esta función devuelve la cadena "Welcome a Python.)
def uppercase_decorator(function): # (Esta función toma "function" como argumento y devuelve una nueva función que envuelve la función original y 
    # convierte la salida a mayúsculas)
    def wrapper():
        func = function()
        make_uppercase = func.upper()
        return make_uppercase
    return wrapper
g = uppercase_decorator(greeting)
print(g())          # WELCOME TO PYTHON

# Explicacion codigo:

# 1- El código define dos funciones: greeting() y uppercase_decorator().
# 2- La función greeting() simplemente devuelve la cadena "Welcome to Python".
# 3- La función uppercase_decorator() toma una función como argumento y devuelve una nueva función que envuelve la función original y convierte la salida
# a mayúsculas.
# 4- A la variable g se le asigna el valor de la función uppercase_decorator(), que es a su vez una función que envuelve a la función greeting().
# 5- Cuando se ejecuta la sentencia print(g()), se llama a la función g(), que a su vez llama a la función greeting() y convierte la salida
# a mayúsculas. La salida de la sentencia print() es, por tanto, la cadena "WELCOME TO PYTHON".

# Esta función decoradora es una función de orden superior que toma una función como parámetro. Tomemos el ejemplo anterior con decorador.

def uppercase_decorator(function):
    def wrapper():
        func = function()
        make_uppercase = func.upper()
        return make_uppercase
    return wrapper
@uppercase_decorator
def greeting():  # (Esta función está decorada por la función uppercase_decorator(). Esto significa que la salida de esta función se convertirá
                 #  a mayúsculas cuando sea invocada.)
    return 'Welcome to Python'
print(greeting())   # WELCOME TO PYTHON

# Otro ejemplo

def split_string_decorator(function):
    def wrapper():
        func = function()
        splitted_string = func.split()
        return splitted_string
    return wrapper
@split_string_decorator
def greeting():
    return 'Welcome to Python'
print(greeting())  # ['Welcome', 'to', 'Python']

# Explicacion codigo

# 1- La función split_string_decorator() toma una función como argumento y devuelve una nueva función que envuelve la función
# original y divide la salida en una lista de cadenas. 
# 2- La sintaxis @split_string_decorator se utiliza para aplicar el decorador split_string_decorator()
# a la función greeting(). Esto significa que la función greeting() será envuelta por la función split_string_decorator() cuando sea llamada.
# 3- Cuando se ejecuta la sentencia print(greeting()), se llama a la función greeting(), que a su vez llama a la función split_string_decorator().
# La función split_string_decorator() divide la salida de la función greeting() en una lista de cadenas, que es devuelta por la sentencia print().
# La salida de la sentencia print() es por tanto la lista de cadenas ["Welcome to Python"].

# Aplicación de varios decoradores a una misma función. Tomamos los dos ejemplos anteriores y aplicamos varios decoradores.

# Primer decorador
def uppercase_decorator(function):
    def wrapper():
        func = function()
        make_uppercase = func.upper()
        return make_uppercase
    return wrapper

# Segundo decorador
def split_string_decorator(function):
    def wrapper():
        func = function()
        splitted_string = func.split()
        return splitted_string
    return wrapper

@split_string_decorator
@uppercase_decorator     # (El orden con los decoradores es importante en este caso - la función .upper() no funciona con listas)
def greeting():
    return 'Welcome to Python'
print(greeting())   # ['WELCOME', 'TO', 'PYTHON']

# Aceptar parámetros en funciones de decorador: La mayoría de las veces necesitamos que nuestras funciones acepten parámetros, por lo que podríamos
# necesitar definir un decorador que acepte parámetros.

def decorator_with_parameters(function): # (Define una función llamada decorador_con_parámeters. Toma un argumento, que es otra función)
    def wrapper_accepting_parameters(para1, para2, para3): # (Define una función llamada wrapper_accepting_parameters. Toma tres argumentos, que se denominan para1, para2 y para3.)
        function(para1, para2, para3)  # (Llama a la función que se pasó a decorator_with_parameters. Pasa los tres argumentos que se pasaron a wrapper_accepting_parameters.)
        print("I live in {}".format(para3)) # (Esto imprime un mensaje que dice "I live in {}". El valor de para3 se utiliza para rellenar el espacio en blanco.)
    return wrapper_accepting_parameters  # (Devuelve la función wrapper_accepting_parameters.)
# (Esto le dice a Python que llame a la función decorator_with_parameters y le pase la función print_full_name. El valor de retorno de
# decorator_with_parameters se asigna entonces a la función print_full_name.)
@decorator_with_parameters
def print_full_name(first_name, last_name, country):
# (Esto imprime un mensaje que dice "I am {} {}. I love to teach". Los valores de first_name, last_name y country se utilizan para rellenar los
# espacios en blanco {})
    print("I am {} {}. I love to teach.".format(
        first_name, last_name, country))
# (Esto llama a la función print_full_name y le pasa los valores "Alvaro", "Nunez", y "Colombia".)
print_full_name("Alvaro", "Nunez",'Colombia')  # I am Alvaro Nunez. I love to teach.
                                               # I live in Colombia
                                               
# Funciones incorporadas de orden superior: Algunas de las funciones incorporadas de orden superior que cubrimos en esta parte son map(),
# filter y reduce. La función lambda se puede pasar como parámetro y el mejor caso de uso de las funciones lambda es en funciones como map,
# filter y reduce.

# Python - Función Map: La función map() es una función incorporada que toma una función y un iterable como parámetros. map(function, iterable)

numbers = [1, 2, 3, 4, 5]  # (Iterable)
def square(x):             # (square es la funcion)
    return x ** 2
numbers_squared = map(square, numbers) # (square = funcion, numbers = iterable)
print(list(numbers_squared))    # [1, 4, 9, 16, 25]

# Apliquemos la función lambda para el mismo objetivo del codigo anterior
numbers_squared = map(lambda x : x ** 2, numbers)  # (lambda x : x ** 2 = funcion, numbers = iterable)
print(list(numbers_squared))    # [1, 4, 9, 16, 25]

# otro ejemplo

numbers_str = ['1', '2', '3', '4', '5']  # (Iterable)
numbers_int = map(int, numbers_str)      # (int = funcion, numbers_str = iterable)
print(list(numbers_int))    # [1, 2, 3, 4, 5]

# otro ejemplo

names = ['Alvaro', 'Claudia', 'Nicolas', 'Sebastian']  #  (Iterable)
def change_to_upper(name):
    return name.upper()
names_upper_cased = map(change_to_upper, names)
print(list(names_upper_cased))    #  ['ALVARO', 'CLAUDIA', 'NICOLAS', 'SEBASTIAN']

# Apliquemos la función lambda para el mismo objetivo del codigo anterior

names_upper_cased = map(lambda name: name.upper(), names)
print(list(names_upper_cased))    # ['ALVARO', 'CLAUDIA', 'NICOLAS', 'SEBASTIAN']

# Lo que en realidad hace map es iterar sobre una lista. Por ejemplo, cambia los nombres a mayúsculas y devuelve una nueva lista.

# Python - Función Filtro: La función filter() llama a la función especificada que devuelve un booleano para cada elemento del iterable 
# especificado (lista) Filtra los elementos que satisfacen los criterios de filtrado.  filter(function, iterable)

# Filtremos numeros pares
numbers = [1, 2, 3, 4, 5]  # (Iterable)
def is_even(num):
    if num % 2 == 0:       # (Funcion)
        return True
    return False
even_numbers = filter(is_even, numbers)  # (is_even = funcion, numbers = iterable)
print(list(even_numbers))       # [2, 4]

# Filtremos numeros impares
numbers = [1, 2, 3, 4, 5]  # (Iterable)
def is_odd(num):           # (Funcion)
    if num % 2 != 0:
        return True
    return False
odd_numbers = filter(is_odd, numbers)  # (V = funcion, numbers = iterable)
print(list(odd_numbers))       # [1, 3, 5]

# Filtrar el nombre con mas de 6 caracteres
names = ['Alvaro', 'Claudia', 'Nicolas', 'Sebastian']   # (Iterable)
def is_name_long(name):                                 # (Funcion)
    if len(name) > 6:
        return True
    return False
long_names = filter(is_name_long, names)
print(list(long_names))         # ['Claudia', 'Nicolas', 'Sebastian']

# Python - Función Reduce: La función reduce() está definida en el módulo functools y debemos importarla desde este módulo. Al igual que map y filter
# toma dos parámetros, una función y un iterable. Sin embargo, no devuelve otro iterable, sino un único valor.

from functools import reduce
numbers_str = ['1', '2', '3', '4', '5']   # (Iterable)
def add_two_nums(x, y):                   # (Funcion)
    return int(x) + int(y)
total = reduce(add_two_nums, numbers_str)
print(total)    # 15


# Ejercicios

# Explique la diferencia entre las funciones map, filter y reduce.

# Las funciones map(), filter() y reduce() son funciones de orden superior en Python que se utilizan para manipular y transformar datos de forma
# eficiente. Aunque tienen similitudes en términos de su capacidad para procesar iterables y aplicar funciones a los elementos, se utilizan para
# propósitos diferentes.

# La función map() se utiliza para aplicar una función a cada elemento de un iterable y devuelve un nuevo iterable con los resultados. Toma dos
# argumentos: la función que se aplicará a cada elemento y el iterable al que se aplicará. La función map() recorre el iterable, aplica la función
# a cada elemento y devuelve un nuevo iterable con los resultados. Es útil cuando se quiere aplicar una operación o transformación a todos los
# elementos de un iterable sin necesidad de utilizar bucles explícitos.

# La función filter() se utiliza para filtrar elementos de un iterable según una condición especificada en una función. Toma dos argumentos:
# la función de filtro que define la condición y el iterable al que se aplicará. La función filter() recorre el iterable y devuelve un nuevo
# iterable que contiene # solo los elementos para los cuales la función de filtro devuelve True. Es útil cuando se desea extraer o filtrar elementos
# que cumplan ciertos criterios.

#  La función reduce() se utiliza para aplicar una función acumulativa a los elementos de un iterable, reduciendo el iterable a un solo valor.
# Toma dos argumentos: la función acumulativa y el iterable al que se aplicará. La función reduce() aplica la función acumulativa sucesivamente
# a pares de elementos del iterable, reduciendo gradualmente el iterable a un solo valor. Es útil cuando se desea realizar una operación acumulativa,
# como calcular la suma o el producto de todos los elementos.

# Explicar la diferencia entre función de orden superior, cierre y decorador

# Función de orden superior: Una función de orden superior es aquella que puede recibir una o más funciones como argumentos y/o devolver una función
# como resultado. En Python, las funciones son ciudadanos de primera clase, lo que significa que se pueden tratar como cualquier otro objeto, como
# variables. Las funciones de orden superior permiten escribir código más modular y flexible, ya que se pueden utilizar para abstraer lógica común,
# pasar comportamiento personalizado a través de argumentos de función y crear funciones más generales y reutilizables.

# Cierre (Closure): Un cierre es una función que guarda internamente referencias a variables locales de su ámbito externo, incluso después de que
# ese ámbito externo haya finalizado su ejecución. Esto significa que la función conserva el estado de las variables capturadas incluso cuando se
# llama desde un ámbito diferente. En otras palabras, el cierre encapsula tanto la función como las variables necesarias para su ejecución.
# Los cierres son útiles cuando se necesita preservar un estado específico en una función y se utilizan comúnmente en la programación funcional
# y en la implementación de decoradores.

# Decorador: Un decorador es una función especial que toma otra función como entrada y devuelve una función modificada. Los decoradores se utilizan
# para extender o modificar el comportamiento de una función sin modificar su implementación original. Esto se logra envolviendo la función original
# en otra función que agrega funcionalidad adicional antes, después o alrededor de la función original. Los decoradores se implementan comúnmente
# utilizando cierres y se pueden aplicar a funciones utilizando la sintaxis del símbolo @. Los decoradores son una forma poderosa de reutilizar y
# extender el comportamiento de las funciones en Python.

# Definir una función de llamada antes de map, filter o reduce, ver ejemplos.

def square(x):
    return x ** 2
numbers = [1, 2, 3, 4, 5]
squared_numbers = list(map(square, numbers))
print(squared_numbers)  # [1, 4, 9, 16, 25]

def is_even(x):
    return x % 2 == 0
numbers = [1, 2, 3, 4, 5]
even_numbers = list(filter(is_even, numbers))
print(even_numbers)  #  [2, 4]

from functools import reduce

def add(x, y):
    return x + y
numbers = [1, 2, 3, 4, 5]
sum_of_numbers = reduce(add, numbers)
print(sum_of_numbers)  # 15

# Utilice el bucle for para imprimir cada país de la lista countries.

countries = ['Estonia', 'Finland', 'Sweden', 'Denmark', 'Norway', 'Iceland']
for country in countries:
    print(country)
# Estonia
# Finland
# Sweden
# Denmark
# Norway
# Iceland

# Utilice for para imprimir cada nombre de la lista names.

names = ['Alvaro', 'Claudia','Nicolas', 'Sebastian']
for name in names:
    print(name)
# Alvaro
# Claudia
# Nicolas
# Sebastian

# Utilice for para imprimir cada número de la lista numbers.

numbers = [1, 2, 3, 4, 5, 6]
for number in numbers:
    print(number)
# 1
# 2
# 3
# 4
# 5
# 6

# Utiliza el mapa para crear una nueva lista cambiando cada país a mayúsculas en la lista countries.

countries = ['Estonia', 'Finland', 'Sweden', 'Denmark', 'Norway', 'Iceland']
def upper_country(country):
  return country.upper()
upper_countries = list(map(upper_country, countries))
print(upper_countries)  # ['ESTONIA', 'FINLAND', 'SWEDEN', 'DENMARK', 'NORWAY', 'ICELAND']

# Otra forma:

countries = ['Estonia', 'Finland', 'Sweden', 'Denmark', 'Norway', 'Iceland']
upper_countries = list(map(str.upper, countries))
print(upper_countries)   # ['ESTONIA', 'FINLAND', 'SWEDEN', 'DENMARK', 'NORWAY', 'ICELAND']

# Usa map para crear una nueva lista cambiando cada número por su cuadrado en la lista numbers

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
def square_number(number):
  return number * number
squared_numbers = list(map(square_number, numbers))
print(squared_numbers)  # [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]

# Otra forma

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
def square_number(number):
  return number ** 2
squared_numbers = list(map(square_number, numbers))
print(squared_numbers)  # [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]

# Otra forma

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
squared_numbers = list(map(lambda x: x**2, numbers))
print(squared_numbers)  # [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]

# Utilice funcion map para cambiar cada nombre a mayúsculas en la lista nombres:

nombres = ['Alvaro', 'Claudia', 'Nicolas', 'Sebastian']
def upper_country(nombre):
  return nombre.upper()
upper_nombres = list(map(upper_country, nombres))
print(upper_nombres)    # ['ALVARO', 'CLAUDIA', 'NICOLAS', 'SEBASTIAN']


# Utilice funcion filter para filtrar los países que contienen "land".

def filter_countries(countries):
    filtered_countries = filter(lambda country: 'land' in country, countries)
    return list(filtered_countries)
countries = ['Estonia', 'Finland', 'Sweden', 'Denmark', 'Norway', 'Iceland']
filtered_list = filter_countries(countries)
print(filtered_list)   # ['Finland', 'Iceland']


# Escriba una función que utilice filter para filtrar la lista countries que no contenga la palabra "land".

def filter_countries(countries):
    filtered_countries = filter(lambda country: 'land' not in country, countries)
    return list(filtered_countries)
countries = ['Estonia', 'Finland', 'Sweden', 'Denmark', 'Norway', 'Iceland']
filtered_list = filter_countries(countries)
print(filtered_list)     # ['Estonia', 'Sweden', 'Denmark', 'Norway']

# Otra forma:

def filter_countries(countries):
  return filter(lambda country: 'land' not in country, countries)
if __name__ == '__main__':
  countries = ['Estonia', 'Finland', 'Sweden', 'Denmark', 'Norway', 'Iceland']
  filtered_countries = filter_countries(countries)
  print(list(filtered_countries))   # ['Estonia', 'Sweden', 'Denmark', 'Norway']
  
  # Escriba una función que utilice la función filter para filtrar los países que tengan exactamente seis caracteres.
  
def filter_countries(countries):
    filtered_countries = filter(lambda country: len(country) == 6, countries)
    return list(filtered_countries)
countries = ['Estonia', 'Finland', 'Sweden', 'Denmark', 'Norway', 'Iceland']
filtered_list = filter_countries(countries)
print(filtered_list)   # ['Sweden', 'Norway']

# Escriba una función que utilice la función filter para listar los países que no tengan exactamente seis caracteres.

def filter_countries(countries):
    filtered_countries = filter(lambda country: len(country) != 6, countries)
    return list(filtered_countries)
countries = ['Estonia', 'Finland', 'Sweden', 'Denmark', 'Norway', 'Iceland']
filtered_list = filter_countries(countries)
print(filtered_list)   # ['Estonia', 'Finland', 'Denmark', 'Iceland']

# Escriba una función que utilice la función filter para filtrar los países que tengan seis o más caracteres.

def filter_countries(countries):
    filtered_countries = filter(lambda country: len(country) >= 6, countries)
    return list(filtered_countries)
countries = ['Estonia', 'Finland', 'Sweden', 'Denmark', 'Norway', 'Iceland']
filtered_list = filter_countries(countries)
print(filtered_list)   # ['Estonia', 'Finland', 'Sweden', 'Denmark', 'Norway', 'Iceland']

# Escriba una función que utilice la función filter para filtrar los países que empiezan por "E".

def filter_countries(countries):
    filtered_countries = filter(lambda country: country.startswith('E'), countries)
    return list(filtered_countries)
countries = ['Estonia', 'Finland', 'Sweden', 'Denmark', 'Norway', 'Iceland']
filtered_list = filter_countries(countries)
print(filtered_list)   # ['Estonia']

# Encadenar dos o más iteradores de lista (p. ej. arr.map(callback).filter(callback).reduce(callback))

import itertools

def chain_iterators(*iterators):
    for iterator in iterators:
        yield from iterator

countries = ['Estonia', 'Finland', 'Sweden', 'Denmark', 'Norway', 'Iceland']
filtered_countries = filter(lambda country: 'land' not in country, countries)
capitalized_countries = map(lambda country: country.upper(), filtered_countries)
chained_iterator = chain_iterators(capitalized_countries, filtered_countries)
print(list(chained_iterator))   # ['ESTONIA', 'SWEDEN', 'DENMARK', 'NORWAY']

# Otro ejemplo

from functools import reduce

def chain_iterators(*iterators):
    for iterator in iterators:
        yield from iterator

def double(x):
    return x * 2

def is_even(x):
    return x % 2 == 0

def sum_numbers(x, y):
    return x + y

arr = [1, 2, 3, 4, 5]

result = chain_iterators(map(double, arr), filter(is_even, arr), [reduce(sum_numbers, arr)])
print(list(result))   # [2, 4, 6, 8, 10, 2, 4, 15]

"""
primera transformación se realiza utilizando map y la función double. Aplica la función double a cada elemento de la lista arr, lo que duplica cada número.
Resultado parcial: [2, 4, 6, 8, 10]

La segunda transformación se realiza utilizando filter y la función is_even. Filtra los elementos de la lista arr y se quedan solo los números pares.
Resultado parcial: [2, 4]

La tercera transformación se realiza utilizando reduce y la función sum_numbers. Combina los elementos de la lista arr utilizando la función sum_numbers, que suma dos números.
Resultado parcial: 15
"""
# Otro ejemplo

from functools import reduce

def chain_iterators(iterators):
    result = None
    for iterator in iterators:
        if result is None:
            result = iterator
        else:
            result = iterator(result)
    return result

arr = [1, 2, 3, 4, 5]

def double(x):
    return x * 2

def is_even(x):
    return x % 2 == 0

def sum_numbers(x, y):
    return x + y

result = chain_iterators([map(double, arr), lambda arr: filter(is_even, arr), lambda arr: reduce(sum_numbers, arr)])
print(result)   # 30

"""
primera transformación se realiza utilizando map y la función double. Aplica la función double a cada elemento de la lista arr, lo que duplica cada número.
Resultado parcial: [2, 4, 6, 8, 10]

La segunda transformación se realiza utilizando filter y la función is_even. Filtra los elementos de la lista arr y se quedan solo los números pares.
Resultado parcial: [2, 4].  Pero como estos numeros ya estan en la lista anterior, no los considera, hasta aqui el resultado es [2, 4, 6, 8, 10]

La tercera transformación se realiza utilizando reduce y la función sum_numbers. Combina los elementos de la lista arr utilizando la función sum_numbers, que suma los números.
[2, 4, 6, 8, 10] Resultado final: 30
"""

# Declara una función llamada get_string_lists que toma una lista como parámetro y devuelve una lista que contiene sólo elementos de cadena(strings).

def get_string_lists(lst):
    return [item for item in lst if isinstance(item, str)]
my_list = [1, 'Hello', 3.14, 'World', True, 'Python']
result = get_string_lists(my_list)
print(result)  # ['Hello', 'World', 'Python']

# otro ejemplo

def get_string_lists(list_items):
  string_list = []
  for item in list_items:
    if isinstance(item, str):
      string_list.append(item)
  return string_list
list_items = [1, 'Hello', 3.14, 'World', True, 'Python']
string_list = get_string_lists(list_items)
print(string_list)   # ['Hello', 'World', 'Python']

# Escriba una función que utilice la función reduce para sumar todos los números de la lista de números.

from functools import reduce
def add_two_nums(x, y):                   
    return int(x) + int(y)
numbers_str = ['1', '2', '3', '4', '5']   
total = reduce(add_two_nums, numbers_str)
print(total)    # 15

# Otra forma

from functools import reduce
def sum_numbers(numbers):
    return reduce(lambda x, y: x + y, numbers)
my_list = [1, 2, 3, 4, 5]
result = sum_numbers(my_list)
print(result)  # 15

# Otra froma

from functools import reduce
def add_numbers(numbers):
  def accumulator(current_sum, number):
    return current_sum + number
  return reduce(accumulator, numbers, 0)
numbers = [1, 2, 3, 4, 5]
sum_of_numbers = add_numbers(numbers)
print(sum_of_numbers)   # 15

# Escriba una función que utilice reduce para concatenar todos los países y producir esta frase: "Estonia, Finlandia, Suecia, Dinamarca, Noruega
# e Islandia son países del norte de Europa"

from functools import reduce
def concatenate_countries(countries):
    concatenated_string = reduce(lambda x, y: f"{x}, {y}", countries)
    return f"{concatenated_string} son países del norte de Europa."
country_list = ['Estonia', 'Finlandia', 'Suecia', 'Dinamarca', 'Noruega', 'Islandia']
result = concatenate_countries(country_list)
print(result)  # "Estonia, Finlandia, Suecia, Dinamarca, Noruega, Islandia son países del norte de Europa."

# Otra forma

from functools import reduce
def concatenate_countries(countries):
  def accumulator(current_string, country):
    if current_string == "":
      return country
    else:
      return current_string + ", " + country
  concatenated_countries = reduce(accumulator, countries, "")
  concatenated_countries += " son países del norte de Europa."
  return concatenated_countries
countries = ['Estonia', 'Finlandia', 'Suecia', 'Dinamarca', 'Noruega', 'Islandia']
concatenated_countries = concatenate_countries(countries)
print(concatenated_countries)  # Estonia, Finlandia, Suecia, Dinamarca, Noruega, Islandia son países del norte de Europa.

# Declara una función llamada categorize_countries que devuelve una lista de países con algún patrón común (puedes encontrar la lista de países en
# este repositorio como countries.js(eg 'land', 'ia', 'island', 'stan')).

# Declare a function called categorize_countries that returns a list of countries with some common pattern. Use la lista ubicada en
# "https://raw.githubusercontent.com/Asabeneh/30-Days-Of-Python/master/data/countries-data.py"


import requests

def categorizar_paises():
    url = "https://raw.githubusercontent.com/Asabeneh/30-Days-Of-Python/master/data/countries-data.py"
    response = requests.get(url)
    data = eval(response.content)

    # Ejemplo: categorizar los países según la letra por la que empiecen
    paises_a = [pais['name'] for pais in data if pais['name'].startswith('A')]
    paises_b = [pais['name'] for pais in data if pais['name'].startswith('B')]
    paises_c = [pais['name'] for pais in data if pais['name'].startswith('C')]

    categorias_paises = {
        'Paises que comienzan con A': paises_a,
        'Paises que comienzan con B': paises_b,
        'Paises que comienzan con C': paises_c
        }

    return categorias_paises

categorias = categorizar_paises()
for categoria, paises in categorias.items():
    print(categoria)
    print(paises)
    print()
"""
Paises que comienzan con A
['Afghanistan', 'Albania', 'Algeria', 'American Samoa', 'Andorra', 'Angola', 'Anguilla', 'Antarctica', 'Antigua and Barbuda', 'Argentina', 'Armenia',
'Aruba', 'Australia', 'Austria', 'Azerbaijan']

Paises que comienzan con B
['Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin', 'Bermuda', 'Bhutan', 'Bolivia (Plurinational State of)',
'Bonaire, Sint Eustatius and Saba', 'Bosnia and Herzegovina', 'Botswana', 'Bouvet Island', 'Brazil', 'British Indian Ocean Territory',
'Brunei Darussalam', 'Bulgaria', 'Burkina Faso', 'Burundi']

Paises que comienzan con C
['Cambodia', 'Cameroon', 'Canada', 'Cabo Verde', 'Cayman Islands', 'Central African Republic', 'Chad', 'Chile', 'China', 'Christmas Island',
'Cocos (Keeling) Islands', 'Colombia', 'Comoros', 'Congo', 'Congo (Democratic Republic of the)', 'Cook Islands', 'Costa Rica', 'Croatia', 'Cuba',
'Curaçao', 'Cyprus', 'Czech Republic', "Côte d'Ivoire"]
"""

# Cree una función que devuelva un diccionario, donde las claves sean las letras iniciales de los países y los valores sean el número de nombres
# de países que empiezan por esa letra.

import requests
from collections import defaultdict

def conteo_iniciales_paises():
    url = "https://raw.githubusercontent.com/Asabeneh/30-Days-Of-Python/master/data/countries-data.py"
    response = requests.get(url)
    data = eval(response.content)
    conteo = defaultdict(int)
    for pais in data:
        inicial = pais["name"][0]
        conteo[inicial] += 1
    return dict(conteo)
conteo = conteo_iniciales_paises()
print(conteo)
"""
{'A': 15, 'Å': 1, 'B': 21, 'U': 8, 'V': 5, 'C': 23, 'D': 4, 'E': 7, 'F': 8, 'G': 16, 'H': 6, 'I': 9, 'J': 4, 'K': 7, 'L': 9, 'M': 23, 'N': 13,
 'O': 1, 'P': 12, 'Q': 1, 'R': 5, 'S': 33, 'T': 14, 'W': 2, 'Y': 1, 'Z': 2}
"""

# Declara una función get_first_ten_countries. Esta devuelve una lista de los diez primeros países de la lista ubicada en
# "https://raw.githubusercontent.com/Asabeneh/30-Days-Of-Python/master/data/countries-data.py"

import requests

def get_first_ten_countries():
    url = "https://raw.githubusercontent.com/Asabeneh/30-Days-Of-Python/master/data/countries-data.py"
    response = requests.get(url)
    data = eval(response.content)
    first_ten_countries = []
    for country in data[:10]:
        first_ten_countries.append(country["name"])
    return first_ten_countries
first_ten_countries = get_first_ten_countries()
print(first_ten_countries)  # ['Afghanistan', 'Åland Islands', 'Albania', 'Algeria', 'American Samoa', 'Andorra', 'Angola', 'Anguilla', 'Antarctica', 'Antigua and Barbuda']

# Declara una función get_last_ten_countries que devuelva los diez últimos países de la lista ubicada en
# "https://raw.githubusercontent.com/Asabeneh/30-Days-Of-Python/master/data/countries-data.py"

import requests

def get_last_ten_countries():
    url = "https://raw.githubusercontent.com/Asabeneh/30-Days-Of-Python/master/data/countries-data.py"
    response = requests.get(url)
    data = eval(response.content)
    last_ten_countries = data[-10:]
    country_names = [country["name"] for country in last_ten_countries]
    return country_names
last_ten_countries = get_last_ten_countries()
print(last_ten_countries)  
""""
['Uruguay', 'Uzbekistan', 'Vanuatu', 'Venezuela (Bolivarian Republic of)', 'Viet Nam', 'Wallis and Futuna',
'Western Sahara', 'Yemen', 'Zambia', 'Zimbabwe']
"""

# Escriba una función para ordenar, de manera separada, los países por "name", por "capital", por "population" de la lista ubicada en
# "https://raw.githubusercontent.com/Asabeneh/30-Days-Of-Python/master/data/countries-data.py"

import requests

def get_sorted_country_names():
    url = "https://raw.githubusercontent.com/Asabeneh/30-Days-Of-Python/master/data/countries-data.py"
    response = requests.get(url)
    data = eval(response.content)
    sorted_names = sorted(data, key=lambda country: country['name'])
    country_names = [country['name'] for country in sorted_names]
    return country_names

def get_sorted_capitals():
    url = "https://raw.githubusercontent.com/Asabeneh/30-Days-Of-Python/master/data/countries-data.py"
    response = requests.get(url)
    data = eval(response.content)
    sorted_capitals = sorted(data, key=lambda country: country['capital'])
    capitals = [country['capital'] for country in sorted_capitals]
    return capitals

import requests

def get_sorted_population():
    url = "https://raw.githubusercontent.com/Asabeneh/30-Days-Of-Python/master/data/countries-data.py"
    response = requests.get(url)
    data = eval(response.content)
    sorted_population = sorted(data, key=lambda country: country['population'])
    population = [country['population'] for country in sorted_population]
    return population


# Ejemplo de obtener la lista ordenada alfabéticamente de los nombres de los países
sorted_country_names = get_sorted_country_names()
print(sorted_country_names)

# Ejemplo de obtener la lista ordenada alfabéticamente de las capitales
sorted_capitals = get_sorted_capitals()
print(sorted_capitals)

# Ejemplo de obtener la lista ordenada por población
sorted_population = get_sorted_population()
print(sorted_population)

# Escriba una función que obtenga una lista que clasifique las diez lenguas (languages) más habladas por lugares (population) de la lista ubicada
# en "https://raw.githubusercontent.com/Asabeneh/30-Days-Of-Python/master/data/countries-data.py"

import requests

def get_top_10_languages():
    url = "https://raw.githubusercontent.com/Asabeneh/30-Days-Of-Python/master/data/countries-data.py"
    response = requests.get(url)
    data = eval(response.content)
    languages = [language for country in data for language in country['languages']]
    language_counts = {}
    for language in languages:
        language_counts[language] = language_counts.get(language, 0) + 1
    sorted_languages = sorted(language_counts.items(), key=lambda item: item[1], reverse=True)
    top_10_languages = [(language, count) for language, count in sorted_languages[:10]]
    return top_10_languages

# Ejemplo de obtener la lista de las diez lenguas más habladas
top_10_languages = get_top_10_languages()
print(top_10_languages)

# [('English', 91), ('French', 45), ('Arabic', 25), ('Spanish', 24), ('Portuguese', 9), ('Russian', 9), ('Dutch', 8), ('German', 7),
# ('Chinese', 5), ('Serbian', 4)]

# Obtener una lista ordenada con los diez países "name" más poblados "population" de la lista ubicada en
# "https://raw.githubusercontent.com/Asabeneh/30-Days-Of-Python/master/data/countries-data.py", puedes seguir estos pasos:

import requests

def get_top_10_populated_countries():
    url = "https://raw.githubusercontent.com/Asabeneh/30-Days-Of-Python/master/data/countries-data.py"
    response = requests.get(url)
    data = eval(response.content)

    top_10_countries = sorted(data, key=lambda country: country['population'], reverse=True)[:10]
    top_10_country_names = [country['name'] for country in top_10_countries]

    return top_10_country_names

# Ejemplo de obtener la lista de los diez países más poblados
top_10_countries = get_top_10_populated_countries()
print(top_10_countries)
# ['China', 'India', 'United States of America', 'Indonesia', 'Brazil', 'Pakistan', 'Nigeria', 'Bangladesh', 'Russian Federation', 'Japan']

# Escriba una función que obtenga una lista de los diez países (name) más poblados, con su población (population) de la lista ubicada
# en "https://raw.githubusercontent.com/Asabeneh/30-Days-Of-Python/master/data/countries-data.py"

import requests

def get_top_10_countries_by_population():
    url = "https://raw.githubusercontent.com/Asabeneh/30-Days-Of-Python/master/data/countries-data.py"
    response = requests.get(url)
    data = eval(response.content)
    top_10_countries = sorted(data, key=lambda country: country['population'], reverse=True)[:10]
    top_10_country_info = [(country['name'], country['population']) for country in top_10_countries]
    return top_10_country_info

# Ejemplo de obtener la lista de los diez países más poblados con su población
top_10_countries = get_top_10_countries_by_population()
print(top_10_countries)
# [('China', 1377422166), ('India', 1295210000), ('United States of America', 323947000), ('Indonesia', 258705000), ('Brazil', 206135893),
# ('Pakistan', 194125062), ('Nigeria', 186988000), ('Bangladesh', 161006790), ('Russian Federation', 146599183), ('Japan', 126960000)]

# otra forma

import requests

def get_top_10_countries_by_population():
    url = "https://raw.githubusercontent.com/Asabeneh/30-Days-Of-Python/master/data/countries-data.py"
    response = requests.get(url)
    data = eval(response.content)
    top_10_countries = sorted(data, key=lambda country: country['population'], reverse=True)[:10]
    top_10_country_info = [(country['name'], country['population']) for country in top_10_countries]
    return top_10_country_info

# Ejemplo de obtener la lista de los diez países más poblados con su población
top_10_countries = get_top_10_countries_by_population()
for country, population in top_10_countries:
    print(country, population)
"""
China 1377422166
India 1295210000
United States of America 323947000
Indonesia 258705000
Brazil 206135893
Pakistan 194125062
Nigeria 186988000
Bangladesh 161006790
Russian Federation 146599183
Japan 126960000
"""