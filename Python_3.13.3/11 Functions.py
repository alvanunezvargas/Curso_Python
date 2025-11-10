# Funciones
# Hasta ahora hemos visto muchas funciones incorporadas en Python. En esta sección, nos centraremos en las funciones personalizadas. ¿Qué es una función?
# Antes de empezar a crear funciones, aprendamos qué es una función y por qué las necesitamos.

# Definición de una función
# Una función es un bloque reutilizable de código o sentencias de programación diseñado para realizar una determinada tarea. Para definir o declarar una
# función, Python proporciona la palabra clave def. La siguiente es la sintaxis para definir una función. El bloque de código de la función se ejecuta sólo
# si la función es llamada o invocada.

# Declarar y llamar a una función
# Cuando creamos una función, la llamamos declarando una función. Cuando empezamos a usarla, la llamamos funcion calling o invoking. Una función puede
# declararse con o sin parámetros.

# def nombre_funcion():
#    códigos
#    códigos
# Llamar a una función
# nombre_funcion()

# Función sin parámetros
# Una función puede declararse sin parámetros.

def generate_full_name ():
    first_name = 'Alvaro'
    last_name = 'Nunez'
    space = ' '
    full_name = first_name + space + last_name
    print(full_name)  # Alvaro Nunez
generate_full_name () # llamar a una función

def add_two_numbers ():
    num_one = 2
    num_two = 3
    total = num_one + num_two
    print(total)  # 5
add_two_numbers()

# Función que Devuelve un Valor - Parte 1
# Las funciones también pueden devolver valores, si una función no tiene una sentencia return, el valor de la función es Ninguno. Reescribamos las funciones
# anteriores utilizando return. De ahora en adelante, obtendremos un valor de una función cuando llamemos a la función y lo imprimamos.

def generate_full_name ():
    first_name = 'Alvaro'
    last_name = 'Nunez'
    space = ' '
    full_name = first_name + space + last_name
    return full_name
print(generate_full_name())  # Alvaro Nunez

def add_two_numbers ():
    num_one = 2
    num_two = 3
    total = num_one + num_two
    return total
print(add_two_numbers())  # 5

# Función con parámetros
# En una función podemos pasar diferentes tipos de datos como parametro (número, cadena, booleano, lista, tupla, diccionario o conjunto)

# Parámetro único: Si nuestra función toma un parámetro debemos llamar a nuestra función con un argumento

# def function_name(parameter):
#    codigo
#    codigo
#    llamando funcion
#  print(function_name(argument))

def greetings (name):
    message = name + ', Todos bienvenidos a Phython!'
    return message
print(greetings('Alvaro'))  # Alvaro, Todos bienvenidos a Phython!

def add_ten(num):
    ten = 10
    return num + ten
print(add_ten(90))  # 100

def square_number(x):
    return x * x
print(square_number(2)) # 4

def area_of_circle (r):
    PI = 3.14
    area = PI * r ** 2
    return area
print(area_of_circle(10))  # 314.0

def sum_of_numbers(n):
    total = 0
    for i in range(n+1):
        total+=i
    print(total)
print(sum_of_numbers(10))  # 55  (suma los numeros desde 0 hastas 10)
print(sum_of_numbers(100)) # 5050 (suma los numeros desde 0 hastas 100)

# Dos parámetros: Una función puede o no tener un parámetro o parámetros. Una función también puede tener dos o más parámetros. Si nuestra función toma
# parámetros debemos llamarla con argumentos. Comprobemos una función con dos parámetros:

# def nombre_funcion(para1, para2):
#     códigos
#     códigos
#   # Llamada a la función
#   print(nombre_funcion(arg1, arg2))

def generate_full_name (first_name, last_name):
    space = ' '
    full_name = first_name + space + last_name
    return full_name
print('Nombre completo: ', generate_full_name('Alvaro','Nunez'))  # Nombre completo:  Alvaro Nunez

def sum_two_numbers (num_one, num_two):
    sum = num_one + num_two
    return sum
print('La suma de los dos numeros es: ', sum_two_numbers(1, 9)) # La suma de los dos numeros es:  10

def calculate_age (current_year, birth_year):
    age = current_year - birth_year
    return age;
print('Edad: ', calculate_age(2023, 1967)) # Edad:  56

def weight_of_object (mass, gravity):
    weight = str(mass * gravity)+ ' N' # primero hay que cambiar el valor a una cadena
    return weight
print('Peso de un objeto en newtons: ', weight_of_object(100, 9.81)) # Peso de un objeto en newtons:  981.0 N

# Pasar argumentos con clave y valor
# Si pasamos los argumentos con clave y valor, el orden de los argumentos no importa.

# def nombre_funcion(para1, para2):
#     códigos
#     códigos
# # Llamada a la función
# print(nombre_funcion(para1 = 'Juan', para2 = 'Doe')) # el orden de los argumentos no importa aquí

def print_fullname(firstname, lastname):
    space = ' '
    full_name = firstname  + space + lastname
    print(full_name)
print_fullname(firstname='Alvaro', lastname='Nunez') # Alvaro Nunez

def add_two_numbers (num1, num2):
    total = num1 + num2
    print(total)
add_two_numbers(num2 = 3, num1 = 2) # 5

# Función que devuelve un valor - Parte 2
# Si no devolvemos un valor con una función, entonces nuestra función está devolviendo None por defecto. Para devolver un valor con una función utilizamos
# la palabra clave return seguida de la variable que estamos devolviendo. Podemos devolver cualquier tipo de datos desde una función.

# Devuelve una cadena:

def print_name(firstname):
    return firstname
print(print_name('Alvaro'))  # Alvaro

def print_full_name(firstname, lastname):
    space = ' '
    full_name = firstname  + space + lastname
    return full_name
print(print_full_name(firstname='Alvaro', lastname='Nunez'))  # Alvaro Nunez

# Devolver un número:

def add_two_numbers (num1, num2):
    total = num1 + num2
    return total
print(add_two_numbers(2, 3)) # 5

def add_numbers(a, b):
    return a + b
print(add_numbers(5, 9)) # 14

# Devolver un booleano

def is_even (n):
    if n % 2 == 0:
        print('even')  # even
        return True    # (return detiene la ejecución de la función, de forma similar a break)
    return False
print(is_even(10)) # True
print(is_even(7)) # False

# Devolución de una lista

def find_even_numbers(n):
    evens = []
    for i in range(n + 1):
        if i % 2 == 0:
            evens.append(i)
    return evens
print(find_even_numbers(10))  # [0, 2, 4, 6, 8, 10]

# Función con parámetros por defecto
# A veces pasamos valores por defecto a los parámetros, cuando invocamos la función. Si no pasamos argumentos al llamar a la función, se utilizarán sus
# valores por defecto.

# def nombre_funcion(param = valor):
#    códigos
#    códigos
# Llamada a la función
# nombre_funcion()
# nombre_funcion(arg)

def greetings (name = 'Alvaro'):
    message = name + ', Bienvenidos a Python!'
    return message
print(greetings())             # Alvaro, Bienvenidos a Python!
print(greetings('Sebastian'))  # Sebastian, Bienvenidos a Python!

def generate_full_name (first_name = 'Alvaro', last_name = 'Nunez'):
    space = ' '
    full_name = first_name + space + last_name
    return full_name
print(generate_full_name())       # Alvaro Nunez
print(generate_full_name('Nicolas','Nunez'))  # Nicolas Nunez

def calculate_age (birth_year,current_year = 2023):
    age = current_year - birth_year
    return age;
print('Edad: ', calculate_age(1967))   # Edad:  56

def weight_of_object (mass, gravity = 9.81):
    weight = str(mass * gravity)+ ' N' # (primero hay que cambiar el valor a cadena)
    return weight
print('Peso de un objeto en newtons: ', weight_of_object(100))         # Peso de un objeto en newtons:  981.0 N
print('Peso de un objeto en newtons: ', weight_of_object(100, 1.62))   # Peso de un objeto en newtons:  162.0 N

# Número arbitrario de argumentos
# Si no sabemos el número de argumentos que pasamos a nuestra función, podemos crear una función que pueda tomar un número arbitrario de argumentos añadiendo
# * antes del nombre del parámetro.

# def nombre_funcion(*args):
#    códigos
#    códigos
#  Llamada a la función
# nombre_funcion(param1, param2, param3,..)

def sum_all_nums(*nums):
    total = 0
    for num in nums:
        total += num    
    return total
print(sum_all_nums(2, 3, 5)) # 10

# Número predeterminado y arbitrario de parámetros en las funciones

def generate_groups (team,*args):
    print(team)
    for i in args:
        print(i)
generate_groups('Team-1','Asabeneh','Brook','David','Eyob')
# Team-1
# Asabeneh
# Brook
# David
# Eyob

# Función como parámetro de otra función

def square_number (n):
    return n * n
def do_something(f, x):
    return f(x)
print(do_something(square_number, 3)) # 9


# Ejercicios

# Declara una función add_two_numbers. Toma dos parámetros y devuelve una suma.

def add_two_numbers(a, b):
    return a + b
result = add_two_numbers(3, 5)
print(result)  # 8

# El área de un círculo se calcula del siguiente modo: área = π x r x r. Escribe una función que calcule el area_del_circulo.

def area_of_circle (r):
    PI = 3.14
    area = PI * r * r
    return area
print(area_of_circle(10))  # 314.0

# otra forma:

import math

def area_del_circulo(radio):
    area = math.pi * radio ** 2
    return area
print(area_del_circulo(5)) # 78.53981633974483

# Escribe una función llamada add_all_nums que tome un número arbitrario de argumentos y sume todos los argumentos. Comprueba si todos los elementos de
# la lista son de tipo numérico. Si no es así da una respuesta razonable

def add_all_nums(*args):
    total = 0
    for num in args:
        if isinstance(num, (int, float, complex)):
            total += num
        else:
            return "Error: Los argumentos deben ser numéricos."
    return total
print(add_all_nums(1, 2, 3, 4))  # 10
print(add_all_nums(1, 2, '3', 4))  # Error: Los argumentos deben ser numéricos.

# La temperatura en °C puede convertirse a °F mediante esta fórmula: °F = (°C x 9/5) + 32. Escribe una función que convierta °C a °F,

def convert_celsius_to_fahrenheit(celsius):
    fahrenheit = (celsius * 9/5) + 32
    return fahrenheit
print(convert_celsius_to_fahrenheit(0))   # 32.0
print(convert_celsius_to_fahrenheit(100)) # 212.0
print(convert_celsius_to_fahrenheit(20))  # 68.0

# Escribe una función llamada comprobar-estación, toma un parámetro de mes y devuelve la estación: Otoño, Invierno, Primavera o Verano.

def comprobar_estacion(mes):
    if mes in [12, 1, 2]:
        return "Invierno"
    elif mes in [3, 4, 5]:
        return "Primavera"
    elif mes in [6, 7, 8]:
        return "Verano"
    elif mes in [9, 10, 11]:
        return "Otoño"
    else:
        return "El mes ingresado no es válido"
print(comprobar_estacion(10))  # Otoño
print(comprobar_estacion(14))  # El mes ingresado no es válido

# Escribe una función llamada calcular_pendiente que devuelva la pendiente de una ecuación lineal

def calcular_pendiente(x1, y1, x2, y2):
    pendiente = (y2 - y1) / (x2 - x1)
    return pendiente
print(calcular_pendiente(0,0,3,6))  # 2.0

# La ecuación cuadrática se calcula del siguiente modo: ax² + bx + c = 0. Escriba una función que calcule el conjunto solución de una ecuación cuadrática,
# solve_quadratic_eqn.

def solve_quadratic_eqn(a, b, c):
    discriminante = b**2 - 4*a*c
    if discriminante > 0:
        raiz_1 = (-b + discriminante**(1/2)) / (2*a)
        raiz_2 = (-b - discriminante**(1/2)) / (2*a)
        return raiz_1, raiz_2
    elif discriminante == 0:
        raiz = -b / (2*a)
        return raiz
    else:
        return "No hay raíces reales"
print(solve_quadratic_eqn(1, -4, 4))  # 2.0
print(solve_quadratic_eqn(1, -7, 10))  # (5.0, 2.0)
print(solve_quadratic_eqn(1, 2, 5))  # No hay raíces reales

# Otra forma: 
    
import math

def solve_quadratic_eqn(a, b, c):
    discriminante = b**2 - 4*a*c
    if discriminante > 0:
        raiz_1 = (-b + math.sqrt(discriminante)) / (2*a)
        raiz_2 = (-b - math.sqrt(discriminante)) / (2*a)
        return raiz_1, raiz_2
    elif discriminante == 0:
        raiz = -b / (2*a)
        return raiz
    else:
        return "No hay raíces reales"
print(solve_quadratic_eqn(1, -4, 4))  # 2.0
print(solve_quadratic_eqn(1, -7, 10))  # (5.0, 2.0)
print(solve_quadratic_eqn(1, 2, 5))  # No hay raíces reales

# Declara una función llamada print_list. Toma una lista como parámetro e imprime cada elemento de la lista.

def print_list(lista):
  for elemento in lista:
    print(elemento)
lista = ["apple", "banana", "cherry"]
print_list(lista)
# apple
# banana
# cherry

# Declara una función llamada lista_inversa. Toma un array como parámetro y devuelve el inverso del array (usa bucles).

def reverse_list(lista):
    reversed_list = []
    for i in range(len(lista)-1, -1, -1):
        reversed_list.append(lista[i])
    return reversed_list
lista = [1, 2, 3, 4, 5]
reversed_list = reverse_list(lista)
print(reversed_list)  # [5, 4, 3, 2, 1]

# Ejemplo de impresion inversa por columna

def lista_inversa(lista):
    invertida = []
    for i in range(len(lista)-1, -1, -1):
        invertida.append(lista[i])
    return invertida
lista = ["apple", "banana", "cherry"]
invertida = lista_inversa(lista)
for fruta in invertida:
    print(fruta)
# cherry
# banana
# apple

# Declare una función llamada capitalize_list_items. Toma una lista como parámetro y devuelve una lista de elementos en mayúsculas

def capitalize_list_items(lst):
    capitalized_lst = []
    for item in lst:
        capitalized_lst.append(item.upper())
    return capitalized_lst
my_list = ['apple', 'banana', 'cherry']
capitalized_list = capitalize_list_items(my_list)
print(capitalized_list)  # ['APPLE', 'BANANA', 'CHERRY']

# Declara una función llamada add_item. Toma como parámetros la lista "food_staff = ['Potato', 'Tomato', 'Mango', 'Milk']" y el  elemento "Meat".
# Devuelve una lista con el elemento añadido al final.

def add_item(food_staff, item):
    food_staff.append(item)
    return food_staff
food_staff = ['Potato', 'Tomato', 'Mango', 'Milk']
new_list = add_item(food_staff, 'Meat')
print(new_list)  # ['Potato', 'Tomato', 'Mango', 'Milk', 'Meat']

# Declara una función llamada add_item. Toma como parámetros la lista "numbers = [2, 3, 7, 9]" y el  elemento "5". Devuelve una lista con el elemento
# añadido al final.

def add_item(numbers, item):
    numbers.append(item)
    return numbers
numbers = [2, 3, 7, 9]
new_list = add_item(numbers, 5)
print(new_list)  # [2, 3, 7, 9, 5]

# Declara una función llamada remove_item. Toma como parámetros la lista "food_staff = ['Potato', 'Tomato', 'Mango', 'Milk']" y el elemento "Mango".
# Devuelve una lista con el elemento eliminado.

def remove_item(food_staff, item):
    if item in food_staff:
        food_staff.remove(item)
    return food_staff
food_staff = ['Potato', 'Tomato', 'Mango', 'Milk']
new_list = remove_item(food_staff, 'Mango')
print(new_list)  # ['Potato', 'Tomato', 'Milk']

# Declara una función llamada remove_item. Toma como parámetros la lista "numbers = [2, 3, 7, 9]" y el elemento "3". Devuelve una lista con el elemento
# eliminado.

def remove_item(numbers, item):
    if item in numbers:
        numbers.remove(item)
    return numbers
numbers = [2, 3, 7, 9]
new_list = remove_item(numbers, 3)
print(new_list) # [2, 7, 9]

# Declara una función llamada suma_de_números. Toma un parámetro numérico y suma todos los números de ese rango.

def suma_de_números(n):
    total = 0
    for i in range(1, n+1):
        total += i
    return total
resultado = suma_de_números(10)
print(resultado)   # 55

def suma_de_números(n):
    total = 0
    for i in range(2, n+1):
        total += i
    return total
resultado = suma_de_números(10)
print(resultado)   # 54

def suma_de_números(n):
    total = 0
    for i in range(5, n+1): # (Va desde 5 hasta n+1)
        total += i
    return total
resultado = suma_de_números(10) # (Son 10 numeros, es decir desde 0 hasta 9, por tanto n+1=10)
print(resultado)   # 45

def suma_de_números(n):
    total = 0
    for i in range(1, n+2):  # (Va desde 1 hasta n+2)
        total += i
    return total
resultado = suma_de_números(10)  # (Son 10 numeros, es decir desde 0 hasta 9, por tanto n+2=11)
print(resultado)   # 66

# Declara una función llamada suma_de_impares. Toma un parámetro numérico y suma todos los números impares de ese rango.

def suma_de_impares(n):
    total = 0
    for i in range(1, n + 1):
        if i % 2 != 0:  # Verificar si el número es impar
            total += i
    return total
resultado = suma_de_impares(10)
print(resultado)  # 25

# Declara una función llamada suma_de_pares. Toma un parámetro numérico y suma todos los números pares en ese - rango.

def suma_de_pares(n):
    total = 0
    for i in range(1, n + 1):
        if i % 2 == 0:  # Verificar si el número es impar
            total += i
    return total
resultado = suma_de_pares(10)
print(resultado)  # 30

# Declare una función llamada pares_e_impares. Tome un parámetro numérico y cuente todos los números pares e impares de ese rango.

def pares_impares(n):
    pares = 0
    impares = 0
    for i in range(1, n + 1):
        if i % 2 == 0:
            pares += 1
        else:
            impares += 1
    return pares, impares
resultado = pares_impares(100)
print(resultado)  # (50, 50) (50 pares y 50 impares)

# Declare una función llamada pares_e_impares. Tome un parámetro numérico y sume todos los números pares e impares de ese rango.

def pares_e_impares(n):
    suma_pares = 0
    suma_impares = 0
    for num in range(1, n+1):
        if num % 2 == 0:
            suma_pares += num
        else:
            suma_impares += num
    return suma_pares, suma_impares
resultado = pares_e_impares(10)
print(resultado)  # (30, 25)  (suma de pares, suma de impares)

# Llama a tu función factorial, toma un número entero como parámetro y devuelve un factorial del número

def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)
numero = 5
resultado = factorial(numero)
print(resultado)  # 120 (Resultado de 5 x 4 x 3 x2 x 1)

# Llama a tu función is_empty, toma un parámetro y comprueba si está vacío o no

def is_empty(parametro):
    if not parametro:
        return True
    else:
        return False
cadena_vacia = ""
cadena_no_vacia = "Hola"
resultado_vacio = is_empty(cadena_vacia)
print(resultado_vacio)  # True (la cadena está vacía)
resultado_no_vacio = is_empty(cadena_no_vacia)
print(resultado_no_vacio)  # False (la cadena no está vacía)

# Escribe diferentes funciones que tomen listas. Calcular promedio, mediana, desviacion estandar, varianza, moda y rango

# Calcular el promedio:

def calcular_promedio(lista):
    suma = sum(lista)
    promedio = suma / len(lista)
    return promedio
lista = [2, 4, 6, 8 ,10, 12, 14 ,16]
resultado = calcular_promedio(lista)
print(resultado)   # 9.0

# Calcular la mediana:

def calcular_mediana(lista):
    lista_ordenada = sorted(lista)
    longitud = len(lista_ordenada)
    if longitud % 2 == 0:
        medio1 = lista_ordenada[longitud // 2 - 1]
        medio2 = lista_ordenada[longitud // 2]
        mediana = (medio1 + medio2) / 2
    else:
        mediana = lista_ordenada[longitud // 2]
    return mediana
lista = [2, 4, 6, 8 ,10, 12, 14 ,16]
resultado = calcular_mediana(lista)
print(resultado)   # 9.0

# Calcular la desviación estándar:

import math

def calcular_desviacion_estandar(lista):
    suma = sum(lista)
    promedio = suma / len(lista)
    promedio = suma / len(lista)
    suma_cuadrados = sum((x - promedio) ** 2 for x in lista)
    desviacion_estandar = math.sqrt(suma_cuadrados / len(lista))
    return desviacion_estandar
lista = [2, 4, 6, 8 ,10, 12, 14 ,16]
resultado = calcular_desviacion_estandar(lista)
print(resultado)   #  4.58257569495584

# Calcular varianza

def calcular_varianza(lista):
    suma =sum(lista)
    promedio = suma / len(lista)
    suma_cuadrados = sum((x - promedio) ** 2 for x in lista)
    varianza = suma_cuadrados / len(lista)
    return varianza
lista = [2, 4, 6, 8 ,10, 12, 14 ,16]
resultado = calcular_varianza(lista)
print(resultado)   # 21.0

# Calcular la moda

from collections import Counter

def calcular_moda(lista):
    contador = Counter(lista)
    moda = contador.most_common(1)[0][0]
    return moda
lista = [2, 4, 2, 6, 4, 6, 8, 6, 8 ,10, 12, 14 ,16]
resultado = calcular_moda(lista)
print(resultado)   # 6

# Calcular el rango de la lista

def calcular_rango(lista):
    rango = max(lista) - min(lista)
    return rango
lista = [2, 4, 6, 8 ,10, 12, 14 ,16]
resultado = calcular_rango(lista)
print(resultado)   # 14

# Escribe una función llamada es_primo, que compruebe si un número es primo.

def es_primo(num):
    if num == 2 or num == 3:
        return True
    if num < 2 or num % 2 == 0:
        return False
    for n in range(3, int(num**0.5)+1, 2):
        if num % n == 0:
            return False
    return True
resultado = es_primo(43)
print(resultado)  # True

# Escribe una función que compruebe si todos los elementos son únicos en la lista.

def all_unique(list):
    return len(list) == len(set(list))
list = [1, 2, 3, 4]
resultado = all_unique(list)
print(resultado)   # True

# Otra forma 

def all_unique(list):
    return len(list) == len(set(list))
list = [1, 2, 3, 4]
if all_unique(list):
  print("Todos los elementos de la lista son únicos.")  # Todos los elementos de la lista son únicos.
else:
  print("Todos los elementos de la lista no son únicos.")

# Escribe una función que compruebe si todos los elementos de la lista son del mismo tipo de datos. 
    
  def same_data_type(lst):
    return all(isinstance(x, type(lst[0])) for x in lst)
lst = [5, "3", 'python', 23, 55, 34]
resultado= same_data_type(lst)
print(resultado)  # False

# Otra forma

def same_data_type(lst):
    return all(isinstance(x, type(lst[0])) for x in lst)
lst = [5, "3", 'python', 23, 55, 34]
if same_data_type(lst):
  print("Todos los elementos de la lista son los mismos tipos de datos.")  
else:
  print("Todos los elementos de la lista no son los mismos tipos de datos.")   # Todos los elementos de la lista no son los mismos tipos de datos.

# Otra forma

def is_same_data_type(list):
  first_item = list[0]
  data_type = type(first_item)
  for item in list:
    if type(item) != data_type:
      return False
  return True
list = [5, "3", 'python', 23, 55, 34]
resultado = is_same_data_type(list)
print(resultado)  # False

# Otra forma

def is_same_data_type(list):
  first_item = list[0]
  data_type = type(first_item)
  for item in list:
    if type(item) != data_type:
      return False
  return True
list = [5, "3", 'python', 23, 55, 34]
if is_same_data_type(list):
  print("Todos los elementos de la lista son los mismos tipos de datos.")  
else:
  print("Todos los elementos de la lista no son los mismos tipos de datos.")   # Todos los elementos de la lista no son los mismos tipos de datos.

# otra forma

def comprobar_tipo_lista(lista):
    if len(lista) == 0:   # (len(lista) devuelve la longitud de la lista si es ==0, esta vacia.)
        return True
    tipo_dato = type(lista[0])
    for elemento in lista:
        if type(elemento) != tipo_dato:
            return False
    return True
lista1 = [1, 2, 3, 4, 5]
lista2 = [1, 2, '3', 4, 5]
lista3 = ['a', 'b', 'c', 'd']
lista4 = []  
print(comprobar_tipo_lista(lista1))  # True
print(comprobar_tipo_lista(lista2))  # False
print(comprobar_tipo_lista(lista3))  # True
print(comprobar_tipo_lista(lista4))  # True

# Escribe una función que compruebe si la variable proporcionada es una variable python válida

def is_valid_variable(variable):
    if not isinstance(variable, str):  # (Comprueba si la variable es una cadena (string))
        return False
    if variable in dir(__builtins__):  # (Comprueba si la variable es una palabra reservada de Python)
        return False
    if variable.isidentifier(): # (Comprueba si la variable es un identificador válido)
        return True
    return False
print(is_valid_variable('hello'))  # True
print(is_valid_variable('123'))  # False (no es un identificador válido)
print(is_valid_variable('if'))  # False (es una palabra reservada)
print(is_valid_variable('my_variable'))  # True

# Ve a la carpeta data y accede al archivo countries-data.py.

# Crea una función llamada idiomas_más_hablados_del_mundo. Debe devolver 10 idiomas más hablados en el mundo en orden descendente

import requests

def idiomas_más_hablados_del_mundo():
    url = "https://raw.githubusercontent.com/Asabeneh/30-Days-Of-Python/master/data/countries-data.py"
    response = requests.get(url)
    data = eval(response.content)
    language_count = {}
    for country in data:
        languages = country["languages"]
        for language in languages:
            if language in language_count:
                language_count[language] += 1
            else:
                language_count[language] = 1
    top_languages = sorted(language_count.items(), key=lambda x: x[1], reverse=True)[:10]
    return [language[0] for language in top_languages]
result = idiomas_más_hablados_del_mundo()
print(result)  # ['English', 'French', 'Arabic', 'Spanish', 'Portuguese', 'Russian', 'Dutch', 'German', 'Chinese', 'Serbian']

# Crea una función llamada países_más_poblados. Debería devolver los 10 o 20 países más poblados en orden descendente.

import requests

def países_más_poblados():
    url = "https://raw.githubusercontent.com/Asabeneh/30-Days-Of-Python/master/data/countries-data.py"
    response = requests.get(url)
    data = eval(response.content)
    sorted_countries = sorted(data, key=lambda x: x["population"], reverse=True)
    top_countries = sorted_countries[:10]
    return [country["name"] for country in top_countries]
result = países_más_poblados()
print(result)  # ['China', 'India', 'United States of America', 'Indonesia', 'Brazil', 'Pakistan', 'Nigeria', 'Bangladesh', 'Russian Federation', 'Japan']