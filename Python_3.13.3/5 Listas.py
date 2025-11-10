# Hay cuatro tipos de datos de colección en Python :

# Lista (List): es una colección ordenada y modificable e indexada. Permite miembros duplicados.
# Tuple: es una colección ordenada y no modificable (inmutable). Permite miembros duplicados.
# Conjunto (Set): es una colección no ordenada, no indexada y no modificable, pero podemos añadir nuevos elementos al conjunto. No permite miembros duplicados.
# Diccionario (Dictionary): es una colección desordenada, modificable e indexada. No admite miembros duplicados.

# Una lista es una colección de diferentes tipos de datos ordenada y modificable. Una lista puede estar vacía o tener elementos de diferentes tipos de datos.

# Cómo crear una lista: En python podemos crar listas de dos formas:

# 1 Uso de la función integrada de lista: list()
lst = list() 
empty_list = list() # esta es una lista vacía, no hay ningún elemento en la lista
print(len(empty_list)) # 0

# 2 Utilizando corchetes, []: list[]
empty_list = [] # esta es una lista vacía, no hay ningún elemento en la lista
print(len(empty_list)) # 0

# Listas con valores iniciales. Usamos len() para encontrar la longitud de una lista.
fruits = ['banana', 'orange', 'mango', 'lemon']                     # (lista de frutas)
vegetables = ['Tomato', 'Potato', 'Cabbage','Onion', 'Carrot']      # (lista de vegetales)
animal_products = ['milk', 'meat', 'butter', 'yoghurt']             # (lista de productos animales)
web_techs = ['HTML', 'CSS', 'JS', 'React','Redux', 'Node', 'MongDB'] # (list of tecnologias web)
countries = ['Finland', 'Estonia', 'Denmark', 'Sweden', 'Norway']   # (Lista de paises)
# Imprime las listas y su longitud
print('Fruits:', fruits) # Fruits: ['banana', 'orange', 'mango', 'lemon']
print('Number of fruits:', len(fruits)) # Number of fruits: 4
print('Vegetables:', vegetables) # Vegetables: ['Tomato', 'Potato', 'Cabbage', 'Onion', 'Carrot']
print('Number of vegetables:', len(vegetables)) # Number of vegetables: 5
print('Animal products:',animal_products) # Animal products: ['milk', 'meat', 'butter', 'yoghurt']
print('Number of animal products:', len(animal_products)) # Number of animal products: 4
print('Web technologies:', web_techs) # Web technologies: ['HTML', 'CSS', 'JS', 'React', 'Redux', 'Node', 'MongDB']
print('Number of web technologies:', len(web_techs)) # Number of web technologies: 7
print('Countries:', countries) # Countries: ['Finland', 'Estonia', 'Denmark', 'Sweden', 'Norway']
print('Number of countries:', len(countries)) # Number of countries: 5

# Las listas pueden tener elementos de distintos tipos de datos
lst = ['Asabeneh', 250, True, {'country':'Finland', 'city':'Helsinki'}]

# Acceso a los elementos de una lista mediante indexación positiva: Accedemos a cada elemento de una lista utilizando su índice. El índice de una 
# lista comienza en 0. La siguiente imagen muestra claramente dónde comienza el índice

# ['banana',   'orange'   'mango'   'lemon']
#     0           1          2         3

fruits = ['banana', 'orange', 'mango', 'lemon']
first_fruit = fruits[0] # (accedemos al primer elemento utilizando su índice)
print(first_fruit)      # banana
second_fruit = fruits[1]
print(second_fruit)     # orange
last_fruit = fruits[3]
print(last_fruit) # lemon

# Otra forma de aceder a la ultma fruta
last_index = len(fruits) - 1
last_fruit1 = fruits[last_index]
print(last_fruit1) # lemon


# Acceso a los elementos de la lista mediante la indexación negativa
# La indexación negativa significa que, empezando por el final, -1 se refiere al último elemento y -2 al penúltimo.

# ['banana',   'orange'   'mango'   'lemon']
#     -4         -3         -2        -1

# Desembalar elementos de la lista
lst = ['item1','item2','item3', 'item4', 'item5']
first_item, second_item, third_item, *rest = lst
print(first_item)     # item1
print(second_item)    # item2
print(third_item)     # item3
print(rest)           # ['item4', 'item5']

fruits = ['banana', 'orange', 'mango', 'lemon','lime','apple']
first_fruit, second_fruit, third_fruit, *rest = fruits
print(first_fruit)     # banana
print(second_fruit)    # orange
print(third_fruit)     # mango
print(rest)            # ['lemon', 'lime', 'apple']

first, second, third,*rest, tenth = [1,2,3,4,5,6,7,8,9,10]
print(first)          # 1
print(second)         # 2
print(third)          # 3
print(rest)           # [4,5,6,7,8,9]
print(tenth)          # 10

countries = ['Germany', 'France','Belgium','Sweden','Denmark','Finland','Norway','Iceland','Estonia']
gr, fr, bg, sw, *scandic, es = countries
print(gr)          # Germany
print(fr)          # France
print(bg)          # Belgium
print(sw)          # Sweden
print(scandic)     # ['Denmark', 'Finland', 'Norway', 'Iceland']
print(es)          # Estonia

# Cortar elementos de una lista 
# Indexación positiva: Podemos especificar un rango de índices positivos especificando el inicio, fin y paso, el valor de retorno será una nueva lista.
# (valores por defecto para inicio = 0, fin = len(lst) - 1 (último elemento), paso = 1)
fruits = ['banana', 'orange', 'mango', 'lemon']
all_fruits = fruits[0:4] # (Toma todos los valores desde indice 0 hasta 4 (indexacion positiva))
print(all_fruits)        # ['banana', 'orange', 'mango', 'lemon']
all_fruits1 = fruits[0:] # (arranca desde la primera fruta o indice 0 y toma todo lo demás)
print(all_fruits1)       # ['banana', 'orange', 'mango', 'lemon']
all_fruits2 = fruits[1:3] # (toma los valores entre indice 1 y 3, no incluye el 3)
print(all_fruits2)        # ['orange', 'mango']
all_fruits3 = fruits[1:] # (toma los valores desde indice 1 en adelante)
print(all_fruits3)       # ['orange', 'mango', 'lemon']
all_fruits4 = fruits[::2] # (toma todos los valores [ : :] con el paso [:2], es decir tomará cada 2do elemento, de todas las frutas)
print(all_fruits4)        # ['banana', 'mango']

# Indexación negativa: Podemos especificar un rango de índices negativos especificando el inicio, final y paso, el valor de retorno será una nueva lista.
fruits = ['banana', 'orange', 'mango', 'lemon']
all_fruits = fruits[-4:]     # (Toma todos los valores desde el primer indice(-1) hasta -4 (indexacion negativa))
print(all_fruits)            # ['banana', 'orange', 'mango', 'lemon']
all_fruits1 = fruits[-3:-1] # (toma los valores entre indice -3 y -1, no incluye el -1)
print(all_fruits1)          # ['orange', 'mango']
all_fruits2 = fruits[-3:]   # (toma los valores desde indice -3 en adelante)
print(all_fruits2)          # ['orange', 'mango', 'lemon']
reverse_fruits = fruits[::-1] # (toma todos los valores [ : :] con el paso [:-1], es decir tomará cada 1er elemento y signo negativo significa invertir el orden)
print(reverse_fruits)         # ['lemon', 'mango', 'orange', 'banana']

# Modificación de listas
# Una lista es una colección ordenada de elementos mutable o modificable. Vamos a modificar la lista de frutas, sustituyendo algunas:
fruits = ['banana', 'orange', 'mango', 'lemon']
fruits[0] = 'avocado'
print(fruits)       #  ['avocado', 'orange', 'mango', 'lemon'] 
fruits[1] = 'apple'
print(fruits)       #  ['avocado', 'apple', 'mango', 'lemon']
last_index = len(fruits) - 1
fruits[last_index] = 'lime'
print(fruits)        #  ['avocado', 'apple', 'mango', 'lime']

# Comprobación de elementos de una lista
# Comprobar si un elemento es miembro de una lista utilizando el operador in. Véase el ejemplo siguiente.
fruits = ['banana', 'orange', 'mango', 'lemon']
does_exist = 'banana' in fruits
print(does_exist)  # True
does_exist = 'lime' in fruits
print(does_exist)  # False

# Añadir elementos a una lista: lst= list()  lst.append(item)
# Para añadir un elemento al final de una lista existente utilizamos el comando append().  
fruits = ['banana', 'orange', 'mango', 'lemon']
fruits.append('apple')
print(fruits)           # ['banana', 'orange', 'mango', 'lemon', 'apple']
fruits.append('lime')   
print(fruits)           # ['banana', 'orange', 'mango', 'lemon', 'apple', 'lime']

# Insertar elementos en una lista: lst = ['item1', 'item2']  lst.insert(index, item)
# Podemos utilizar el método insert() para insertar un único elemento en un índice especificado de una lista. Tenga en cuenta que los demás elementos 
# se desplazan a la derecha. El método insert() recibe dos argumentos: el índice y el elemento a insertar.
fruits = ['banana', 'orange', 'mango', 'lemon']
fruits.insert(2, 'apple')   # inserte una apple entre la orange y mango
print(fruits)              # ['banana', 'orange', 'apple', 'mango', 'lemon']
fruits.insert(3, 'lime')    # (inserte una lime entre apple y mango)
print(fruits)              # ['banana', 'orange', 'apple', 'lime', 'mango', 'lemon']

# Eliminar elementos de una lista: lst = ['item1', 'item2']  lst.remove(item)
# El método remove elimina un elemento especificado de una lista.
fruits = ['banana', 'orange', 'apple', 'lime', 'mango', 'lemon']
fruits.remove('lime')
print(fruits)            # ['banana', 'orange', 'apple', 'mango', 'lemon']
fruits.remove('apple')
print(fruits)            # ['banana', 'orange', 'mango', 'lemon']

# Eliminar elementos con Pop  lst = ['item1', 'item2']   lst.pop() (ultimo item)   lst.pop(index)
# El método pop() elimina el índice especificado, (o el último elemento si no se especifica el índice):
fruits = ['banana', 'orange', 'mango', 'lemon']
fruits.pop()        # (comando que por defecto elimina ultimo elmento de la lista)
print(fruits)       # ['banana', 'orange', 'mango']
fruits.pop(0)
print(fruits)       # ['orange', 'mango']

# Eliminar elementos con Del: lst = ['item1', 'item2']   del lst[index] (solo un unico item))   del lst (borrar la lista completamente)
# La palabra clave del elimina el índice especificado y también se puede utilizar para eliminar elementos dentro del rango del índice. También puede
# eliminar la lista por completo
fruits = ['banana', 'orange', 'mango', 'lemon', 'kiwi', 'lime']
del fruits[0]
print(fruits)       # ['orange', 'mango', 'lemon', 'kiwi', 'lime']
del fruits[1]
print(fruits)       # ['orange', 'lemon', 'kiwi', 'lime']
del fruits[1:3]     # (borra los valores entre indice 1 y 3, no incluye el 3)
print(fruits)       # ['orange', 'lime']
del fruits          # (borra la lista completamente)
print(fruits)       # NameError: name 'fruits' is not defined 

# Borrar elementos de la lista: lst = ['item1', 'item2']   lst.clear()
# El método clear() vacía la lista:
fruits = ['banana', 'orange', 'mango', 'lemon']
fruits.clear()
print(fruits)    # []

# Copiar una lista: lst = ['item1', 'item2']   lst_copy = lst.copy()
# Es posible copiar una lista reasignándola a una nueva variable de la siguiente forma: lista2 = lista1. Ahora, lista2 es una referencia de lista1,
# cualquier cambio que hagamos en lista2 también modificará la original, lista1. Pero hay muchos casos en los que no nos gusta modificar el original 
# sino tener una copia diferente. Una forma de evitar el problema anterior es usar copy().
fruits = ['banana', 'orange', 'mango', 'lemon']
fruits_copy = fruits.copy()
print(fruits_copy)       # ['banana', 'orange', 'mango', 'lemon']

# Unir listas
# Existen varias formas de unir, o concatenar, dos o más listas en Python.
# Operador más (+) list3 = list1 + list2
positive_numbers = [1, 2, 3, 4, 5]
zero = [0]
negative_numbers = [-5,-4,-3,-2,-1]
integers = negative_numbers + zero + positive_numbers
print(integers) # [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

fruits = ['banana', 'orange', 'mango', 'lemon']
vegetables = ['Tomato', 'Potato', 'Cabbage', 'Onion', 'Carrot']
fruits_and_vegetables = fruits + vegetables
print(fruits_and_vegetables ) # ['banana', 'orange', 'mango', 'lemon', 'Tomato', 'Potato', 'Cabbage', 'Onion', 'Carrot']

# Unir usando el método extend() El método extend() permite añadir una lista a otra. Véase el ejemplo siguiente:
list1 = ['item1', 'item2']
list2 = ['item3', 'item4', 'item5']
list1.extend(list2)
print(list1)   # ['item1', 'item2', 'item3', 'item4', 'item5']

num1 = [0, 1, 2, 3]
num2= [4, 5, 6]
num1.extend(num2)
print('Numbers:', num1) # Numbers: [0, 1, 2, 3, 4, 5, 6]

negative_numbers = [-5,-4,-3,-2,-1]
positive_numbers = [1, 2, 3,4,5]
zero = [0]
negative_numbers.extend(zero)
print(negative_numbers)  # [-5, -4, -3, -2, -1, 0]
negative_numbers.extend(positive_numbers)
print('Integers:', negative_numbers) # Integers: [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

fruits = ['banana', 'orange', 'mango', 'lemon']
vegetables = ['Tomato', 'Potato', 'Cabbage', 'Onion', 'Carrot']
fruits.extend(vegetables)
print('Fruits and vegetables:', fruits ) # Fruits and vegetables: ['banana', 'orange', 'mango', 'lemon', 'Tomato', 'Potato', 'Cabbage', 'Onion', 'Carrot']

# Contar elementos de una lista  lst = ['item1', 'item2']  lst.count(item)
# El método count() devuelve el número de veces que aparece un elemento en una lista:
fruits = ['banana', 'orange', 'mango', 'lemon']
print(fruits.index('orange'))   # 1

ages = [22, 19, 24, 25, 26, 24, 25, 24]
print(ages.index(24))           # 2

# Invertir una lista: lst = ['item1', 'item2']  lst.reverse()
# El método reverse() invierte el orden de una lista.
fruits = ['banana', 'orange', 'mango', 'lemon']
fruits.reverse()
print(fruits) # ['lemon', 'mango', 'orange', 'banana']

ages = [22, 19, 24, 25, 26, 24, 25, 24]
ages.reverse()
print(ages) # [24, 25, 24, 26, 25, 24, 19, 22]

# Ordenar elementos de una lista: lst = ['item1', 'item2']  lst.sort() (ascendente)   lst.sort(reverse=True) (descendente)
# Para ordenar listas podemos utilizar el método sort() o las funciones integradas sorted(). El método sort() reordena los elementos de la lista en orden 
# ascendente y modifica la lista original. Si un argumento del método sort() inverso es igual a true, ordenará la lista en orden descendente.
fruits = ['banana', 'orange', 'mango', 'lemon']
fruits.sort()
print(fruits)             # ['banana', 'lemon', 'mango', 'orange'] (ordenados alfabéticamente)
fruits.sort(reverse=True)
print(fruits)             # ['orange', 'mango', 'lemon', 'banana']

ages = [22, 19, 24, 25, 26, 24, 25, 24]
ages.sort()
print(ages) #  [19, 22, 24, 24, 24, 25, 25, 26]
ages.sort(reverse=True)
print(ages) #  [26, 25, 25, 24, 24, 24, 22, 19]

# sorted(): devuelve la lista ordenada sin modificar la lista original.
fruits = ['banana', 'orange', 'mango', 'lemon']
print(sorted(fruits))   # ['banana', 'lemon', 'mango', 'orange']

# Orden inverso, sin modificar la lista original.
fruits = ['banana', 'orange', 'mango', 'lemon']
print(sorted(fruits,reverse=True))  # ['orange', 'mango', 'lemon', 'banana']
print(fruits)  # ['banana', 'orange', 'mango', 'lemon']

# upper() Cambiar el nombre de una lista de minuscula a mayuscula por numero de indexacion.  list =list[x].upper()
companias = ['Facebook', 'Google', 'Microsoft', 'Apple', 'IBM', 'Oracle', 'Amazon']
companias[2]=companias[2].upper()
print(companias)    # ['Facebook', 'Google', 'MICROSOFT', 'Apple', 'IBM', 'Oracle', 'Amazon']


# Ejercicios

# Declarar una lista vacía
list = []
my_list = list()

# Declarar una lista con más de 5 elementos
my_list2 = ['banana', 'lemon', 'mango', 'orange']

# Averiguar la longitud de la lista
print(len(my_list2))

# Obtener el primer elemento, el elemento central y el último elemento de la lista
frutas = ['manzana', 'banano', 'naranja', 'mango' 'uva', 'pitaya']
primer_fruta =frutas[0]
print('la primer fruta es:',primer_fruta)            # la primer fruta es: manzana
indice_central = len(frutas) // 2
fruta_central = frutas[indice_central]
print('la fruta central es:',fruta_central)         # la fruta central es: naranja
ultimo_indice = len(frutas)-1
ultima_fruta = frutas[ultimo_indice]
print('La ultima fruta es:',ultima_fruta)           # La ultima fruta es: pitaya

# Declara una lista llamada datos_mezclados, pon (Alvaro, 55, 1.70, casado, Armenia)
datos_mezclados = ['Alvaro', 55, 1.70, 'casado', 'Armenia']

# Declare una variable de lista llamada companias y asígnele los valores iniciales Facebook, Google, Microsoft, Apple, IBM, Oracle y Amazon.
companias = ['Facebook', 'Google', 'Microsoft', 'Apple', 'IBM', 'Oracle', 'Amazon']

# Imprime la lista con print()
print(datos_mezclados)

# Imprimir el número de empresas de la lista
print(companias)

# Imprima la primera, central y ultima compañia
companias = ['Facebook', 'Google', 'Microsoft', 'Apple', 'IBM', 'Oracle', 'Amazon']
primer_comapania =companias[0]
print('la primer compañia es:',primer_comapania)  # la primer compañia es: Facebook
indice_central1 = len(companias) // 2
compania_central = companias[indice_central1]
print('la compañia central es:',compania_central) # la compañia central es: Apple
ultimo_indice1 = len(companias)-1
ultima_compania = companias[ultimo_indice1]
print('La ultima comapañia es:',ultima_compania)  # La ultima comapañia es: Amazon

# Imprimir la lista después de modificar una de las empresas
companias = ['Facebook', 'Google', 'Microsoft', 'Apple', 'IBM', 'Oracle', 'Amazon']
companias_copy = companias.copy()  # (crear una copia de la lista original "companias" para no modificarla)
companias_copy[0] = 'Alibaba'
print(companias_copy)             # ['Alibaba', 'Google', 'Microsoft', 'Apple', 'IBM', 'Oracle', 'Amazon']

# Añadir dos empresas a companias
companias = ['Facebook', 'Google', 'Microsoft', 'Apple', 'IBM', 'Oracle', 'Amazon']
companias_copy.append('Etzy')
print(companias_copy)                # ['Facebook', 'Google', 'Microsoft', 'Apple', 'IBM', 'Oracle', 'Amazon', 'Etzy']
companias_copy.insert(0, 'Alibaba')
print(companias_copy)                # ['Alibaba', 'Facebook', 'Google', 'Microsoft', 'Apple', 'IBM', 'Oracle', 'Amazon', 'Etzy']

# Insertar una empresa informática en el centro de la lista de empresas
companias = ['Facebook', 'Google', 'Microsoft', 'Apple', 'IBM', 'Oracle', 'Amazon']
companias_copy = companias.copy()    # (crear una copia de la lista original "companias" para no modificarla)
indice_central2 = len(companias) // 2
companias_copy.insert(indice_central2, 'Aliexpres')
print(companias_copy)    # ['Facebook', 'Google', 'Microsoft', 'Aliexpres', 'Apple', 'IBM', 'Oracle', 'Amazon']

# Cambie uno de los nombres de companias a mayúsculas (¡excluida IBM!)
companias = ['Facebook', 'Google', 'Microsoft', 'Apple', 'IBM', 'Oracle', 'Amazon']
companias_copy = companias.copy()   # (crear una copia de la lista original "companias" para no modificarla)
companias_copy[2]=companias_copy[2].upper()
print(companias_copy)    # ['Facebook', 'Google', 'MICROSOFT', 'Apple', 'IBM', 'Oracle', 'Amazon']

# Une las companias con una cadena '#; '
companias = ['Facebook', 'Google', 'Microsoft', 'Apple', 'IBM', 'Oracle', 'Amazon']
companias_copy = companias.copy()   # (crear una copia de la lista original "companias" para no modificarla)
joined_companias = '#; '.join(companias_copy)   # Facebook#; Google#; Microsoft#; Apple#; IBM#; Oracle#; Amazon
print(joined_companias)

# Comprobar si una determinada empresa existe en la lista companias
companias = ['Facebook', 'Google', 'Microsoft', 'Apple', 'IBM', 'Oracle', 'Amazon']
does_exist = 'Google' in companias
print(does_exist)    # True
does_exist1 = 'Aliexpres' in companias
print(does_exist1)   # false

# Ordena la lista con el método sort()
companias = ['Facebook', 'Google', 'Microsoft', 'Apple', 'IBM', 'Oracle', 'Amazon']
companias_copy = companias.copy()  # (crear una copia de la lista original "companias" para no modificarla)
companias_copy.sort()          
print(companias_copy)          # ['Amazon', 'Apple', 'Facebook', 'Google', 'IBM', 'Microsoft', 'Oracle']   
companias_copy.sort(reverse=True)
print(companias_copy)          # ['Oracle', 'Microsoft', 'IBM', 'Google', 'Facebook', 'Apple', 'Amazon']

# Invierte la lista en orden descendente utilizando el método reverse()
companias = ['Facebook', 'Google', 'Microsoft', 'Apple', 'IBM', 'Oracle', 'Amazon']
companias_copy = companias.copy()  # (crear una copia de la lista original "companias" para no modificarla)
companias_copy.reverse()
print(companias_copy)   # ['Amazon', 'Oracle', 'IBM', 'Apple', 'Microsoft', 'Google', 'Facebook']

# Corta las 3 primeras empresas de la lista
companias = ['Facebook', 'Google', 'Microsoft', 'Apple', 'IBM', 'Oracle', 'Amazon']
companias_copy = companias.copy()  # (crear una copia de la lista original "companias" para no modificarla)
companias_copy.remove('Facebook')
companias_copy.remove('Google')
companias_copy.remove('Microsoft')
print(companias_copy)                 # ['Apple', 'IBM', 'Oracle', 'Amazon']
companias_copy1 = companias.copy()  # (crear una copia de la lista original "comapanias" para no modificarla)
companias_copy1.pop(0)
companias_copy1.pop(0)
companias_copy1.pop(0)
print(companias_copy1)                # ['Apple', 'IBM', 'Oracle', 'Amazon']
companias_copy2 = companias.copy()  # (crear una copia de la lista original "companias" para no modificarla)
del companias_copy2[0:3]
print(companias_copy2)                # ['Apple', 'IBM', 'Oracle', 'Amazon'] 

# Elimina las 3 últimas empresas de la lista
companias = ['Facebook', 'Google', 'Microsoft', 'Apple', 'IBM', 'Oracle', 'Amazon']
companias_copy3 = companias.copy()  # (crear una copia de la lista original "companias" para no modificarla)
del companias_copy3[-3:]
print(companias_copy3)    # ['Facebook', 'Google', 'Microsoft', 'Apple']

# Elimina de la lista a la empresa o empresas de TI intermedias
companias = ['Facebook', 'Google', 'Microsoft', 'Apple', 'IBM', 'Oracle', 'Amazon']
companias_copy4 = companias.copy()  # (crear una copia de la lista original "companias" para no modificarla)
indice_central4 = len(companias_copy4) // 2
companias_copy4.pop(indice_central4)
print(companias_copy4)                # ['Facebook', 'Google', 'Microsoft', 'IBM', 'Oracle', 'Amazon']

# Elimine la primera empresa informática de la lista
companias = ['Facebook', 'Google', 'Microsoft', 'Apple', 'IBM', 'Oracle', 'Amazon']
companias_copy5 = companias.copy()  # (crear una copia de la lista original "companias" para no modificarla)
companias_copy5.remove('Facebook')
print(companias_copy5)              # ['Google', 'Microsoft', 'Apple', 'IBM', 'Oracle', 'Amazon']

# Elimine de la lista la empresa o empresas de TI del medio
companias = ['Facebook', 'Google', 'Microsoft', 'Apple', 'IBM', 'Oracle', 'Amazon']
companias_copy6 = companias.copy()  # (crear una copia de la lista original "companias" para no modificarla)
del companias_copy6[2:5]
print(companias_copy6)           # ['Facebook', 'Google', 'Oracle', 'Amazon']

# Elimine la última empresa informática de la lista
companias = ['Facebook', 'Google', 'Microsoft', 'Apple', 'IBM', 'Oracle', 'Amazon']
companias_copy7 = companias.copy()  # (crear una copia de la lista original "companias" para no modificarla)
companias_copy7.pop()
print(companias_copy7)   #   ['Facebook', 'Google', 'Microsoft', 'Apple', 'IBM', 'Oracle']

# Eliminar todas las empresas de TI de la lista
companias = ['Facebook', 'Google', 'Microsoft', 'Apple', 'IBM', 'Oracle', 'Amazon']
companias_copy8 = companias.copy()  # (crear una copia de la lista original "companias" para no modificarla)
companias_copy8.clear()
print(companias_copy8)       # []

# Destruye la lista de empresas de TI
companias = ['Facebook', 'Google', 'Microsoft', 'Apple', 'IBM', 'Oracle', 'Amazon']
companias_copy9 = companias.copy()  # (crear una copia de la lista original "companias" para no modificarla)
del companias_copy9
print(companias_copy9)       # NameError: name 'companias_copy9' is not defined

# Unir las siguientes listas:
# front_end = ['HTML', 'CSS', 'JS', 'React', 'Redux'] y back_end = ['Node','Express', 'MongoDB']
front_end = ['HTML', 'CSS', 'JS', 'React', 'Redux']
back_end = ['Node','Express', 'MongoDB']
front_back = front_end + back_end  
print(front_back)           # ['HTML', 'CSS', 'JS', 'React', 'Redux', 'Node', 'Express', 'MongoDB']
front_end.extend(back_end)
print(front_end)            # ['HTML', 'CSS', 'JS', 'React', 'Redux', 'Node', 'Express', 'MongoDB']
print(back_end)             # ['Node', 'Express', 'MongoDB']

# La siguiente es una lista de las edades de 10 estudiantes:
edades = [19, 22, 19, 24, 20, 25, 26, 24, 25, 24]

# Ordena la lista y encuentra la edad mínima y máxima
print(sorted(edades))               # [19, 19, 20, 22, 24, 24, 24, 25, 25, 26]
print(sorted(edades,reverse=True))  # [26, 25, 25, 24, 24, 24, 22, 20, 19, 19]

# Añade de nuevo a la lista la edad mínima y la edad máxima
edades = [19, 22, 19, 24, 20, 25, 26, 24, 25, 24]
edades.append('19')
edades.append('26')
print(edades)    # [19, 22, 19, 24, 20, 25, 26, 24, 25, 24, '19', '26']

# Hallar la mediana de la lista "edades1=[19, 22, 19, 24, 20, 25, 26, 24, 25, 24, '19', '26']
# 1er paso Convertir los valores en numeros enteros
edades1=[19, 22, 19, 24, 20, 25, 26, 24, 25, 24, '19', '26']
edades1=[int(e) for e in edades1 if isinstance(e, int) or e.isdigit()]
# Segundo paso, ordenar la lista
edades1.sor()
# Calcular la mediana
edades1.sort()
n = len(edades1)
if n % 2 == 0:
    mediana = (edades1[n//2-1] + edades1[n//2])/2
else:
    mediana = edades1[n//2]
print(mediana)           # 24
# Encontrar la edad media
suma_edades = sum(edades1)
numero_edades = len(edades1)
edad_media = suma_edades / numero_edades
print('La media media es: ', edad_media)    # La eada media es:  22.75
# Encuentra el rango de las edades
edades1=[19, 22, 19, 24, 20, 25, 26, 24, 25, 24, '19', '26']
edades1=[int(x) for x in edades1]  # otra forma de convertir a numeros enteros
edad_minima=min(edades1)
edad_maxima = max(edades1)
rango_edades = edad_maxima - edad_minima
print('El rango de edades es:',rango_edades)   # El rango de edades es: 7

# Compara el valor de (min - media) y (max - media), utiliza el método abs()
dif_min = abs(edad_media - edad_minima)
dif_max = abs(edad_media - edad_maxima)

# Encontrar el país o países del centro
indice_central4 = len(countries) // 2
print(indice_central4)

countries = [
  'Afghanistan',
  'Albania',
  'Algeria',
  'Andorra',
  'Angola',
  'Antigua and Barbuda',
  'Argentina',
  'Armenia',
  'Australia',
  'Austria',
  'Azerbaijan',
  'Bahamas',
  'Bahrain',
  'Bangladesh',
  'Barbados',
  'Belarus',
  'Belgium',
  'Belize',
  'Benin',
  'Bhutan',
  'Bolivia',
  'Bosnia and Herzegovina',
  'Botswana',
  'Brazil',
  'Brunei',
  'Bulgaria',
  'Burkina Faso',
  'Burundi',
  'Cambodia',
  'Cameroon',
  'Canada',
  'Cape Verde',
  'Central African Republic',
  'Chad',
  'Chile',
  'China',
  'Colombi',
  'Comoros',
  'Congo (Brazzaville)',
  'Congo',
  'Costa Rica',
  "Cote d'Ivoire",
  'Croatia',
  'Cuba',
  'Cyprus',
  'Czech Republic',
  'Denmark',
  'Djibouti',
  'Dominica',
  'Dominican Republic',
  'East Timor (Timor Timur)',
  'Ecuador',
  'Egypt',
  'El Salvador',
  'Equatorial Guinea',
  'Eritrea',
  'Estonia',
  'Ethiopia',
  'Fiji',
  'Finland',
  'France',
  'Gabon',
  'Gambia, The',
  'Georgia',
  'Germany',
  'Ghana',
  'Greece',
  'Grenada',
  'Guatemala',
  'Guinea',
  'Guinea-Bissau',
  'Guyana',
  'Haiti',
  'Honduras',
  'Hungary',
  'Iceland',
  'India',
  'Indonesia',
  'Iran',
  'Iraq',
  'Ireland',
  'Israel',
  'Italy',
  'Jamaica',
  'Japan',
  'Jordan',
  'Kazakhstan',
  'Kenya',
  'Kiribati',
  'Korea, North',
  'Korea, South',
  'Kuwait',
  'Kyrgyzstan',
  'Laos',
  'Latvia',
  'Lebanon',
  'Lesotho',
  'Liberia',
  'Libya',
  'Liechtenstein',
  'Lithuania',
  'Luxembourg',
  'Macedonia',
  'Madagascar',
  'Malawi',
  'Malaysia',
  'Maldives',
  'Mali',
  'Malta',
  'Marshall Islands',
  'Mauritania',
  'Mauritius',
  'Mexico',
  'Micronesia',
  'Moldova',
  'Monaco',
  'Mongolia',
  'Morocco',
  'Mozambique',
  'Myanmar',
  'Namibia',
  'Nauru',
  'Nepal',
  'Netherlands',
  'New Zealand',
  'Nicaragua',
  'Niger',
  'Nigeria',
  'Norway',
  'Oman',
  'Pakistan',
  'Palau',
  'Panama',
  'Papua New Guinea',
  'Paraguay',
  'Peru',
  'Philippines',
  'Poland',
  'Portugal',
  'Qatar',
  'Romania',
  'Russia',
  'Rwanda',
  'Saint Kitts and Nevis',
  'Saint Lucia',
  'Saint Vincent',
  'Samoa',
  'San Marino',
  'Sao Tome and Principe',
  'Saudi Arabia',
  'Senegal',
  'Serbia and Montenegro',
  'Seychelles',
  'Sierra Leone',
  'Singapore',
  'Slovakia',
  'Slovenia',
  'Solomon Islands',
  'Somalia',
  'South Africa',
  'Spain',
  'Sri Lanka',
  'Sudan',
  'Suriname',
  'Swaziland',
  'Sweden',
  'Switzerland',
  'Syria',
  'Taiwan',
  'Tajikistan',
  'Tanzania',
  'Thailand',
  'Togo',
  'Tonga',
  'Trinidad and Tobago',
  'Tunisia',
  'Turkey',
  'Turkmenistan',
  'Tuvalu',
  'Uganda',
  'Ukraine',
  'United Arab Emirates',
  'United Kingdom',
  'United States',
  'Uruguay',
  'Uzbekistan',
  'Vanuatu',
  'Vatican City',
  'Venezuela',
  'Vietnam',
  'Yemen',
  'Zambia',
  'Zimbabwe',
]
indice_central4 = (len(countries) // 2)-1
print(countries[indice_central4])    # Lebanon

# Divida la lista ?countries? en dos listas iguales si es par si no un país más para la primera mitad.
if len(countries) % 2 == 0:
    half = len(countries) // 2
    first_half = countries[:half]
    second_half = countries[half:]
else:
    half = len(countries) // 2 + 1
    first_half = countries[:half]
    second_half = countries[half:]
print(first_half)
print(second_half)

# En la lista ['China', 'Russia', 'USA', 'Finland', 'Sweden', 'Norway', 'Denmark'] Desembalar los tres primeros países y el resto como países escandinavos
lst1 = ['China', 'Russia', 'USA', 'Finland', 'Sweden', 'Norway', 'Denmark']
p1, p2, p3, *escandinavos = lst1
print(p1)    # China
print(p2)    # Russia
print(p3)    # USA
print(escandinavos)  # ['Finland', 'Sweden', 'Norway', 'Denmark']        