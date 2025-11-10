# Un tuple es una colección de diferentes tipos de datos ordenados e inmutables. Las tuplas se escriben con corchetes, (). Una vez creada una tupla,
# no podemos cambiar sus valores. No podemos utilizar los métodos add, insert, remove en una tupla porque no es modificable (mutable). A diferencia 
# de las listas, las tuplas tienen pocos métodos.
# Métodos relacionados con las tuplas:

# tuple(): para crear una tupla vacía
# count(): para contar el número de un elemento especificado en una tupla
# index(): para encontrar el índice de un elemento especificado en una tupla
# ° operator: para unir dos o más tuplas y crear una nueva tupla

# Creación de una tupla
# Tuple vacía: Creación de una tupla vacía
empty_tuple = ()
empty_tuple = tuple()

# Tuple con valores iniciales
tpl = ('item1', 'item2','item3')
fruits = ('banana', 'orange', 'mango', 'lemon')

# Longitud de la tupla
# Utilizamos el método len() para obtener la longitud de una tupla.
tpl = ('item1', 'item2', 'item3')
len(tpl)

# Acceso a los elementos de tuple
# Indexación positiva De forma similar al tipo de datos lista, utilizamos la indexación positiva o negativa para acceder a los elementos de tuple.

# ()'banana',   'orange'   'mango'   'lemon')
#     0           1          2         3

tpl = ('item1', 'item2', 'item3')
first_item = tpl[0]
second_item = tpl[1]

fruits = ('banana', 'orange', 'mango', 'lemon')
first_fruit = fruits[0]
second_fruit = fruits[1]
last_index =len(fruits) - 1
last_fruit = fruits[last_index]

# Indexación negativa: La indexación negativa significa que empezando por el final, -1 se refiere al último elemento, -2 al penúltimo y el negativo 
# de la longitud de la lista/tupla se refiere al primer elemento.

# ()'banana',   'orange'   'mango'   'lemon')
#     -4          -3         -2        -1

tpl = ('item1', 'item2', 'item3','item4')
first_item = tpl[-4]
second_item = tpl[-3]

fruits = ('banana', 'orange', 'mango', 'lemon')
first_fruit = fruits[-4]
second_fruit = fruits[-3]
last_fruit = fruits[-1]

# Cortar tuples
# Podemos cortar una sub-tuple especificando un rango de índices donde empezar y donde terminar en la tupla, el valor de retorno será una nueva tupla 
# con los elementos especificados.
tpl = ('item1', 'item2', 'item3','item4')
all_items = tpl[0:4]         # todos los objetos
all_items = tpl[0:]          # todos los objetos
middle_two_items = tpl[1:3]
print(middle_two_items)     # ('item2', 'item3')

fruits = ('banana', 'orange', 'mango', 'lemon')
all_fruits = fruits[0:4]    # Todos los objetos
all_fruits= fruits[0:]      # Todos los objetos
orange_mango = fruits[1:3]  # no incluye el elemento en el índice 3
print(orange_mango)         # ('orange', 'mango')
orange_to_the_rest = fruits[1:]
print(orange_to_the_rest)   # ('orange', 'mango', 'lemon')

# Rango de índices negativos
tpl = ('item1', 'item2', 'item3','item4')
all_items = tpl[-4:]         # Todos los objetos
middle_two_items = tpl[-3:-1]  # no incluye el elemento del índice positivo 3 ó indice negativo -1
print(middle_two_items)      # ('item2', 'item3')

fruits = ('banana', 'orange', 'mango', 'lemon')
all_fruits = fruits[-4:]    # Todos los objetos
orange_mango = fruits[-3:-1]  # No incluye el objeto de indice positivo 3 o indice negativo -1
orange_to_the_rest = fruits[-3:]
print(orange_to_the_rest)    # ('orange', 'mango', 'lemon')

# Cambiar tuplas por listas: tpl = ('item1', 'item2', 'item3','item4')   lst = list(tpl)
# Podemos cambiar tuplas a listas y listas a tuplas. La tupla es inmutable si queremos modificar una tupla debemos cambiarla a lista.
fruits = ('banana', 'orange', 'mango', 'lemon')
fruits = list(fruits)
fruits[0] = 'apple'
print(fruits)     # ['apple', 'orange', 'mango', 'lemon']
fruits = tuple(fruits)
print(fruits)     # ('apple', 'orange', 'mango', 'lemon')

# Comprobación de un elemento en una tupla: tpl = ('item1', 'item2', 'item3','item4')   'item2' in tpl # True
# Podemos comprobar si un elemento existe o no en una tupla utilizando in, que devuelve un booleano.
fruits = ('banana', 'orange', 'mango', 'lemon')
print('orange' in fruits) # True
print('apple' in fruits)  # False
fruits[0] = 'apple'       # TypeError: 'tuple' el objeto no admite la asignación de elementos

# Unir tuplas: tpl1 = ('item1', 'item2', 'item3')  tpl2 = ('item4', 'item5','item6')   tpl3 = tpl1 + tpl2
# Podemos unir dos o más tuplas utilizando el operador +.
fruits = ('banana', 'orange', 'mango', 'lemon')
vegetables = ('Tomato', 'Potato', 'Cabbage','Onion', 'Carrot')
fruits_and_vegetables = fruits + vegetables
print(fruits_and_vegetables)    # ('banana', 'orange', 'mango', 'lemon', 'Tomato', 'Potato', 'Cabbage', 'Onion', 'Carrot')

# Borrar tuplas: tpl1 = ('item1', 'item2', 'item3')  del tpl1
# No es posible eliminar un único elemento de una tupla, pero sí es posible eliminar la propia tupla utilizando del.
fruits = ('banana', 'orange', 'mango', 'lemon')
del fruits

# Ejercicios
# Crear una tupla vacía:
empty_tuple = ()
empty_tuple = tuple()

# Crea una tupla que contenga los nombres de tus hermanas y hermanos (también puedes crear hermanos imaginarios).
sisters = ('olga', 'constanza', 'diana')
brothers = ('gustavo', 'ricardo', 'ana')

# Une las tuplas brothers y sisters y asígnalas a siblings
siblings = sisters + brothers
print(siblings)            # ('olga', 'constanza', 'diana', 'gustavo', 'ricardo', 'ana')

# ¿Cuántos hermanos tiene?
print(len(siblings))   # 6

# Modifica la tupla siblings y añade el nombre de tu padre y de tu madre y asígnalo a family_members
siblings = list(siblings)
siblings.append('jorge')
siblings.append('rubby')
family_members = tuple(siblings)
print(siblings)         # # ('olga', 'constanza', 'diana', 'gustavo', 'ricardo', 'ana', 'jorge', 'rubby')

# Crea tuplas de frutas, verduras y productos animales. Une las tres tuplas y asígnalo a una variable llamada food_stuff_tp.

fruits = ('apple', 'banana', 'orange', 'grape')
vegetables = ('carrot', 'potato','spinach', 'kale')
animal_products = ('beef', 'chicken', 'eggs', 'milk')
food_stuff_tp = fruits + vegetables + animal_products
print(food_stuff_tp) # ('apple', 'banana', 'orange', 'grape', 'carrot', 'potato', 'spinach', 'kale', 'beef', 'chicken', 'eggs', 'milk')

# Cambiar la tupla food_stuff_tp por una lista food_stuff_lt
food_stuff_lt = list(food_stuff_tp)
print(food_stuff_lt) # ['apple', 'banana', 'orange', 'grape', 'carrot', 'potato', 'spinach', 'kale', 'beef', 'chicken', 'eggs', 'milk']

# Corta el elemento o elementos centrales de la tupla food_stuff_tp o de la lista food_stuff_lt.
middle_index = len(food_stuff_lt) // 2  # Índice central de la lista
if len(food_stuff_lt) % 2 == 0:
    # Si la lista tiene un número par de elementos, se deben cortar dos elementos centrales
    food_stuff_lt = food_stuff_lt[:middle_index-1] + food_stuff_lt[middle_index+1:]
else:
    # Si la lista tiene un número impar de elementos, se corta el elemento central
    food_stuff_lt = food_stuff_lt[:middle_index] + food_stuff_lt[middle_index+1:]
    
print(food_stuff_lt) # ['apple', 'banana', 'orange', 'grape', 'carrot', 'kale', 'beef', 'chicken', 'eggs', 'milk']

# Corta los tres primeros y los tres últimos elementos de la lista food_staff_lt
food_stuff_lt = food_stuff_lt[3:-3]
print(food_stuff_lt) # ['grape', 'carrot', 'kale', 'beef']

# Borrar completamente la tupla food_staff_tp
del food_stuff_tp
print(food_stuff_tp)  # NameError

# Comprueba si existe un elemento en la tupla:
fruits = ('apple', 'banana', 'orange', 'grape')
print('banana' in fruits)  # True
print('kiwi' in fruits)    # False

# Compruebe si "Estonia" y Iceland son países nórdicos en la tupla "paises_nordicos = ('Denmark', 'Finland', 'Iceland', 'Norway', 'Sweden')"
paises_nordicos = ('Denmark', 'Finland', 'Iceland', 'Norway', 'Sweden')

if 'Estonia' in paises_nordicos:
    print('Estonia es un país nórdico')
else:
    print('Estonia no es un país nórdico')
                                               # Estonia no es un país nórdico
    
if 'Iceland' in paises_nordicos:
    print('Iceland es un país nórdico')
else:
    print('Iceland no es un país nórdico')
                                                # Iceland es un país nórdico