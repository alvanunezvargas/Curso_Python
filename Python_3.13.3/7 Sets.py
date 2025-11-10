# Conjuntos (Sets)
# Un conjunto es una colección de elementos. La definición matemática de conjunto se puede aplicar también en Python. Un conjunto es una colección 
# de elementos distintos no ordenados ni indexados. En Python el conjunto se utiliza para almacenar elementos únicos, y es posible encontrar la unión,
# intersección, diferencia, diferencia simétrica, subconjunto, superconjunto y conjunto disjunto entre conjuntos.

# Crear un conjunto
# Utilizamos llaves {} para crear un conjunto o la función incorporada set().

# Creación de un conjunto vacío
st = {}
st = set()

# Creación de un conjunto con elementos iniciales
st = {'item1', 'item2', 'item3', 'item4'}

# Ejemplo
fruits = {'banana', 'orange', 'mango', 'lemon'}

# Obtener la longitud de un conjunto
# Usamos el método len() para encontrar la longitud de un conjunto.
st = {'item1', 'item2', 'item3', 'item4'}
print(len(st))  # 4

fruits = {'banana', 'orange', 'mango', 'lemon','avocado'}
print(len(fruits))  # 5

# Acceso a los elementos de un conjunto
# Utilizamos bucles para acceder a los elementos. Lo veremos en la sección de bucles

# Comprobación de un elemento
# Para comprobar si un elemento existe en una lista se utiliza el operador de pertenencia.
st = {'item1', 'item2', 'item3', 'item4'}
print("Does set st contain item3? ", 'item3' in st) # Does set st contain item3? True

fruits = {'banana', 'orange', 'mango', 'lemon'}
print('mango' in fruits ) # True

# Añadir elementos a un conjunto
# Una vez creado un conjunto no podemos cambiar ningún elemento, sin embargo podemos añadir y eliminar elementos.

# Añadir un elemento utilizando add()
st = {'item1', 'item2', 'item3', 'item4'}
st.add('item5')
print(st)     # {'item5', 'item1', 'item3', 'item2', 'item4'}

fruits = {'banana', 'orange', 'mango', 'lemon'}
fruits.add('lime')
print(fruits)  # {'lime', 'mango', 'orange', 'banana', 'lemon'}

# Eliminar elementos de un conjunto
# Podemos eliminar un elemento de un conjunto utilizando el método remove(). Si el elemento no se encuentra, el método remove() producirá errores,
# por lo que es bueno comprobar si el elemento existe en el conjunto dado. Sin embargo, el método discard() no genera ningún error.
st = {'item1', 'item2', 'item3', 'item4'}
st.remove('item2')
print(st)        # {'item3', 'item4', 'item1'} 

# El método pop() elimina un elemento aleatorio de una lista y devolverá un elemento aleatorio del conjunto
fruits = {'banana', 'orange', 'mango', 'lemon'}
fruits.pop()  
print(fruits)   # {'lemon', 'mango', 'orange'}  (pop() Remueve aleatoriamente cualquier objeto dentro de un conjunto)

fruits = {'banana', 'orange', 'mango', 'lemon'}
removed_item = fruits.pop()
print(removed_item)  # banana
print(fruits)        # {'mango', 'lemon', 'orange'}

it_companies = {'Facebook', 'Google', 'Microsoft', 'Apple', 'IBM', 'Oracle', 'Amazon'}
it_companies.discard('Acer')
print(it_companies)            # (No hace nada)
it_companies.remove('Acer')
print(it_companies)            # KeyError: 'Acer' (Detiene la corrida si hay secuencia posterior de comandos)

# Borrar elementos de un conjunto
# Si queremos borrar o vaciar el conjunto utilizamos el método clear.
st = {'item1', 'item2', 'item3', 'item4'}
st.clear()
print(st)    # Set

fruits = {'banana', 'orange', 'mango', 'lemon'}
fruits.clear()
print(fruits)   # set()

# Borrar un conjunto
# Si queremos borrar el propio conjunto utilizamos el operador del.
st = {'item1', 'item2', 'item3', 'item4'}
del st
print(st)   # NameError: name 'st' is not defined

fruits = {'banana', 'orange', 'mango', 'lemon'}
del fruits
print(fruits)    # NameError: name 'fruits' is not defined

# Convertir lista en conjunto
# Podemos convertir lista en conjunto y conjunto en lista. La conversión de lista a conjunto elimina los duplicados y sólo se reservarán los elementos únicos.
lst = ['item1', 'item2', 'item3', 'item4', 'item1']
st = set(lst)  
print(st)    # {'item2', 'item4', 'item1', 'item3'}

fruits = ['banana', 'orange', 'mango', 'lemon','orange', 'banana']
fruits_s = set(fruits) 
print(fruits_s)    # {'mango', 'lemon', 'banana', 'orange'}

# Unir conjuntos
# Podemos unir dos conjuntos utilizando el método union() o update().

# Unión Este método devuelve un nuevo conjunto
st1 = {'item1', 'item2', 'item3', 'item4'}
st2 = {'item5', 'item6', 'item7', 'item8'}
st3 = st1.union(st2)
print(st3)  # {'item6', 'item2', 'item1', 'item7', 'item5', 'item3', 'item4', 'item8'}

A = {19, 22, 24, 20, 25, 26}
B = {19, 22, 20, 25, 26, 24, 28, 27}
union = A | B
print(union)    # {19, 20, 22, 24, 25, 26, 27, 28}

fruits = {'banana', 'orange', 'mango', 'lemon'}
vegetables = {'tomato', 'potato', 'cabbage','onion', 'carrot'}
print(fruits.union(vegetables)) # {'lemon', 'carrot', 'tomato', 'banana', 'mango', 'orange', 'cabbage', 'potato', 'onion'}

# Update este método inserta un conjunto en un conjunto dado
st1 = {'item1', 'item2', 'item3', 'item4'}
st2 = {'item5', 'item6', 'item7', 'item8'}
st1.update(st2)  # (el contenido de st2 se añade a st1)
print(st1)   # {'item7', 'item2', 'item4', 'item1', 'item8', 'item6', 'item3', 'item5'}

fruits = {'banana', 'orange', 'mango', 'lemon'}
vegetables = {'tomato', 'potato', 'cabbage','onion', 'carrot'}
fruits.update(vegetables) # (el contenido de vegetables se añade a fruits)
print(fruits) # {'cabbage', 'carrot', 'orange', 'onion', 'potato', 'lemon', 'banana', 'tomato', 'mango'}

# Búsqueda de elementos de intersección
# La intersección devuelve un conjunto de elementos que se encuentran en ambos conjuntos.
set1 = {1, 2, 3, 4, 5}
set2 = {3, 4, 5, 6, 7}
result = set1.intersection(set2)
print(result)  # {3, 4, 5}


set1 = {1, 2, 3, 4, 5}
set2 = {3, 4, 5, 6, 7}
set3 = {2, 4, 6, 8}
result = set1.intersection(set2, set3)
print(result)  # {4}

list1 = [1, 2, 3, 4, 5]
list2 = [3, 4, 5, 6, 7]
result = set(list1).intersection(set(list2))
print(result)  # {3, 4, 5}

# Comprobación de subconjunto y superconjunto
# Un conjunto puede ser subconjunto o superconjunto de otros conjuntos:

# Subconjunto: issubset()  Se utiliza para comprobar si un conjunto es un subconjunto de otro
# Superconjunto: issuperset() Se utiliza para comprobar si un conjunto contiene todos los elementos de otro conjunto
whole_numbers = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
even_numbers = {0, 2, 4, 6, 8, 10}
print(even_numbers.issubset(whole_numbers))   # True   (Responde si todos los elementos de "even_numbers" estan contenidos en "whole_numbers" ó "even_numbers" es subconjunto de "whole_numbers")
print(whole_numbers.issubset(even_numbers))   # False, (Responde si todos los elementos de "whole_numbers" estan contenidos en "even_numbers" ó "whole_numbers" es subconjunto de "even_numbers")
print(whole_numbers.issuperset(even_numbers)) # True   (Responde si todos los elementos de "even_numbers" estan contenidos en "whole_numbers" ó "whole_numbers" es superconjunto de "even_numbers")
print(even_numbers.issuperset(whole_numbers)) # False  (Responde si todos los elementos de "whole_numbers" estan contenidos en "even_numbers" ó "even_numbers" es superconjunto de "whole_numbers")

A = {1, 2, 3, 4, 5}
B = {1, 2, 3}
C = {6, 7}
print(B.issubset(A)) # True       (Responde si todos los elementos de "B" estan contenidos en "A" ó "B" es subconjunto de "A")
print(B.issuperset(A))  # False   (Responde si todos los elementos de "A" estan contenidos en "B" ó A es superconjunto de "A")
print(C.issubset(A)) # False      (Responde si todos los elementos de "C" estan contenidos en "A" ó "C" es subconjunto de "A")
print(C.issuperset(A)) # False    (Responde si todos los elementos de "A" estan contenidos en "C" ó C es superconjunto de "A")

fruits = {'apple', 'banana', 'cherry', 'orange', 'kiwi'}
my_fruits = {'banana', 'orange', 'kiwi'}
print(my_fruits.issubset(fruits)) # True   (Responde si todos los elementos de "my_fruits" estan contenidos en "fruits" ó "my_fruits" es subconjunto de "fruits")
print(fruits.issubset(my_fruits)) # False      (Responde si todos los elementos de "fruits" estan contenidos en "my_fruits" ó "fruits" es subconjunto de "my_fruits")
print(my_fruits.issuperset(fruits)) # False    (Responde si todos los elementos de "fruits" estan contenidos en "my_fruits" ó "my fruits" es superconjunto de "fruits")
print(fruits.issuperset(my_fruits)) # True    (Responde si todos los elementos de "my_fruits" estan contenidos en "fruits" ó "fruits" es superconjunto de "my_fruits") )

set1 = {1, 2, 3, 4, 5}
set2 = {3, 4}
set3 = {5, 6, 7}
print(set1.issuperset(set2))  # True
print(set1.issuperset(set3))  # False

set1 = {1, 2, 3, 4, 5}
set2 = {1, 2, 3, 4, 5}
print(set1.issuperset(set2))  # True

set1 = {1, 2, 3, 4, 5}
set2 = {1, 2, 3, 4, 5}
set3 = {5, 6, 7}
print(set2.issuperset(set1))  # True
print(set3.issuperset(set1))  # False

# Comprobación de la diferencia entre dos conjuntos
# Devuelve la diferencia entre dos conjuntos.
st1 = {'item1', 'item2', 'item3', 'item4'}
st2 = {'item2', 'item3'}
print(st2.difference(st1)) # set()   (Devuelve los elementos de conjunto st2 que no se encuentran en st1)
print(st1.difference(st2)) # {'item4', 'item1'} (Devuelve los elementos de conjunto st1 que no se encuentran en st2)

whole_numbers = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
even_numbers = {0, 2, 4, 6, 8, 10}
print(whole_numbers.difference(even_numbers))  # {1, 3, 5, 7, 9}
print(even_numbers.difference( whole_numbers)) # set()

python = {'p', 'y', 't', 'o','n'}
dragon = {'d', 'r', 'a', 'g', 'o','n'}
print(python.difference(dragon))     # {'p', 'y', 't'}
print(dragon.difference(python))     # {'g', 'a', 'd', 'r'}

# Hallar la diferencia simétrica entre dos conjuntos
# Devuelve la diferencia simétrica entre dos conjuntos. Significa que devuelve un conjunto que contiene todos los elementos de ambos conjuntos, 
# excepto, los elementos que están presentes en ambos conjuntos, matemáticamente: (A\B) ∪ (B\A)
#st1 = {'item1', 'item2', 'item3', 'item4'}
st2 = {'item2', 'item3'}
# it means (A\B)∪(B\A)
print(st2.symmetric_difference(st1)) # {'item1', 'item4'} (Devuelve los elementos de conjunto que no son comunes en los dos conjuntos)
print(st1.symmetric_difference(st2)) # {'item1', 'item4'} (Devuelve los elementos de conjunto que no son comunes en los dos conjuntos)


whole_numbers = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
some_numbers = {1, 2, 3, 4, 5}
print(whole_numbers.symmetric_difference(some_numbers)) # {0, 6, 7, 8, 9, 10}  (Devuelve los elementos de conjunto que no son comunes en los dos conjuntos)
print(some_numbers .symmetric_difference(whole_numbers))  # {0, 6, 7, 8, 9, 10} (Devuelve los elementos de conjunto que no son comunes en los dos conjuntos)

python = {'p', 'y', 't', 'h', 'o','n'}
dragon = {'d', 'r', 'a', 'g', 'o','n'}
print(python.symmetric_difference(dragon))  # {'y', 't', 'r', 'd', 'a', 'g', 'h', 'p'}
print(dragon.symmetric_difference(python))  # {'y', 't', 'r', 'd', 'a', 'g', 'h', 'p'}

# Unir conjuntos
# Si dos conjuntos no tienen un elemento o elementos comunes los llamamos conjuntos disjuntos. Podemos comprobar si dos conjuntos son conjuntos o 
# disjuntos utilizando el método isdisjoint().
st1 = {'item1', 'item2', 'item3', 'item4'}
st2 = {'item2', 'item3'}
print(st2.isdisjoint(st1))  # False  (Tienen elmentos comunes))

odd_numbers = {0, 2, 4 ,6, 8}
even_numbers = {1, 3, 5, 7, 9}
print(even_numbers.isdisjoint(odd_numbers)) # True  (No tienen elmentos comunes)

python = {'p', 'y', 't', 'h', 'o','n'}
dragon = {'d', 'r', 'a', 'g', 'o','n'}
print(python.isdisjoint(dragon))  # False


# Ejercicios

it_companies = {'Facebook', 'Google', 'Microsoft', 'Apple', 'IBM', 'Oracle', 'Amazon'}
A = {19, 22, 24, 20, 25, 26}
B = {19, 22, 20, 25, 26, 24, 28, 27}
age = [22, 19, 24, 25, 26, 24, 25, 24]

# Hallar la longitud del conjunto it_companies
print(len(it_companies))  # 7

# Añadir "Twitter" a it_companies
it_companies.add('Twitter')
print(it_companies)      # {'Apple', 'Twitter', 'Microsoft', 'Oracle', 'Facebook', 'Amazon', 'IBM', 'Google'}

# Insertar varias empresas informáticas a la vez en el conjunto it_companies
other_companies= ('Twitter', 'Dell', 'Acer', 'Alibaba')
it_companies.update(other_companies)
print(it_companies)   # {'Alibaba', 'Amazon', 'Dell', 'IBM', 'Oracle', 'Google', 'Microsoft', 'Apple', 'Facebook', 'Twitter', 'Acer'}

# Eliminar una de las empresas del conjunto it_companies
it_companies.remove('Facebook')
print(it_companies)         # {'Apple', 'Amazon', 'Google', 'IBM', 'Microsoft', 'Oracle'}
# ó
it_companies.pop()
print(it_companies)         # {'Oracle', 'IBM', 'Amazon', 'Google', 'Facebook', 'Apple'}

# ¿Cuál es la diferencia entre remove() and discard()
# remove() genera un KeyError si el elemento a eliminar no está en el conjunto, mientras que discard() no genera un error y simplemente no hace nada 
# en ese caso. Por lo tanto, remove() debe utilizarse cuando se espera que el elemento esté siempre presente en el conjunto, mientras que discard() debe 
# utilizarse cuando se desea eliminar un elemento si está ahí, pero no pasa nada si no está.
it_companies.discard('Acer')
print(it_companies)            # No hace nada
it_companies.remove('Acer')
print(it_companies)            # KeyError: 'Acer' (Detiene la corrida)

# Unir A y B
union = A | B
print(union)    # {19, 20, 22, 24, 25, 26, 27, 28}

# Buscar A intersección B
result = A.intersection(B)
print(result)                 # {19, 20, 22, 24, 25, 26}
result1 = A.intersection(B)
print(result1)                # {19, 20, 22, 24, 25, 26}

# Es A subconjunto de B
print(A.issubset(B))   # True

# ¿Son A y B conjuntos disjuntos?
print(A.isdisjoint(B))   # False

# Unir A con B y B con A
union_AB = A | B
union_BA = B | A
print("A unido con B:", union_AB)  # A unido con B: {19, 20, 22, 24, 25, 26, 27, 28}
print("B unido con A:", union_BA)  # B unido con A: {19, 20, 22, 24, 25, 26, 27, 28}

# Cuál es la diferencia simétrica entre A y B
print(A.symmetric_difference(B))  # {27, 28}

# Eliminar los conjuntos por completo
del A
print(A)  # NameError: name 'A' is not defined
del B
print(B)  # NameError: name 'B' is not defined

# Convierte "age" en un conjunto y compara la longitud de la lista y del conjunto, ¿cuál es mayor?
age_set = set(age)
print(age_set)                                          # {19, 22, 24, 25, 26}  (El conjunto descarta valores repetidos de la lista)
print("Longitud de la lista age:", len(age))            # Longitud de la lista age: 8
print("Longitud del conjunto age_set:", len(age_set))   # Longitud del conjunto age_set: 5

# Explique la diferencia entre los siguientes tipos de datos: cadena, lista, tupla y conjunto

# Cadenas o strings (str): Una cadena es una secuencia de caracteres que se utiliza para representar texto en Python. Las cadenas se definen entre comillas 
# simples ('...') o dobles ("..."). Las cadenas son inmutables, lo que significa que una vez que se crea una cadena, no se puede cambiar. Las operaciones
# comunes en cadenas incluyen la concatenación, la indexación y el slicing.

# Listas (list): Una lista es una colección ordenada y mutable de elementos que pueden ser de diferentes tipos de datos (números, cadenas, booleanos,
# otras listas, etc.). Las listas se definen utilizando corchetes ([]) y los elementos se separan por comas. Las operaciones comunes en listas incluyen 
# la indexación, el slicing, la adición y eliminación de elementos.

# Tuplas (tuple): Una tupla es similar a una lista, pero es inmutable, lo que significa que una vez que se crea una tupla, no se puede cambiar. Las tuplas
# se definen utilizando paréntesis (()) y los elementos se separan por comas. Las operaciones comunes en tuplas incluyen la indexación y el slicing.

# Conjuntos (set): Un conjunto es una colección no ordenada y mutable de elementos únicos. Los conjuntos se definen utilizando llaves ({}) o la función set(),
# y los elementos se separan por comas. Las operaciones comunes en conjuntos incluyen la adición y eliminación de elementos, la unión, la intersección y
# la diferencia.

# "I am a teacher and I love to inspire and teach people" ¿Cuántas palabras únicas se han utilizado en la frase? Utilice los métodos de división split() y 
# ajuste strip() para obtener las palabras únicas.
sentence = "I am a teacher and I love to inspire and teach people"
words = sentence.split()  # (Dividir la oración en palabras)
print(words)      # ['I', 'am', 'a', 'teacher', 'and', 'I', 'love', 'to', 'inspire', 'and', 'teach', 'people']
words = [word.strip() for word in words]  # (Eliminar espacios adicionales alrededor de cada palabra)
print(words)              # ['I', 'am', 'a', 'teacher', 'and', 'I', 'love', 'to', 'inspire', 'and', 'teach', 'people']
unique_words = set(words)  # (Crear un conjunto de palabras únicas)
print("Número de palabras únicas:", len(unique_words))  # Número de palabras únicas: 10
print("Palabras únicas:", unique_words)                 # Palabras únicas: {'I', 'a', 'to', 'inspire', 'people', 'love', 'teach', 'and', 'am', 'teacher'}
