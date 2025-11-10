# Esto es un primer comentario
# esto es un segundo comentario
# python se está comiendo el mundo

# Tipos de datos

# 1 Integer (int): Numeros enteros negativos, cero y positivos) Ejemplo: ... -3, -2, -1, 0, 1, 2, 3 ...
#   Float: Numeros decimales. Ejemplo ... -3.5, -2.25, -1.0, 0.0, 1.1, 2.2, 3.5 ...
#   Complex: Numeros complejos. Ejemplo 1 + j, 2 + 4j

# 2 String (str): Conjunto de uno o varios caracteres entre comillas simples o dobles. Si una cadena tiene más de una frase, se utilizan comillas triples.
'python'
"Finland"
"""Esto es un comentario multilínea
el comentario multilínea ocupa varias líneas 
python se está comiendo el mundo
"""

# 3 Booleans: Un tipo de dato booleano es un valor Verdadero o Falso. T y F deben ir siempre en mayúsculas.
True # ¿Está encendida la luz? Si está encendida, el valor es Verdadero
False # ¿Está encendida la luz? Si está apagada, el valor es Falso

# 4 List Clist): En Python es una colección ordenada que permite almacenar diferentes tipos de datos. Una lista es similar a un array en JavaScript. 
# Las listas se direccionan con []
[0, 1, 2, 3, 4, 5] # todos son los mismos tipos de datos - una lista de números
['Banana', 'Orange', 'Mango', 'Avocado'] # todas son los mismos tipos de datos - una lista de strings (frutas)
['Banana', 10, False, 9.81]  # Diferentes tipos de datos en la lista list - string, integer, boolean and float

# 5 Tuple (tuple): Una tupla es una colección ordenada de diferentes tipos de datos como una lista, pero las tuplas no pueden modificarse una vez creadas. 
# Son inmutables. Los tuples se direccionan con ()
('Asabeneh', 'Pawel', 'Brook', 'Abraham', 'Lidiya') # Names
('Earth', 'Jupiter', 'Neptune', 'Mars', 'Venus', 'Saturn', 'Uranus', 'Mercury') # planets

# 6 Set (set): Es una colección de tipos de datos similar a list y tuple que no se repiten. A diferencia de list y tuple, set no es una colección ordenada 
# de elementos. Puede contener elementos de diferentes tipos de datos. Se pueden iniciar con valores de datos separados por una coma y encerrados entre llaves {}.
set_1={1,2,'python'}
{3.14, 9.81, 2.7} # El orden no es importante en un set

# 7 Dictionary (dict): Es una colección ordenada de valores de dato. Se utilizan para almacenar pares clave-valor. Los valores pueden ser de cualquier 
# tipo de datos y también se pueden modificar. Usamos {} corchetes o el dict() método para crear un diccionario vacío
{
'first_name':'Alvaro',
'last_name':'Nunez',
'country':'Colombia', 
'age':55, 
'is_married':True,
'skills':['JS', 'React', 'Node', 'Python']
}

# Verificar tipos de datos
print(type(10))          # int (Numero entero)
print(type(3.14))        # float (Numero decimal)
print(type(True))        # bool (Representa un valor de verdad; es decir, TRUE o FALSE)
print(type(1 + 3j))      # complex (Numero complejo)
print(type('Asabeneh'))  # String
print(type([1, 2, 3]))   # List
print(type({'name':'Asabeneh'})) # Dictionary
print(type({9.8, 3.14, 2.7}))    # Set
print(type((9.8, 3.14, 2.7)))    # Tuple