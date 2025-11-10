"""
Tipos de error en Python
Cuando escribimos código es frecuente que cometamos un error. Si nuestro código no se ejecuta, el intérprete de Python mostrará un mensaje con
información sobre dónde se produce el problema y el tipo de error. A veces también nos dará sugerencias sobre una posible solución. Entender los
diferentes tipos de errores en lenguajes de programación nos ayudará a depurar nuestro código rápidamente y también nos hará mejores en lo que hacemos.

Veamos los tipos de error más comunes uno por uno. Primero abramos nuestra shell interactiva de Python. Ve al terminal de tu ordenador y escribe
'python'. Se abrirá el intérprete de comandos interactivo de Python.
"""

# SyntaxError: Este error ocurre cuando hay un error de sintaxis en el código.
# Ejemplo:

if x = 5:
    print('Hello')  # SyntaxError: invalid syntax. Maybe you meant '==' or ':=' instead of '='?

# NameError: Este error ocurre cuando se utiliza un nombre que no está definido en el programa.
# Ejemplo

print(variable)   # NameError: name 'variable' is not defined. Did you mean: 'callable'?

# IndexError: Este error ocurre cuando se accede a un índice inválido en una secuencia.
# Ejemplo:

lista = [1, 2, 3]
print(lista[3])  # IndexError: list index out of range

# ModuleNotFoundError: Este error ocurre cuando se intenta importar un módulo que no está instalado o no existe.
# Ejemplo:

import non_existent_module    # ModuleNotFoundError: No module named 'non_existent_module'

# AttributeError: Este error ocurre cuando se intenta acceder a un atributo que no existe en un objeto.
# Ejemplo:

cadena = "Hola"
print(cadena.uppercase())  # AttributeError: 'str' object has no attribute 'uppercase'

# KeyError: Este error ocurre cuando se intenta acceder a una clave que no existe en un diccionario.
# Ejemplo

diccionario = {'nombre': 'Juan', 'edad': 25}
print(diccionario['apellido'])    # KeyError: 'apellido'

# TypeError: Este error ocurre cuando se realiza una operación con operandos de tipos incompatibles.
# Ejemplo:

resultado = '10' + 5   # TypeError: can only concatenate str (not "int") to str

# ImportError: Este error ocurre cuando no se puede importar un módulo o una función desde un módulo.
# Ejemplo:

from math import non_existent_function    # ImportError: cannot import name 'non_existent_function' from 'math' (unknown location)

# ValueError: Este error ocurre cuando se pasa un argumento con un valor incorrecto a una función.
# Ejemplo:

numero = int('abc')  # ValueError: invalid literal for int() with base 10: 'abc'

# ZeroDivisionError: Este error ocurre cuando se intenta dividir un número entre cero.
# Ejemplo:

resultado = 10 / 0   # ZeroDivisionError: division by zero

# Otros errores:

# RecursionError: Este error ocurre cuando una función se llama a sí misma de manera recursiva sin alcanzar un caso base.

# IOError: Este error ocurre cuando ocurre un problema de entrada/salida, como no poder abrir o leer un archivo.

# MemoryError: Este error ocurre cuando el programa se queda sin memoria disponible.

# AssertionError: Este error ocurre cuando una afirmación condicional falla.

# KeyboardInterrupt: Este error ocurre cuando se interrumpe la ejecución del programa con la combinación de teclas Ctrl+C.

# FileNotFoundErros Este error ocurre cuando no se encuentra un archivo)