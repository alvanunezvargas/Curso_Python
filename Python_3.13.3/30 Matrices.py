# Determinar si un vector es una combinacion lineal de otro vector:

def es_combinacion_lineal(vector1, vector2):
    # Verifica si los vectores tienen la misma longitud
    if len(vector1) != len(vector2):
        return False
    
    # Calcula el cociente entre cada par de elementos
    cocientes = [vector2[i] / vector1[i] if vector1[i] != 0 else None for i in range(len(vector1))]
    
    # Verifica si todos los cocientes son iguales
    if all(cociente == cocientes[0] for cociente in cocientes):
        return True
    else:
        return False

# Ejemplo de uso
vector1 = [1, 2, 3]
vector2 = [2, 4, 6]

if es_combinacion_lineal(vector1, vector2):
    print("El vector 2 es una combinación lineal del vector 1.")
else:
    print("El vector 2 NO es una combinación lineal del vector 1.")


# Verifica si un vector dado w, pertenece al plano de dos vectores u y v.

import numpy as np

u = np.array([1, 0, 1])
v = np.array([3, 2, 0])

n = np.cross(u, v)
print(n)
w = np.array([1, 4, 0])
producto_escalar = np.dot(w, n)
print(producto_escalar)
if producto_escalar == 0:
    print('El vector w pertenece al mismo plano de u y v')
else:
    print('El vector w NO pertenece al mismo plano de u y v')
    

# Determina si tres vectores dados se encuentran en la misma linea recta, es decir si son colineales.

import numpy as np

P = np.array([0, -2, 4])
Q = np.array([1, -3, 5])

vector_director = Q - P
print("Vector director:", vector_director)

R = np.array([4, -6, 8])
vector_PR = R - P

# Calcula el cociente de los vectores
cociente = vector_PR / vector_director

# Verifica si todos los elementos del cociente son iguales
es_colineal = np.allclose(cociente, cociente[0])

if es_colineal:
    print("Los vectores son colineales.")
else:
    print("Los vectores NO son colineales.")


# Evaluar si dos vectores son Ortogonales.

import numpy as np

u = np.array([1, 5, -2])
v = np.array([2, 0, 1])

producto_escalar = np.dot(u, v)
print(producto_escalar)

if producto_escalar == 0:
    print("Los vectores son Ortogobnales.")
else:
    print("Los vectores NO son Ortogonales.")


"""
Para calcular la proyección de un vector u sobre otro vector v, podemos usar la fórmula: 

proy (v) u = (u ⋅ v / ∥v∥^2) v

Donde ⋅ representa el producto punto y ∥v∥ es la magnitud de v. Usaremos esta fórmula para calcular la proyección de u sobre v y viceversa.
"""

import numpy as np

def calcular_proyeccion(u, v):
    proyeccion_u_sobre_v = np.dot(u, v) / np.dot(v, v) * v
    proyeccion_v_sobre_u = np.dot(v, u) / np.dot(u, u) * u
    return proyeccion_u_sobre_v, proyeccion_v_sobre_u

# Definimos los vectores u y v
u = np.array([2, 2, 3])
v = np.array([1, -2, 0])

# Calculamos las proyecciones
proyeccion_u_sobre_v, proyeccion_v_sobre_u = calcular_proyeccion(u, v)

# Imprimimos los resultados
print("Proyección de u sobre v:", proyeccion_u_sobre_v)
print("Proyección de v sobre u:", proyeccion_v_sobre_u)


"""
Dados dos vectores a y b de 3 dimensiones, distintos de cero, encontrar un tercer vector c distinto de cero, que sea perpendicular a ambos a = (a1, a2, a3), b = (b1, b2, b3) y c = (x, y z) y que sea distinto de cero. El producto cruz de a y b sera a x b = (a2b3 - a3b2, a3b1 - a1b3, a1b2 - a2b1) = (x, y, z). Si el producto cruz es cero, entonces a y b son paralelos, por lo que ajustamos un componente de c para que sea distinto de cero. Si a1 es distinto de cero, entonces c = (-a2, a1, 0), si no, c = (0, -a3, a2).
"""

import numpy as np

def encontrar_vector_perpendicular(a, b):
    # Calculamos el producto cruz entre los vectores a y b
    c = np.cross(a, b)
    
    # Si el vector c resultante es cero (es decir, a y b son paralelos), ajustamos un componente
    if np.allclose(c, np.zeros(3)):
        if a[0] != 0:
            c = np.array([-a[1], a[0], 0])
        else:
            c = np.array([0, -a[2], a[1]])
    
    return c

# Definimos los vectores a y b (distintos de cero)
a = np.array([1, 1, -2])
b = np.array([-3, 1, 0])

# Encontramos un vector c perpendicular a ambos
c = encontrar_vector_perpendicular(a, b)

print("El vector c perpendicular a ambos a y b es:", c)


"""
Dados dos vectores a y b de 3 dimensiones, distintos de cero, encontrar dos vectores unitarios ortogonales a estos
"""

import numpy as np

# Vectores a y b (3 dimensiones, diferentes de cero)
a = np.array([3, 2, 1])
b = np.array([-1, 1, 0])

# Paso 1: Producto cruz
c = np.cross(a, b)

# Paso 2: Normalización de c
u1 = c / np.linalg.norm(c)

# Paso 3: Generación de un vector aleatorio diferente de cero
d = np.random.rand(3)

# Paso 4: Producto cruz entre c y d
e = np.cross(c, d)

# Paso 5: Normalización de e
u2 = e / np.linalg.norm(e)

print("Vector unitario ortogonal a 'a' y 'b':", u1)
print("Otro vector unitario ortogonal a 'a', 'b' y el primer vector unitario:", u2)


# Encontrar un vector ortogonal al plano que pasa por P = [1 0 1], Q = [-2 1 -3] y R = [4 2 5]

import numpy as np

# Definir los puntos P, Q y R
P = np.array([1, 0, 1])
Q = np.array([-2, 1, -3])
R = np.array([4, 2, 5])

# Paso 1: Calcular el vector PQ
PQ = Q - P

# Paso 2: Calcular el vector PR
PR = R - P

# Paso 3: Calcular el producto cruz de PQ y PR para encontrar un vector ortogonal al plano
vector_ortogonal = np.cross(PQ, PR)

print("Un vector ortogonal al plano es:", vector_ortogonal)

# GRADED FUNCTION
import numpy as np

# Our function will go through the matrix replacing each row in order turning it into echelon form.
# If at any point it fails because it can't put a 1 in the leading diagonal,
# we will return the value True, otherwise, we will return False.
# There is no need to edit this function.
def isSingular(A) :
    B = np.array(A, dtype=np.float_) # Make B as a copy of A, since we're going to alter it's values.
    try:
        fixRowZero(B)
        fixRowOne(B)
        fixRowTwo(B)
        fixRowThree(B)
    except MatrixIsSingular:
        return True
    return False

# This next line defines our error flag. For when things go wrong if the matrix is singular.
# There is no need to edit this line.
class MatrixIsSingular(Exception): pass

# For Row Zero, all we require is the first element is equal to 1.
# We'll divide the row by the value of A[0, 0].
# This will get us in trouble though if A[0, 0] equals 0, so first we'll test for that,
# and if this is true, we'll add one of the lower rows to the first one before the division.
# We'll repeat the test going down each lower row until we can do the division.
# There is no need to edit this function.
def fixRowZero(A) :
    if A[0,0] == 0 :
        A[0] = A[0] + A[1]
    if A[0,0] == 0 :
        A[0] = A[0] + A[2]
    if A[0,0] == 0 :
        A[0] = A[0] + A[3]
    if A[0,0] == 0 :
        raise MatrixIsSingular()
    A[0] = A[0] / A[0,0]
    return A

# First we'll set the sub-diagonal elements to zero, i.e. A[1,0].
# Next we want the diagonal element to be equal to one.
# We'll divide the row by the value of A[1, 1].
# Again, we need to test if this is zero.
# If so, we'll add a lower row and repeat setting the sub-diagonal elements to zero.
# There is no need to edit this function.
def fixRowOne(A) :
    A[1] = A[1] - A[1,0] * A[0]
    if A[1,1] == 0 :
        A[1] = A[1] + A[2]
        A[1] = A[1] - A[1,0] * A[0]
    if A[1,1] == 0 :
        A[1] = A[1] + A[3]
        A[1] = A[1] - A[1,0] * A[0]
    if A[1,1] == 0 :
        raise MatrixIsSingular()
    A[1] = A[1] / A[1,1]
    return A

# This is the first function that you should complete.
# Follow the instructions inside the function at each comment.
def fixRowTwo(A) :
    # Insert code below to set the sub-diagonal elements of row two to zero (there are two of them).
    A[2] = A[2] - A[2,0] * A[0]
    A[2] = A[2] - A[2,1] * A[1]
    # Next we'll test that the diagonal element is not zero.
    if A[2,2] == 0 :
        # Insert code below that adds a lower row to row 2.
        A[2] = A[2] + A[3]
        A[2] = A[2] - A[2,0] * A[0]
        A[2] = A[2] - A[2,1] * A[1]
        # Now repeat your code which sets the sub-diagonal elements to zero.
        
        
    if A[2,2] == 0 :
        raise MatrixIsSingular()
    # Finally set the diagonal element to one by dividing the whole row by that element.
    A[2] = A[2] / A[2,2]
    return A

# You should also complete this function
# Follow the instructions inside the function at each comment.
def fixRowThree(A) :
    # Insert code below to set the sub-diagonal elements of row three to zero.
    A[3] = A[3] - A[3,0] * A[0]
    A[3] = A[3] - A[3,1] * A[1]
    A[3] = A[3] - A[3,2] * A[2]
    if A[3,3] == 0 :
        A[3] = A[3] - A[3,0] * A[2]
    # Complete the if statement to test if the diagonal element is zero.
    if A[3,3] == 0 :
        raise MatrixIsSingular()
    # Transform the row to set the diagonal element to one.
    A[3] = A[3] / A[3,3]
    return A

A = np.array([
        [2, 0, 0, 0],
        [0, 3, 0, 0],
        [0, 0, 4, 4],
        [0, 0, 5, 5]
    ], dtype=float)

A_copy = np.array(A, dtype=np.float_) # Make a copy of A

print(isSingular(A))
print(fixRowZero(A))
print(fixRowOne(A))
print(fixRowTwo(A))
print(fixRowThree(A))


# Otra forma de hacerlo:

import numpy as np

class MatrixIsSingular(Exception): pass

def fixRow(A, row, singular_check_col):
    if A[row, singular_check_col] == 0:
        for r in range(row+1, 4):
            if A[r, singular_check_col] != 0:
                A[row] += A[r]
                break
    if A[row, singular_check_col] == 0:
        raise MatrixIsSingular()
    A[row] = A[row] / A[row, singular_check_col]
    print(f"After fixRow {row}:")
    print(A)
    return A

def eliminate(A, row, eliminate_below=True):
    factor_col = row if eliminate_below else row - 1
    for r in range((row+1) if eliminate_below else (row-1), 4 if eliminate_below else -1, 1 if eliminate_below else -1):
        A[r] -= A[row] * A[r, factor_col]
    print(f"After eliminate {row}:")
    print(A)
    return A

def isSingular(A):
    B = np.array(A, dtype=np.float_) # Make B as a copy of A, since we're going to alter it's values.
    try:
        for i in range(4):
            B = fixRow(B, i, i)
            if i < 3: B = eliminate(B, i)
        B = eliminate(B, 3, False)
    except MatrixIsSingular:
        return True
    return False

A = np.array([
        [1, -2, -1, 3],
        [-1, 3, -2, -2],
        [2, 0, 1, 1],
        [1, -2, 2, 3]
    ], dtype=float)

print(isSingular(A))


"""
Este código transforma un vector `a_B` de la base B a la base canónica: 
"""
import numpy as np

# Definir la base B en coordenadas canonicas.
B = np.array([
    [3, 1],
    [1, 1]
])

# Definir un vector a_B en las coordenadas de B
a_B = np.array([3/2, 1/2])

# Para transformar coordenadas de a con base B a la base canónica, multiplicamos a por B
a_canonical = np.dot(B, a_B)

print("a in canonical basis: ", a_canonical)


"""
Este código transforma un vector `a_B` de la base canonica a la base B: 
"""

import numpy as np

# Definir la base B en las coordenadas canonicas
B = np.array([
    [3, 1],
    [1, 1],
     ])

# Definir un vector c en las coordenadas canónicas
c_canonical = np.array([5, 2])

# Para transformar coordenadas canonicas a la base B, multiplicamos c_B por la inversa de B
c_B = np.dot(np.linalg.inv(B), c_canonical)

print("c in B basis: ", c_B)


""" 
Determinar las coordenadas de los vectores en la base B. Trazar dos vectores ortogonales de angulo de 45° con respecto a la base canonica y determinar las coordenadas en la base B de posicion ([[3 1], [1 1]]) y en la base canónica.
"""
import numpy as np

# Definir las matrices A, B y C
B = np.array([[3, 1], [1, 1]])  # Coordenadas de la base B
B_inversa = np.linalg.inv(B)  # Inversa de coordenadas de la base B
Rc = (1/np.sqrt(2)) * np.array([[1, -1], [1, 1]]) # Coordenadas de los vectores base puestos 45° en la base canónica


# Coordenadas de los vectores 45° en la base B
Binv_Rc = np.dot(B_inversa, Rc)
Rb = np.dot(Binv_Rc, B)           

print(Rb)


"""
Obtener las coordenadas canonicas de Rb en la base B.
"""

import numpy as np

# Definir las matrices A, B y C
B = np.array([[3, 1], [1, 1]])  # Coordenadas de la base B
B_inversa = np.linalg.inv(B)  # Inversa de coordenadas de la base B
Rc = (1/np.sqrt(2)) * np.array([[1, -1], [1, 1]]) # Coordenadas de los vectores base puestos 45° en la base canónica

# Coordenadas de los vectores 45° en la base B
Binv_Rc = np.dot(B_inversa, Rc)
Rb = np.dot(Binv_Rc, B)           

# Para transformar las coordenadas de la base B a las coordenadas canónicas, multiplicamos Rb por B
R_canonical = np.dot(B, Rb)

print("R in canonical basis: ", R_canonical)


"""
Función para realizar el procedimiento de Gram-Schmidt, que toma una lista de vectores y forma una base ortonormal a partir de este conjunto. El procedimiento nos permite determinar la dimensión del espacio abarcado por los vectores base, que es igual o menor que el espacio que ocupan los vectores.

Empezaremos completando una función para 4 vectores base.
"""

import numpy as np
import numpy.linalg as la

verySmallNumber = 1e-14 # That's 1×10⁻¹⁴ = 0.00000000000001

# Nuestra primera función realizará el procedimiento de Gram-Schmidt para vectores de 4 bases.
# Tomaremos esta lista de vectores como las columnas de una matriz, A.
# Luego revisaremos los vectores uno por uno y los configuraremos para que sean ortogonales.
# a todos los vectores anteriores. Antes de normalizar.
# Siga las instrucciones dentro de la función en cada comentario.
# Se le indicará dónde agregar el código para completar la función.

def gsBasis4(A) :
     B = np.array(A, dtype=np.float_) # Haz que B sea una copia de A, ya que vamos a alterar sus valores.
     # La columna cero es fácil, ya que no tiene otros vectores para normalizarla.
     # Todo lo que hay que hacer es normalizarlo. Es decir. dividir por su módulo o norma.
     B[:, 0] = B[:, 0] / la.norm(B[:, 0])
     # Para la primera columna, necesitamos restar cualquier superposición con nuestro nuevo vector cero.
     B[:, 1] = B[:, 1] - B[:, 1] @ B[:, 0] * B[:, 0]

# Si queda algo después de esa resta, entonces B[:, 1] es linealmente independiente de B[:, 0]
     # Si este es el caso, podemos normalizarlo. De lo contrario, pondremos ese vector en cero.
     if la.norm(B[:, 1]) > verySmallNumber :
        B[:, 1] = B[:, 1] / la.norm(B[:, 1])
     else :
        B[:, 1] = np.zeros_like(B[:, 1])

# Ahora necesitamos repetir el proceso para la columna 2.
     # Insertamos dos líneas de código, la primera para restar la superposición con el vector cero,
     # y el segundo para restar la superposición con el primero.
     B[:, 2] = B[:, 2] - B[:, 2] @ B[:, 0] * B[:, 0]  # Restar la superposición con el vector cero
     B[:, 2] = B[:, 2] - B[:, 2] @ B[:, 1] * B[:, 1]  # Restar la superposición con el primero
# Nuevamente necesitaremos normalizar nuestro nuevo vector.
     # Copie y adapte el fragmento de normalización de arriba a la columna 2.
     if la.norm(B[:, 2]) > verySmallNumber :
        B[:, 2] = B[:, 2] / la.norm(B[:, 2])
     else :
        B[:, 2] = np.zeros_like(B[:, 2])

# Finalmente, la columna tres:
     # Insertar código para restar la superposición con los primeros tres vectores.
     B[:, 3] = B[:, 3] - B[:, 3] @ B[:, 0] * B[:, 0]  # Restar la superposición con el vector cero
     B[:, 3] = B[:, 3] - B[:, 3] @ B[:, 1] * B[:, 1]  # Restar la superposición con el primer vector
     B[:, 3] = B[:, 3] - B[:, 3] @ B[:, 2] * B[:, 2]  # Restar la superposición con el segundo vector

# Ahora normaliza si es posible
     if la.norm(B[:, 3]) > verySmallNumber :
         B[:, 3] = B[:, 3] / la.norm(B[:, 3])
     else :
         B[:, 3] = np.zeros_like(B[:, 3])

     return B
 
# La segunda parte de este ejercicio generalizará el procedimiento.
# Anteriormente, solo podíamos tener cuatro vectores y había muchas repeticiones en el código.
# Usaremos un bucle "for" aquí para iterar el proceso para cada vector.
 
def gsBasis(A) :
    B = np.array(A, dtype=np.float_)  # Haz que B sea una copia de A, ya que vamos a alterar sus valores.
     # Recorre todos los vectores, comenzando con cero, etiquétalos con i
     
    for i in range(B.shape[1]) :
        # Recorre todos los índices desde 0 hasta i-1 (excluyendo i)
        for j in range(i) :
            # Completa el código para restar la superposición con los vectores anteriores.
            # Necesitarás el vector actual B[:, i] y un vector anterior B[:, j]
            B[:, i] = B[:, i] - B[:, i] @ B[:, j] * B[:, j]
        # A continuación, inserta código para realizar la prueba de normalización para B[:, i]
        if la.norm(B[:, i]) > verySmallNumber :
            B[:, i] = B[:, i] / la.norm(B[:, i])
        else :
            B[:, i] = np.zeros_like(B[:, i])

    return B

# Esta función utiliza el proceso de Gram-schmidt para calcular la dimensión.
# abarcado por una lista de vectores.
# Dado que cada vector está normalizado a uno o es cero,
# la suma de todas las normas será la dimensión.

def dimensions(A) :
    return np.sum(la.norm(gsBasis4(A), axis=0))
 
# Probemos las funciones con algunos ejemplos: 

V = np.array([[1,0,2,6],
              [0,1,8,2],
              [2,8,3,1],
              [1,-6,2,3]], dtype=np.float_)
gsBasis4(V)

# Aplicar el proceso de Gram-Schmidt
B = gsBasis4(V)

# Imprimir la matriz resultante
print("Matriz después del proceso de Gram-Schmidt:")
print(B)

# Calcular e imprimir la dimensión del espacio abarcado por los vectores
dim = dimensions(V)
print("\nDimensión del espacio abarcado por los vectores:", dim)


"""
Este código de Python resuelve un problema que implica reflexión de un punto de un vector en relación a un plano dado, utilizando álgebra lineal y matrices. A continuación, se explica paso a paso:

Se define la matriz E, que representa una transformación lineal. Esta matriz define cómo se transforman los puntos del espacio.

Se define la matriz TE, que representa la matriz de reflexión respecto a un plano dado. En este caso, la matriz de reflexión TE refleja los puntos a lo largo del eje z, ya que el valor en la última fila y columna es -1, invirtiendo la coordenada z.

Se calcula la inversa de la matriz E utilizando np.linalg.inv(E). La matriz inversa de E es necesaria para revertir la transformación y llevar de vuelta los puntos al espacio original después de la reflexión.

Se define el vector r, que representa el punto del vector que se reflejará en relación al plano dado.

Se realiza la multiplicación de matrices para obtener el resultado final. Primero, se multiplica la matriz E por la matriz de reflexión TE (obteniendo resultado_intermedio1). Luego, se multiplica el resultado intermedio por la matriz inversa de 
E (obteniendo resultado_intermedio2). Finalmente, se multiplica el resultado intermedio por el vector r (obteniendo resultado_final), lo que da como resultado el punto del vector después de la reflexión.

En resumen, este código utiliza matrices y álgebra lineal para realizar una reflexión de un punto de un vector en relación a un plano dado, aplicando transformaciones lineales y operaciones de multiplicación de matrices.
"""

import numpy as np

# Definir las matrices y el vector
E = np.array([
    [1/np.sqrt(3), 1/np.sqrt(2), 1/np.sqrt(6)],
    [1/np.sqrt(3), -1/np.sqrt(2), 1/np.sqrt(6)],
    [1/np.sqrt(3), 0, -2/np.sqrt(6)]
])

TE = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, -1]
])

E_inv = np.linalg.inv(E)

r = np.array([2, 3, 5])

# Multiplicación de matrices
resultado_intermedio1 = np.dot(E, TE)
resultado_intermedio2 = np.dot(resultado_intermedio1, E_inv)
resultado_final = np.dot(resultado_intermedio2, r)

print("Resultado final de la multiplicación:")
print(resultado_final)


"""
Panda Desorientado: Reflejos en el Espejo

Antecedentes:

El oso panda está confundido. Está tratando de entender cómo se deberían ver las cosas cuando se reflejan en un espejo, pero obtiene resultados incorrectos. En las coordenadas de Panda, el espejo se encuentra a lo largo del primer eje. Sin embargo, como es típico de los osos, su sistema de coordenadas no es ortonormal, por lo que lo que él cree que es la dirección perpendicular al espejo no es realmente la dirección en la que el espejo refleja. ¡Ayuda a Panda a escribir un código que haga sus cálculos matriciales correctamente!.

Instrucciones
En esta tarea escribirás una función Python que producirá una matriz de transformación para vectores reflejados en un espejo con un ángulo arbitrario.

A partir de la última tarea, en la que escribiste un código para construir una base ortonormal que abarca un conjunto de vectores de entrada, aquí tomarás una matriz que toma una forma simple en esa base, y la transformarás en nuestra base de partida. Recuerde que desde el último video,

T = E * Te * E_inv

Escribirás una función que construya esta matriz. Esta evaluación no es conceptualmente complicada, pero desarrollará y pondrá a prueba tu capacidad para expresar ideas matemáticas en código. Como tal, tu entrega final de código será relativamente corta, pero recibirás menos estructura sobre cómo escribirlo.

Matrices en Python
En este ejercicio volveremos a utilizar el operador @. Recuerda que en el ejercicio anterior utilizamos este operador para obtener el producto punto de vectores. En general el operador combinará vectores y/o matrices de la forma esperada en álgebra lineal, es decir, será el producto punto de vectores, la multiplicación de matrices, o la operación con matrices sobre un vector, dependiendo de su entrada. Por ejemplo, para calcular las siguientes expresiones,

a=s⋅t
s=At
M=AB

se utilizaría el código

a = s @ t
s = A @ t
M = A @ B

(Esto contrasta con el operador ∗, que realiza la multiplicación elemento a elemento, o multiplicación por un escalar).
"""
# Este código define un paquete.
# Ejecutar esta celda una vez al principio para cargar las dependencias necesarias para que el código funcione
import numpy as np
from numpy.linalg import norm, inv
from numpy import transpose
from readonly.bearNecessities import *

# FUNCIÓN CALIFICADA
# Debes editar esta celda..

# En esta función, vas a devolver la matriz de transformación T.
# La construirás a partir de un conjunto de base ortonormal E que crearás a partir de la Base del Oso
# y una matriz de transformación en las coordenadas del espejo TE.
def build_reflection_matrix(bearBasis) : # El parámetro bearBasis es una matriz de 2x2 que se pasa a la función.
    # Usa la función gsBasis en bearBasis para obtener la base ortonormal del espejo
    B = bearBasis
    E = gsBasis(B)
    TE = np.array([[1, 0],
                   [0, -1]])
    T = E@TE@transpose(E)
    return T

def gsBasis(A) :
    B = np.array(A, dtype=np.float_) 
    for i in range(B.shape[1]) :
        for j in range(i) :
            B[:, i] = B[:, i] - B[:, i] @ B[:, j] * B[:, j]
        if np.linalg.norm(B[:, i]) > 1e-10:
            B[:, i] /= np.linalg.norm(B[:, i])
        else :
            B[:, i] = np.zeros_like(B[:, i])
    return B
    
"""
Prueba tu código antes de enviarlo
Para probar el código que escribiste anteriormente, ejecuta la celda (selecciona la celda de arriba y luego presiona el botón de reproducción [ ▶| ] o presiona shift-enter). Luego puedes usar el código de abajo para probar tu función. No necesitas enviar esta celda; puedes editarla y ejecutarla tantas veces como quieras.

El código de abajo mostrará una imagen de Panda Bear. Si has implementado correctamente la función anterior, también verás el reflejo de Bear en su espejo. Los ejes naranjas son la base de Bear y los ejes rosados son la base ortonormal del espejo.
"""

# Primero, carga Pyplot, una biblioteca para graficar..
# matplotlib inline
import matplotlib.pyplot as plt

# Esta es la matriz de los vectores base del oso.
# (Una vez que hayas completado el ejercicio una vez, observa qué sucede cuando cambias la base del oso.)
bearBasis = np.array(
    [[1,   -1],
     [1.5, 2]])
# Esta línea utiliza tu código para construir una matriz de transformación que podremos usar.
T = build_reflection_matrix(bearBasis)

# El oso está dibujado como un conjunto de polígonos, cuyos vértices se encuentran en una lista matricial de vectores columna.
# Tenemos tres de estas listas matriciales no cuadradas: bear_white_fur, bear_black_fur y bear_face.
# Vamos a crear nuevas listas de vértices aplicando la matriz T que has calculado.
reflected_bear_white_fur = T @ bear_white_fur
reflected_bear_black_fur = T @ bear_black_fur
reflected_bear_face = T @ bear_face

# La siguiente línea ejecuta un código para configurar el entorno gráfico.
ax = draw_mirror(bearBasis)

# Primero vamos a graficar al oso, su pelaje blanco, su pelaje negro y su cara..
ax.fill(bear_white_fur[0], bear_white_fur[1], color=bear_white, zorder=1)
ax.fill(bear_black_fur[0], bear_black_fur[1], color=bear_black, zorder=2)
ax.plot(bear_face[0], bear_face[1], color=bear_white, zorder=3)

# Siguiente paso: graficaremos el reflejo del oso.
ax.fill(reflected_bear_white_fur[0], reflected_bear_white_fur[1], color=bear_white, zorder=1)
ax.fill(reflected_bear_black_fur[0], reflected_bear_black_fur[1], color=bear_black, zorder=2)
ax.plot(reflected_bear_face[0], reflected_bear_face[1], color=bear_white, zorder=3);

# Valores y vectores propios

import numpy as np

def eigen(T):
    eigenvalues, eigenvectors = np.linalg.eig(T)
    return eigenvalues, eigenvectors

# Define la matriz T
T = np.array([[-1-np.sqrt(3)/2, -1+(np.sqrt(3)/2)],
              [1, 1]])

# Obtiene los valores propios y vectores propios
eigenvalues, eigenvectors = eigen(T)

print("Valores propios:")
print(eigenvalues)
print("\nVectores propios:")
print(eigenvectors)

"""
La diagonalización de matrices:  es un proceso en álgebra lineal que consiste en transformar una matriz A en una forma diagonal utilizando una matriz de cambio de base. Esto implica encontrar una matriz invertible P y una matriz diagonal D tal que:

D = P*D*P^-1

Donde:

A    Es la matriz original que se desea diagonalizar.
P    Es una matriz que contiene los vectores propios de A.
D    Es la matriz diagonal resultante de A.
P^-1 Es la inversa de la matriz que contiene los vectores propios de A, es decir P.

Los calculos incluyen la determinacion de valores propios y vectores propios.

Cuando se desea determinar la diagonalizacion D elevado a la potencia n, tenga en cuenta en el codigo introducir el valor de n. 
"""
# Calcule la diagonalizacion: 

import numpy as np

# Definir la matriz A
A = np.array([[3/2, -1], [-1/2, 1/2]])

# Calcular valores propios y vectores propios
valores_propios, vectores_propios = np.linalg.eig(A)

# Mostrar los valores propios
print("Valores propios:")
print(valores_propios)

# Mostrar los vectores propios
print("\nVectores propios:")
print(vectores_propios)

# Construir la matriz diagonal
matriz_diagonal = np.diag(valores_propios)
print("\nMatriz diagonal:")
print(matriz_diagonal)

# Matriz inversa de vectores propios
matriz_inversa_Vec_propios = np.linalg.inv(vectores_propios)
print("Matriz inversa de A:")
print(matriz_inversa_Vec_propios)

# matirz diagonal elevada a la OJO n
n = 2
D = np.linalg.matrix_power(matriz_diagonal, n)
print("D elevado a la n es")
print(D)

# Calculo de la diagonalizacion de A^2
Diagonalizacion = np.dot(np.dot(vectores_propios, D), matriz_inversa_Vec_propios)
print("Diagonalizacion de A:")
print(Diagonalizacion)


# Utilizar variable simbolica en lugar de numero para calcular la matriz diagonal

import numpy as np
from sympy import symbols, Matrix

# Define la variable simbólica 'a'
a = symbols('a')

# Define la matriz diagonal D en términos de 'a'
D = Matrix([[a, 0], [0, a]])

# Define la matriz de vectores propios
eigenvectors = Matrix([[1, 1], [0, 1]])

# Calcula la matriz original T
T = eigenvectors * D * eigenvectors.inv()

print("Matriz original T:")
print(T)


# Ecuacuon resultante de las caracteristicas polinomiales de una matriz

from sympy import symbols, Matrix

# Define la variable simbólica 'a'
L = symbols('L')

a = 3/2
b= -1
c= -1/2
d = 1/2
# Caracteristica polinomial de la ecuación
EC = L**2 - (a+d)*L + (a*d - b*c)

print(EC)


# Elevar una matriz a una potencia

import numpy as np

# Define la matriz T
T = np.array([[1, 0],
              [2, -1]])

# Calcula T^n: En este ejemplo n=3
T_cubed = np.linalg.matrix_power(T, 3)

print("T^3:")
print(T_cubed)


"""
PageRank

Este cuaderno te permitirá afianzar tus conocimientos sobre vectores y valores propios explorando el algoritmo PageRank. Se divide en dos partes: la primera es una hoja de trabajo para familiarizarte con el funcionamiento del algoritmo; aquí analizaremos un micro-internet con menos de 10 sitios web y veremos qué hace y qué problemas puede haber. La segunda parte es una evaluación que pondrá a prueba la aplicación de la teoría de matrices propias a este problema, escribiendo código para calcular el PageRank de una red grande que representa una subsección de internet.

Parte 1 - Ficha de trabajo

Introducción

PageRank (desarrollado por Larry Page y Sergey Brin) revolucionó la búsqueda en la web al generar una lista clasificada de páginas web basada en la conectividad subyacente de la web. El algoritmo PageRank se basa en un internauta aleatorio ideal que, al llegar a una página, pasa a la siguiente haciendo clic en un enlace. El internauta tiene la misma probabilidad de hacer clic en cualquier enlace de la página y, cuando llega a una página sin enlaces, tiene la misma probabilidad de pasar a cualquier otra página escribiendo su URL. Además, el internauta puede optar ocasionalmente por teclear una URL aleatoria en lugar de seguir los enlaces de una página. El PageRank es el orden de clasificación de las páginas, desde la más probable hasta la menos probable.
"""

# Antes de comenzar, carguemos las bibliotecas.
import numpy as np
import numpy.linalg as la
from readonly.PageRankFunctions import *
np.set_printoptions(suppress=True)

"""
PageRank como un problema de álgebra lineal

Imaginemos un micro-internet con solo 6 sitios web ((Avocado, Bullseye, CatBabel, Dromeda, eTings, and FaceSpace). Cada sitio web se enlaza con algunos de los otros, formando una red como la que se muestra: (Ver archivo word "Problemas con vectores")

El principio de diseño de PageRank es que los sitios web importantes tendrán enlaces de otros sitios web importantes. Este principio un tanto recursivo será la base de nuestro razonamiento.

Imagine que tenemos 100 usuarios indecisos en nuestro micro-internet, cada uno visitando un único sitio web a la vez. Cada minuto, los usuarios siguen un enlace en su sitio web actual para navegar a otro sitio dentro del micro-internet. Después de un tiempo, los sitios web con más enlaces tendrán más usuarios visitándolos. A largo plazo, por cada usuario que abandone un sitio web en un minuto, otro ingresará, manteniendo constante el número total de usuarios en cada sitio web. El PageRank es simplemente la clasificación de los sitios web según la cantidad de usuarios que tienen al final de este proceso.

r = [ra rb rc rd re rf]

Supongamos que el número de usuarios en cada sitio web en el minuto i+1 está relacionado con el número de usuarios en el minuto i mediante una transformación matricial.

r^(i+1) = Lr^i

L = [Laa Lba Lca Lda Lea Lfa; Lab Lbb Lcb Ldb Leb Lfb; Lac Lbc Lcc Ldc Lec Lfc; Lad Lbd Lcd Ldd Led Lfd; Lae Lbe Lce Lde Lee Lfe; Laf Lbf Lcf Ldf Lef Lff]

donde las columnas representan la probabilidad de abandonar un sitio web para ir a cualquier otro sitio web, y suman uno. Las filas determinan la probabilidad de entrar a un sitio web desde cualquier otro, aunque estas no necesitan sumar uno. El comportamiento a largo plazo de este sistema es cuando r^(i+1) = r^i así que omitiremos los superíndices aquí, y esto nos permite escribir:

Lr=r

que es una ecuación de valores propios para la matriz L, con valor propio 1 (esto está garantizado por la estructura probabilística de la matriz L)

Complete la matriz L a continuación, hemos dejado en blanco la columna para los sitios web a los que enlaza el sitio web FaceSpace (F). Recuerde, esta es la probabilidad de hacer clic en otro sitio web desde este, por lo que cada columna debe sumar uno (escalando por el número de enlaces).
"""

# Reemplace los ??? aquí con la probabilidad de hacer clic en un enlace hacia cada sitio web al salir del Sitio web F (FaceSpace)..
L = np.array([[0,   1/2, 1/3, 0, 0,   0 ],
              [1/3, 0,   0,   0, 1/2, 0 ],
              [1/3, 1/2, 0,   1, 0,   1/2 ],
              [1/3, 0,   1/3, 0, 1/2, 1/2 ],
              [0,   0,   0,   0, 0,   0 ],
              [0,   0,   1/3, 0, 0,   0 ]])

"""
En principio, podríamos usar una biblioteca de álgebra lineal, como se muestra a continuación, para calcular los valores y vectores propios. Y esto funcionaría para un sistema pequeño. Pero esto se vuelve inmanejable para sistemas grandes. Y dado que solo nos importa el eigenvector principal (el que tiene el eigenvalor más grande, que será 1 en este caso), podemos usar el método de la iteración de potencia que escalará mejor y será más rápido para sistemas grandes.

Utilice el código a continuación para echar un vistazo al PageRank para este microinternet.
"""

eVals, eVecs = la.eig(L) # Calcula los valores y vectores propios
order = np.absolute(eVals).argsort()[::-1] # Los ordena por sus valores propios.
eVals = eVals[order]
eVecs = eVecs[:,order]

r = eVecs[:, 0] # Establece r como el eigenvector principal.
100 * np.real(r / np.sum(r)) # Normaliza r para que sume uno, luego multiplica por 100 para obtener porcentajes.

print(r)

"""
Podemos observar en esta lista la cantidad de usuarios indecisos que esperamos encontrar en cada sitio web después de largos periodos de tiempo. Colocándolos en orden de popularidad (basado en esta métrica), el PageRank de este microinternet es:

CatBabel, Dromeda, Avocado, FaceSpace, Bullseye, eTings

Volviendo al diagrama del microinternet, ¿es esto lo que esperarías? Convéncete de que, basándote en qué páginas parecen importantes según las que otras enlazan con ellas, este es un ranking sensato.

Ahora intentemos obtener el mismo resultado usando el método de la Iteración de Potencia que se cubrió en el video. Este método será mucho mejor para tratar con sistemas grandes.

Primero, configuremos nuestro vector inicial, r^(0), para que tengamos nuestros 100 usuarios indecisos distribuidos de manera equitativa en cada uno de nuestros 6 sitios web.
"""

r = 100 * np.ones(6) / 6 # Establece este vector (6 entradas de 1/6 × cada 100).
print(r) # Muestra su valor


# A continuación, actualicemos el vector al siguiente minuto utilizando la matriz L. Ejecuta la siguiente celda múltiples veces, hasta que el resultado se estabilice.

r = L @ r # Aplica la matriz L a r
print(r) # Muestra su valor
# Ejecuta esta celda varias veces hasta que converja a la respuesta correcta.

# Podemos automatizar la aplicación de esta matriz varias veces de la siguiente manera,

r = 100 * np.ones(6) / 6 # Establece este vector (6 entradas de 1/6 × cada 100)
for i in np.arange(100) : # Repita 100 veces
    r = L @ r
print(r)

# O aún mejor, podemos seguir ejecutando hasta que alcancemos la tolerancia requerida.

r = 100 * np.ones(6) / 6 # Establece este vector (6 entradas de 1/6 × cada 100)
lastR = r
r = L @ r
i = 0
while la.norm(lastR - r) > 0.01 :
    lastR = r
    r = L @ r
    i += 1
print(str(i) + " iterations to convergence.")
print(r)

"""
Observa cómo el orden del PageRank se establece con bastante rapidez, y el vector converge hacia el valor que calculamos anteriormente después de unas pocas decenas de repeticiones.

¡Enhorabuena! ¡Acabas de calcular tu primer PageRank!


Parámetro de amortiguación

El sistema que acabamos de estudiar convergió con bastante rapidez a la respuesta correcta. Consideremos una extensión de nuestro micro-internet donde las cosas empiezan a ir mal.

Supongamos que se añade un nuevo sitio web al micro-internet: el sitio web de Geoff. Este sitio web está enlazado por FaceSpace y solo enlaza a sí mismo. (Ver archivo word "Problemas con vectores")

Intuitivamente, solo FaceSpace, que está en la mitad inferior del ranking de páginas, enlaza a este sitio web entre los otros dos a los que enlaza, por lo que podríamos esperar que el sitio de Geoff tenga un puntaje de PageRank correspondientemente bajo.

Construye la nueva matriz L para el micro-internet expandido y usa la Iteración de Potencia en el vector de usuarios indecisos. Veamos qué pasa...
"""

 # Lo llamaremos L2, para diferenciarlo del L anterior.
L2 = np.array([[0,   1/2, 1/3, 0, 0,   0,   0],
               [1/3, 0,   0,   0, 1/2, 0,   0],
               [1/3, 1/2, 0,   1, 0,   1/3, 0],
               [1/3, 0,   1/3, 0, 1/2, 1/3, 0],
               [0,   0,   0,   0, 0,   0,   0],
               [0,   0,   1/3, 0, 0,   0,   0],
               [0,   0,   0,   0, 0,   1/3, 1]])

r = 100 * np.ones(7) / 7 # Configura este vector (6 entradas de 1/6 × Cada 100)
lastR = r
r = L2 @ r
i = 0
while la.norm(lastR - r) > 0.01 :
    lastR = r
    r = L2 @ r
    i += 1
print(str(i) + " iterations to convergence.")
print(r)

"""
Parece que Geoff acapara todo el tráfico en el microinternet y, de alguna manera, se posiciona en el primer lugar del PageRank. Este comportamiento es comprensible, ya que una vez que un "Usuario" llega al sitio web de Geoff, no puede salir, ya que todos los enlaces vuelven a Geoff.

Para combatir esto, podemos agregar una pequeña probabilidad de que los "Usuarios" no sigan ningún enlace en una página web, sino que visiten un sitio web del microinternet al azar. Digamos que la probabilidad de que sigan un enlace es d, por lo tanto, la probabilidad de elegir un sitio web al azar es 1-d. Podemos usar una nueva matriz para calcular cada minuto dónde visitan los usuarios.

M = D L + ((1 - d / n) J)

Donde J es una matriz n×n donde todos los elementos son uno.

Si d es uno, tenemos el mismo caso que vimos anteriormente, mientras que si d es cero, siempre visitaremos una página web al azar y por lo tanto todas las páginas web tendrán la misma probabilidad y serán clasificadas de igual manera. Para que esta extensión funcione mejor, 1-d debería ser relativamente pequeño, aunque no entraremos en una discusión sobre cuán pequeño exactamente.

Volvamos a probar este PageRank con esta extensión.
"""

d = 0.5 # Siéntete libre de jugar con este parámetro después de ejecutar el código una vez.
M = d * L2 + (1-d)/7 * np.ones([7, 7]) # np.ones() es la matriz J, con unos en cada entrada.


r = 100 * np.ones(7) / 7 # Configura este vector (6 entradas de 1/6 × Cada 100)
lastR = r
r = M @ r
i = 0
while la.norm(lastR - r) > 0.01 :
    lastR = r
    r = M @ r
    i += 1
print(str(i) + " iterations to convergence.")
print(r)

"""
Esto definitivamente es una mejora, ya que PageRank asigna valores más razonables a los "usuarios indecisos" que terminan en cada página web. Sin embargo, este método aún predice que la página web de Geoff tiene una clasificación alta. Esto podría verse como una consecuencia del uso de una red pequeña. También podríamos solucionar el problema al no contar los enlaces a sí mismos al crear la matriz L (y si una página web no tiene enlaces salientes, hacer que se enlace a todas las páginas web de manera equitativa). No profundizaremos en esta vía, ya que está dentro del ámbito de las mejoras a PageRank, más que de los problemas de autovalores.

Con la comprensión que has adquirido de PageRank, ahora estás en una buena posición para crear tu propio código para calcular el PageRank de un sitio web con miles de entradas.

Parte 2 - Evaluación

En esta evaluación, se le pedirá que cree una función que pueda calcular el PageRank para una matriz de probabilidad arbitrariamente grande. Esta tarea final del curso proporcionará menos orientación que las evaluaciones anteriores. Se espera que utilice código de secciones anteriores de la hoja de trabajo y lo adapte a sus necesidades.
"""

# PAQUETE
# Aquí están las importaciones nuevamente, por si acaso las necesitas.
# No es necesario editar o enviar esta celda.
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from numpy import arange
from readonly.PageRankFunctions import *
np.set_printoptions(suppress=True)


# FUNCIÓN CALIFICADA
# Completa esta función para proporcionar el PageRank para un internet de tamaño arbitrario.
# Es decir, el eigenvector principal del sistema amortiguado, utilizando el método de iteración de potencia.
# Normalizar columnas para que su suma sea igual a 1 (a menos que no haya enlaces externos, en cuyo caso déjalos como están)
# Los inputs de la función son la linkMatrix y d el parámetro de amortiguación, como se define en esta hoja de trabajo.
# (El parámetro de amortiguación, d, será establecido por la función; no es necesario establecerlo tú mismo).

def pageRank(linkMatrix, d) :
    n = linkMatrix.shape[0]
    M = d * linkMatrix + (1-d)/n * np.ones([n, n]) # np.ones() es la matriz J, con unos en cada entrada
    r = 100 * np.ones(n) / n # Inicializa el vector r con valores iniciales arbitrarios, cada pagina tiene la misma probabilidad, es decir 100/n
    lastR = r
    r = M @ r
    i = 0
    while la.norm(lastR - r) > 0.01 :
        lastR = r
        r = M @ r
        i += 1
    print(str(i) + " iterations to convergence.")
    print(r)
    return r

# Utiliza la siguiente función para generar internets de diferentes tamaños: generate_internet(n)

linkMatrix = generate_internet(10)
d = 0.85
pageRank(linkMatrix, d)

"""
Prueba tu código antes de enviarlo.

Para probar el código que has escrito arriba, ejecuta la celda (selecciona la celda anterior, luego presiona el botón de reproducción [ ▶| ] o presiona shift-enter). Luego puedes usar el código siguiente para probar tu función. No es necesario que envíes esta celda; puedes editarla y ejecutarla tantas veces como quieras.
"""

# Prueba tu método de PageRank contra el método "eig" incorporado.
# Deberías ver que el tuyo es mucho más rápido para internets grandes.


# Ten en cuenta que esto está calculando los valores propios de la matriz de enlaces, L,
# Sin ningún amortiguamiento. Puede dar resultados diferentes a tu función de PageRank.
# Si lo deseas, podrías modificar esta celda para incluir amortiguamiento.
# (Sin embargo, no hay crédito por esto)
eVals, eVecs = la.eig(linkMatrix) # Obtiene los valores y vectores propios
order = np.absolute(eVals).argsort()[::-1] # Los ordena por sus valores propios
eVals = eVals[order]
eVecs = eVecs[:,order]

r = eVecs[:, 0]
100 * np.real(r / np.sum(r))
print(r)
plt.bar(arange(r.shape[0]), r);


# Puede que desees ver el PageRank de forma gráfica.
# Este código dibujará un gráfico de barras, para cada sitio web numerado en el internet generado,
# La altura de cada barra será la puntuación en el PageRank.
# Ejecuta este código para ver el PageRank de cada internet que generes.
# Con suerte, deberías ver lo que podrías esperar
# - hay algunos grupos de sitios web importantes, ¡pero la mayoría en internet son basura!

r = pageRank(generate_internet(100), 0.9)
plt.bar(arange(r.shape[0]), r);