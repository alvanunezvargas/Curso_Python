
"""
"Python para el Análisis Estadístico

La Estadística es una disciplina que se encarga de la colección, organización, visualización, análisis, interpretación y presentación de datos. Es una rama de las Matemáticas recomendada como prerrequisito para incursionar en el mundo de la ciencia de datos y el aprendizaje automático. Aunque la Estadística abarca un amplio espectro, en esta sección nos centraremos en sus aspectos más relevantes. Una vez superado este desafío, podrás elegir entre diversas trayectorias, como el desarrollo web, el análisis de datos, el aprendizaje automático y la ciencia de datos. Independientemente del camino que elijas, en algún momento de tu carrera te toparás con datos que deberás procesar. Poseer conocimientos estadísticos te ayudará a tomar decisiones basadas en datos, como se suele decir, 'los datos hablan'.

Datos

Pero, ¿qué son los datos? Los datos son conjuntos de caracteres recopilados y traducidos con un propósito, generalmente para su análisis. Estos conjuntos pueden incluir texto, números, imágenes, sonido o vídeo. Sin un contexto adecuado, los datos carecen de significado, tanto para un ser humano como para una computadora. Para dar sentido a los datos, es necesario trabajar con ellos empleando diversas herramientas.

El flujo de trabajo en el análisis de datos, la ciencia de datos o el aprendizaje automático comienza con la obtención de datos. Estos datos pueden provenir de una fuente existente o generarse específicamente para el propósito en cuestión. Además, los datos pueden ser estructurados o no estructurados, y su tamaño puede variar desde pequeñas muestras hasta conjuntos masivos. La mayoría de los tipos de datos que manejaremos se han abordado en la sección sobre manejo de archivos.

Módulo Estadística

Python cuenta con un módulo de estadística que proporciona funciones para calcular estadísticas matemáticas a partir de datos numéricos. Es importante destacar que este módulo no pretende competir con las bibliotecas de terceros como NumPy, SciPy o paquetes estadísticos completos diseñados para profesionales en estadística, como Minitab, SAS y Matlab. Más bien, se enfoca en el nivel de las calculadoras gráficas y científicas.

NumPy

En la primera sección, presentamos Python como un lenguaje de programación versátil por sí mismo, pero su verdadero poder se despliega en combinación con otras bibliotecas populares como NumPy, SciPy, Matplotlib, Pandas, entre otras, lo que lo convierte en un entorno poderoso para la computación científica.

NumPy es la biblioteca central para la computación científica en Python, proporcionando un objeto de matriz multidimensional de alto rendimiento y herramientas para trabajar con arrays.

Hasta este punto, hemos estado utilizando Visual Studio Code (VSCode), pero de aquí en adelante, recomiendo el uso de Jupyter Notebook. Para acceder a Jupyter Notebook, vamos a instalar Anaconda. Si estás utilizando Anaconda, la mayoría de los paquetes comunes ya están incluidos, por lo que no necesitarás instalar paquetes adicionales."
"""

# Creación de una matriz numpy:

import numpy as np
        
python_list = [1,2,3,4,5] # (Creación de una lista python)
print('Type:', type (python_list)) # <class 'list'>
print(python_list) # [1, 2, 3, 4, 5]

two_dimensional_list = [[0,1,2], [3,4,5], [6,7,8]]

print(two_dimensional_list)  # [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

# Creación de matriz Numpy(Numérico Python) a partir de lista python
numpy_array_from_list = np.array(python_list)
print(type (numpy_array_from_list))   # <class 'numpy.ndarray'>
print(numpy_array_from_list) # [1 2 3 4 5]


# Creación de una matriz numpy de tipo float: Creación de una matriz numpy float a partir de una lista con un parámetro de tipo de datos float

import numpy as np
        
# Creación de una lista python
python_list = [1,2,3,4,5]

numy_array_from_list2 = np.array(python_list, dtype=float)
print(numy_array_from_list2) # [1. 2. 3. 4. 5.]

# Creación de una matriz numpy booleana: Creación de una matriz numpy booleana a partir de una lista.

import numpy as np
        
# Creación de una lista python
python_list = [0, 1, -1, 0, 0]

numpy_bool_array = np.array(python_list, dtype=bool)
print(numpy_bool_array) # [False  True  True False False]

# Creación de matrices multidimensionales con numpy: Una matriz numpy puede tener una o varias filas y columnas.

import numpy as np
        
two_dimensional_list = [[0,1,2], [3,4,5], [6,7,8]]
numpy_two_dimensional_list = np.array(two_dimensional_list)
print(type (numpy_two_dimensional_list))  # <class 'numpy.ndarray'>
print(numpy_two_dimensional_list)
"""
[[0 1 2]
 [3 4 5]
 [6 7 8]]
"""

# Conversión de una matriz numpy en una lista: Una matriz numpy puede convertirse en una lista con el método tolist().

import numpy as np

python_list = [1,2,3,4,5]
two_dimensional_list = [[0,1,2], [3,4,5], [6,7,8]]

numpy_array_from_list = np.array(python_list)  # (Convertir lista "python_list" en una matriz numpy)
print(type (numpy_array_from_list))  # <class 'numpy.ndarray'>
np_to_list = numpy_array_from_list.tolist()  # (Convertir la matriz numpy "numpy_array_from_list" en una lista)
print(type (np_to_list))  # <class 'list'>
numpy_two_dimensional_list = np.array(two_dimensional_list)  # (Convertir lista "two_dimensional_list" en una matriz numpy)
print(type (numpy_two_dimensional_list))  # <class 'numpy.ndarray'>
np_to_list1 = numpy_two_dimensional_list.tolist()   # (Convertir la matriz numpy "numpy_two_dimensional_list" en una lista)
print(type (np_to_list))  # <class 'list'>
print('one dimensional array:', np_to_list)  # one dimensional array: [1, 2, 3, 4, 5]
print('two dimensional array: ', np_to_list1) # two dimensional array:  [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

# Creación de una matriz numpy a partir de una tupla: Una tupla es una colección ordenada e inmutable de elementos. Una tupla se define mediante paréntesis ().

import numpy as np

python_tuple = (1,2,3,4,5)
print(type (python_tuple)) # <class 'tuple'>
print('python_tuple: ', python_tuple) # python_tuple:  (1, 2, 3, 4, 5)

numpy_array_from_tuple = np.array(python_tuple)
print(type (numpy_array_from_tuple)) # <class 'numpy.ndarray'>
print('numpy_array_from_tuple: ', numpy_array_from_tuple) # numpy_array_from_tuple:  [1 2 3 4 5]

# Forma del matriz numpy: El método shape proporciona la forma de la matriz como una tupla. La primera es la fila y la segunda es la columna. Si la matriz es unidimensional devuelve el tamaño de la matriz.

import numpy as np

two_dimensional_list = [[0,1,2], [3,4,5], [6,7,8]]
numpy_two_dimensional_list = np.array(two_dimensional_list)
nums = np.array([1, 2, 3, 4, 5])
print(nums)  # [1 2 3 4 5]
print('shape of nums: ', nums.shape)  # shape of nums:  (5,)  (Fil, Col)
print(numpy_two_dimensional_list)
"""
[[0 1 2]
 [3 4 5]
 [6 7 8]]
"""
print('shape of numpy_two_dimensional_list: ', numpy_two_dimensional_list.shape)  # shape of numpy_two_dimensional_list:  (3, 3)  (Fil, Col)
three_by_four_array = np.array([[0, 1, 2, 3],
        [4,5,6,7],
        [8,9,10, 11]])
print(three_by_four_array.shape)  # (3, 4) (Fil, Col)

# Tipo de datos de la matriz numpy: Tipos de datos: str, int, float, complex, bool, list, None.

import numpy as np

int_lists = [-3, -2, -1, 0, 1, 2,3]
int_array = np.array(int_lists)
float_array = np.array(int_lists, dtype=float)  # (Tiene los mismos valores de "int_lists" se ha especificado que su tipo de datos sea float dtype = tipo de datos)

print(int_array)  # [-3 -2 -1  0  1  2  3]
print(int_array.dtype)  # int32
print(float_array)  # [-3. -2. -1.  0.  1.  2.  3.]
print(float_array.dtype)  # float64

# Tamaño de una matriz numpy: En numpy para conocer el número de elementos de una matriz numpy usamos size.

import numpy as np

numpy_array_from_list = np.array([1, 2, 3, 4, 5])
two_dimensional_list = np.array([[0, 1, 2],
                              [3, 4, 5],
                              [6, 7, 8]])

print('The size:', numpy_array_from_list.size) # The size: 5
print('The size:', two_dimensional_list.size)  # The size: 9

"""
Operaciones matemáticas con numpy:

NumPy array no es exactamente como python list. Para hacer una operación matemática en una lista de Python tenemos que hacer un bucle a través de los elementos, pero numpy permite hacer cualquier operación matemática sin necesidad de hacer un bucle. Operación matemática:

Suma (+)
Resta (-)
Multiplicación (*)
División (/)
Módulos (%)
División por el suelo(//)
Exponencial(**)
"""

# Suma: Sumar 10 a cada elmento de la matriz numpy.

import numpy as np

numpy_array_from_list = np.array([1, 2, 3, 4, 5])
print('original array: ', numpy_array_from_list)
ten_plus_original = numpy_array_from_list  + 10
print(ten_plus_original)
"""
original array:  [1 2 3 4 5]
[11 12 13 14 15]
"""

# Resta: Restar 10 a cada elmento de la matriz numpy.

import numpy as np

numpy_array_from_list = np.array([1, 2, 3, 4, 5])
print('original array: ', numpy_array_from_list)
ten_minus_original = numpy_array_from_list  - 10
print(ten_minus_original)
"""
original array:  [1 2 3 4 5]
[-9 -8 -7 -6 -5]
"""

# Multiplicación: Multiplicar 10 a cada elmento de la matriz numpy.

import numpy as np

numpy_array_from_list = np.array([1, 2, 3, 4, 5])
print('original array: ', numpy_array_from_list)
ten_times_original = numpy_array_from_list * 10
print(ten_times_original)
"""
original array:  [1 2 3 4 5]
[10 20 30 40 50]
"""

# Division: Dividir 10 a cada elmento de la matriz numpy.

import numpy as np

numpy_array_from_list = np.array([1, 2, 3, 4, 5])
print('original array: ', numpy_array_from_list)
ten_times_original = numpy_array_from_list / 10
print(ten_times_original)
"""
original array:  [1 2 3 4 5]
[0.1 0.2 0.3 0.4 0.5]
"""

# Residuo de la division (Modulus): Obtener el residuo de la división de 10 a cada elmento de la matriz numpy.

import numpy as np

numpy_array_from_list = np.array([1, 2, 3, 4, 5])
print('original array: ', numpy_array_from_list)
ten_times_original = numpy_array_from_list % 3
print(ten_times_original)
"""
original array:  [1 2 3 4 5]
[1 2 0 1 2]
"""

# El resultado de la división sin el residuo: Obtener el resultado de la división sin el residuo de 10 a cada elmento de la matriz numpy.   

import numpy as np

numpy_array_from_list = np.array([1, 2, 3, 4, 5])
print('original array: ', numpy_array_from_list)
ten_times_original = numpy_array_from_list // 10
print(ten_times_original)
"""
original array:  [1 2 3 4 5]
[0 0 0 0 0]
"""

# Exponenecial: Obtener el resultado de la exponencial de 2 a cada elmento de la matriz numpy.

import numpy as np

numpy_array_from_list = np.array([1, 2, 3, 4, 5])
print('original array: ', numpy_array_from_list)
ten_times_original = numpy_array_from_list  ** 2
print(ten_times_original)
"""
original array:  [1 2 3 4 5]
[ 1  4  9 16 25]
"""

# Comprobación de los tipos de datos: Podemos comprobar el tipo de datos con dtype

import numpy as np

numpy_int_arr = np.array([1,2,3,4])
numpy_float_arr = np.array([1.1, 2.0,3.2])
numpy_bool_arr = np.array([-3, -2, 0, 1,2,3], dtype='bool')
print(numpy_int_arr.dtype)  # int32
print(numpy_float_arr.dtype)  # float64
print(numpy_bool_arr.dtype)  # bool

# Conversión de tipos: Podemos convertir los tipos de datos dela matriz numpy

# Int a Float

import numpy as np

numpy_int_arr = np.array([1,2,3,4], dtype='float')
print(numpy_int_arr.dtype)  # float64
print(repr(numpy_int_arr))  # array([1., 2., 3., 4.])  (Obtener una representacion de cadena de la matriz Numpy usando la funcion "repr()")
print(type(numpy_int_arr))  # <class 'numpy.ndarray'>

# Int a booleano

import numpy as np

numpy_int_arr = np.array([-3, -2, 0, 1,2,3], dtype='bool')
print(numpy_int_arr.dtype)  # bool
print(repr(numpy_int_arr))  # array([ True,  True, False,  True,  True,  True])
print(type(numpy_int_arr))  # <class 'numpy.ndarray'>

# Int a Cadena (string)

import numpy as np

numpy_float_list = [1, 2, 3, 4, 5]
numpy_int_arr = np.array(numpy_float_list).astype('int').astype('str')
print(numpy_int_arr.dtype)  # <U11
print(repr(numpy_int_arr))  #  array(['1', '2', '3', '4', '5'], dtype='<U11')
print(type(numpy_int_arr))  #  <class 'numpy.ndarray'>

# Matrices multidimensionales: Podemos crear matrices multidimensionales con numpy.

import numpy as np

two_dimension_array = np.array([(1,2,3),(4,5,6), (7,8,9)])
print(type (two_dimension_array))  # <class 'numpy.ndarray'>
print(two_dimension_array)
"""
[[1 2 3]
 [4 5 6]
 [7 8 9]]
"""
print('Shape: ', two_dimension_array.shape)  # Shape:  (3, 3)
print('Size:', two_dimension_array.size)  # Size: 9
print('Data type:', two_dimension_array.dtype)  # Data type: int32

# Obtención de elementos de una matriz numpy: Podemos obtener elementos de una matriz numpy con el índice.

import numpy as np

two_dimension_array = np.array([[1,2,3],[4,5,6], [7,8,9]])
first_row = two_dimension_array[0]
second_row = two_dimension_array[1]
third_row = two_dimension_array[2]
print('First row:', first_row)  # First row: [1 2 3]
print('Second row:', second_row)  # Second row: [4 5 6]
print('Third row: ', third_row)  # Third row:  [7 8 9]


import numpy as np

two_dimension_array = np.array([[1,2,3],[4,5,6], [7,8,9]])
first_column= two_dimension_array[:,0]  # (Todas las filas y la columna 0)
second_column = two_dimension_array[:,1] # (Todas las fiulas y la columna 1)
third_column = two_dimension_array[:,2]  # (Todas las filas y la columna 2) 
print('First column:', first_column)  # First column: [1 4 7]
print('Second column:', second_column)  # Second column: [2 5 8]
print('Third column: ', third_column) # Third column:  [3 6 9]
print(two_dimension_array)
"""[[1 2 3]
 [4 5 6]
 [7 8 9]]
"""

# Realizar cortes (slicing) en matrices con NumPy

import numpy as np

two_dimension_array = np.array([[1,2,3],[4,5,6], [7,8,9]])
first_two_rows_and_columns = two_dimension_array[0:2, 0:2]  # (Seleccionar las dos primeras filas y las primeras dos columnas)
print(first_two_rows_and_columns)
"""
[[1 2]
 [4 5]]
"""

# Cómo invertir las filas y toda la matriz

import numpy as np

two_dimension_array = np.array([[1,2,3],[4,5,6], [7,8,9]])
# Invertir las filas de la matriz
reversed_rows_array = two_dimension_array[::-1]  # (Seleccionat todas las filas en orden inverso)
# Invertir toda la matriz
reversed_array = two_dimension_array[::-1, ::-1]  # (Seleccionat todas las filas y columnas en orden inverso)
print(reversed_rows_array)
"""
[[7 8 9]
 [4 5 6]
 [1 2 3]]
"""
print(reversed_array)
"""
[[9 8 7]
 [6 5 4]
 [3 2 1]]
"""

# Cómo cambiar valores de la matriz

import numpy as np

two_dimension_array = np.array([[1,2,3],[4,5,6], [7,8,9]])
print(two_dimension_array)
"""
[[1 2 3]
 [4 5 6]
 [7 8 9]]
"""
two_dimension_array[1,1] = 55
two_dimension_array[1,2] =44
print(two_dimension_array)

"""
[[ 1  2  3]
 [ 4 55 44]
 [ 7  8  9]]
"""

# Crear matriz con todas las entradas establecidas en 0

import numpy as np

numpy_zeroes = np.zeros((3,3),dtype=int,order='C')
print(numpy_zeroes)
"""
[[0 0 0]
 [0 0 0]
 [0 0 0]]
 
Este código utiliza la función `np.zeros()` de NumPy para crear una nueva matriz NumPy bidimensional llamado `numpy_zeroes` con todas las entradas establecidas en 0. 

La función `np.zeros()` toma dos argumentos: la forma de la matriz y el tipo de datos de la matriz. En este caso, hemos especificado una forma de `(3,3)` para crear una matriz bidimensional de 3 filas y 3 columnas, y hemos especificado un tipo de datos de `int` para crear una matriz de enteros. 

También hemos especificado el argumento `order='C'` para indicar que la matriz debe ser almacenada en el orden de fila principal (C-style), lo que significa que las filas se almacenan de forma contigua en la memoria.
"""

# Crear matriz con todas las entradas establecidas en 1. Multiplicar cada elmento x 2.

import numpy as np

numpy_ones = np.ones((3,3),dtype=int,order='C')
print(numpy_ones)
"""
[[1 1 1]
 [1 1 1]
 [1 1 1]]
"""
twoes = numpy_ones * 2
print(twoes)
"""
[[2 2 2]
 [2 2 2]
 [2 2 2]]
"""
# Como cambiar el arreglo de la matriz numpy 3x3 a un arreglo 3x2.

import numpy as np

first_shape  = np.array([(1,2,3), (4,5,6)])
print(first_shape)
"""
[[1 2 3]
 [4 5 6]]
"""
reshaped = first_shape.reshape(3,2)
print(reshaped)
"""
[[1 2]
 [3 4]
 [5 6]]
"""

# Como aplanar una matriz numpy bidimensional en una matriz numpy unidimensional.

import numpy as np

first_shape  = np.array([(1,2,3), (4,5,6)])
flattened = first_shape.flatten()
print(flattened)  # [1 2 3 4 5 6]

# Apilar horizontalmente dos matrices numpy.

import numpy as np

np_list_one = np.array([1,2,3])
np_list_two = np.array([4,5,6])
print(np_list_one + np_list_two)  # [5 7 9]
print('Horizontal Append:', np.hstack((np_list_one, np_list_two)))  # Horizontal Append: [1 2 3 4 5 6]

# o

import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
c = np.hstack((a, b))
print(c)
"""
[[1 2 5 6]
 [3 4 7 8]]
"""

# Apilar verticalmente dos matrices numpy.

import numpy as np

np_list_one = np.array([1,2,3])
np_list_two = np.array([4,5,6])
print('Vertical Append:', np.vstack((np_list_one, np_list_two)))
"""
Vertical Append: [[1 2 3]
 [4 5 6]]
"""
# o

import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
c = np.vstack((a, b))
print(c)
"""
[[1 2]
 [3 4]
 [5 6]
 [7 8]]
"""

# Apilar matrices numpy en profundidad.

import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
c = np.dstack((a, b))
print(c)
"""
[[[1 5]
  [2 6]]

 [[3 7]
  [4 8]]]
"""

# Generar numeros aleatorios con numpy.

import numpy as np

random_float = np.random.random()
print(random_float)  # 0.7928290133043481

# o

import numpy as np

random_floats = np.random.random(5)
print(random_floats)  # [0.96365374 0.3956329  0.20047189 0.31315143 0.06052349]

# Generar un número entero aleatorio entre 0 y 10.

import numpy as np

random_int = np.random.randint(0, 11)
print(random_int)  # 8

# Generando un entero aleatorio entre 2 y 9, y creando una matriz de una fila o unidimensional.

import numpy as np

random_int = np.random.randint(2,10, size=4)
print(random_int) # [9 6 4 3]

# Generando enteros aleatorios entre 0 y 9, en una matriz de tres filas o tridimensional.


import numpy as np

random_int = np.random.randint(2,10, size=(3,3))
print(random_int)
"""
[[6 5 8]
 [4 4 5]
 [7 6 3]]
"""

# Generar una matriz unidimensional que contiene 80 numeros aleatorios en una distribución normal con una media de 79 y una desviación estándar de 15.

import numpy as np

normal_array = np.random.normal(79, 15, 80)
print(normal_array)
"""
[ 82.28171831  88.67982551 103.14688451  81.62974123  81.90545184
  71.77548936  85.98330679  67.91488909  72.6544264   79.53187352
  54.79547747  65.18094434  73.29424481  73.32919138  62.67703491
  77.92225161  82.15254204  68.84131987  89.14936554  82.85107674
 106.19630266  68.35279477  96.30475369 103.80028307  82.53874137
  78.64326425 102.35832805  82.39427785  81.65648906  80.82970664
  61.41189308  91.55603856  79.58040551  89.71965762 100.18718079
  84.90466457  77.87267975  77.82040842  92.19988494  72.86141569
  81.16133795  82.9497546   87.11237591  72.64901602  78.44805021
  73.39274559  83.4845996   80.87135486  73.66506603  91.62092883
  55.83666946 104.71705118 103.66675507  93.88967903  87.52119062
  87.11631592  82.8703907   74.18314358  65.7161256   74.3943861
  95.72168942  63.24168554  85.4379992   99.42987786  97.22332027
  81.38654919  80.68844661  89.67249286  90.35297279  64.62122683
  70.00391803  85.31445792  82.19446903  75.60673042  60.22562648
  61.65192617  92.51630276  42.82104831  63.83437892  77.3691235 ]
"""

# Visualizar la distribución de un conjunto de datos utilizando un histograma.

import numpy as np

normal_array = np.random.normal(79, 15, 80)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
plt.hist(normal_array, color="grey", bins=50)
plt.show()

"""
Primero, se importan las bibliotecas Matplotlib y Seaborn utilizando las sentencias `import matplotlib.pyplot as plt` y `import seaborn as sns`. Luego, se utiliza la función `sns.set()` para establecer el estilo predeterminado de Seaborn para los gráficos.

A continuación, se utiliza la función `plt.hist()` de Matplotlib para crear un histograma dela matriz NumPy `normal_array`. La función `plt.hist()` toma varios argumentos, incluyendo el array NumPy que se va a graficar, el número de contenedores (bins) para el histograma y el color de las barras del histograma.

En este caso, hemos especificado la matriz NumPy `normal_array` como el primer argumento, un número de contenedores de 50 y el color gris para las barras del histograma.
"""

# Matrices en numpy

import numpy as np

four_by_four_matrix = np.matrix(np.ones((4,4), dtype=float))
print(four_by_four_matrix)
"""
[[1. 1. 1. 1.]
 [1. 1. 1. 1.]
 [1. 1. 1. 1.]
 [1. 1. 1. 1.]]
"""
np.asarray(four_by_four_matrix)[2] = 2  # (np.asarray convierte lista de python en una matriz; se accede a la tercera fila (indice 2) y se asigna el valor de 2)
print(four_by_four_matrix)
"""
[[1. 1. 1. 1.]
 [1. 1. 1. 1.]
 [2. 2. 2. 2.]
 [1. 1. 1. 1.]]
"""

"""
Numpy numpy.arange(): Es una función que crea una matriz de NumPy con números enteros equiespaciados. La función tiene tres parámetros:

start: El número inicial del array.
stop: El número final del array.
step: El incremento entre números. El valor predeterminado es 1.
Por ejemplo, el siguiente código crea un array de NumPy con los números del 0 al 9:
"""

"""
Numpy numpy.arange()" es una función que crea un array de NumPy con números enteros equiespaciados. La función tiene tres parámetros:

start: El número inicial del array.
stop: El número final del array.
step: El incremento entre números. El valor predeterminado es 1.
"""


lst = range(0, 11, 2)
print(list(lst))  # [0, 2, 4, 6, 8, 10]    (Se usa "list" para convertir la funcion lst en una lista)
for l in lst:
    print(l)
"""
0
2
4
6
8
10
"""
print(type(l))  # <class 'int'>

# De manara analoga se puede obtener el mismo resultado con numpy.arange().

import numpy as np
whole_numbers = np.arange(0, 11, 2)
print(type(whole_numbers))  # <class 'numpy.ndarray'>
print(whole_numbers)  # [ 0  2  4  6  8 10]

# Otros ejemplos:

import numpy as np

numeros = np.arange(0, 20)
print(numeros)  # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]


import numpy as np

numeros = np.arange(0, 20, 2)
print(numeros)  # [ 0  2  4  6  8 10 12 14 16 18]


import numpy as np

numeros_impares = np.arange(1, 20, 2)
print(numeros_impares)  # [ 1  3  5  7  9 11 13 15 17 19]
numeros_pares = np.arange(2, 20, 2)
print(numeros_pares)  # [ 2  4  6  8 10 12 14 16 18]

# Creación de secuencias de números mediante linspace: Se puede utilizar para crear 10 valores de 1 a 5 espaciados uniformemente.

import numpy as np

print(np.linspace(1.0, 5.0, num=10))  
"""
[1.         1.44444444 1.88888889 2.33333333 2.77777778 3.22222222
 3.66666667 4.11111111 4.55555556 5.        ]
 """

# Ahora veamos como no incluir el último valor del intervalo.

import numpy as np

print(np.linspace(1.0, 5.0, num=10, endpoint=False)) # [1.  1.4 1.8 2.2 2.6 3.  3.4 3.8 4.2 4.6]

"""
numpy.logspace(): es una función que crea un array de NumPy con números equiespaciados en escala logarítmica. La función tiene estos parámetros:

numpy.logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None)

start: El valor inicial del rango.
stop: El valor final del rango.
num: Número de muestras a generar. Por defecto es 50.
endpoint: Si es verdadero, se incluye el valor final en el rango. Por defecto es verdadero.
base: La base de la secuencia logarítmica. Por defecto es 10.0.
dtype: El tipo de datos de la matriz devuelta. Por defecto es None.
"""

import numpy as np

print(np.logspace(2, 4.0, num=4))  # [  100.           464.15888336  2154.43469003 10000.        ]

# Comprobar el tamaño de una matriz.

import numpy as np

x = np.array([1,2,3], dtype=np.complex128)
print(x)  # [1.+0.j 2.+0.j 3.+0.j]
print(x.itemsize)  # 16  (Imprimir el tamaño de cada elmento de la matriz en bytes, debido a que cada numero complejo de 128 bits, ocupa 16 bytes de memoria)

# indexación y segmentación de matrices NumPy en Python.

import numpy as np

np_list = np.array([(1,2,3), (4,5,6)])
print(np_list)
"""
[[1 2 3]
 [4 5 6]]
"""
print('First row: ', np_list[0])  # First row:  [1 2 3]
print('Second row: ', np_list[1])  # Second row:  [4 5 6]
print('First column: ', np_list[:,0])  # First column:  [1 4]
print('Second column: ', np_list[:,1])  # Second column:  [2 5]
print('Third column: ', np_list[:,2])  # Third column:  [3 6]

"""
Funciones Estadísticas de NumPy: NumPy tiene funciones estadísticas muy útiles para encontrar el mínimo, máximo, media, desviacion estandar, mediana, varianza y percentiles de los elementos dados en la matriz. 
"""

import numpy as np

np_normal_dis = np.random.normal(5, 0.5, 100)
#  Mínimo, máximo, media, desviacion estandar, mediana, varianza y percentiles
print('min: ', np_normal_dis.min())  # min:  3.4679345994243733
print('max: ', np_normal_dis.max())  # max:  6.632742085433915
print('mean: ',np_normal_dis.mean())  # mean:  4.951944620171893
print('sd: ', np_normal_dis.std())  # sd:  0.5431523441790276
print('median: ', np.median(np_normal_dis))  # median:  5.007369985571231
print('variance: ', np.var(np_normal_dis))  # variance:  0.22239960923791682
print('25th percentile: ', np.percentile(np_normal_dis, 25))  # 25th percentile:  4.720907910063371
print('50th percentile: ', np.percentile(np_normal_dis, 50))  # 50th percentile:  4.965821992010044
print('75th percentile: ', np.percentile(np_normal_dis, 75))  # 75th percentile:  5.3450209698064715

# Estadistica en arreglos multidimensionales.

import numpy as np

two_dimension_array = np.array([(1,2,3),(4,5,6), (7,8,9)])
#  Mínimo, máximo, media, desviacion estandar, mediana, varianza y percentiles
print('min: ', two_dimension_array.min())  # min:  1
print('max: ', two_dimension_array.max())  # max:  9
print('mean: ',two_dimension_array.mean())  # mean:  5.0
print('sd: ', two_dimension_array.std())  # sd:  2.581988897471611
print('median: ', np.median(two_dimension_array))  # median:  5.0
print('variance: ', np.var(two_dimension_array))  # variance:  6.666666666666667
print('25th percentile: ', np.percentile(two_dimension_array, 25))  # 25th percentile:  3.0
print('50th percentile: ', np.percentile(two_dimension_array, 50))  # 50th percentile:  5.0
print('75th percentile: ', np.percentile(two_dimension_array, 75))  # 75th percentile:  7.0


import numpy as np

two_dimension_array = np.array([(1,2,3),(4,5,6), (7,8,9)])
print('Column with minimum: ', np.amin(two_dimension_array,axis=0))  # Column with minimum:  [1 2 3]
print('Column with maximum: ', np.amax(two_dimension_array,axis=0))  # Column with maximum:  [7 8 9]
print('Row with minimum: ', np.amin(two_dimension_array,axis=1))  # Row with minimum:  [1 4 7]
print('Row with maximum: ', np.amax(two_dimension_array,axis=1))  # Row with maximum:  [3 6 9]


# Cómo crear secuencias repetidas.

import numpy as np

a = [1,2,3]

# Repetir la totalidad de 'a' dos veces
print('Tile: ', np.tile(a, 2))  # Tile:  [1 2 3 1 2 3]

# Repetir cada elemento de 'a' dos veces
print('Repeat: ', np.repeat(a, 2))  # Repeat:  [1 1 2 2 3 3]

# Cómo generar números aleatorios.

import numpy as np

one_random_num = np.random.random()  # (Numero aleatorio entre 0 y 1)
print(one_random_num)  # 0.8357240026610266

# Números aleatorios entre [0,1) de forma 2,3

import  numpy as np

r = np.random.random(size=[2,3])
print(r)
"""
[[0.99673474 0.73073869 0.21092296]
 [0.96814211 0.99056842 0.25033231]]
 
La función `np.random.random()` genera números aleatorios en el rango de 0 a 1 utilizando una distribución uniforme continua. Esto significa que cada número generado tiene la misma probabilidad de estar en cualquier lugar del rango de 0 a 1.
"""
# Números aleatorios entre [0,1) de forma 2,2

import  numpy as np

rand = np.random.rand(2,2)
print(rand)
"""
[[0.96249004 0.26997127]
 [0.59028376 0.53359834]]
 
La función `np.random.rand()` también genera números aleatorios en el rango de 0 a 1. La principal diferencia entre `np.random.random(size=[2,2])` y `np.random.rand(2,2)` es la forma en que se especifican las dimensiones de la matriz que se desea crear. En `np.random.random(size=[2,2])`, se especifica el tamaño de la matriz como un argumento de la función en forma de una lista `[2,2]`. En `np.random.rand(2,2)`, se especifican las dimensiones de la matriz como argumentos separados por comas `2,2`.
"""
print(rand.shape)  # (2, 2)

# Números aleatorios de forma 2,2

import  numpy as np

rand2 = np.random.randn(2,2)
print(rand2)
"""
[[-0.31307994 -0.73810265]
 [-0.87741911 -1.8987479 ]]
 
La función `np.random.randn()` genera números aleatorios a partir de una distribución normal estándar, que es una distribución con una media de cero y una desviación estándar de uno. A diferencia de `np.random.random()` y `np.random.rand()`, que generan números aleatorios en un rango específico, `np.random.randn()` genera números aleatorios a partir de una distribución normal estándar, lo que significa que los números generados pueden ser positivos o negativos y pueden estar lejos de cero.
"""
print(rand2.shape)  # (2, 2)

# Generar una lista de 10 caracteres aleatorios seleccionados de una lista.

import  numpy as np

print(np.random.choice(['a', 'e', 'i', 'o', 'u'], size=10))  # ['e' 'i' 'i' 'e' 'e' 'u' 'i' 'a' 'e' 'e']

# Generar una lista de 10 caracteres aleatorios seleccionados de una lista con una probabilidad específica para cada vocal

import  numpy as np

print(np.random.choice(['a', 'e', 'i', 'o', 'u'], size=10, p=[0.3, 0.1, 0.1, 0.4, 0.1]))
# ['a' 'a' 'a' 'a' 'o' 'a' 'o' 'e' 'e' 'u']

"""
En el argumento `p`, se especifica la probabilidad de cada elemento en la lista de elementos. En este caso, la probabilidad de 'a' es 0.3, la probabilidad de 'e' es 0.1, la probabilidad de 'i' es 0.1, la probabilidad de 'o' es 0.4 y la probabilidad de 'u' es 0.1. Esto significa que 'o' tiene la mayor probabilidad de ser seleccionado, seguido de 'a', 'e', 'i' y 'u'
"""

# Generar una matriz de forma 5,3 de números enteros aleatorios en el rango de 0 a 9.

import  numpy as np

rand_int = np.random.randint(0, 10, size=[5,3])
print(rand_int)
"""
[[9 5 8]
 [0 4 9]
 [1 8 7]
 [3 4 6]
 [0 4 1]]
"""

# Generar matriz, calcular la estadistica y graficar histograma.

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

np_normal_dis = np.random.normal(5, 0.5, 1000)
np_normal_dis
## min, max, mean, median, sd  Mínimo, máximo, media, mediana, moda y desviacion estandar
print('min: ', np.min(np_normal_dis))  # min:  3.3607736976089955
print('max: ', np.max(np_normal_dis))  # max:  6.649935329271893
print('mean: ', np.mean(np_normal_dis))  # mean:  5.003687597653529
print('median: ', np.median(np_normal_dis))  # median:  5.011490029081008
print('mode: ', stats.mode(np_normal_dis))  # mode:  ModeResult(mode=3.3607736976089955, count=1)
print('sd: ', np.std(np_normal_dis))  # sd:  0.4911350373777632
plt.hist(np_normal_dis, color="grey", bins=21)
plt.show()

# Álgebra lineal
# Producto escalar: producto de dos matrices

import numpy as np

f = np.array([1,2,3])
g = np.array([4,5,3])
# 1*4+2*5+3*3 = 23
print(np.dot (f, g)) # 23

# Multiplicación de matrices:

import numpy as np

h = [[1,2],[3,4]]
i = [[5,6],[7,8]]
"""
1*5+2*7 = 19
1*6+2*8 = 22
3*5+4*7 = 43
3*6+4*8 = 50
"""
print(np.matmul(h, i))
"""
[[19 22]
 [43 50]]
"""
# El determinante de una matriz 2*2 se calcula de la siguiente manera

import numpy as np

i = [[5,6],[7,8]]
# 5*8-7*6 = -2
print(np.linalg.det(i))  # -2.000000000000005"

# Solución de un sistema de ecuaciones lineales.

import numpy as np

# Sistema de dos ecuaciones y dos incógnitas
# x + 2y = 1
# 3x + 5y = 2 
a = np.array([[1, 2], [3, 5]])
b = np.array([1, 2])
print(np.linalg.solve(a, b)) # [-1.  1.]


# Crea una matriz de ajedrez de 8x8 con ceros y unos utilizando la indexación de matriz.

import numpy as np

Z = np.zeros((8,8))
Z[1::2,::2] = 1
Z[::2,1::2] = 1
print(Z)
"""
[[0. 1. 0. 1. 0. 1. 0. 1.]
 [1. 0. 1. 0. 1. 0. 1. 0.]
 [0. 1. 0. 1. 0. 1. 0. 1.]
 [1. 0. 1. 0. 1. 0. 1. 0.]
 [0. 1. 0. 1. 0. 1. 0. 1.]
 [1. 0. 1. 0. 1. 0. 1. 0.]
 [0. 1. 0. 1. 0. 1. 0. 1.]
 [1. 0. 1. 0. 1. 0. 1. 0.]]

El código proporcionado crea una matriz NumPy de 8x8 llamada `Z` y la inicializa con ceros utilizando la función `np.zeros()`. Luego, se asigna el valor 1 a ciertos elementos de la matriz `Z` utilizando la indexación de matriz.

La primera línea de indexación `Z[1::2,::2] = 1` asigna el valor 1 a los elementos de la matriz `Z` que se encuentran en filas impares y columnas pares. La notación `1::2` significa que se seleccionan todas las filas a partir de la segunda fila (índice 1) y se salta una fila entre cada selección. La notación `::2` significa que se seleccionan todas las columnas y se salta una columna entre cada selección. Por lo tanto, esta línea de código asigna el valor 1 a los elementos de la matriz `Z` que se encuentran en las filas impares y las columnas pares.

La segunda línea de indexación `Z[::2,1::2] = 1` asigna el valor 1 a los elementos de la matriz `Z` que se encuentran en filas pares y columnas impares. La notación `::2` significa que se seleccionan todas las filas y se salta una fila entre cada selección. La notación `1::2` significa que se seleccionan todas las columnas a partir de la segunda columna (índice 1) y se salta una columna entre cada selección. Por lo tanto, esta línea de código asigna el valor 1 a los elementos de la matriz `Z` que se encuentran en las filas pares y las columnas impares.
"""

# Crear una lista de números enteros consecutivos que se desplacen en dos unidades.

new_list = [ x + 2 for x in range(0, 11)]
print(new_list) # [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

"""
Crea una nueva lista llamada `new_list` utilizando una comprensión de lista en Python. La comprensión de lista utiliza la sintaxis `[expresión for variable in iterable]` para crear una nueva lista a partir de un iterable.

En este caso, la expresión es `x + 2`, que suma 2 a cada elemento `x` del iterable. El iterable es `range(0, 11)`, que crea una secuencia de números enteros desde 0 hasta 10 (11 no está incluido). Por lo tanto, la comprensión de lista crea una nueva lista que contiene los números enteros desde 2 hasta 12.
"""

# Crear una lista de números enteros consecutivos que se desplacen en dos unidades con numpy.

import numpy as np

np_arr = np.array(range(0, 11))
print(np_arr) # [ 0  1  2  3  4  5  6  7  8  9 10]
np_arr1 = np.array(range(0, 11)) + 2
print(np_arr1)  # [ 2  3  4  5  6  7  8  9 10 11 12]

# Ecuaciones lineales.

import numpy as np
import matplotlib.pyplot as plt

temp = np.array([1,2,3,4,5])
pressure = temp * 2 + 5
print(pressure)  # [ 7  9 11 13 15]

plt.plot(temp,pressure)
plt.xlabel('Temperature in °C')
plt.ylabel('Pressure in atm')
plt.title('Temperature vs Pressure')
plt.xticks(np.arange(0, 6, step=0.5))
plt.show()

# Graficar una distribución normal gaussiana usando numpy: Numpy puede generar números aleatorios. Para crear una muestra aleatoria, necesitamos la media (mu), sigma (desviación estándar) y una cantidad de datos.

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

mu = 28
sigma = 15
samples = 100000

x = np.random.normal(mu, sigma, samples)
ax = sns.histplot(x)
ax.set(xlabel="x", ylabel='y')
plt.show()

"""
Para resumir, las principales diferencias con las listas de Python son:

Las matrices admiten operaciones vectorizadas, mientras que las listas no.
Una vez que se crea una matriz, no puede cambiar su tamaño. Tendrás que crear una nueva matriz o sobrescribir la existente.
Cada matriz tiene un solo tipo de dtype. Todos los elementos que contiene deben ser de ese tipo.
Una matriz numpy equivalente ocupa mucho menos espacio que una lista de Python.
Las matrices admiten la indexación booleana. Veamos el siguiente ejemplo:

import numpy as np

# Crear un array
arr = np.array([1, 2, 3, 4, 5])

# Crear una máscara booleana basada en una condición
mask = arr > 3

# Aplicar la máscara para seleccionar los elementos que cumplen con la condición
filtered_arr = arr[mask]

print(filtered_arr)  # Resultado: [4, 5]
"""