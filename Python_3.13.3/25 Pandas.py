"""
Pandas es una herramienta de análisis de datos y estructuras de datos de código abierto, de alto rendimiento y fácil de usar para el lenguaje de programación Python. Pandas agrega estructuras de datos y herramientas diseñadas para trabajar con datos similares a tablas, que son series y marcos de datos. Pandas proporciona herramientas para la manipulación de datos:

Reorganización
Fusionando
Clasificación
Rebanar
Agregación
Imputación

La estructura de datos de Pandas se basa en Series y DataFrames.

Una serie es una columna y un DataFrame es una tabla multidimensional formada por una colección de series. Para crear una serie de pandas, debemos usar numpy para crear matrices unidimensionales o una lista de Python.

Nombres Serie Pandas

	Nombre
0	Alvaro
1	Claudia
2	Nicolas
3	Sebastian

	Pais
0	Colombia
1	USA
2	Nueva Zelanda
3	Inglaterra

	Ciudad
0	Armenia
1	Miami
2	Wellington
3	Londres

Como se puede observar, una serie de Pandas representa únicamente una columna de datos. Sin embargo, si deseamos trabajar con múltiples columnas, recurrimos a los DataFrames. A continuación, se muestra un ejemplo de DataFrames de Pandas:


	Nombre	     Pais	        Ciudad	    Peso	Altura
0	Alvaro	   Colombia	        Armenia	     75	     170
1	Claudia	     USA	         Miami	     62	     158
2	Nicolas	   Nueva Zelanda   Wellington	 82	     180
3	Sebastian	Inglaterra	    Londres	     72	     175

A continuación, veremos cómo importar pandas y cómo crear Series y DataFrames usando pandas.
"""
import pandas as pd

nums = [1, 2, 3, 4,5]
s = pd.Series(nums)
print(s)
"""
0    1
1    2
2    3
3    4
4    5
dtype: int64
"""
# Creando la serie Pandas con índice personalizado

import pandas as pd

nums = [1, 2, 3, 4, 5]
s = pd.Series(nums, index=[1, 2, 3, 4, 5])
print(s)
"""
1    1
2    2
3    3
4    4
5    5
dtype: int64
"""

import pandas as pd

fruits = ['Orange','Banana','Mango']
fruits = pd.Series(fruits, index=[1, 2, 3])
print(fruits)
"""
1    Orange
2    Banana
3     Mango
dtype: object
"""

# Crear series Pandas a partir de un diccionario

import pandas as pd

dct = {'name':'Alvaro','country':'Colombia','city':'Armenia'}
s = pd.Series(dct)
print(s)
"""
name         Alvaro
country    Colombia
city        Armenia
dtype: object
"""

# Creando una serie de Pandas constantes

import pandas as pd

s = pd.Series(10, index = [1, 2, 3])
print(s)
"""
1    10
2    10
3    10
dtype: int64
"""

# Creando una serie Pandas usando Linspace

import pandas as pd
import numpy as np

s = pd.Series(np.linspace(5, 20, 10)) # linspace (inicio, fin, numero elementos)
print(s)
"""
0     5.000000
1     6.666667
2     8.333333
3    10.000000
4    11.666667
5    13.333333
6    15.000000
7    16.666667
8    18.333333
9    20.000000
dtype: float64
"""

"""
Crear una serie de Pandas utilizando Linspace DataFrames: Los marcos de datos de Pandas son tablas de datos bidimensionales con columnas y filas. Los marcos de datos de Pandas se pueden crear de diferentes maneras.

Crear marcos de datos a partir de una lista de listas
"""

import pandas as pd

data = [
    ['Alvaro', 'Colombia', 'Armenia'], 
    ['Claudia', 'UK', 'London'],
    ['Nicolas', 'Sweden', 'Stockholm'],
    ['Sebastian', 'New Zeland', 'Wellington']
]
df = pd.DataFrame(data, columns=['Names','Country','City'])
print(df)
"""
       Names     Country        City
0     Alvaro    Colombia     Armenia
1    Claudia          UK      London
2    Nicolas      Sweden   Stockholm
3  Sebastian  New Zeland  Wellington
"""

# Crear un dataframe usando diccionarios

import pandas as pd

data = {'Nombre': ['Alvaro', 'Claudia', 'Nicolas', 'Sebastian'], 'Pais':[
    'Colombia', 'UK', 'Sweden', 'New Zeland'], 'Ciudad': ['Armenia', 'Londres', 'Stockholm', 'Wellington']}
df = pd.DataFrame(data)
print(df)
"""
      Nombre        Pais      Ciudad
0     Alvaro    Colombia     Armenia
1    Claudia          UK     Londres
2    Nicolas      Sweden   Stockholm
3  Sebastian  New Zeland  Wellington
"""

# Lectura de archivos CSV usando Pandas

import pandas as pd

df = pd.read_csv(r'C:\Users\alvan\Downloads\weight-height.csv')
print(df)
"""
      Gender     Height      Weight
0       Male  73.847017  241.893563
1       Male  68.781904  162.310473
2       Male  74.110105  212.740856
3       Male  71.730978  220.042470
4       Male  69.881796  206.349801
...      ...        ...         ...
9995  Female  66.172652  136.777454
9996  Female  67.067155  170.867906
9997  Female  63.867992  128.475319
9998  Female  69.034243  163.852461
9999  Female  61.944246  113.649103

[10000 rows x 3 columns]
"""
# Exploración de datos: Leamos solo las primeras 5 filas usando head()

import pandas as pd

df = pd.read_csv(r'C:\Users\alvan\Downloads\weight-height.csv')
print(df.head())  # (Imprime las primeras 5 filas del dataframe, podemos aumentar las filas que queremos ver agregando un numero dentro de los parentesis)
"""
  Gender     Height      Weight
0   Male  73.847017  241.893563
1   Male  68.781904  162.310473
2   Male  74.110105  212.740856
3   Male  71.730978  220.042470
4   Male  69.881796  206.349801
"""

# Exploremos también las últimas actualizaciona del dataframe utilizando los métodos tail().

import pandas as pd

df = pd.read_csv(r'C:\Users\alvan\Downloads\weight-height.csv')
print(df.tail()) # (Imprime las ultimas 5 filas del dataframe, podemos aumentar las filas que queremos ver agregando un numero dentro de los parentesis)
"""
      Gender     Height      Weight
9995  Female  66.172652  136.777454
9996  Female  67.067155  170.867906
9997  Female  63.867992  128.475319
9998  Female  69.034243  163.852461
9999  Female  61.944246  113.649103
"""

"""
Como puede ver, el archivo csv tiene tres filas: Gnder, Height y Weight. Si el DataFrame tuviera filas largas, sería difícil conocer todas las columnas. Por tanto, deberíamos utilizar un método para conocer las columnas. No sabemos el número de filas. Utilicemos el método de la forma.
"""
import pandas as pd

df = pd.read_csv(r'C:\Users\alvan\Downloads\weight-height.csv')
print(df.shape)  # (10000, 3)

# Consigamos todas las columnas usando columns

import pandas as pd

df = pd.read_csv(r'C:\Users\alvan\Downloads\weight-height.csv')
print(df.columns)  # Index(['Gender', 'Height', 'Weight'], dtype='object')


# Ahora, obtengamos una columna específica usando la clave de column.

import pandas as pd

df = pd.read_csv(r'C:\Users\alvan\Downloads\weight-height.csv')
heights = df['Height'] 
print(heights)
"""
0       73.847017
1       68.781904
2       74.110105
3       71.730978
4       69.881796
          ...    
9995    66.172652
9996    67.067155
9997    63.867992
9998    69.034243
9999    61.944246
Name: Height, Length: 10000, dtype: float64
"""
weights = df['Weight']
print(weights)
"""
0       241.893563
1       162.310473
2       212.740856
3       220.042470
4       206.349801
           ...    
9995    136.777454
9996    170.867906
9997    128.475319
9998    163.852461
9999    113.649103
Name: Weight, Length: 10000, dtype: float64
"""
print(len(heights) == len(weights))  # True

# El método describe() proporciona valores estadísticos descriptivos de un conjunto de datos.

import pandas as pd

df = pd.read_csv(r'C:\Users\alvan\Downloads\weight-height.csv')
heights = df['Height']
print(heights.describe())
"""
count    10000.000000
mean        66.367560
std          3.847528
min         54.263133
25%         63.505620
50%         66.318070
75%         69.174262
max         78.998742
Name: Height, dtype: float64
"""
weights = df['Weight']
print(weights.describe())
"""
count    10000.000000
mean       161.440357
std         32.108439
min         64.700127
25%        135.818051
50%        161.212928
75%        187.169525
max        269.989699
Name: Weight, dtype: float64
"""

# describe() también puede proporcionar información estadística del dataframe

import pandas as pd

df = pd.read_csv(r'C:\Users\alvan\Downloads\weight-height.csv')
print(df.describe())
"""
             Height        Weight
count  10000.000000  10000.000000
mean      66.367560    161.440357
std        3.847528     32.108439
min       54.263133     64.700127
25%       63.505620    135.818051
50%       66.318070    161.212928
75%       69.174262    187.169525
max       78.998742    269.989699
"""

# La función `info()` muestra el número de filas y columnas del DataFrame, el nombre y el tipo de datos de cada columna, y la cantidad de valores no nulos en cada columna.

import pandas as pd

df = pd.read_csv(r'C:\Users\alvan\Downloads\weight-height.csv')
print(df.info())
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10000 entries, 0 to 9999
Data columns (total 3 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   Gender  10000 non-null  object 
 1   Height  10000 non-null  float64
 2   Weight  10000 non-null  float64
dtypes: float64(2), object(1)
memory usage: 234.5+ KB
None
"""

"""
Modificar un dataframe: Podemos crear un nuevo DataFrame, podemos crear una nueva columna y agregarla al DataFrame, podemos eliminar una columna existente de un DataFrame, podemos modificar una columna existente en un DataFrame, podemos cambiar el tipo de datos de los valores de las columnas en el DataFrame

Creando un dataframe: Importamos los paquetes pandas y numpy,
"""
import pandas as pd
import numpy as np
data = [
    {"Name": "Alvaro", "Country":"Colombia","City":"Armenia"},
    {"Name": "Claudia", "Country":"Colombia","City":"Calarca"},
    {"Name": "Nicolas", "Country":"Colombia","City":"Armenia"},
    {"Name": "Sebastian", "Country":"Colombia","City":"Bucaramanga"}]
df = pd.DataFrame(data)
print(df)
"""
        Name   Country         City
0     Alvaro  Colombia      Armenia
1    Claudia  Colombia      Calarca
2    Nicolas  Colombia      Armenia
3  Sebastian  Colombia  Bucaramanga
"""

"""
Agregar una columna a un DataFrame es como agregar una clave a un diccionario.

Primero usemos el ejemplo anterior para crear un DataFrame. Después de crear el DataFrame, comenzaremos a modificar las columnas y los valores de las columnas.

Agregar una nueva columna: Agreguemos una columna Wight en el DataFrame
"""
import pandas as pd
import numpy as np
data = [
    {"Name": "Alvaro", "Country":"Colombia","City":"Armenia"},
    {"Name": "Claudia", "Country":"Colombia","City":"Calarca"},
    {"Name": "Nicolas", "Country":"Colombia","City":"Armenia"},
    {"Name": "Sebastian", "Country":"Colombia","City":"Bucaramanga"}]
df = pd.DataFrame(data)
weights = [75, 62, 80, 72]
df['Weight'] = weights
print(df)
"""
        Name   Country         City  Weight
0     Alvaro  Colombia      Armenia      75
1    Claudia  Colombia      Calarca      62
2    Nicolas  Colombia      Armenia      80
3  Sebastian  Colombia  Bucaramanga      72
"""

# Agreguemos también una columna de Height en el DataFrame

import pandas as pd
import numpy as np
data = [
    {"Name": "Alvaro", "Country":"Colombia","City":"Armenia"},
    {"Name": "Claudia", "Country":"Colombia","City":"Calarca"},
    {"Name": "Nicolas", "Country":"Colombia","City":"Armenia"},
    {"Name": "Sebastian", "Country":"Colombia","City":"Bucaramanga"}]
df = pd.DataFrame(data)
weights = [75, 62, 80, 72]
df['Weight'] = weights
heights = [170, 160, 180, 175]
df['Height'] = heights
print(df)
"""
        Name   Country         City  Weight  Height
0     Alvaro  Colombia      Armenia      75     170
1    Claudia  Colombia      Calarca      62     160
2    Nicolas  Colombia      Armenia      80     180
3  Sebastian  Colombia  Bucaramanga      72     175
"""

"""
Agreguemos una columna adicional llamada BMI (índice de masa corporal) calculando su BMI utilizando su masa y altura. El BMI es la masa dividida por la altura al cuadrado (en metros) - Peso/Altura * Altura.

Como ves, la altura está en centímetros, por lo que deberíamos cambiarla a metros. Modifiquemos la fila de altura.
"""
import pandas as pd
import numpy as np
data = [
    {"Name": "Alvaro", "Country":"Colombia","City":"Armenia"},
    {"Name": "Claudia", "Country":"Colombia","City":"Calarca"},
    {"Name": "Nicolas", "Country":"Colombia","City":"Armenia"},
    {"Name": "Sebastian", "Country":"Colombia","City":"Bucaramanga"}]
df = pd.DataFrame(data)
weights = [75, 62, 80, 72]
df['Weight'] = weights
heights = [170, 160, 180, 175]
df['Height'] = heights
df['Height'] = df['Height'] * 0.01
print(df)
"""
        Name   Country         City  Weight  Height
0     Alvaro  Colombia      Armenia      75    1.70
1    Claudia  Colombia      Calarca      62    1.60
2    Nicolas  Colombia      Armenia      80    1.80
3  Sebastian  Colombia  Bucaramanga      72    1.75
"""
# Calculemos el BMI y agreguemos una columna BMI al DataFrame

import pandas as pd

data = [
    {"Name": "Alvaro", "Country":"Colombia","City":"Armenia"},
    {"Name": "Claudia", "Country":"Colombia","City":"Calarca"},
    {"Name": "Nicolas", "Country":"Colombia","City":"Armenia"},
    {"Name": "Sebastian", "Country":"Colombia","City":"Bucaramanga"}]
df = pd.DataFrame(data)
weights = [75, 62, 80, 72]
df['Weight'] = weights
heights = [170, 160, 180, 175]
df['Height'] = heights
df['Height'] = df['Height'] * 0.01
def calculate_bmi ():
    weights = df['Weight']
    heights = df['Height']
    bmi = []
    for w,h in zip(weights, heights):
        b = w/(h*h)
        bmi.append(b)
    return bmi
bmi = calculate_bmi()
df['BMI'] = bmi
print(df)
"""
        Name   Country         City  Weight  Height        BMI
0     Alvaro  Colombia      Armenia      75    1.70  25.951557
1    Claudia  Colombia      Calarca      62    1.60  24.218750
2    Nicolas  Colombia      Armenia      80    1.80  24.691358
3  Sebastian  Colombia  Bucaramanga      72    1.75  23.510204
"""

# Un codigo mas optimizado seria:

import pandas as pd

data = [
    {"Name": "Alvaro", "Country":"Colombia","City":"Armenia"},
    {"Name": "Claudia", "Country":"Colombia","City":"Calarca"},
    {"Name": "Nicolas", "Country":"Colombia","City":"Armenia"},
    {"Name": "Sebastian", "Country":"Colombia","City":"Bucaramanga"}
]
df = pd.DataFrame(data)
df['Weight'] = [75, 62, 80, 72]
df['Height'] = [170, 160, 180, 175]
df['Height'] = df['Height'] * 0.01

# Utilizamos la función apply() para calcular el BMI de cada persona
df['BMI'] = df.apply(lambda row: row['Weight'] / (row['Height'] ** 2), axis=1)
print(df)
"""
        Name   Country         City  Weight  Height        BMI
0     Alvaro  Colombia      Armenia      75    1.70  25.951557
1    Claudia  Colombia      Calarca      62    1.60  24.218750
2    Nicolas  Colombia      Armenia      80    1.80  24.691358
3  Sebastian  Colombia  Bucaramanga      72    1.75  23.510204
"""

# Formatear columnas de DataFrame: Los valores de la columna BMI del DataFrame son flotantes con muchos dígitos significativos después del decimal. Cambiémoslo a un dígito significativo después del punto. Adicionamos la siguiente linea. df['BMI'] = round(df['BMI'], 1)

import pandas as pd

data = [
    {"Name": "Alvaro", "Country":"Colombia","City":"Armenia"},
    {"Name": "Claudia", "Country":"Colombia","City":"Calarca"},
    {"Name": "Nicolas", "Country":"Colombia","City":"Armenia"},
    {"Name": "Sebastian", "Country":"Colombia","City":"Bucaramanga"}]
df = pd.DataFrame(data)
df['Weight'] = [75, 62, 80, 72]
df['Height'] = [170, 160, 180, 175]
df['Height'] = df['Height'] * 0.01
df['BMI'] = df.apply(lambda row: row['Weight'] / (row['Height'] ** 2), axis=1)
df['BMI'] = round(df['BMI'], 1)
print(df)
"""
        Name   Country         City  Weight  Height   BMI
0     Alvaro  Colombia      Armenia      75    1.70  26.0
1    Claudia  Colombia      Calarca      62    1.60  24.2
2    Nicolas  Colombia      Armenia      80    1.80  24.7
3  Sebastian  Colombia  Bucaramanga      72    1.75  23.5
"""

# Agreguemos las columnas Birth Year y Current Year.

import pandas as pd

data = [
    {"Name": "Alvaro", "Country":"Colombia","City":"Armenia"},
    {"Name": "Claudia", "Country":"Colombia","City":"Calarca"},
    {"Name": "Nicolas", "Country":"Colombia","City":"Armenia"},
    {"Name": "Sebastian", "Country":"Colombia","City":"Bucaramanga"}]
df = pd.DataFrame(data)
df['Weight'] = [75, 62, 80, 72]
df['Height'] = [170, 160, 180, 175]
df['Height'] = df['Height'] * 0.01
df['BMI'] = df.apply(lambda row: row['Weight'] / (row['Height'] ** 2), axis=1)
df['BMI'] = round(df['BMI'], 1)
df['Birth Year'] = ['1967', '1978', '2001', '2004']
df['Current Year'] = pd.Series(2023, index=[0,1,2, 3])
print(df)
"""
        Name   Country         City  Weight  Height   BMI Birth Year  Current Year
0     Alvaro  Colombia      Armenia      75    1.70  26.0       1967          2023
1    Claudia  Colombia      Calarca      62    1.60  24.2       1978          2023
2    Nicolas  Colombia      Armenia      80    1.80  24.7       2001          2023
3  Sebastian  Colombia  Bucaramanga      72    1.75  23.5       2004          2023
"""

# Comprobación de tipos de datos de valores de columna: Podemos comprobar los tipos de datos de los valores de columna utilizando el atributo dtypes.

import pandas as pd

data = [
    {"Name": "Alvaro", "Country":"Colombia","City":"Armenia"},
    {"Name": "Claudia", "Country":"Colombia","City":"Calarca"},
    {"Name": "Nicolas", "Country":"Colombia","City":"Armenia"},
    {"Name": "Sebastian", "Country":"Colombia","City":"Bucaramanga"}]
df = pd.DataFrame(data)
df['Weight'] = [75, 62, 80, 72]
df['Height'] = [170, 160, 180, 175]
df['Height'] = df['Height'] * 0.01
df['BMI'] = df.apply(lambda row: row['Weight'] / (row['Height'] ** 2), axis=1)
df['BMI'] = round(df['BMI'], 1)
df['Birth Year'] = ['1967', '1978', '2001', '2004']
df['Current Year'] = pd.Series(2023, index=[0,1,2, 3])
print(df.Name.dtype) # object
print(df.Country.dtype) # object
print(df.City.dtype)  # object
print(df.Weight.dtype)  # int64
print(df.Height.dtype)  # float64
print(df.BMI.dtype) # float64
print(df['Birth Year'].dtype)  # object
print(df['Current Year'].dtype) # int64

# Birth Year, da un objeto de cadena, deberíamos cambiarlo a número, adicionando la siguiente linea: df['Birth Year'] = df['Birth Year'].astype('int')

import pandas as pd

data = [
    {"Name": "Alvaro", "Country":"Colombia","City":"Armenia"},
    {"Name": "Claudia", "Country":"Colombia","City":"Calarca"},
    {"Name": "Nicolas", "Country":"Colombia","City":"Armenia"},
    {"Name": "Sebastian", "Country":"Colombia","City":"Bucaramanga"}]
df = pd.DataFrame(data)
df['Weight'] = [75, 62, 80, 72]
df['Height'] = [170, 160, 180, 175]
df['Height'] = df['Height'] * 0.01
df['BMI'] = df.apply(lambda row: row['Weight'] / (row['Height'] ** 2), axis=1)
df['BMI'] = round(df['BMI'], 1)
df['Birth Year'] = ['1967', '1978', '2001', '2004']
df['Current Year'] = pd.Series(2023, index=[0,1,2, 3])
df['Birth Year'] = df['Birth Year'].astype('int')
print(df['Birth Year'].dtype)  # int32

