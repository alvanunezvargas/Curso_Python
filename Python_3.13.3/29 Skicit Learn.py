"""
HDBSCAN (Agrupamiento basado en densidad jerárquico)

Tengo un archivo Excel con varias hojas y voy a trabajar en la hoja llamada "Espectro", que contiene 1075 x 1783 datos, correspondientes a los espectros de crudos de recibo en una refineria. La primera fila contiene las identificaciones de las propiedades de cada columna. ¿Qué código de Python necesito para realizar un HDBSCAN (Agrupamiento basado en densidad jerárquico) utilizando scikit-learn?
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import HDBSCAN
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import HDBSCAN
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Cargar el archivo Excel excluyendo la primera fila (encabezado de la página)
df = pd.read_excel(r"C:\Users\alvan\OneDrive\Documentos\MATLAB\Examples\R2020b\matlab\StoreDifferentDataTypesInCellArrayExample\CRUDOS-RAMAN.xlsx", sheet_name="Espectro", skiprows=1)

# Asegurarse de que todas las filas tengan valores
df.dropna(inplace=True)

X = df.values

scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X)

clusterer = HDBSCAN(min_cluster_size=15)  
clusterer.fit(X_scaled)

labels = clusterer.labels_

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f'Número de clusters: {n_clusters}')

# Reduce dimension to 2 with t-SNE
tsne = TSNE(n_components=2, random_state=0)
X_2d = tsne.fit_transform(X_scaled)

# Visualize the clusters
plt.figure(figsize=(6, 5))
colors = 'c', 'b', 'g', 'r', 'm', 'y', 'k', 'w', 'orange', 'purple'
for i, c in zip(range(n_clusters), colors):
    plt.scatter(X_2d[labels == i, 0], X_2d[labels == i, 1], c=c, label=f'Cluster {i}')
plt.legend()
plt.show()

# Add cluster labels to DataFrame and save to Excel
df['Cluster'] = pd.Series(labels)
df['X'] = X_2d[:, 0]
df['Y'] = X_2d[:, 1]
df.to_excel(r"C:\Users\alvan\OneDrive\Documentos\MATLAB\Examples\R2020b\matlab\StoreDifferentDataTypesInCellArrayExample\CRUDOS-RAMAN-Clusters.xlsx", index=False)

"""
En el algoritmo HDBSCAN, el parámetro min_cluster_size se establece en 15. Esto significa que los puntos de datos que no están dentro de un grupo de al menos 15 puntos de datos se consideran ruido y se etiquetan como -1. En este ejemplo se identificaron y graficaron 4 clusters, 0, 1, 2 y 3. Los puntos de datos que no pertenecen a ninguno de estos clusters se etiquetan como -1. En este ejemplo, 715 de los 1075 puntos de datos se etiquetaron como ruido.

fit_transform` hace dos cosas: primero, calcula la media y la desviación estándar de tus datos (esto es el método `fit`). Luego, utiliza estos valores para normalizar tus datos (esto es el método `transform`)


Normalizacion de datos con fit_transform

fit_transform realiza dos funciones: primero, calcula la media y la desviación estándar de tus datos (esto es el método `fit`). Luego, utiliza estos valores para normalizar tus datos (esto es el método `transform`)

En este caso, el archivo de origen, en la hoja "Propiedades" tiene la identificacion de las caracteristicas en el encabezado de columna, es decir la primera fila y los valores de las observaciones en el resto de filas. Que codigo de python utilizo para normalizar a lo largo de las columnas, donde cada fila representa una observacion y cada columna una caracteristica.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Cargar el archivo Excel
archivo_entrada = r"C:\Users\alvan\OneDrive\Documentos\MATLAB\Examples\R2020b\matlab\StoreDifferentDataTypesInCellArrayExample\CRUDOS-RAMAN.xlsx"  
archivo_salida = r"C:\Users\alvan\OneDrive\Documentos\MATLAB\Examples\R2020b\matlab\StoreDifferentDataTypesInCellArrayExample\Normalizacion.xlsx"  

# Leer el archivo Excel en un DataFrame de pandas
datos = pd.read_excel(archivo_entrada, sheet_name="Propiedades")

# Convertir todos los nombres de las columnas a strings
datos.columns = datos.columns.astype(str)

# Normalizar los datos utilizando StandardScaler de scikit-learn
scaler = StandardScaler()
datos_normalizados = scaler.fit_transform(datos)

# Crear un nuevo DataFrame con los datos normalizados
df_normalizado = pd.DataFrame(datos_normalizados, columns=datos.columns)

# Guardar el nuevo DataFrame en un archivo Excel
df_normalizado.to_excel(archivo_salida, index=False)

print(f"Datos normalizados guardados en {archivo_salida}")


"""
Máquina de Vectores de Soporte (SVM) para regresión

Este script de Python realiza varias tareas relacionadas con el preprocesamiento de datos y el entrenamiento de un modelo de Máquina de Vectores de Soporte (SVM) para la regresión.

1. **Carga los datos**: Los datos se cargan desde un archivo Excel en un DataFrame de pandas. Los nombres de las columnas se convierten a strings.

2. **Divide los datos**: Los datos se dividen en características (X) y etiquetas (y). Las características son todas las columnas excepto la primera, y las etiquetas son solo la primera columna.

3. **Maneja los valores faltantes**: Se crea un `SimpleImputer` que reemplaza los valores NaN con la media de la columna. Este imputer se ajusta y transforma las características.

4. **Divide los datos en conjuntos de entrenamiento y prueba**: Los datos se dividen en un conjunto de entrenamiento y un conjunto de prueba, con el 20% de los datos reservados para pruebas.

5. **Escala los datos**: Se crea un `StandardScaler` que se ajusta a los datos de entrenamiento. Este scaler se utiliza para transformar tanto los datos de entrenamiento como los de prueba.

6. **Entrena el modelo**: Se crea un modelo `svm.SVR` (una Máquina de Vectores de Soporte para la regresión) y se entrena con los datos de entrenamiento escalados.

7. **Guarda el modelo**: El modelo entrenado se guarda en un archivo llamado 'modelo_svm.pkl' para su uso futuro.

8. **Carga el modelo y hace una predicción**: El modelo se carga desde el archivo 'modelo_svm.pkl'. Luego, se carga una muestra de datos desde un archivo Excel, se convierte en un array numpy, se escala utilizando el mismo scaler que se utilizó para los datos de entrenamiento, y se utiliza para hacer una predicción con el modelo. La predicción se imprime en la consola.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn import svm
import joblib
import numpy as np

# Cargar el archivo Excel
archivo_excel = r"C:\Users\alvan\OneDrive\Documentos\MATLAB\Examples\R2020b\matlab\StoreDifferentDataTypesInCellArrayExample\CRUDOS-RAMAN.xlsx"   # Reemplaza con la ruta correcta de tu archivo
df = pd.read_excel(archivo_excel, sheet_name="Propiedades", header=0)  # Especificar que la primera fila contiene nombres de propiedades

# Convertir todos los nombres de las columnas a strings
df.columns = df.columns.astype(str)

# Dividir los datos en características (X) y etiquetas (y)
X = df.iloc[:,1:]  # Seleccionar todas las filas : y todas las columnas a partir de la segunda columna 1:
y = df.iloc[:,0]   # Seleccionar todas las filas : y solo la primeera columna 0

# Crear un imputer que reemplace los valores NaN con la media de la columna
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# Ajustar el imputer a los datos de entrenamiento y transformarlos
X_imputed = imputer.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Inicializar y ajustar el scaler solo en el conjunto de entrenamiento
scaler = StandardScaler().fit(X_train)

# Transformar los conjuntos de entrenamiento y prueba utilizando el scaler (normalizar) ajustado
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar el modelo
clf = svm.SVR()
clf.fit(X_train_scaled, y_train)

# Guardar el modelo
joblib.dump(clf, 'modelo_svm.pkl')

# En este código, svm.SVC() crea un nuevo modelo SVM, clf.fit(X_train_scaled, y_train) entrena el modelo utilizando los datos de entrenamiento escalados y las etiquetas de entrenamiento, y joblib.dump(clf, 'modelo_svm.pkl') guarda el modelo entrenado en un archivo llamado 'modelo_svm.pkl'.

# Para cargar el modelo y usarlo para hacer predicciones, puedes hacer algo como esto:

# Cargar el modelo
clf = joblib.load('modelo_svm.pkl')

# Cargar los datos de la muestra desde un archivo Excel
archivo_excel_muestra = r"C:\Users\alvan\OneDrive\Documentos\MATLAB\Examples\R2020b\matlab\StoreDifferentDataTypesInCellArrayExample\Muestra.xlsx"  # Reemplaza con la ruta al archivo Excel de la muestra
df_muestra = pd.read_excel(archivo_excel_muestra, header=0)  # Asume que la primera fila contiene los nombres de las columnas

# Convertir el DataFrame de la muestra en un array numpy
muestra = df_muestra.to_numpy()

# Escalar la muestra utilizando el mismo scaler que se utilizó para los datos de entrenamiento
muestra_scaled = scaler.transform(muestra)

# Hacer una predicción
# Supongamos que 'muestra' es un nuevo dato que quieres clasificar
muestra_scaled = scaler.transform(muestra)
prediccion = clf.predict(muestra_scaled)

print(f"La muestra tiene un API de {prediccion}")

"""
El LabelEncoder se utiliza para convertir etiquetas de clases categóricas en números.
"""

import pandas as pd
from sklearn import preprocessing

# Crear un DataFrame de ejemplo
data = {'Nombre': ['Fluffy', 'Buddy', 'Tweety', 'Mittens'],
        'Clase': ['Gato', 'Perro', 'Pájaro', 'Gato']}

df = pd.DataFrame(data)

# Visualizar el DataFrame original
print("DataFrame original:")
print(df)

# Aplicar LabelEncoder a la columna 'Clase'
le = preprocessing.LabelEncoder()
df['Clase'] = le.fit_transform(df['Clase'])

# Visualizar el DataFrame después de la transformación
print("\nDataFrame después de la transformación:")
print(df)

"""
DataFrame original:
    Nombre   Clase
0   Fluffy    Gato
1    Buddy   Perro
2   Tweety  Pájaro
3  Mittens    Gato

DataFrame después de la transformación:
    Nombre  Clase
0   Fluffy      0
1    Buddy      1
2   Tweety      2
3  Mittens      0

En este ejemplo, LabelEncoder se utiliza para transformar las etiquetas de la columna 'Clase' en valores numéricos. La salida mostrará el DataFrame original y el DataFrame después de aplicar la transformación, donde las etiquetas de 'Gato' se han convertido en 0, 'Perro' en 1 y 'Pájaro' en 2.

Otro ejemplo mas complejo seria:
"""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Cargar el conjunto de datos de diabetes
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# Convertir base de datos diabetis en un archivo excel
import pandas as pd
from sklearn import datasets

# Cargar el conjunto de datos de diabetes
diabetes = datasets.load_diabetes()

# Crear un DataFrame de pandas con los datos y las columnas del conjunto de datos
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)

# Agregar la variable objetivo al DataFrame
df['target'] = diabetes.target

# Guardar el DataFrame en un archivo Excel
df.to_excel(r'C:\Users\alvan\OneDrive\Documentos\MATLAB\Examples\R2020b\matlab\StoreDifferentDataTypesInCellArrayExample\diabetes.xlsx', index=False)

# Convertir el problema en un problema de clasificación binaria: (Tomar un conjunto de datos originalmente diseñado para resolver un problema de clasificación con más de dos clases y transformarlo en un problema de clasificación con dos clases. La variable objetivo 'y' contenía información continua sobre la progresión de la enfermedad. Para convertirlo en un problema de clasificación binaria, se establecera un umbral (threshold). En este caso, se utiliza el valor medio de la variable objetivo 'y' como umbral. Si el valor de la variable objetivo (y) es mayor que el umbral, etiquetamos la instancia como perteneciente a la clase positiva (1), lo que podría indicar la presencia de diabetes. Si es menor o igual al umbral, etiquetamos la instancia como perteneciente a la clase negativa (0), lo que podría indicar la ausencia de diabetes.)
# Por ejemplo, consideraremos que un paciente tiene diabetes si el nivel de glucosa es mayor que el valor medio
threshold = np.mean(y)
y_binary = (y > threshold).astype(int)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# Escalar características utilizando StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Utilizar LabelEncoder para codificar las etiquetas de clase
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# Entrenar un modelo de regresión logística
model = LogisticRegression()
model.fit(X_train_scaled, y_train_encoded)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test_scaled)

# Decodificar las predicciones a las etiquetas originales
y_pred_original = le.inverse_transform(y_pred)

# Evaluar la precisión del modelo
accuracy = accuracy_score(y_test, y_pred_original)
print(f'Precisión del modelo: {accuracy:.2f}')


"""
SimpleImputer de scikit-learn se utiliza para asignar valores faltantes en una base de datos
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# Crear un conjunto de datos simulado con valores faltantes
data = {
    'Feature1': [1, 2, np.nan, 4, 5],
    'Feature2': [5, np.nan, 3, 7, 8],
    'Feature3': [9, 10, 11, np.nan, 13]
}

df = pd.DataFrame(data)

print("Datos originales:")
print(df)

# Seleccionar las características (columnas) que deseas transformar
X = df.values

# Crear el imputador con estrategia 'mean' (media)
imputer = SimpleImputer(strategy='mean')

# Aplicar la transformación al conjunto de datos
X = imputer.fit_transform(X)

# Crear un nuevo DataFrame con los datos imputados
df_imputed = pd.DataFrame(X, columns=df.columns)

print("\nDatos después de la imputación:")
print(df_imputed)

"""
Datos originales:
   Feature1  Feature2  Feature3
0       1.0       5.0       9.0
1       2.0       NaN      10.0
2       NaN       3.0      11.0
3       4.0       7.0       NaN
4       5.0       8.0      13.0

Datos después de la imputación:
   Feature1  Feature2  Feature3
0       1.0      5.00      9.00
1       2.0      5.75     10.00
2       3.0      3.00     11.00
3       4.0      7.00     10.75
4       5.0      8.00     13.00

En este ejemplo, el conjunto de datos original tiene algunos valores faltantes (NaN). Luego, se utiliza SimpleImputer de scikit-learn con la estrategia 'mean' para reemplazar los valores faltantes con la media de cada columna. El resultado es un nuevo conjunto de datos (df_imputed) donde los valores faltantes han sido imputados con la media de sus respectivas columnas. Otras estrategias para remplazar los valores faltantes son:

median: Imputa los valores faltantes con la mediana de la columna.
imputer = SimpleImputer(strategy='median')

most_frequent: Imputa los valores faltantes con el valor más frecuente (moda) de la columna.
imputer = SimpleImputer(strategy='most_frequent')

constant: Imputa los valores faltantes con un valor constante proporcionado por el parámetro fill_value. Este valor se especifica al crear el objeto SimpleImputer.
imputer = SimpleImputer(strategy='constant', fill_value=0)

Estas estrategias permiten adaptar la imputación a la naturaleza de los datos y al contexto del problema. Por ejemplo, la estrategia de la media (mean) y mediana (median) son adecuadas para datos numéricos, mientras que la estrategia más frecuente (most_frequent) es útil para variables categóricas. La estrategia constante (constant) puede ser útil si se tiene un valor específico con el que se desean imputar los valores faltantes.
"""

#  SelectKBest y f_regression de scikit-learn se utilizan para seleccionar las mejores características de un conjunto de datos utilizando la prueba F de regresión.

import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression

# Cargar el conjunto de datos de diabetes
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# Crear un DataFrame de pandas con los datos y las columnas del conjunto de datos
df = pd.DataFrame(X, columns=diabetes.feature_names)

# Agregar la variable objetivo al DataFrame
df['target'] = y

# Guardar el DataFrame en un archivo Excel
df.to_excel(r'C:\Users\alvan\OneDrive\Documentos\MATLAB\Examples\R2020b\matlab\StoreDifferentDataTypesInCellArrayExample\diabetes.xlsx', index=False)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)

# Crear el selector de características con prueba F de regresión y k=3 (seleccionar 3 mejores características)
selector = SelectKBest(score_func=f_regression, k=3)

# Ajustar y transformar los datos de entrenamiento
X_train_selected = selector.fit_transform(X_train, y_train)

# Transformar los datos de prueba utilizando las características seleccionadas
X_test_selected = selector.transform(X_test)

# Imprimir las características seleccionadas en el conjunto de entrenamiento
print("Características seleccionadas:")
selected_features = np.where(selector.get_support())[0]
print(np.array(diabetes.feature_names)[selected_features])

# Imprimir algunas estadísticas
print("\nEstadísticas de prueba:")
print("Score de prueba F:", selector.scores_)
print("P-values:", selector.pvalues_)

# Imprimir los datos de entrenamiento y prueba después de la selección de características
print("\nDatos de entrenamiento después de la selección de características:")
print(X_train_selected)

print("\nDatos de prueba después de la selección de características:")
print(X_test_selected)

"""
Primero, se divide el conjunto de datos en conjuntos de entrenamiento y prueba utilizando la función `train_test_split`. Se reserva el 20% de los datos para pruebas y se establece una semilla aleatoria para garantizar que los resultados sean reproducibles.

Luego, se crea un objeto `SelectKBest` que se configura para usar la prueba F de regresión (`f_regression`) y seleccionar las tres mejores características (`k=3`). Este objeto se ajusta a los datos de entrenamiento, lo que significa que calcula las puntuaciones F y los p-values para cada característica. Luego, selecciona las tres características con las puntuaciones F más altas.

Después de ajustar el selector a los datos de entrenamiento, se transforman los datos de entrenamiento y prueba. Esto significa que se eliminan todas las características excepto las tres seleccionadas.

A continuación, se imprime el nombre de las características seleccionadas. Esto se hace utilizando el método `get_support` del selector, que devuelve una máscara booleana de las características seleccionadas. Esta máscara se utiliza para indexar el array de nombres de características.

Finalmente, se imprimen algunas estadísticas de la prueba F, incluyendo las puntuaciones F y los p-values para cada característica. También se imprimen los datos de entrenamiento y prueba después de la selección de características.
"""

# PolynomialFeatures se aplica para generar características polinomiales a partir de las características existentes 

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Cargar el conjunto de datos de diabetes
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aplicar PolynomialFeatures para generar características polinómicas de segundo grado
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Entrenar un modelo de regresión lineal con características polinómicas
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test_poly)

# Calcular el error cuadrático medio (MSE)
mse = mean_squared_error(y_test, y_pred)

print(f"Error cuadrático medio en el conjunto de prueba: {mse}")

# Obtener los nombres de las características polinómicas
poly_feature_names = poly.get_feature_names_out(diabetes.feature_names)

# Crear un DataFrame para las características polinómicas
df_poly = pd.DataFrame(X_train_poly, columns=poly_feature_names)

print("\nPrimeras filas del conjunto de datos polinómico:")
print(df_poly.head())

"""
Error cuadrático medio en el conjunto de prueba: 3096.0283073442642

Primeras filas del conjunto de datos polinómico:
     1       age       sex       bmi        bp        s1        s2  ...     s3 s6      s4^2     s4 s5     s4 s6          s5^2     s5 s6      s6^2        
0  1.0  0.070769  0.050680  0.012117  0.056301  0.034206  0.049416  ...  0.000043  0.001177  0.000939 -0.000037  7.487912e-04 -0.000029  0.000001        
1  1.0 -0.009147  0.050680 -0.018062 -0.033213 -0.020832  0.012152  ... -0.001430  0.005071  0.000019  0.001398  7.424434e-08  0.000005  0.000385        
2  1.0  0.005383 -0.044642  0.049840  0.097615 -0.015328 -0.016345  ...  0.000089  0.000007 -0.000044  0.000035  2.902277e-04 -0.000230  0.000182        
3  1.0 -0.027310 -0.044642 -0.035307 -0.029770 -0.056607 -0.058620  ... -0.003915  0.001560  0.001970  0.005114  2.487261e-03  0.006458  0.016766        
4  1.0 -0.023677 -0.044642 -0.065486 -0.081413 -0.038720 -0.053610  ... -0.002537  0.005836  0.002836  0.003247  1.378551e-03  0.001578  0.001806        

[5 rows x 66 columns]

Primero, se divide el conjunto de datos en conjuntos de entrenamiento y prueba utilizando la función `train_test_split`. Se reserva el 20% de los datos para pruebas y se establece una semilla aleatoria para garantizar que los resultados sean reproducibles.

Luego, se crea un objeto `PolynomialFeatures` con un grado de 2, lo que significa que generará todas las características polinómicas de segundo grado de los datos. Estas características incluyen los cuadrados de todas las características originales, así como todas las interacciones de dos vías entre diferentes características.

Después de crear el objeto `PolynomialFeatures`, se ajusta a los datos de entrenamiento y se transforman los datos de entrenamiento y prueba. Esto significa que se generan las características polinómicas de los datos.

A continuación, se entrena un modelo de regresión lineal con las características polinómicas. Se ajusta el modelo a los datos de entrenamiento y luego se utilizan para predecir los valores objetivo en el conjunto de prueba.

Luego, se calcula el error cuadrático medio (MSE) de las predicciones en el conjunto de prueba. El MSE es una medida común de la precisión de un modelo de regresión.

Finalmente, se obtienen los nombres de las características polinómicas generadas y se crean un DataFrame de pandas con los datos de entrenamiento y los nombres de las características. Se imprime las primeras filas de este DataFrame para verificar que las características polinómicas se hayan generado correctamente.
"""

