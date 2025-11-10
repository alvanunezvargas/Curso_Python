"""
Python con MongoDB
Python es una tecnología backend y se puede conectar con diferentes aplicaciones de bases de datos. Se puede conectar tanto a bases de datos SQL como noSQL. En esta sección, conectaremos Python con la base de datos MongoDB que es una base de datos noSQL.

MongoDB
MongoDB es una base de datos NoSQL. MongoDB almacena los datos en un documento similar a JSON, lo que hace a MongoDB muy flexible y escalable. Veamos las diferentes terminologías de las bases de datos SQL y NoSQL. La siguiente tabla mostrará la diferencia entre bases de datos SQL y NoSQL.

           SQL                          NoSQL
Bases de datos (Database)          Bases de datos (Database)
Tablas (Tables)                    Colecciones (Collections)
Filas (Rows)                       Documentos (Documents)
Columnas (Columns)                 Campos (Fields)
Index (Index)                      Index (Index)
Unirse (Join)                      Incorporación y vinculación (Embedding and linking)
Agrupado por (Group by)            Agregación (Aggregación)

Instalamos pymongo, desde el terminal de VSCode, ejecutamos el siguiente comando:

pip install pymongo dnspython

El módulo "dnspython" debe estar instalado para usar mongodb+srv:// URIs. Las URIs de MongoDB+SRV son un tipo especial de URI que utiliza el sistema de nombres de dominio (DNS) para resolver el nombre de dominio del cluster MongoDB. El módulo "dnspython" proporciona las funciones necesarias para resolver nombres de dominio.

En esta sección, nos centraremos en una base de datos NoSQL MongoDB. Vamos a registrarnos en mongoDB "https://www.mongodb.com/es" haciendo clic en el botón de inicio de sesión y luego en registrarnos en la página siguiente.

Despues de registrarse en MongoDB y tener abierta tu sesion, vamos a "database" en DEPLOYMENT" buscamos el boton "Connect" y en la opcion "Connect your application" seleccionamos "Drivers Access your Atlas data Using MongoDB's native drivers (e.g. Node.js, Go, etc.)" y en la opcion "3.Add your connection string into your application code", copiamos la cadena de conexion.

mongodb+srv://alvanunez:<password>@alvanunez.gvkokpv.mongodb.net/?retryWrites=true&w=majority

Ahora remeplazamos <password> por la contraseña que usamos para registrarnos en MongoDB.

mongodb+srv://alvanunez:Nicolas1@alvanunez.gvkokpv.mongodb.net/?retryWrites=true&w=majority

Conexión de la aplicación Flask al cluster MongoDB: 
"""

from flask import Flask, render_template
import os # importing operating system module
import pymongo

MONGODB_URI = 'mongodb+srv://alvanunez:Nicolas1@alvanunez.gvkokpv.mongodb.net/?retryWrites=true&w=majority'
client = pymongo.MongoClient(MONGODB_URI)
print(client.list_database_names())

app = Flask(__name__)
if __name__ == '__main__':
    # for deployment we use the environ
    # to make it work for both production and development
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)

"""
Crear una base de datos y una colección: Vamos a crear una base de datos y colección en mongoDB. Vamos a crear una base de datos con el nombre thirty_days_of_python y una colección llamada estudiantes.

Podemos crear la base de datos con alguna de estas dos formas:

db = client.name_of_databse
db = client['name_of_database']
"""

import pymongo

MONGODB_URI = 'mongodb+srv://alvanunez:Nicolas1@alvanunez.gvkokpv.mongodb.net/?retryWrites=true&w=majority'
client = pymongo.MongoClient(MONGODB_URI) 
# Creación de base de datos
db = client['thirty_days_of_python']  
# Creación de la colección  
collection = db['students']
collection.insert_one({'name': 'Alvaro', 'country': 'Colombia', 'city': 'Armenia', 'age': 56})

"""
collection = db['students'] Ese código está creando una coleccion llamada "students" 
db.students - se refiere a la colección "students" en la base de datos actual.
.insert_one() - es el método para insertar un solo documento en la colección.
Dentro del paréntesis pasamos un objeto JSON con los campos y valores del documento:
'name': 'Alvaro' - campo "name" con valor "Alvaro"
'country': 'Colombia' - campo "country" con valor "Colombia"
'city': 'Armenia' - campo "city" con valor "Armenia"
'age': 56 - campo "age" con valor numérico 56
En resumen, esto insertará un documento equivalente a un registro de una tabla SQL, con los campos name, country, city y age, en la colección "students" de MongoDB.
Los documentos en MongoDB son estructuras JSON flexibles y dinámicas, no requieren un esquema fijo previo como en SQL. Podemos insertar documentos con diferentes campos en la misma colección.

Vamos MongoDB y con la sesion abierta vamos a "database" en DEPLOYMENT", buscamos el boton "View Monitoring" luego vamos a "Collections" y veremos en la lista de base de datos que se creo "thirty_days_of_python" y dentro de ella la coleccion "students" con el documento que insertamos. Si observas la figura, el documento se ha creado con un id largo que actúa como clave primaria. Cada vez que creamos un documento, mongoDB crea un id único para él.

El mensaje que observaremos en la terminal de VSCode es el siguiente:

... ['sample_airbnb', 'sample_analytics', 'sample_geospatial', 'sample_guides', 'sample_mflix', 'sample_restaurants', 'sample_supplies', 'sample_training', 'sample_weatherdata', 'thirty_days_of_python', 'admin', 'local']....

Se muestran todas las bases de datos que por defecto se instalaron al abrir la cuenta en MongoDB y la que acabamos de crear "thirty_days_of_python"

El método insert_one() inserta un documento cada vez, si queremos insertar muchos documentos a la vez, podemos utilizar el método insert_many() o el bucle for con el metodo insert_one(). Podemos usar cualquiera de estas dos lineas.

collection.insert_many(students) 
ó
for student in students:
    db.students.insert_one(student)
"""

import pymongo

MONGODB_URI = 'mongodb+srv://alvanunez:Nicolas1@alvanunez.gvkokpv.mongodb.net/?retryWrites=true&w=majority'
client = pymongo.MongoClient(MONGODB_URI) 

# Definir la base de datos y la colección
db = client['thirty_days_of_python'] 
collection = db['students']

# Creación de documentos
students = [
    {'name':'Claudia','country':'Colombia','city':'Calarca','age':45},
    {'name':'Nicolas','country':'Colombia','city':'Armenia','age':22},
    {'name':'Sebastian','country':'Colombia','city':'Bucaramanga','age':19},
]
# Insertar documentos en la colección
collection.insert_many(students)


"""
Búsqueda en MongoDB: Los métodos find() y findOne() son métodos comunes para encontrar datos en una colección en la base de datos mongoDB. Es similar a la sentencia SELECT en una base de datos MySQL. Utilicemos el método find_one() para obtener un documento de una colección de la base de datos.

*find_one({"_id": ObjectId("id"}): Obtiene la primera ocurrencia si no se proporciona un id
"""

import pymongo

MONGODB_URI = 'mongodb+srv://alvanunez:Nicolas1@alvanunez.gvkokpv.mongodb.net/?retryWrites=true&w=majority'
client = pymongo.MongoClient(MONGODB_URI)

db = client['thirty_days_of_python']
student = db.students.find_one()
print(student)  # {'_id': ObjectId('6544e65b2ef9e0e201dc4081'), 'name': 'Alvaro', 'country': 'Colombia', 'city': 'Armenia', 'age': 56}


"""
Podemos obtener un documento específico utilizando un _id específico. Veamos un ejemplo, usar un id para obtener el objeto. Ejemplo '_id':ObjectId('6544ede69773b62c960ddea8').  Para ello debemos importar ObjectId de bson.objectid. `bson` es un módulo en Python que se utiliza para la serialización y deserialización de datos en el formato BSON (Binary JSON), que es el formato de almacenamiento de datos primario utilizado por MongoDB.
`ObjectId` es una clase en el módulo `bson.objectid` que se utiliza para crear y manipular los identificadores únicos de MongoDB. En MongoDB, cada documento almacenado en una colección requiere un campo `_id` que es único para esa colección. `ObjectId` se utiliza comúnmente como el valor para este campo `_id`, ya que proporciona una forma rápida y fácil de generar identificadores únicos.
"""

import pymongo
from bson.objectid import ObjectId

MONGODB_URI = 'mongodb+srv://alvanunez:Nicolas1@alvanunez.gvkokpv.mongodb.net/?retryWrites=true&w=majority'
client = pymongo.MongoClient(MONGODB_URI)

db = client['thirty_days_of_python']
student = db.students.find_one({'_id':ObjectId('6544ede69773b62c960ddea8')})
print(student) # {'_id': ObjectId('6544ede69773b62c960ddea8'), 'name': 'Claudia', 'country': 'Colombia', 'city': 'Calarca', 'age': 45}


"""
Buscar todas las entradas de una colección en una base de datos en MongoDB.
"""

import pymongo

MONGODB_URI = 'mongodb+srv://alvanunez:Nicolas1@alvanunez.gvkokpv.mongodb.net/?retryWrites=true&w=majority'
client = pymongo.MongoClient(MONGODB_URI)

db = client['thirty_days_of_python'] 
students = db.students.find()
for student in students:
    print(student)
    
"""
{'_id': ObjectId('6544e65b2ef9e0e201dc4081'), 'name': 'Alvaro', 'country': 'Colombia', 'city': 'Armenia', 'age': 56}
{'_id': ObjectId('6544e66876ec1a2441521756'), 'name': 'Alvaro', 'country': 'Colombia', 'city': 'Armenia', 'age': 56}
{'_id': ObjectId('6544ede69773b62c960ddea8'), 'name': 'Claudia', 'country': 'Colombia', 'city': 'Calarca', 'age': 45}
{'_id': ObjectId('6544ede69773b62c960ddea9'), 'name': 'Nicolas', 'country': 'Colombia', 'city': 'Armenia', 'age': 22}
{'_id': ObjectId('6544ede69773b62c960ddeaa'), 'name': 'Sebastian', 'country': 'Colombia', 'city': 'Bucaramanga', 'age': 19}

Buscar con consulta: En mongoDB, busque y tome un objeto de consulta. Podemos pasar un objeto de consulta y podemos filtrar los documentos que queramos.
"""

import pymongo

MONGODB_URI = 'mongodb+srv://alvanunez:Nicolas1@alvanunez.gvkokpv.mongodb.net/?retryWrites=true&w=majority'
client = pymongo.MongoClient(MONGODB_URI)
db = client['thirty_days_of_python'] 

query = {
    "city":"Bucaramanga"
}
students = db.students.find(query)

for student in students:
    print(student)
# {'_id': ObjectId('6544ede69773b62c960ddeaa'), 'name': 'Sebastian', 'country': 'Colombia', 'city': 'Bucaramanga', 'age': 19}

# Buscar consulta con modificador

import pymongo

MONGODB_URI = 'mongodb+srv://alvanunez:Nicolas1@alvanunez.gvkokpv.mongodb.net/?retryWrites=true&w=majority'
client = pymongo.MongoClient(MONGODB_URI)
db = client['thirty_days_of_python'] 

query = {
    "country":"Colombia",
    "city":"Calarca"
}
students = db.students.find(query)

for student in students:
    print(student)
# {'_id': ObjectId('6544ede69773b62c960ddea8'), 'name': 'Claudia', 'country': 'Colombia', 'city': 'Calarca', 'age': 45}
    
# Veamos otra consulta modificada

import pymongo

MONGODB_URI = 'mongodb+srv://alvanunez:Nicolas1@alvanunez.gvkokpv.mongodb.net/?retryWrites=true&w=majority'
client = pymongo.MongoClient(MONGODB_URI)
db = client['thirty_days_of_python'] 

query = {"age":{"$gt":30}}
students = db.students.find(query)
for student in students:
    print(student)
"""
{'_id': ObjectId('6544e65b2ef9e0e201dc4081'), 'name': 'Alvaro', 'country': 'Colombia', 'city': 'Armenia', 'age': 56}
{'_id': ObjectId('6544ede69773b62c960ddea8'), 'name': 'Claudia', 'country': 'Colombia', 'city': 'Calarca', 'age': 45}

query = {"age":{"$gt":30}}
Esta consulta se puede traducir como "selecciona los documentos donde el campo 'age' es mayor que 30". Aquí está lo que hace cada parte:
- `"age"`: Este es el campo que estamos consultando. En este caso, estamos buscando documentos donde el campo 'age' tenga un cierto valor.
- `{"$gt":30}`: Este es un operador de consulta de MongoDB. `$gt` significa "mayor que". Entonces, `{"$gt":30}` significa "mayor que 30".
- `query = ...`: Esto simplemente asigna nuestra consulta a la variable `query`.

Por lo tanto, si pasas esta consulta a un método como `collection.find(query)`, buscará todos los documentos en la colección donde el valor del campo 'age' sea mayor que 30.
"""

# Ahora vamos a consultar los menores de 30 años

import pymongo

MONGODB_URI = 'mongodb+srv://alvanunez:Nicolas1@alvanunez.gvkokpv.mongodb.net/?retryWrites=true&w=majority'
client = pymongo.MongoClient(MONGODB_URI)
db = client['thirty_days_of_python'] 

query = {"age":{"$lt":30}} 
students = db.students.find(query)
for student in students:
    print(student)
"""
{'_id': ObjectId('6544ede69773b62c960ddea9'), 'name': 'Nicolas', 'country': 'Colombia', 'city': 'Armenia', 'age': 22}
{'_id': ObjectId('6544ede69773b62c960ddeaa'), 'name': 'Sebastian', 'country': 'Colombia', 'city': 'Bucaramanga', 'age': 19}
"""
   
# Consulta con operador de MongoDB "$lt" significa "menor que" y "$gt" significa "mayor que".

# Podemos limitar el numero de documentos que queremos obtener.

import pymongo

MONGODB_URI = 'mongodb+srv://alvanunez:Nicolas1@alvanunez.gvkokpv.mongodb.net/?retryWrites=true&w=majority'
client = pymongo.MongoClient(MONGODB_URI)

db = client['thirty_days_of_python'] 
students = db.students.find().limit(2)
for student in students:
    print(student)
"""
{'_id': ObjectId('6544e65b2ef9e0e201dc4081'), 'name': 'Alvaro', 'country': 'Colombia', 'city': 'Armenia', 'age': 56}
{'_id': ObjectId('6544ede69773b62c960ddea8'), 'name': 'Claudia', 'country': 'Colombia', 'city': 'Calarca', 'age': 45}


Buscar con clasificación: De forma predeterminada, la clasificación es en orden ascendente. Podemos cambiar la clasificación a orden descendente agregando el parámetro -1. Miremos diferentes ejemplos de clasificación.
"""

import pymongo

MONGODB_URI = 'mongodb+srv://alvanunez:Nicolas1@alvanunez.gvkokpv.mongodb.net/?retryWrites=true&w=majority'
client = pymongo.MongoClient(MONGODB_URI)

db = client['thirty_days_of_python'] 
students = db.students.find().sort('name')
for student in students:
    print(student)
"""
{'_id': ObjectId('6544e65b2ef9e0e201dc4081'), 'name': 'Alvaro', 'country': 'Colombia', 'city': 'Armenia', 'age': 56}
{'_id': ObjectId('6544ede69773b62c960ddea8'), 'name': 'Claudia', 'country': 'Colombia', 'city': 'Calarca', 'age': 45}
{'_id': ObjectId('6544ede69773b62c960ddea9'), 'name': 'Nicolas', 'country': 'Colombia', 'city': 'Armenia', 'age': 22}
{'_id': ObjectId('6544ede69773b62c960ddeaa'), 'name': 'Sebastian', 'country': 'Colombia', 'city': 'Bucaramanga', 'age': 19}
"""
 
# Cambiemos la linea de students

students = db.students.find().sort('name',-1)

"""
{'_id': ObjectId('6544ede69773b62c960ddeaa'), 'name': 'Sebastian', 'country': 'Colombia', 'city': 'Bucaramanga', 'age': 19}
{'_id': ObjectId('6544ede69773b62c960ddea9'), 'name': 'Nicolas', 'country': 'Colombia', 'city': 'Armenia', 'age': 22}
{'_id': ObjectId('6544ede69773b62c960ddea8'), 'name': 'Claudia', 'country': 'Colombia', 'city': 'Calarca', 'age': 45}
{'_id': ObjectId('6544e65b2ef9e0e201dc4081'), 'name': 'Alvaro', 'country': 'Colombia', 'city': 'Armenia', 'age': 56}
"""

students = db.students.find().sort('age')

"""
{'_id': ObjectId('6544ede69773b62c960ddeaa'), 'name': 'Sebastian', 'country': 'Colombia', 'city': 'Bucaramanga', 'age': 19}
{'_id': ObjectId('6544ede69773b62c960ddea9'), 'name': 'Nicolas', 'country': 'Colombia', 'city': 'Armenia', 'age': 22}
{'_id': ObjectId('6544ede69773b62c960ddea8'), 'name': 'Claudia', 'country': 'Colombia', 'city': 'Calarca', 'age': 45}
{'_id': ObjectId('6544e65b2ef9e0e201dc4081'), 'name': 'Alvaro', 'country': 'Colombia', 'city': 'Armenia', 'age': 56}
"""


students = db.students.find().sort('age',-1)

"""
{'_id': ObjectId('6544e65b2ef9e0e201dc4081'), 'name': 'Alvaro', 'country': 'Colombia', 'city': 'Armenia', 'age': 56}
{'_id': ObjectId('6544ede69773b62c960ddea8'), 'name': 'Claudia', 'country': 'Colombia', 'city': 'Calarca', 'age': 45}
{'_id': ObjectId('6544ede69773b62c960ddea9'), 'name': 'Nicolas', 'country': 'Colombia', 'city': 'Armenia', 'age': 22}
{'_id': ObjectId('6544ede69773b62c960ddeaa'), 'name': 'Sebastian', 'country': 'Colombia', 'city': 'Bucaramanga', 'age': 19}
"""

"""
Actualizar con consulta: Usaremos el método update_one() para actualizar un elemento. Se necesitan dos objetos, uno es una consulta y el segundo es el nuevo objeto. La primera persona, Alvaro, vamos a cambiarle la edad de 56 a 57.
"""

import pymongo

MONGODB_URI = 'mongodb+srv://alvanunez:Nicolas1@alvanunez.gvkokpv.mongodb.net/?retryWrites=true&w=majority'
client = pymongo.MongoClient(MONGODB_URI)

db = client['thirty_days_of_python'] 
query = {'age':56}
new_value = {'$set':{'age':57}}

students = db.students.update_one(query, new_value)
for student in students:
    print(student)

# Ahora observemos en MongoDB que se actualizo el documento.

# Cuando queremos actualizar muchos documentos a la vez usamos el método update_many().

import pymongo

MONGODB_URI = 'mongodb+srv://alvanunez:Nicolas1@alvanunez.gvkokpv.mongodb.net/?retryWrites=true&w=majority'
client = pymongo.MongoClient(MONGODB_URI)

db = client['thirty_days_of_python'] 
query = {}  # Selecciona todos los documentos
new_value = {'$set':{'age':35}}  # Establece la nueva edad
db.students.update_many(query, new_value)

# En todos los documentos "query = {} ", la edad cambia a 35.

"""
Eliminar documento: El método delete_one() elimina un documento. delete_one() toma un parámetro de objeto de consulta. Solo elimina la primera aparición. Agreguemos a Nestor a la coleccion y luego lo borramos. .
"""

import pymongo

MONGODB_URI = 'mongodb+srv://alvanunez:Nicolas1@alvanunez.gvkokpv.mongodb.net/?retryWrites=true&w=majority'
client = pymongo.MongoClient(MONGODB_URI) 

# Definir la base de datos y la colección
db = client['thirty_days_of_python'] 
collection = db['students']

# Creación de documentos
students = [
    {'name':'Nestor','country':'Israel','city':'Telavid','age':20},
    ]

# Insertar documentos en la colección
collection.insert_one(students)

# Ahora eliminemos a Nestor

import pymongo

MONGODB_URI = 'mongodb+srv://alvanunez:Nicolas1@alvanunez.gvkokpv.mongodb.net/?retryWrites=true&w=majority'
client = pymongo.MongoClient(MONGODB_URI) 

# Definir la base de datos y la colección
db = client['thirty_days_of_python'] 
query = {'name':'Nestor'}
db.students.delete_one(query)

# Vamos a MongoDB y observamos que se elimino el documento.
"""
Cuando queremos eliminar muchos documentos usamos el método delete_many(), que requiere un objeto de consulta. Si pasamos un objeto de consulta vacío a delete_many({}), eliminará todos los documentos de la colección.
"""

import pymongo

MONGODB_URI = 'mongodb+srv://alvanunez:Nicolas1@alvanunez.gvkokpv.mongodb.net/?retryWrites=true&w=majority'
client = pymongo.MongoClient(MONGODB_URI) 

# Definir la base de datos y la colección
db = client['thirty_days_of_python'] 
db.students.delete_many({})

# Vamos a MongoDB y observamos que se eliminaron todos los documentos.

# Borrar una colección: Usando el método drop() podemos eliminar una colección de una base de datos.

import pymongo

MONGODB_URI = 'mongodb+srv://alvanunez:Nicolas1@alvanunez.gvkokpv.mongodb.net/?retryWrites=true&w=majority'
client = pymongo.MongoClient(MONGODB_URI) 

db = client['thirty_days_of_python']
collection = db['students']
db.students.drop()