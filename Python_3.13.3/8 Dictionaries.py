# Diccionarios
# Un diccionario es una colección de datos no ordenados, modificables (mutables) y emparejados (clave: valor).
# Crear un diccionario
# Para crear un diccionario se utilizan llaves, {} o la función incorporada dict().
empty_dict = {}
dict = dict()

# Diccionario con valores
dct = {'key1':'value1', 'key2':'value2', 'key3':'value3', 'key4':'value4'}

person = {
    'first_name':'Alvaro',
    'last_name':'Nunez',
    'age':55,
    'country':'Colombia',
    'is_marred':True,
    'skills':['MatLab', 'Quality Control', 'Fuel oils', 'volumetric correction factors', 'Python'],
    'address':{
        'street':'Avenue 19',
        'zipcode':'630004'
    }
    }

# El diccionario anterior muestra que un valor puede ser de cualquier tipo de datos: cadena, booleano, lista, tupla, conjunto o diccionario.

# Longitud del diccionario
# Comprueba el número de pares 'clave: valor' del diccionario.
dct = {'key1':'value1', 'key2':'value2', 'key3':'value3', 'key4':'value4'}
print(len(dct)) # 4

person = {
    'first_name':'Alvaro',
    'last_name':'Nunez',
    'age':55,
    'country':'Colombia',
    'is_marred':True,
    'skills':['MatLab', 'Quality Control', 'Fuel oils', 'volumetric correction factors', 'Python'],
    'address':{
        'street':'Avenue 19',
        'zipcode':'630004'
    }
    }
print(len(person)) # 7

# Acceso a los elementos del diccionario: Podemos acceder a los "valores" del Diccionario haciendo referencia a su nombre clave.
dct = {'key1':'value1', 'key2':'value2', 'key3':'value3', 'key4':'value4'}
print(dct['key1']) # value1
print(dct['key4']) # value4

person = {
    'first_name':'Alvaro',
    'last_name':'Nunez',
    'age':55,
    'country':'Colombia',
    'is_marred':True,
    'skills':['MatLab', 'Quality Control', 'Fuel oils', 'volumetric correction factors', 'Python'],
    'address':{
        'street':'Avenue 19',
        'zipcode':'630004'
    }
    }
print(person['first_name']) # Alvaro
print(person['country'])    # Colombia
print(person['skills'])     # ['MatLab', 'Quality Control', 'Fuel oils', 'volumetric correction factors', 'Python']
print(person['skills'][0])  # MatLab
print(person['address']['street']) # Avenue 19
print(person['city'])       # Error

# Al acceder a un elemento por el nombre de la clave se produce un error si la clave no existe. Para evitar este error primero tenemos que comprobar
# si existe una clave o podemos utilizar el método get. El método get devuelve None, que es un tipo de datos de objeto NoneType, si la clave no existe.
person = {
    'first_name':'Alvaro',
    'last_name':'Nunez',
    'age':55,
    'country':'Colombia',
    'is_marred':True,
    'skills':['MatLab', 'Quality Control', 'Fuel oils', 'volumetric correction factors', 'Python'],
    'address':{
        'street':'Avenue 19',
        'zipcode':'630004'
    }
    }
print(person.get('first_name')) # Alvaro
print(person.get('country'))    # Colombia
print(person.get('skills')) # ['MatLab', 'Quality Control', 'Fuel oils', 'volumetric correction factors', 'Python']
print(person.get('city'))   # None

# Añadir elementos a un diccionario: Podemos añadir nuevos pares de clave y valor a un diccionario
dct = {'key1':'value1', 'key2':'value2', 'key3':'value3', 'key4':'value4'}
dct['key5'] = 'value5'
print(dct)  # {'key1': 'value1', 'key2': 'value2', 'key3': 'value3', 'key4': 'value4', 'key5': 'value5'}

person = {
    'first_name':'Alvaro',
    'last_name':'Nunez',
    'age':55,
    'country':'Colombia',
    'is_marred':True,
    'skills':['MatLab', 'Quality Control', 'Fuel oils', 'volumetric correction factors', 'Python'],
    'address':{
        'street':'Avenue 19',
        'zipcode':'630004'
    }
    }
person['job_title'] = 'Instructor'
person['skills'].append('chromatography')
print(person) # {'first_name': 'Alvaro', 'last_name': 'Nunez', 'age': 55, 'country': 'Colombia', 'is_marred': True, 'skills': ['MatLab', 'Quality Control', 'Fuel oils', 'volumetric correction factors', 'Python', 'chromatography'], 'address': {'street': 'Avenue 19', 'zipcode': '630004'}, 'job_title': 'Instructor'}

# Modificación de elementos de un diccionario: Podemos modificar los elementos de un diccionario
dct = {'key1':'value1', 'key2':'value2', 'key3':'value3', 'key4':'value4'}
dct['key1'] = 'value-one'
print(dct) # {'key1': 'value-one', 'key2': 'value2', 'key3': 'value3', 'key4': 'value4'}

person = {
    'first_name':'Alvaro',
    'last_name':'Nunez',
    'age':55,
    'country':'Colombia',
    'is_marred':True,
    'skills':['MatLab', 'Quality Control', 'Fuel oils', 'volumetric correction factors', 'Python'],
    'address':{
        'street':'Avenue 19',
        'zipcode':'630004'
    }
    }
person['first_name'] = 'Eyob'
person['age'] = 52
print(person) # {'first_name': 'Eyob', 'last_name': 'Nunez', 'age': 52, 'country': 'Colombia', 'is_marred': True, 'skills': ['MatLab', 'Quality Control', 'Fuel oils', 'volumetric correction factors', 'Python'], 'address': {'street': 'Avenue 19', 'zipcode': '630004'}}

# Comprobación de claves en un diccionario: Utilice el operador "in" para comprobar si una clave existe en un diccionario
dct = {'key1':'value1', 'key2':'value2', 'key3':'value3', 'key4':'value4'}
print('key2' in dct) # True
print('key5' in dct) # False

# Eliminar pares de clave y valor de un diccionario
# pop(clave): elimina el elemento con el nombre de clave especificado:
# popitem(): elimina el último elemento
# del: elimina un elemento con el nombre de clave especificado
dct = {'key1':'value1', 'key2':'value2', 'key3':'value3', 'key4':'value4'}
dct.pop('key1')  # elimina el par clave : valor key1':'value1
print(dct) # {'key2': 'value2', 'key3': 'value3', 'key4': 'value4'}
dct = {'key1':'value1', 'key2':'value2', 'key3':'value3', 'key4':'value4'}
dct.popitem()   # elimina el último par clave : valor
print(dct)  # {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}
dct = {'key1':'value1', 'key2':'value2', 'key3':'value3', 'key4':'value4'}
del dct['key2']  # elimina el par clave : valor key2':'value2
print(dct)  # {'key1': 'value1', 'key3': 'value3', 'key4': 'value4'}

person = {
    'first_name':'Alvaro',
    'last_name':'Nunez',
    'age':55,
    'country':'Colombia',
    'is_marred':True,
    'skills':['MatLab', 'Quality Control', 'Fuel oils', 'volumetric correction factors', 'Python'],
    'address':{
        'street':'Avenue 19',
        'zipcode':'630004'
    }
    }
person.pop('first_name')        # elimina el par clave : valor "first_name"
print(person)
person.popitem()                # elimina el ultimo par clave : "valor"
print(person)
del person['is_marred']        # elimina el par clave : valor "is_married"
print(person)

# Cambiar el diccionario a una lista de elementos
# El método items() cambia el diccionario a una lista de tuplas.
dct = {'key1':'value1', 'key2':'value2', 'key3':'value3', 'key4':'value4'}
print(dct.items()) # dict_items([('key1', 'value1'), ('key2', 'value2'), ('key3', 'value3'), ('key4', 'value4')])

# Borrar un diccionario: Si no queremos los elementos de un diccionario podemos borrarlos utilizando el método clear()
dct = {'key1':'value1', 'key2':'value2', 'key3':'value3', 'key4':'value4'}
print(dct.clear()) # None

# Borrar un diccionario: Si no utilizamos el diccionario podemos borrarlo completamente utilizando el método del
dct = {'key1':'value1', 'key2':'value2', 'key3':'value3', 'key4':'value4'}
del dct
print(dct)  # NameError: name 'dct' is not defined.

# Copiar un diccionario: Podemos copiar un diccionario utilizando el método copy(). Usando copy podemos evitar la mutación del diccionario original.
dct = {'key1':'value1', 'key2':'value2', 'key3':'value3', 'key4':'value4'}
dct_copy = dct.copy() 
print(dct_copy)  #  {'key1': 'value1', 'key2': 'value2', 'key3': 'value3', 'key4': 'value4'}

# Obtener las claves de un diccionario como una lista: El método keys() nos proporciona todas las "claves" de un diccionario en forma de lista.
dct = {'key1':'value1', 'key2':'value2', 'key3':'value3', 'key4':'value4'}
keys = dct.keys()
print(keys)  # dict_keys(['key1', 'key2', 'key3', 'key4'])

# Obtener los valores de un diccionario como una lista: El método "values" nos da todos los "valores" de un diccionario como una lista.
dct = {'key1':'value1', 'key2':'value2', 'key3':'value3', 'key4':'value4'}
values = dct.values()
print(values)     # dict_values(['value1', 'value2', 'value3', 'value4'])


# Ejercicios

# Crear un diccionario vacío llamado dog
dog = {}
# ó
dog = dict()

# Añadir nombre, color, raza, patas, edad al diccionario dog
dog['nombre'] = 'Fido'
dog['color'] = 'marrón'
dog['raza'] = 'Labrador Retriever'
dog['patas'] = 4
dog['edad'] = 3
print(dog)  # {'nombre': 'Fido', 'color': 'marrón', 'raza': 'Labrador Retriever', 'patas': 4, 'edad': 3}

# Cree un diccionario de estudiantes y añada nombre, apellidos, sexo, edad, estado civil, aptitudes, país, ciudad y dirección como claves del diccionario.
estudiantes = {}

estudiantes['estudiante1'] = {'nombre': 'Juan', 'apellidos': 'Pérez García', 'sexo': 'Masculino', 'edad': 22, 'estado civil': 'Soltero', 'aptitudes': ['Programación', 'Diseño gráfico'], 'país': 'España', 'ciudad': 'Madrid', 'dirección': 'Calle Mayor 123'}
estudiantes['estudiante2'] = {'nombre': 'María', 'apellidos': 'González López', 'sexo': 'Femenino', 'edad': 24, 'estado civil': 'Casada', 'aptitudes': ['Inglés', 'Marketing'], 'país': 'México', 'ciudad': 'Ciudad de México', 'dirección': 'Avenida Reforma 456'}
estudiantes['estudiante3'] = {'nombre': 'Pablo', 'apellidos': 'Ramírez Sánchez', 'sexo': 'Masculino', 'edad': 20, 'estado civil': 'Soltero', 'aptitudes': ['Música', 'Escritura creativa'], 'país': 'Argentina', 'ciudad': 'Buenos Aires', 'dirección': 'Avenida Corrientes 789'}

print(estudiantes)

# Obtener la longitud del diccionario del alumno
print(len(estudiantes))  # 3

# Obtén el valor de aptitudes y comprueba el tipo de datos, debe ser una lista
print(estudiantes['estudiante1']['aptitudes'])  # ['Programación', 'Diseño gráfico']
print(estudiantes['estudiante2']['aptitudes'])  # ['Inglés', 'Marketing']
print(estudiantes['estudiante3']['aptitudes'])  # ['Música', 'Escritura creativa']
# ó
for estudiante in estudiantes.values():
    print(estudiante['aptitudes'])
    
# Modifica los valores de las habilidades añadiendo una o dos habilidades
estudiantes['estudiante1']['aptitudes'].append('Marketing')
estudiantes['estudiante2']['aptitudes'].extend(['Programación', 'Escritura creativa'])
estudiantes['estudiante3']['aptitudes'].extend(['Pintura', 'Artes plasticas'])

print(estudiantes['estudiante1']['aptitudes'])  # ['Programación', 'Diseño gráfico', 'Marketing']
print(estudiantes['estudiante2']['aptitudes'])  # ['Inglés', 'Marketing', 'Programación', 'Escritura creativa']
print(estudiantes['estudiante3']['aptitudes'])  # ['Música', 'Escritura creativa', 'Pintura', 'Artes plasticas']

# Obtener las claves del diccionario como una lista
keys = estudiantes.keys()
print(keys)  # dict_keys(['estudiante1', 'estudiante2', 'estudiante3'])
# ó por estudiante
keys = estudiantes['estudiante1'].keys()
print(keys)  # dict_keys(['nombre', 'apellidos', 'sexo', 'edad', 'estado civil', 'aptitudes', 'país', 'ciudad', 'dirección'])

# Obtener los valores del diccionario como una lista
values = estudiantes['estudiante1'].values()
print(values) # dict_values(['Juan', 'Pérez García', 'Masculino', 22, 'Soltero', ['Programación', 'Diseño gráfico', 'Marketing'], 'España', 'Madrid', 'Calle Mayor 123'])

# Cambia el diccionario a una lista de tuplas utilizando el método items()
print(estudiantes['estudiante1'].items())  # dict_items([('nombre', 'Juan'), ('apellidos', 'Pérez García'), ('sexo', 'Masculino'), ('edad', 22), ('estado civil', 'Soltero'), ('aptitudes', ['Programación', 'Diseño gráfico', 'Marketing']), ('país', 'España'), ('ciudad', 'Madrid'), ('dirección', 'Calle Mayor 123')])

# Eliminar uno de los elementos del diccionario
del estudiantes['estudiante1']['aptitudes']
print(estudiantes['estudiante1']) # {'nombre': 'Juan', 'apellidos': 'Pérez García', 'sexo': 'Masculino', 'edad': 22, 'estado civil': 'Soltero', 'país': 'España', 'ciudad': 'Madrid', 'dirección': 'Calle Mayor 123'}

# Eliminar uno de los diccionarios
del estudiantes['estudiante1']
print(estudiantes)