# En Python tenemos muchas funciones incorporadas. Algunas de las funciones integradas de Python más utilizadas son las siguientes: 
# print(), len(), type(), int(), float(), str(), input(), list(), dict(), min(), max(), sum(), sorted(), open(), file(), help(), y dir().

# abs(x): Devuelve el valos absoluto de un numero
print(abs(-5))   # 

# all (iterable) : Devuelve True si todos los elementos de un iterable (como una list, tuple o set) son verdaderos, y False en caso contrario.
# si esta vacio rerorna True
numeros=[1,5,0,345,54]
print(all(numeros)) #False   
numeros1=[1,5,10,345,54]
print(all(numeros1)) # True

# any (iterable) : Devuelve True si algún elemento del iterable es verdadero. Si el iterable está vacío, devuelve Falso
numeros = [12, 67, 2, 67]
print(any(numeros))  # True
texto = ["Proyectos", "con", "Python"]
print(any(texto))  # True
valores = [5, "h", True, None]
print(any(valores))  # True
nombre = ""
print(any(nombre))  # False

# Len (x): Cuenta el numero de caracteres, incluido el espacio
print(len("Hola mundo"))  # 10

# str (x): Convierte numero a en string "x"
print(str(10)) # "10"

# int (x): Convierte la variable en numero
print(int("10"))  # 10
      
# float (x): Convierte a numero decimal
print(float(10)) # 10.0

# input ("x"): Se utiliza para obtener información del usuario. Solicita la entrada del usuario y lee una línea. Después de leer los datos, 
# los convierte en un string y los devuelve.
print(input("Ingrese su nombre")) # Ingrese su nombre

# ascii (""): Devuelve un string que contiene una representación imprimible de un objeto de caracteres ASCII y reemplaza los caracteres que no son ascii 
# con los carácteres de escape \x, \u o \U.

print(ascii("Python es interesante"))  # Python es interesante
print(ascii("Pythön es interesante"))  # Pyth\xf6n es interesante
print("Pyth\xf6n es interesante")     # Python es interesante

# bin (x): Devuelve la representación binaria de un entero especificado.
print(bin(10))  # 0b1010

# bool (x): El método python bool() convierte el valor en booleano (Verdadero o Falso) utilizando el procedimiento de prueba de verdad estándar.
# Falso si el valor se omite o es falso y Verdadero si el valor es verdadero
test = []  
print(test,'is',bool(test)) # is False
test = [0]  
print(test,'is',bool(test)) # is True 
test = 0.0  
print(test,'is',bool(test))  # 0.0 es False

# Variables en python
Primer_nombre = 'Alvaro'
Apellido_pm = 'Nunez'
Pais_ = 'Colombia'
Ciudad_ = 'Armenia'
Edad_ = 55
es_casado = True
habilidades_ = ['Quality Control', 'Matlab', 'Estadistica Descriptiva', 'Python']
Informacion_personal = {'Primer nombre':'Alvaro', 'Apellido':'Nunez', 'Pais':'Colombia', 'Ciudad':'Armenia', 'Estado civil':'Casado'}

# Imprimir los valores almacenados en las variables
print('Primer nombre:', Primer_nombre) # Alvaro"
print('Longitud primer nombre:', len(Primer_nombre)) # 6
print('Apellido: ', Apellido_pm) # Nunez
print('Longitud apellido: ', len(Apellido_pm)) # 5
print('Pais: ', Pais_) # Colombia
print('Ciudad: ', Ciudad_) # Armenia
print('Edad: ', Edad_) # 55
print('Casado: ', es_casado) # True
print('Habilidades: ', habilidades_) # ['Quality Control', 'Matlab', 'Estadistica Descriptiva', 'Python']
print('Informacion personal: ', Informacion_personal) # {'Primer nombre': 'Alvaro', 'Apellido': 'Nunez', 'Pais': 'Colombia', 'Ciudad': 'Armenia', 'Estado civil': 'Casado'}

# input: El objetivo es parar un programa en Python, esperar que el usuario introduzca un dato mediante el teclado, y tras apretar la tecla enter
# (o intro o return), almacenar este dato en una variable
Primer_nombre=input("Cual es su nombre:")
Edad_=input("Cuantos años tiene:")
print(Primer_nombre)
print(Edad_)

# Convertir datos en otro tipo de dato
# int a float
num_int = 10
print('num_int',num_int)  # 10
num_float = float(num_int)
print('num_float:', num_float) # 10.0

# float a int
gravity = 9.81
print(int(gravity)) # 9

# int a str
num_int = 10
print(num_int)                  # 10
num_str = str(num_int)
print(num_str)                  # '10'

# str a list
first_name = 'Alvaro'
print(first_name)  # 'Alvaro'
first_name_to_list = list(first_name)
print(first_name_to_list)  # ['A', 'l', 'v', 'a', 'r', 'o']