"""
Clases y objetos
Python es un lenguaje de programación orientado a objetos. Todo en Python es un objeto, con sus propiedades y métodos. Un número, cadena, lista, diccionario, tupla, conjunto, etc. utilizado en un programa es un objeto de una clase incorporada correspondiente. Creamos una clase para crear un objeto. Una clase es como un constructor de objetos, o un "plano" para crear objetos. Instanciamos una clase para crear un objeto. La clase define los atributos y el comportamiento del objeto, mientras que el objeto, por su parte, representa a la clase.

Hemos estado trabajando con clases y objetos desde el principio de este reto sin saberlo. Cada elemento en un programa Python es un objeto de una clase. Comprobemos si todo en python es una clase:
"""

num = 10
tipo_num = type(num)
print(tipo_num)  # <class 'int'>

cadena = 'string'
tipo_cadena = type(cadena)
print(tipo_cadena)  # <class 'str'>

booleano = True
tipo_booleano = type(booleano)
print(tipo_booleano)  # <class 'bool'>

lista = []
tipo_lista = type(lista)
print(tipo_lista)  # <class 'list'>

tupla = ()
tipo_tupla = type(tupla)
print(tipo_tupla)  # <class 'tuple'>

conjunto = set()
tipo_conjunto = type(conjunto)
print(tipo_conjunto)  # <class 'set'>

diccionario = {}
tipo_diccionario = type(diccionario)
print(tipo_diccionario)  # <class 'dict'>


"""
Crear una clase: Para crear una clase necesitamos la palabra clave class seguida del nombre y dos puntos.

# syntax
class ClassName:
  code goes here
  
Crear un objeto: Podemos crear un objeto llamando a la clase.
"""

class Person:
  pass
print(Person)  # <class '__main__.Person'>

p = Person()
print(p)  # <__main__.Person object at 0x000002526E274250>

"""
Constructor de clase
En los ejemplos anteriores, hemos creado un objeto a partir de la clase Person. Sin embargo, una clase sin constructor no es realmente útil en aplicaciones reales. Usemos la función constructor para hacer nuestra clase más útil. Python también tiene una función constructora init() incorporada. La función constructora init tiene el parámetro self que es una referencia a la instancia actual de la clase Examples:
"""

class Person:
      def __init__ (self, nombre):
        # self permite adjuntar parámetros a la clase
          self.nombre =nombre

p = Person('Alvaro')
print(p.nombre)
print(p)
"""
Alvaro
<__main__.Person object at 0x000002572C214190>
"""

# Adicionemos mas parametros a la funcion constructora init():

class Person:
      def __init__(self, firstname, lastname, age, country, city):
          self.firstname = firstname
          self.lastname = lastname
          self.age = age
          self.country = country
          self.city = city


p = Person('Alvaro', 'Nunez', 56, 'Colombia', 'Armenia')
print(p.firstname)  # Alvaro
print(p.lastname)  # Nunez
print(p.age)  # 56
print(p.country)  # Colombia
print(p.city)  # Armenia


# Métodos de los objetos: Los objetos pueden tener métodos. Los métodos son funciones que pertenecen al objeto.

class Person:
      def __init__(self, firstname, lastname, age, country, city):
          self.firstname = firstname
          self.lastname = lastname
          self.age = age
          self.country = country
          self.city = city
      def person_info(self):
        return f'{self.firstname} {self.lastname} is {self.age} years old. He lives in {self.city}, {self.country}'

p = Person('Alvaro', 'Nunez', 56, 'Colombia', 'Armenia')
print(p.person_info()) # Alvaro Nunez is 56 years old. He lives in Armenia, Colombia

"""
Métodos por Defecto de Objetos
A veces, es posible que desees tener valores predeterminados para los métodos de tu objeto. Si proporcionamos valores predeterminados para los parámetros en el constructor, podemos evitar errores cuando llamamos o creamos nuestra clase sin parámetros. Veamos cómo se ve esto:
"""

class Person:
      def __init__(self, firstname='Alvaro', lastname='Nunez', age=56, country='Colombia', city='Armenia'):
          self.firstname = firstname
          self.lastname = lastname
          self.age = age
          self.country = country
          self.city = city

      def person_info(self):
        return f'{self.firstname} {self.lastname} is {self.age} years old. He lives in {self.city}, {self.country}.'

p1 = Person()
print(p1.person_info())  # Alvaro Nunez is 56 years old. He lives in Armenia, Colombia.
p2 = Person('Sebastian', 'Nunez', 19, 'Armenia', 'Colombia')
print(p2.person_info())  # Sebastian Nunez is 19 years old. He lives in Colombia, Armenia.

"""
Método para Modificar los Valores por Defecto de una Clase
En el siguiente ejemplo, la clase persona, todos los parámetros del constructor tienen valores por defecto. Además de eso, tenemos el parámetro skills, al que podemos acceder usando un método. Creamos el método add_skill para añadir habilidades a la lista de habilidades.
"""

class Person:
      def __init__(self, firstname='Alvaro', lastname='Nunez', age=56, country='Colombia', city='Armenia'):
          self.firstname = firstname
          self.lastname = lastname
          self.age = age
          self.country = country
          self.city = city
          self.skills = []

      def person_info(self):
        return f'{self.firstname} {self.lastname} is {self.age} years old. He lives in {self.city}, {self.country}.'
      def add_skill(self, skill):
          self.skills.append(skill)

p1 = Person()
print(p1.person_info())  # Alvaro Nunez is 56 years old. He lives in Armenia, Colombia.
p1.add_skill('MATLAB')
p1.add_skill('Quality Control')
p1.add_skill('Python')
p2 = Person('Nicolas', 'Nunez', 22, 'Armenia', 'Quindio') 
print(p2.person_info()) # Nicolas Nunez is 22 years old. He lives in Quindio, Armenia.
print(p1.skills)  # ['MATLAB', 'Quality Control', 'Python']
print(p2.skills)  # []

"""
Herencia: Usando la herencia podemos reutilizar el código de la clase padre. La herencia nos permite definir una clase que hereda todos los métodos y propiedades de la clase padre. La clase padre o super o clase base es la clase que da todos los métodos y propiedades. La clase hija es la clase que hereda de otra clase o clase padre. Vamos a crear una clase estudiante heredando de la clase persona.
"""
class Person:
      def __init__(self, firstname='Alvaro', lastname='Nunez', age=56, country='Colombia', city='Armenia'):
          self.firstname = firstname
          self.lastname = lastname
          self.age = age
          self.country = country
          self.city = city
          self.skills = []

      def person_info(self):
        return f'{self.firstname} {self.lastname} is {self.age} years old. He lives in {self.city}, {self.country}.'
      def add_skill(self, skill):
          self.skills.append(skill)

class Student(Person):
    pass

s1 = Student('Nicolas', 'Nunez', 22, 'Armenia', 'Quindio')
s2 = Student('Sebastian', 'Nunez', 19, 'Armenia', 'Colombia')
print(s1.person_info())  # Nicolas Nunez is 22 years old. He lives in Quindio, Armenia.
s1.add_skill('Soccer')
s1.add_skill('Business')
s1.add_skill('Resilience')
print(s1.skills)  # ['Soccer', 'Business', 'Resilience']

print(s2.person_info())  # Sebastian Nunez is 19 years old. He lives in Colombia, Armenia.
s2.add_skill('Organizing')
s2.add_skill('Marketing')
s2.add_skill('Digital Marketing')
print(s2.skills)  # ['Organizing', 'Marketing', 'Digital Marketing']


# Anulación del método padre

class Person:
    def __init__(self, firstname='Alvaro', lastname='Nunez', age=56, country='Colombia', city='Armenia'):
        self.firstname = firstname
        self.lastname = lastname
        self.age = age
        self.country = country
        self.city = city
        self.skills = []

    def person_info(self):
        return f'{self.firstname} {self.lastname} is {self.age} years old. He lives in {self.city}, {self.country}.'
    def add_skill(self, skill):
        self.skills.append(skill)

class Student(Person):

 def __init__ (self, firstname='Alvaro', lastname='Nunez', age=56, country='Colombia', city='Armenia', gender='male'):
        self.gender = gender
        super().__init__(firstname, lastname,age, country, city)
 def person_info(self):
        gender = 'He' if self.gender =='male' else 'She'
        return f'{self.firstname} {self.lastname} is {self.age} years old. {gender} lives in {self.city}, {self.country}.'

s1 = Student('Nicolas', 'Nunez', 22, 'Colombia', 'Armenia', 'male')
s2 = Student('Sebastian', 'Nunez', 19, 'Armenia', 'Colombia', 'male')
print(s1.person_info())  # Nicolas Nunez is 22 years old. He lives in Armenia, Colombia.
s1.add_skill('Soccer')
s1.add_skill('Business')
s1.add_skill('Resilience')
print(s1.skills)  # ['Soccer', 'Business', 'Resilience']

print(s2.person_info())  # Sebastian Nunez is 19 years old. He lives in Colombia, Armenia.
s2.add_skill('Organizing')
s2.add_skill('Marketing')
s2.add_skill('Digital Marketing')
print(s2.skills)  # ['Organizing', 'Marketing', 'Digital Marketing']

"""
Podemos utilizar la función incorporada super() o el nombre padre Persona para heredar automáticamente los métodos y propiedades de su padre. En el ejemplo anterior anulamos el método padre. El método hijo tiene una característica diferente, puede identificar, si el género es masculino o femenino y asignar el pronombre apropiado(Él/Ella).
"""

# Ejercicios

"""
Crear una clase llamada Estadística con las funciones que realicen cálculos estadísticos para determinar el numero de datos, la media, mediana, moda, rango, varianza, desviación típica , mínimo, máximo, recuento, percentil y distribución de frecuencias de la siguiente lista:  "ages = [31, 26, 34, 37, 27, 26, 32, 32, 26, 27, 27, 24, 32, 33, 27, 25, 26, 38, 37, 31, 34, 24, 33, 29, 26]"
"""

class Estadistica:
    def __init__(self, datos):
        self.datos = datos

    def numero_de_datos(self):
        return len(self.datos)

    def media(self):
        return sum(self.datos) / len(self.datos)

    def mediana(self):
        datos_ordenados = sorted(self.datos)
        n = len(datos_ordenados)
        if n % 2 == 1:
            return datos_ordenados[n // 2]
        else:
            mediana_inf = datos_ordenados[(n // 2) - 1]
            mediana_sup = datos_ordenados[n // 2]
            return (mediana_inf + mediana_sup) / 2

    def moda(self):
        from collections import Counter
        conteo = Counter(self.datos)
        moda = conteo.most_common(1)
        return moda[0][0]

    def rango(self):
        return max(self.datos) - min(self.datos)

    def varianza(self):
        media = self.media()
        sumatoria = sum((x - media) ** 2 for x in self.datos)
        return sumatoria / (len(self.datos) - 1)

    def desviacion_tipica(self):
        import math
        return math.sqrt(self.varianza())

    def minimo(self):
        return min(self.datos)

    def maximo(self):
        return max(self.datos)

    def recuento(self, valor):
        return self.datos.count(valor)

    def percentil(self, p):
        datos_ordenados = sorted(self.datos)
        n = len(datos_ordenados)
        posicion = (p / 100) * (n - 1)
        if int(posicion) == posicion:
            return datos_ordenados[int(posicion)]
        else:
            inf = datos_ordenados[int(posicion)]
            sup = datos_ordenados[int(posicion) + 1]
            fraccion = posicion - int(posicion)
            return inf + (sup - inf) * fraccion

    def distribucion_de_frecuencias(self):
        from collections import Counter
        conteo = Counter(self.datos)
        return dict(conteo)

# Datos de ejemplo
ages = [31, 26, 34, 37, 27, 26, 32, 32, 26, 27, 27, 24, 32, 33, 27, 25, 26, 38, 37, 31, 34, 24, 33, 29, 26]

# Crear una instancia de la clase Estadistica
estadisticas = Estadistica(ages)

# Ejemplos de uso
print("Número de datos:", estadisticas.numero_de_datos())  # Número de datos: 25
print("Media:", estadisticas.media())  # Media: 29.76
print("Mediana:", estadisticas.mediana())  # Mediana: 29
print("Moda:", estadisticas.moda())  # Moda: 26
print("Rango:", estadisticas.rango())  # Rango: 14
print("Varianza:", estadisticas.varianza())  # Varianza: 18.273333333333333
print("Desviación típica:", estadisticas.desviacion_tipica())  # Desviación típica: 4.2747319604079665
print("Mínimo:", estadisticas.minimo())  # Mínimo: 24
print("Máximo:", estadisticas.maximo())  # Máximo: 38
print("Recuento de 26:", estadisticas.recuento(26))  # Recuento de 26: 5
print("Percentil 75:", estadisticas.percentil(75))  # Percentil 75: 33
print("Distribución de frecuencias:", estadisticas.distribucion_de_frecuencias()) # Distribución de frecuencias: {31: 2, 26: 5, 34: 2, 37: 2, 27: 4, 32: 3, 24: 2, 33: 2, 25: 1, 38: 1, 29: 1}

# Otra forma:

from collections import Counter

class Estadistica:
    def __init__(self, datos):
        self.datos = datos

    def numero_de_datos(self):
        return len(self.datos)

    def media(self):
        return sum(self.datos) / len(self.datos)

    def mediana(self):
        datos_ordenados = sorted(self.datos)
        n = len(datos_ordenados)
        if n % 2 == 1:
            return datos_ordenados[n // 2]
        else:
            mediana_inf = datos_ordenados[(n // 2) - 1]
            mediana_sup = datos_ordenados[n // 2]
            return (mediana_inf + mediana_sup) / 2

    def moda(self):
        conteo = Counter(self.datos)
        moda = conteo.most_common(1)
        return moda[0][0]

    def rango(self):
        return max(self.datos) - min(self.datos)

    def varianza(self):
        media = self.media()
        sumatoria = sum((x - media) ** 2 for x in self.datos)
        return sumatoria / len(self.datos)

    def desviacion_tipica(self):
        return self.varianza() ** 0.5

    def minimo(self):
        return min(self.datos)

    def maximo(self):
        return max(self.datos)

    def recuento(self, valor):
        return self.datos.count(valor)

    def percentil(self, p):
        datos_ordenados = sorted(self.datos)
        n = len(datos_ordenados)
        posicion = (p / 100) * (n - 1)
        if int(posicion) == posicion:
            return datos_ordenados[int(posicion)]
        else:
            inf = datos_ordenados[int(posicion)]
            sup = datos_ordenados[int(posicion) + 1]
            fraccion = posicion - int(posicion)
            return inf + (sup - inf) * fraccion

    def distribucion_de_frecuencias(self):
        conteo = Counter(self.datos)
        return dict(conteo)

# Datos de ejemplo
ages = [31, 26, 34, 37, 27, 26, 32, 32, 26, 27, 27, 24, 32, 33, 27, 25, 26, 38, 37, 31, 34, 24, 33, 29, 26]

# Crear una instancia de la clase Estadistica
estadisticas = Estadistica(ages)

# Ejemplos de uso
print("Número de datos:", estadisticas.numero_de_datos())  # Número de datos: 25
print("Media:", estadisticas.media())  # Media: 29.76
print("Mediana:", estadisticas.mediana())  # Mediana: 29
print("Moda:", estadisticas.moda())  # Moda: 26
print("Rango:", estadisticas.rango())  # Rango: 14
print("Varianza:", estadisticas.varianza())  # Varianza: 18.273333333333333
print("Desviación típica:", estadisticas.desviacion_tipica())  # Desviación típica: 4.2747319604079665
print("Mínimo:", estadisticas.minimo())  # Mínimo: 24
print("Máximo:", estadisticas.maximo())  # Máximo: 38
print("Recuento de 26:", estadisticas.recuento(26))  # Recuento de 26: 5
print("Percentil 75:", estadisticas.percentil(75))  # Percentil 75: 33
print("Distribución de frecuencias:", estadisticas.distribucion_de_frecuencias()) # Distribución de frecuencias: {31: 2, 26: 5, 34: 2, 37: 2, 27: 4, 32: 3, 24: 2, 33: 2, 25: 1, 38: 1, 29: 1}

"""
Crea una clase llamada PersonAccount. Tiene las propiedades nombre, apellido, ingresos, gastos y los métodos total_ingresos, total_gastos, contable_info, add_ingreso, add_gasto y contable_balance. Ingresos es un conjunto de ingresos y su descripción. Lo mismo ocurre con los gastos.
"""

class PersonAccount:
    def __init__(self, nombre, apellido):
        self.nombre = nombre
        self.apellido = apellido
        self.ingresos = {}
        self.gastos = {}

    def total_ingresos(self):
        return sum(self.ingresos.values())

    def total_gastos(self):
        return sum(self.gastos.values())

    def contable_info(self):
        return f"Nombre: {self.nombre} {self.apellido}\n" \
               f"Total de ingresos: {self.total_ingresos()}\n" \
               f"Total de gastos: {self.total_gastos()}\n" \
               f"Balance: {self.contable_balance()}"

    def add_ingreso(self, descripcion, monto):
        if descripcion in self.ingresos:
            self.ingresos[descripcion] += monto
        else:
            self.ingresos[descripcion] = monto

    def add_gasto(self, descripcion, monto):
        if descripcion in self.gastos:
            self.gastos[descripcion] += monto
        else:
            self.gastos[descripcion] = monto

    def contable_balance(self):
        return self.total_ingresos() - self.total_gastos()

# Ejemplo de uso:
persona = PersonAccount("Alvaro", "Nunez")
persona.add_ingreso("Salario", 3000)
persona.add_ingreso("Venta de productos", 500)
persona.add_gasto("Alquiler", 1200)
persona.add_gasto("Comida", 300)
persona.add_gasto("Transporte", 150)

print(persona.contable_info())

"""
Nombre: Alvaro Nunez
Total de ingresos: 3500
Total de gastos: 1650
Balance: 1850
"""

# Otra forma:

class PersonAccount:
    def __init__(self, nombre, apellido):
        self.nombre = nombre
        self.apellido = apellido
        self.ingresos = {}
        self.gastos = {}

    def total_ingresos(self):
        return sum(self.ingresos.values())

    def total_gastos(self):
        return sum(self.gastos.values())

    def contable_info(self):
        total_ingresos = self.total_ingresos()
        total_gastos = self.total_gastos()
        balance = total_ingresos - total_gastos
        return f"Nombre: {self.nombre} {self.apellido}\n" \
               f"Total de ingresos: {total_ingresos}\n" \
               f"Total de gastos: {total_gastos}\n" \
               f"Balance: {balance}"

    def add_ingreso(self, descripcion, monto):
        self.ingresos.setdefault(descripcion, 0)
        self.ingresos[descripcion] += monto

    def add_gasto(self, descripcion, monto):
        self.gastos.setdefault(descripcion, 0)
        self.gastos[descripcion] += monto

# Ejemplo de uso:
persona = PersonAccount("Alvaro", "Nunez")
persona.add_ingreso("Salario", 3000)
persona.add_ingreso("Venta de productos", 500)
persona.add_gasto("Alquiler", 1200)
persona.add_gasto("Comida", 300)
persona.add_gasto("Transporte", 150)

print(persona.contable_info())

"""
Nombre: Alvaro Nunez
Total de ingresos: 3500
Total de gastos: 1650
Balance: 1850
"""