# Python datetime
# Python tiene el módulo datetime para manejar la fecha y la hora.
# El comando import datetime importa el módulo datetime de la biblioteca estándar de Python. El módulo datetime proporciona clases y funciones para
# trabajar con fechas y horas en Python.

import datetime
print(dir(datetime))
""""
['MAXYEAR', 'MINYEAR', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', 'date', 'datetime',
 'datetime_CAPI', 'sys', 'time', 'timedelta', 'timezone', 'tzinfo']

MAXYEAR: Es una constante que representa el año máximo permitido en el módulo datetime.

MINYEAR: Es una constante que representa el año mínimo permitido en el módulo datetime.

__builtins__: Es una referencia al módulo que contiene los nombres incorporados predefinidos en Python. Proporciona acceso a funciones y clases
incorporadas en el lenguaje.

__cached__: Es una referencia al archivo de caché generado cuando se importa el módulo datetime. Almacenar el resultado en caché mejora el tiempo
de carga en futuras importaciones.

__doc__: Es una cadena de documentación que contiene información sobre el módulo datetime.

__file__: Es una cadena que representa la ruta del archivo donde se encuentra el módulo datetime.

__loader__: Es un objeto que carga el módulo datetime en tiempo de ejecución.

__name__: Es una cadena que representa el nombre del módulo datetime.

__package__: Es una cadena que representa el nombre del paquete al que pertenece el módulo datetime.

__spec__: Es un objeto que contiene información sobre la especificación del módulo datetime.

date: Es una clase que representa una fecha en Python. Proporciona métodos para acceder y manipular fechas.

datetime: Es una clase que combina la fecha y la hora en un solo objeto. Proporciona métodos para trabajar con fechas y horas juntas.

datetime_CAPI: Es una referencia a la interfaz de programación de aplicaciones (API) de bajo nivel utilizada internamente por el módulo datetime.
No es relevante para la mayoría de los usuarios.

sys: Es un módulo incorporado que proporciona acceso a variables y funciones relacionadas con el intérprete de Python.

time: Es una clase que representa una hora del día en Python. Proporciona métodos para acceder y manipular horas.

timedelta: Es una clase que representa una duración o un intervalo de tiempo. Se utiliza para realizar cálculos y operaciones aritméticas con fechas
y horas.

timezone: Es una clase que representa un desplazamiento de tiempo desde UTC (Tiempo Universal Coordinado). Se utiliza para trabajar con diferentes
zonas horarias.

tzinfo: Es una clase abstracta que define la interfaz para las clases de zona horaria en Python. Se utiliza como base para crear clases de zona
horaria personalizadas.

Estas clases y funciones forman parte del módulo datetime de Python y son ampliamente utilizadas para trabajar con fechas, horas y duraciones en
programas Python. Proporcionan una amplia gama de funcionalidades para manipular y gestionar datos de tiempo en diferentes formatos y zonas horarias.
"""

""""
Con Los comandos incorporados dir o help es posible conocer las funciones disponibles en un determinado módulo. Como puedes ver, en el módulo
datetime hay muchas funciones, pero nos centraremos en date, datetime, time y timedelta. Veámoslas una a una.
"""

from datetime import datetime
now = datetime.now()
print(now)                                # 2023-07-11 14:52:20.569192
day = now.day                    
month = now.month                
year = now.year                  
hour = now.hour                  
minute = now.minute              
second = now.second
timestamp = now.timestamp()
print(day, month, year, hour, minute)     # 11 7 2023 14 52
print('timestamp', timestamp)             # timestamp 1689105140.569192  (Representa el número de segundos transcurridos desde el 1 de enero de 1970.)
print(f'{day}/{month}/{year}, {hour}:{minute}') # 11/7/2023, 14:52

# Formateo de la fecha con strftime

from datetime import datetime
new_year = datetime(2020, 1, 1)
print(new_year)      # 2020-01-01 00:00:00
day = new_year.day
month = new_year.month
year = new_year.year
hour = new_year.hour
minute = new_year.minute
second = new_year.second
print(day, month, year, hour, minute) # 1 1 2020 0 0
print(f'{day}/{month}/{year}, {hour}:{minute}')  # 1/1/2020, 0:0

# Veamos como simplificamos el codigo:

from datetime import datetime
now = datetime.now()                            
t = now.strftime("%H:%M:%S")                    # (Formato: H:M:S)
print("time:", t)                               # time: 15:07:36
time_one = now.strftime("%m/%d/%Y, %H:%M:%S")   # (Formato: mm/dd/YY, H:M:S)
print("time one:", time_one)                    # time one: 07/11/2023, 15:07:36
time_two = now.strftime("%d/%m/%Y, %H:%M:%S")   # (Formato: dd/mm/YY, H:M:S)
print("time two:", time_two)                    # time two: 11/07/2023, 15:07:36

# Conversión de una cadena (string) de texto en un objeto de tiempo, utilizando la función "strptime"

from datetime import datetime
date_string = "5 December, 2019"             # (Cadena de texto que representa una fecha en el formato "día Mes, Año".)
print("date_string =", date_string)          # date_string = 5 December, 2019
date_object = datetime.strptime(date_string, "%d %B, %Y")  # (Analiza la cadena de texto en un objeto de (%d) dia, (%B,) nombre mes y (%Y) año)
print("date_object =", date_object)          # date_object = 2019-12-05 00:00:00

# Utilización de la fecha a partir de datetime

from datetime import date
d = date(2020, 1, 1)
print(d)                             # 2020-01-01
print('Current date:', d.today())    # Current date: 2023-07-12
today = date.today()                 # (Ojeto fecha de hoy)
print("Current year:", today.year)   # Current year: 2023
print("Current month:", today.month) # Current month: 7
print("Current day:", today.day)     # Current day: 12

# Objetos temporales para representar el tiempo

from datetime import time
a = time()                                # (tiempo(hora = 0, minuto = 0, segundo = 0))
print("a =", a)                           # a = 00:00:00
b = time(10, 30, 50)                      # (hora(hora, minuto y segundo))
print("b =", b)                           # b = 10:30:50
c = time(hour=10, minute=30, second=50)   # (hora(hora, minuto y segundo))
print("c =", c)                           # c = 10:30:50
d = time(10, 30, 50, 200555)              # (time(hora, minuto, segundo, microsegundo))
print("d =", d)                           # d = 10:30:50.200555

# Diferencia entre dos puntos en el tiempo usando "date" y "datetime"

from datetime import date
from datetime import datetime
today = date(year=2019, month=12, day=5)
new_year = date(year=2020, month=1, day=1)
time_left_for_newyear = new_year - today
print('Time left for new year: ', time_left_for_newyear)   # Time left for new year:  27 days, 0:00:00
t1 = datetime(year = 2019, month = 12, day = 5, hour = 0, minute = 59, second = 0)
t2 = datetime(year = 2020, month = 1, day = 1, hour = 0, minute = 0, second = 0)
diff = t2 - t1
print('Time left for new year:', diff)  # Time left for new year: 26 days, 23:01:00

# Diferencia entre dos puntos temporales usando "timedelta"

from datetime import timedelta
t1 = timedelta(weeks=12, days=10, hours=4, seconds=20)
t2 = timedelta(days=7, hours=5, minutes=3, seconds=30)
t3 = t1 - t2
print("t3 =", t3)   # t3 = 86 days, 22:56:50


# Ejercicios

# Obtener el día, mes, año, hora, minuto y fecha actual del módulo datetime

from datetime import datetime
now = datetime.now()  # (Obtener la fecha y hora actual)
day = now.day         # (Obtener el día actual)
month = now.month     # (Obtener el mes actual)
year = now.year       # (Obtener el año actual)
hour = now.hour       # (Obtener la hora actual)
minute = now.minute   # # Obtener los minutos actuales
print("Día:", day, "Mes:", month, "Año:", year, "Hora:", hour, "Minuto:", minute) # Día: 12 Mes: 7 Año: 2023 Hora: 16 Minuto: 40

# Formatea la fecha actual utilizando este formato "%m/%d/%Y, %H:%M:%S")

from datetime import datetime
now = datetime.now()
formatted_date = now.strftime("%m/%d/%Y, %H:%M:%S")
print("Fecha formateada:", formatted_date)   # Fecha formateada: 07/12/2023, 16:42:52

# Hoy es 5 diciembre, 2019. Cambiar esta cadena de tiempo al tiempo.

from datetime import datetime
date_string = "5 December, 2019"   # (Nota el mes debe estar escrito en ingles, de lo contrario error)
date_object = datetime.strptime(date_string, "%d %B, %Y")
print("Objeto de tiempo:", date_object)   # Objeto de tiempo: 2019-12-05 00:00:00

# Calcula la diferencia horaria entre ahora y el año nuevo.

from datetime import datetime
t1 = datetime.now()
t2 = datetime(year = 2024, month = 1, day = 1, hour = 0, minute = 0, second = 0)
diff = t2 - t1
print('Tiempo restante para año nuevo:', diff)  # Tiempo restante para año nuevo: 172 days, 7:02:30.419360

# Otra forma

from datetime import datetime, timedelta
now = datetime.now()
new_year = datetime(now.year + 1, 1, 1, 0, 0, 0) # Obtener la fecha y hora del Año Nuevo (1 de enero del próximo año a las 00:00:00)
time_difference = new_year - now
print("Diferencia horaria hasta el Año Nuevo:", time_difference.days, "días,", time_difference.seconds // 3600, "horas,", (time_difference.seconds // 60) % 60, "minutos")
# Diferencia horaria hasta el Año Nuevo: 172 días, 6 horas, 57 minutos

import datetime
def time_difference(new_year, now):
  difference = new_year - now
  return difference.total_seconds()
now = datetime.datetime.now()
new_year = datetime.datetime(2023, 1, 1)
difference = time_difference(new_year, now)
print(f"La diferencia horaria entre ahora y año nuevo es {difference} segundos.")
# La diferencia horaria entre ahora y año nuevo es -16650575.4323 segundos.

# Otra forma

import datetime
def time_difference(now, new_year):
  difference = new_year - now
  days = difference.days
  hours = difference.seconds // 3600
  minutes = (difference.seconds // 60) % 60
  seconds = difference.seconds % 60
  return days, hours, minutes, seconds
now = datetime.datetime.now()
new_year = datetime.datetime(2023, 1, 1)
days, hours, minutes, seconds = time_difference(now, new_year)
print(f"La diferencia horaria entre ahora y año nuevo es: \n" f"{days} dias, {hours} horas, {minutes} minutos, y {seconds} segundos.")
"""
La diferencia horaria entre ahora y año nuevo es: 
-193 dias, 6 horas, 45 minutos, y 25 segundos.
"""

# Calcula la diferencia horaria entre el 1 de enero de 1970 y ahora.

from datetime import datetime
t1 = datetime.now()
t2 = datetime(year = 1970, month = 1, day = 1, hour = 0, minute = 0, second = 0)
diff = t1 - t2
print('La diferencia horaria entre el 1 de enero de 1970 y ahora es:', diff) 
# la diferencia horaria entre el 1 de enero de 1970 y ahora es: 19550 days, 17:19:22.104776

# Otra froma

from datetime import datetime
now = datetime.now()
epoch = datetime(1970, 1, 1, 0, 0, 0)
time_difference = now - epoch
print("Diferencia horaria desde el 1 de enero de 1970:", time_difference.days, "días,", time_difference.seconds // 3600, "horas,", (time_difference.seconds // 60) % 60, "minutos,", time_difference.seconds % 60, "segundos")
# Diferencia horaria desde el 1 de enero de 1970: 19550 días, 17 horas, 21 minutos, 25 segundos

# Piensa, ¿para qué puedes utilizar el módulo datetime? Por ejemplo:
  # Análisis de series temporales
  # Para obtener una marca de tiempo de cualquier actividad en una aplicación
  # Añadir entradas en un blog

  # Análisis de series temporales

"""
Un ejemplo práctico del análisis de series temporales utilizando el módulo datetime podría ser el análisis de datos de ventas diarias en una
tienda. Supongamos que tienes un conjunto de datos que registra las ventas diarias de una tienda durante varios años.

Para analizar estas series temporales, podrías utilizar el módulo datetime para realizar operaciones como agrupar las ventas por mes, calcular
la media mensual de ventas, identificar los días de la semana con mayor o menor actividad, etc.
"""
from datetime import datetime

# Supongamos que tenemos una lista de tuplas con el formato (fecha, ventas)
sales_data = [
    ("2022-01-01", 100),
    ("2022-01-02", 150),
    ("2022-02-01", 200),
    ("2022-02-02", 250),
    # ...
]

# Creamos un diccionario para almacenar las ventas mensuales
monthly_sales = {}

# Recorremos los datos de ventas
for date_str, sales in sales_data:
    # Convertimos la cadena de fecha a un objeto datetime
    date = datetime.strptime(date_str, "%Y-%m-%d")
    
    # Obtenemos el mes y el año de la fecha
    month = date.month
    year = date.year
    
    # Creamos una clave única para identificar el mes y el año
    key = (year, month)
    
    # Actualizamos el total de ventas para ese mes
    if key in monthly_sales:
        monthly_sales[key] += sales
    else:
        monthly_sales[key] = sales

# Calculamos la media mensual de ventas
for key, sales in monthly_sales.items():
    year, month = key
    average_sales = sales / 30  # Suponiendo 30 días en un mes
    print(f"Mes {month} del año {year}: Ventas promedio = {average_sales}")

"""
Mes 1 del año 2022: Ventas promedio = 8.333333333333334
Mes 2 del año 2022: Ventas promedio = 15.0
"""

  # Para obtener una marca de tiempo de cualquier actividad en una aplicación
  
"""
Un ejemplo práctico de utilizar el módulo datetime para obtener una marca de tiempo de cualquier actividad en una aplicación sería registrar
el momento en que se realizan ciertas acciones o eventos dentro de la aplicación.

Supongamos que tienes una aplicación de chat y deseas registrar la hora en que se envía un mensaje. Puedes utilizar el módulo datetime para
obtener la marca de tiempo actual y almacenarla junto con el mensaje en una base de datos u otro sistema de almacenamiento.
"""

from datetime import datetime

def enviar_mensaje(usuario, mensaje):
    # Obtener la marca de tiempo actual
    timestamp = datetime.now()
    
    # Realizar acciones relacionadas con el envío del mensaje
    
    # ...
    
    # Imprimir el mensaje y la marca de tiempo en la consola
    print(f"Mensaje enviado por {usuario} a las {timestamp}: {mensaje}")

# Ejemplo de uso
usuario = "John"
mensaje = "Hola, ¿cómo estás?"
enviar_mensaje(usuario, mensaje)
# Mensaje enviado por John a las 2023-07-12 17:37:44.881752: Hola, ¿cómo estás?

  # Añadir entradas en un blog

"""
En este ejemplo, la función agregar_entrada_blog recibe un título y contenido para una entrada del blog. Luego, utiliza el módulo datetime para
obtener la fecha y hora actual y guardar esa información junto con el título y contenido en una base de datos o archivo. En este caso, simplemente
imprimimos la información en la consola para simular el proceso de guardado.
"""
from datetime import datetime

def agregar_entrada_blog(titulo, contenido):
    # Obtener la fecha y hora actual
    fecha_actual = datetime.now()

    # Guardar la entrada del blog en una base de datos o archivo
    # Aquí solo imprimiremos la información en la consola como ejemplo
    print("Nueva entrada de blog:")
    print("Fecha:", fecha_actual)
    print("Título:", titulo)
    print("Contenido:")
    print(contenido)

# Ejemplo de uso
titulo = "Mi primera entrada"
contenido = "¡Hola a todos! Este es mi primer artículo en el blog."
agregar_entrada_blog(titulo, contenido)
"""
Nueva entrada de blog:
Fecha: 2023-07-12 17:43:03.988895
Título: Mi primera entrada
Contenido:
¡Hola a todos! Este es mi primer artículo en el blog.
"""