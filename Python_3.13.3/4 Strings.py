# Cualquier tipo de dato escrito como texto entre comillas simples, dobles o triples es un string. Para comprobar la longitud de un string se utiliza 
# el método len().
greeting = 'Hello, World!'
print(greeting) # Hello, World!
print(len(greeting)) # 13

# Multiples strings se crean utilizando comillas simples triples (''') o comillas dobles triples ("""). Véase el ejemplo siguiente.
multiline_string = '''Soy profesor y disfruto enseñando.
No encontré nada tan gratificante como capacitar a la gente.
Por eso creé 30 días de python.'''
print(multiline_string)  # Soy profesor y disfruto enseñando....

# Otra forma de hacer lo mismo:
multiline_string = """Soy profesor y disfruto enseñando.
No encontré nada tan gratificante como capacitar a la gente.
Por eso creé 30 días de python."""
print(multiline_string) # Soy profesor y disfruto enseñando......

# Concatenacion de strings:
first_name = 'Alvaro'
last_name = 'Nunez'
space = ' '
full_name = first_name  +  space + last_name
print(full_name) # Alvaro Nunez

# Secuencias de escape en strings: En Python y otros lenguajes de programación (\) seguido de un carácter es una secuencia de escape. Ejemplos
# \n: Nueva linea
# \t: Tabulador
# \\: Barra oblicua
# \': Comilla simple (')
# \": Comilla doble (")
print('I hope everyone is enjoying the Python Challenge.\nAre you ?') # I hope everyone is enjoying the Python Challenge.
                                                                      # Are you ?  (corta el texto en otor renglon)
print('Days\tTopics\tExercises')  # Days    Topics  Exercises  (añadir espacio de tabulación o 4 espacios)                                                                     
print('Day 1\t3\t5') # Day 1   3       5
print('Day 2\t3\t5') # Day 2   3       5
print('Day 3\t3\t5') # Day 3   3       5
print('This is a backslash  symbol (\\)') # This is a backslash  symbol (\)  (Para escribir una barra invertida)
print('Programming language starts with \"Hello, World!\"') # Programming language starts with starts with "Hello, World!" (Escribir una comilla doble)

# Hay muchas formas de formatear strings. Veamos algunas de ellas. El operador "%" se utiliza para formatear un conjunto de variables encerradas 
# en un "tuple" (una lista de tamaño fijo), junto con un formato string, que contiene texto normal junto con "especificadores de argumento", 
# símbolos especiales como "%s", "%d", "%f", "%.número de dígitosf"
# %s - String (o cualquier objeto con un string, como números)
# %d - Números enteros
# %f - Números decimales
# "%.number of digitsf" - Números decimales con numero de digitos definidos
first_name = 'Alvaro'
last_name = 'Nunez'
language = 'Python'
formated_string = 'I am %s %s. I teach %s' %(first_name, last_name, language)
print(formated_string) # I am Alvaro Nunez. I teach Python (%s %s %s refiere a primer, segundo y tercer string en first_name, last_name y language)
# los primeros formatos de string %s %s traen las dos primeras variables (first_name, last_name, language) y el otro %s trae en la tercera variable, (language)
# por ello el final del codigo formated_string termina con %(first_name, last_name, language), refiriendo que el primer %s trae la variable first_name y asi sucecivamente 

radius = 10
pi = 3.14
area = pi * radius ** 2
formated_string = 'The area of circle with a radius %d is %.2f.' %(radius, area)
print(formated_string) # The area of circle with a radius 10 is 314.00.  (%d refiere al primer numero entero y %.2f a dos decimales en el resultado del area)

radius = 10
pi = 3.14
area = pi * radius ** 2
formated_string = 'The area of circle with a radius %d y pi %.3f is %.2f.' %(radius, pi, area)
print(formated_string) # The area of circle with a radius 10 y pi 3.140 is 314.00. (%d refiere al primer numero entero (radios), %.3f tres cifras decimales de (pi)
# y %.2f a dos decimales en el resultado del (area))

python_libraries = ['Django', 'Flask', 'NumPy', 'Matplotlib','Pandas']
formated_string = 'The following are python libraries:%s' % (python_libraries)
print(formated_string) # The following are python libraries:['Django', 'Flask', 'NumPy', 'Matplotlib', 'Pandas']

# Nuevo estilo de formatos (str.format). Este formato se introdujo en la versión 3 de Python
first_name = 'Alvaro'
last_name = 'Nunez'
language = 'Python'
formated_string = 'I am {} {}. I teach {}'.format(first_name, last_name, language)
print(formated_string) # I am Alvaro Nunez. I teach Python ({}{}{} refiere a primer, segundo y tercer string en first_name, last_name y language)
# El primer formato de strig opera en la primera variable de .format(first_name, last_name, language) la segunada en la segunda variable y asi sucecivamente.

a = 4
b = 3
print('{} + {} = {}'.format(a, b, a + b)) # 4 + 3 = 7
print('{} - {} = {}'.format(a, b, a - b)) # 4 - 3 = 1
print('{} * {} = {}'.format(a, b, a * b)) # 4 * 3 = 12
print('{} / {} = {:.2f}'.format(a, b, a / b)) # 4 / 3 = 1.33 (lo limita a dos dígitos después del decimal)
print('{} % {} = {}'.format(a, b, a % b)) # 4 % 3 = 1
print('{} // {} = {}'.format(a, b, a // b)) # 4 // 3 = 1
print('{} ** {} = {}'.format(a, b, a ** b)) # 4 ** 3 = 64

radius = 10
pi = 3.14
area = pi * radius ** 2
formated_string = 'The area of a circle with a radius {} is {:.2f}.'.format(radius, area)
print(formated_string) # The area of a circle with a radius 10 is 314.00.

radius = 10
pi = 3.14
area = pi * radius ** 2
formated_string = 'The area of a circle with a radius {} and pi {:.3f} is {:.2f}.'.format(radius, pi, area)
print(formated_string) # The area of a circle with a radius 10 and pi 3.140 is 314.00.

# Interpolación de strins / f-strings (Python 3.6+)
# Otro nuevo formato string es la interpolación de strings, f-strings. Los strings empiezan por f y podemos colocar los datos en sus posiciones correspondientes.
a = 4
b = 3
print(f'{a} + {b} = {a + b}') # 4 + 3 = 7
print(f'{a} - {b} = {a - b}') # 4 - 3 = 1
print(f'{a} * {b} = {a * b}') # 4 * 3 = 12
print(f'{a} / {b} = {a / b:.2f}') # 4 / 3 = 1.33
print(f'{a} % {b} = {a % b}') # 4 % 3 = 1
print(f'{a} // {b} = {a // b}') # 4 // 3 = 1
print(f'{a} ** {b} = {a ** b}') # 4 ** 3 = 64

# Los strings de Python como secuencias de caracteres
# Los strings de Python son secuencias de caracteres, y comparten sus métodos básicos de acceso con otras secuencias ordenadas de objetos de Python: 
# listas y tuples. La forma más sencilla de extraer caracteres individuales de los strings (y miembros individuales de cualquier secuencia) 
# es descomprimirlos en las variables correspondientes.
language = 'Python'
a,b,c,d,e,f = language # asignar a cada caracter de secuencia una variable
print(a) # P
print(b) # y
print(c) # t
print(d) # h
print(e) # o
print(f) # n

# Acceso a caracteres de cadenas por índice: En programación, la indexacion positiva la cuenta empieza por cero. Por lo tanto, la primera letra de una 
# cadena está en el índice cero y la última letra de una cadena es la longitud de una cadena menos uno.

# ['p',   'y'   't'   'h'   'o'   'n']
#   0      1     2     3     4     5

# Acceso a caracteres de strings por índice
# En programación, las cuentas empiezan por cero. Por lo tanto, la primera letra de un string está en el índice cero y la última letra de un string 
# es la longitud de un string menos uno.
language = 'Python'
first_letter = language[0]
print(first_letter) # P
second_letter = language[1]
print(second_letter) # y
third_letter = language[2]
print(third_letter) # t
fourth_letter = language[3]
print(fourth_letter) # h
fifth_letter=language[4]
print(fifth_letter) # o
last_index = len(language) - 1
last_letter = language[last_index]
print(last_letter) # n

# Si queremos empezar por el extremo derecho podemos utilizar la indexación negativa. -1 es el último índice.

# ['p',   'y'   't'   'h'   'o'   'n']
#  -6     -5    -4     -3   -2     -1

language = 'Python'
last_letter = language[-1]
print(last_letter) # n
second_last = language[-2]
print(second_last) # o

# Cortar strings en Python
# En Python podemos dividir strings en substrings.
language = 'Python'
first_three = language[0:3] # Toma desde caracter de índice 0 hasta 3 pero no incluye 3
print(first_three) # Pyt
last_three = language[3:6] # Toma desde caracter de indice 3 hasta 6 pero no incluye 6 (no hay indice 6, per esto toma el ultimo caracter que es indice 5)
print(last_three) # hon

# Otra forma es:
last_three = language[-3:] # Toma los ultimos 3 caracteres
print(last_three)   # hon
last_three = language[3:]  # Toma desde caracter de indice 3 hasta el ultimo
print(last_three)   # hon
last_three = language[-4:] # Toma los ultimos 4 caracteres
print(last_three)   # thon
last_three = language[4:] # Toma desde caracter de indice 4 hasta el ultimo
print(last_three)   # on

# Invertir un string. Podemos invertir strings fácilmente en python.
greeting = 'Hola Mundo!'
print(greeting[::-1]) # !odnuM aloH

# Saltar caracteres durante el corte. Es posible omitir caracteres durante el corte pasando el argumento "paso" al método de corte.
language = 'Python'
alva = language[0:6:2] # Tomar caracteres de indice 0 hasta 6-1 [0:6..], y el valor de "paso" [..:2] indica la cantidad de caracteres que se salta en cada iteración.
print(alva) # Pto

language = 'Parangutirimicuaro'
pto = language[0:19:2] # Tomar caracteres de indice 0 hasta 19-1 [0:19..], y el valor de "paso" [..:2] indica la cantidad de caracteres que se salta en cada iteración.
print(pto) # Prnuiiiur

# Métodos de string  Existen muchos métodos que nos permiten formatear string. Veamos algunos de los métodos siguiente ejemplo:

# capitalize(): Convierte el primer carácter de la cadena en mayúscula.
challenge = 'thirty days of python'
print(challenge.capitalize()) # 'Thirty days of python'

# count(): Devuelve las ocurrencias de los substring en el string, count(substring, start=.., end=..). El inicio es un índice inicial para contar y el 
# fin es el último índice para contar.
challenge = 'thirty days of python'
print(challenge.count('y')) # 3  (Se utiliza la función "count()" para contar el número de veces que aparece el carácter 'y')

# La función "count()" puede aceptar dos argumentos adicionales opcionales: "start" y "end". En este caso, se especifica "y" como el subconjunto 
# a buscar y se establece "7" como el valor de "start" y "14" como el valor de "end". Esto significa que se buscará el carácter 'y' solo en 
# el indice del string de la variable "challenge" que va desde el índice 7 hasta el índice 13 (excluyendo el índice 14). Veamos:
challenge = 'thirty days of python'
print(challenge.count('y', 7, 14)) # 1

# De manera analoga
print(challenge.count('y', 0, 6)) # 1
print(challenge.count('y', 0, 5)) # 0

# endswith(): Comprueba si una cadena termina con una terminación especificada.
challenge = 'thirty days of python'
print(challenge.endswith('on'))   # True
print(challenge.endswith('tion')) # False

# expandtabs(): Sustituye el carácter de tabulación por espacios. Tambien se puede ingresar el tamaño de los espacios
challenge = 'thirty\tdays\tof\tpython'
print(challenge.expandtabs())   # thirty  days    of      python (3 espacios despues primer palabra, 5 despues de la segunda y luego 7)
print(challenge.expandtabs(10)) # thirty    days      of        python (5 espacios despues primer palabra, 7 despues de la segunda y luego 9)

# find(): Devuelve el numero índice de la primera aparición de un substring, si no se encuentra devuelve -1
challenge = 'thirty days of python'
print(challenge.find('y'))  # 5
print(challenge.find('th')) # 0
print(challenge.find('zo')) # -1

# rfind(): Devuelve el numero índice de la última aparición de un substring, si no se encuentra devuelve -1
challenge = 'thirty days of python'
print(challenge.rfind('y'))  # 16
print(challenge.rfind('th')) # 17
print(challenge.rfind('zo')) # -1

# format(): formatea la cadena en una salida más agradable
first_name = 'Alvaro'
last_name = 'Nunez'
age = 55
job = 'learner'
country = 'Colombia'
sentence = 'I am {} {}. I am {} years old. I am a {}. I live in {}.'.format(first_name, last_name, age, job, country)
print(sentence) # I am Alvaro Nunez. I am 55 years old. I am a learner. I live in Colombia.

radius = 10
pi = 3.14
area = pi * radius ** 2
result = 'The area of a circle with radius {} is {}'.format(str(radius), str(area)) #(str convierte numeros en string)
print(result) # The area of a circle with radius 10 is 314.0

# index(): Devuelve el numero índice más bajo de un substring, los argumentos adicionales indican el índice inicial y final (por defecto 0 y longitud de string - 1). 
# Si no se encuentra el substring, el resultado sera valueError. index() busca desde el inicio de la cadena hacia el final devolviendo el numero de indice de caracter
challenge = 'thirty days of python'
sub_string = 'da'
print(challenge.index(sub_string))  # 7
print(challenge.index(sub_string, 7)) # 7
print(challenge.index(sub_string, 8)) # ValueError: substring not found (Busca "da" desde "ays of python")
print(challenge.index(sub_string, 9)) # ValueError: substring not found (Busca "da" desde "ys of python")

saludo = 'Hola mundo'
sub_string = 'o'
print(saludo.index(sub_string))  # 1

# rindex(): De manera analoga a index(), pero busca desde el final de la cadena hacia el inicio, devolviendo el numero de indice de caracter
challenge = 'thirty days of python'
sub_string = 'da'
print(challenge.rindex(sub_string))  # 7
print(challenge.rindex(sub_string, 7)) # 7
print(challenge.rindex(sub_string, 8)) # ValueError: substring not found

saludo = 'Hola mundo'
sub_string = 'o'
print(saludo.rindex(sub_string)) # 9

challenge = 'You cannot end a sentence with because because because is a conjunction'
sub_string = 'because'
print(challenge.index(sub_string)) # 31
print(challenge.rindex(sub_string)) # 47

# isalnum(): Comprueba si hay carácteres alfanuméricos
challenge = 'ThirtyDaysPython'
print(challenge.isalnum()) # True

challenge = '30DaysPython'
print(challenge.isalnum()) # True

challenge = 'thirty days of python'
print(challenge.isalnum()) # False (El espacio no es un carácter alfanumérico)

challenge = 'thirty days of python 2019'
print(challenge.isalnum()) # False (2019 no es alfanumerico)

# isalpha(): Comprueba si todos los elementos del string son caracteres del alfabeto (a-z y A-Z).
challenge = 'thirty days of python'
print(challenge.isalpha()) # False (El espacio queda excluido como caracter alfabetico)
challenge = 'ThirtyDaysPython'
print(challenge.isalpha()) # True
num = '123'
print(num.isalpha())      # False

# isdecimal(): Comprueba si todos los caracteres de un string son numericos (0-9). El resultado sera "False" si estan presentes caracteres diferentes 
# a numeros, si hay espacios o si son numeros arabigos o chinos
challenge = 'thirty days of python'
print(challenge.isdecimal())  # False
challenge = '123'
print(challenge.isdecimal())  # True
challenge = '\u00B2'
print(challenge.isdigit())   # True (Los carácteres "\u00B2" corresponde al símbolo de "superíndice 2" o "exponente 2" en Unicode)
challenge = '12 3'
print(challenge.isdecimal())  # False
num = '\u00BD' # El caracter "\u00BD" corresponde al numero decimal 1/2
print(num.isdecimal()) # False

# isdigit(): Comprueba si todos los caracteres de un string son números (0-9 y algunos otros caracteres unicode para números). El resultado sera "False"
# si estan presentes caracteres diferentes a numeros o si hay espacios. El resultado sera "True" si hay numeros arabigos o chinos
challenge = 'Thirty'
print(challenge.isdigit()) # False
challenge = '30'
print(challenge.isdigit())   # True
challenge = '\u00B2'
print(challenge.isdigit())   # True  (El carácter \u00B2 corresponde al símbolo de "superíndice 2" o "exponente 2" en Unicode)
num = '\u00BD' # El caracter '\u00BD' corresponde al numero decimal 1/2
print(num.isdigit()) # False


# isnumeric(): Comprueba si todos los caracteres de una cadena son números o están relacionados con números 
# (igual que isdigit(), sólo que acepta más símbolos, como ½).
num = '10'
print(num.isnumeric()) # True
num = '\u00BD' # El caracter '\u00BD' corresponde al numero decimal 1/2
print(num.isnumeric()) # True
num = '10.5'
print(num.isnumeric()) # False

# isidentifier(): Comprueba si un identificador es válido - comprueba si un string es un nombre de variable válido.
challenge = '30DaysOfPython'
print(challenge.isidentifier()) # False (Porque empieza por un número)
challenge = 'thirty_days_of_python'
print(challenge.isidentifier()) # True

# islower(): Comprueba si todos los caracteres del alfabeto de la cadena están en minúsculas.
challenge = 'thirty days of python'
print(challenge.islower()) # True
challenge = 'Thirty days of python'
print(challenge.islower()) # False

# isupper(): Comprueba si todos los caracteres del alfabeto de la cadena están en mayúsculas.
challenge = 'Thirty Days of Python'
print(challenge.isupper()) #  False
challenge = 'THIRTY DAYS OF PYTHON'
print(challenge.isupper()) # True

# join(): Devuelve una cadena concatenada
web_tech = ['HTML', 'CSS', 'JavaScript', 'React']
result1 = ''.join(web_tech)
print(result1) # HTMLCSSJavaScriptReact
result2 = ' '.join(web_tech)
print(result2) # HTML CSS JavaScript React
result3 = '& '.join(web_tech)
print(result3) # HTML& CSS& JavaScript& React

# strip(): Elimina todos los caracteres dados empezando por el principio y el final de la cadena.
challenge = 'thirty days of pythoonnn'
print(challenge.strip('noth')) # irty days of py
# En este caso, se están eliminando los caracteres 'n', 'o', 't' y 'h'. Inicialmente se busca el caracter 'n' al principio de la primera frase y al final de la
# ultima frase, entonces se elimina, asi la frase queda 'thirty days of pythoonn' nuevamente se vuelve a buscar el primer caracter 'n' en la primera frase 
# y al final de la frase resultante , entonces se elimina, asi la frase queda 'thirty days of pythoon' nuevamente se vuelve a buscar el primer caracter 'n' 
# en la primera frase y al final de la frase resultante, entonces se elimina, asi la frase queda 'thirty days of pythoo' nuevamente se vuelve a buscar el 
# primer caracter 'n' en la primera frase y al final de la frase resultante, si no se encuentra, se procede a buscar el segundo caracter 'o' en la primera frase 
# y al final de la frase resultante entonces se elimina, asi la frase queda 'thirty days of pytho', nuevamente se vuelve a buscar el primer caracter 'n' en 
# la primera frase y al final de la frase resultante si no se encuentra, se procede a buscar el segundo caracter 'o' en la primera frase y al final de la 
# frase resultante, entonces se elimina, asi la frase queda 'thirty days of pyth', nuevamente se vuelve a buscar el primer caracter 'n' en la primera frase 
# y al final de la frase resultante si no se encuentra, se procede a buscar el segundo caracter 'o' en la primera frase y al final de la frase resultante
# si no se encuentra, se procede a buscar el tercer caracter 't'en la primera frase y al final de la frase resultante entonces se elimina, asi la frase queda
# 'hirty days of pyth' nuevamente se vuelve a buscar el primer caracter 'n' en la primera frase y al final de la frase resultante si no se encuentra, se procede
# a buscar el segundo caracter 'o' en la primera frase y al final de la frase resultante, si no se encuentra, se procede a buscar el tercer caracter 't'en la 
# primera frase y al final de la frase resultante, si no se encuentra, se procede a buscar el cuarto caracter 'h'en la primera frase y al final de la frase resultante,
# entonces se eliminan, asi la frase queda 'irty days of pyt'. De manera analoga se procede nuevamente a buscar el primer caracter en la primera frase y al final 
# de la frase resultante y asi sucecivamente hasta que llegamos a la frase final 'irty days of py' 

# replace(): Sustituye el substring por un string dado
challenge = 'thirty days of python'
print(challenge.replace('python', 'coding')) # thirty days of coding

# split(): Se utiliza para dividir una cadena (string) en una lista de subcadenas (substring) o palabras, con un argumento específico
challenge = 'thirty days of python'
print(challenge.split()) # ['thirty', 'days', 'of', 'python']  (Divide la cadena, sin argumento, por defecto utiliza espacio entre palabras)
challenge = 'thirty, days, of, python'
print(challenge.split()) # ['thirty,', 'days,', 'of,', 'python']
challenge = 'thirty, days, of, python'
print(challenge.split(', ')) # ['thirty', 'days', 'of', 'python'] (Con argumento, a cada palabra o subconjunto de la cadena elimine ",")

# swapcase(): Convierte todos los caracteres de mayúsculas a minúsculas y viceversa.
challenge = 'thirty days of python'
print(challenge.swapcase())   # THIRTY DAYS OF PYTHON
challenge = 'Thirty Days Of Python'
print(challenge.swapcase())  # tHIRTY dAYS oF pYTHON

# startswith(): Comprueba si el string empieza por el string especificado
challenge = 'thirty days of python'
print(challenge.startswith('thirty')) # True
print(challenge.startswith('python')) # False

# endtswith(): Comprueba si el string termina por el string especificado
challenge = 'thirty days of python'
print(challenge.endswith('thirty')) # False
print(challenge.endswith('python')) # True



# Ejercicios

# Crea un acrónimo o una abreviatura para el nombre "Python para todos".
nombre = "Python para todos"
acronimo = "".join([word[0] for word in nombre.split()]).upper()
print(acronimo) # PPT
# Primero dividimos la cadena original en palabras usando el método split(). Luego, usamos una comprensión de lista para obtener la primera letra de 
# cada palabra y las unimos en una sola cadena usando el método join(). Finalmente, convertimos la cadena resultante en mayúsculas usando el método 
# upper(). El acrónimo resultante es "PPT".

# Extraer la frase "porque porque porque" en la siguiente frase: "No se puede terminar una frase con porque porque porque es una conjunción".
sentence = "No se puede terminar una frase con porque porque porque es una conjunción"
prase = sentence[35:55]
print(prase) # porque porque porque

# "Coding for all      ", elimina los espacios del string dado.
string = "Coding for all      "
result = string.strip()
print(result) # Coding for all

# La siguiente lista contiene los nombres de algunas librerías python: ['Django', 'Flask', 'Bottle', 'Pyramid', 'Falcon']. Une la lista con una cadena 
# hash (#) con espacio
libraries = ['Django', 'Flask', 'Bottle', 'Pyramid', 'Falcon']
result = '# '.join(libraries)
print(result) # Django# Flask# Bottle# Pyramid# Falcon

# Utilice la secuencia de escape de línea nueva para separar las frases siguientes.
# Estoy disfrutando con este reto.
# Me pregunto qué será lo siguiente.
print('Me pregunto qué será lo siguiente.\nI just wonder what is next.') # Me pregunto qué será lo siguiente.
                                                                         # I just wonder what is next.
                                                                         
# Utilice una secuencia de escape de tabulación para escribir las siguientes líneas.
#Name            Age     Country         City
#Alvaro          55      Colombia        Armenia
print('Name    \tAge\tCountry \tCity\nAlvaro  \t55\tColombia\tArmenia')                                                                      