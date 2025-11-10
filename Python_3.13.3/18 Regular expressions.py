"""
Expresiones regulares
Una expresión regular o RegEx es una cadena de texto especial que ayuda a encontrar patrones en los datos. Una RegEx se puede utilizar para comprobar
si algún patrón existe en un tipo de datos diferente. Para usar RegEx en python primero debemos importar el módulo RegEx que se llama re.

El módulo re
Después de importar el módulo podemos utilizarlo para detectar o encontrar patrones.
"""
import re

"""
Métodos en el módulo re
Para encontrar un patrón utilizamos diferentes conjuntos de caracteres re que permiten buscar una coincidencia en una cadena.

re.match(): busca sólo al principio de la primera línea de la cadena y devuelve los objetos coincidentes si los encuentra, en caso contrario devuelve None.
re.search: Devuelve un objeto coincidente si lo hay en cualquier parte de la cadena, incluidas las cadenas multilínea.
re.findall: Devuelve una lista con todas las coincidencias
re.split: Toma una cadena, la divide en los puntos de coincidencia y devuelve una lista
re.sub: Sustituye una o varias coincidencias dentro de una cadena

Match:
sintaxis
re.match(substring, string, re.I)
substring es una cadena o un patrón, string es el texto que buscamos como patrón , re.I ignora mayúsculas y minúsculas
"""

import re

txt = 'I love to teach python and javaScript'
match = re.match('I love to teach', txt, re.I)   # (re.match() busca una coincidencia desde el principio de la cadena txt con el patrón proporcionado
# 'I love to teach'. re.I indica que la coincidencia es indiferente a mayúsculas y minúsculas, lo que significa que coincidirá con "I love to teach"
# sin importar si las letras están en mayúsculas o minúsculas. El resultado de re.match() se almacena en la variable match.)
print(match)  # <re.Match object; span=(0, 15), match='I love to teach'> (imprime el resultado de la coincidencia)
span = match.span()  # (match.span() devuelve una tupla con las posiciones de inicio y finalización de la coincidencia encontrada.)
print(span)     # (0, 15)
start, end = span  # (Las posiciones de inicio y finalización se desempaquetan en las variables start y end.)
print(start, end)  # 0, 15
substring = txt[start:end]  # (Usando las posiciones de inicio y finalización obtenidas de la coincidencia, se extrae la subcadena coincidente de la cadena txt)
print(substring)       # I love to teach

"""
Como puede ver en el ejemplo anterior, el patrón que buscamos (o la subcadena que buscamos) es "I love to teach". La función match devuelve un objeto
sólo si el texto empieza por el patrón.
"""

import re

txt = 'I love to teach python and javaScript'
match = re.match('I like to teach', txt, re.I)
print(match)  # None

# La cadena no concuerda con "I like to teach", por lo tanto no hubo coincidencia y el método match devolvió None.

"""
Search
re.search(substring, string, re.I)
Buscar un patrón (substring) en una cadena de texto (string) con la opción de coincidencia insensible a mayúsculas y minúsculas (re.I)
"""

import re

txt = "Python is the most beautiful language that a human being has ever created. I recommend python for a first programming language"
match = re.search('first', txt, re.I)  # (re.search() busca la palabra "first" en la cadena txt con la opción de coincidencia indiferente a
# mayúsculas y minúsculas (re.I))
print(match)  # <re.Match object; span=(100, 105), match='first'>
span = match.span() # (Podemos obtener la posición inicial y final de la coincidencia como tupla usando span)
print(span)     # (100, 105)
start, end = span  # (Busquemos la posición inicial y final a partir del span)
print(start, end)  # 100 105
substring = txt[start:end]
print(substring)       # first

"""
Como puede ver, "search" es mucho mejor que "match" porque puede buscar el patrón en todo el texto. Search devuelve un objeto match con
la primera coincidencia encontrada, de lo contrario devuelve "None". Una función re mucho mejor es findall. Esta función busca el patrón a través de
toda la cadena y devuelve todas las coincidencias como una lista.
"""

# Buscar todas las coincidencias con findall: findall() devuelve todas las coincidencias en forma de lista.

import re

txt = "Python is the most beautiful language that a human being has ever created. I recommend python for a first programming language"
matches = re.findall('language', txt, re.I)  # (re.findall() busca todas las ocurrencias de la palabra "language" en la cadena txt con la opción
# de coincidencia indiferente a mayúsculas y minúsculas (re.I))
print(matches)  # ['language', 'language']

# Otro ejemplo:
    
import re

txt = "Python is the most beautiful language that a human being has ever created. I recommend python for a first programming language"
matches = re.findall('python', txt, re.I)  
print(matches)  # ['Python', 'python']

# Dado que utilizamos re.I, se incluyen tanto las minúsculas como las mayúsculas. Si no tenemos la bandera re.I, tendremos que escribir nuestro
# patrón de otra manera. Comprobémos:

import re

txt = '''Python is the most beautiful language that a human being has ever created.
I recommend python for a first programming language'''
matches = re.findall('Python|python', txt)
print(matches) # ['Python', 'python']

# Otra forma seria:

import re

txt = "Python is the most beautiful language that a human being has ever created. I recommend python for a first programming language"
matches = re.findall('[Pp]ython', txt, re.I)
print(matches)   # ['Python', 'python']

# Sustitución de una subcadena (substring): Cambiar Python y python por JavaScript

import re

txt = '''Python is the most beautiful language that a human being has ever created.
I recommend python for a first programming language'''
match_replaced = re.sub('Python|python', 'JavaScript', txt, re.I)
print(match_replaced) # JavaScript is the most beautiful language that a human being has ever created. I recommend JavaScript for a first
# programming language

# Otra forma es:

import re

txt = '''Python is the most beautiful language that a human being has ever created.
I recommend python for a first programming language'''
match_replaced = re.sub('[Pp]ython', 'JavaScript', txt, re.I)
print(match_replaced) # JavaScript is the most beautiful language that a human being has ever created. I recommend JavaScript for a first
# programming language

# Añadamos un ejemplo más. La siguiente cadena es realmente difícil de leer a menos que eliminemos el símbolo %. Sustituir el % por una
# cadena vacía limpiará el texto.

import re

txt = '''%I a%m te%%a%%che%r% a%n%d %% I l%o%ve te%ach%ing. 
T%he%re i%s n%o%th%ing as r%ewarding a%s e%duc%at%i%ng a%n%d e%m%p%ow%er%ing p%e%o%ple.
I fo%und te%a%ching m%ore i%n%t%er%%es%ting t%h%an any other %jobs. 
D%o%es thi%s m%ot%iv%a%te %y%o%u to b%e a t%e%a%cher?'''
matches = re.sub('%', '', txt)
print(matches)
"""
I am teacher and  I love teaching. 
There is nothing as rewarding as educating and empowering people.
I found teaching more interesting than any other jobs. 
Does this motivate you to be a teacher?
"""

# Dividir texto utilizando RegEx Split:

import re

txt = '''I am teacher and  I love teaching.
There is nothing as rewarding as educating and empowering people.
I found teaching more interesting than any other jobs.
Does this motivate you to be a teacher?'''
print(re.split('\n', txt))
"""
['I am teacher and  I love teaching.', 'There is nothing as rewarding as educating and empowering people.',
'I found teaching more interesting than any other jobs.', 'Does this motivate you to be a teacher?']
"""

# Utilizamos re.split('\n', txt) para dividir la cadena de texto txt en una lista de subcadenas. La función re.split() toma dos argumentos:
# El primer argumento es el patrón de expresión regular, que en este caso es '\n'. Esto significa que estamos dividiendo el texto en cada salto
# de línea encontrado en la cadena.
# El segundo argumento es la cadena de texto "txt" en la que queremos realizar la división.

# El resultado de re.split('\n', txt) es una lista que contiene las líneas individuales del texto. Cada elemento de la lista es una línea del texto original.

# Escribir patrones RegEx: Para declarar una variable de cadena utilizamos una comilla simple o doble. Para declarar la variable RegEx r''. El
# siguiente patrón sólo identifica apple con minúsculas, para hacerlo indiferente a mayúsculas y minúsculas deberíamos reescribir nuestro patrón o
# añadir una bandera.

# Escribir patrones RegEx: Para declarar una variable de cadena utilizamos una comilla simple o doble. Para declarar la variable RegEx r''.
# El siguiente patrón sólo identifica apple con minúsculas, para hacerlo indiferente a mayúsculas y minúsculas deberíamos reescribir nuestro patrón
# o añadir una bandera.

import re

regex_pattern = r'apple'
txt = 'Apple and banana are fruits. An old cliche says an apple a day a doctor way has been replaced by a banana a day keeps the doctor far far away. '
matches = re.findall(regex_pattern, txt)
print(matches)  # ['apple']

# Definimos una expresión regular llamada regex_pattern. Esta expresión regular busca la secuencia de caracteres "apple" de forma literal.
# La r antes de la cadena indica una cadena "raw" o sin procesar, lo que significa que los caracteres de escape se interpretan literalmente.
# Definimos una cadena de texto llamada txt. Esta cadena contiene varias ocurrencias de la palabra "apple", una comienzan con una "A" mayúscula
# y otra con una "a" minúscula.
# Utilizamos re.findall(regex_pattern, txt) para buscar todas las ocurrencias del patrón "apple" (sin importar las mayúsculas y minúsculas) en la
# cadena txt. La función re.findall() toma dos argumentos:
# El primer argumento es el patrón de expresión regular, que en este caso es regex_pattern.
# El segundo argumento es la cadena de texto en la que queremos buscar las coincidencias, que es txt en este caso.
# La función re.findall() busca todas las ocurrencias del patrón en la cadena y devuelve una lista con todas las coincidencias encontradas.
# print(matches) : Finalmente, se imprime la lista resultante: ['apple']

# Para hacerlo indiferente a mayúsculas y minúsculas añadimos la bandera re.I.

import re

regex_pattern = r'apple'
txt = 'Apple and banana are fruits. An old cliche says an apple a day a doctor way has been replaced by a banana a day keeps the doctor far far away. '
matches = re.findall(regex_pattern, txt, re.I)
print(matches)  # ['Apple', 'apple']

# O podemos utilizar un método de conjunto de caracteres [Aa]pple:

import re

regex_pattern = r'[Aa]pple'
txt = 'Apple and banana are fruits. An old cliche says an apple a day a doctor way has been replaced by a banana a day keeps the doctor far far away. '
matches = re.findall(regex_pattern, txt)
print(matches)  # ['Apple', 'apple']

# El metodo de conjunto de caracteres lo podriamos definir de las siguiente maneras:
# [a-c] significa, a o b o c
# [a-z] significa, cualquier minuscula de la a a la z
# [A-Z] significa, cualquier mayuscula de la A a la Z
# [0-3] significa, 0 o 1 o 2 o 3
# [0-9] significa cualquier número del 0 al 9
# [A-Za-z0-9] cualquier carácter, es decir, de la a a la z, de la A a la Z o del 0 al 9

# \: se utiliza para capturar caracteres especiales:

import re

cadena = r'Este es un ejemplo: Python es genial.'
coincidencias = re.findall(r':', cadena)
print(coincidencias)  # [':']

# \d significa: coincide cuando la cadena contiene dígitos (números del 0 al 9)

import re

cadena = r'123 abc 456'
digitos = re.findall(r'\d', cadena)
print(digitos)  # ['1', '2', '3', '4', '5', '6']

# o

import re

txt = 'This regular expression example was made on December 6,  2019 and revised on July 8, 2021'
regex_pattern = r'\d{4}'  # ("\d{4}" Extrae los digidos de 4 numeros)
matches = re.findall(regex_pattern, txt)
print(matches)  # ['2019', '2021']

# o

import re

txt = 'This regular expression example was made on December 6,  2019 and revised on July 8, 2021'
regex_pattern = r'\d{1,4}'   # ("\d{1,4}" Extrae los digidos de 1 a 4 numeros)
matches = re.findall(regex_pattern, txt)
print(matches)  # ['6', '2019', '8', '2021']

# 0

import re

regex_pattern = r'\d+'  # ("\d+" Son carácteres de uno o mas dijitos)
txt = 'This regular expression example was made on December 6,  2019 and revised on July 8, 2021'
matches = re.findall(regex_pattern, txt)
print(matches)  # ['6', '2019', '8', '2021']

# \D significa: coincidencia cuando la cadena no contiene dígitos

import re

cadena = r'123 abc 456'
digitos = re.findall(r'\D', cadena)
print(digitos)  # [' ', 'a', 'b', 'c', ' ']

#  Conceptos básicos de expresiones regulares

# 1) Cualquier carácter excepto nueva línea, Expresión Regular: '.'

import re

texto = "abcedario and axc"
coincidencias = re.findall("a.c", texto) # (La expresión regular "a.c" significa que cualquier cadena que comience con la letra a,
# seguida de cualquier carácter y despues seguida de la letra c.)"""
print(coincidencias)  # ['abc', 'axc']

# 2) El carácter 'a': Expresión Regular: "a"

import re

texto = "Apple and banana are fruits."
coincidencias = re.findall("a", texto) # (La expresión regular "a" significa que cualquier cadena que contenga la letra a  )
print(coincidencias) # ['a', 'a', 'a', 'a', 'a']

# 3) La cadena 'ab': Expresión Regular: "ab"

import re

texto = "abcde"
coincidencias = re.findall("ab", texto) # (La expresión regular "ab" significa que cualquier cadena que contenga las letras
# "a" y "b" seguidas.)
print(coincidencias)  # ['ab']

# 4) 0 (vacio) o más a's: Expresión Regular: "a*"

import re

texto = "a abc aa aab"
coincidencias = re.findall("a*", texto) # (La expresión regular "a*" significa que cualquier cadena que contenga cero (vacio)
# o más caracteres a.)
print(coincidencias)  # ['a', '', 'a', '', '', '', 'aa', '', 'aa', '', '']

# 5) Escapa un carácter especial: Expresión Regular: "\"

import re

texto = "Hello, world. This is a test."
coincidencias = re.findall("\.", texto) #(La expresión regular "\." significa encontrar todas las apariciones del carácter "."
# en el texto.)
print(coincidencias)  # ['.', '.']

# Cuantificadores de expresiones regulares

# 1) 0 (vacio) o más: Símbolo: "*"

import re

texto = "bb bcd bbc bbbc"
coincidencias = re.findall("b*", texto) # (La expresión regular "b*" significa cualquier cadena que contenga cero (vacio)
# o más caracteres a.)
print(coincidencias)  # ['bb', '', 'b', '', '', '', 'bb', '', '', 'bbb', '', '']

# 2) 1 o más: Símbolo: "+"

import re

texto = "bb bcd acde bbbc"
coincidencias = re.findall("b+", texto) # (La expresión regular "b+" significa encontrar todas las apariciones
# de la expresión regular "b" en el texto. 
print(coincidencias)  # ['bb', 'b', 'bbb']

# 3) 0 o 1: Símbolo: "?"

import re

texto = "color colour coloor"
coincidencias = re.findall("colou?r", texto) # (La expresión regular "colou?r" significa cualquier cadena que contenga los caracteres
# "colo" y "colou" seguidos de carácter obligatorio "r". El signo de interrogación "?" a la derecha del carácter "u" significa
# que "u" puede estar presente o no en la cadena)
print(coincidencias)  # ['color', 'colour']

# 4) Exactamente 2: Símbolo: "{2}"

import re

texto = "aa abc aac aabc"
coincidencias = re.findall("a{2}", texto) # (La expresión regular "a{2}" cualquier cadena que contenga dos caracteres "a")
print(coincidencias)  # ['aa', 'aa', 'aa']

# 5) Entre 2 y 5: Símbolo: "{2,5}"

import re

texto = "aa abc aac aabc aaaa aaaaa"
coincidencias = re.findall("a{2,5}", texto) # (La expresión regular "a{2,5}" cualquier cadena que contenga entre dos y cinco caracteres "a")
print(coincidencias)  # ['aa', 'aa', 'aa', 'aaaa', 'aaaaa']

# 6) 2 o más: Símbolo: {2,}

import re

texto = "aa abc aac aaaaaa"
coincidencias = re.findall("a{2,}", texto) # (La expresión regular "a{2,}" cualquier cadena que contenga entre dos y mas caracteres "a")
print(coincidencias)  # ['aa', 'aa', 'aaaaaa']

# Clases de caracteres de expresiones regulares

# 1) Un carácter de: a, b, c, d: Sintaxis: [ab-d]

import re

texto = "abcd1234efgh"
coincidencias = re.findall(r'[ab-d]', texto) # (La expresión regular es "[ab-d]" que significa coincidir con cualquier carácter que sea
# "a", "b", "c" o "d".)
print(coincidencias)  # ['a', 'b', 'c', 'd']

# o

import re

regex_pattern = r'[a].'  # (Este corchete significa a y . significa cualquier carácter excepto nueva línea)
txt = '''Apple and banana are fruits'''
matches = re.findall(regex_pattern, txt)
print(matches)  # ['an', 'an', 'an', 'a ', 'ar']

# o

import re

regex_pattern = r'[a].+'  # (Esta expresión regular coincide con cualquier cadena que comience con la letra a y + luego los demas 
# caracteres si los hay.)
txt = '''Apple and banana are fruits'''
matches = re.findall(regex_pattern, txt)
print(matches)  # ['and banana are fruits']

# o

import re

txt = '''I am not sure if there is a convention how to write the word e-mail.
Some people write it as email others may write it as Email or E-mail.'''
regex_pattern = r'[Ee]-?mail'  # (Esta expresión regular coincide con cualquier cadena que empiece por la letra E o e, seguida de
# un guión opcional, seguida de la palabra "mail". El carácter ? en la expresión regular significa que el guión es opcional.)
matches = re.findall(regex_pattern, txt)
print(matches)  # ['e-mail', 'email', 'Email', 'E-mail']

# 2) Un carácter excepto a, b, c, d: Sintaxis: "[^abcd]"

import re

texto = "abcd1234efgh"
coincidencias = re.findall(r'[^ab-d]', texto) # (La expresión regular es "[^ab-d]" que significa no coincida con loa carácteres
# "a", "b", "c" o "d".)
print(coincidencias)  # ['1', '2', '3', '4', 'e', 'f', 'g', 'h']

# 3) Carácter de retroceso: Sintaxis: "\\"

import re

texto = "Este es un ejemplo de backslash: \\"
coincidencias = re.findall(r'\\', texto) # (La expresión regular es '\\', que significa coincidir con "\\")
print(coincidencias)  # ['\\']

# 4) Un dígito: Sintaxis: "\d"

import re

texto = "123abc456"
coincidencias = re.findall(r'\d', texto) # (La expresión regular es '\d', que significa coincidir con digitos numericos)
print(coincidencias)  # ['1', '2', '3', '4', '5', '6']

# 5) Un no dígito: Sintaxis: "\D"

import re

texto = "123abc456"
coincidencias = re.findall(r'\D', texto) # (La expresión regular es '\D', que significa NO coincidir con digitos numericos)
print(coincidencias)  # ['a', 'b', 'c']

# 6) Un espacio en blanco: Sintaxis: "\s"

import re

texto = "Texto con espacios   y tabulaciones\t.\nSalto de linea"
coincidencias = re.findall(r'\s', texto) # (La expresión regular es "\s" que significa coincidir con cualquier espacio en blanco
# tabulaciones y saltos de línea)
print(coincidencias)  # [' ', ' ', ' ', ' ', ' ', ' ', '\t', '\n', ' ', ' ']

# 7) Un no espacio en blanco: Sintaxis: "\S"

import re

texto = "Texto con espacios   y tabulaciones\t.\nSalto de linea"
coincidencias = re.findall(r'\S', texto) # (La expresión regular es "\S" que significa NO coincidir con ningun espacio en blanco)
print(coincidencias) 
# ['T', 'e', 'x', 't', 'o', 'c', 'o', 'n', 'e', 's', 'p', 'a', 'c', 'i', 'o', 's', 'y', 't', 'a', 'b', 'u', 'l', 'a', 'c', 
# 'i', 'o', 'n', 'e', 's', '.', 'S', 'a', 'l', 't', 'o', 'd', 'e', 'l', 'i', 'n', 'e', 'a']

# 8) Un carácter de palabra: Sintaxis: "\w"

import re

texto = "Python es genial! abc1234."
coincidencias = re.findall(r'\w', texto) # (La expresión regular es "\w" que significa coincidir con cualquier carácter
# alfanumérico)
print(coincidencias)  # ['P', 'y', 't', 'h', 'o', 'n', 'e', 's', 'g', 'e', 'n', 'i', 'a', 'l', 'a', 'b', 'c', '1', '2', '3', '4']

# 9) Un carácter no alfanumérico: Sintaxis: "\W"

import re

texto = "Python es&genial!%"
coincidencias = re.findall(r'\W', texto) # (La expresión regular es "\W" que significa NO coincidir con carácter
# alfanumérico)
print(coincidencias)  # [' ', '&', '!', '%']


# Grupos de expresiones regulares

# 1) Captura de grupos: Sintaxis: "(expresión)"

import re

texto = "Mi número de teléfono es 123-456-7890 y el otro es 987-654-3210."
coincidencias = re.findall(r'(\d{3}-\d{3}-\d{4})', texto) # (La expresión regular es "(\d{3}-\d{3}-\d{4})", que significa una
# cadena que comienza con tres dígitos, seguida de un guion, seguida de tres dígitos, seguida de un guion, seguida de cuatro dígitos.)
print(coincidencias)  # ['123-456-7890', '987-654-3210']

# o

import re

texto = "Mi número de teléfono es 123-456-7890 y el otro es 987-654-3210."
coincidencias = re.findall(r'(\d{3}-\d{3}-\d{4})', texto)
for telefono in coincidencias:
    print("Número de teléfono encontrado:", telefono)
"""
Número de teléfono encontrado: 123-456-7890
Número de teléfono encontrado: 987-654-3210
"""

# 2) Grupo sin captura: Sintaxis: "(?:expresión)"

import re

texto = "Visita mi sitio web en http://www.ejemplo.com y también https://www.ejemplo2.com."
coincidencias = re.findall(r'(?:https?://)(www\..+)', texto)
"""
"(?:https?://)" Esta parte del patrón busca una cadena que comience con "http://" o "https://"
"(?: ...)" ?: al principio indica que no queremos capturar esta parte en el resultado.
"(www\..+)" Esta parte busca una cadena que comience con "www." y capture el resto de la URL. El www\. busca la cadena
"www." literalmente, y .+ coincide con uno o más caracteres (cualquier carácter) después de "www.".
"""
print(coincidencias)  # ['www.ejemplo.com y también https://www.ejemplo2.com.']

# o

import re

texto = "Visita mi sitio web en http://www.ejemplo.com y también https://www.ejemplo2.com."
coincidencias = re.findall(r'(?:https?://)(www\..+)', texto)
for sitio_web in coincidencias:
    print("Sitio web encontrado:", sitio_web)
# Sitio web encontrado: www.ejemplo.com y también https://www.ejemplo2.com.

# 3) Coincide con el grupo capturado Y: Sintaxis: "\number"

import re

texto = "Mis cumpleaños son el 15-05-1990 y el 25-12-2020."
coincidencias = re.findall(r'(\d{2})-(\d{2})-(\d{4})', texto) # (La expresión regular es "(\d{2})-(\d{2})-(\d{4})"", que significa una
# cadena que comienza con dos dígitos, seguida de un guion, seguida de dos dígitos, seguida de un guion, seguida de cuatro dígitos.)
print(coincidencias)  # [('15', '05', '1990'), ('25', '12', '2020')]

# o

import re

texto = "Mis cumpleaños son el 15-05-1990 y el 25-12-2020."
coincidencias = re.findall(r'(\d{2})-(\d{2})-(\d{4})', texto)
for fecha in coincidencias:
    dia, mes, año = fecha
    print(f"Fecha válida encontrada: {dia}-{mes}-{año}")
"""
Fecha válida encontrada: 15-05-1990
Fecha válida encontrada: 25-12-2020
"""

# Aserciones de expresiones regulares

# 1) Inicio de cadena (^): Sintaxis: "^"

import re

txt = 'This regular expression example was made on December 6,  2019 and revised on July 8, 2021'
regex_pattern = r'^This'  # ()"^This" ^arranca con This. No admite this minuscula)
matches = re.findall(regex_pattern, txt)
print(matches)  # ['This']

# o

import re

texto = "Python es un lenguaje de programación."
coincidencia = re.search(r'^Python', texto) # (la función re.search() se utiliza para encontrar la primera aparición de la expresión
# regular "^Python" en la cadena de texto texto. 
if coincidencia:
    print("Coincidió al inicio de la cadena.")
else:
    print("No coincidió al inicio de la cadena.")
# Coincidió al inicio de la cadena.

# 0

import re

txt = 'This regular expression example was made on December 6,  2019 and revised on July 8, 2021'
regex_pattern = r'[^A-Za-z ]+'  # ()"[^A-Za-z ]+" ^Caracteres que no sean letras mayusculas (A-Z), ni minusculas (a-z), ni espacios y 
# + que sean caracteres seguidos, diferentes de los exceptuados.) 
matches = re.findall(regex_pattern, txt)
print(matches)  # ['6,', '2019', '8,', '2021']

# 2) Fin de cadena ($):Sintaxis: "$"

import re

texto = "Python es un lenguaje de programación."
coincidencia = re.search(r'programación.$', texto) # (la función re.search() se utiliza para encontrar la última aparición de la
# expresión regular "programación" en la cadena de texto)
if coincidencia:
    print("Coincidió al final de la cadena.")
else:
    print("No coincidió al final de la cadena.")
# Coincidió al final de la cadena.

# 3) Límite de palabra (\b): Sintaxis: "\b"

import re

texto = "Python es un lenguaje de programación."
coincidencias = re.findall(r'\bun\b', texto) # (La línea re.findall(r'\bun\b', texto) utiliza la función re.findall() para encontrar
# todas las apariciones del patrón de caracteres "un" en la cadena de texto.)
print(coincidencias) # ['un']

# 4) Sin límite de palabra (\B): Sintaxis: "\B"

import re

texto = "Python es xunx lenguaje de programación."
coincidencias = re.findall(r'\Bun\B', texto) # (La línea re.findall(r'\Bun\B', texto) utiliza la función re.findall() para encontrar
# todas las apariciones del patrón de caracteres "un" que no sea un limite de palabra.)
print(coincidencias)  # ['un']

# 5 Previsión positiva (?=...): Sintaxis: "(?=...)"

import re

texto = "Python 3.8 es una versión popular."
coincidencia = re.search(r'\d+(?=\.\d+)', texto)  # (La expresión regular r'\d+(?=\.\d+)' se descompone de la siguiente manera:
# \d+: Coincide con uno o más dígitos y (?=.\d+) es una "previsión positiva". Coincide con la posición en la cadena donde hay un
# punto "." seguido de uno o más dígitos \d+, pero no consume los caracteres. En otras palabras, verifica si después de los dígitos
# hay un punto decimal y más dígitos sin incluirlos en la coincidencia)
if coincidencia:
    print("Coincidió con la versión de Python:", coincidencia.group())
else:
    print("No se encontró ninguna versión.")
# Coincidió con la versión de Python: 3

# 6) Previsión negativa (?!...): Sintaxis: "(?!...)"

import re

texto = "Python 3.8 es una versión popular."
coincidencia = re.search(r'\d+(?!\.\d+)', texto) # (La expresión regular r'\d+(?!\.\d+)' se descompone de la siguiente manera: 
# "\d+" Coincide con uno o más dígitos (en este caso, busca números en la cadena). (?!\.\d+): Esta es una "previsión negativa".
# Coincide con la posición en la cadena donde NO hay un punto "." seguido de uno o más dígitos "\d+". En otras palabras, verifica
# si después de los dígitos NO hay un punto decimal y más dígitos.)
if coincidencia:
    print("Coincidió con una versión que no tiene punto decimal:", coincidencia.group())
else:
    print("No se encontró ninguna versión sin punto decimal.")
# Coincidió con una versión que no tiene punto decimal: 8

# Indicadores de expresiones regulares

# 1) Coincidencia global ("g"): Este indicador se usa para encontrar todas las coincidencias en lugar de detenerse después de la primera
# coincidencia.

import re

texto = "Hola, este es un ejemplo de texto con palabras repetidas. Ejemplo, ejemplo, ejemplo."
coincidencias = re.findall(r'ejemplo', texto, re.I) # (utiliza la expresión regular r'ejemplo' para encontrar todas las apariciones
# del patrón de caracteres ejemplo en la cadena de texto texto.)
print(coincidencias) # ['ejemplo', 'Ejemplo', 'ejemplo', 'ejemplo']

# 2) Ignorar mayúsculas y minúsculas ("i"): Al utilizar este indicador, la expresión regular considerará las letras mayúsculas y
# minúsculas como equivalentes. Por lo tanto, "A" coincidirá con "a".

import re

texto = "Python es genial, python también lo es."
coincidencias = re.findall(r'python', texto, re.I) # (utiliza la expresión regular r'python' para encontrar todas las apariciones
# del patrón de caracteres "python" y "re.I" para que sea indiferente las mayusculas y minisculas.)
print(coincidencias)  # ['Python', 'python']

# 3) ^ y $ coinciden con el inicio y el final de línea: Estos son anclajes utilizados para indicar que la coincidencia debe ocurrir
# al principio ("^"") o al final ("$"") de una línea de texto, respectivamente.

import re

texto = "Este es un texto, e\nque contiene varias líneas.\nEs importante. Es"
coincidencias_inicio = re.findall(r'^Es', texto, re.M)  # (La expresion regular "r'^Es" busca al principio de la linea la palabra "Es".
# El indicador "re.M" permite que "^" coincida con el principio de cada linea)
coincidencias_fin = re.findall(r'Es$', texto, re.M)  # (La expresion regular "r'Es$" busca al final de la linea la palabra "Es".
# El indicador "re.M" permite que "$" coincida con el final de cada linea)
print("Coincidencias al principio de línea:", coincidencias_inicio) # Coincidencias al principio de línea: ['Es', 'Es']
print("Coincidencias al final de línea:", coincidencias_fin) # Coincidencias al final de línea: ['Es']

# Caracteres especiales de expresiones regulares.

# 1) Nueva línea: "\n" representa un salto de línea en el texto.

import re

texto = "Este es un texto.\nCon dos líneas."
lineas = re.split('\n', texto) # (Utilizamos "re.split" y La expresion regular "\n", para imprimir cada linea)
print(lineas) # ['Este es un texto.', 'Con dos líneas.']

# o

import re

texto = "Este es un texto.\nCon dos líneas."
lineas = re.split('\n', texto)
for linea in lineas:
    print(linea)
"""
Este es un texto.
Con dos líneas.
"""

# 2) Retorno de carro: "\r" representa un retorno de carro en el texto.

import re

texto = "Hola,\r¿Cómo estás?"
texto_limpio = re.sub('\r', '', texto) # (Utilizando re.sub() para eliminar el retorno de carro "\r")
print(texto_limpio) # Hola,¿Cómo estás?

# 3) Tabulador: \t representa un tabulador en el texto.

import re

texto = "Nombre:\tJuan\tEdad:\t30"
campos = re.split('\t', texto) # (Utilizando re.split() para dividir el texto en campos)
print(campos) # ['Nombre:', 'Juan', 'Edad:', '30']

# o

import re

texto = "Nombre:\tJuan\nEdad:\t30"
campos = re.split('\t', texto) # (Utilizando re.split() para dividir el texto en campos)
for campo in campos:
    print(campo)
"""
Nombre:
Juan
Edad:
30
"""

# 4) Carácter nulo: "\0" representa el carácter nulo.

import re

texto = "Este es un\0texto con\0carácter nulo."
texto_limpio = re.sub('\0',"", texto) # (Utilizando re.sub() para eliminar el carácter nulo "\0")
print(texto_limpio) # Este es untexto concarácter nulo.

# o

import re

texto = "Este es un\0texto con\0carácter nulo."
texto_limpio = re.sub('\0'," ", texto) # (Utilizando re.sub() para eliminar el carácter nulo "\0")
print(texto_limpio) # Este es un texto con carácter nulo.

# 5) Carácter octal YY: "\o{YY}"" representa un carácter en notación octal. Debe reemplazar "{YY}"" por el valor octal deseado.

import re

texto = "Este es un ejemplo con el carácter \o101 (A en octal)."
caracteres_octal = re.findall(r'\\o(\d{3})', texto) # (Utiliza la función re.findall() para buscar todas las coincidencias en el texto
# que sigan el patrón especificado por la expresión regular. La expresión regular r'\\o(\d{3})' se desglosa de la siguiente manera:
# "\\" Esto coincide con el carácter de barra invertida "\" en la cadena de texto. Dado que "\" es un carácter de escape en las
# expresiones regulares, necesitas usar "\\"" para representar una barra invertida literal. "o" Esto coincide con el carácter
# literal "o" después de la barra invertida. "(\d{3})" Esto define un grupo de captura ( ... ) que coincide con tres dígitos "\d{3}"
# en notación octal)
for caracter in caracteres_octal: # (Luego, se realiza un bucle "for" para recorrer la lista de caracteres octales encontrados.)
    print("Carácter en octal:", chr(int(caracter, 8)))  # (Se utiliza chr(int(caracter, 8)) para convertir cada número octal en un
    # carácter. "int(caracter, 8)" convierte el número octal en un entero y "chr()" convierte el entero en el carácter correspondiente.)
# Respuesta: Carácter en octal: A

# 6) Carácter hexadecimal "\xYY" Representa un carácter ASCII en formato hexadecimal. "YY" debe ser reemplazado por el código hexadecimal
# de un carácter. Por ejemplo, \x41 representa el carácter 'A' en hexadecimal.

import re

texto = "Este es un ejemplo \x41scii"
coincidencias = re.findall(r'\x41', texto) # (Usar expresión regular para encontrar y coincidir con el carácter hexadecimal \xYY "\x41")
for coincidencia in coincidencias:
    print("Coincidencia encontrada:", coincidencia)  # Coincidencia encontrada: A

# 7) Carácter hexadecimal \uYYYY: Representa un carácter Unicode en formato hexadecimal. YYYY debe ser reemplazado por el código Unicode
# de un carácter. Por ejemplo, \u00A9 representa el símbolo de derechos de autor (©) en Unicode.

import re

texto = "Este es un ejemplo \u00A9 de Unicode" # (Usar expresión regular para encontrar y coincidir con el carácter Unicode \uYYYY
# "\u00A9)"
coincidencias = re.findall(r'\u00A9', texto)
for coincidencia in coincidencias:
    print("Coincidencia encontrada:", coincidencia)  # Coincidencia encontrada: ©


# 8) Carácter de control \cY: Representa un carácter de control en ASCII. Y debe ser reemplazado por un carácter que indique una
# función de control específica. Por ejemplo, \cC representa el carácter de control Ctrl+C, que se utiliza comúnmente para interrumpir
# procesos en sistemas Unix.

import re

texto = "Este es un ejemplo con el carácter \cG (Control-G)."  #  (La cadena contiene un carácter de control \cG que representa el
# Control-G.)
caracteres_control = re.findall(r'\\c([A-Za-z])', texto) # (Se utiliza re.findall() para buscar todas las ocurrencias de caracteres
# de control en la cadena texto. La expresión regular r'\\c([A-Za-z])' se utiliza para buscar un backslash (\) seguido de una letra
# en mayúscula o minúscula. El patrón \\c se usa para coincidir con el carácter de control \c en la cadena.)
for caracter in caracteres_control:
    control_char = ord(caracter) & 0x1F  # ("ord(caracter)"" convierte el caracter encontrado en su valor ordinal, que es un número
    # entero que representa su posición en la tabla de caracteres ASCII. "& 0x1F" se utiliza para realizar una operación bit a bit
    # con el valor ordinal. En este caso, se aplica una máscara para mantener solo los 5 bits más bajos del valor ordinal. Esto se
    # hace porque los caracteres de control están representados por los códigos ASCII del 0 al 31 (inclusive) y esos 5 bits son
    # suficientes para representarlos.)
    print(f"Carácter de control: {chr(control_char)} (Control-{caracter})") # ("chr(control_char)"" convierte el valor resultante de
    # nuevo en un carácter. "(Control-{caracter})"" se utiliza para mostrar una descripción del carácter de control que se ha
    # encontrado en la cadena original. Cómo funciona: "{caracter}" es una variable que contiene el carácter de control encontrado
    # en la cadena original. "(Control-{caracter})"" es una cadena de formato que se utiliza para mostrar el carácter de control
    # junto con la palabra "Control-" para proporcionar una descripción más legible.)
    
    # Respuesta: "Carácter de control:  (Control-G)"
    
# Sustitución de expresiones regulares

# 1) Inserta ($$): Este patrón se usa para insertar un signo de dólar ($) en el texto de sustitución.

import re

texto = "El precio del producto es $precio."
resultado = re.sub(r'\$precio', r'$$', texto) # (La función "re.sub()" utiliza la expresión regular para reemplazar el patrón de
# caracteres "\$precio" = "$precio" en la cadena de texto y lo reemplaza "r'$$'" por $$)
print(resultado) # El precio del producto es $$.

# 2) Insertar coincidencia completa ($&): Este patrón se utiliza para insertar la coincidencia completa (la cadena que coincide con la
# expresión regular) en el texto de sustitución.

import re

texto = "Hola, mundo."
resultado = re.sub(r'Hola', r'[$&]', texto)
print(resultado) # [$&], mundo.

# 3) Insertar cadena siguiente ($'): Este patrón se usa para insertar la parte del texto que sigue a la coincidencia en la cadena
# original en el texto de sustitución.

import re

texto = "El sol brilla, el cielo está despejado."
resultado = re.sub(r'brilla,', r'brilla, y $\'', texto)
print(resultado)  # El sol brilla, y $\' el cielo está despejado.

# 4) Insertar grupo capturado Y ($Y): Este patrón se utiliza para insertar un grupo capturado específico (identificado por su número
# de índice o nombre) en el texto de sustitución.

import re

texto = "Mi número de teléfono es (123) 456-7890."
resultado = re.sub(r'\((\d+)\) (\d+)-(\d+)', r'Número de área: \1, Número de teléfono: \2-\3', texto)
print(resultado) # Mi número de teléfono es Número de área: 123, Número de teléfono: 456-7890.


# Corchete cuadrado: Utilicemos corchetes para incluir busqueda de minúsculas y mayúsculas.

import re

regex_pattern = r'[Aa]pple' # (Este corchete significa A o a)
txt = 'Apple and banana are fruits. An old cliche says an apple a day a doctor way has been replaced by a banana a day keeps the doctor far far away.'
matches = re.findall(regex_pattern, txt)
print(matches)  # ['Apple', 'apple']

# Si queremos buscar el banano, escribimos el patrón de la siguiente manera:

import re

regex_pattern = r'[Aa]pple|[Bb]anana' # (Estoscorchetes significan A o a y B o b)
txt = 'Apple and banana are fruits. An old cliche says an apple a day a doctor way has been replaced by a banana a day keeps the doctor far far away.'
matches = re.findall(regex_pattern, txt)
print(matches)  # ['Apple', 'banana', 'apple', 'banana']


# Ejercicios

# ¿Cuál es la palabra más frecuente en el siguiente parrafo?

import re
from collections import Counter

parrafo = 'I love teaching. If you do not love teaching what else can you love. I love Python if you do not love something which can give you all the capabilities to develop an application what else can you love.'
palabras = re.findall(r'\b\w+\b', parrafo) # ('\b\w+\b' genera una lista con las palabras contenidas en parrafo)
frecuencia_palabras = Counter(palabras) # (Toma una lista de elementos y devuelve un diccionario que cuenta el 
# número de veces que aparece cada elemento en la lista.)
palabra_mas_frecuente, frecuencia = frecuencia_palabras.most_common(1)[0] # (`palabra_mas_frecuente, frecuencia`
# Es una asignación múltiple que desempaqueta la tupla seleccionada en dos variables separadas. `most_common(1)`:
# Es un método de la clase `Counter` que devuelve una lista de tuplas, donde cada tupla contiene una palabra y
# su frecuencia, ordenadas por frecuencia de mayor a menor. El argumento `1` indica que solo queremos la palabra
# más frecuente. `[0]`: Es un índice que selecciona la primera tupla de la lista devuelta por `most_common(1)`)
print(f'La palabra más frecuente es "{palabra_mas_frecuente}" con una frecuencia de {frecuencia} veces.')
# La palabra más frecuente es "love" con una frecuencia de 6 veces."

# Genere un codigo python que utilice expresiones regulares para extraer la distancia entre los dos puntos más alejados de
# la lista "puntos":  
# puntos = ['-12', '-4', '-3', '-1', '0', '4', '8']

import re

# Lista de puntos
puntos = ['-12', '-4', '-3', '-1', '0', '4', '8']
numero = re.findall(r'-?\d+', ' '.join(puntos)) # (`re.findall(r'-?\d+', ' '.join(puntos))`: Es una expresión regular que encuentra
# todos los números enteros (positivos y negativos) en la lista de puntos. La expresión regular `-?\d+` coincide con cualquier
# secuencia de uno o más dígitos (`\d+`) que pueden estar precedidos por un signo negativo opcional (`-?`). El método
# `re.findall()` devuelve una lista de todas las coincidencias encontradas en la lista de puntos. `' '.join(puntos)`: Es una
# función que convierte la lista de puntos en una cadena de texto separada por espacios. Esto es necesario para que la expresión
# regular pueda encontrar todos los números enteros en la lista de puntos.)
numeros = [int(numero) for numero in numero]
distancia = max(numeros) - min(numeros)
print("Distancia entre los puntos más alejados:", distancia)  # Distancia entre los puntos más alejados: 20

# Utilizando expresiones regulares, escriba un patrón que identifique si una cadena es una variable python válida

import re

def es_variable_valida(cadena):
    patron = r'^[a-zA-Z_]\w*$' # (r'^[a-zA-Z_]\w*$'`: Define un patrón de expresión regular que coincide con cualquier cadena
    # que comience con una letra o un guión bajo (`[a-zA-Z_]`) seguida más letras o de cero, números o guion bajo (`\w*`) y la 
    # coincidencia debe llegar hasta el final de la cadena '$'.)
    return bool(re.match(patron, cadena)) # (La función `re.match()` busca el patrón de expresión regular `patron` en la cadena
# `cadena`. Si la cadena coincide con el patrón, la función devuelve un objeto de coincidencia. Si la cadena no coincide con el
# patrón, la función devuelve `None`. La función `bool()` se utiliza para convertir el resultado de `re.match()` en un valor
# booleano. Si `re.match()` devuelve un objeto de coincidencia, `bool()` devuelve `True`. Si `re.match()` devuelve `None`, `bool()`
# devuelve `False`.)

# Ejemplos de uso
print(es_variable_valida('variable_1'))  # True
print(es_variable_valida('123_variable'))  # False
print(es_variable_valida('mi-variable'))  # False
print(es_variable_valida('_otra_variable'))  # True
print(es_variable_valida('123novalida'))  # False

# o

import re

def es_variable_valida(cadena):
    patron = r'^[a-zA-Z_][a-zA-Z0-9_]*$' # (`^[a-zA-Z_]`: Coincide con cualquier cadena que comience con una letra o un guión bajo
    # (`_`). `[a-zA-Z0-9_]*`: Coincide con cero o más letras, números o guiones bajos (`_`). `$`: Indica el final de la cadena.)
    return bool(re.match(patron, cadena))

print(es_variable_valida('variable_1'))  # True
print(es_variable_valida('123_variable'))  # False
print(es_variable_valida('mi-variable'))  # False
print(es_variable_valida('_otra_variable'))  # True
print(es_variable_valida('123novalida'))  # False

# Genere un script python que limpie la cadena 'sentence'. Después de la limpieza, cuente tres palabras más frecuentes en la
# cadena.

import re
from collections import Counter

sentence = '''%I $am@% a %tea@cher%, &and& I lo%#ve %tea@ching%;. No hay nada mas gratificante que educar y ayudar a la gente. Me parece que tomar el te es mas interesante que cualquier otro trabajo. ¿Te motiva esto a ser tea@cher?'''

cleaned_sentence = re.sub(r'[^a-zA-Z\s]', '', sentence) # ( `re.sub()` del módulo `re` de Python se utiliza para reemplazar todas
# las ocurrencias de una expresión regular en una cadena. '[^a-zA-Z\s]'` se utiliza para buscar todos los caracteres que no (^) son
# letras (`a-z` y `A-Z`) ni espacios en blanco (`\s`) en una cadena. '': Esto es lo que se utilizará para reemplazar las
# coincidencias encontradas. En este caso, se reemplazarán por una cadena vacía, lo que significa que se eliminarán del texto
# en la cadena 'sentence')

words = cleaned_sentence.split() # (Divide la cadena `cleaned_sentence` en una lista de palabras utilizando los espacios en blanco
# como separadores de palabras. La variable `words` es una lista que contiene todas las palabras de la cadena `cleaned_sentence`.)

word_counts = Counter(words) # (Toma una lista de elementos y devuelve un diccionario que cuenta el número de veces que aparece
# cada elemento en la lista.)

# Las tres palabras más comunes
most_common_words = word_counts.most_common(3) # (`most_common(3)`: El método `most_common()` se utiliza para obtener las palabras
# más comunes en el diccionario de frecuencia de palabras. El argumento `3` indica que se deben devolver las tres palabras más
# comunes.)

print("Las tres palabras más comunes en la cadena son:")
for word, count in most_common_words:
    print(f"'{word}': {count} veces")
"""
Las tres palabras más comunes en la cadena son:
'a': 3 veces
'que': 3 veces
'I': 2 veces
"""
# o

import re
from collections import Counter

sentence = '''%I $am@% a %tea@cher%, &and& I lo%#ve %tea@ching%;. No hay nada mas gratificante que educar y ayudar a la gente. Me parece que tomar el te es mas interesante que cualquier otro trabajo. ¿Te motiva esto a ser tea@cher?'''

matches = re.sub('[%$@&#!?;]', '', sentence)  # (Eliminar los caracteres especiales %$@&#!?;)

palabras = matches.split()  # (Convertir la cadena en una lista de palabras)
frecuencia_palabras = Counter(palabras)  # (Contar la frecuencia de cada palabra)
palabras_mas_frecuentes = frecuencia_palabras.most_common(3)  # (Obtener las tres palabras más frecuentes)

print(f'Las tres palabras más frecuentes son:')
for palabra, frecuencia in palabras_mas_frecuentes:
    print(f'"{palabra}" aparece {frecuencia} veces.')
"""
Las tres palabras más frecuentes son:
"a" aparece 3 veces.
"que" aparece 3 veces.
"I" aparece 2 veces.
"""