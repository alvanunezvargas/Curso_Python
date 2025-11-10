# Operadores arimeticos
print(2 + 3)             # 5 (Suma (+))
print(3 - 1)             # 2 (Resta (-))
print(2 * 3)             # 6 (Multiplicacion (*))
print(3 / 2)             # 1.5 (Division (/))
print(3 ** 2)            # 9 (Numero elevado al exponente (**))
print(10 % 3)            # 1 (Residuo de la division (Modulus) (%))
print(10 // 3)           # 3 (Cociente de la division (Floor divition), en numero entero, sin decimal (//))

# Calcule el area de un circulo
radius = 10                           # radius of a circle
area_of_circle = 3.14 * radius ** 2   # two * sign means exponent or power
print('Area of a circle:', area_of_circle)  # Area of a circle: 314.0

# Calcule area de rectangulo
length = 10
width = 20
area_of_rectangle = length * width
print('Area of rectangle:', area_of_rectangle)  # Area of rectangle: 200

# Operadores de comparacion
print(3 > 2)     # True (porque 3 es mayor que 2)
print(3 >= 2)    # True (porque 3 es mayor que 2)
print(3 < 2)     # False  (porque 3 es mayor que 2)
print(2 < 3)     # True (porque 2 es menor que 3)
print(2 <= 3)    # True (porque 2 es menor que 3)
print(3 == 2)    # False (porque 3 no es igual a 2 (==: Es igual a))
print(3 != 2)    # True (porque 3 no es igual a 2  (=!: No es igua a))
print(len('mango') == len('avocado'))  # False
print(len('mango') != len('avocado'))  # True
print(len('mango') < len('avocado'))   # True
print(len('milk') != len('meat'))      # False
print(len('milk') == len('meat'))      # True
print(len('tomato') == len('potato'))  # True
print(len('python') > len('dragon'))   # False

# Comparaciuones para determinar si es falso o verdadero
print('True == True: ', True == True)  # True == True:  True
print('True == False: ', True == False)  # True == False:  False
print('False == False:', False == False)  # True == False:  False

# Además del operador de comparación anterior, Python usa:
# - is: Devuelve verdadero si ambas variables son el mismo objeto (x is y)
# - is not: Devuelve verdadero si ambas variables no son el mismo objeto (x is not y)
# - in: Devuelve True si la lista consultada contiene un determinado elemento (x in y)
# - not in: Devuelve True si la lista consultada no tiene un determinado elemento (x in y)
print('1 is 1', 1 is 1)                   # 1 is 1 True
print('1 is not 2', 1 is not 2)           # 1 is not 2 True
print('A in Asabeneh', 'A' in 'Asabeneh') # A in Asabeneh True
print('B in Asabeneh', 'B' in 'Asabeneh') # B in Asabeneh False
print('coding' in 'coding for all') # True
print('a in an:', 'a' in 'an')      # a in an: True
print('4 is 2 ** 2:', 4 is 2 ** 2)   # 4 is 2 ** 2: True

# Operadores logicos
# Python usa palabras clave and, or y not para operadores lógicos. Los operadores lógicos se utilizan para combinar sentencias condicionales:
print(3 > 2 and 4 > 3) # True (porque ambas afirmaciones son verdaderas)
print(3 > 2 and 4 < 3) # False (porque la segunda afirmación es falsa)
print(3 < 2 and 4 < 3) # False (porque ambas afirmaciones son falsas)
print('True and True: ', True and True) # True and True:  True
print(3 > 2 or 4 > 3)  # True - porque ambas afirmaciones son ciertas
print(3 > 2 or 4 < 3)  # True - porque una de las afirmaciones es cierta
print(3 < 2 or 4 < 3)  # False - porque ambas afirmaciones son falsas
print('True or False:', True or False)  # True or False: True
print(not 3 > 2)     # False (porque 3 > 2 es verdadero, entonces no verdadero da falso)
print(not True)      # False (Negación, el operador not convierte verdadero en falso)
print(not False)     # True
print(not not True)  # True
print(not not False) # False

# Escriba un script que solicite al usuario que ingrese la base y la altura del triángulo y calcule el área de este triángulo (área = 0,5 x b x h).
base=float(input("enter the base of the triangle: "))  # enter tha base of the triangle:   (usted debe ingresar la base del trinagulo)
height=float(input("enter the height of the triangle: ")) # enter the height of the triangle: (usted debe ingresar la altura del trinagulo)
area=0.5*base*height
print("The area of the triangle is:" , area)  # The area of the triangle is: xx

# Calcule la pendiente, la intersección x y la intersección y de y = 2x -2
equation="y=2x-2"
slope=2
x_intercept=-slope/2
y_intercept=-2
print("Equation:", equation)  # Equation: y=2x-2
print("Slope:", slope)  # Slope: 2
print("X-intercept:", x_intercept)  # X-intercept: -1.0
print("Y-intercept:",y_intercept)  # Y-intercept: -2

# La pendiente es (m = y2-y1/x2-x1). Encuentre la pendiente y la distancia euclidiana entre el punto (2, 2) y el punto (6,10) en Python
x1, y1= 2, 2
x2, y2=6, 10
slope=(y2-y1)/(x2-x1)
distance=((x2-x1)**2+(y2-y1)**2)**0.5
print("Slope:", slope) # Slope: 2.0
print("Distance:", distance) # Distance: 8.94427190999916

# Encuentre las longitudes de las palabras 'python' y 'dragon' y haga una declaración de comparación falsa.
word1="python"
word2="dragon"
lenght1=len(word1)
lenght2=len(word2)
print(not lenght1==lenght2)  # False
if lenght1==lenght2:
    print("The lengths are equal.")
else:
    print("The lengths are not equal.")  # The lengths are equal.
    
# Use un operador para verificar si 'on' se encuentra tanto en 'python' como en 'dragon'
print('on in python:', 'on' in 'python') # on in python: True
print('on' in 'python') # True

# No hay 'on' tanto en dragon como en python
print( 'on' in 'dragon' and "on" in "python")  # True
print(not('on' in 'dragon' and "on" in "python"))  # False

world1="python"
world2="dragon"
if "on" not in world1 and "on" not in world2:  #(se utiliza el operador "not in" para verificar si la palabra "on"esta contenida..)
    print("'on' is not in both words.")
else:
    print("'on' is in one of both words.")  # 'on' is in one of both words.