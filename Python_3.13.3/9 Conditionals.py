# Por defecto, las sentencias en un script Python se ejecutan secuencialmente de arriba a abajo. Si la lógica de procesamiento lo requiere, el flujo
# secuencial de ejecución puede alterarse de dos maneras:

# Ejecución condicional: un bloque de una o más sentencias se ejecutará si una determinada expresión es verdadera

# Ejecución repetitiva: un bloque de una o más sentencias se ejecutará repetidamente mientras cierta expresión sea verdadera. En esta sección,
# cubriremos las sentencias if, else, elif. Los operadores lógicos y de comparación que aprendimos en secciones anteriores serán útiles aquí.

# Condición If: En python y otros lenguajes de programación la palabra clave if se utiliza para comprobar si una condición es verdadera y ejecutar
# el código del bloque. Recuerde la sangría después de los dos puntos.
a = 3
if a > 0:
    print('A es un número positivo')  # A es un número positivo
# Como puede ver en el ejemplo anterior, 3 es mayor que 0. La condición era verdadera y se ejecutó el código del bloque. Sin embargo, si la condición
# es falsa, no vemos el resultado. Para ver el resultado de la condición falsa, debemos tener otro bloque, que va a ser else.  

# Si Else: Si la condición es verdadera se ejecutará el primer bloque, si no lo es se ejecutará la condición else.
# if condition:
#    Esta parte del código se ejecuta para condiciones verdaderas
# else:
#     Esta parte del código se ejecuta para condiciones falsas
a = 3
if a < 0:
    print('A es un número negativo')
else:
    print('A es un número positivo')  # A es un número positivo
# La condición anterior resulta falsa, por lo que se ejecuta el bloque else. ¿Qué pasa si nuestra condición es más de dos? Podríamos utilizar _ elif_."

# Si Elif Else: En nuestra vida cotidiana, tomamos decisiones a diario. No tomamos decisiones comprobando una o dos condiciones, sino múltiples condiciones.
# Al igual que en la vida, la programación también está llena de condiciones. Usamos elif cuando tenemos múltiples condiciones.
# if condition:
#     code
# elif condition:
#     code
# else:
#     code
a = 0
if a > 0:
    print('A es un número positivo')
elif a < 0:
    print('A es un número negativo')
else:
    print('A es cero')  # A es cero

# Mano corta (Short hand): code if condition else code
a = 3
print('A es positivo') if a > 0 else print('A es negativo') # (# Si primera condición es cumplida, se imprimirá 'A es positivo)

# Condiciones anidadas: Las condiciones pueden anidarse
# if condition:
#     code
#     if condition:
#     code

a = 0
if a > 0:
    if a % 2 == 0:
        print('A es un entero positivo y par')
    else:
        print('A es un número positivo')
elif a == 0:
    print('A es cero')                    # A es cero
else:
    print('A es un número negativo')

# Si a = 10   'A es un entero positivo y par'
# Si 1 = 11   'A es un número positivo'
# Si a = -1   'A es un número negativo'
# Podemos evitar escribir condiciones anidadas utilizando el operador lógico 'and'.

# Condición If y operadores lógicos
# if condition and condition:
#     code

a = 0
if a > 0 and a % 2 == 0:
        print('A es un entero par y positivo')
elif a > 0 and a % 2 != 0:  # (“!=” en Python se llama operador de desigualdad. Este operador se utiliza para evaluar si dos valores son diferentes)
     print('A es un entero positivo')
elif a == 0:
    print('A es cero')     # A es cero
else:
    print('A es negativo')

# Operadores lógicos If y Or
# if condición or condición:
#     código

usuario = 'Alvaro'
nivel_acceso = 3
if usuario == 'admin' or nivel_acceso >= 4:
        print('¡Acceso concedido!')
else:
    print('¡Acceso denegado!')  # ¡Acceso denegado!

# Si usuario = 'admin' ¡Acceso concedido!
# Si nivel_acceso = 4  ¡Acceso concedido!


# Ejercicios

# Obtener la entrada del usuario mediante input("Introduzca su edad: "). Si el usuario tiene 18 años o más, dar respuesta: "Tiene edad suficiente para conducir."
# Si es menor de 18, se le pide que "espere los años que le faltan". Salida:
# Introduzca su edad: 30
# Tiene edad suficiente para aprender a conducir.
# Salida:
# Introduzca su edad: 15
# Necesitas 3 años más para aprender a conducir.
edad = int(input("Introduzca su edad: "))  # (La función “input()” se utiliza para obtener la entrada del usuario. La función “int()” se utiliza para convertir la entrada del usuario en un número entero)

if edad >= 18:
    print("Tiene edad suficiente para conducir.")
else:
    anos_faltantes = 18 - edad
    print(f"Necesitas {anos_faltantes} años más para aprender a conducir.")

# Si edad 18 "Tiene edad suficiente para conducir."
# Si edad 15 "Necesitas 3 años más para aprender a conducir."

# Compara los valores de mi_edad y tu_edad utilizando if ... else. ¿Quién es mayor (tú o yo)? Utilice input("Introduzca su edad: ") para obtener la edad
# como entrada. Puede utilizar una condición anidada para imprimir 'año' para 1 año de diferencia en la edad, 'años' para diferencias mayores, y un 
# texto personalizado si mi_edad = tu_edad. Salida:
# Introduce tu edad: 30
# Eres 5 años mayor que yo.
tu_edad = int(input("Introduce tu edad: "))
mi_edad = 25 # asumimos que tengo 25 años

if tu_edad > mi_edad:
    diferencia = tu_edad - mi_edad
    if diferencia == 1:
        print(f"Eres {diferencia} año mayor que yo.")
    else:
        print(f"Eres {diferencia} años mayor que yo.")
elif tu_edad < mi_edad:
    diferencia = mi_edad - tu_edad
    if diferencia == 1:
        print(f"Soy {diferencia} año mayor que tú.")
    else:
        print(f"Soy {diferencia} años mayor que tú.")
else:
    print("Tenemos la misma edad.")

# Si tu edad 30 "Eres 5 años mayor que yo."
# Si tu edad 25 "Tenemos la misma edad"
# Si tu edad 24 "Soy 1 año mayor que tú."
# Si tu edad 20 "Soy 5 años mayor que tú."

# Obtiene dos números del usuario usando el prompt de entrada. Si a es mayor que b devuelve a es mayor que b, si a es menor que b devuelve a es menor 
# que b, si no a es igual a b.
a = int(input("A introduce tu edad: "))
b = int(input("B introduce tu edad: "))

if a > b:
    print(f"{a} es mayor que {b}.")
elif a < b:
    print(f"{a} es menor que {b}.")
else:
    print(f"{a} es igual a {b}.")

# Si a 30 y b 25 "30 es mayor que 25."
# Si a 25 y b 30 "25 es menor que 30."
# Si a 30 y b 30 "30 es igual a 30."

# Escribe un código que califique a los estudiantes según sus puntuaciones:
# 80-100, A
# 70-89, B
# 60-69, C
# 50-59, D
# 0-49, F
puntuacion = int(input("Introduce la puntuación del estudiante: "))

if puntuacion >= 80 and puntuacion <= 100:
    print("El estudiante obtuvo una A")
elif puntuacion >= 70 and puntuacion <= 89:
    print("El estudiante obtuvo una B")
elif puntuacion >= 60 and puntuacion <= 69:
    print("El estudiante obtuvo una C")
elif puntuacion >= 50 and puntuacion <= 59:
    print("El estudiante obtuvo una D")
elif puntuacion >= 0 and puntuacion <= 49:
    print("El estudiante obtuvo una F")
else:
    print("La puntuación introducida NO es inválida. Introduce un número entre 0 y 100.")
    
# Comprueba si la estación es otoño, invierno, primavera o verano. Si la entrada del usuario es Septiembre, Octubre o Noviembre, la estación es Otoño.
# Diciembre, Enero o Febrero, la estación es Invierno. Marzo, Abril o Mayo, la estación es Primavera. Junio, Julio o Agosto, la estación es Verano.  
    
mes = input("Introduce el mes del año: ")

if mes == 'Septiembre' or mes == 'Octubre' or mes == 'Noviembre':
    print("La estacion es Otoño")
elif mes == 'Diciembre' or mes == 'Enero' or mes == 'Febrero':
    print("La estacion es Invierno")
elif mes == 'Marzo' or mes== 'Abril' or mes == 'Mayo':
    print("La estacion es Primavera")
elif mes == 'Junio' or mes== 'Julio' or mes == 'Agosto':
    print("La estacion es Verano")
else:
    print("Rectifique el mes con la primera letra en mayuscula")

# otra forma de codigo:

mes = input("Introduce el mes del año: ")

if mes in ["Septiembre", "Octubre", "Noviembre"]:
    print("La estación del año es Otoño.")
elif mes in ["Diciembre", "Enero", "Febrero"]:
    print("La estación del año es Invierno.")
elif mes in ["Marzo", "Abril", "Mayo"]:
    print("La estación del año es Primavera.")
elif mes in ["Junio", "Julio", "Agosto"]:
    print("La estación del año es Verano.")
else:
    print("Rectifique el mes con la primera letra en mayuscula.")

# La siguiente lista contiene algunas frutas:
# frutas = ['plátano', 'naranja', 'mango', 'limón']
# Si una fruta no existe en la lista, añádela a la lista e imprime la lista modificada. Si la fruta existe print('Esa fruta ya existe en la lista')
frutas = ['platano', 'naranja', 'mango', 'limon']

fruta_nueva = input("Introduce una fruta: ")  # manzana

if fruta_nueva in frutas:
    print("Esa fruta ya existe en la lista.")
else:
    frutas.append(fruta_nueva)
    print("Fruta añadida correctamente. La lista de frutas es:", frutas)  # Fruta añadida correctamente. La lista de frutas es: ['platano', 'naranja', 'mango', 'limon', 'manzana']

# Aquí tenemos un diccionario de personas. ¡Siéntete libre de modificarlo!
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
# Comprueba si el diccionario de la persona tiene la clave skills, si es así imprime el skills del medio en la lista de skills.

if 'skills' in person:
    skills = person['skills']
    mid_skill = skills[len(skills)//2]
    print("La habilidad del medio es:", mid_skill) # La habilidad del medio es: Fuel oils
    
# Comprueba si el diccionario de personas tiene la clave skills, si es así comprueba si la persona tiene el skill 'Python' e imprime el resultado.
if 'skills' in person:
    if 'Python' in person['skills']:
        print('La persona tiene la habilidad de Python.')   # La persona tiene la habilidad de Python.
    else:
        print('La persona no tiene la habilidad de Python.')
else:
    print('La persona no tiene habilidades.')

# Si skills de una persona solo tienen JavaScript y React, imprime('Es un desarrollador front-end'), si skills de la persona
# tienen Node, Python, MongoDB, imprime('Es un desarrollador back-end'), si skills de la persona tienen React, Node y MongoDB, imprime
# ('Es un desarrollador fullstack'), si no imprime('título desconocido') - ¡para resultados más precisos se pueden anidar más condiciones!        
if 'skills' in person:
    skills = person['skills']
    if 'JavaScript' in skills and 'React' in skills:
        print('Es un desarrollador front-end')
    elif 'Node' in skills and 'Python' in skills and 'MongoDB' in skills:
        print('Es un desarrollador back-end')
    elif 'React' in skills and 'Node' in skills and 'MongoDB' in skills:
        print('Es un desarrollador fullstack')
    else:
        print('título desconocido')  # título desconocido
else:
    print('No se encontraron habilidades para esta persona.')

# Otra forma de hacerlo

if 'skills' in person:
    skills
    if skills in ['JavaScript', 'React']:
        print('Es un desarrollador front-end')
    elif skills in ['Node', 'Python', 'MongoDB']:
        print('Es un desarrollador back-end')
    elif skills in ['React', 'Node', 'MongoDB']:
        print('Es un desarrollador fullstack')
    else:
        print('título desconocido')  # título desconocido
else:
    print('No se encontraron habilidades para esta persona.')

# Si la persona está casada y vive en Colombia, imprima la información en el siguiente formato: Alvaro Nunez vive en Colombia. El es casado
if person['is_marred'] == True and person['country'] == 'Colombia':
    print(f"{person['first_name']} {person['last_name']} vive en {person['country']}. El es casado")  # Alvaro Nunez vive en Colombia. El es casado