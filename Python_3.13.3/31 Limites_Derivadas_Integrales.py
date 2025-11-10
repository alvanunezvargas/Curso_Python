# Regla de la suma: Calcular la derivada de una funcion f(x)=5*x^2, utilizando la definicion de la derivada como el limite:〖f^' (x)=lim┬(Δx→0)〗⁡((f(x+Δx)-f(x) )/|Δx| )

import sympy as sp

# Definir variables y función
x, delta_x = sp.symbols('x delta_x')
f_x = 5*x**2

# Definir la expresión de la derivada
f_prime_x = (f_x.subs(x, x + delta_x) - f_x) / delta_x

# Calcular el límite cuando delta_x tiende a 0
limit_result = sp.limit(f_prime_x, delta_x, 0)

print("El límite de f'(x) cuando Δx tiende a 0 es:", limit_result)

# Simplificar la expresion
expr_simplified = sp.nsimplify(limit_result)

# Mostrar el resultado de la expresion simplificada
print("La expresion simplificada de la derivada es", expr_simplified)

# Evaluar la derivada en un punto especifico
x_valor = sp.pi
derivada_en_x = f_prime_x.subs(x, x_valor)
print("La derivada en el punto es:", derivada_en_x)

# Simplificar la expresion
expr_simplified1 = sp.nsimplify(derivada_en_x)

# Mostrar el resultado de la expresion simplificada
print("La expresion simplificada de la derivada en el punto especifico es", expr_simplified1)


# Regla de la potencia y producto: Determinar la derivada de f(x)

import sympy as sp

# Definir el símbolo x
x = sp.Symbol('x')

# Definir la función
f_x = sp.exp(x)

# Calcular la derivada de f(x) respecto a x
f_prime_x = sp.diff(f_x, x)

# Imprimir la derivada
print("La derivada es:", f_prime_x)

# Simplificar la expresion
expr_simplified = sp.nsimplify(f_prime_x)

# Mostrar el resultado de la expresion simplificada
print("La expresion simplificada de la derivada es", expr_simplified)


# Determinar la integral de f'(x)

import sympy as sp

# Definir el símbolo x
x = sp.Symbol('x')

# Definir la función f'(x)
f_prime_x = f_x = sp.sqrt(x)

# Calcular la integral de f'(x) con respecto a x
integral_f_prime_x = sp.integrate(f_prime_x, x)

# Imprimir la integral
print("La integral de f'(x) es:", integral_f_prime_x)

# Simplificar la expresion
expr_simplified = sp.nsimplify(integral_f_prime_x)

# Mostrar el resultado de la expresion simplificada
print("La expresion simplificada de la derivada es", expr_simplified)


# Para comprobar si dos expresiones son iguales

import numpy as np

# Definimos una función para cada expresión
def expr1(m):
    return -(2/3) * (np.exp(m) - 1 ) * np.exp(m) + np.exp(m)

def expr2(m):
    return (1/3) * np.exp(m) * (5 - 2 * np.exp(m))                                                                                                                                          

# Definimos un rango de valores de m
m_values = np.linspace(-5, 5, 1000)

# Evaluamos ambas expresiones para cada valor de m
values_expr1 = expr1(m_values)
values_expr2 = expr2(m_values)

# Comprobamos si ambas expresiones son iguales con una tolerancia de 1e-10
are_equal = np.allclose(values_expr1, values_expr2, atol=1e-10)

# Imprimimos el resultado
if are_equal:
    print("Las expresiones son iguales para todos los valores de m.")
else:
    print("Las expresiones no son iguales para todos los valores de m.")
    

# Regla de la cadena: Calcular la derivada

import sympy as sp

# Definimos las variables
m = sp.Symbol('m')

# Definimos las funciónes p(m) y h(p)
p_m = sp.exp(m) - 1
h_p_m = -1/3 * (p_m)**2 + p_m + 1/5

# Calculamos la derivada de h(p(m)) respecto a m
dh_dm = sp.diff(h_p_m, m)

# Imprimimos el resultado
print("La derivada de h(p(m)) respecto a m es:", dh_dm)

# Simplificar la expresion
expr_simplified = sp.nsimplify(dh_dm)

# Mostrar el resultado de la expresion simplificada
print("La expresion simplificada de la derivada es", expr_simplified)


# Calculo de derivadas parciales

import sympy as sp

# Definimos los símbolos
x, y, z= sp.symbols('x y z')

# Definimos la función
f = x**2 + 3 * sp.exp(y) * sp.exp(z) + sp.cos(x) * sp.sin(z)

# Calculamos la derivada con respecto a x
df_dx = sp.diff(f, x)
print("Derivada de f(x, y) con respecto a x:")
print(df_dx)

# Simplificar las expresiones
df_dx_simplified = sp.nsimplify(df_dx)
print("La expresion simplificada de la derivada de x es", df_dx_simplified)

# Calculamos la derivada con respecto a y
df_dy = sp.diff(f, y)
print("\nDerivada de f(x, y) con respecto a y:")
print(df_dy)

# Simplificar las expresiones
df_dy_simplified = sp.nsimplify(df_dy)
print("La expresion simplificada de la derivada de x es", df_dy_simplified)

# Calculamos la derivada con respecto a z
df_dz = sp.diff(f, z)
print("\nDerivada de f(x, y) con respecto a y:")
print(df_dz)

# Simplificar las expresiones
df_dz_simplified = sp.nsimplify(df_dz)
print("La expresion simplificada de la derivada de x es", df_dz_simplified)

# Definir el jacobiano
jacobian = sp.Matrix([[df_dx_simplified], [df_dy_simplified], [df_dz_simplified]])

# Imprimir el jacobiano
print("\nEl jacobiano de f es:")
print(jacobian)

# Definir los valores de x, y, z
x_val = 0
y_val = 0
z_val = 0

# Sustituir los valores en el jacobiano
jacobian_val = jacobian.subs({x: x_val, y: y_val, z: z_val})

# Imprimir el jacobiano en el punto
print("\nEl jacobiano de f en el punto (x, y, z) = (0, 0, 0) es:")
print(jacobian_val)


# Obtener la derivada de la funcion f(x, y, z) con respecto a x, y, Y z donde x, y Y z son funciones de t

import sympy as sp

# Definir los símbolos y las funciones de t
t = sp.Symbol('t')
x = t + 1
y = t - 1
z = t**2

# Definir la función f(x, y, z)
f = sp.cos(x) * sp.sin(y) * sp.exp(2*z)

# Calcular la derivada de f con respecto a t
df_dt = sp.diff(f, t)

# Imprimir la derivada
print("La expresion de la derivada es", df_dt)

# Evaluar la derivada en un punto especifico
derivada_en_t = df_dt.subs({x: t - 1, y: t**2, z: 1/t})

# Imprimir la derivada en el punto
print("La derivada en el punto es:", derivada_en_t)

# Simplificar las expresiones
df_dt_simplified = sp.nsimplify(df_dt)
derivada_en_t_simplified = sp.nsimplify(derivada_en_t)

# Imprimir las expresiones simplificadas
print("La expresion simplificada de la derivada es", df_dt_simplified)
print("La expresion simplificada de la derivada en el punto especifico es", derivada_en_t_simplified)


# Calcular la matriz jacobiana de una función f(x, y, z, ...n) y evaluarla en un punto específico. Calcular la determinante

import sympy as sp

# Definir los símbolos y las funciones de t
x, y, z = sp.symbols('x y z')

# Definir la función f(x, y, z)
f_u = x + sp.sin(y) + z
f_v = y + sp.sin(x) - z
f_w = x * y + z

# Calcular la derivada de f con respecto a x, y, z
dfu_dx, dfu_dy, dfu_dz = sp.diff(f_u, x), sp.diff(f_u, y), sp.diff(f_u, z)
dfv_dx, dfv_dy, dfv_dz = sp.diff(f_v, x), sp.diff(f_v, y), sp.diff(f_v, z)
dfw_dx, dfw_dy, dfw_dz = sp.diff(f_w, x), sp.diff(f_w, y), sp.diff(f_w, z)

# Definir el jacobiano
jacobian = sp.Matrix([[dfu_dx, dfu_dy, dfu_dz], [dfv_dx, dfv_dy, dfv_dz], [dfw_dx, dfw_dy, dfw_dz]])

# Definir los valores de x, y, z
x_val, y_val, z_val = 0, 1, 2

# Sustituir los valores en el jacobiano
jacobian_val = jacobian.subs({x: x_val, y: y_val, z: z_val})

# Calcular el valor numérico del jacobiano
jacobian_num = jacobian_val.evalf()

# Calcular el determinante del jacobiano
det_jacobian = jacobian.det()

# Calcular el determinante del jacobiano con valores de (x, y, z ...N)
det_jacobianN = jacobian_val.det()

# Calcular el valor numérico del determinante del jacobiano
det_jacobian_num = det_jacobianN.evalf()

# Imprimir los resultados
print("\nEl jacobiano de f es:")
print(jacobian)
print("\nLa expresion del jacobiano de f en el punto (x, y, z,....n) es:")
print(jacobian_val)
print("\nEl valor numérico del jacobiano en el punto (x, y, z,....n) es:")
print(jacobian_num)
print("\nEl determinante del jacobiano es:")
print(det_jacobian)
print("\nLa expresion para el calculo del determinante del jacobiano en el punto (x, y, z,....n) es:")
print(det_jacobianN)
print("\nEl valor numérico del determinante del jacobiano en el punto (x, y, z,....n) es:")
print(det_jacobian_num)

# Otra forma mas optimas de calcular el jacobiano y el determinante

import sympy as sp

# Definir los símbolos
symbols = sp.symbols('x y z')

# Definir las funciones: (symbols[0] = x, symbols[1] = y, symbols[2] = z)
f_u = symbols[0] + sp.sin(symbols[1]) + symbols[2]
f_v = symbols[1] + sp.sin(symbols[0]) - symbols[2]
f_w = symbols[0] * symbols[1] + symbols[2]
functions = [f_u, f_v, f_w]

# Calcular las derivadas de las funciones con respecto a cada símbolo
derivatives = [[f.diff(s) for s in symbols] for f in functions]

# Definir el jacobiano
jacobian = sp.Matrix(derivatives)

# Definir los valores de los símbolos
values = {s: v for s, v in zip(symbols, [0, 1, 2])}

# Sustituir los valores en el jacobiano
jacobian_val = jacobian.subs(values)

# Calcular el valor numérico del jacobiano
jacobian_num = jacobian_val.evalf()

# Calcular el determinante del jacobiano
det_jacobian = jacobian.det()

# Calcular el determinante del jacobiano con valores de (x, y, z ...N)
det_jacobianN = jacobian_val.det()

# Calcular el valor numérico del determinante del jacobiano
det_jacobian_num = det_jacobianN.evalf()

# Imprimir los resultados
print("\nEl jacobiano de f es:")
print(jacobian)
print("\nLa expresion del jacobiano de f en el punto (x, y, z,....n) es:")
print(jacobian_val)
print("\nEl valor numérico del jacobiano en el punto (x, y, z,....n) es:")
print(jacobian_num)
print("\nEl determinante del jacobiano es:")
print(det_jacobian)
print("\nLa expresion para el calculo del determinante del jacobiano en el punto (x, y, z,....n) es:")
print(det_jacobianN)
print("\nEl valor numérico del determinante del jacobiano en el punto (x, y, z,....n) es:")
print(det_jacobian_num)


# Para obtener la matriz hessiana de la función f(x, y, z) y evaluarla en un punto especifico (x, y, z, ... n) se puede utilizar la función hessian de la librería sympy.

import sympy as sp

# Definir los símbolos
x, y, z = sp.symbols('x y z')

# Definir la función
f = x * y * sp.cos(z) - sp.sin(x) * sp.exp(y) * z**3

# Calcular la matriz hessiana
hessian_matrix = sp.hessian(f, (x, y, z))

# Definir los valores de los símbolos
values = {x: 0, y: 0, z: 0}

# Sustituir los valores en la matriz hessiana y calcular su valor numérico
hessian_num = hessian_matrix.subs(values).evalf()

# Imprimir la matriz hessiana
print("Matriz Hessiana:")
sp.pprint(hessian_matrix)

# Imprimir la matriz hessiana evaluada en los puntos (x, y, z) = (0, 0, 0)
print("\nMatriz Hessiana en los puntos (x, y, z) = (0, 0, 0):")
sp.pprint(hessian_num)


# Definir la fucncion compuesta de f(u) a partir de las funciones f(x), x(u) y u(t) y luego calcula la derivada de f(u) con respecto a t.

import sympy as sp

# Definir las variables simbólicas
x, u, t = sp.symbols('x u t')

# Definir las funciones dadas
f_x = 5 * x
x_u = 1 - u
u_t = t ** 2

# Calcular las derivadas parciales
df_dx = sp.diff(f_x, x)  # Derivada de f(x) con respecto a x
dx_du = sp.diff(x_u, u)   # Derivada de x(u) con respecto a u
du_dt = sp.diff(u_t, t)   # Derivada de u(t) con respecto a t

# Aplicar la regla de la cadena
df_du = df_dx.subs(x, x_u) * dx_du  # Derivada de f(u) con respecto a u

# Mostrar la derivada de f(u)
print("La derivada de f(u) es:", df_du)


# Calculo de la derivada de una función compuesta por aplicacion de la regla de la cadena multivariada

# df/dt = df/dx dx/dt

# Calculo de la derivada de una función compuesta por aplicacion de la regla de la cadena

import sympy as sp

# Definir las variables simbólicas
x1, x2, x3, t = sp.symbols('x1 x2 x3 t')

# Definir las funciones dadas
f_x = x1**3 * sp.cos(x2) * sp.exp(x3)
x1_t = 2 * t
x2_t = 1 - t**2
x3_t = sp.exp(t)

# Calcular las derivadas parciales
df_dx1 = sp.diff(f_x, x1)  # Derivada de f(x) con respecto a x1
df_dx2 = sp.diff(f_x, x2)  # Derivada de f(x) con respecto a x2
df_dx3 = sp.diff(f_x, x3)  # Derivada de f(x) con respecto a x3

dx1_dt = sp.diff(x1_t, t)  # Derivada de x1 con respecto a t
dx2_dt = sp.diff(x2_t, t)  # Derivada de x2 con respecto a t
dx3_dt = sp.diff(x3_t, t)  # Derivada de x3 con respecto a t

# Construir matrices jacobianas
df_dx = sp.Matrix([df_dx1, df_dx2, df_dx3])  # Jacobiana de f(x) con respecto a x

# Jacobiana de x(t) con respecto a t
dx_dt = sp.Matrix([dx1_dt, dx2_dt, dx3_dt])  # Vector dx/dt

# Calcular df/dt
df_dt = df_dx.transpose() * dx_dt

# Mostrar df/dt
print(df_dx.transpose())
print(dx_dt)


# df/dt = df/dx dx/du dx/dt

import sympy as sp

# Definir las variables simbólicas
x1, x2, x3, u1, u2, t = sp.symbols('x1 x2 x3 u1 u2 t')

# Definir las funciones dadas
f_x = sp.sin(x1) * sp.cos(x2) * sp.exp(x3)
x1_u1_u2 = sp.sin(u1) + sp.cos(u2)
x2_u1_u2 = sp.cos(u1) - sp.sin(u2)
x3_u1_u2 = sp.exp(u1+u2)
u1_t = 1 + t/2
u2_t = 1 - t/2

# Calcular las derivadas parciales
df_dx1 = sp.diff(f_x, x1)  # Derivada de f(x) con respecto a x1
df_dx2 = sp.diff(f_x, x2)  # Derivada de f(x) con respecto a x2
df_dx3 = sp.diff(f_x, x3)  # Derivada de f(x) con respecto a x3

dx1_du1 = sp.diff(x1_u1_u2, u1)  # Derivada de x1(u1, u2) con respecto a u1
dx1_du2 = sp.diff(x1_u1_u2, u2)  # Derivada de x1(u1, u2) con respecto a u2
dx2_du1 = sp.diff(x2_u1_u2, u1)  # Derivada de x2(u1, u2) con respecto a u1
dx2_du2 = sp.diff(x2_u1_u2, u2)  # Derivada de x2(u1, u2) con respecto a u2
dx3_du1 = sp.diff(x3_u1_u2, u1)  # Derivada de x3(u1, u2) con respecto a u1
dx3_du2 = sp.diff(x3_u1_u2, u2)  # Derivada de x3(u1, u2) con respecto a u2

du1_dt = sp.diff(u1_t, t)  # Derivada de u1(t) con respecto a t
du2_dt = sp.diff(u2_t, t)  # Derivada de u2(t) con respecto a t

# Construir matrices jacobianas
df_dx = sp.Matrix([df_dx1, df_dx2, df_dx3])  # Jacobiana de f(x) con respecto a x
dx_du = sp.Matrix([[dx1_du1, dx1_du2], [dx2_du1, dx2_du2], [dx3_du1, dx3_du2]])  # Jacobiana de x(u) con respecto a u
du_dt = sp.Matrix([du1_dt, du2_dt])  # Vector du/dt

# Calcular df/dt
df_dt = df_dx.transpose() * dx_du * du_dt

# Mostrar df/dt
print(df_dx.transpose())
print(dx_du)
print(du_dt)


# Calcular la derivada del coste de una red neuronal utilizando la regla de la cadena.

import sympy as sp

# Definir las variables simbólicas
a1, a0, ô, y, z1, w1, b1  = sp.symbols('a1 a0 ô y z1 w1 b1')

# Definir las funciones dadas
C_k = (a1 - y)**2
a_z = ô * z1
z_w = w1 * a0 + b1

# Calcular las derivadas parciales
df_C_k = sp.diff(C_k, a1)  # Derivada de C_k con respecto a a1
df_a_z = sp.diff(a_z, z1)   # Derivada de a_z con respecto a z
df_z_w = sp.diff(z_w, w1)  # Derivada de z_w con respecto a w
df_z_b1 = sp.diff(z_w, b1)  # Derivada de z_w con respecto a b1

print(df_C_k)
print(df_a_z)
print(df_z_w)
print(df_z_b1)


# Calcula la derivada de la funcion de costo (cuadrado de la diferencia entre la activacion a1 y el valor objetivo y) con respecto al sesgo b

import numpy as np

# Primero definimos nuestra funcion sigma.
sigma = np.tanh

# A continuación, define la ecuación de propagación directa
def a1 (w1, b1, a0) :
  z = w1 * a0 + b1
  return sigma(z)

# La función de costo individual es el cuadrado de la diferencia entre 
# la salida de la red y la salida de los datos de entrenamiento.
def C (w1, b1, x, y) :
  return (a1(w1, b1, x) - y)**2

# Esta función devuelve la derivada de la función de costo 
# con respecto al peso.
def dCdw (w1, b1, x, y) :
  z = w1 * x + b1
  dCda = 2 * (a1(w1, b1, x) - y) # Derivada del costo con activación
  dadz = 1/np.cosh(z)**2 # Derivada de la activación con suma ponderada z
  dzdw = x # Derivada de la suma ponderada z con respecto al peso w
  return dCda * dadz * dzdw # Devuelve el producto de la regla de la cadena

# Esta función calcula la derivada de la función de costo 
# con respecto al sesgo.
# Es muy similar a la función anterior
# así que deberías poder completarla
def dCdb (w1, b1, x, y) :
  z = w1 * x + b1
  dCda = 2 * (a1(w1, b1, x) - y)
  dadz = 1/np.cosh(z)**2
  """ Modifica la siguiente línea para obtener la derivada 
  de la suma ponderada, z, con respecto al sesgo, b """
  dzdb = 1
  return dCda * dadz * dzdb

""" Prueba tu código antes de enviarlo:"""
# Comencemos con un peso y un sesgo no ajustados.
w1 = 2.3
b1 = -1.2
# Podemos probar con un único par de datos de punto x e y.
x = 0
y = 1
# Muestra cómo cambiaría el costo
# con un pequeño cambio en el sesgo
print( dCdb(w1, b1, x, y) )


# Calcular la funcion de costo (cuadrado de la diferencia entre la activacion x y el valor objetivo y) de esta red.

import numpy as np

# Defina la funcion de activacion
sigma = np.tanh

# Usemos un peso y un sesgo inicial aleatorio.
W = np.array([[-0.94529712, -0.2667356 , -0.91219181],
              [ 2.05529992,  1.21797092,  0.22914497]])
b = np.array([ 0.61273249,  1.6422662 ])

# Definamos la función de propagación directa
def a1 (a0) :
  # Observa que la siguiente línea es casi igual a la anterior,
  # excepto que usamos la multiplicación matricial con el operador "@" en lugar de la multiplicación escalar 
  # con el operador "*".
  z = W @ a0 + b
  # Todo lo demás es igual..,
  return sigma(z)

# A continuación, si un ejemplo de entrenamiento es,
x = np.array([0.7, 0.6, 0.2])
y = np.array([0.9, 0.6])

# Entonces, la función de costo es,
d = a1(x) - y # Diferencia vectorial entre la activación observada (a1(x)) y la activación esperada (y)
C = d @ d # Valor absoluto al cuadrado de la diferencia.
print(C)



"""
En esta tarea, entrenará una red neuronal para dibujar una curva implementando la retropropagación por la regla de la cadena para calcular los jacobianos de la función de coste.

A continuación, entrenará la red neuronal mediante un método de descenso más pronunciado estocástico (preimplementado) y dibujará una serie de curvas para mostrar el progreso del entrenamiento.

Retropropagación ¶
Instrucciones:

En esta tarea, entrenarás una red neuronal para dibujar una curva. La curva toma una variable de entrada, la distancia recorrida a lo largo de la curva de 0 a 1, y devuelve 2 salidas, las coordenadas 2D de la posición de los puntos en la curva.

Para ayudar a capturar la complejidad de la curva, utilizaremos dos capas ocultas en nuestra red con 6 y 7 neuronas respectivamente. Una neurona en la entrada y dos neuronas en la salida.

Se te pedirá completar funciones que calculen el Jacobiano de la función de costo, con respecto a los pesos y sesgos de la red. Tu código formará parte de un algoritmo estocástico de descenso más pronunciado que entrenará tu red.

Matrices en Python:

Recuerda de las tareas del curso anterior en esta especialización que las matrices pueden multiplicarse de dos maneras en Python:

Por elementos: Cuando dos matrices tienen las mismas dimensiones, los elementos de la matriz en la misma posición en cada matriz se multiplican entre sí. En Python, esto se usa con el operador '∗'.

A = B * C

Multiplicación de matrices: Cuando el número de columnas de la primera matriz es igual al número de filas de la segunda. En Python, esto se usa con el operador '@'.

Esta es una manera más eficiente de calcular el producto matricial completo cuando las dimensiones son compatibles. A diferencia de la multiplicación elemental, que se realiza elemento por elemento, la multiplicación con '@' calcula directamente el producto de matrices siguiendo las reglas estándar de la multiplicación matricial.

Recuerda que para que la multiplicación de matrices sea válida, el número de columnas de la primera matriz debe ser igual al número de filas de la segunda matriz.

Esta tarea no evaluará en qué casos se utiliza cada tipo de multiplicación, pero se utilizarán ambas en el código inicial que se te proporciona. No necesitas cambiarlas ni preocuparte por sus detalles específicos.

Cómo enviar la tarea:

Para completar la tarea, edita el código en las celdas que se te indican a continuación. Una vez que hayas terminado y estés satisfecho con él, pulsa el botón "Enviar tarea" en la parte superior de este cuaderno. Prueba tu código utilizando las celdas de la parte inferior del cuaderno antes de enviarlo.

Por favor, no cambies ninguno de los nombres de las funciones, ya que serán comprobados por el script de calificación.

Propagación directa:

En la siguiente celda, definiremos funciones para configurar nuestra red neuronal. Estas funciones incluyen una función de activación, σ(z), su derivada, σ′(z), una función para inicializar pesos y sesgos, y una función que calcula cada activación de la red utilizando la propagación directa.

Recuerda las ecuaciones de propagación directa:

a(n) = σ (z(n))
z(n) = w(n) a(n-1) + b(n)

En esta tarea utilizaremos la función logística como nuestra función de activación, en lugar de la más familiar tangente hiperbólica (tanh).

σ(z) = 1 / (1 + exp(−z))
"""

# APLICACION
import numpy as np
import matplotlib.pyplot as plt

# Importaciones necesarias:
# Se cargan las dependencias necesarias para el funcionamiento del script.
# Se define la función de activación y su derivada.
sigma = lambda z : 1 / (1 + np.exp(-z))
d_sigma = lambda z : np.cosh(z/2)**(-2) / 4

# Esta función inicializa la red neuronal con su estructura y además reinicia cualquier entrenamiento previo realizado.
def reset_network (n1 = 6, n2 = 7, random=np.random) :
    global W1, W2, W3, b1, b2, b3
    W1 = random.randn(n1, 1) / 2
    W2 = random.randn(n2, n1) / 2
    W3 = random.randn(2, n2) / 2
    b1 = random.randn(n1, 1) / 2
    b2 = random.randn(n2, 1) / 2
    b3 = random.randn(2, 1) / 2

# Esta función propaga cada activación hacia la siguiente capa. Devuelve todas las sumas ponderadas y activaciones.
def network_function(a0) :
    z1 = W1 @ a0 + b1
    a1 = sigma(z1)
    z2 = W2 @ a1 + b2
    a2 = sigma(z2)
    z3 = W3 @ a2 + b3
    a3 = sigma(z3)
    return a0, z1, a1, z2, a2, z3, a3

# Esta función calcula el costo de la red neuronal en relación con un conjunto de datos de entrenamiento.
def cost(x, y) :
    return np.linalg.norm(network_function(x)[-1] - y)**2 / x.size
print(cost)

"""
En las siguientes celdas, se te pedirá completar las funciones para el Jacobiano de la función de costo con respecto a los pesos y sesgos. Comenzaremos con la capa 3, que es la más fácil, y luego trabajaremos hacia atrás a través de las capas.

Definiremos nuestros Jacobians como:

JW(3) = ∂C/∂W(3)
Jb(3) = ∂C/∂b(3)

etc., donde C es la función de costo promedio sobre el conjunto de entrenamiento. Esto significa que

C = 1/N ∑ Ck 

En los cuestionarios de práctica, calculaste lo siguiente:

∂C/∂W(3) = ∂C/∂a(3) * ∂a(3)/∂z(3) * ∂z(3)/∂W(3)

Para el peso, [derivada parcial de la función de costo con respecto a la activación] * [derivada de la activación con respecto al peso] y de manera similar para el sesgo [derivada parcial de la función de costo con respecto a la activación] * [derivada de la activación con respecto al sesgo]

∂C/∂b(3) = ∂C/∂a(3) * ∂a(3)/∂z(3) * ∂z(3)/∂b(3)

Las derivadas parciales se expresan de la siguiente forma:

∂C/∂a(3) = 2(a(3) - y)
∂a(3)/∂z(3) = σ′(z(3))
∂z(3)/∂W(3) = a(2)
∂z(3)/∂b(3) = 1
∂a(3)/∂a(2) = W(3)
∂a(2)/∂a(1) = W(2)
∂a(1)/∂z(1) = σ′(z(1))
∂z(1)/∂W(1) = a(0)
∂z(1)/∂b(1) = 1

Te proporcionaremos la función "J_W3 (JW(3))" para que veas cómo funciona. De esta manera, deberías poder adaptar la función "J_b3" por ti mismo, con un poco de ayuda.
"""
# FUNCION EVALUADA

# Primera función (J_w3): Esta función calcula el Jacobiano de la función de costo con respecto a los pesos de la capa 3. No necesitas editarla.
def J_W3 (x, y) :
    # Primero obtiene todas las activaciones y sumas ponderadas en cada capa de la red.
    a0, z1, a1, z2, a2, z3, a3 = network_function(x)
    # Utiliza la variable "J" para almacenar partes del resultado, actualizándola en cada línea.
    # Calcula ∂C/∂a(3) usando las expresiones proporcionadas.
    J = 2 * (a3 - y)
    # Multiplica el resultado por la derivada de sigma, evaluada en z3.
    J = J * d_sigma(z3)
    # Realiza el producto punto (a lo largo del eje que contiene los ejemplos de entrenamiento) con la derivada parcial final,
    # es decir, ∂z(3)/∂W(3) = a(2)
    # Divide por el número de ejemplos de entrenamiento para obtener el promedio.
    J = J @ a2.T / x.size
    # Devuelve el resultado final.
    return J

# En esta función, implementarás el Jacobiano para el sesgo.
# Como verás en las derivadas parciales, solo la última es diferente.
# Las dos primeras derivadas parciales son las mismas que en la función anterior.
# ===Instrucciones de edición===
def J_b3 (x, y) :
    # Primero, configura las activaciones (lo mismo que en la función anterior).
    a0, z1, a1, z2, a2, z3, a3 = network_function(x)
    # Luego, implementa las dos primeras derivadas parciales del Jacobiano.
    # ===Puedes copiar las dos líneas de la función anterior para configurar los dos primeros términos del Jacobiano===
    J = 2 * (a3 - y)
    J = J * d_sigma(z3)
    # Para la línea final, no necesitas multiplicar por dz3/db3 porque eso equivale a multiplicar por 1.
    # Sin embargo, sí necesitas sumar el resultado final sobre todos los ejemplos de entrenamiento.
    # No es necesario editar esta linea.
    J = np.sum(J, axis=1, keepdims=True) / x.size
    return J

"""
A continuación, haremos el Jacobiano para la capa 2. Las derivadas parciales para esto son:

∂C/∂W(2) = ∂C/∂a(3) * ((∂a(3)/∂a(2)) * ∂a(2)/∂z(2)) * ∂z(2)/∂W(2)
∂C/∂b(2) = ∂C/∂a(3) * ((∂a(3)/∂a(2)) * ∂a(2)/∂z(2)) * ∂z(2)/∂b(2)

Esto es muy similar a la capa anterior, con dos excepciones:

Hay una nueva derivada parcial, entre paréntesis, ∂a(3)/∂a(2)
Los términos después de los paréntesis ahora están un nivel más abajo.

Recuerda que la nueva derivada parcial toma la siguiente forma:

∂a(3)/∂a(2) = ∂a(3)/∂z(3) * ∂z(3)/∂a(2) = W(3) * σ′(z(3))

Para mostrar cómo cambia esto las cosas, implementaremos el Jacobiano para el peso nuevamente y te pediremos que lo implementes para el sesgo.
"""
# FUNCION EVALUADA

# Compara esta función con J_W3 para ver cómo cambia.
# No es necesario editar esta función.
def J_W2 (x, y) :
    # Las dos primeras líneas son idénticas a las de J_W3.
    a0, z1, a1, z2, a2, z3, a3 = network_function(x)    
    J = 2 * (a3 - y)
    # Las siguientes dos líneas implementan da3/da2, primero σ' y luego W3.
    J = J * d_sigma(z3)
    J = (J.T @ W3).T
    # Luego, las líneas finales son las mismas que en J_W3, pero con el número de capa reducido.
    J = J * d_sigma(z2)
    J = J @ a1.T / x.size
    return J

# Como antes, completa todas las líneas incompletas.
# ===EDITE LA FUNCION===
def J_b2 (x, y) :
    a0, z1, a1, z2, a2, z3, a3 = network_function(x)
    J = 2 * (a3 - y)
    J = J * d_sigma(z3)
    J = (J.T @ W3).T
    J = J * d_sigma(z2)
    J = np.sum(J, axis=1, keepdims=True) / x.size
    return J


"""
La capa 1 es muy similar a la capa 2, pero con un término de derivada parcial adicional

∂C/∂W(1) = ∂C/∂a(3) * (∂a(3)/∂a(2) * ∂a(2)/∂a(1)) * ∂a(1)/∂z(1) * ∂z(1)/∂W(1)
∂C/∂b(1) = ∂C/∂a(3) * (∂a(3)/∂a(2) * ∂a(2)/∂a(1)) * ∂a(1)/∂z(1) * ∂z(1)/∂b(1)

Debería poder adaptar las líneas de las celdas anteriores para completar el Jacobiano del peso y del sesgo
"""

# FUNCIÓN EVALUADA

# Complete las lineas.
# ===Edite la funcion===
def J_W1 (x, y) :
    a0, z1, a1, z2, a2, z3, a3 = network_function(x)
    J = 2 * (a3 - y)
    J = J * d_sigma(z3)
    J = (J.T @ W3).T
    J = J * d_sigma(z2)
    J = (J.T @ W2).T
    J = J * d_sigma(z1)
    J = J @ a0.T / x.size
    return J

# Complete las lineas.
# ===Edite la funcion===
def J_b1 (x, y) :
    a0, z1, a1, z2, a2, z3, a3 = network_function(x)
    J = 2 * (a3 - y)
    J = J * d_sigma(z3)
    J = (J.T @ W3).T
    J = J * d_sigma(z2)
    J = (J.T @ W2).T
    J = J * d_sigma(z1)
    J = np.sum(J, axis=1, keepdims=True) / x.size
    return J

"""
Primero, generamos datos de entrenamiento y creamos una red neuronal con pesos y sesgos asignados aleatoriamente.
"""
# Generar datos de entrenamiento
x_train = np.linspace(0, 2*np.pi, 100)
y_train = np.sin(x_train)

# Inicializar los pesos y sesgos de la red con valores aleatorios
reset_network()

# Realiza un paso de optimización por descenso de gradiente.

def gradient_descent_step(x, y, learning_rate):
    global W1, W2, W3, b1, b2, b3
    
    # Calcular las jacobianas
    dJ_dW1 = J_W1(x, y)
    dJ_db1 = J_b1(x, y)
    dJ_dW2 = J_W2(x, y)
    dJ_db2 = J_b2(x, y)
    dJ_dW3 = J_W3(x, y)
    dJ_db3 = J_b3(x, y)
    
    # Actualizar los pesos y sesgos utilizando el descenso de gradiente
    W1 -= learning_rate * dJ_dW1
    b1 -= learning_rate * dJ_db1
    W2 -= learning_rate * dJ_dW2
    b2 -= learning_rate * dJ_db2
    W3 -= learning_rate * dJ_dW3
    b3 -= learning_rate * dJ_db3

# Entrenar la red neuronal
def plot_training(x, y, iterations=10000, aggression=7, noise=1):
    plt.scatter(x, y, color='green', label='Training Data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Training Progress')
    plt.legend()
    plt.show()
    
    for i in range(iterations):
        gradient_descent_step(x, y, aggression / (1 + noise * i))
        if i == iterations - 1:
            predictions = network_function(x)[-1]
            plt.plot(x, y, 'g-', label='Training Data')
            plt.plot(x, predictions, 'm-', label='Iteration {}'.format(i))
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend()
            plt.title('Training Progress')
            plt.show()
            
plot_training(x_train, y_train)

"""
Se necesitan aproximadamente 50,000 iteraciones para entrenar esta red. Sin embargo, podemos dividir esto: 10,000 iteraciones deberían tardar aproximadamente un minuto en ejecutarse. Ejecuta la línea siguiente tantas veces como quieras.

Si lo deseas, puedes cambiar los parámetros del algoritmo de descenso más pronunciado (profundizaremos en esto en ejercicios futuros), pero puedes modificar la cantidad de iteraciones representadas, la agresividad del descenso por la Jacobiana y la cantidad de ruido que se agrega.

También puedes editar los parámetros de la red neuronal, es decir, asignarle diferentes cantidades de neuronas en las capas ocultas llamando a la función:

reset_network(n1, n2)
"""

"""
Determinar la fórmula de la serie de Taylor para aproximar términos de la función f(x) = ? expandida alrededor del punto p = ?

Las aproximaciones de Taylor son una herramienta poderosa en matemáticas que utilizan polinomios para aproximar funciones complejas. Se basan en un teorema importante llamado Teorema de Taylor que establece que cualquier función suficientemente "agradable" (diferenciable) en un punto, se puede aproximar localmente por un polinomio.
"""

import sympy as sp
import matplotlib.pyplot as plt
import numpy as np

def taylor_series(f, p, n):
    x = sp.Symbol('x')
    terms = []

    # Calcular las derivadas de la función en el punto p
    derivadas = [f]
    for i in range(1, n + 1):
        derivada = sp.diff(derivadas[-1], x)
        derivadas.append(derivada)

    # Construir y mostrar los términos de la serie de Taylor
    for i in range(n + 1):
        termino = derivadas[i].subs(x, p) * (x - p)**i / sp.factorial(i)
        print(f"Término {i}: {termino}")
        terms.append(termino)

    return terms

# Definir la función, el punto y el orden de la aproximación
x = sp.symbols('x')
f = x * sp.sin(x)
p = 0
n = 6

# Imprimir la serie de Taylor
print(f"Serie de Taylor para aproximar la función {f} alrededor del punto p = {p} en los {n} primeros terminos es:")
taylor_terms = taylor_series(f, p, n)

# Calcular e imprimir las derivadas hasta el orden n
current_derivative = f
for i in range(1, n+1):
    current_derivative = sp.diff(current_derivative, x)
    print(f"\nDerivada de orden {i} de f(x) con respecto a x:")
    print(current_derivative)

# Convertir la función sympy a una función numpy para graficar
f_lambdified = sp.lambdify(x, f, "numpy")
x_vals1 = np.linspace(-10, -0.01, 200)  # Ajustar el rango de x para evitar x = 0
x_vals2 = np.linspace(0.01, 10, 200)  # Ajustar el rango de x para evitar x = 1
x_vals = np.concatenate((x_vals1, x_vals2))
y_vals_func = f_lambdified(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals_func, label='Función original', linewidth=2)

# Definir los límites de los ejes x e y
plt.xlim(-10, 10)
plt.ylim(-10, 10)

for i, term in enumerate(taylor_terms):
    term_lambdified = sp.lambdify(x, term, "numpy")
    y_vals_term = np.array([term_lambdified(val) for val in x_vals])
    plt.plot(x_vals, y_vals_term, label=f'Término de Taylor {i}')

plt.legend()
plt.grid(True)
plt.title('Función original y términos de la serie de Taylor')
plt.show()


# Graficar funciones en graficas independientes en una sola vista.

import numpy as np
import matplotlib.pyplot as plt

# Definir el rango de valores de x
x = np.linspace(-2*np.pi, 2*np.pi, 400)

# Definir las funciones
y1 = x*np.cos(x) + np.sin(x)
y2 = -x*np.sin(x) + 2*np.cos(x)
y3 = -x*np.cos(x) - 3*np.sin(x)
y4 = x*np.sin(x) - 4*np.cos(x)
y5 = x*np.cos(x) + 5*np.sin(x)
y6= -x*np.sin(x) + 6*np.cos(x)

# Crear la figura y los subplots
fig, axs = plt.subplots(3, 2, figsize=(10, 8))

# Graficar la primera función
axs[0, 0].plot(x, y1)
axs[0, 0].set_title('Función 1')

# Graficar la segunda función
axs[0, 1].plot(x, y2)
axs[0, 1].set_title('Función 2')

# Graficar la tercera función
axs[1, 0].plot(x, y3)
axs[1, 0].set_title('Función 3')

# Graficar la cuarta función
axs[1, 1].plot(x, y4)
axs[1, 1].set_title('Función 4')

# Graficar la quinta función
axs[2, 0].plot(x, y5)
axs[2, 0].set_title('Función 5')

# Graficar la sexta función
axs[2, 1].plot(x, y6)
axs[2, 1].set_title('Función 6')

# Ajustar diseño y mostrar las gráficas
plt.tight_layout()
plt.show()


"""
Determinar la fórmula de la serie de Taylor para aproximar términos de la función f(x, y) = ? expandida alrededor del punto p = x, y. Determinar la matriz jacobiana y hessina de la función f(x, y) = ? en el punto p = x, y.

Las aproximaciones de Taylor son una herramienta poderosa en matemáticas que utilizan polinomios para aproximar funciones complejas. Se basan en un teorema importante llamado Teorema de Taylor que establece que cualquier función suficientemente "agradable" (diferenciable) en un punto, se puede aproximar localmente por un polinomio.
"""

import sympy as sp
from sympy import simplify

def taylor_series_2d(f, p, n):
    x, y = sp.symbols('x y')
    terms = []

    # Calcular las derivadas parciales de la función en el punto p
    derivadas = {(0, 0): f}
    for i in range(1, n + 1):
        for j in range(i + 1):
            derivada = sp.diff(derivadas.get((j - 1, i - j), 0), x) if j > 0 else sp.diff(derivadas.get((j, i - j - 1), 0), y)
            derivadas[(j, i - j)] = derivada

    # Construir y mostrar los términos de la serie de Taylor
    for i in range(n + 1):
        for j in range(i + 1):
            termino = derivadas[(j, i - j)].subs({x: p[0], y: p[1]}) * (x - p[0])**j * (y - p[1])**(i - j) / (sp.factorial(j) * sp.factorial(i - j))
            print(f"Término ({j}, {i - j}): {termino}")
            terms.append(termino)

    return terms

# Definir la función, el punto y el orden de la aproximación
x, y = sp.symbols('x y')
f = sp.sin(sp.pi*x-x**2*y)
p = (1, sp.pi)
n = 2

# Imprimir la serie de Taylor
print(f"Serie de Taylor para aproximar la función {f} alrededor del punto p = {p} en los {n} primeros terminos es:")
taylor_terms = taylor_series_2d(f, p, n)

# Imprimimos las derivadas de x e y

df_dx = sp.diff(f, x)
print("\n1ra Derivada de f(x, y) con respecto a x:")
print(simplify(df_dx))

df_dy = sp.diff(f, y)
print("\n1ra Derivada de f(x, y) con respecto a y:")
print(simplify(df_dy))

d2f_dx = sp.diff(df_dx, x)
print("\n2da Derivada de f(x, y) con respecto a x:")
print(simplify(d2f_dx))

d2f_dy = sp.diff(df_dy, y)
print("\n2da Derivada de f(x, y) con respecto a y:")
print(simplify(d2f_dy))

d2f_dxdy = sp.diff(f, x, y) 
print("\n2da Derivada mixta de f(x, y) con respecto a x y:")
print(simplify(d2f_dxdy))

# Evaluar las primeras derivadas en el punto (x, y)

df_dx_eval = df_dx.subs({x: p[0], y: p[1]})
df_dy_eval = df_dy.subs({x: p[0], y: p[1]})

# Construir la matriz Jacobiana
Jf = sp.Matrix([[df_dx_eval], [df_dy_eval]])

print("\nMatriz Jacobiana Jf:")
print(Jf)

# Evaluar las segundas derivadas en el punto (x, y)

d2f_dx_eval = d2f_dx.subs({x: p[0], y: p[1]})
d2f_dy_eval = d2f_dy.subs({x: p[0], y: p[1]})
d2f_dxdy_eval = d2f_dxdy.subs({x: p[0], y: p[1]})

# Construir la matriz Hessiana
Hf = sp.Matrix([[d2f_dx_eval, d2f_dxdy_eval],
             [d2f_dxdy_eval, d2f_dy_eval]])

print("\nMatriz Hessiana Hf:")
print(Hf)


"""
Este codigo implementa el método de Newton-Raphson para encontrar las raíces (o ceros) de una función real. Debe ingresar la funcion f(x) y el valor inicial x1 para ver los resultados intermedios de la iteracion. Inicialmente se imprimiran las racices, la derivada de f(x) y la tabla de resultados intermedios de la iteracciones con un valor inicial x1. Finalmente se graficara la funcion f(x). La tabla impresa muestra en cada linea el valor de x, f(x) y f'(x) en cada iteracion. La primera linea corresponde al valor de f(x) y f'(x) a partir del valor inicial x1 introducido. La segunda linea corresponde al valor de f(x) y f'(x) obtenido de sustituir con el valor de x de la fila inmediatamente anterior y asi sucesivamente.
"""
import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def newton_raphson(f, x0, tol=1e-6, max_iter=100):
    x = sp.Symbol('x')
    df = sp.diff(f, x)
    f_func = sp.lambdify(x, f)
    df_func = sp.lambdify(x, df)
    
    iteration = 0
    x_prev = x0
    fx = f_func(x_prev)
    dfx = df_func(x_prev)
    while iteration < max_iter and abs(fx) >= tol:
        if dfx == 0:
            break
        x_next = x_prev - fx / dfx
        if abs(x_next - x_prev) < tol:
            return x_next, iteration
        x_prev = x_next
        fx = f_func(x_prev)
        dfx = df_func(x_prev)
        iteration += 1
    
    return None, iteration

x = sp.Symbol('x')
f = (x**6/6) - 3*x**4 - (2*x**3/3) + (27*x**2/2) + 18*x - 30

roots = []
for x0 in [-10, -5, 0, 5, 10]:
    root, iteration = newton_raphson(f, x0)
    if root is not None:
        roots.append(root)

if roots:
    print("Se encontraron las siguientes raíces:")
    for root in roots:
        print("Raíz:", root)
else:
    print("No se encontraron raíces en el intervalo dado.")

diff_x = sp.diff(f, x)
print("La derivada de f(x) es:", diff_x)

def f_val(x_val):
    return f.subs(x, x_val)

def d_f_val(x_val):
    return diff_x.subs(x, x_val)

x1 = 1.0 # Introduzca el valor inicial, al menos con 1 decimal. Ejemplo: 1.0

d = {"x" : [], "f(x)": [], "f'(x)": []}
for i in range(0, 20):
    f_val_x1 = f_val(x1)
    d_f_val_x1 = d_f_val(x1)
    x1 = x1 - f_val_x1 / d_f_val_x1
    d["x"].append(x1)
    d["f(x)"].append(f_val_x1)
    d["f'(x)"].append(d_f_val_x1)

df = pd.DataFrame(d, columns=['x', 'f(x)', "f'(x)"])
print(df)

# Convertir la función a una función numérica
f_lambdified = sp.lambdify(x, f, "numpy")

# Crear un array de valores x
x_vals = np.linspace(-10, 10, 400)

# Calcular los valores y correspondientes
y_vals = f_lambdified(x_vals)

# Crear la gráfica
plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label=str(f))

# Establecer los límites de los ejes x e y
plt.xlim([-10, 10])
plt.ylim([-100, 100])

plt.title('Gráfica de la función')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.legend()
plt.show()


"""
Encontrar el mínimo de la función f(x, y) por el método de descenso de gradiente. El codigo abajo utiliza el codigo de descenso de gradiente para minimizar una funcion de dos variables. Grafica la funcion con la proyeccion del mapa de líneas de nivel en el plano xy y en otra grafica, en la misma vista, traza el gradiente perpendicular a las lineas de nivel en cualquier punto del espacio.
"""
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sympy import symbols, diff, lambdify
import sympy as sp

# Definir los símbolos y la función
x, y = symbols('x y')
f = x*sp.exp(-x**2 - y**2)

# Crear funciones numéricas para f y su gradiente
f_func = lambdify((x, y), f, "numpy")
grad_f_func = lambdify((x, y), [diff(f, x), diff(f, y)], "numpy")

# Parámetros de optimización
learning_rate = 0.1
num_iterations = 1000
initial_point = np.array([0.0, 0.0])

# Descenso de gradiente
current_point = initial_point.copy()
for _ in range(num_iterations):
    gradient = grad_f_func(current_point[0], current_point[1])
    current_point -= learning_rate * np.array(gradient)

# Resultado
minimum = round(f_func(current_point[0], current_point[1]), 1)
print("El mínimo de la función es:", minimum)
print("El punto de mínimo es:", [round(i, 1) for i in current_point])

df_x = diff(f, x)
df_y = diff(f, y)
print(f'La primera derivada con respeto a x de la funcion {f} es, \n{df_x}')
print(f'La primera derivada con respeto a y de la funcion {f} es, \n{df_y}')

# Calcular el gradiente en una cuadrícula de puntos
x_vals_grad = np.linspace(-2, 2, 20)
y_vals_grad = np.linspace(-2, 2, 20)
x_mesh_grad, y_mesh_grad = np.meshgrid(x_vals_grad, y_vals_grad)
gradient_x, gradient_y = grad_f_func(x_mesh_grad, y_mesh_grad)

# Crear una cuadrícula de puntos en el espacio 3D
x_vals = np.linspace(-2, 2, 100)
y_vals = np.linspace(-2, 2, 100)
x_mesh, y_mesh = np.meshgrid(x_vals, y_vals)
z = f_func(x_mesh, y_mesh)
z_min, z_max = np.min(z), np.max(z)

# Crear una nueva figura
fig = plt.figure(figsize=(12, 6))

# Agregar la primera subgráfica (gráfica 3D)
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(x_mesh, y_mesh, z, cmap='viridis')
ax1.contour(x_mesh, y_mesh, z, zdir='z', offset=z_min, levels=np.linspace(z_min, z_max, 20), cmap='viridis')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('f(x, y)')
ax1.set_title(f'Gráfica de {f}')
ax1.set_xticks(np.arange(-2, 2.5, 0.5))  
ax1.set_yticks(np.arange(-2, 2.5, 0.5))  
ax1.set_zticks(np.linspace(z_min, z_max, 5))

# Agregar la segunda subgráfica (gráfico del gradiente)
ax2 = fig.add_subplot(122)
contour = ax2.contour(x_mesh_grad, y_mesh_grad, f_func(x_mesh_grad, y_mesh_grad), cmap='viridis')
ax2.quiver(x_mesh_grad, y_mesh_grad, gradient_x, gradient_y, color='r')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('Gráfico del gradiente perpendicular a las líneas de nivel')

# Mostrar la gráfica
plt.tight_layout()
plt.show()


"""
Encontrar los puntos críticos (mínimos y máximos) de una función f(x, y) = f, sujeta a una restricción g(x, y) = g, utilizando el método de los multiplicadores de Lagrange, el cual considera la siguiente solucion df/dx = λ*dg/dx; df/dy = λ*dg/dy; donde λ es el multiplicador de Lagrange, df/dx y df/dy son las derivadas parciales de f(x, y) con respecto a x e y, dg/dx y dg/dy son las derivadas parciales de g(x, y) con respecto a x e y. El código abajo implementa el método de los multiplicadores de Lagrange para encontrar los puntos críticos de una función f(x, y) sujeta a una restricción g(x, y) = 0. La repuesta se observa en forma simbolica
"""
import sympy as sp

# Definir variables simbólicas
x, y, a, l = sp.symbols('x y a lambda')

# Definir la función y la restricción
f = x**2 * y   # Funcion
g = x**2 + y**2 - a**2  # Restriccion

# Definir la función Lagrangiana
L = f - l * g

# Resolver el sistema de ecuaciones df/dx = λ * dg/dx, df/dy = λ * dg/dy y la restricción g(x, y) = 0
solutions = sp.solve([sp.diff(L, x), sp.diff(L, y), sp.diff(L, l), g], (x, y, l, a))

# Imprimir los resultados
for sol in solutions:
    print("Punto crítico: (x =", sol[0], ", y =", sol[1], "), lambda =", sol[2], ", a =", sol[3])
    print("Valor de f en el punto crítico:", f.subs({x: sol[0], y: sol[1]}))
    print()
    
"""
Este código utiliza el método de multiplicadores de Lagrange para encontrar los puntos críticos de un sistema de ecuaciones no lineales de la función f(x, y) = f, sujeta a una restricción g(x, y) = g. Para cada punto de inicio, utiliza `scipy.optimize.root` para encontrar un punto donde el gradiente del Lagrangiano es cero. Vamos a encontrar los ceros de una ecuacion vectorial 3D en un sistema de ecuaciones no lineales con una restricción. La ecuacion vectorial seria ∇L(x,y,λ) = ([[df/dx, - λ dg/dx], [df/dy , λ dg/dy], [-g(x)]]). El código imprime las coordenadas de cada punto crítico, el valor de la función `f` en ese punto, y el valor de `λ` (el multiplicador de Lagrange) en ese punto. La repuesta se observa en forma numerica.
"""
import numpy as np
import sympy as sp
from scipy import optimize

x, y = sp.symbols('x y')  # Use sp.symbols to create multiple symbols
f = -sp.exp(x-y**2+x*y)
g = sp.cosh(y) + x - 2
dfdx = sp.diff(f, x)
dfdy = sp.diff(f, y)
dgdx = sp.diff(g, x)
dgdy = sp.diff(g, y)

def DL (xyλ) :
    [x_val, y_val, λ] = xyλ
    return np.array([
            float(dfdx.subs({'x': x_val, 'y': y_val}) - λ * dgdx.subs({'x': x_val, 'y': y_val})),
            float(dfdy.subs({'x': x_val, 'y': y_val}) - λ * dgdy.subs({'x': x_val, 'y': y_val})),
            float(- g.subs({'x': x_val, 'y': y_val}))
        ])

start_points = [[-1, -1, 0], [1, 1, 0], [0, 0, 0], [2, 2, 0], [-2, -2, 0]]

for i, start in enumerate(start_points):
    x_val, y_val, λ = optimize.root(DL, start).x
    print(f"Raíz {i+1}:")
    print(f"x = {x_val}")
    print(f"y = {y_val}")
    print(f"λ = {λ}")
    print(f"f(x, y) = {f.subs({'x': x_val, 'y': y_val})}")
    print()
    

"""
El codigo realiza una regresión lineal en un conjunto de datos y calcula varias estadísticas relacionadas con la regresión. Primero, se realiza la regresión lineal utilizando `scipy.stats.linregress`, Luego, se calcula la pendiente, la intersección, el coeficiente de correlación, el valor p, el error estándar y el error cuadrático.
"""

import numpy as np
from scipy import stats

  # Example data
xdat = np.array([0.4, 0.5, 0.6, 0.7, 0.8])
ydat = np.array([0.1, 0.25, 0.55, 0.75, 0.85])

# Realizar la regresión lineal
slope, intercept, r_value, p_value, std_err = stats.linregress(xdat, ydat)

# Calcule error de la funcion
X2 = np.sum((ydat - slope * xdat - intercept)**2)

# Mostrar los resultados
print("Pendiente (m):", slope)
print("Intercepto (c):", intercept)
print("Coeficiente de correlación:", r_value)
print("Valor p:", p_value)
print("Error estándar:", std_err)
print(f"El error cuadrático X^2 = {X2}")


"""
Ajuste de la distribución de datos de altura

Instrucciones

En esta evaluación, escribirás código para realizar un descenso por el gradiente más pronunciado para ajustar un modelo gaussiano a la distribución de datos de altura que se introdujo por primera vez en Matemáticas para el aprendizaje automático: Álgebra lineal.

El algoritmo es el mismo que encontraste en Descenso por el gradiente en un arenero, pero esta vez, en lugar de descender una función predefinida, descenderemos la función Chi-cuadrado (X^2), que es a la vez una función de los parámetros que queremos optimizar y de los datos a los que se ajusta el modelo.

Cómo enviar la tarea

Completa todas las tareas que se te piden en la hoja de trabajo. Cuando hayas terminado y estés satisfecho con tu código, presiona el botón "Enviar tarea" en la parte superior de este cuaderno.

Comienza

Ejecuta la celda siguiente para cargar las dependencias y generar la primera figura en esta hoja de trabajo.

Si tenemos datos de las alturas de las personas en una población, se pueden representar en un histograma, es decir, un gráfico de barras donde cada barra tiene un ancho que representa un rango de alturas y un área que es la probabilidad de encontrar a una persona con una altura en ese rango. Podemos buscar modelar esos datos con una función, como una Gaussiana, que podemos especificar con dos parámetros, en lugar de almacenar todos los datos en el histograma.

La función Gaussiana se da como:

f(x, mu, sig) = (1 / (sig * sqrt(2*pi))) * exp(- (x - mu)**2 / (2 * sig**2))

La primera figura impresa muestra los datos en naranja, el modelo en magenta y la zona donde se superponen en verde. Este modelo en particular no se ha ajustado bien, ya que no hay una superposición adecuada.

Recuerda de los vídeos la definición de x^2 como la diferencia al cuadrado entre los datos y el modelo, es decir:

X^2 = [y - f(x, mu, sig)]^2

Esto se representa en la figura como la suma de los cuadrados de las barras rosadas y naranjas.

No olvides que x e y se representan como vectores aquí, ya que son listas de todos los puntos de datos, el 
|abs-squared|^2 codifica el cuadrado y la suma de los residuales en cada barra.

Para mejorar el ajuste, queremos alterar los parámetros mu y sig y ver cómo eso cambia el X^2. Es decir, necesitaremos calcular el Jacobiano:

J = [d(x^2)/dmu, d(x^2)/dsig]

Veamos el primer término, d(x^2)/dmu, usando la regla de la cadena multivariante, se puede escribir como:

d(x^2)/dmu = -2(y - f(x, mu, sig)) * df/dmu (x, mu, sig)

Existe una expresión similar para d(x^2)/dsig; intenta resolver esta expresión por tu cuenta.

Los Jacobianos dependen de las derivadas df/du y df/dsig. Escribe funciones a continuación para ellas.
"""

# Run this cell first to load the dependancies for this assessment,
# and generate the first figure.
from readonly.HeightsModule import *

# PACKAGE
import matplotlib.pyplot as plt
import numpy as np

# GRADED FUNCTION

# This is the Gaussian function.
def f (x,mu,sig) :
    return np.exp(-(x-mu)**2/(2*sig**2)) / np.sqrt(2*np.pi) / sig

# Next up, the derivative with respect to μ.
# If you wish, you may want to express this as f(x, mu, sig) multiplied by chain rule terms.
# === COMPLETE THIS FUNCTION ===
def dfdmu (x,mu,sig) :
    return f(x, mu, sig) * (x - mu) / (sig ** 2)

# Finally in this cell, the derivative with respect to σ.
# === COMPLETE THIS FUNCTION ===
def dfdsig (x,mu,sig) :
    return f(x, mu, sig) * (-1 / sig + ((x - mu) ** 2) / sig ** 3 )

"""
A continuación, recuerda que el descenso más pronunciado se moverá en el espacio de parámetros proporcional al negativo del Jacobiano, es decir:

[[δμ], [δσ]] ∝ -J

Siendo la proporcionalidad la "agresividad" del algoritmo.

Modifica la función a continuación para incluir el término d(x^2)/dsig del Jacobiano, el término d(x^2)/dmu ya está incluido para ti.
"""

# GRADED FUNCTION

# Complete the expression for the Jacobian, the first term is done for you.
# Implement the second.
# === COMPLETE THIS FUNCTION ===
def steepest_step (x, y, mu, sig, aggression) :
    J = np.array([
        -2*(y - f(x,mu,sig)) @ dfdmu(x,mu,sig),
        -2*(y - f(x,mu,sig)) @ dfdsig(x,mu,sig) # Replace the ??? with the second element of the Jacobian.
    ])
    step = -J * aggression
    return step

"""
Prueba tu código antes de enviarlo

Para probar el código que has escrito anteriormente, ejecuta todas las celdas anteriores (selecciona cada celda y luego presiona el botón de reproducción [ ▶| ] o presiona shift-enter). Luego puedes usar el código a continuación para probar tu función. No es necesario que envíes estas celdas; puedes editarlas y ejecutarlas tantas veces como quieras.
"""
# First get the heights data, ranges and frequencies
x,y = heights_data()

# Next we'll assign trial values for these.
mu = 155 ; sig = 6
# We'll keep a track of these so we can plot their evolution.
p = np.array([[mu, sig]])

# Plot the histogram for our parameter guess
histogram(f, [mu, sig])
# Do a few rounds of steepest descent.
for i in range(50) :
    dmu, dsig = steepest_step(x, y, mu, sig, 2000)
    mu += dmu
    sig += dsig
    p = np.append(p, [[mu,sig]], axis=0)

# Calcular X^2 y f(x, mu, sig)
X_squared = (y - f(x, mu, sig))**2
f_value = f(x, mu, sig)

# Imprimir los resultados
print("X^2 = ", X_squared)
print("f(x, mu, sig) = ", f_value)

# Plot the path through parameter space.
contour(f, p)
# Plot the final histogram.
histogram(f, [mu, sig])


"""
Este codigo realiza una funcion igual al anterior, utilizando la optimización de mínimos cuadrados no lineales para ajustar una distribición normal a un conjunto de datos de altura. El código imprime los parámetros ajustados, el valor de X^2 y f(x, mu, sig) en cada iteración. Finalmente, traza el histograma final.
"""

from readonly.HeightsModule import *
from scipy.stats import norm
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

# Obtener los datos de altura
x, y = heights_data()

# Definir la función de densidad de probabilidad de una distribución normal
def f(x, mu, sig):
    return norm.pdf(x, mu, sig)

# Ajustar la distribución normal a los datos
params, _ = curve_fit(f, x, y, p0=[178, 6])

# Imprimir los parámetros ajustados
print("mu = ", params[0])
print("sig = ", params[1])

# Calcular X^2 y f(x, mu, sig)
X_squared = (y - f(x, *params))**2
f_value = f(x, *params)

# Imprimir los resultados
print("X^2 = ", X_squared)
print("f(x, mu, sig) = ", f_value)

# Trazar el histograma final
histogram(f, params)


"""
Dada la matriz A, calcular la matriz de covarianza y sus valores y vectores propios.
"""

import numpy as np

# The dataset to be plotted
A = np.array([[1,1],
                [-3,-3],
                [2,2],
                [0,2],
                [-2,0],
                [1,3],
                [4,4]])

# Define the covariance matrix
cov_matrix = np.cov(A, rowvar=False)

# Calculate the eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Form the diagonal matrix of eigenvalues
D = np.diag(eigenvalues)

print("Covariance matrix:")
print(cov_matrix)
print("\nEigenvalues:")
print(eigenvalues)
print("\nDiagonal matrix D of eigenvalues:")
print(D)


"""
Dada una matriz, determinar varianza y covarianza con y sin sesgo.
"""

import numpy as np
import matplotlib.pyplot as plt

# Datos
data = np.array([[3,2],[2,4]])
# data *= 2
# data += [2, 2]


# Separar x e y
x = data[:, 0]
y = data[:, 1]

# Calcular media de x e y
mean_x = np.mean(x)
mean_y = np.mean(y)

# Calcular varianza de x sin sesgo cov (x) (1/(n-1) ∑ (xi - xp)^2)
var_x_unbiased = np.var(x, ddof=1)

# Calcular varianza de y sin sesgo cov (y) (1/(n-1) ∑ (yi - yp)^2)
var_y_unbiased = np.var(y, ddof=1)

# Calcular varianza de x con sesgo cov (x) (1/n) ∑ (xi - xp)^2)
var_x_biased = np.var(x)

# Calcular varianza de y con sesgo cov (y) (1/n) ∑ (yi - yp)^2)
var_y_biased = np.var(y)


# Calcular covarianza de x e y sin sesgo cov (x, y) (1/(n-1) ∑ (xi - xp) (yi - yp)
cov_xyunbiased = np.cov(x, y)[0, 1]

# Calcular covarianza de x e y sin sesgo cov (x, y) (1/n) ∑ (xi - xp) (yi - yp)
cov_xy_biased = np.cov(x, y, bias=True)[0, 1]

# Calcular covarianza de x e y sin sesgo cov (x, y) (1/(n-1) ∑ (xi - xp) (yi - yp)
cov_yxunbiased = np.cov(y, x)[0, 1]

# Calcular covarianza de x e y sin sesgo cov (x, y) (1/n) ∑ (xi - xp) (yi - yp)
cov_yxbiased = np.cov(y, x, bias=True)[0, 1]

print("Media de x:", mean_x)
print("Media de y:", mean_y)
print("Varianza sin sesgo (n-1) x:", var_x_unbiased)
print("Varianza sin sesgo (n-1) y:", var_y_unbiased)
print("Varianza con sesgo (n) x:", var_x_biased)
print("Varianza con sesgo (n) y:", var_y_biased)
print("Covarianza sin sesgo (x, y) (n-1):", cov_xyunbiased)
print("Covarianza con sesgo (x, y) (n):", cov_xy_biased)
print("Covarianza sin sesgo (y, x) (n-1):", cov_yxunbiased)
print("Covarianza con sesgo (y, x) (n):", cov_yxbiased)

matriz = np.array([[var_x_unbiased, cov_xyunbiased], [cov_yxunbiased, var_y_unbiased]])
matriz1 = np.array([[var_x_biased, cov_xy_biased], [cov_yxbiased, var_y_biased]])
print("Matriz de covarianza sin sesgo (n-1):", matriz)
print("Matriz de covarianza con sesgo (n):", matriz1)


"""
Dada la matriz de covarianza con sesgo, determinar la matriz base que genera los datos con la matriz de covarianza objetivo. 
"""

import numpy as np

# Matriz de covarianza objetivo
cov_target = np.array([[1, 0.8], [0.8, 1]], dtype=np.float64)

# Añadir una pequeña cantidad a la diagonal para asegurarte de que la matriz es definida positiva
cov_target += np.eye(*cov_target.shape) * 1e-6

# Descomposición de Cholesky
L = np.linalg.cholesky(cov_target)

# Generar datos aleatorios
n_samples = 1000
X = np.random.normal(size=(n_samples, 2))

# Obtener el conjunto de datos que tiene la matriz de covarianza objetivo
data = np.dot(X, L.T)

# Verificar la matriz de covarianza de los datos generados
cov_data = np.cov(data, rowvar=False)
print(cov_data)


"""
Diagonalizar una matriz
En concreto, se le dará una matriz diagonalizable 𝐴 y tendrá que encontrar las matrices S y D tales que:  A=SDS^-1
Recuerda que para hacer esto, primero debes encontrar todos los valores y vectores propios de 𝐴. Entonces, 𝑆 es la matriz de todos los vectores propios dispuestos como columnas, y 𝐷 es la matriz de los valores propios correspondientes dispuestos a lo largo de la diagonal. 
A efectos de prueba, cada vector propio en S debe ser de longitud unitaria. Este será siempre el caso si utiliza np.linalg.eig(). Sin embargo, si no utiliza esta función, dependiendo de su implementación, puede que tenga que normalizar los vectores propios. Además, los valores propios deben aparecer en orden no decreciente.

Traducción realizada con la versión gratuita del traductor DeepL.com
"""

import numpy as np

def diagonalize(A):
   
    # Obtener los valores y vectores propios de A
    eig_vals, S = np.linalg.eig(A)

    # Crear la matriz D con los valores propios en la diagonal
    D = np.diag(eig_vals)

    # Calcular la inversa de S
    S_inv = np.linalg.inv(S)

    return S, D, S_inv
"""
# Test the function
A = np.array([[4, -9, 6, 12],
              [0, -1, 4, 6],
              [2, -11, 8, 16],
              [-1, 3, 0, -1]])
S, D, S_inv = diagonalize(A)
print("S:", S)
print("D:", D)
print("S_inv:", S_inv)
"""
A = np.array([[1, 5],
              [2, 4]])
S_exp = np.array([[-0.92847669, -0.70710678],
                  [ 0.37139068, -0.70710678]])
D_exp = np.array([[-1, 0],
                  [0, 6]])
S_inv_exp = np.array([[-0.76930926,  0.76930926],
                      [-0.40406102, -1.01015254]])

S, D, S_inv = diagonalize(A)
np.testing.assert_allclose(S_exp, S, rtol=1e-5, atol=1e-10)
np.testing.assert_allclose(D_exp, D, rtol=1e-5, atol=1e-10)
np.testing.assert_allclose(S_inv_exp, S_inv, rtol=1e-5, atol=1e-10)

A = np.array([[4, -9, 6, 12],
              [0, -1, 4, 6],
              [2, -11, 8, 16],
              [-1, 3, 0, -1]])
S_exp = np.array([[-5.00000000e-01, 8.01783726e-01, 9.04534034e-01, 3.77964473e-01],
                  [-5.00000000e-01, 5.34522484e-01, 3.01511345e-01, 7.55928946e-01],
                  [5.00000000e-01, -7.95591412e-15, 3.01511345e-01, 3.77964473e-01],
                  [-5.00000000e-01, 2.67261242e-01, -2.21106380e-15, 3.77964473e-01]])

D_exp = np.array([[1, 0, 0, 0],
                  [0, 2, 0, 0],
                  [0, 0, 3, 0],
                  [0, 0, 0, 4]])
S_inv_exp = np.array([[-2.00000000e+00, 1.00000000e+01, -4.00000000e+00, -1.40000000e+01],
                      [-3.74165739e+00, 2.24499443e+01, -1.12249722e+01, -2.99332591e+01],
                      [3.31662479e+00, -1.32664992e+01, 6.63324958e+00, 1.65831240e+01],
                      [-8.81212207e-16, -2.64575131e+00, 2.64575131e+00, 5.29150262e+00]])

S, D, S_inv = diagonalize(A)
np.testing.assert_allclose(S_exp, S, rtol=1e-5, atol=1e-10)
np.testing.assert_allclose(D_exp, D, rtol=1e-5, atol=1e-10)
np.testing.assert_allclose(S_inv_exp, S_inv, rtol=1e-5, atol=1e-10)

print("All tests passed!")


"""
En esta función, implementarás la multiplicación de polinomios. Se te darán dos matrices numpy unidimensionales A y B. los coeficientes de los dos polinomios, donde ai es el coeficiente de xi en A. Debes calcular los coeficientes de A B.
Más formalmente, si C es la matriz unidimensional resultante, entonces ci = ∑ aj * bk.
A y B pueden tener tamaños diferentes.
"""

import numpy as np

def multiply_polynomials(A, B):
    
    C = np.convolve(A, B)

    return C

A = np.array([1, 2])
B = np.array([3, 4])
C_exp = np.array([3, 10, 8])
np.testing.assert_allclose(multiply_polynomials(A, B), C_exp, rtol=1e-5, atol=1e-10)

A = np.array([5, 6])
B = np.array([1, 3, 5, 9])
C_exp = np.array([5, 21, 43, 75, 54])
np.testing.assert_allclose(multiply_polynomials(A, B), C_exp, rtol=1e-5, atol=1e-10)
np.testing.assert_allclose(multiply_polynomials(B, A), C_exp, rtol=1e-5, atol=1e-10)

print("All tests passed!")


"""
Calcula la media de las columnas y la compara con la media de las filas de una matriz.
"""

import numpy as np
from numpy.testing import assert_allclose 

def mean_naive(X):
    """Calcula la media muestral para un conjunto de datos iterando sobre el mismo.

Argumentos:

X: ndarray de forma (N, D) que representa el conjunto de datos. N es el tamaño del conjunto de datos (el número de puntos de datos) y D es la dimensionalidad de cada punto de datos.

Devuelve:
media: ndarray de forma (D,), la media muestral del conjunto de datos X.
    """
    "Compute the mean for a dataset X by iterating over the data points"
    # X is of size (D,N) where D is the dimensionality and N the number of data points
    D, N = X.shape
    mean = np.zeros(N)  # Initialize mean as a 1D array
    for d in range(D):
        mean += X[d, :]  # Add each data point to mean
    mean /= D  # Divide by the number of dimensions
    return mean.reshape(1, -1)  # Reshape the result to a 2D array with a single row


# Test case 1
X = np.array([[0, 1, 1], 
              [1, 2, 1]])
expected_mean = np.array([0.5, 1.5, 1.]).reshape(1, -1)
assert_allclose(mean_naive(X), expected_mean, rtol=1e-5)

# Test case 2
X = np.array([[0, 1, 0], 
              [2, 3, 1]])
expected_mean = np.array([1, 2, 0.5]).reshape(1, -1)
assert_allclose(mean_naive(X), expected_mean, rtol=1e-5)

# Test covariance is zero
X = np.array([[0, 1], 
              [0, 1]])
expected_mean = np.array([0., 1.]).reshape(1, -1)
assert_allclose(mean_naive(X), expected_mean, rtol=1e-5)

print("All tests passed!")

# Otra froma de hacerlo es:

import numpy as np
from numpy.testing import assert_allclose

def mean(X):

    """
    Calcula la media muestral de un conjunto de datos iterando sobre el mismo.

  Argumentos:
    X: `ndarray` de forma (N, D) que representa el conjunto de datos. N 
      es el tamaño del conjunto de datos (el número de puntos de datos) 
      y D es la dimensionalidad de cada punto de datos.

  Devuelve:
    `ndarray`: `ndarray` con forma (D,), la media muestral del conjunto de datos `X`.
    """
    # Inicializamos la media
    N, D = X.shape
    m = np.zeros((D,))
    for i in range(N):
       # Sumamos cada punto de datos a la variable `m`
       m += X[i]
    # Dividimos por el número de puntos de datos
    m /= N
    return m

# Test case 1
X = np.array([[0., 1., 1.], 
              [1., 2., 1.]])
expected_mean = np.array([0.5, 1.5, 1.])
assert_allclose(mean(X), expected_mean, rtol=1e-5)

# Test case 2
X = np.array([[0., 1., 0.], 
              [2., 3., 1.]])
expected_mean = np.array([1., 2., 0.5])
assert_allclose(mean(X), expected_mean, rtol=1e-5)

# Test covariance is zero
X = np.array([[0., 1.], 
              [0., 1.]])
expected_mean = np.array([0., 1.])
assert_allclose(mean(X), expected_mean, rtol=1e-5)

print("All tests passed!")

# Otra forma mas optimizada de hacerlo es:

import numpy as np
from numpy.testing import assert_allclose 

def mean_optimized(X):
   
    return np.mean(X, axis=0)

# Test case 1
X = np.array([[0, 1, 1], 
              [1, 2, 1]])
expected_mean = np.array([0.5, 1.5, 1.])
assert_allclose(mean_optimized(X), expected_mean, rtol=1e-5)

# Test case 2
X = np.array([[0, 1, 0], 
              [2, 3, 1]])
expected_mean = np.array([1, 2, 0.5])
assert_allclose(mean_optimized(X), expected_mean, rtol=1e-5)

# Test covariance is zero
X = np.array([[0, 1], 
              [0, 1]])
expected_mean = np.array([0., 1.])
assert_allclose(mean_optimized(X), expected_mean, rtol=1e-5)

print("All tests passed!")

# Otra froma de hac4erlo es:

import numpy as np
from numpy.testing import assert_allclose

def cov(X):
  
  N, D = X.shape  
  mean = np.mean(X, axis=0)
  X_centered = X - mean
  covariance = np.dot(X_centered.T, X_centered) / (N)
  return covariance

# Test case 1
X = np.array([[0., 1.], 
              [1., 2.],
     [0., 1.], 
     [1., 2.]])
expected_cov = np.array(
    [[0.25, 0.25],
    [0.25, 0.25]])

assert_allclose(cov(X), expected_cov, rtol=1e-5)

# Test case 2
X = np.array([[0., 1.], 
              [2., 3.]])
expected_cov = np.array(
    [[1., 1.],
    [1., 1.]])

assert_allclose(cov(X), expected_cov, rtol=1e-5)

# Test covariance is zero
X = np.array([[0., 1.], 
              [0., 1.],
              [0., 1.]])
expected_cov = np.zeros((2, 2))

assert_allclose(cov(X), expected_cov, rtol=1e-5)

print("All tests passed!")

# Otra forma mucho mas optimizada de hacerlo es:

import numpy as np
from numpy.testing import assert_allclose

def cov(X):
    """Calcula la matriz de covarianza de un conjunto de datos usando la función np.cov de NumPy.

    Argumentos:
        X: ndarray de forma (N, D) que representa el conjunto de datos. N es el tamaño del conjunto de datos (el número de puntos de datos) y D es la dimensionalidad de cada punto de datos.
    Devuelve:
        covariance: ndarray de forma (D, D), la matriz de covarianza del conjunto de datos X.
    """
    return np.cov(X, rowvar=False, bias=True)

# Test case 1
X = np.array([[0., 1.], 
              [1., 2.],
     [0., 1.], 
     [1., 2.]])
expected_cov = np.array(
    [[0.25, 0.25],
    [0.25, 0.25]])

assert_allclose(cov(X), expected_cov, rtol=1e-5)

# Test case 2
X = np.array([[0., 1.], 
              [2., 3.]])
expected_cov = np.array(
    [[1., 1.],
    [1., 1.]])

assert_allclose(cov(X), expected_cov, rtol=1e-5)

# Test covariance is zero
X = np.array([[0., 1.], 
              [0., 1.],
              [0., 1.]])
expected_cov = np.zeros((2, 2))

assert_allclose(cov(X), expected_cov, rtol=1e-5)

print("All tests passed!")


"""
Una transformación afín es una operación lineal que incluye una combinación de rotación, escala, traslación y cizallamiento (deformación en una dirección). En el contexto de conjuntos de datos, la transformación afín se aplica a cada punto del conjunto de datos, modificando su posición en el espacio.

Media de la Transformación Afín:
La media de la transformación afín de un conjunto de datos puede ser calculada aplicando la transformación afín directamente al vector de medias. Si tenemos un conjunto de datos representado por la matriz 
X de tamaño (N,D), y queremos saber cuál es la media después de aplicar la transformación afín Axi+b para cada punto de datos xi, simplemente calculamos Aμ+b, donde μ es el vector de medias original de X.
En resumen, si μ es el vector de medias original, entonces la media después de la transformación afín será Aμ+b.

Covarianza de la Transformación Afín:
La covarianza de la transformación afín de un conjunto de datos es un poco más complicada de calcular. Si tenemos una matriz de datos X de tamaño (N,D) y queremos saber cuál es la covarianza cuando aplicamos la transformación afín Axi+b para cada punto de datos xi, necesitamos considerar cómo la covarianza se transforma bajo una transformación lineal.
La covarianza de la transformación afín AX+b puede ser calculada usando la siguiente fórmula:
Cov(AX+b)=ACov(X)A^T
Donde Cov(X) es la matriz de covarianza original de X. Esto se debe a que la covarianza de una variable aleatoria transformada linealmente es afectada por la transformación de manera específica, en este caso, multiplicando por A y su transpuesta A^T.
Es importante notar que la covarianza transformada nos dará información sobre cómo se relacionan las variables después de la transformación afín, teniendo en cuenta cómo la transformación afecta tanto la escala como la orientación de los datos.

En resumen, al aplicar una transformación afín Axi+b a un conjunto de datos X, la media se transforma a Aμ+b, mientras que la covarianza se transforma a ACov(X)A^T. Estos cálculos son fundamentales para entender cómo se modifican las propiedades estadísticas de los datos bajo una transformación afín.
"""

import numpy as np
from numpy.testing import assert_allclose

def affine_mean(mean, A, b):
    
    """
  Calcula el vector medio después de una transformación afín.

  Argumentos:
    mean: `ndarray` de forma (D,), el vector de la media muestral para algún conjunto de datos.
    A: `ndarray` de forma (D, D), la matriz de transformación afín.
    b: `ndarray` de forma (D,), el vector de traslación afín.

  Devuelve:
    `ndarray`: vector de la media muestral de la forma (D,) después de la transformación afín.
  """
    # affine_m tiene forma (D,)
    affine_m = np.zeros(mean.shape)
    
    # Aplicar la transformación afín a la media original
    affine_m = np.dot(A, mean) + b

    return affine_m

def affine_covariance(S, A, b):
  """
  Calcula la matriz de covarianza después de la transformación afín.

  Argumentos:
    S: `ndarray` de forma (D, D), la matriz de covarianza de la muestra para algún conjunto de datos.
    A: `ndarray` de forma (D, D), la matriz de transformación afín.
    b: `ndarray` de forma (D,), el vector de traslación afín.

  Devuelve:
    `ndarray`: la matriz de covarianza muestral de la forma (D, D) después de la transformación afín.
  """
  # `affine_cov` tiene forma (D, D)
  affine_cov = np.zeros(S.shape)  

  # Aplicamos la transformación afín a la matriz de covarianza
  affine_cov = np.dot(A, np.dot(S, A.T))

  return affine_cov

A = np.array([[0, 1], [2, 3]])
b = np.ones(2)
m = np.full((2,), 2)
S = np.eye(2)*2

expected_affine_mean = np.array([ 3., 11.])
expected_affine_cov = np.array(
    [[ 2.,  6.],
    [ 6., 26.]])

assert_allclose(affine_mean(m, A, b), expected_affine_mean, rtol=1e-4)

print("All tests passed!")


"""
Este código verifica la definida positiva, el producto interior, la simetría y la bilinealidad, definido por la matriz `A` y los vectores `x`, `y`, `z`. Para la bilinealidad, verifica las dos propiedades de linealidad en cada argumento. Para la definición positiva, verifica que el producto interno de un vector consigo mismo es siempre no negativo, y que solo es cero si el vector es el vector cero. Para la simetría, verifica que el producto interno de `x` e `y` es igual al producto interno de `y` e `x`.
"""

import numpy as np

A = np.array([[1, 0], [0, 1]])
x = np.array([1,1])
y = np.array([2,-1])
z = np.array([2,2])
alpha = 2

def B(x, y):
    """Calcula la función B(x, y) = x^T * A * y = x^T * (A * y)."""
    global A
    return np.dot(x, np.dot(A, y))
 
def is_positive_definite(f, x):
    """Laa función es definida positiva si <x, x> >= 0."""
    return f(x, x) > 0

def is_inner_product(f, x, y, z, a):
    """La función es un producto interior si es definida positiva."""
    return is_positive_definite(f, x) and is_bilinear(f, x, y, z, a) and is_symmetric(f, x, y)

def is_symmetric(f, x, y):
    """La función es simétrica si <x, y> = <y, x>."""
    return np.allclose(f(x, y), f(y, x))

def is_bilinear(f, x, y, z, a):
    """La función es bilineal si <λx + z, y> = λ <x, y> + <z, y>; y si,
    <x, λy + z> = λ <x,y> + < λ, z>. Si es simetrica, solo se verifica una de las dos propiedades."""
    return np.allclose(f(a*x + y, z), a*f(x, z) + f(y, z)) and np.allclose(f(x, a*y + z), a*f(x, y) + f(x, z))


c = np.dot(A, y)
v = np.dot(x, np.dot(A, y))

print("La funcion es definida positiva: ", is_positive_definite(B, x))
print("La funcion tiene producto interior: ", is_inner_product(B, x, y, z, alpha))
print("La funcion es simetrica: ", is_symmetric(B, x, y))
print("La funcion es bilinear: ", is_bilinear(B, x, y, z, alpha))
print("El producto de (A * y): ",c)  
print("El producto de x^T * (A * y): ", v)


# Otra forma de hacerlo es:

import numpy as np

# Define the matrix A and vectors x, y, z
A = np.array([[2, 1], [1, 2]])
x = np.array([1, 2])
y = np.array([3, 4])
z = np.array([5, 6])
lambda_ = 1

# Definida positiva. Es un producto interior si es definida positiva
# <x, x> >= 0
print("Definición positiva cumple <x, x> >= 0: ", np.dot(x, x) >= 0)
# <x, x> = 0 implies x = 0
zero_vector = np.array([0, 0])
print("Definición positiva cumple <x, x> = 0 implies x = 0: ", np.dot(zero_vector, zero_vector) == 0)

# Simetría
# <x, y> = <y, x>
print("Simetría cumple <x, y> = <y, x>: ", np.isclose(np.dot(x, y), np.dot(y, x)))

# Bilinealidad
# <λx + z, y> =  λ <x, y> + <z, y>
lhs = np.dot(lambda_* x + z, y)
rhs = lambda_ * np.dot(x, y) + np.dot(z, y)
print("Bilinealidad cumple <λx + z, y> =  λ <x, y> + <z, y>: ", np.isclose(lhs, rhs))
# <x, λy + z> = λ <x,y> + < λ, z>
lhs = np.dot(x, lambda_*y + z)
rhs = lambda_ * np.dot(x, y) + lambda_ * np.dot(x, z)
print("Bilinealidad cumple <x, λy + z> = λ <x,y> + < λ, z>: ", np.isclose(lhs, rhs))


"""
Este código calcula la distancia entre `x` y `y` utilizando dos definiciones diferentes de producto interno: el producto punto estándar (que da lugar a la distancia euclidiana) y un nuevo producto interno definido por la matriz `A`.
"""
import numpy as np
import sympy as sp

# Define vectors x and y
x = np.array([1/2, -1, -1/2])
y = np.array([0, 1, 0])

# Compute the difference vector
diff = x - y
print("Difference vector: ", diff)

# Compute the Euclidean distance (using the dot product as the inner product)
euclidean_distance = sp.sqrt(np.dot(diff, diff))
print("Euclidean distance: ", euclidean_distance)

# Define a new inner product
A = np.array([[2, 1, 0], [1, 2, -1], [0, -1, 2]])

# Compute the distance using the new inner product
new_distance = sp.sqrt(np.dot(diff, np.dot(A, diff)))
print("Distance with new inner product: ", new_distance)


"""
Este codigo se utiliza para calcular el ángulo entre dos vectores `x` e `y` con respecto a una matriz `A`.
"""
import numpy as np
# La matriz A define el producto interior
A = np.array([[1, -1/2],[-1/2,5]])
x = np.array([0,-1])
y = np.array([1,1])

def find_angle(A, x, y):
    """Compute the angle"""
    inner_prod = x.T @ A @ y
    norm_x = np.sqrt(x.T @ A @ x)
    norm_y = np.sqrt(y.T @ A @ y)
    alpha = inner_prod/(norm_x*norm_y)
    angle = np.arccos(alpha)
    return np.round(angle,2) 

tranf_x = A @ x  # Transformación lineal de x con la matriz A
tranf_y = A @ y  # Transformación lineal de y con la matriz A

print("El ángulo en radianes entre x e y en la matriz A es: ", find_angle(A, x, y))
print("El angulo en grados entre x e y en la matriz A es: ", np.degrees(find_angle(A, x, y)))
print("La transformación de x es: ", tranf_x)
print("La transformación de y es: ", tranf_y)

# Otra forma de hacerlo es:

import numpy as np

def inner_product(A, x, y):
    """Compute the inner product of vectors 'x' and 'y' with respect to matrix 'A'"""
    return np.dot(x, np.dot(A, y))

def length(A, x):
    """Compute the length of a vector 'x' with respect to matrix 'A'"""
    return np.sqrt(inner_product(A, x, x))

def angle_between(A, x, y):
    """Compute the angle in degrees between vectors 'x' and 'y' with respect to matrix 'A'"""
    cos_theta = inner_product(A, x, y) / (length(A, x) * length(A, y))
    return np.arccos(np.clip(cos_theta, -1, 1)) # Convert the angle to degrees

A = np.array([[2, 0], [0, 1]])
x = np.array([1, 1])
y = np.array([-1, 1])

print("El ángulo en radianes entre x e y en la matriz A es: ", angle_between(A, x, y))
print("El angulo en grados entre x e y en la matriz A es: ", np.degrees(angle_between(A, x, y)))


# Algoritmo del vecino más próximo KNN

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

iris = datasets.load_iris()
X = iris.data[:, :2]  # use first two version for simplicity
y = iris.target

def pairwise_distance_matrix(X, Y):
    return np.sqrt(np.maximum(-2 * np.dot(X, Y.T) + np.sum(Y**2, axis=1) + np.sum(X**2, axis=1)[:, np.newaxis], 0))

def KNN(k, X, y, Xtest):
    num_classes = len(np.unique(y))
    idx = np.argsort(pairwise_distance_matrix(X, Xtest).T, axis=1)[:, :k]
    ypred = np.zeros((Xtest.shape[0], num_classes))
    for m in range(Xtest.shape[0]):
        klasses = y[idx[m]]    
        for k in np.unique(klasses):
            ypred[m, k] = len(klasses[klasses == k])  # Count occurrences of each class
    return np.argmax(ypred, axis=1)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

K = 3  # Define K, you can change this value depending on your needs

ypred = [KNN(K, X, y, xtest.reshape(1, -1)) for xtest in np.array([xx.ravel(), yy.ravel()]).T]

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

plt.figure()
plt.pcolormesh(xx, yy, np.array(ypred).reshape(xx.shape), cmap=cmap_light)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class classification (k = %i)" % K)
plt.show()

"""
Proyección ortogonal

La expresion πu(x) = b * ((x·b)/(||b||^2)) representa la fórmula para calcular la proyección ortogonal de un vector x sobre un subespacio unidimensional u generado por el vector base b. ((x·b)/(||b||^2)) es el escalar λ que representa la coordenada de la proyección con respecto a la base b e b es el vector base que abarca u

Tenemos un vector x en dos dimensiones, y x se puede representar como una combinación lineal de los vectores base de r2. También tenemos un subespacio unidimensional u con un vector base b. Eso significa que todos los vectores en u se pueden representar como lambda tiempo b para algún lambda. Ahora estamos interesados en encontrar un vector en u que sea el más cercano a x.   Resulta que no podemos encontrar el vector en u más cercano a x, sino la proyección ortogonal de x sobre u.
Esto significa que el vector diferencia de x y su proyección es ortogonal a u. En general, estamos buscando la proyección ortogonal de x sobre u, y denotaremos esta proyección por πu(x). La proyección tiene dos propiedades importantes. En primer lugar, dado que πu(x) está en u, se deduce que existe un lambda en r tal que πu(x) puede escribirse como λ veces, b el múltiplo del vector base que abarca u. El lambda es la coordenada de la proyección con respecto a la base b del subespacio u. 
Proyeccion ortogonal de x sobre u lo representamos como πu(x) la cual tiene dos propiedades importantes:
1) En primer lugar, dado que πu(x) está en u, se deduce que existe un lambda en r tal que πu(x) puede escribirse como λ veces, b el múltiplo del vector base que abarca u. El lambda es la coordenada de la proyección con respecto a la base b del subespacio u.  πu(x) ∈ u ⇒ ⊢ λ∈R : πu(x) = λb; donde λ es la coordenada de la proyeccion con respecto a la base b del subespacio u.
2) La segunda propiedad es que el vector diferencia de x y su proyección sobre u es ortogonal a u. Esto significa que es ortogonal al vector base que abarca u. La segunda propiedad es que el producto interior entre b y la diferencia entre pi u de x y x es cero <b πu(x)-x> = 0
Esta es la condición de ortogonalidad. Estas propiedades generalmente se mantienen para cualquier x en RD y subespacios unidimensionales u.
"""

import numpy as np

def orthogonal_projection(x, b):
    """
    Calcula la proyección ortogonal de x sobre el subespacio u con vector base b.

    Args:
        x: ndarray de forma (D,) que representa el vector a proyectar.
        b: ndarray de forma (D,) que representa el vector base del subespacio.

    Returns:
        La proyección ortogonal de x sobre el subespacio u en términos de λb.
    """
    lambda_ = np.dot(x, b) / np.linalg.norm(b)**2
    return lambda_, lambda_ * b

# Prueba con vectores en R2
x = np.array([1, 2])
b = np.array([2, 2])
lambda_, projection = orthogonal_projection(x, b)
print(np.dot(x, b))
print(np.linalg.norm(b)**2)
print(f"λ = {lambda_}, proyección = {projection}")

"""
La matriz de proyección 'P' para proyectar cualquier vector en R^3 sobre el subespacio abarcado por el vector base b = [b1, b2, b3] se puede calcular utilizando la fórmula P = (b b^T) / (b^T b). Esta matriz de proyección'P' es una matriz cuadrada 3 x 3 que puede usarse para proyectar cualquier vector en R^3 sobre el subespacio unidimensional generado por   
"""

import numpy as np

def projection_matrix(b):
    """
    Calcula la matriz de proyección para el subespacio abarcado por el vector base b.

    Args:
        b: ndarray de forma (D,) que representa el vector base del subespacio.

    Returns:
        La matriz de proyección P.
    """
    b = np.array(b).reshape(-1, 1)  # Convertir b en una columna vector
    P = b @ b.T / (b.T @ b)
    return P

# Prueba con el vector base en R3
b = np.array([1, 2, 2])
P = projection_matrix(b)
print(P)

"""
Para proyectar un vector 'v' sobre un subespacio utilizando una matriz de proyección 'P', se realiza multiplicando la matriz 'P' por el vector v. Esto se puede expresar como P v donde 'P' es la matriz de proyeccion y 'v' es el vector que se está proyectando
"""

import numpy as np

# Matriz de proyección dada
P = (1/25)*np.array([[9, 0, 12],[0, 0, 0], [12, 0, 16]])

# Vector a proyectar
v = np.array([1, 1, 1])

# Proyección de v sobre el subespacio
projection = P @ v

print(projection)


"""
El error de reconstrucción se refiere a la diferencia entre el vector original y su proyección sobre el subespacio. Se calcula tomando la norma del vector que resulta de la resta entre el vector original y su proyección. Matemáticamente, esto se expresa como ||x -πu(x)||, donde 'x' es el vector original y 'πu(x)' es su proyección sobre u.
"""

import numpy as np

# Punto de datos original
original = np.array([1, 1, 1])

# Proyección del punto de datos original
projection = (1/9)*np.array([5, 10, 10])

# Error de reconstrucción
reconstruction_error = np.linalg.norm(original - projection)

print(reconstruction_error)


"""
Tenemos un vector "x" en un espacio tridimensional. Definimos un subespacio bidimensional "u" con vectores base "b1" y "b2". 

"u = [b1, b2], y representa el plano. Buscamos la proyección ortogonal de "x" sobre "u", la llamaremos "πu(x)". Esta proyección se verá como un punto sobre el plano "u".

Ya podemos hacer dos observaciones:

πu(x) está en "u": se puede representar como una combinación lineal de los vectores base de "u" (usando coeficientes λ1 y λ2). La diferencia entre "x" y πu(x) es ortogonal a "u": es perpendicular a todos los vectores base de "u". Podemos usar el producto interno para comprobar que esto se cumple tanto para "b1" como para "b2".

Ahora vamos a generalizar esta idea:

"x" es un vector D-dimensional. Buscamos una proyección sobre un subespacio M-dimensional "u". Podemos definir un vector "lambda" con todos los coeficientes λi y una matriz "B" que contiene todos los vectores base de "u" como columnas. Con esto, podemos escribir 

πu(x) = λ1b1 + λ2b2. →   πu(x) = Σ(M,i=1)λibi

Siguiendo la segunda propiedad, es que el vector de diferencias de x - πu(x) que son orthogonales a todos los vectores base 'u'. Ahora podemos utilizar el producto interno 

<πu(x) - x, b1> = 0
<πu(x) - x, b2> = 0  → <πu(x) - xi, bi>= 0, i = 1 .... M

Pero ahora, vamos a formular para el caso general, donde x es un vector D-dimensional, y vamos a localizar un subespacio M-dimensional 'u'.

πu(x) = Bλ
<πu(x) - xi, bi>= <Bλ - xi, bi> = 0
<Bλ, bi> - <xibi> = 0  i = 1 .... M
λ^T B^T - x^T bi = 0   i = 1 .... M
λ^T B^T B - x^T B = 0
λ^T   = x^T B (B^T B)^-1
λ     = (B^T B)^-1 B^T x
πu(x) = Bλ = B (B^T B)^-1 B^T x  →  proyección de matriz = B (B^T B)^-1 B^T
En una base ortonormal B^T B es la matriz identidad
πu(x) = B^T B x
"""

import numpy as np

# Vector original x
x = np.array([12, 0, 0])

# Vectores base del subespacio u
b1 = np.array([1, 1, 1])
b2 = np.array([0, 1, 2])

# Matriz B que contiene apilacion de b1 y b2 como columnas respectivamente.
B = np.column_stack((b1, b2))

# Calculamos B^T * B y su inversa
BTB = B.T @ B
BTB_inv = np.linalg.inv(BTB)

# Calculamos lambda
lambda_ = BTB_inv @ B.T @ x

# Proyección de x sobre u
projection = B @ lambda_

# Matriz de proyección
P = B @ BTB_inv @ B.T

print("La apilacion de b1 y b2 en la Matriz B:", B)
print("La transpuesta de la matriz B es: ", B.T)
print("Matriz B^T * B:", BTB)
print("Matriz inversa de B^T * B:", BTB_inv)
print("Las coordenadas del punto proyectado con respecto a b1 y b2 son:", lambda_)
print("Proyección de x sobre u:", projection)
print("Matriz de proyección:", P)
print("La matriz de proyección es simétrica:", np.allclose(P, P.T))

"""
El rango de una matriz es el número máximo de columnas linealmente independientes en la matriz. En el caso de la matriz de proyección `P`, el rango es igual a la dimensión del subespacio sobre el que se está proyectando. En este caso, se está proyectando sobre un subespacio definido por dos vectores base (`b1` y `b2`), por lo que el rango de la matriz de proyección debería ser 2, no 1.
"""
print("El rango de la matriz de proyección es:", np.linalg.matrix_rank(P))


"""
El proceso que estás describiendo es la proyección ortogonal de datos de alta dimensión a un subespacio de dimensión inferior, que es un concepto clave en técnicas de reducción de dimensionalidad como el Análisis de Componentes Principales (PCA).

El proceso de derivación de las coordenadas de un dato en un espacio de dimensión inferior utilizando proyección ortogonal sobre un subespacio principal.

a) xn = Σ(m, j=1)  Bjn bj.  Combinación lineal Bjn por bj, donde bj forman la base ortonormal de nuestro subespacio.
b) función de pérdida, j = (1/N) Σ (N, n=1) ||xn - x̄n||^2
c) Derivada parcial de nuestra función de pérdida con respecto a x̄n (dj/dx̄n) = - (2/N) (xn - x̄n)^T
d) (dj/dBin) = (dj/dx̄n) * (dx̄n/dBin) 
e) (dx̄n/dBin) = bi,  i=1 ... M
f) (dj/dBin) = -(2/N) (xn - x̄n)^T bi
g) (dj/dBin)  = -(2/N) (xn - Σ(m, j=1)  Bjn bj)^T bi = -(2/N) ((xn)^T bi - Bin (bi)^T bi), como (bi)^T bi = 1 entonces
h) (dj/dBin) = -(2/N) ((xn)^T bi - Bin)
i) Bin = (xn)^T bi

Este código primero calcula la proyección de los datos en el subespacio principal. Luego, calcula la función de pérdida, que es la distancia cuadrada media entre los datos originales y su proyección. Después, calcula la derivada de la función de pérdida con respecto a la proyección de los datos. A continuación, calcula la derivada de la proyección de los datos con respecto a los vectores base. Finalmente, calcula la derivada de la función de pérdida con respecto a los vectores base y actualiza los vectores base en la dirección opuesta al gradiente para minimizar la función de pérdida.

Nota: Este código asume que estás utilizando el descenso de gradiente para minimizar la función de pérdida, y que tienes una tasa de aprendizaje predefinida. Además, este código no incluye ninguna lógica para detener el entrenamiento, por lo que deberías ejecutarlo dentro de un bucle y establecer una condición de parada adecuada.
"""
import numpy as np

# Generar datos aleatorios
N = 100  # número de muestras
D = 10  # dimensión de los datos originales
M = 2  # dimensión del subespacio principal
X = np.random.randn(N, D)

# Inicializar los vectores base aleatoriamente
B = np.random.randn(D, M)

# Definir la tasa de aprendizaje
learning_rate = 0.01

# Ejecutar el código de proyección y actualización de vectores base
for _ in range(1000):  # ejecutar durante 1000 iteraciones
    # Calcular la proyección de los datos en el subespacio principal
    Xn = np.dot(X, B)  # Xn.shape = (N, M)

    # Calcular la función de pérdida
    loss = np.mean(np.sum((X - np.dot(Xn, B.T))**2, axis=1))

    # Calcular la derivada de la función de pérdida con respecto a Xn
    dloss_dXn = -2/N * (X - np.dot(Xn, B.T))

    # Calcular la derivada de Xn con respecto a B
    dXn_dB = np.einsum('nd,nm->nmd', X, np.ones((N, M)))  # dXn_dB.shape = (N, D, M)

    # Calcular la derivada de la función de pérdida con respecto a B
    dloss_dB = np.einsum('nd,nmd->md', dloss_dXn, dXn_dB)

    # Actualizar B
    B -= learning_rate * dloss_dB.T  # Transpose dloss_dB

    # Imprimir la función de pérdida cada 100 iteraciones
    if _ % 100 == 0:
        print(f"Iteration {_}, Loss: {loss}")

# Normalizar B
B = B / np.linalg.norm(B, axis=0)       

# Calcular la proyección de X en el subespacio principal
Xn = B @ (B.T @ X.T)  # Xn.shape = (M, N)

# Transponer Xn a la forma (N, M)
Xn = Xn.T


"""
Estás describiendo el proceso de proyección ortogonal de datos en un subespacio principal, que es una parte clave del Análisis de Componentes Principales (PCA). Aquí está una descripción paso a paso de lo que estás haciendo:

a) `xn = Σ(m, j=1) Bjn bj.`: Estás proyectando cada punto de datos `xn` en el subespacio principal. Cada `Bjn` es el coeficiente de la proyección de `xn` en la dirección del vector base `bj`.

b) `j = (1/N) Σ (N, n=1) ||xn - x̄n||^2`: Esta es la función de pérdida que estás minimizando. Es el error cuadrático medio de la reconstrucción, que mide la diferencia entre cada punto de datos original `xn` y su reconstrucción `x̄n` a partir de la proyección en el subespacio principal.

c) `(dj/dx̄n) = - (2/N) (xn - x̄n)^T`: Esta es la derivada de la función de pérdida con respecto a `x̄n`. Estás utilizando esto para actualizar `x̄n` en cada iteración del algoritmo de optimización.

d) `Bjn = (xn)^T bj, j = 1 .... M`: Esta es la forma de calcular los coeficientes `Bjn` de la proyección. Cada `Bjn` es simplemente el producto escalar de `xn` y `bj`.

e) `xn - x̄n = Σ (D, j=M+1) (bj)^T xn) bj`: Esta es otra forma de expresar la diferencia entre `xn` y `x̄n`. Estás expresando esta diferencia como una suma de proyecciones en las direcciones de los vectores base que no están en el subespacio principal.

f) `j = Σ (D, j=M+1) (bj)^T S bj`: Esta es una forma de expresar la función de pérdida en términos de la matriz de covarianza `S` de los datos y los vectores base `bj`. Estás sumando las varianzas de los datos cuando se proyectan en las direcciones de los vectores base que no están en el subespacio principal.

Estos son los pasos básicos del PCA. El objetivo es encontrar los vectores base `bj` que minimizan la función de pérdida `j`, lo que equivale a maximizar la varianza de los datos cuando se proyectan en el subespacio principal.
"""

import numpy as np
from sklearn.decomposition import PCA

# Generar algunos datos aleatorios
np.random.seed(0)
X = np.random.rand(100, 10)

# a) Proyectar los datos en el subespacio principal
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# b) La función de pérdida es el error cuadrático medio de la reconstrucción
X_reconstructed = pca.inverse_transform(X_pca)
loss = np.mean((X - X_reconstructed) ** 2)

# c) La derivada de la función de pérdida con respecto a X_pca se calcula automáticamente durante el entrenamiento del PCA

# d) Los coeficientes de la proyección son simplemente las componentes principales
B = pca.components_

# e) La diferencia entre X y X_reconstructed es la proyección de los datos en el subespacio que se ignora
difference = X - X_reconstructed

# f) La función de pérdida también se puede expresar en términos de la matriz de covarianza de los datos
covariance_matrix = np.cov(X, rowvar=False)
loss_from_covariance = np.trace(covariance_matrix) - np.sum(pca.explained_variance_)

print("Loss from reconstruction error: ", loss)
print("Loss from covariance matrix: ", loss_from_covariance)


import numpy as np
from numpy.testing import assert_allclose

def PCA(X, num_components):
    """
    Args:
        X: ndarray de tamaño (N, D), donde D es la dimensión de los datos
           y N es el número de puntos de datos
        num_componentes: el número de componentes principales a utilizar.
    Devuelve
        los datos reconstruidos, la media muestral de X, los valores principales
        y componentes principales
    """

    # SU CÓDIGO AQUÍ
    # su solución debe aprovechar las funciones que ha implementado arriba.
    ### Descomente y modifique el código siguiente
# primero realice la normalización en los dígitos para que tengan media cero y varianza unitaria
    N, D = X.shape
    mean = np.mean(X, axis=0)
    Xbar = X - mean 
# A continuación, calcular la matriz de covarianza de datos S
    S = np.dot(Xbar.T, Xbar) / X.shape[0] 

# # A continuación, encontrar los valores propios y los correspondientes vectores propios de S
    eig_vals, eig_vecs = np.linalg.eig(S)
# Toma el `num_components` superior de eig_vals y eig_vecs,
# Estos serán los correspondientes valores principales y componentes
# Recuerde que los vectores propios son las columnas de la matriz `eig_vecs`.
    principal_vals = eig_vals[:num_components]
    principal_components = eig_vecs[:, :num_components]

# Debido a errores de precisión, los vectores propios pueden resultar complejos, por lo que sólo hay que tomar sus partes reales.
# Reconstruir los datos proyectando los datos normalizados sobre la base de los componentes principales.
# Recuerde que los puntos de datos en X_normalized están dispuestos a lo largo de las filas
# Pero al proyectar, necesitamos que estén ordenados a lo largo de las columnas 
# Observa que hemos restado la media de X, así que asegúrate de añadirla de nuevo...
# a los datos reconstruidos
    reconst = np.dot(Xbar, principal_components) @ principal_components.T + mean
    principal_components = np.real(principal_components)
    
    return reconst, mean, principal_vals, principal_components

X = np.array([[3, 6, 7],
              [8, 9, 0],
              [1, 5, 2]])

reconst, mean, principal_vals, principal_components = PCA(X, 1)

print('Cheacking mean...')
mean_exp = np.array([4, 20 / 3, 3])
np.testing.assert_allclose(mean, mean_exp, rtol=1e-5)
print('Mean is computed correctly!')

print('Checking principal values...')
principal_vals_exp = np.array([15.39677773])
np.testing.assert_allclose(principal_vals, principal_vals_exp, rtol=1e-5)
print('Principal Values are computed correctly!')

print('Checking principal components...')
principal_components_exp = np.array([[-0.68811066],
                                     [-0.40362611],
                                     [ 0.60298398]])
np.testing.assert_allclose(principal_components, principal_components_exp, rtol=1e-5)
print("Principal components are computed correctly!")

print('Checking reconstructed data...')
reconst_exp = np.array([[ 1.68166528,  5.30679755,  5.03153182],
                        [ 7.7868029 ,  8.8878974 , -0.31833472],
                        [ 2.53153182,  5.80530505,  4.2868029 ]])
np.testing.assert_allclose(reconst, reconst_exp, rtol=1e-5)
print("Reconstructed data is computed correctly!")


"""
El análisis de componentes principales (PCA, por sus siglas en inglés) es un método utilizado para reducir la dimensionalidad de un conjunto de datos, manteniendo la mayor cantidad posible de información. Los pasos implicados en el análisis de componentes principales son los siguientes:

Normalizar los datos:

Este paso implica restar la media de los datos y, a menudo, dividir los datos por la desviación estándar para asegurarse de que todas las variables contribuyan de manera similar al análisis. Sin embargo, a veces solo se resta la media para centrar los datos sin escalarlos.
Construir la matriz de covarianza:

La matriz de covarianza captura la relación entre las diferentes variables en el conjunto de datos. Para un conjunto de datos de dimensionalidad D, la matriz de covarianza es una matriz simétrica DxD donde cada elemento representa la covarianza entre dos variables.
Calcular los vectores propios y los valores propios de la matriz de covarianza:

Los vectores propios (eigenvectores) y los valores propios (eigenvalues) de la matriz de covarianza son fundamentales en PCA. Los vectores propios representan las direcciones de máxima variabilidad en los datos, mientras que los valores propios representan la cantidad de variabilidad explicada por cada vector propio. El eigenvector asociado con el valor propio más grande es el eigenvector de componente principal.
Proyectar los datos en el nuevo espacio dimensional reducido:

Los nuevos datos de dimensionalidad reducida se obtienen proyectando los datos originales sobre los vectores propios seleccionados (también conocidos como componentes principales). Esto se logra multiplicando la matriz de datos normalizados por la matriz de vectores propios seleccionados.
Una medida de la correlación entre diferentes variables viene dada por la Matriz de covarianza
Las direcciones en las que se extienden los datos vienen dadas por los Vectores propios
La magnitud de la dispersión de los datos a lo largo de los componentes principales viene dada por los Valores propios.
"""

import numpy as np

# Generar algunos datos aleatorios
X = np.array([[1, 0],
              [1, 1],
              [1, 2]])

# Dimension de la matriz X. N = Numero de Filas, D = Numero de Columnas
N, D = X.shape

# 1) Normalizar los datos
X_normalized = X - np.mean(X, axis=0)

# 2) Construir la matriz de covarianza de los datos
covariance_matrix_S = np.dot(X_normalized.T, X_normalized) / X.shape[0]

# 3) Calcular el vector propio y los valores propios de la matriz de covarianza
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix_S)

# Ordenar los vectores propios por sus valores propios correspondientes, en orden descendente
idx = eigenvalues.argsort()[::-1]   
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:,idx]

# 4) Encontrar los nuevos datos de dimensionalidad reducida
X_pca = np.dot(X_normalized, eigenvectors)

# Calcular la matriz de proyección
P = np.dot(eigenvectors, eigenvectors.T)
P1= X @ np.linalg.inv(np.transpose(X) @ X) @ np.transpose(X)

print("La matriz de covarianza es:\n",covariance_matrix_S )
print("Los valores propios son:\n" ,eigenvalues)
print("Los vectores propios son:\n", eigenvectors)
print("Datos de dimensionalidad reducida:\n", X_pca)


"""
Vamos a realizar un analisis de componentes principales PCA en algún conjunto de datos 𝑋 para 𝑀 componentes principales. A continuación, tenemos que realizar los siguientes pasos, que se dividen en partes:

1 Normalización de los datos (normalizar).
2 Encontrar los valores propios y los correspondientes vectores propios de la matriz de covarianza 𝑆. Ordenar por los mayores valores propios y los correspondientes vectores propios (eig).
3 Calcule la matriz de proyección ortogonal y utilícela para proyectar los datos en el subespacio abarcado por los vectores propios.
"""

import numpy as np
from numpy.testing import assert_allclose

def PCA(X, num_components):
    """
    Args:
        X: ndarray de tamaño (N, D), donde D es la dimensión de los datos
           y N es el número de puntos de datos
        num_componentes: el número de componentes principales a utilizar.
    Devuelve
        los datos reconstruidos, la media muestral de X, los valores principales
        y componentes principales
    """

    # SU CÓDIGO AQUÍ
    # su solución debe aprovechar las funciones que ha implementado arriba.
    ### Descomente y modifique el código siguiente
# primero realice la normalización en los dígitos para que tengan media cero y varianza unitaria
    N, D = X.shape
    mean = np.mean(X, axis=0)
    Xbar = X - mean 
# A continuación, calcular la matriz de covarianza de datos S
    S = np.dot(Xbar.T, Xbar) / X.shape[0] 

# # A continuación, encontrar los valores propios y los correspondientes vectores propios de S
    eig_vals, eig_vecs = np.linalg.eig(S)
# Toma el `num_components` superior de eig_vals y eig_vecs,
# Estos serán los correspondientes valores principales y componentes
# Recuerde que los vectores propios son las columnas de la matriz `eig_vecs`.
    principal_vals = eig_vals[:num_components]
    principal_components = eig_vecs[:, :num_components]

# Debido a errores de precisión, los vectores propios pueden resultar complejos, por lo que sólo hay que tomar sus partes reales.
# Reconstruir los datos proyectando los datos normalizados sobre la base de los componentes principales.
# Recuerde que los puntos de datos en X_normalized están dispuestos a lo largo de las filas
# Pero al proyectar, necesitamos que estén ordenados a lo largo de las columnas 
# Observa que hemos restado la media de X, así que asegúrate de añadirla de nuevo...
# a los datos reconstruidos
    reconst = np.dot(Xbar, principal_components) @ principal_components.T + mean
    principal_components = np.real(principal_components)
    
    return reconst, mean, principal_vals, principal_components

X = np.array([[3, 6, 7],
              [8, 9, 0],
              [1, 5, 2]])

reconst, mean, principal_vals, principal_components = PCA(X, 1)

print('Cheacking mean...')
mean_exp = np.array([4, 20 / 3, 3])
np.testing.assert_allclose(mean, mean_exp, rtol=1e-5)
print('Mean is computed correctly!')

print('Checking principal values...')
principal_vals_exp = np.array([15.39677773])
np.testing.assert_allclose(principal_vals, principal_vals_exp, rtol=1e-5)
print('Principal Values are computed correctly!')

print('Checking principal components...')
principal_components_exp = np.array([[-0.68811066],
                                     [-0.40362611],
                                     [ 0.60298398]])
np.testing.assert_allclose(principal_components, principal_components_exp, rtol=1e-5)
print("Principal components are computed correctly!")

print('Checking reconstructed data...')
reconst_exp = np.array([[ 1.68166528,  5.30679755,  5.03153182],
                        [ 7.7868029 ,  8.8878974 , -0.31833472],
                        [ 2.53153182,  5.80530505,  4.2868029 ]])
np.testing.assert_allclose(reconst, reconst_exp, rtol=1e-5)
print("Reconstructed data is computed correctly!")