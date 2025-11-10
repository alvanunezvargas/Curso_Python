from sympy import symbols, exp, simplify, O

# Definir la variable simbólica
x, Δ = symbols('x Δ')

# Definir las expresiones
expresion1 = (1/exp(4)) * (1 + 2 * (x-2))
expresion2 = (1/exp(4)) * (1 - 2 * (x-2)) 
expresion3 = (1/exp(4)) * (1 - 2 * (x-2)) + 4*(x - 2)**2 
expresion4 = (1/exp(2)) * (2 * (x-2))
expresion5 = (1/exp(4)) * (4-2*x)

# Simplificar las expresiones
expresion1_simplificada = simplify(expresion1)
expresion2_simplificada = simplify(expresion2)
expresion3_simplificada = simplify(expresion3)
expresion4_simplificada = simplify(expresion4)
expresion5_simplificada = simplify(expresion5)

print("Expresión 1 simplificada:")
print(expresion1_simplificada)

print("\nExpresión 2 simplificada:")
print(expresion2_simplificada)

print("\nExpresión 3 simplificada:")
print(expresion3_simplificada)

print("\nExpresión 4 simplificada:")
print(expresion4_simplificada)

print("\nExpresión 5 simplificada:")
print(expresion5_simplificada)
