import pandas as pd

df = pd.read_csv("C:/Users/Alvaro/Downloads/Palabras Sopa de Letras Bruto.csv", sep=';')

print(df.columns.tolist())
