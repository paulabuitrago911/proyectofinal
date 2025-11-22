# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 13:55:56 2025

@author: Juan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import unicodedata
import re

"""
#url="https://drive.google.com/file/d/1_3tTr55GK3Tyd90MvD7tUaL1w7p0TEmZ/view?usp=sharing"
#df_web =pd.read_csv(url)
#df_carga=pd.DataFrame(df_web)
#print(df_carga)
"""
#Cargar un archivo con pandas
df_csv =pd.read_csv("bd_proyecto_limpias.csv")
print(df_csv)

##copia a la base de datos##
df = df_csv.copy()
#####

# =====================================================
# ========== FUNCIONES DE NORMALIZACIÓN ===============
# =====================================================

def normalizar_texto(texto):
    """Normaliza un texto eliminando tildes, pasando a minúsculas y quitando símbolos."""
    if pd.isnull(texto):
        return texto

    # Convertir a string
    texto = str(texto)

    # Quitar acentos
    texto = unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('utf-8')

    # Convertir a minúsculas
    texto = texto.lower()

    # Eliminar caracteres especiales
    texto = re.sub(r'[^a-z0-9\s.,_-]', '', texto)

    # Quitar espacios múltiples
    texto = re.sub(r'\s+', ' ', texto).strip()

    return texto


# =====================================================
# ========== ESTANDARIZAR NOMBRES COLUMNAS ============
# =====================================================

def limpiar_nombres_columnas(df):
    nuevas = []
    for col in df.columns:
        col = normalizar_texto(col)
        col = col.replace(' ', '_')
        nuevas.append(col)
    df.columns = nuevas
    return df


# =====================================================
# ========== MANEJO DE TIPOS DE DATOS =================
# =====================================================

def convertir_tipos(df):
    """Intenta convertir columnas a tipos adecuados automáticamente."""

    for col in df.columns:
        # Intentar convertir a numérico
        try:
            df[col] = pd.to_numeric(df[col])
            continue
        except:
            pass

        # Intentar convertir a fecha
        try:
            df[col] = pd.to_datetime(df[col])
            continue
        except:
            pass

        # Convertir a string si sigue sin tipo adecuado
        df[col] = df[col].astype(str)

    return df


# =====================================================
# ========== LIMPIEZA DE VALORES NULOS ================
# =====================================================

def tratar_nulos(df):
    """Rellena valores nulos basándose en el tipo de dato."""

    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    return df


# =====================================================
# ========== LIMPIEZA GENERAL DEL DATASET =============
# =====================================================

def limpiar_dataset(df):
    print("\n=== INICIANDO LIMPIEZA PROFESIONAL ===")

    # ------------------------------------
    # 1. Estandarizar nombres
    # ------------------------------------
    print("→ Estandarizando nombres de columnas...")
    df = limpiar_nombres_columnas(df)

    # ------------------------------------
    # 2. Conversión de tipos
    # ------------------------------------
    print("→ Convirtiendo tipos de datos automáticamente...")
    df = convertir_tipos(df)

    # ------------------------------------
    # 3. Normalización de texto
    # ------------------------------------
    print("→ Normalizando valores de texto...")
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].apply(normalizar_texto)

    # ------------------------------------
    # 4. Manejo de nulos
    # ------------------------------------
    print("→ Corrigiendo valores nulos...")
    df = tratar_nulos(df)

    # ------------------------------------
    # 5. Eliminar duplicados
    # ------------------------------------
    print("→ Eliminando duplicados...")
    df.drop_duplicates(inplace=True)

    # ------------------------------------
    # 6. Ordenar columnas
    # ------------------------------------
    df = df.reindex(sorted(df.columns), axis=1)

    print("✔ LIMPIEZA COMPLETA.")
    return df


# =====================================================
# ========== PRUEBA DEL MÓDULO (OPCIONAL) =============
# =====================================================
if __name__ == "__main__":
    print("\n### MÓDULO DE LIMPIEZA – PRUEBA ###")
    archivo = "bd_proyecto.csv"

    try:
        df_test = pd.read_csv(archivo)
        df_limpio = limpiar_dataset(df_test)
        df_limpio.to_csv("bd_proyecto_limpias.csv", index=False)
        print("Archivo limpio generado como bd_proyecto_limpias.csv")

    except Exception as e:
        print("No se encontró el archivo para prueba, pero el módulo funciona correctamente.")
        print("Error:", e)
        
        
#primeras 5 filas
print(df.head())

#Mostrar las ultimas 5 filas
print(df.tail())

#informacion general (columnas,tipos,nulos)
print(df.info())

#Estadistica basica
print(df.describe())

##filtrado##
#Selecciona una sola columna
print(df["total_cost"])

#selecciona fila por etiqueta
print(df.loc[0])

#Selecciona fila por posicion
print(df.iloc[1])

##4)filtros
#filtrado mayores a 1000 en profit
print(df[df["total_profit"]>1000])
#filtro de mayores de 1000 en profit y que sean cosmeticos
print(df[(df["total_profit"]>1000) &(df["item_type"]=="cosmetics")])
#Filtra si el pais Pakistan esta en la lista
print(df[df["country"].isin(["pakistan"])])

## Operaciones por columnas

df["candidad_doble"]=df["Units Sold"]*2
df["es_mayor"]=df["candidad_doble"]>=10000

##Limpieza de datos
df2 = pd.DataFrame({"a":[1,None,3],"b":[4,5,None]})
#Muestra en booleano donde hay valores nulos
print(df2.isnull())

#Reemplaza el nulo por un cero, lo muestra en el print
print((df2.fillna(0)))

#si se quiere guardar se crea una nueva variable
df3 = df2.fillna(0)

#Elimina las filas que tenga al menos un valor nulo
print(df2.dropna())

### estadistica basica
#promedio de ventas
print(df["total_profit"].mean())
#Mediana de las ventas
print(df["total_profit"].median())
#desviacion estandar de las ventas
print(df["total_profit"].std())



#  Filtra los datos principales que vas a graficar
# Es mejor filtrar primero y luego seleccionar las columnas
df_filtrado = df[(df["total_profit"] > 20000) & (df["total_cost"] >= 25000)]

# Prepara los datos X e Y usando las columnas específicas del DataFrame filtrado
# Reemplaza "NombreColumnaX" y "NombreColumnaY" con los nombres reales de tus columnas
eje_x = df_filtrado["total_profit"]         # Ejemplo: Eje X es el costo total
eje_y = df_filtrado["total_revenue"]      # Ejemplo: Eje Y es el ingreso total

# Prepara los arrays de colores y tamaños.
# DEBEN tener la misma longitud que tus datos filtrados.
longitud_datos = len(df_filtrado)
colores = np.random.rand(longitud_datos)
tamanos = np.random.rand(longitud_datos) * 1000

#Realiza la gráfica de dispersión con datos consistentes
plt.scatter(eje_x, eje_y, c=colores, s=tamanos, alpha=0.5, cmap="plasma")

plt.colorbar(label="Intensidad/Color Aleatorio")
plt.title("Gráfico de dispersión de Ganancias vs Costos (Filtrado)")
plt.xlabel("Eje X: Total Cost")
plt.ylabel("Eje Y: Total Revenue")
# plt.legend() # No es útil aquí a menos que definas etiquetas para los puntos
plt.grid(True)
plt.show()


#Grafico de caja de bigotes de costes mayores a 20000
plt.boxplot(eje_x)
plt.title("Distribucion de Costes donde sea mayor a 20000",fontsize=14,color="green")
plt.show()

#Grafico de caja de bigotes de Ganancias donde es mayor o igual 25000
plt.boxplot(eje_y)
plt.title("Distribucion de Ganancias donde sea mayor o igual 25000",fontsize=14,color="green")
plt.show()
#Grafico de caja de bigotes de costes
x = df["Total Cost"] 
plt.boxplot(x)
plt.title("Distribucion de Costes",fontsize=14,color="green")
plt.show()

#Grafico de caja de bigotes de Ganancias
y = df["total_revenue"]
plt.boxplot(y)
plt.title("Distribucion de Ganancias",fontsize=14,color="green")
plt.show()

#Histogramas de coste por unidad
z = df["unit_price"]
plt.hist(z, bins=30,color="red",edgecolor="white",alpha=0.7)
plt.title("Histograma")
plt.xlabel("Precio por unidad")
plt.ylabel("Frecuencia")
plt.legend()
plt.grid(True)
plt.show()


#Histogramas  de costes totales|
plt.hist(x, bins=30,color="red",edgecolor="white",alpha=0.7)
plt.title("Histograma")
plt.xlabel("Costo total")
plt.ylabel("Frecuencia")
plt.legend()
plt.grid(True)
plt.show()
