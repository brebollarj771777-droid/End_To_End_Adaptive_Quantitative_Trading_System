"""
Trading System Neural Network Pipeline

Creates a GRU/Dense-based Neural Network Model to analyze portfolio's historical data, predict financial position probabilities and
use adaptive leverage to take well-informed trading decisions.
"""

#Third Party Library Imports.

import numpy as np
import sqlite3 as sq3
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dense, GRU, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from sqlalchemy import create_engine


#########################
#USER-DEFINED FUNCTIONS.#
#########################

##########################
#1. Data Window Creation.#
##########################

#Create data windows to improve tendency analysis, with an offset on the Target to avoid look-ahead bias.
def Crear_Ventanas(df,objetivo,ventana):
  X,Y = [],[]
  for i in range(ventana,len(df)):
    X.append(df.values[i-ventana:i])
    Y.append(objetivo.values[i - 1])
  return np.array(X),np.array(Y)

########################################
#2. Data Fetching From SQLite Database.#
########################################

# Select asset data ordered by ticker, processing each asset individually for further analysis.

def Obtencion_De_Datos(ticker):

  with sq3.connect("Activos Financieros Inv.db") as conexion:
    datos = pd.read_sql(f'''
     SELECT Fecha,
            Ticker,
            Rendimientos_Logaritmicos,
            Media_Movil20Dias,
            Media_Movil10Dias,
            Volatilidad_Movil20Dias,
            Volatilidad_Movil10Dias,
            Volumen_Normalizado_20,
            Volumen_Normalizado_10,
            ATR_Normalizado_20,
            ATR_Normalizado_10,
            RSI_Normalizado_20,
            RSI_Normalizado_10,
            ADX_Normalizado_20,
            ADX_Normalizado_10,
            row_number() OVER (ORDER BY Fecha ASC) AS Fila
     FROM Datos_Modelo WHERE Ticker = '{ticker}'
  ''', conexion)

  return datos

###############################
#3. Features Table Assembling.#
###############################

#Calculate 20-day Window Average Normalized ATR, choosing between 10 or 20-day window metrics based on the result.

def Obtencion_De_Features(datos):
  features = datos.copy()
  
  #Daily volatility threshold set on 1.5%
  umbral_vol_crit_ = 0.015
  
  #To calculate ATR, only future model-training data is used to avoid look-ahead bias.
  limite_entrenamiento = int(len(features) * 0.8)
  volatilidad_promedio = features["ATR_Normalizado_20"].iloc[:limite_entrenamiento].mean()
  if volatilidad_promedio >= umbral_vol_crit_:
    ventana = 10
    features = features[features["Fila"] >= 10]
    features.drop(columns = ["Media_Movil20Dias","Volatilidad_Movil20Dias", "Volumen_Normalizado_20", "ATR_Normalizado_20", "RSI_Normalizado_20","ADX_Normalizado_20" ,"Fila"], inplace = True)
  else:
    ventana = 20
    features = features[features["Fila"] >= 20]
    features.drop(columns = ["Media_Movil10Dias","Volatilidad_Movil10Dias", "Volumen_Normalizado_10" ,"ATR_Normalizado_10", "RSI_Normalizado_10","ADX_Normalizado_10" ,"Fila"], inplace = True)

  return features,ventana, volatilidad_promedio

############################
#4. Target Column Creation.#
############################

#Create target column based on 5-day moving average of logarithmic returns criteria.

def Obtencion_De_Objetivo(datos,features,ventana):
  auxiliar = datos.copy()
  rend_log_auxiliar = auxiliar["Rendimientos_Logaritmicos"].copy()
  rend_log_auxiliar.dropna(inplace = True)
  auxiliar_MA5 = rend_log_auxiliar.rolling(window = 5).mean().shift(1)
  rend_log = features["Rendimientos_Logaritmicos"].copy()
  auxiliar_MA5 = auxiliar_MA5.loc[rend_log.index]
  
#Calculate average ATR for position threshold. 
  if ventana == 10:
    ATR_medio = features["ATR_Normalizado_10"].mean().copy()
  else:
    ATR_medio = features["ATR_Normalizado_20"].mean().copy()

#Threshold for target column to determine financial position.

  umbral_adaptativo = 0.15 * ATR_medio

#Establish conditions to decide financial position.

  condiciones = [
      (rend_log > auxiliar_MA5 + umbral_adaptativo),
      (rend_log < auxiliar_MA5 - umbral_adaptativo)
    ]

  features["Objetivo"] = np.select(condiciones, [1,-1], default = 0)

  features["Objetivo_Futuro"] = features["Objetivo"].shift(-1)

  return features

###########################
#5. Train-test Split Data.#
###########################

#Split features table' data on 80% training data and 20% test data. 
def Datos_Entrenamiento_Prueba(features):

  datos_entrenamiento = features.copy()
  datos_entrenamiento.dropna(inplace = True)
  datos_entrenamiento.drop(columns = ["Objetivo"], inplace = True)

  division = int(0.8*len(datos_entrenamiento))
  datosentrenamiento_train = datos_entrenamiento.iloc[:division]
  datosentrenamiento_test = datos_entrenamiento.iloc[division:]

  return datosentrenamiento_train, datosentrenamiento_test

##################
#6. Data Scaling.#
##################

# Standardize feature sets using training statistics to prevent data leakage into the test partition.
def Escalado_De_Datos(datosentrenamiento_train, datosentrenamiento_test):
  escalador_previo = StandardScaler()
  escalador_previo.fit(datosentrenamiento_train.drop(columns = ["Fecha", "Ticker", "Objetivo_Futuro"]))
  datos_entrenamiento_escalado = escalador_previo.transform(datosentrenamiento_train.drop(columns = ["Fecha", "Ticker", "Objetivo_Futuro"]))
  datos_prueba_escalado = escalador_previo.transform(datosentrenamiento_test.drop(columns = ["Fecha", "Ticker", "Objetivo_Futuro"]))

  return datos_entrenamiento_escalado, datos_prueba_escalado


# Format scaled arrays into DataFrames and generate 3D sequence tensors for recurrent neural network processing.
def Creacion_De_Ventanas(datos_entrenamiento_escalado, datos_prueba_escalado, datosentrenamiento_train, datosentrenamiento_test, Crear_Ventanas, ventana):
  if ventana == 10:
    columnas = ["Rendimientos_Logaritmicos", "Media_Movil10Dias", "Volatilidad_Movil10Dias", "Volumen_Normalizado_10", "ATR_Normalizado10Dias", "RSI_Normalizado10Dias", "ADX_Normalizado10Dias"]
  else:
    columnas = ["Rendimientos_Logaritmicos", "Media_Movil20Dias", "Volatilidad_Movil20Dias", "Volumen_Normalizado_20", "ATR_Normalizado20Dias", "RSI_Normalizado20Dias", "ADX_Normalizado20Dias"]
  datos_train = pd.DataFrame(datos_entrenamiento_escalado, columns = columnas, index = datosentrenamiento_train.index)
  datos_test = pd.DataFrame(datos_prueba_escalado, columns = columnas, index = datosentrenamiento_test.index)

  X_entrenamiento, Y_entrenamiento = Crear_Ventanas(datos_train, datosentrenamiento_train["Objetivo_Futuro"], ventana)
  X_prueba, Y_prueba = Crear_Ventanas(datos_test, datosentrenamiento_test["Objetivo_Futuro"], ventana)

  return X_entrenamiento, Y_entrenamiento, X_prueba, Y_prueba

##############################################
# 7. Model Construction & Training Execution.#
##############################################

# Assemble the GRU deep network architecture, handle class weighting, and reindex labels for categorical crossentropy.
def Construccion_Del_Modelo(X_entrenamiento, Y_entrenamiento, Y_prueba, ventana):
  pesos = compute_class_weight(class_weight = 'balanced', classes = np.unique(Y_entrenamiento), y = Y_entrenamiento)
  pesos = dict(enumerate(pesos))

  Modelo = Sequential([
    Input(shape = (X_entrenamiento.shape[1], X_entrenamiento.shape[2])),
    GRU(64, activation = 'tanh', return_sequences = True),
    Dropout(0.2),
    GRU(32, activation = 'tanh', return_sequences = False),
    Dropout(0.2),
    Dense(16, activation = 'relu'),
    Dropout(0.2),
    Dense(3, activation = 'softmax')
  ])

  Modelo.compile(optimizer = Adam(learning_rate= 0.0001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
  
  #Fit model output to softmax function allowed values (0,1,2)
  
  Y_entrenamiento_preparado = Y_entrenamiento + 1
  Y_prueba_preparado = Y_prueba + 1

  return Modelo, pesos, Y_entrenamiento_preparado, Y_prueba_preparado

# Train model using EarlyStopping to avoid overfitting, returning trading position predictions and target probabilities.
def Ejecucion_Del_Modelo(Modelo, X_entrenamiento, Y_entrenamiento, X_prueba, Y_prueba, pesos, Y_entrenamiento_preparado, Y_prueba_preparado):
  tf.keras.backend.clear_session()

  detencion = EarlyStopping(monitor = 'val_loss', patience = 15, verbose = 0, restore_best_weights = True)

  historial_modelo = Modelo.fit(
    X_entrenamiento,
    Y_entrenamiento_preparado,
    validation_data = (X_prueba, Y_prueba_preparado),
    class_weight = pesos,
    epochs = 100,
    batch_size = 32,
    callbacks = [detencion],
    verbose = 0)

  prediccion_resultado = Modelo.predict(X_prueba, verbose = 0)

  predicciones_mapeadas = np.argmax(prediccion_resultado, axis = 1)
  
  #Return model output to original state
  Y_predicciones_trading = predicciones_mapeadas - 1
  Y_prueba_trading = Y_prueba_preparado - 1

  return historial_modelo, Y_predicciones_trading, Y_prueba_trading, prediccion_resultado

#########################################
# 8. Backtesting & Dynamic Leverage Logic.#
#########################################

# Simulate active trading strategy with dynamic leverage based on prediction certainty, ADX trends, transaction fees, and borrowing costs.

def Resultados_Estrategia(datosentrenamiento_test, Y_predicciones_trading, Y_prueba_trading, prediccion_resultado, ventana, volatilidad_promedio):

  rendimientos_test_reales = datosentrenamiento_test["Rendimientos_Logaritmicos"].values[ventana:]

  # Base transaction commission (0.1%)
  
  comision = 0.001

  cambios_posicion = np.diff(Y_predicciones_trading, prepend=Y_predicciones_trading[0])

  comisiones = np.where(cambios_posicion != 0, comision, 0)
  
  # Long position probability score
  
  p = prediccion_resultado[:,2]

  apalancamiento = 0

  diferencia = np.abs(p - 1/3)

  if ventana == 10:
    adx_datos_test = datosentrenamiento_test["ADX_Normalizado_10"].values[ventana:]
  else:
    adx_datos_test = datosentrenamiento_test["ADX_Normalizado_20"].values[ventana:]
  # Dynamic leverage condition matrix
  condiciones_apalancamiento = [
    (diferencia >= 0.2) & (adx_datos_test >= 0.25),
    ((diferencia >= 0.15) & (diferencia < 0.2) & (adx_datos_test >= 0.25)) | ((diferencia >= 0.2) & (adx_datos_test < 0.25)),
    (diferencia >= 0.05) & (diferencia < 0.15),
  ]

  apalancamiento = np.select(condiciones_apalancamiento, [2, 1.5, 1.25], default = 1)
  
  # Financial financing and borrow fees setup
  
  tasa_interes_anual = 0.06

  tasa_interes_diaria = tasa_interes_anual / 252

  if volatilidad_promedio < 0.015:
    tasa_prestamo_anual = 0.0075
  else:
    tasa_prestamo_anual = 0.015

  tasa_prestamo_diaria = tasa_prestamo_anual / 252

  # Portfolio return metrics computation
  
  retornos_estrategia_apalancados = (Y_predicciones_trading * rendimientos_test_reales) * apalancamiento

  comisiones_apalancadas = comisiones * apalancamiento

  costo_de_interes = np.where(Y_predicciones_trading != 0, (apalancamiento - 1) * (tasa_interes_diaria), 0)

  costo_prestamo_short = np.where(Y_predicciones_trading == -1, apalancamiento * tasa_prestamo_diaria, 0)

  costo_financiero_total = costo_de_interes + costo_prestamo_short

  retornos_estrategia_apalancados_netos = retornos_estrategia_apalancados - comisiones_apalancadas - costo_financiero_total
  
  # Cumulative log-return compounding
  
  retorno_acumulado_estrategia_apalancados = np.exp(np.cumsum(retornos_estrategia_apalancados_netos)) - 1

  retorno_acumulado_mercado = np.exp(np.cumsum(rendimientos_test_reales)) - 1

  return Y_predicciones_trading, apalancamiento, retornos_estrategia_apalancados_netos, rendimientos_test_reales, retorno_acumulado_estrategia_apalancados, retorno_acumulado_mercado

#####################################################
# MAIN EXECUTION PIPELINE & RESULTS CONSOLIDATION.#
#####################################################

# Extract unique asset tickers from the database
with sq3.connect("Activos Financieros Inv.db") as conexion:
  tickers = pd.read_sql("SELECT DISTINCT Ticker FROM Datos_Modelo", conexion).values
  tickers = tickers.flatten()

resultados_finales = []

# Iterative execution loop across all assets
for ticker in tickers:
  datos = Obtencion_De_Datos(ticker)
  features, ventana, volatilidad_promedio = Obtencion_De_Features(datos)
  features_obj = Obtencion_De_Objetivo(datos,features,ventana)
  datos_entrenamiento, datos_prueba = Datos_Entrenamiento_Prueba(features_obj)
  datos_entrenamiento_escalado, datos_prueba_escalado = Escalado_De_Datos(datos_entrenamiento, datos_prueba)
  X_entrenamiento, Y_entrenamiento, X_prueba, Y_prueba = Creacion_De_Ventanas(datos_entrenamiento_escalado, datos_prueba_escalado, datos_entrenamiento, datos_prueba, Crear_Ventanas, ventana)
  Modelo, pesos, Y_entrenamiento_preparado, Y_prueba_preparado = Construccion_Del_Modelo(X_entrenamiento, Y_entrenamiento, Y_prueba, ventana)
  historial_modelo, Y_predicciones_trading, Y_prueba_trading, prediccion_resultado = Ejecucion_Del_Modelo(Modelo, X_entrenamiento, Y_entrenamiento_preparado, X_prueba, Y_prueba, pesos, Y_entrenamiento_preparado, Y_prueba_preparado)
  Y_predicciones_trading, apalancamiento, retornos_estrategia_apalancados_netos, rendimientos_test_reales, retorno_acumulado_estrategia_apalancados, retorno_acumulado_mercado = Resultados_Estrategia(datos_prueba, Y_predicciones_trading, Y_prueba_trading, prediccion_resultado, ventana, volatilidad_promedio)
  resultados = pd.DataFrame({
  'Fecha' : datos_prueba['Fecha'].values[ventana:],
  'Ticker' : ticker,
  'Posicion_Estrategia' : Y_predicciones_trading,
  'Apalancamiento' : apalancamiento,
  'Rendimiento_Diario_Estrategia' : retornos_estrategia_apalancados_netos,
  'Rendimientos_Diario_Benchmark' : rendimientos_test_reales,
  'Rendimiento_Acumulado_Estrategia' : retorno_acumulado_estrategia_apalancados,
  'Rendimiento_Acumulado_Benchmark' : retorno_acumulado_mercado
  })
  resultados_finales.append(resultados)

# Combine results for all tickers into a single strategy DataFrame

Datos_Estrategia = pd.concat(resultados_finales)

# Store final strategy backtest output into SQLite database
conexion = create_engine('sqlite:///Datos Estrategia.db')
Datos_Estrategia.to_sql('Datos_Estrategia_Benchmark', conexion, if_exists = 'replace', index = False)
# Optional: Download database file if running on Google Colab
try:
    from google.colab import files
    files.download('Datos_Estrategia.db')
except ImportError:
    pass  # Executing locally; database is saved in the working directory