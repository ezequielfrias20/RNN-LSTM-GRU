import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

def predict_and_plot(model, df, feature_columns, target_column, scaler):
    """
    Función para predecir los siguientes 10 pasos con un modelo previamente entrenado,
    usando los últimos 60 datos del dataframe y mostrar los resultados en un gráfico.
    
    Args:
        model_path (str): Ruta al modelo guardado.
        df (pd.DataFrame): Dataframe con los datos, debe contener las columnas de features y la columna objetivo.
        feature_columns (list): Lista de nombres de las columnas que se utilizarán como características.
        target_column (str): El nombre de la columna objetivo (lo que se quiere predecir).
    
    Returns:
        np.array: Predicción de los próximos 10 pasos.
    """
    # Cargar el modelo previamente guardado
    # model = load_model(model_path)
    
    # Obtener los últimos 60 datos
    last_60_data = df[feature_columns].iloc[-30:].values  # Últimos 60 pasos de las características
    last_60_data = np.reshape(last_60_data, (1, 30, len(feature_columns)))  # Redimensionar para el modelo

    # Predicción de los siguientes 10 pasos
    y_pred_scaled = model.predict(last_60_data)

    # 3. Desescalar las predicciones para todas las features
    y_pred_unscaled =  y_pred_scaled # np.zeros_like(y_pred_scaled)

    for i, feature in enumerate(features):
        min_val = scaler.data_min_[i]
        max_val = scaler.data_max_[i]
        # y_pred_unscaled[:, :, i] = y_pred_scaled[:, :, i] * (max_val - min_val) + min_val
        y_pred_unscaled[:, :, i] = scaler.inverse_transform(y_pred_scaled[:, :, i])

    # 4. Tomar solo la primera predicción de cada secuencia (más simple)
    y_pred_flat = y_pred_unscaled[:, 0, :]  # (n_samples, n_features)

    print("Predicción escalada:", y_pred_unscaled)
    
    # Convertir la predicción a un array de 1D (de forma: (10, 1)) si es necesario
    y_pred = y_pred_unscaled[0, :, 0]  # Suponiendo que la salida es de la forma (1, 10, 1)
    
    # Graficar los 60 datos y los 10 predichos
    plt.figure(figsize=(12, 6))
    
    # Los 60 primeros pasos
    plt.plot(range(len(df[feature_columns])), df[target_column].values, label='Datos actuales (últimos 60 pasos)', color='blue')
    
    # Los 10 pasos predichos
    plt.plot(range(len(df[feature_columns]) - 10, len(df[feature_columns])), y_pred, label='Predicción (próximos 10 pasos)', color='red', linestyle='--')
    
    plt.xlabel('Pasos de tiempo')
    plt.ylabel(target_column)
    plt.title('Predicción de los siguientes 10 pasos con el modelo')
    plt.legend()
    plt.show()
    
    return y_pred

# Ejemplo de uso:
# df es tu dataframe, feature_columns son las columnas de características que usas,
# y target_column es la columna que quieres predecir (por ejemplo 'metric' o 'value').

import pandas as pd
from utils.save import save_model_and_scalers, load_model_and_scalers
from utils.data.prepare_data import get_data_firestore, get_data_firestore_df

NAME_MODEL = 'saved_model_lstm_multi_2_step_10_delayVideo'

model, scaler, scaler_ouput, features, features_ouput = load_model_and_scalers(
        NAME_MODEL)

# Cargar tus datos (supongamos que ya están en un DataFrame)
df = get_data_firestore_df('metrics', fields_to_extract=None)

df_filtrado = df[df['jitterVideo'] != 0].copy()

df_filtrado['delayVideo'] = df_filtrado['roundTripTimeVideo'] / 2
df_filtrado['delayAudio'] = df_filtrado['roundTripTimeAudio'] / 2
df_filtrado['packetLossRateVideo'] = (df_filtrado['packetsLostVideo'] / (
    df_filtrado['packetsReceivedVideo'] + df_filtrado['packetsLostVideo']))*100
df_filtrado['packetLossRateAudio'] = (df_filtrado['packetsLostAudio'] / (
    df_filtrado['packetsReceivedAudio'] + df_filtrado['packetsLostAudio']))*100

features = [
    'delayVideo',
    # 'delayAudio',
    # 'jitterVideo',
    # 'jitterAudio',
    # 'packetLossRateVideo',
    # 'packetLossRateAudio'
]

escenario_1 = df_filtrado[df_filtrado['roomId'] == '5ea95487-0a31-4e33-9263-c31717b81b5e'].copy()
escenario_2 = df_filtrado[df_filtrado['roomId'] == 'b3e9e3d7-cc40-484a-a327-18e0f9dac1c7'].copy()
escenario_3 = df_filtrado[df_filtrado['roomId'] == 'cfbfff06-9780-4b48-883b-bb453d285a75'].copy()
escenario_4 = df_filtrado[df_filtrado['roomId'] == 'a37de4c6-83a2-4b34-b173-7e79b325c983'].copy()

# def select_escenario():
#     if SAVE_MODEL:
#         return df_filtrado[features]
#     if ESCENARIO == 1:
#         return escenario_1[features]
#     elif ESCENARIO == 2:
#         return escenario_2[features]
#     elif ESCENARIO == 3:
#         return escenario_3[features]
#     elif ESCENARIO == 4:
#         return escenario_4[features]
#     else:
#         return df_filtrado[features]

data = escenario_3[features]

target_column = 'delayVideo'  # Columna objetivo

# Predicción y gráfico
predictions = predict_and_plot(model, data, features, target_column, scaler)

