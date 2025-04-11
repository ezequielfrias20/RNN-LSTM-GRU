import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Reshape
from tensorflow.keras.callbacks import EarlyStopping
import json

# Preprocesamiento de datos adaptado para intervalos de 5 segundos
def preprocess_data(data_list, look_back=6, forecast_steps=24):  # 24 pasos = 2 minutos (24*5=120 segundos)
    """
    Preprocesa los datos para el modelo LSTM con intervalos de 5 segundos.
    
    Args:
        data_list: Lista de diccionarios con los datos de entrada.
        look_back: Número de pasos anteriores a considerar (6 pasos = 30 segundos de historia).
        forecast_steps: Número de pasos a predecir (24 pasos = 2 minutos).
        
    Returns:
        X, y: Arrays numpy preparados para el modelo.
        scalers: Diccionario de escaladores para inversión posterior.
        features: Lista de características usadas.
    """
    # Convertir a DataFrame
    df = pd.DataFrame(data_list)
    
    # Convertir timestamp a datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.sort_values('timestamp')
    
    # Seleccionar características relevantes
    features = [
        'jitterAudio', 'jitterVideo',
        'packetsLostAudio', 'packetsLostVideo',
        'roundTripTimeAudio', 'roundTripTimeVideo',
        'bytesReceivedAudio', 'bytesReceivedVideo'
    ]
    
    # Calcular throughput (bytes/segundo) en ventanas de 5 segundos
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds()
    df['throughputAudio'] = df['bytesReceivedAudio'].diff() / df['time_diff']
    df['throughputVideo'] = df['bytesReceivedVideo'].diff() / df['time_diff']
    features.extend(['throughputAudio', 'throughputVideo'])
    
    # Eliminar primera fila con NaN por el diff()
    df = df.iloc[1:]
    
    # Normalizar datos
    scalers = {}
    for feature in features:
        scaler = MinMaxScaler()
        df[feature] = scaler.fit_transform(df[[feature]])
        scalers[feature] = scaler
    
    # Crear secuencias para LSTM
    X, y = [], []
    for i in range(look_back, len(df) - forecast_steps):
        X.append(df[features].values[i - look_back:i])
        y.append(df[features].values[i:i + forecast_steps])
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y, scalers, features

# Construcción del modelo (similar pero ajustado para nuevos parámetros)
def build_model(input_shape, output_steps, n_features):
    """
    Construye un modelo LSTM para predicción de secuencias.
    
    Args:
        input_shape: Forma de los datos de entrada (look_back, n_features).
        output_steps: Número de pasos a predecir.
        n_features: Número de características.
        
    Returns:
        model: Modelo Keras compilado.
    """
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dense(output_steps * n_features),
        Reshape((output_steps, n_features))
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model

# Predicción adaptada para intervalos de 5 segundos
def predict_future(model, initial_data, scalers, features, look_back=6, forecast_steps=24):
    """
    Realiza predicciones futuras y revierte la normalización para datos cada 5 segundos.
    
    Args:
        model: Modelo entrenado.
        initial_data: Datos iniciales para la predicción.
        scalers: Diccionario de escaladores.
        features: Lista de características.
        look_back: Número de pasos anteriores (6 pasos = 30 segundos).
        forecast_steps: Pasos a predecir (24 pasos = 2 minutos).
        
    Returns:
        predictions: Diccionario con las predicciones en valores originales.
    """
    # Normalizar los datos iniciales
    scaled_data = []
    for i, feature in enumerate(features):
        scaled = scalers[feature].transform(initial_data[[feature]])
        scaled_data.append(scaled)
    
    scaled_data = np.array(scaled_data).T[0]
    
    # Hacer predicción
    prediction = model.predict(np.array([scaled_data[-look_back:]]))
    
    # Revertir normalización
    predictions = {}
    for i, feature in enumerate(features):
        pred_feature = prediction[0, :, i].reshape(-1, 1)
        predictions[feature] = scalers[feature].inverse_transform(pred_feature).flatten()
    
    return predictions

# Función principal adaptada
def main(input_data):
    # Simular datos históricos (en producción usarías datos reales)
    historical_data = []
    base_time = input_data[0]['timestamp']
    for i in range(72):  # 72 puntos históricos = 6 minutos de historia
        new_point = input_data[0].copy()
        # Variar ligeramente los valores
        for key in ['jitterAudio', 'jitterVideo', 'packetsLostAudio', 'packetsLostVideo',
                   'roundTripTimeAudio', 'roundTripTimeVideo', 'bytesReceivedAudio', 'bytesReceivedVideo']:
            new_point[key] = new_point[key] * (1 + np.random.uniform(-0.1, 0.1))
        new_point['timestamp'] = base_time - (72 - i) * 5000  # 5 segundos entre puntos (5000 ms)
        historical_data.append(new_point)
    
    historical_data.extend(input_data)
    
    # Preprocesar datos con nuevos parámetros
    X, y, scalers, features = preprocess_data(historical_data, look_back=6, forecast_steps=24)
    
    # Dividir en train y test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Construir y entrenar el modelo
    model = build_model(X_train.shape[1:], y.shape[1], len(features))
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5)
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=16,  # Batch size reducido para mejor ajuste
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1
    )
    
    # Preparar datos iniciales para predicción (últimos 6 puntos = 30 segundos)
    initial_df = pd.DataFrame(historical_data[-6:])
    
    # Hacer predicción
    predictions = predict_future(model, initial_df, scalers, features, look_back=6, forecast_steps=24)
    
    # Obtener valores predichos para el minuto 2 (paso 24)
    result = {
        'predicted_jitter_audio_2min': predictions['jitterAudio'][23],
        'predicted_jitter_video_2min': predictions['jitterVideo'][23],
        'predicted_packet_loss_audio_2min': predictions['packetsLostAudio'][23],
        'predicted_packet_loss_video_2min': predictions['packetsLostVideo'][23],
        'predicted_delay_audio_2min': predictions['roundTripTimeAudio'][23],
        'predicted_delay_video_2min': predictions['roundTripTimeVideo'][23],
        'predicted_throughput_audio_2min': predictions['throughputAudio'][23],
        'predicted_throughput_video_2min': predictions['throughputVideo'][23],
        'prediction_time_interval': '5 seconds',
        'prediction_horizon': '24 steps (2 minutes)'
    }
    
    return result

# Ejemplo de uso
if __name__ == "__main__":
    # Datos de entrada de ejemplo
    input_data = [{
        'packetsReceivedVideo': 279082,
        'roundTripTimeAudio': 505.798,
        'bytesReceived': 291914683,
        'jitterAudio': 18,
        'bytesSentAudio': 8290292,
        'networkType': '3g',
        'packetsReceivedAudio': 101584,
        'bytesReceivedVideo': 267775910,
        'roomId': '6d914414-7a99-4d0c-a40f-e5911a0dc694',
        'bytesSentVideo': 251731886,
        'bytesSent': 275296085,
        'packetsLostAudio': 5943,
        'bytesReceivedAudio': 8034740,
        'availableOutgoingBitrate': 132619,
        'jitterVideo': 978,
        'roundTripTimeVideo': 517.242,
        'packetsLostVideo': 6333,
        'timestamp': 1743899556388.829
    }]
    
    # Obtener predicciones
    predictions = main(input_data)
    
    print("Predicciones para después de 2 minutos (24 intervalos de 5 segundos):")
    print(json.dumps(predictions, indent=2))