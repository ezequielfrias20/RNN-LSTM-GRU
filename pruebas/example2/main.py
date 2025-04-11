import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from utils.graphic import generar_grafico
from utils.data.prepare_data import get_data_firestore
from utils.data.preprocess_data import preprocess_data, preprocess_data_2, preprocess_data_3
from utils.data.scaler_data import prepare_sequences
from utils.models.LSTM.model_1 import build_model, build_model_2

# Carga las variables del archivo .env
load_dotenv()

# Configuración
LOOKBACK = 60  # Número de pasos anteriores a considerar para la predicción
# Número de pasos a predecir (asumiendo datos cada 5 segundos para 2 minutos)
PREDICTION_HORIZON = 24
# Número de muestras que se procesan antes de actualizar los pesos del modelo, Valores típicos: 32, 64, 128. El modelo verá 32 muestras por cada actualización.
BATCH_SIZE = 32
# Epocas. Número de veces que el modelo recorre todo el dataset durante el entrenamiento.
EPOCHS = 100
# Fracción de los datos de entrenamiento que se usará para validación (evaluación durante el entrenamiento). Si es 0.2, el 20% de X_train/y_train se usa para validar (no se aprende de ellos).
VALIDATION_SPLIT = 0.2

# Obtener array de datos
data = get_data_firestore('metrics', [])

def train_and_evaluate(data):
    # Preprocesamiento
    # df, input_features, output_features = preprocess_data(data)
    data2 = pd.read_csv("../../data/datos_tcp.csv")
    df, input_features, output_features = preprocess_data_3(data2)

    # Preparar secuencias
    X, y, input_scaler, output_scaler = prepare_sequences(
        df, input_features, output_features, LOOKBACK, PREDICTION_HORIZON
    )

    # Dividir en train y test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Construir modelo
    model = build_model_2((LOOKBACK, len(input_features)),
                          PREDICTION_HORIZON, output_features)

    # Entrenar
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT,
        # Controla cuánta información se muestra durante el entrenamiento. 1: Muestra barra de progreso y métricas por época (recomendado).
        verbose=1
    )

    # Evaluar
    loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss[0]}, Test MAE: {loss[1]}")

    return model, X_test, y_test, input_scaler, output_scaler, input_features, output_features



model, X_test, y_test, input_scaler, output_scaler, input_features, output_features = train_and_evaluate(
    data)

y_pred = model.predict(X_test)

y_test_unscaled = output_scaler.inverse_transform(y_test)
y_pred_unscaled = output_scaler.inverse_transform(y_pred)

# Gráfica para cada feature de salida
for i in range(y_test_unscaled.shape[1]):
    plt.figure(figsize=(10, 5))
    plt.plot(y_test_unscaled[:, i], label='Real', color='blue', alpha=0.7)
    plt.plot(y_pred_unscaled[:, i],
             label='Predicho', color='red', linestyle='--')
    plt.title(f'Comparación Real vs Predicho - {output_features[i]}')
    plt.xlabel('Muestras')
    plt.ylabel('Valor')
    plt.legend()
    plt.grid(True)
    plt.show()
