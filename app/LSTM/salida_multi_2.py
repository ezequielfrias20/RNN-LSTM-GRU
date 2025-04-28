import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Reshape
from sklearn.model_selection import train_test_split
from utils.data.prepare_data import get_data_firestore, get_data_firestore_df
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from utils.save import save_model_and_scalers, load_model_and_scalers
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ------------------------------
# Hiperparámetros
# ------------------------------
LOOKBACK = 30  # Número de pasos anteriores a considerar para la predicción

# Número de pasos a predecir (asumiendo datos cada 2 segundos para 2 minutos)
PREDICTION_HORIZON = 10

# Número de muestras que se procesan antes de actualizar los pesos del modelo, Valores típicos: 32, 64, 128. El modelo verá 32 muestras por cada actualización.
BATCH_SIZE = 16

# Epocas. Número de veces que el modelo recorre todo el dataset durante el entrenamiento.
EPOCHS = 100

# Fracción de los datos de entrenamiento que se usará para validación (evaluación durante el entrenamiento). Si es 0.2, el 20% de X_train/y_train se usa para validar (no se aprende de ellos).
VALIDATION_SPLIT = 0.1

LOAD_MODEL = False

ESCENARIO = 1

SAVE_MODEL = True

NAME_MODEL = 'saved_model_lstm_multi_2_step_10_delayVideo'


# ------------------------------
# Procesamiento de datos
# ------------------------------

# Cargar tus datos (supongamos que ya están en un DataFrame)
df = get_data_firestore_df('metrics', fields_to_extract=None, force_refresh=True)

# Filtramos los datos atipicos 
df_filtrado = df[
    (df['jitterVideo'] != 0) & 
    (df['roundTripTimeVideo'] <= 2000) & 
    (df['roundTripTimeVideo'] >= 0)
].copy()

df_filtrado['delayVideo'] = df_filtrado['roundTripTimeVideo'] / 2
df_filtrado['delayAudio'] = df_filtrado['roundTripTimeAudio'] / 2
df_filtrado['packetLossRateVideo'] = (df_filtrado['packetsLostVideo'] / (
    df_filtrado['packetsReceivedVideo'] + df_filtrado['packetsLostVideo']))*100
df_filtrado['packetLossRateAudio'] = (df_filtrado['packetsLostAudio'] / (
    df_filtrado['packetsReceivedAudio'] + df_filtrado['packetsLostAudio']))*100

features = [
    'delayVideo',
    'delayAudio',
    'jitterVideo',
    'jitterAudio',
    'packetLossRateVideo',
    'packetLossRateAudio'
]

escenario_1 = df_filtrado[df_filtrado['roomId'] ==
                          '5ea95487-0a31-4e33-9263-c31717b81b5e'].copy()
escenario_2 = df_filtrado[df_filtrado['roomId'] ==
                          'b3e9e3d7-cc40-484a-a327-18e0f9dac1c7'].copy()
escenario_3 = df_filtrado[df_filtrado['roomId'] ==
                          'cfbfff06-9780-4b48-883b-bb453d285a75'].copy()
escenario_4 = df_filtrado[df_filtrado['roomId'] ==
                          'a37de4c6-83a2-4b34-b173-7e79b325c983'].copy()


def select_escenario():
    if SAVE_MODEL:
        return df_filtrado[features]
    if ESCENARIO == 1:
        return escenario_1[features]
    elif ESCENARIO == 2:
        return escenario_2[features]
    elif ESCENARIO == 3:
        return escenario_3[features]
    elif ESCENARIO == 4:
        return escenario_4[features]
    else:
        return df_filtrado[features]


data = select_escenario()

# Head de datos, te da las primeras 5 filas
print(data.head())
print(data.info())  # Información básica Tipo de datos
# Descripción general, te da estadicsticas generales (La media, la desviación estandar, el minimo, )
print(data.describe())


# Primero, separamos train + temp (validación + test)
# 70% train, 30% para validación + test
train_data, temp_data = train_test_split(data, test_size=0.3, shuffle=False)

# Luego, separamos validation y test
val_data, test_data = train_test_split(
    temp_data, test_size=0.5, shuffle=False)  # 15% validación, 15% test

print(f"Train len: {len(train_data)}")
print(f"Validation len: {len(val_data)}")
print(f"Test len: {len(test_data)}")

# ------------------------------
# Escalado de datos
# ------------------------------

# 1. Fit solo en train
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data)

# 2. Transform en validation y test
val_scaled = scaler.transform(val_data)
test_scaled = scaler.transform(test_data)

# ------------------------------
# Crear secuencias temporales
# ------------------------------


def create_sequences(data, input_steps, output_steps):
    X, y = [], []
    for i in range(len(data) - input_steps - output_steps + 1):
        X.append(data[i:i+input_steps])
        y.append(data[i+input_steps:i+input_steps+output_steps])
    return np.array(X), np.array(y)


X_train, y_train = create_sequences(train_scaled, LOOKBACK, PREDICTION_HORIZON)
X_val, y_val = create_sequences(val_scaled, LOOKBACK, PREDICTION_HORIZON)
X_test, y_test = create_sequences(test_scaled, LOOKBACK, PREDICTION_HORIZON)

if (LOAD_MODEL):
    # Cargar modelo y escaladores
    model, scaler, scaler_ouput, features, features_ouput = load_model_and_scalers(
        NAME_MODEL)
else:

    # ------------------------------
    # Entrenamiento del modelo
    # ------------------------------

    # Entrenar
    early_stop = EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True)

    model = Sequential([
        LSTM(64, input_shape=(LOOKBACK, len(features))),
        RepeatVector(PREDICTION_HORIZON),  # Repite el vector 60 veces
        # Otra LSTM para expandir a secuencia
        LSTM(64, return_sequences=True),
        # Predice 6 features en cada paso
        TimeDistributed(Dense(len(features)))
    ])

    model.compile(optimizer='adam', loss='mse', metrics=["mae", "mse", RootMeanSquaredError()])

    # 7. Entrenamiento usando validación
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=1,
        callbacks=[early_stop],
    )
    if (SAVE_MODEL):
        save_model_and_scalers(model, scaler,
                               scaler, features, features, NAME_MODEL)


# ------------------------------
# Evaluación
# ------------------------------


# 8. Evaluar en test
test_loss = model.evaluate(X_test, y_test)
print(f"Test Loss (Escalado): {test_loss}")
print("Loss: (Escalado)", test_loss[0])
print("MAE: (Escalado)", test_loss[1])
print("MSE: (Escalado)", test_loss[2])
print("RMSE: (Escalado)", test_loss[3])

# Convertir métricas a escala original (si se usa StandardScaler)
mae_original = test_loss[1] * scaler.scale_[0]
rmse_original = test_loss[3] * scaler.scale_[0]
print(f'MAE en escala original: {mae_original}')
print(f'RMSE en escala original: {rmse_original}')

# ------------------------------
# Graficar Train + Test + Predictions
# ------------------------------


# 1. Obtener todos los datos reales (sin escalado) para cada feature
data_real = df_filtrado[features].values

# 2. Predicción sobre el set de test
# Shape: (n_samples, pred_steps, n_features)
y_pred_scaled = model.predict(X_test)

# 3. Desescalar las predicciones para todas las features
y_pred_unscaled = np.zeros_like(y_pred_scaled)

for i, feature in enumerate(features):
    min_val = scaler.data_min_[i]
    max_val = scaler.data_max_[i]
    y_pred_unscaled[:, :, i] = y_pred_scaled[:, :, i] * \
        (max_val - min_val) + min_val

# 4. Tomar solo la primera predicción de cada secuencia (más simple)
y_pred_flat = y_pred_unscaled[:, 0, :]  # (n_samples, n_features)

# 5. Definir tamaños
train_size = len(train_data)
val_size = len(val_data)
test_size = len(test_data)

start_pred = train_size + val_size + LOOKBACK
x_pred = np.arange(start_pred, start_pred + len(y_pred_flat))

# 6. Plot para cada feature
for idx, feature in enumerate(features):
    plt.figure(figsize=(18, 6))

    # Real values
    plt.plot(data_real[:, idx],
             label=f'Datos Reales ({feature})', color='blue')

    # Predicted values
    plt.plot(x_pred, y_pred_flat[:, idx],
             label=f'Predicción ({feature})', color='red')

    # Líneas de separación
    plt.axvline(train_size, color='green', linestyle='--', label='Fin Train')
    plt.axvline(train_size + val_size, color='orange',
                linestyle='--', label='Fin Validation')

    plt.title(f'Evolución de {feature} y Predicción sobre Test')
    plt.xlabel('Índice temporal')
    plt.ylabel(feature)
    plt.legend()
    plt.grid(True)
    plt.show()
