import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from utils.data.prepare_data import get_data_firestore
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam


# ------------------------------
# Hiperparámetros
# ------------------------------
LOOKBACK = 24  # Número de pasos anteriores a considerar para la predicción

# Número de pasos a predecir (asumiendo datos cada 5 segundos para 2 minutos)
PREDICTION_HORIZON = 1

# Número de muestras que se procesan antes de actualizar los pesos del modelo, Valores típicos: 32, 64, 128. El modelo verá 32 muestras por cada actualización.
BATCH_SIZE = 16

# Epocas. Número de veces que el modelo recorre todo el dataset durante el entrenamiento.
EPOCHS = 100

# Fracción de los datos de entrenamiento que se usará para validación (evaluación durante el entrenamiento). Si es 0.2, el 20% de X_train/y_train se usa para validar (no se aprende de ellos).
VALIDATION_SPLIT = 0.1


# ------------------------------
# Procesamiento de datos
# ------------------------------

# Cargar tus datos (supongamos que ya están en un DataFrame)
data_json = get_data_firestore('metrics', [])
df = pd.DataFrame(data_json)

# Convertir timestamp a datetime y ordenar
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

df_filtrado = df[df['jitterVideo'] != 0]

df_filtrado['delayVideo'] = df_filtrado['roundTripTimeVideo']
df_filtrado['delayAudio'] = df_filtrado['roundTripTimeAudio']
df_filtrado['packetLossRateVideo'] = df_filtrado['packetsLostVideo'] / \
    (df_filtrado['packetsReceivedVideo'] + df_filtrado['packetsLostVideo'])
df_filtrado['packetLossRateAudio'] = df_filtrado['packetsLostAudio'] / \
    (df_filtrado['packetsReceivedAudio'] + df_filtrado['packetsLostAudio'])

features = [
        'delayVideo',
        'delayAudio',
        'jitterVideo',
        'jitterAudio',
        'packetLossRateVideo',
        'packetLossRateAudio'
        ]

print(df_filtrado[features].head()) # Head de datos, te da las primeras 5 filas 
print(df_filtrado[features].info()) # Información básica Tipo de datos
print(df_filtrado[features].describe()) # Descripción general, te da estadicsticas generales (La media, la desviación estandar, el minimo, )
data = df_filtrado[features].values

# ------------------------------
# Escalado de datos
# ------------------------------
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# ------------------------------
# Crear secuencias temporales
# ------------------------------


def create_sequences(data, input_steps, output_steps):
    X, y = [], []
    for i in range(len(data) - input_steps - output_steps + 1):
        X.append(data[i:i+input_steps])
        y.append(data[i+input_steps:i+input_steps+output_steps])
    return np.array(X), np.array(y)


X, y = create_sequences(scaled_data, LOOKBACK, PREDICTION_HORIZON)

# ------------------------------
# Dividir en entrenamiento y prueba
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False)

# ------------------------------
# Construcción del modelo LSTM
# ------------------------------
model = Sequential()
model.add(LSTM(128, activation='tanh', return_sequences=False,
          input_shape=(LOOKBACK, len(features))))
model.add(Dense(PREDICTION_HORIZON * len(features)))  # Salida total
model.compile(optimizer='adam', loss='mse', metrics=["mae", "mse", RootMeanSquaredError()])

# Revisa NaN o infinitos en los datos
print("NaN en X_train:", np.isnan(X_train).any())
print("Inf en X_train:", np.isinf(X_train).any())
print("NaN en y_train:", np.isnan(y_train).any())
print("Inf en y_train:", np.isinf(y_train).any())

# Revisa estadísticas básicas
print("Mínimos:", X_train.min())
print("Máximos:", X_train.max())

# ------------------------------
# Entrenamiento del modelo
# ------------------------------

# Entrenar
early_stop = EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train,
    # Flatten para que coincida con la salida
    y_train.reshape(y_train.shape[0], -1),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    verbose=1,
    callbacks=[early_stop],
)

# ------------------------------
# Evaluación
# ------------------------------
metrics_dict = model.evaluate(X_test, y_test.reshape(y_test.shape[0], -1))
print("Loss:", metrics_dict[0])
print("MAE:", metrics_dict[1])
print("MSE:", metrics_dict[2])
print("RMSE:", metrics_dict[3])
