import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping
from utils.data.prepare_data import get_data_firestore
import os # Sistema operativo con python

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # SUPRIMIR ALGUNAS ADVERTENCIAS DE TENSORFLOW

# ------------------------------
# Hiperparámetros
# ------------------------------
LOOKBACK = 24
PREDICTION_HORIZON = 24
BATCH_SIZE = 16
EPOCHS = 30
N_SPLITS = 5

# ------------------------------
# Cargar y preparar datos
# ------------------------------
data_json = get_data_firestore('metrics', [])
df = pd.DataFrame(data_json)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

df_filtrado = df[df['jitterVideo'] != 0]
df_filtrado['delayVideo'] = df_filtrado['roundTripTimeVideo']
df_filtrado['delayAudio'] = df_filtrado['roundTripTimeAudio']
df_filtrado['packetLossRateVideo'] = df_filtrado['packetsLostVideo'] / (
    df_filtrado['packetsReceivedVideo'] + df_filtrado['packetsLostVideo'])
df_filtrado['packetLossRateAudio'] = df_filtrado['packetsLostAudio'] / (
    df_filtrado['packetsReceivedAudio'] + df_filtrado['packetsLostAudio'])

features = [
    'delayVideo',
    'delayAudio',
    'jitterVideo',
    'jitterAudio',
    'packetLossRateVideo',
    'packetLossRateAudio'
]

data = df_filtrado[features].values
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# ------------------------------
# Crear secuencias
# ------------------------------
def create_sequences(data, input_steps, output_steps):
    X, y = [], []
    for i in range(len(data) - input_steps - output_steps + 1):
        X.append(data[i:i+input_steps])
        y.append(data[i+input_steps:i+input_steps+output_steps])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, LOOKBACK, PREDICTION_HORIZON)
y = y.reshape(y.shape[0], -1)

# ------------------------------
# Construcción del modelo GRU
# ------------------------------
def build_model(input_shape, output_size):
    model = Sequential()
    model.add(GRU(128, activation='tanh', input_shape=input_shape))
    model.add(Dense(output_size))
    model.compile(optimizer='adam', loss='mse')
    return model

# ------------------------------
# Validación cruzada tipo serie temporal
# ------------------------------
def time_series_cross_validation(X, y, n_splits):
    split_size = len(X) // (n_splits + 1)
    results = []

    for i in range(n_splits):
        train_end = split_size * (i + 1)
        val_start = train_end
        val_end = val_start + split_size

        if val_end > len(X):
            break

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[val_start:val_end], y[val_start:val_end]

        model = build_model(input_shape=(X.shape[1], X.shape[2]), output_size=y.shape[1])
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                  validation_data=(X_val, y_val), callbacks=[early_stop], verbose=0)

        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        results.append(mse)

    return results

# ------------------------------
# Ejecutar validación cruzada
# ------------------------------
mse_scores = time_series_cross_validation(X, y, N_SPLITS)
print("MSE por fold:", mse_scores)
print("MSE promedio:", np.mean(mse_scores))

rmse_scores = [np.sqrt(mse) for mse in mse_scores]
print("RMSE por fold:", rmse_scores)
print("RMSE promedio:", np.mean(rmse_scores))