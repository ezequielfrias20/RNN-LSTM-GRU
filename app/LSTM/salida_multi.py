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
import matplotlib.pyplot as plt


# ------------------------------
# Hiperparámetros
# ------------------------------
LOOKBACK = 24  # Número de pasos anteriores a considerar para la predicción

# Número de pasos a predecir (asumiendo datos cada 5 segundos para 2 minutos)
PREDICTION_HORIZON = 24

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

model.summary()

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


# ------------------------------
# Predicciones
# ------------------------------

# 1. Obtener predicciones
y_pred_scaled = model.predict(X_test)

# 2. Reescalar predicciones y valores reales al rango original
y_pred = scaler.inverse_transform(
    y_pred_scaled.reshape(-1, len(features))
).reshape(y_pred_scaled.shape[0], PREDICTION_HORIZON, len(features))

y_true = scaler.inverse_transform(
    y_test.reshape(-1, len(features))
).reshape(y_test.shape[0], PREDICTION_HORIZON, len(features))


# ------------------------------
# Graficar resultados
# ------------------------------


# 3. Elegimos cuántos pasos futuros mostrar por gráfico
time_step_to_plot = 0  # ver otro paso futuro (0 al 23)

# 4. Creamos grilla de gráficos
n_features = len(features)
cols = 3
rows = (n_features + cols - 1) // cols  # Ajusta número de filas

# ------------------------------
# Graficar de valores reales vs predicciones
# ------------------------------

fig, axs = plt.subplots(rows, cols, figsize=(15, 5 * rows))
axs = axs.flatten()

for i in range(n_features):
    axs[i].plot(y_true[:, time_step_to_plot, i], label='Real', marker='o', linestyle='-')
    axs[i].plot(y_pred[:, time_step_to_plot, i], label='Predicción', marker='x', linestyle='--')
    axs[i].set_title(features[i])
    axs[i].set_xlabel("Muestra")
    axs[i].set_ylabel("Valor")
    axs[i].legend()
    axs[i].grid(True)

# Si hay más espacios que variables, vaciamos los sobrantes
for j in range(n_features, len(axs)):
    fig.delaxes(axs[j])

plt.tight_layout()
plt.show()

# ------------------------------
# Configura qué muestra visualizar
# ------------------------------
sample_idx = 0  # Cambia este índice para ver otra secuencia de test

# ------------------------------
# Graficar de valores reales 24 vs predicciones
# ------------------------------
fig, axs = plt.subplots(2, 3, figsize=(18, 10))
axs = axs.flatten()  # Para iterar fácilmente

for i in range(len(features)):
    axs[i].plot(y_true[sample_idx, :, i], label='Real', marker='o')
    axs[i].plot(y_pred[sample_idx, :, i], label='Predicción', marker='x')
    axs[i].set_title(f'{features[i]}')
    axs[i].set_xlabel('Paso futuro')
    axs[i].set_ylabel('Valor')
    axs[i].grid(True)
    axs[i].legend()

plt.suptitle(f'Predicción de 24 pasos futuros para muestra {sample_idx}', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# ------------------------------
# Gráfica de valores escalados (normalizados)
# ------------------------------

# sample_idx indica la muestra específica que quieres ver
sample_idx = 0  # Puedes cambiarlo para ver otra muestra

fig, axs = plt.subplots(2, 3, figsize=(18, 10))
axs = axs.flatten()

for i in range(len(features)):
    axs[i].plot(y_test[sample_idx, :, i], label='Real Escalado', marker='o')
    axs[i].plot(y_pred_scaled[sample_idx].reshape(PREDICTION_HORIZON, len(features))[:, i],
                label='Predicción Escalada', marker='x')
    axs[i].set_title(f'{features[i]} (Escalado)')
    axs[i].set_xlabel('Paso futuro')
    axs[i].set_ylabel('Valor normalizado')
    axs[i].grid(True)
    axs[i].legend()

plt.suptitle(f'Valores escalados para muestra {sample_idx}', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()