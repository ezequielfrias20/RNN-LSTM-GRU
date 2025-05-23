import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from sklearn.model_selection import train_test_split
from utils.data.prepare_data import get_data_firestore
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from utils.save import save_model_and_scalers, load_model_and_scalers
from utils.models.LSTM.model_1 import build_model_3

# ------------------------------
# Hiperparámetros
# ------------------------------
LOOKBACK = 24
PREDICTION_HORIZON = 1
BATCH_SIZE = 16
EPOCHS = 100
VALIDATION_SPLIT = 0.1
LOAD_MODEL = False
SAVE_MODEL = True
NAME_MODEL = 'saved_model_gru_1'

# ------------------------------
# Carga y preprocesamiento de datos
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
    'delayVideo', 'delayAudio',
    'jitterVideo', 'jitterAudio',
    'packetLossRateVideo', 'packetLossRateAudio'
]

data = df_filtrado[features].values

# ------------------------------
# Escalado
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
# Train/Test Split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

if (LOAD_MODEL):
    # Cargar modelo y escaladores
    model, scaler, scaler_ouput, features, features_ouput = load_model_and_scalers(NAME_MODEL)
else:

    # ------------------------------
    # Modelo con GRU
    # ------------------------------
    # model = Sequential()
    # model.add(GRU(128, activation='tanh', return_sequences=False, input_shape=(LOOKBACK, len(features))))
    # model.add(Dense(PREDICTION_HORIZON * len(features)))  # Salida

    # model.compile(
    #     optimizer=Adam(learning_rate=0.001),
    #     loss='mse',
    #     metrics=["mae", "mse", RootMeanSquaredError()]
    # )

    model = build_model_3((LOOKBACK, len(features)), PREDICTION_HORIZON, features, features, LOOKBACK)



    model.summary()

    # ------------------------------
    # Revisar datos
    # ------------------------------
    print("NaN en X_train:", np.isnan(X_train).any())
    print("Inf en X_train:", np.isinf(X_train).any())
    print("NaN en y_train:", np.isnan(y_train).any())
    print("Inf en y_train:", np.isinf(y_train).any())
    print("Mínimos:", X_train.min())
    print("Máximos:", X_train.max())

    # ------------------------------
    # Entrenamiento
    # ------------------------------
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        X_train,
        y_train.reshape(y_train.shape[0], -1),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        verbose=1,
        callbacks=[early_stop],
    )
    if (SAVE_MODEL):
        save_model_and_scalers(model, scaler,
                               scaler, features, features, NAME_MODEL)

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