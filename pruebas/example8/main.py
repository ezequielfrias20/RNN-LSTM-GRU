# importing the libraries
import numpy as np
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from utils.save import save_model_and_scalers, load_model_and_scalers
from utils.data.prepare_data import get_data_firestore, get_data_firestore_df

# ------------------------------
# CONFIGURACION
# ------------------------------
LOAD_MODEL = True  # Cargar modelo existente

WINDOW_SIZE = 60  # Tamaño de la ventana para la secuencia

# Número de muestras que se procesan antes de actualizar los pesos del modelo, Valores típicos: 16, 32, 64, 128. El modelo verá 32 muestras por cada actualización.
BATCH_SIZE = 16

# Epocas. Número de veces que el modelo recorre todo el dataset durante el entrenamiento.
EPOCHS = 100

# Fracción de los datos de entrenamiento que se usará para validación (evaluación durante el entrenamiento). Si es 0.2, el 20% de X_train/y_train se usa para validar (no se aprende de ellos).
VALIDATION_SPLIT = 0.2


# ------------------------------
# Carga de datos
# ------------------------------
data = get_data_firestore_df(
    'metrics', fields_to_extract=None, force_refresh=False)
# data = pd.read_csv('./metrics.csv')


# ------------------------------
# Procesamiento de datos
# ------------------------------

# Filtramos los datos atipicos
data_df = data[
    (data['jitterVideo'] != 0) &
    (data['roundTripTimeVideo'] <= 2000) &
    (data['roundTripTimeVideo'] >= 0) &
    (data['roundTripTimeAudio'] <= 2000) &
    (data['roundTripTimeAudio'] >= 0) &
    (data['jitterAudio'] <= 60)
].copy()

# Calculamos las nuevas columnas (delay y packetLossRate)
data_df['delayVideo'] = data_df['roundTripTimeVideo'] / 2
data_df['delayAudio'] = data_df['roundTripTimeAudio'] / 2
data_df['packetLossRateVideo'] = (data_df['packetsLostVideo'] / (
    data_df['packetsReceivedVideo'] + data_df['packetsLostVideo']))*100
data_df['packetLossRateAudio'] = (data_df['packetsLostAudio'] / (
    data_df['packetsReceivedAudio'] + data_df['packetsLostAudio']))*100

# Caracteristicas a usar
features = [
    'delayVideo',
    'delayAudio',
    'jitterVideo',
    'jitterAudio',
    'packetLossRateVideo',
    'packetLossRateAudio',
    'date',
]

# Filtramos las columnas con las caracteristicas
data_df = data_df[features]

# Evaluamos la forma de los datos
print('data_df Shape', data_df.shape)
print(data_df.head())
print(data_df.isna().sum())
print(data_df.info())

# converting the dataype of 'Date' col to 'datetime'
# data_df['date'] = pd.to_datetime(data_df['date'])
# data_df = data_df.sort_values('date')

# Columna 'Fecha' sea índice
data_df.set_index('date', inplace=True)

print(data_df.info())

# sort the indexes
# data_df.sort_index(inplace = True)

print(data_df.head())

# ------------------------------
# Normalizar los datos
# ------------------------------
scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(data_df[data_df.columns])

# Convertimos el array en un Dataframe
apple_scaled_df = pd.DataFrame(
    scaled_values, columns=data_df.columns, index=data_df.index)


# ------------------------------
# Crear secuencias temporales
# ------------------------------

# Tamaño de la ventana
window_size = WINDOW_SIZE

# Función para crear la secuencia


def create_sequence(data, window_size):
    X = []
    y = []
    for i in range(window_size, len(data)):
        X.append(data.iloc[i-window_size:i].values)
        y.append(data.iloc[i].values)
    return np.array(X), np.array(y)


X, y = create_sequence(apple_scaled_df, window_size)

# Separación de la data en train y test, un 80% para train y un 20% para test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False)

if (LOAD_MODEL):
    model, scaler, scaler, features, features = load_model_and_scalers(
        model_dir="model")
else:

    # ------------------------------
    # Entrenamiento del modelo
    # ------------------------------

    model = keras.Sequential([
        keras.layers.LSTM(units=128, activation='tanh', return_sequences=False,
                          input_shape=(X_train.shape[1], X_train.shape[2])),
        keras.layers.Dense(y_train.shape[1])
    ])

    model.summary()

    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mae', 'mse', 'RootMeanSquaredError'])

    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=10,
                                   restore_best_weights=True)

    history = model.fit(X_train, y_train,
                        validation_split=VALIDATION_SPLIT,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        callbacks=[early_stopping])

    save_model_and_scalers(model, scaler,
                           scaler, features, features, 'model')


# ------------------------------
# Evaluación
# ------------------------------


test_loss = model.evaluate(X_test, y_test)
print(f"Test Loss (Escalado): {test_loss}")
print("RMSE: (Escalado)", test_loss[0])
print("MAE: (Escalado)", test_loss[1])
print("MSE: (Escalado)", test_loss[2])
# print("RMSE: (Escalado)", test_loss[3])

# ------------------------------
# Predicciones
# ------------------------------

predictions = model.predict(X_test)

# Escalamamos de manera inversa para obtener los valores reales
predictions = scaler.inverse_transform(predictions)
y_test_rescaled = scaler.inverse_transform(y_test)

# ------------------------------
# Grafico de resultado de el 20% de test vs la predicción
# ------------------------------
plt.figure(figsize=(14, 7))

for i, col in enumerate(apple_scaled_df.columns):
    plt.subplot(2, 3, i + 1)
    plt.plot(y_test_rescaled[:, i], color='blue', label=f'Actual {col}')
    plt.plot(predictions[:, i], color='red', label=f'Predicted {col}')
    plt.title(f'{col} Prediction')
    plt.xlabel('Time')
    plt.ylabel(f'{col}')
    plt.legend()

plt.tight_layout()
plt.show()

# ================================================== PRUEBAS ==================================================

escenario_1 = '5ea95487-0a31-4e33-9263-c31717b81b5e' # Calidad optima
escenario_2 = 'b3e9e3d7-cc40-484a-a327-18e0f9dac1c7' # Calidad media
escenario_3 = 'cfbfff06-9780-4b48-883b-bb453d285a75'
escenario_4 = 'a37de4c6-83a2-4b34-b173-7e79b325c983'

# Funcion para filtrar la data por el escenario seleccionado
def escenario(escenario, data):
    if (escenario == 1):
        return data[
            # (data['jitterVideo'] != 0) &
            # (data['delayVideo'] <= 120) &
            # (data['delayAudio'] <= 100) &
            # (data['jitterAudio'] <= 20) &
            # (data['jitterVideo'] <= 40) &
            (data['roomId'] == escenario_1)
        ].copy()
    if (escenario == 2):
        return data[
            # (data['jitterVideo'] != 0) &
            # (data['delayVideo'] <= 250) &
            # (data['delayAudio'] <= 200) &
            # (data['jitterAudio'] <= 20) &
            # (data['jitterVideo'] <= 25) &
            (data['roomId'] == escenario_2)
        ].copy()
    if (escenario == 3):
        return data[
            # (data['jitterVideo'] != 0) &
            (data['delayVideo'] <= 600) &
            # (data['delayAudio'] <= 400) &
            # (data['jitterAudio'] <= 20) &
            # (data['jitterVideo'] <= 40) &
            (data['roomId'] == escenario_3)
        ].copy()
    if (escenario == 4):
        return data[
            # (data['jitterVideo'] != 0) &
            # (data['delayVideo'] <= 120) &
            # (data['delayAudio'] <= 100) &
            # (data['jitterAudio'] <= 20) &
            # (data['jitterVideo'] <= 40) &
            (data['roomId'] == escenario_4)
        ].copy()

# ------------------------------
# Preparación de datos para la predicción
# ------------------------------

# Agregamos las columnas que nos interesan
data['delayVideo'] = data['roundTripTimeVideo'] / 2
data['delayAudio'] = data['roundTripTimeAudio'] / 2
data['packetLossRateVideo'] = (data['packetsLostVideo'] / (
    data['packetsReceivedVideo'] + data['packetsLostVideo']))*100
data['packetLossRateAudio'] = (data['packetsLostAudio'] / (
    data['packetsReceivedAudio'] + data['packetsLostAudio']))*100

# Nuevo dataframe con los datos filtrados
df = escenario(4, data)

features = [
    'delayVideo',
    'delayAudio',
    'jitterVideo',
    'jitterAudio',
    'packetLossRateVideo',
    'packetLossRateAudio',
    'date',
]

df = df[features]

print('df Shape', df.shape)
print(df.head())

print(df.isna().sum())
print(df.info())

# converting the dataype of 'Date' col to 'datetime'
# df['date'] = pd.to_datetime(df['date'])
# df = df.sort_values('date')

# making the 'Date' col as index
df.set_index('date', inplace=True)

print(df.info())

# sort the indexes
# df.sort_index(inplace = True)

print(df.head())

# Normalizar los datos con el mismo scaler que se uso para el entrenamiento
scaled_values = scaler.transform(df[df.columns])

# Convertimos el array en un Dataframe
scaled_df = pd.DataFrame(scaled_values, columns=df.columns, index=df.index)

window_size = WINDOW_SIZE


def create_sequence(data, window_size):
    X = []
    y = []
    for i in range(window_size, len(data)):
        X.append(data.iloc[i-window_size:i].values)
        y.append(data.iloc[i].values)
    return np.array(X), np.array(y)


X_prueba, y_prueba = create_sequence(scaled_df, window_size)

# train-test-split

X_train_nuevo, X_test_prueba, y_train_nuevo, y_test_prueba = train_test_split(
    X_prueba, y_prueba, test_size=0.35, shuffle=False)

# Making predictions on the test data
predictions = model.predict(X_test_prueba)

# Inverse scaling to get the original values
predictions = scaler.inverse_transform(predictions)
y_test_rescaled = scaler.inverse_transform(y_test_prueba)

# # Plotting the results
# plt.figure(figsize=(14, 7))

# for i, col in enumerate(scaled_df.columns):
#     plt.subplot(2, 3, i + 1)
#     plt.plot(y_test_rescaled[:, i], color='blue', label=f'Actual {col}', marker='o')
#     plt.title(f'{col} Prediction')
#     plt.xlabel('Time')
#     plt.ylabel(f'{col}')
#     plt.legend()

# plt.tight_layout()
# plt.show()

# Plotting the results

# Funcion para definir el titulo de cada grafico
def title(field):
    if (field == "delayVideo"):
        return "Delay Video (ms)"
    if (field == "delayAudio"):
        return "Delay Audio (ms)"
    if (field == "jitterVideo"):
        return "Jitter Video (ms)"
    if (field == "jitterAudio"):
        return "Jitter Audio (ms)"
    if (field == "packetLossRateVideo"):
        return "Tasa de perdida de Paquete Video (%)"
    if (field == "packetLossRateAudio"):
        return "Tasa de perdida de Paquete Audio (%)"
    return ""


# ------------------------------
# Grafico de los datos de delay 
# ------------------------------

plt.figure(figsize=(14, 7))

for i, col in enumerate(df.columns[:2]):
    plt.subplot(2, 1, i + 1)
    plt.plot(range(1, len(df)+1), df[col], color='#0072BD',
             label=f'Real {title(col)}', marker='o')
    plt.title(f'{title(col)}')
    plt.xlabel('Pasos de Tiempo')
    plt.ylabel(f'{title(col)}')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ------------------------------
# Grafico de los datos de jitter 
# ------------------------------

plt.figure(figsize=(14, 7))

for i, col in enumerate(df.columns[2:4]):
    plt.subplot(2, 1, i + 1)
    plt.plot(range(1, len(df)+1), df[col], color='#0072BD',
             label=f'Real {title(col)}', marker='o')
    plt.title(f'{title(col)}')
    plt.xlabel('Pasos de Tiempo')
    plt.ylabel(f'{title(col)}')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ------------------------------
# Grafico de los datos de Tasa de Perdida de Paquetes 
# ------------------------------

plt.figure(figsize=(14, 7))

for i, col in enumerate(df.columns[4:]):
    plt.subplot(2, 1, i + 1)
    plt.plot(range(1, len(df)+1), df[col], color='#0072BD',
             label=f'Real {title(col)}', marker='o')
    plt.title(f'{title(col)}')
    plt.xlabel('Pasos de Tiempo')
    plt.ylabel(f'{title(col)}')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ------------------------------
# Grafico de los resultados de la predicción vs los datos reales 
# ------------------------------

plt.figure(figsize=(14, 7))

for i, col in enumerate(scaled_df.columns):
    plt.subplot(2, 3, i + 1)
    plt.plot(y_test_rescaled[:, i], color='#0072BD',
             label=f'Real {title(col)}', marker='o')
    plt.plot(predictions[:, i], color='#4DBEEE',
             label=f'Predicción {title(col)}', marker='o')
    plt.title(f'Predicción {title(col)}')
    plt.xlabel('Pasos de Tiempo')
    plt.ylabel(f'{title(col)}')
    plt.legend()

plt.tight_layout()
plt.show()
