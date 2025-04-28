# importing the libraries
import numpy as np
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from utils.save import save_model_and_scalers, load_model_and_scalers

# Configuracion
LOAD_MODEL = True

data = pd.read_csv('./metrics.csv')

apple_df = data[
    (data['jitterVideo'] != 0) &
    (data['roundTripTimeVideo'] <= 2000) &
    (data['roundTripTimeVideo'] >= 0) &
    (data['roundTripTimeAudio'] <= 2000) &
    (data['roundTripTimeAudio'] >= 0) &
    (data['jitterAudio'] <= 60)
].copy()

apple_df['delayVideo'] = apple_df['roundTripTimeVideo'] / 2
apple_df['delayAudio'] = apple_df['roundTripTimeAudio'] / 2
apple_df['packetLossRateVideo'] = (apple_df['packetsLostVideo'] / (
    apple_df['packetsReceivedVideo'] + apple_df['packetsLostVideo']))*100
apple_df['packetLossRateAudio'] = (apple_df['packetsLostAudio'] / (
    apple_df['packetsReceivedAudio'] + apple_df['packetsLostAudio']))*100

features = [
    'delayVideo',
    'delayAudio',
    'jitterVideo',
    'jitterAudio',
    'packetLossRateVideo',
    'packetLossRateAudio',
    'date',
]

apple_df = apple_df[features]

print('apple_df Shape', apple_df.shape)
print(apple_df.head())

print(apple_df.isna().sum())
print(apple_df.info())

# converting the dataype of 'Date' col to 'datetime'
apple_df['date'] = pd.to_datetime(apple_df['date'])
apple_df = apple_df.sort_values('date')

# making the 'Date' col as index
apple_df.set_index('date', inplace=True)

print(apple_df.info())

# sort the indexes
# apple_df.sort_index(inplace = True)

print(apple_df.head())

# normalizing the data
scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(apple_df[apple_df.columns])

# converting the array into dataframe
apple_scaled_df = pd.DataFrame(
    scaled_values, columns=apple_df.columns, index=apple_df.index)

window_size = 60


def create_sequence(data, window_size):
    X = []
    y = []
    for i in range(window_size, len(data)):
        X.append(data.iloc[i-window_size:i].values)
        y.append(data.iloc[i].values)
    return np.array(X), np.array(y)


X, y = create_sequence(apple_scaled_df, window_size)

# train-test-split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False)

if (LOAD_MODEL):
    model, scaler, scaler, features, features = load_model_and_scalers(
        model_dir="model")
else:
    # model = keras.Sequential([
    #     # Adding the first LSTM layer with Dropout
    #     keras.layers.LSTM(units=50, return_sequences=True,
    #                       input_shape=(X_train.shape[1], X_train.shape[2])),
    #     keras.layers.Dropout(0.3),

    #     # Adding the second LSTM layer with Dropout
    #     keras.layers.LSTM(units=50, return_sequences=True),
    #     keras.layers.Dropout(0.3),

    #     # Adding the third LSTM layer with Dropout
    #     keras.layers.LSTM(units=50, return_sequences=False),
    #     keras.layers.Dropout(0.3),

    #     # Adding a Dense output layer
    #     keras.layers.Dense(y_train.shape[1])
    # ])

    model = keras.Sequential([
        # Adding the first LSTM layer with Dropout
        keras.layers.LSTM(units=128, activation='tanh', return_sequences=False,
                          input_shape=(X_train.shape[1], X_train.shape[2])),
        # keras.layers.Dropout(0.3),

        # # Adding the second LSTM layer with Dropout
        # keras.layers.LSTM(units=50, return_sequences=True),
        # keras.layers.Dropout(0.3),

        # # Adding the third LSTM layer with Dropout
        # keras.layers.LSTM(units=50, return_sequences=False),
        # keras.layers.Dropout(0.3),

        # Adding a Dense output layer
        keras.layers.Dense(y_train.shape[1])
    ])

    # model = Sequential()
    # model.add(LSTM(128, activation='tanh', return_sequences=False,
    #         input_shape=(LOOKBACK, len(features))))
    # model.add(Dense(PREDICTION_HORIZON * len(features)))  # Salida total
    # model.compile(optimizer='adam', loss='mse', metrics=["mae", "mse", RootMeanSquaredError()])

    model.summary()

    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['RootMeanSquaredError', 'mae'])

    # Early stopping condition
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=10,
                                   restore_best_weights=True)

    history = model.fit(X_train, y_train,
                        validation_split=0.2,
                        epochs=100,
                        batch_size=16,
                        callbacks=[early_stopping])

    save_model_and_scalers(model, scaler,
                           scaler, features, features, 'model')

# 8. Evaluar en test
test_loss = model.evaluate(X_test, y_test)
print(f"Test Loss (Escalado): {test_loss}")

# Making predictions on the test data
predictions = model.predict(X_test)

# Inverse scaling to get the original values
predictions = scaler.inverse_transform(predictions)
y_test_rescaled = scaler.inverse_transform(y_test)

# Plotting the results
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

escenario_1 = '5ea95487-0a31-4e33-9263-c31717b81b5e'
escenario_2 = 'b3e9e3d7-cc40-484a-a327-18e0f9dac1c7'
escenario_3 = 'cfbfff06-9780-4b48-883b-bb453d285a75'
escenario_4 = 'a37de4c6-83a2-4b34-b173-7e79b325c983'


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
            # (data['delayVideo'] <= 120) &
            # (data['delayAudio'] <= 100) &
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


data['delayVideo'] = data['roundTripTimeVideo'] / 2
data['delayAudio'] = data['roundTripTimeAudio'] / 2
data['packetLossRateVideo'] = (data['packetsLostVideo'] / (
    data['packetsReceivedVideo'] + data['packetsLostVideo']))*100
data['packetLossRateAudio'] = (data['packetsLostAudio'] / (
    data['packetsReceivedAudio'] + data['packetsLostAudio']))*100

df = escenario(1, data)

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
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# making the 'Date' col as index
df.set_index('date', inplace=True)

print(df.info())

# sort the indexes
# df.sort_index(inplace = True)

print(df.head())

# normalizing the data
# scaler = MinMaxScaler()
scaled_values = scaler.transform(df[df.columns])

# converting the array into dataframe
scaled_df = pd.DataFrame(scaled_values, columns=df.columns, index=df.index)

window_size = 60


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
    X_prueba, y_prueba, test_size=0.4, shuffle=False)

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

plt.figure(figsize=(14, 7))

for i, col in enumerate(df.columns[:3]):
    plt.subplot(3, 1, i + 1)
    plt.plot(range(1, len(df)+1), df[col], color='#0072BD',
             label=f'Real {title(col)}', marker='o')
    plt.title(f'{title(col)}')
    plt.xlabel('Pasos de Tiempo')
    plt.ylabel(f'{title(col)}')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 7))

for i, col in enumerate(df.columns[3:]):
    plt.subplot(3, 1, i + 1)
    plt.plot(range(1, len(df)+1),df[col], color='#0072BD',
             label=f'Real {title(col)}', marker='o')
    plt.title(f'{title(col)}')
    plt.xlabel('Pasos de Tiempo')
    plt.ylabel(f'{title(col)}')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

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
