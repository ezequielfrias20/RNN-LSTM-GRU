from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.metrics import RootMeanSquaredError
import tensorflow as tf


def build_model(input_shape, PREDICTION_HORIZON, output_features):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(64, activation='relu'),
        # Salida para múltiples pasos
        Dense(PREDICTION_HORIZON * len(output_features))
    ])

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )

    return model


def build_model_2(input_shape, PREDICTION_HORIZON, output_features):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        LSTM(64, return_sequences=False),
        Dense(64, activation='relu'),
        Dropout(0.5),
        # Salida para múltiples pasos
        Dense(PREDICTION_HORIZON * len(output_features))
    ])

    model.compile(
        optimizer="adam",
        loss="mae",
        metrics=[RootMeanSquaredError()]
    )

    return model


def build_model_3(input_shape, PREDICTION_HORIZON, output_features, input_features, lookback):
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer='l2'),
                      input_shape=input_shape),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(PREDICTION_HORIZON * len(output_features))
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.Huber(),
        metrics=['mae', 'mse']
    )

    return model

def build_model_4(input_shape, PREDICTION_HORIZON, output_features, input_features, lookback):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        LSTM(64, return_sequences=False),
        Dense(128, activation='relu'),
        Dropout(0.5),
        # Salida para múltiples pasos
        Dense(PREDICTION_HORIZON * len(output_features))
    ])

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.Huber(),
        metrics=["mae", RootMeanSquaredError()]
    )

    return model


def build_model_6(input_shape, PREDICTION_HORIZON, output_features, input_features, lookback):
    model = Sequential([
    LSTM(128, activation='tanh', return_sequences=True, input_shape=input_shape),
    Dropout(0.2),
    LSTM(64, activation='tanh', return_sequences=False),  # Última capa LSTM
    Dropout(0.2),
    Dense(PREDICTION_HORIZON * len(output_features))
])
    model.compile(
        optimizer="adam",
        loss="mae",
        metrics=["mae", 'mse', RootMeanSquaredError()]
    )

    return model
