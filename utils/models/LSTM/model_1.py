from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.metrics import RootMeanSquaredError

def build_model(input_shape, PREDICTION_HORIZON, output_features):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(PREDICTION_HORIZON * len(output_features))  # Salida para múltiples pasos
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
        # LSTM(64, return_sequences=True),
        # Dropout(0.2),
        # LSTM(32),
        # Dropout(0.2),
        # Dense(64, activation='relu'),
        Dense(PREDICTION_HORIZON * len(output_features))  # Salida para múltiples pasos
    ])
    
    model.compile(
    optimizer="adam",
    loss="mae",
    metrics=[RootMeanSquaredError()]
    )
    
    return model