import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping



# Hiperparámetros a explorar
def create_model(input_shape, output_size, units_lstm1, units_lstm2, dropout, learning_rate):
    model = Sequential([
        LSTM(units_lstm1, return_sequences=True, activation='tanh', input_shape=input_shape),
        Dropout(dropout),
        LSTM(units_lstm2, return_sequences=False, activation='tanh'),
        Dropout(dropout),
        Dense(output_size)
    ])

    optimizer = Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='mae',
        metrics=['mae', 'mse', RootMeanSquaredError()]
    )

    return model


def seleccion_parametros(X_train, y_train, X_val, y_val, output_features, PREDICTION_HORIZON):
    param_grid = {
        'units_lstm1': [64, 128],
        'units_lstm2': [32, 64],
        'dropout': [0.0, 0.2],
        'batch_size': [16, 32],
        'learning_rate': [0.001, 0.0005]
    }

    # Variables necesarias
    input_shape = (X_train.shape[1], X_train.shape[2])
    output_size = PREDICTION_HORIZON * len(output_features)

    best_model = None
    best_mae = float('inf')
    best_params = None

    # Combinaciones de hiperparámetros
    from itertools import product

    for u1, u2, d, bs, lr in product(param_grid['units_lstm1'],
                                    param_grid['units_lstm2'],
                                    param_grid['dropout'],
                                    param_grid['batch_size'],
                                    param_grid['learning_rate']):
        
        print(f"🔍 Probando configuración: units_lstm1={u1}, units_lstm2={u2}, dropout={d}, batch_size={bs}, learning_rate={lr}")
        
        model = create_model(input_shape, output_size, u1, u2, d, lr)

        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=bs,
            validation_split=0.2,
            callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
            verbose=0
        )

        mae = model.evaluate(X_val, y_val, verbose=0)[1]  # MAE

        print(f"📈 MAE obtenido: {mae:.4f}")

        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_params = {
                'units_lstm1': u1,
                'units_lstm2': u2,
                'dropout': d,
                'batch_size': bs,
                'learning_rate': lr
            }

    print("\n✅ Mejor configuración encontrada:")
    print(best_params)
    print(f"📉 Mejor MAE: {best_mae:.4f}")