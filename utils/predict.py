import numpy as np
import pandas as pd

def predict_qos(model, recent_data_buffer, input_scaler, output_scaler, input_features, output_features, PREDICTION_HORIZON):
    """
    recent_data_buffer: Lista de los últimos LOOKBACK registros (cada uno es un dict)
    """

    # Esto funciona si es lista de dicts
    input_df = pd.DataFrame([recent_data_buffer])

    # Preprocesamiento (one-hot encoding, etc.)
    input_df = pd.get_dummies(input_df)

    # Asegurar que tenemos todas las columnas esperadas
    for feature in input_features:
        if feature not in input_df.columns:
            input_df[feature] = 0  # Valor por defecto

    input_df = input_df[input_features]

    # Escalar
    scaled_input = input_scaler.transform(
        input_df)  # Forma (LOOKBACK, n_features)

    # Dar forma para predicción (1, LOOKBACK, n_features)
    prediction = model.predict(
        scaled_input[np.newaxis, ...])  # Mejor que reshape

    # Reformatear y desescalar
    prediction = prediction.reshape(PREDICTION_HORIZON, len(output_features))
    prediction = output_scaler.inverse_transform(prediction)

    predictions = prediction

    # Crear salida
    result = {
        'delayVideo': np.mean(prediction[:, 0]),
        'delayAudio': np.mean(prediction[:, 1]),
        'jitterVideo': np.mean(prediction[:, 2]),
        'jitterAudio': np.mean(prediction[:, 3]),
        'packetLossRateVideo': np.mean(prediction[:, 4]),
        'packetLossRateAudio': np.mean(prediction[:, 5]),
        'throughputVideo': np.mean(prediction[:, 6]),
        'throughputAudio': np.mean(prediction[:, 7])
    }

    return result, predictions