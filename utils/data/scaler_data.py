from sklearn.preprocessing import MinMaxScaler
import numpy as np

def prepare_sequences(data, input_features, output_features, lookback, horizon):
    # horizon: Cuántos pasos futuros predecir.
    # Escalar datos
    input_scaler = MinMaxScaler()
    output_scaler = MinMaxScaler()
    
    X = input_scaler.fit_transform(data[input_features])
    y = output_scaler.fit_transform(data[output_features])
    
    # Crear secuencias para LSTM
    X_seq, y_seq = [], []
    
    for i in range(lookback, len(data) - horizon):
        X_seq.append(X[i-lookback:i])
        y_seq.append(y[i:i+horizon].flatten())  # Aplanar las salidas futuras
        
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    return X_seq, y_seq, input_scaler, output_scaler