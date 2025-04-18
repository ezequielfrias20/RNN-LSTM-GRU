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

def prepare_sequences_5(df_train, df_val, df_test , input_features, output_features, lookback, horizon):
    scaler = MinMaxScaler()

    # Paso 1: Convertir a float32
    for col in input_features:
        df_train[col] = df_train[col].astype("float64")
        df_val[col] = df_val[col].astype("float64")
        df_test[col] = df_test[col].astype("float64")

    # Paso 2: Escalar los datos
    scaled_train = scaler.fit_transform(df_train)
    scaled_test = scaler.transform(df_test)
    scaled_val = scaler.transform(df_val)

    # Paso 3: Crear Secuencias para LSTM
    X_train, y_train = [], []
    for i in range(lookback, len(scaled_train) - horizon):
        X_train.append(scaled_train[i-lookback:i])
        y_train.append(scaled_train[i:i+horizon].flatten())

    X_test, y_test = [], []
    for i in range(lookback, len(scaled_test) - horizon):
        X_test.append(scaled_test[i-lookback:i])
        y_test.append(scaled_test[i:i+horizon].flatten())

    X_val, y_val = [], []
    for i in range(lookback, len(scaled_val) - horizon):
        X_val.append(scaled_val[i-lookback:i])
        y_val.append(scaled_val[i:i+horizon].flatten())
    
    return np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val), np.array(X_test), np.array(y_test), scaler