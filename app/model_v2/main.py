import numpy as np
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Reshape # Añadido Reshape
from utils.save import save_model_and_scalers, load_model_and_scalers
from utils.data.prepare_data import get_data_firestore, get_data_firestore_df

# ------------------------------
# CONFIGURACION
# ------------------------------
LOAD_MODEL = True  # Cargar modelo existente o entrenar uno nuevo
PREDICTION_TYPE = 'multi_step'  # 'one_step' o 'multi_step'
N_FUTURE_STEPS = 30  # Número de pasos futuros a predecir (solo para multi_step)
MODEL_BASE_DIR = 'model' # Directorio base para guardar modelos

WINDOW_SIZE = 60
BATCH_SIZE = 16 # 32
EPOCHS = 50 # Reducido para pruebas rápidas, aumenta a 100 o más para resultados reales
VALIDATION_SPLIT = 0.2

# Determinar el directorio específico del modelo basado en el tipo de predicción
if PREDICTION_TYPE == 'one_step':
    MODEL_DIR = f"{MODEL_BASE_DIR}_one_step"
    N_OUTPUT_STEPS = 1
elif PREDICTION_TYPE == 'multi_step':
    MODEL_DIR = f"{MODEL_BASE_DIR}_multi_step_{N_FUTURE_STEPS}"
    N_OUTPUT_STEPS = N_FUTURE_STEPS
else:
    raise ValueError("PREDICTION_TYPE debe ser 'one_step' o 'multi_step'")

# ------------------------------
# Carga de datos
# ------------------------------
data_firestore = get_data_firestore_df(
    'metrics', fields_to_extract=None, force_refresh=False)
# data_firestore = pd.read_csv('./metrics.csv') # Descomenta si usas CSV local

# ------------------------------
# Procesamiento de datos
# ------------------------------
data_df = data_firestore[
    (data_firestore['jitterVideo'] != 0) &
    (data_firestore['roundTripTimeVideo'] <= 2000) &
    (data_firestore['roundTripTimeVideo'] >= 0) &
    (data_firestore['roundTripTimeAudio'] <= 2000) &
    (data_firestore['roundTripTimeAudio'] >= 0) &
    (data_firestore['jitterAudio'] <= 60)
].copy()

data_df['delayVideo'] = data_df['roundTripTimeVideo'] / 2
data_df['delayAudio'] = data_df['roundTripTimeAudio'] / 2


data_df['packetLossRateVideo'] = np.where(
    (data_df['packetsReceivedVideo'] + data_df['packetsLostVideo']) == 0,
    0,  # o algún otro valor apropiado si el denominador es cero
    (data_df['packetsLostVideo'] / (data_df['packetsReceivedVideo'] + data_df['packetsLostVideo'])) * 100
)
data_df['packetLossRateAudio'] = np.where(
    (data_df['packetsReceivedAudio'] + data_df['packetsLostAudio']) == 0,
    0,  # o algún otro valor apropiado si el denominador es cero
    (data_df['packetsLostAudio'] / (data_df['packetsReceivedAudio'] + data_df['packetsLostAudio'])) * 100
)


features_to_use = [
    'delayVideo',
    'delayAudio',
    'jitterVideo',
    'jitterAudio',
    'packetLossRateVideo',
    'packetLossRateAudio',
    'date',
]
data_df = data_df[features_to_use]
data_df.set_index('date', inplace=True)
data_df.sort_index(inplace=True) # Asegurar orden cronológico

# Eliminar NaNs que podrían surgir de cálculos o datos originales
data_df.dropna(inplace=True)

print('data_df Shape después de procesar y dropear NaNs:', data_df.shape)
print(data_df.head())
print("Valores Nulos:\n", data_df.isna().sum())
# data_df.info()

if data_df.empty or len(data_df) < WINDOW_SIZE + N_OUTPUT_STEPS:
    raise ValueError(f"No hay suficientes datos después del preprocesamiento para crear secuencias. "
                     f"Datos disponibles: {len(data_df)}, "
                     f"Necesarios (WINDOW_SIZE + N_OUTPUT_STEPS): {WINDOW_SIZE + N_OUTPUT_STEPS}")

# ------------------------------
# Normalizar los datos
# ------------------------------
scaler = MinMaxScaler()
# Guardamos las columnas y el índice antes de escalar para reconstruir el DataFrame
original_columns = data_df.columns
original_index = data_df.index

scaled_values = scaler.fit_transform(data_df)
apple_scaled_df = pd.DataFrame(scaled_values, columns=original_columns, index=original_index)
N_FEATURES = apple_scaled_df.shape[1]

# ------------------------------
# Crear secuencias temporales
# ------------------------------
def create_sequence_flexible(data, window_size, n_future_steps, n_features):
    """
    Crea secuencias de entrada (X) y salida (y).
    Para one-step: y es (n_samples, n_features)
    Para multi-step: y es (n_samples, n_future_steps, n_features)
    """
    X, y = [], []
    for i in range(window_size, len(data) - n_future_steps + 1):
        X.append(data.iloc[i-window_size:i].values) # Ventana de entrada
        # Para y, tomamos los siguientes n_future_steps para todas las features
        y_slice = data.iloc[i : i + n_future_steps].values
        if n_future_steps == 1:
            y.append(y_slice.ravel()) # (1, n_features) -> (n_features,)
        else:
            y.append(y_slice) # (n_future_steps, n_features)
    return np.array(X), np.array(y)

X, y = create_sequence_flexible(apple_scaled_df, WINDOW_SIZE, N_OUTPUT_STEPS, N_FEATURES)

print(f"Shape of X: {X.shape}") # (n_samples, window_size, n_features)
print(f"Shape of y: {y.shape}") # (n_samples, n_features) para one-step o (n_samples, n_future_steps, n_features) para multi-step

if X.shape[0] == 0:
    raise ValueError("No se pudieron crear secuencias. Verifica los datos y los parámetros WINDOW_SIZE/N_FUTURE_STEPS.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")


model = None
if LOAD_MODEL:
    try:
        model, loaded_scaler,loaded_scaler_ouput, loaded_features, loaded_features_ouput = load_model_and_scalers(model_dir=MODEL_DIR)
        if model is not None:
            scaler = loaded_scaler # Sobrescribir el scaler con el cargado
            features_to_use = loaded_features # Usar las features con las que se entrenó
            print(f"Modelo cargado desde {MODEL_DIR}")
        else:
            print(f"No se encontró modelo en {MODEL_DIR} o falló la carga. Entrenando uno nuevo.")
            LOAD_MODEL = False # Forzar entrenamiento
    except Exception as e:
        print(f"Error al cargar el modelo desde {MODEL_DIR}: {e}. Entrenando uno nuevo.")
        LOAD_MODEL = False # Forzar entrenamiento


if not LOAD_MODEL or model is None:
    print("Entrenando un nuevo modelo...")
    # ------------------------------
    # Entrenamiento del modelo
    # ------------------------------
    model = keras.Sequential()
    model.add(LSTM(units=128, activation='tanh', return_sequences=False, # O True si apilas LSTMs
                   input_shape=(WINDOW_SIZE, N_FEATURES)))

    if PREDICTION_TYPE == 'one_step':
        model.add(Dense(N_FEATURES)) # Salida: (N_FEATURES)
    elif PREDICTION_TYPE == 'multi_step':
        # La capa Dense produce todos los pasos y features aplanados
        model.add(Dense(N_OUTPUT_STEPS * N_FEATURES))
        # Reformateamos para tener (N_OUTPUT_STEPS, N_FEATURES) como salida
        model.add(Reshape((N_OUTPUT_STEPS, N_FEATURES)))

    model.summary()

    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mae', 'mse', 'RootMeanSquaredError'])

    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=10,
                                   restore_best_weights=True)

    history = model.fit(X_train, y_train,
                        validation_split=VALIDATION_SPLIT,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        callbacks=[early_stopping],
                        verbose=1) # Añadido verbose

    save_model_and_scalers(model, scaler, scaler, list(apple_scaled_df.columns), list(apple_scaled_df.columns), MODEL_DIR)
    print(f"Modelo y scalers guardados en {MODEL_DIR}")


# ------------------------------
# Evaluación
# ------------------------------
test_loss_metrics = model.evaluate(X_test, y_test)
print(f"Test Loss (Total): {test_loss_metrics[0]}")
print(f"Test MAE: {test_loss_metrics[1]}")
print(f"Test MSE: {test_loss_metrics[2]}")
print(f"Test RMSE: {test_loss_metrics[3]}")


# ------------------------------
# Predicciones
# ------------------------------
predictions_scaled = model.predict(X_test)
print(f"Shape de predicciones escaladas: {predictions_scaled.shape}")

# Reescalar las predicciones y y_test
# predictions_scaled será (n_samples, n_features) para one-step
# y (n_samples, n_future_steps, n_features) para multi-step.
# y_test tiene la misma forma.

if PREDICTION_TYPE == 'one_step':
    # predictions_scaled ya es (n_samples, n_features)
    # y_test ya es (n_samples, n_features)
    predictions_rescaled = scaler.inverse_transform(predictions_scaled)
    y_test_rescaled = scaler.inverse_transform(y_test)
elif PREDICTION_TYPE == 'multi_step':
    # Necesitamos reescalar cada paso de tiempo.
    # Reshape a 2D, reescala, y luego reshape de vuelta a 3D.
    num_samples = predictions_scaled.shape[0]
    
    predictions_rescaled_flat = scaler.inverse_transform(predictions_scaled.reshape(-1, N_FEATURES))
    predictions_rescaled = predictions_rescaled_flat.reshape(num_samples, N_OUTPUT_STEPS, N_FEATURES)
    
    y_test_rescaled_flat = scaler.inverse_transform(y_test.reshape(-1, N_FEATURES))
    y_test_rescaled = y_test_rescaled_flat.reshape(num_samples, N_OUTPUT_STEPS, N_FEATURES)

print(f"Shape de y_test_rescaled: {y_test_rescaled.shape}")
print(f"Shape de predictions_rescaled: {predictions_rescaled.shape}")

# ------------------------------
# Grafico de resultado
# ------------------------------
plt.figure(figsize=(18, 12))
num_features_to_plot = N_FEATURES

if PREDICTION_TYPE == 'one_step':
    for i in range(num_features_to_plot):
        plt.subplot(3, 2, i + 1) # Asumiendo 6 features
        plt.plot(y_test_rescaled[:, i], color='blue', label=f'Actual {apple_scaled_df.columns[i]}')
        plt.plot(predictions_rescaled[:, i], color='red', linestyle='--', label=f'Predicted {apple_scaled_df.columns[i]}')
        plt.title(f'{apple_scaled_df.columns[i]} - One-Step Prediction')
        plt.xlabel('Time Step en Test Set')
        plt.ylabel(f'{apple_scaled_df.columns[i]}')
        plt.legend()
elif PREDICTION_TYPE == 'multi_step':
    # Para multi-step, graficamos la predicción de N_FUTURE_STEPS para algunos ejemplos del test set.
    # O podemos graficar cómo se desempeña la predicción para un paso específico en el futuro (e.g., el 1er paso, el 5to paso, etc.)
    
    # Opción 1: Mostrar el pronóstico completo para algunas muestras de prueba
    samples_to_plot = min(3, X_test.shape[0]) # Graficar hasta 3 pronósticos completos
    for sample_idx in range(samples_to_plot):
        plt.figure(figsize=(18, 12))
        plt.suptitle(f'Multi-Step Forecast ({N_OUTPUT_STEPS} steps) for Test Sample {sample_idx}', fontsize=16)
        for feature_idx in range(num_features_to_plot):
            plt.subplot(3, 2, feature_idx + 1)
            # El eje X para estos plots será el paso futuro (0 a N_OUTPUT_STEPS-1)
            time_steps_future = np.arange(N_OUTPUT_STEPS)
            plt.plot(time_steps_future, y_test_rescaled[sample_idx, :, feature_idx], color='blue', marker='o', label=f'Actual {apple_scaled_df.columns[feature_idx]}')
            plt.plot(time_steps_future, predictions_rescaled[sample_idx, :, feature_idx], color='red', marker='x', linestyle='--', label=f'Predicted {apple_scaled_df.columns[feature_idx]}')
            plt.title(f'{apple_scaled_df.columns[feature_idx]}')
            plt.xlabel('Future Time Step')
            plt.ylabel('Value')
            plt.legend()
        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Ajustar para suptitle
        plt.show()

    # Opción 2: Graficar la predicción de un paso específico en el futuro a través de todas las muestras de prueba
    # (similar al gráfico one-step pero eligiendo qué paso futuro mostrar)
    step_to_plot = 0 # El primer paso futuro (puedes cambiarlo a N_FUTURE_STEPS-1 para el último)
    plt.figure(figsize=(18, 12))
    plt.suptitle(f'Multi-Step Prediction for Future Step {step_to_plot+1} (across all test samples)', fontsize=16)
    for i in range(num_features_to_plot):
        plt.subplot(3, 2, i + 1)
        plt.plot(y_test_rescaled[:, step_to_plot, i], color='blue', label=f'Actual {apple_scaled_df.columns[i]}')
        plt.plot(predictions_rescaled[:, step_to_plot, i], color='red', linestyle='--', label=f'Predicted {apple_scaled_df.columns[i]}')
        plt.title(f'{apple_scaled_df.columns[i]} - Predicción del {step_to_plot+1}° paso futuro')
        plt.xlabel('Sample en Test Set')
        plt.ylabel(f'{apple_scaled_df.columns[i]}')
        plt.legend()


plt.tight_layout(rect=[0, 0, 1, 0.96]) # Ajustar para suptitle si existe
plt.show()


# ================================================== PRUEBAS CON NUEVOS DATOS (ESCENARIOS) ==================================================
# La lógica de "PRUEBAS" necesitará ser adaptada de forma similar.
# Principalmente, cómo preparas X_prueba y cómo interpretas las predicciones.

escenario_1_id = '5ea95487-0a31-4e33-9263-c31717b81b5e' # Calidad optima
escenario_2_id = 'b3e9e3d7-cc40-484a-a327-18e0f9dac1c7' # Calidad media
escenario_3_id = 'cfbfff06-9780-4b48-883b-bb453d285a75'
escenario_4_id = 'a37de4c6-83a2-4b34-b173-7e79b325c983'

def escenario(escenario_num, data_input):
    # Asegurarse de que las columnas existan antes de filtrar
    required_cols_for_calc = ['roundTripTimeVideo', 'roundTripTimeAudio', 'packetsLostVideo', 'packetsReceivedVideo', 'packetsLostAudio', 'packetsReceivedAudio']
    if not all(col in data_input.columns for col in required_cols_for_calc):
        print("Advertencia: Faltan columnas para calcular delay/packetLossRate en la función escenario.")
        # Podrías retornar data_input o manejar el error como prefieras
    
    # Es mejor calcular estas columnas una vez fuera de esta función si 'data_input' es el dataframe global.
    # Si 'data_input' ya tiene estas columnas, esta parte es redundante o podría causar problemas.
    # Para este ejemplo, asumiré que data_input es el 'data_firestore' original.
    
    # Crear copia para no modificar el dataframe original que se pasa
    df_esc = data_input.copy()

    df_esc['delayVideo'] = df_esc['roundTripTimeVideo'] / 2
    df_esc['delayAudio'] = df_esc['roundTripTimeAudio'] / 2
    df_esc['packetLossRateVideo'] = np.where(
        (df_esc['packetsReceivedVideo'] + df_esc['packetsLostVideo']) == 0, 0,
        (df_esc['packetsLostVideo'] / (df_esc['packetsReceivedVideo'] + df_esc['packetsLostVideo'])) * 100
    )
    df_esc['packetLossRateAudio'] = np.where(
        (df_esc['packetsReceivedAudio'] + df_esc['packetsLostAudio']) == 0, 0,
        (df_esc['packetsLostAudio'] / (df_esc['packetsReceivedAudio'] + df_esc['packetsLostAudio'])) * 100
    )

    if escenario_num == 1:
        return df_esc[df_esc['roomId'] == escenario_1_id].copy()
    if escenario_num == 2:
        return df_esc[df_esc['roomId'] == escenario_2_id].copy()
    if escenario_num == 3:
        # Ejemplo de filtro adicional si es necesario, como en tu código original
        return df_esc[
            (df_esc['roomId'] == escenario_3_id) #& (df_esc['delayVideo'] <= 600)
        ].copy()
    if escenario_num == 4:
        return df_esc[df_esc['roomId'] == escenario_4_id].copy()
    return pd.DataFrame() # Retornar dataframe vacío si el escenario no es válido

# ------------------------------
# Preparación de datos para la predicción con un escenario
# ------------------------------
# Usamos data_firestore original aquí, ya que la función escenario recalcula/selecciona features
df_prueba_escenario = escenario(4, data_firestore) # Elige el escenario 1, 2, 3, o 4

features_for_prediction = [
    'delayVideo', 'delayAudio', 'jitterVideo', 'jitterAudio',
    'packetLossRateVideo', 'packetLossRateAudio', 'date'
]
# Asegurarse de que 'jitterVideo' y 'jitterAudio' existan si no se calcularon en `escenario`
# Si vienen de `data_firestore` directamente, deben estar ahí.
if 'jitterVideo' not in df_prueba_escenario.columns and 'jitterVideo' in data_firestore.columns:
    df_prueba_escenario = pd.merge(df_prueba_escenario, data_firestore[['date', 'jitterVideo', 'jitterAudio']], on='date', how='left')


df_prueba_escenario = df_prueba_escenario[features_for_prediction].copy() # Usar .copy() para evitar SettingWithCopyWarning
df_prueba_escenario.dropna(inplace=True) # Importante
df_prueba_escenario.set_index('date', inplace=True)
df_prueba_escenario.sort_index(inplace=True)

print('df_prueba_escenario Shape:', df_prueba_escenario.shape)
print(df_prueba_escenario.head())
# print(df_prueba_escenario.isna().sum())
# print(df_prueba_escenario.info())

if len(df_prueba_escenario) < WINDOW_SIZE + N_OUTPUT_STEPS:
    print(f"No hay suficientes datos en el escenario para predicción ({len(df_prueba_escenario)}). Se necesitan al menos {WINDOW_SIZE + N_OUTPUT_STEPS}.")
else:
    # Normalizar los datos con el MISMO scaler del entrenamiento
    # Asegurarse que las columnas estén en el mismo orden que cuando se 'fitteó' el scaler
    df_prueba_escenario_cols_ordered = df_prueba_escenario[apple_scaled_df.columns] # Reordenar/seleccionar columnas
    
    scaled_values_prueba = scaler.transform(df_prueba_escenario_cols_ordered)
    scaled_df_prueba = pd.DataFrame(scaled_values_prueba, columns=df_prueba_escenario_cols_ordered.columns, index=df_prueba_escenario_cols_ordered.index)

    X_escenario, y_escenario_actual = create_sequence_flexible(scaled_df_prueba, WINDOW_SIZE, N_OUTPUT_STEPS, N_FEATURES)

    if X_escenario.shape[0] > 0:
        # Realizar predicciones
        predictions_escenario_scaled = model.predict(X_escenario)

        # Reescalar
        if PREDICTION_TYPE == 'one_step':
            predictions_escenario_rescaled = scaler.inverse_transform(predictions_escenario_scaled)
            y_escenario_actual_rescaled = scaler.inverse_transform(y_escenario_actual)
        elif PREDICTION_TYPE == 'multi_step':
            num_samples_esc = predictions_escenario_scaled.shape[0]
            pred_flat = scaler.inverse_transform(predictions_escenario_scaled.reshape(-1, N_FEATURES))
            predictions_escenario_rescaled = pred_flat.reshape(num_samples_esc, N_OUTPUT_STEPS, N_FEATURES)
            
            y_flat = scaler.inverse_transform(y_escenario_actual.reshape(-1, N_FEATURES))
            y_escenario_actual_rescaled = y_flat.reshape(num_samples_esc, N_OUTPUT_STEPS, N_FEATURES)

        # Graficar resultados para el escenario
        # (Similar a la sección de gráficos de prueba, adaptado para los datos del escenario)
        print(f"Predicciones para escenario generadas. Shape: {predictions_escenario_rescaled.shape}")

        # Ejemplo de gráfico para el escenario (similar al de test)
        plt.figure(figsize=(18, 12))
        feature_names = scaled_df_prueba.columns # O apple_scaled_df.columns

        if PREDICTION_TYPE == 'one_step':
            plt.suptitle('Predicciones One-Step para Escenario', fontsize=16)
            for i in range(N_FEATURES):
                plt.subplot(3, 2, i + 1)
                plt.plot(y_escenario_actual_rescaled[:, i], color='blue', label=f'Actual {feature_names[i]}')
                plt.plot(predictions_escenario_rescaled[:, i], color='red', linestyle='--', label=f'Predicción {feature_names[i]}')
                plt.title(f'{feature_names[i]}')
                plt.xlabel('Paso de Tiempo en Escenario')
                plt.ylabel(f'{feature_names[i]}')
                plt.legend()
        elif PREDICTION_TYPE == 'multi_step':
            # Plot del primer pronóstico completo del escenario como ejemplo
            sample_idx_esc = 0 
            plt.suptitle(f'Pronóstico Multi-Step ({N_OUTPUT_STEPS} pasos) para Escenario (Muestra {sample_idx_esc})', fontsize=16)
            for feature_idx in range(N_FEATURES):
                plt.subplot(3, 2, feature_idx + 1)
                time_steps_future = np.arange(N_OUTPUT_STEPS)
                plt.plot(time_steps_future, y_escenario_actual_rescaled[sample_idx_esc, :, feature_idx], color='blue', marker='o', label=f'Actual {feature_names[feature_idx]}')
                plt.plot(time_steps_future, predictions_escenario_rescaled[sample_idx_esc, :, feature_idx], color='red', marker='x', linestyle='--', label=f'Predicción {feature_names[feature_idx]}')
                plt.title(f'{feature_names[feature_idx]}')
                plt.xlabel('Paso Futuro')
                plt.ylabel('Valor')
                plt.legend()
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

        # Tus gráficos originales de solo datos reales del escenario (delay, jitter, packetloss)
        # (Estos no cambian ya que son sobre df_prueba_escenario que ya tiene valores reales)
        def title_custom(field): # Renombré tu función 'title' a 'title_custom' para evitar conflicto con plt.title
            if (field == "delayVideo"): return "Delay Video (ms)"
            if (field == "delayAudio"): return "Delay Audio (ms)"
            # ... (resto de tus títulos)
            return field
        
        plt.figure(figsize=(14,7))
        for i, col in enumerate(df_prueba_escenario.columns[:2]): # delay
             plt.subplot(2,1,i+1)
             plt.plot(range(len(df_prueba_escenario)), df_prueba_escenario[col], color='#0072BD', label=f'Real {title_custom(col)}', marker='o')
             # ... (resto de tu código de ploteo)
        # ... (Repetir para jitter y packetLossRate)
        # plt.show() # Muestra cada figura individualmente o todas al final
        print("Plots de datos del escenario listos (descomentar plt.show() si es necesario).")

    else:
        print("No hay suficientes datos en X_escenario para hacer predicciones.")