from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from dotenv import load_dotenv
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
import pandas as pd

from utils.data.prepare_data import get_data_firestore

from utils.data.preprocess_data import preprocess_data_5
from utils.data.scaler_data import prepare_sequences_5
from utils.models.LSTM.model_1 import build_model_3, build_model_4, build_model_6

pd.options.mode.chained_assignment = None

# Configuración
LOOKBACK = 60  # Número de pasos anteriores a considerar para la predicción
# Número de pasos a predecir (asumiendo datos cada 5 segundos para 2 minutos)
PREDICTION_HORIZON = 24
# Número de muestras que se procesan antes de actualizar los pesos del modelo, Valores típicos: 32, 64, 128. El modelo verá 32 muestras por cada actualización.
BATCH_SIZE = 32
# Epocas. Número de veces que el modelo recorre todo el dataset durante el entrenamiento.
EPOCHS = 1000
# Fracción de los datos de entrenamiento que se usará para validación (evaluación durante el entrenamiento). Si es 0.2, el 20% de X_train/y_train se usa para validar (no se aprende de ellos).
VALIDATION_SPLIT = 0.2

NEW_MODEL = True

# Obtener array de datos
data = get_data_firestore('metrics', [])

# Preprocesamiento
df, input_features, output_features = preprocess_data_5(data)

# Número de datos de los sets de entrenamiento, validación y prueba
N = df.shape[0]  # Cantidad total de datos
NTRAIN = int(0.7*N)
NVAL = int(0.15*N)
NTEST = N - NTRAIN - NVAL

# Seleccionar las filas correspondientes
df_train = df.iloc[0:NTRAIN, :]
df_val = df.iloc[NTRAIN:NTRAIN+NVAL, :]
df_test = df.iloc[NTRAIN+NVAL:, :]

# Imprimir información en pantalla
print(f'Tamaño set original: {df.shape}')
print(f'Tamaño set de entrenamiento: {df_train.shape}')
print(f'Tamaño set de validación: {df_val.shape}')
print(f'Tamaño set de prueba: {df_test.shape}')

X_train, y_train, X_val, y_val, X_test, y_test, scaler = prepare_sequences_5(
    df_train, df_val, df_test, input_features, output_features, LOOKBACK, PREDICTION_HORIZON)

print(f'Tamaño de X_train: {X_train.shape}')

# MODELO LSTM
model = build_model_6((LOOKBACK, len(input_features)),
                      PREDICTION_HORIZON, output_features, input_features, LOOKBACK)


# Entrenar
early_stop = EarlyStopping( 
    monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=VALIDATION_SPLIT,
    callbacks=[early_stop],
    # Controla cuánta información se muestra durante el entrenamiento. 1: Muestra barra de progreso y métricas por época (recomendado).
    verbose=1
)


# Evaluar
metrics_dict = model.evaluate(X_val, y_val, verbose=0)
print("Loss:", metrics_dict[0])
print("MAE:", metrics_dict[1])
print("MSE:", metrics_dict[2])
print("RMSE:", metrics_dict[3])


plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.show()

# y_pred = model.predict(X_test)

# n_features = len(output_features)
# plt.figure(figsize=(18, 3 * n_features))

# # Seleccionar paso del horizonte (ej: primer paso)
# HORIZON_STEP = 5  # 0 para el primer paso, 1 para el segundo, etc.

# # Reestructurar y_test y y_pred a 3D (muestras, horizonte, características)
# n_samples = y_test.shape[0]
# y_test_3d = y_test.reshape(n_samples, PREDICTION_HORIZON, len(output_features))
# y_pred_3d = y_pred.reshape(n_samples, PREDICTION_HORIZON, len(output_features))

# for i, feature in enumerate(output_features, 1):
#     feature_idx = output_features.index(feature)
#     y_test_feature = y_test_3d[:, HORIZON_STEP, feature_idx]
#     y_pred_feature = y_pred_3d[:, HORIZON_STEP, feature_idx]

#     plt.subplot(n_features, 1, i)
#     plt.plot(y_test_feature, label='Real')
#     plt.plot(y_pred_feature, label='Predicho', linestyle='--')
#     plt.title(f'{feature} (Paso {HORIZON_STEP + 1})')
#     plt.legend()
#     plt.grid(True)

# plt.tight_layout()
# plt.show()
