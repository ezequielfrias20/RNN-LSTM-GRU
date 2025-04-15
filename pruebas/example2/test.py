from utils.graphic import generar_grafico
from utils.data.prepare_data import get_data_firestore
from utils.data.preprocess_data import preprocess_data, preprocess_data_2, preprocess_data_3
from utils.data.scaler_data import prepare_sequences
from utils.models.LSTM.model_1 import build_model, build_model_2
from utils.save import save_model_and_scalers

data = get_data_firestore('metrics', [])

# Configuración
LOOKBACK = 60  # Número de pasos anteriores a considerar para la predicción
# Número de pasos a predecir (asumiendo datos cada 5 segundos para 2 minutos)
PREDICTION_HORIZON = 24
# Número de muestras que se procesan antes de actualizar los pesos del modelo, Valores típicos: 32, 64, 128. El modelo verá 32 muestras por cada actualización.
BATCH_SIZE = 32
# Epocas. Número de veces que el modelo recorre todo el dataset durante el entrenamiento.
EPOCHS = 100
# Fracción de los datos de entrenamiento que se usará para validación (evaluación durante el entrenamiento). Si es 0.2, el 20% de X_train/y_train se usa para validar (no se aprende de ellos).
VALIDATION_SPLIT = 0.2

df, input_features, output_features = preprocess_data_3(data)

# Preparar secuencias
X, y, input_scaler, output_scaler = prepare_sequences(
    df, input_features, output_features, LOOKBACK, PREDICTION_HORIZON
)
