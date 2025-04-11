import tensorflow as tf
import joblib
import json
import os

def save_model_and_scalers(model, input_scaler, output_scaler, input_features, output_features, model_dir="saved_model"):
    # Crear directorio si no existe
    os.makedirs(model_dir, exist_ok=True)
    
    # Guardar el modelo de TensorFlow/Keras
    model.save(os.path.join(model_dir, "tcp_model.keras"))
    
    # Guardar los scalers
    joblib.dump(input_scaler, os.path.join(model_dir, "input_scaler.save"))
    joblib.dump(output_scaler, os.path.join(model_dir, "output_scaler.save"))
    
    # Guardar las características (features)
    with open(os.path.join(model_dir, "features.json"), "w") as f:
        json.dump({
            "input_features": input_features,
            "output_features": output_features
        }, f)

def load_model_and_scalers(model_dir="saved_model"):
    # Cargar el modelo
    model = tf.keras.models.load_model(os.path.join(model_dir, "tcp_model.keras"))
    
    # Cargar los scalers
    input_scaler = joblib.load(os.path.join(model_dir, "input_scaler.save"))
    output_scaler = joblib.load(os.path.join(model_dir, "output_scaler.save"))
    
    # Cargar las características
    with open(os.path.join(model_dir, "features.json"), "r") as f:
        features = json.load(f)
    
    return model, input_scaler, output_scaler, features["input_features"], features["output_features"]