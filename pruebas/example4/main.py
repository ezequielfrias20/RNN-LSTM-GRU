from tensorflow import keras # Para construir el modelo
import pandas as pd # Para cargar nuestros datos
import numpy as np # Crear matrices en 3D
from sklearn.preprocessing import StandardScaler # Preprocesamiento de datos, escalador estandar y una varianza a nuestros datos
import matplotlib.pyplot as plt # Gráficas
import seaborn as sns # Gráficas avanzadas
import os # Sistema operativo con python
from datetime import datetime
from utils.data.prepare_data import get_data_firestore
from utils.data.preprocess_data import preprocess_data
from dotenv import load_dotenv

load_dotenv()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # SUPRIMIR ALGUNAS ADVERTENCIAS DE TENSORFLOW

data = get_data_firestore('metrics', [])
indices = np.arange(len(data))

jitterAudio = [item["jitterAudio"] for item in data]
jitterVideo = [item["jitterVideo"] for item in data]
roundTripTimeAudio = [item["roundTripTimeAudio"] for item in data]
roundTripTimeVideo = [item["roundTripTimeVideo"] for item in data]

# # Plot 1 - jitter
# plt.figure(figsize=(12,8)) #(ancho, altura)
# plt.plot(indices, jitterAudio, label= 'Jitter Audio', color="blue")
# plt.plot(indices, jitterVideo, label= 'Jitter Video', color="red")
# plt.title("Jitter")
# plt.legend()
# plt.show()

# Crear figura con 2 subplots (2 filas, 1 columna)
fig, axs = plt.subplots(2, 2, figsize=(16, 8))  # 2 filas, 1 columna

# Gráfico 1
axs[0, 0].plot(indices, jitterAudio, color='blue')
axs[0, 0].set_title('Jitter Audio')
axs[0, 0].grid(True)

# Gráfico 2
axs[0, 1].plot(indices, jitterVideo, color='red')
axs[0, 1].set_title('Jitter Video')
axs[0, 1].grid(True)

# Gráfico 3
axs[1, 0].plot(indices, roundTripTimeAudio, color='orange')
axs[1, 0].set_title('RTT Audio')
axs[1, 0].grid(True)

# Gráfico 4
axs[1, 1].plot(indices, roundTripTimeVideo, color='green')
axs[1, 1].set_title('RTT Video')
axs[1, 1].grid(True)

plt.suptitle('Parametros QoS', fontsize=14)
plt.tight_layout()
# plt.show()


df, input_features, output_features = preprocess_data(data) # Preprocesamiento de la data

print(df.head()) # Head de datos, te da las primeras 5 filas 
print(df.info()) # Información básica Tipo de datos
print(df.describe()) # Descripción general, te da estadicsticas generales (La media, la desviación estandar, el minimo, )



