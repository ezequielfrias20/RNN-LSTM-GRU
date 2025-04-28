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
df = pd.DataFrame(data) # Convertir a dataframe
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')
# df = pd.get_dummies(df, columns=['networkType', 'roomId'])

escenario_1 = df[df['roomId'] == '5ea95487-0a31-4e33-9263-c31717b81b5e']
escenario_2 = df[df['roomId'] == 'b3e9e3d7-cc40-484a-a327-18e0f9dac1c7']
escenario_3 = df[df['roomId'] == 'cfbfff06-9780-4b48-883b-bb453d285a75']
escenario_4 = df[df['roomId'] == 'a37de4c6-83a2-4b34-b173-7e79b325c983']

# =================
# TOTAL DE DATA QOS
# =================

# Crear figura con 2 subplots (2 filas, 1 columna)
fig, axs = plt.subplots(2, 2, figsize=(16, 8))  # 2 filas, 1 columna

# Gráfico 1
axs[0, 0].plot(indices, df['jitterAudio'], color='blue')
axs[0, 0].set_title('Jitter Audio')
axs[0, 0].grid(True)

# Gráfico 2
axs[0, 1].plot(indices, df['jitterVideo'], color='red')
axs[0, 1].set_title('Jitter Video')
axs[0, 1].grid(True)

# Gráfico 3
axs[1, 0].plot(indices, df['roundTripTimeAudio'], color='orange')
axs[1, 0].set_title('RTT Audio')
axs[1, 0].grid(True)

# Gráfico 4
axs[1, 1].plot(indices, df['roundTripTimeVideo'], color='green')
axs[1, 1].set_title('RTT Video')
axs[1, 1].grid(True)

plt.suptitle('Parametros QoS', fontsize=14)
plt.tight_layout()
plt.show()

# =================
# ESCENARIO 1 - CALIDAD OPTIMA
# =================

# Gráfico 1
axs[0, 0].plot(range(len(escenario_1)), escenario_1['jitterAudio'], color='blue')
axs[0, 0].set_title('Jitter Audio')
axs[0, 0].grid(True)

# Gráfico 2
axs[0, 1].plot(range(len(escenario_1)), escenario_1['jitterVideo'], color='red')
axs[0, 1].set_title('Jitter Video')
axs[0, 1].grid(True)

# Gráfico 3
axs[1, 0].plot(range(len(escenario_1)), escenario_1['roundTripTimeAudio'], color='orange')
axs[1, 0].set_title('RTT Audio')
axs[1, 0].grid(True)

# Gráfico 4
axs[1, 1].plot(range(len(escenario_1)), escenario_1['roundTripTimeVideo'], color='green')
axs[1, 1].set_title('RTT Video')
axs[1, 1].grid(True)

plt.suptitle('Escenario 1 - Calidad Optima', fontsize=14)
plt.tight_layout()
plt.show()

# =================
# ESCENARIO 2 - DEGRADACION CRITICA
# =================

# Crear figura con 2 subplots (2 filas, 1 columna)
fig, axs = plt.subplots(2, 2, figsize=(16, 8))  # 2 filas, 1 columna

# Gráfico 1
axs[0, 0].plot(range(len(escenario_2)), escenario_2['jitterAudio'], color='blue')
axs[0, 0].set_title('Jitter Audio')
axs[0, 0].grid(True)

# Gráfico 2
axs[0, 1].plot(range(len(escenario_2)), escenario_2['jitterVideo'], color='red')
axs[0, 1].set_title('Jitter Video')
axs[0, 1].grid(True)

# Gráfico 3
axs[1, 0].plot(range(len(escenario_2)), escenario_2['roundTripTimeAudio'], color='orange')
axs[1, 0].set_title('RTT Audio')
axs[1, 0].grid(True)

# Gráfico 4
axs[1, 1].plot(range(len(escenario_2)), escenario_2['roundTripTimeVideo'], color='green')
axs[1, 1].set_title('RTT Video')
axs[1, 1].grid(True)

plt.suptitle('Escenario 2 - Degradacion Critica', fontsize=14)
plt.tight_layout()
plt.show()

# =================
# ESCENARIO 3 - CALIDAD CRITICA
# =================

# Gráfico 1
axs[0, 0].plot(range(len(escenario_3)), escenario_3['jitterAudio'], color='blue')
axs[0, 0].set_title('Jitter Audio')
axs[0, 0].grid(True)

# Gráfico 2
axs[0, 1].plot(range(len(escenario_3)), escenario_3['jitterVideo'], color='red')
axs[0, 1].set_title('Jitter Video')
axs[0, 1].grid(True)

# Gráfico 3
axs[1, 0].plot(range(len(escenario_3)), escenario_3['roundTripTimeAudio'], color='orange')
axs[1, 0].set_title('RTT Audio')
axs[1, 0].grid(True)

# Gráfico 4
axs[1, 1].plot(range(len(escenario_3)), escenario_3['roundTripTimeVideo'], color='green')
axs[1, 1].set_title('RTT Video')
axs[1, 1].grid(True)

plt.suptitle('Escenario 3 - Calidad Critica', fontsize=14)
plt.tight_layout()
plt.show()

# =================
# ESCENARIO 4 - CONDICIONES EXTREMAS
# =================

# Gráfico 1
axs[0, 0].plot(range(len(escenario_4)), escenario_4['jitterAudio'], color='blue')
axs[0, 0].set_title('Jitter Audio')
axs[0, 0].grid(True)

# Gráfico 2
axs[0, 1].plot(range(len(escenario_4)), escenario_4['jitterVideo'], color='red')
axs[0, 1].set_title('Jitter Video')
axs[0, 1].grid(True)

# Gráfico 3
axs[1, 0].plot(range(len(escenario_4)), escenario_4['roundTripTimeAudio'], color='orange')
axs[1, 0].set_title('RTT Audio')
axs[1, 0].grid(True)

# Gráfico 4
axs[1, 1].plot(range(len(escenario_4)), escenario_4['roundTripTimeVideo'], color='green')
axs[1, 1].set_title('RTT Video')
axs[1, 1].grid(True)

plt.suptitle('Escenario 4 - Condiciones Extremas', fontsize=14)
plt.tight_layout()
plt.show()


df, input_features, output_features = preprocess_data(data) # Preprocesamiento de la data

print(df.head()) # Head de datos, te da las primeras 5 filas 
print(df.info()) # Información básica Tipo de datos
print(df.describe()) # Descripción general, te da estadicsticas generales (La media, la desviación estandar, el minimo, )



