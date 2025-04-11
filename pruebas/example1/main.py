# Primer ejercicio para predecir valores en la bolsa de valores ==> VIDEO: https://www.youtube.com/watch?v=94PlBzgeq90
from tensorflow import keras # Para construir el modelo
import pandas as pd # Para cargar nuestros datos
import numpy as np # Crear matrices en 3D
from sklearn.preprocessing import StandardScaler # Preprocesamiento de datos, escalador estandar y una varianza a nuestros datos
import matplotlib.pyplot as plt # Gráficas
import seaborn as sns # Gráficas avanzadas
import os # Sistema operativo con python
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # SUPRIMIR ALGUNAS ADVERTENCIAS DE TENSORFLOW

data = pd.read_csv("data/MicrosoftStock.csv")

print(data.head()) # Head de datos, te da las primeras 5 filas 
print(data.info()) # Información básica Tipo de datos
print(data.describe()) # Descripción general, te da estadicsticas generales (La media, la desviación estandar, el minimo, )

# Initial Data Visualization
# Plot 1 - Open and Close Prices of time
plt.figure(figsize=(12,8)) #(ancho, altura)
plt.plot(data['date'], data['open'], label= 'Open', color="blue")
plt.plot(data['date'], data['close'], label= 'Close', color="red")
plt.title("Open-Close Price over Time")
plt.legend() # Poner una leyenda
# plt.show() # Mostrat Grafico 


# Plot 2 - Trading Volume (check for outliers)
plt.figure(figsize=(12,8)) #(ancho, altura)
plt.plot(data['date'], data['volume'], label= 'Volume', color="orange")
plt.title("Stock Volume over Time")
# plt.show() # Mostrat Grafico 

# Eliminar law columnas no numericas
numeric_data = data.select_dtypes(include=["int64", "float64"])

# Plot 3 - Check for correlation between feautures
plt.figure(figsize=(8,6))
sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm") # Mapa de calor
plt.title("Feature Correlation Heatmap")
# plt.show()

# Convertir la hora en un Date time y luego crea un filtro de fecha
data['date'] = pd.to_datetime(data['date'])

# El filtro nos ayuda a poner un rango de fecha 
prediction = data.loc[
    (data['date'] > datetime(2013,1,1)) &
    (data['date'] < datetime(2018,1,1))
]

# Plot 4 - Price over Time
plt.plot(data['date'], data['close'], color="blue")
plt.xlabel("Date")
plt.ylabel("Close")
plt.title("Price over Time")

# Prepare for the LSTM Model (Sequential)
stock_close = data.filter(['close']) # Solo quiero los datos de cierre

# Convertir en una matriz de numpy
dataset = stock_close.values # convert to numpy array

#longitud del array
training_data_len = int(np.ceil(len(dataset) * 0.95)) # Solo quiero el 95% de los datos

# Preprocessing Stages ( Es como dar la media y la varianza en numpy)
scaler = StandardScaler() # Vamos a escalar los datos
scaled_data  = scaler.fit_transform(dataset)

# Datos de entrenamiento
training_data = scaled_data[:training_data_len] # Los datos hasta el 95% de ellos 

# Caracteristicas de entrenamiento
x_train, y_train = [], [] # Estos son los datos que le damos al modelo para que aprenda, La x son todas las caracteristicas y la y son todos los precios de cierre

# Create a sliding window for our stock (60 Days)
for i in range(60, len(training_data)):
    x_train.append(training_data[i - 60: i , 0])
    y_train.append(training_data[i, 0])

# Ahora hay que convertirlas en array de numpy
x_train, y_train = np.array(x_train), np.array(y_train) # Matrices

# Hay que llevarla a matrices en 3D para que el tensor pueda interpretarlo. Las LSTM esperan datos en 3D:
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) ## Formato para LSTM: (muestras, pasos_temporales, características)

# Contruir el Modelo
model = keras.models.Sequential()

# Primera capa
model.add(keras.layers.LSTM(64, return_sequences=True, input_shape=(x_train.shape[1],1)))

# Segunda capa
model.add(keras.layers.LSTM(64, return_sequences=False))

# Tercera capa
model.add(keras.layers.Dense(128, activation="relu")) # Activacion para series de tiempo

# Cuarta capa: Evitar el sobreajuste
model.add(keras.layers.Dropout(0.5))

# Capa final:
model.add(keras.layers.Dense(1)) # En la capa de salida se usa cada neurona dependiendo de la cantidad de reasultados que quieres que te retorne

# Si quisiera ver como se realizo el modelo
model.summary()
# Compilar el modelo
model.compile(
    optimizer="adam",
    loss="mae",
    metrics=[keras.metrics.RootMeanSquaredError()]
)

# Entrenar el model
training  = model.fit(x_train, y_train, epochs= 20, batch_size = 32)

# Preparar los datos de prueba
test_data = scaled_data[training_data_len-60:]
x_test, y_test = [], dataset[training_data_len:]

# Crearemos las secuencias de prueba
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1 ))

# Hacer las predicciones

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Plotting Data
train = data[:training_data_len]
test = data[training_data_len:]

test = test.copy()

test['Predictions'] = predictions

plt.figure(figsize=(12,8)) #(ancho, altura)
plt.plot(train['date'], train['close'], label="Train (Actual)", color="blue")
plt.plot(test['date'], test['close'], label="Test (Actual)", color="orange")
plt.plot(test['date'], test['Predictions'], label="Predictions", color="red")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.title("Our Stock Prediction")
plt.legend()
plt.show()