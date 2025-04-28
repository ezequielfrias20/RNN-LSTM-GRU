import numpy as np
from google.cloud import firestore
from google.oauth2 import service_account
import json
import os
from dotenv import load_dotenv
import pandas as pd

"""
    Datos a recolectar: 

    - Jitter (ms)
    - delay (ms): RTT / 2
    - Tasa de perdidas de paquetes (%):  (packetLost/ (packetLost + packetsReceived)) * 100
    - Throughput (bits/ ms):  ((diferencia de bytesSent) / (diferencia timestamp)) * 8

"""

def get_data_firestore_return_array(collection_name, fields_to_extract):
    """
    Obtiene datos de Firestore y los convierte en un array de NumPy.

    Args:
        collection_name (str): Nombre de la colección en Firestore.
        fields_to_extract (list): Lista de campos a extraer de cada documento.

    Returns:
        numpy.ndarray: Array con los datos extraídos, listo para entrenamiento.
    """
    # Configurar las credenciales de Firebase
    creds_json = json.loads(os.getenv('FIREBASE_CREDENTIALS'))
    creds = service_account.Credentials.from_service_account_info(creds_json)
    db = firestore.Client(credentials=creds)

    # Obtener todos los documentos de la colección
    docs = db.collection(collection_name).stream()

    # Extraer los datos y convertirlos en una lista de diccionarios
    data = []
    for doc in docs:
        doc_data = doc.to_dict()
        # Extraer solo los campos especificados
        row = [doc_data.get(field, None) for field in fields_to_extract]
        data.append(row)

    # Convertir la lista en un array de NumPy
    np_data = np.array(data, dtype=np.float32)  # Asegura que los datos sean numéricos

    return np_data

def get_data_firestore(collection_name, fields_to_extract, path = '../../firebase-creds.json'):
    with open(path) as f:
        creds_json = json.load(f)
    """
    Obtiene datos de Firestore y los convierte en un array de NumPy.

    Args:
        collection_name (str): Nombre de la colección en Firestore.
        fields_to_extract (list): Lista de campos a extraer de cada documento.

    Returns:
        numpy.ndarray: Array con los datos extraídos, listo para entrenamiento.
    """
    # Configurar las credenciales de Firebase
    # creds_json = json.loads(os.getenv('FIREBASE_CREDENTIALS'))
    creds = service_account.Credentials.from_service_account_info(creds_json)
    db = firestore.Client(credentials=creds)

    # Obtener todos los documentos de la colección
    docs = db.collection(collection_name).order_by('timestamp', direction=firestore.Query.ASCENDING).stream()

    # Extraer los datos y convertirlos en una lista de diccionarios
    data = []
    for doc in docs:
        doc_data = doc.to_dict()
        # Extraer solo los campos especificados
        
        row = [doc_data.get(field, None) for field in fields_to_extract]
        data.append(doc_data if len(fields_to_extract) == 0 else row)

    return data

def get_data_firestore_df(collection_name, fields_to_extract=None, path='../../firebase-creds.json', force_refresh=False):
    """
    Obtiene datos de Firestore y los convierte en un DataFrame.
    
    Args:
        collection_name (str): Nombre de la colección en Firestore.
        fields_to_extract (list, optional): Lista de campos a extraer de cada documento. Si None, extrae todos.
        path (str, optional): Ruta al archivo de credenciales de Firebase.
        force_refresh (bool, optional): Si True, fuerza la actualización desde Firestore.

    Returns:
        pandas.DataFrame: DataFrame con los datos extraídos.
    """
    
    csv_filename = f'{collection_name}.csv'

    if os.path.exists(csv_filename) and not force_refresh:
        df = pd.read_csv(csv_filename)
        # Asegurarse que si hay columna "date", sea tipo datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
        return df

    # Cargar credenciales
    with open(path) as f:
        creds_json = json.load(f)

    creds = service_account.Credentials.from_service_account_info(creds_json)
    db = firestore.Client(credentials=creds)

    # Obtener documentos de la colección ordenados por 'date'
    docs = db.collection(collection_name).order_by('date', direction=firestore.Query.ASCENDING).stream()

    data = []
    for doc in docs:
        doc_data = doc.to_dict()
        if fields_to_extract:
            row = {field: doc_data.get(field, None) for field in fields_to_extract}
        else:
            row = doc_data
        data.append(row)

    # Convertir a DataFrame
    df = pd.DataFrame(data)

    # Si hay campo "date", parsearlo y ordenar
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

    # Guardar en CSV
    df.to_csv(csv_filename, index=False)

    return df