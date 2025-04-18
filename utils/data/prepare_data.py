import numpy as np
from google.cloud import firestore
from google.oauth2 import service_account
import json
import os
from dotenv import load_dotenv

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