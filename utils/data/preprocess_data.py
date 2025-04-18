import pandas as pd
from datetime import datetime


def preprocess_data(data):
    # Convertir datos JSON a DataFrame
    df = pd.DataFrame(data)

    # Convertir date a datetime y ordenar
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    # df = df.sort_values('timestamp')

    # Codificar variables categóricas (como networkType)
    # Conviertes los string en valores
    df = pd.get_dummies(df, columns=['networkType', 'roomId'])

    # # Calcular variables objetivo
    df['delayVideo'] = df['roundTripTimeVideo']
    df['delayAudio'] = df['roundTripTimeAudio']
    # df['packetLossRateVideo'] = df['packetsLostVideo'] / \
    #     (df['packetsReceivedVideo'] + df['packetsLostVideo'])
    # df['packetLossRateAudio'] = df['packetsLostAudio'] / \
    #     (df['packetsReceivedAudio'] + df['packetsLostAudio'])
    # df['throughputVideo'] = df['bytesReceivedVideo'] * \
    #     8 / (5 * 1024)  # kbps (asumiendo intervalos de 5s)
    # df['throughputAudio'] = df['bytesReceivedAudio'] * 8 / (5 * 1024)  # kbps

    # Definir características de entrada y salida
    input_features = [
        'delayVideo',
        'packetsReceivedVideo', 'roundTripTimeAudio', 'bytesReceived',
        'jitterAudio', 'bytesSentAudio', 'packetsReceivedAudio',
        'bytesReceivedVideo', 'bytesSentVideo', 'bytesSent',
        'packetsLostAudio', 'bytesReceivedAudio', 'availableOutgoingBitrate',
        'jitterVideo',
        'roundTripTimeVideo', 'packetsLostVideo',
    ]
    output_features =[
        # 'delayVideo',
        # 'delayAudio',
        # 'jitterVideo',
        # 'jitterAudio',
        # 'packetLossRateVideo', 'packetLossRateAudio',
        # 'throughputVideo', 'throughputAudio'
    ]

    output_features =[
        'delayVideo',
        'packetsReceivedVideo', 'roundTripTimeAudio', 'bytesReceived',
        'jitterAudio', 'bytesSentAudio', 'packetsReceivedAudio',
        'bytesReceivedVideo', 'bytesSentVideo', 'bytesSent',
        'packetsLostAudio', 'bytesReceivedAudio', 'availableOutgoingBitrate',
        'jitterVideo',
        'roundTripTimeVideo', 'packetsLostVideo',
    ]

    return df, input_features, output_features


def preprocess_data_2(df):
    df['date'] = pd.to_datetime(df['date'])
    prediction = df.loc[
        (df['date'] > datetime(2013, 1, 1)) &
        (df['date'] < datetime(2018, 1, 1))
    ]
    stock_close = df.filter(['close'])

    # Definir características de entrada y salida
    input_features = [
        'close',
    ]

    output_features = [
        'close',
    ]

    return stock_close, input_features, output_features


def preprocess_data_3(data):
    # Convertir datos JSON a DataFrame
    df = data

    # Definir características de entrada y salida
    input_features = [
        "TCPOutputPacket", "TCPOutputDelay", "TCPOutputJitter", "TCPOutputPloss",
        "TCPInputPacket", "TCPInputDelay", "TCPInputJitter", "TCPInputPloss",
        "TCPInputRetrans"
    ]

    output_features = [
        "TCPOutputPacket", "TCPOutputDelay", "TCPOutputJitter", "TCPOutputPloss",
        "TCPInputPacket", "TCPInputDelay", "TCPInputJitter", "TCPInputPloss",
        "TCPInputRetrans"
    ]

    return df, input_features, output_features

def preprocess_data_5(data):
    # Convertir datos JSON a DataFrame
    df = pd.DataFrame(data)

    # Convertir timestamp a datetime y ordenar
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # Codificar variables categóricas (como networkType)
    # Conviertes los string en valores
    df = pd.get_dummies(df, columns=['networkType', 'roomId'])

    # Calcular variables objetivo
    df['delayVideo'] = df['roundTripTimeVideo']
    df['delayAudio'] = df['roundTripTimeAudio']
    df['packetLossRateVideo'] = df['packetsLostVideo'] / \
        (df['packetsReceivedVideo'] + df['packetsLostVideo'])
    df['packetLossRateAudio'] = df['packetsLostAudio'] / \
        (df['packetsReceivedAudio'] + df['packetsLostAudio'])
    df['throughputVideo'] = df['bytesReceivedVideo'] * \
        8 / (5 * 1024)  # kbps (asumiendo intervalos de 5s)
    df['throughputAudio'] = df['bytesReceivedAudio'] * 8 / (5 * 1024)  # kbps

    df_filtrado = df[df['jitterVideo'] != 0]

    # Definir características de entrada y salida
    input_features = [
        'delayVideo', 'delayAudio',
        'jitterVideo',
        'jitterAudio',
        'packetLossRateVideo', 'packetLossRateAudio',
        'throughputVideo', 'throughputAudio'
    ]

    output_features = [
        'delayVideo', 'delayAudio',
        'jitterVideo',
        'jitterAudio',
        'packetLossRateVideo', 'packetLossRateAudio',
        'throughputVideo', 'throughputAudio'
    ]

    return df_filtrado[input_features], input_features, output_features
