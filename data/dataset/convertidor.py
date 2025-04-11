import pandas as pd

# Definir todas las columnas del archivo original
columnas_completas = [
    "RequestID", "NbClients", "BottleneckBW", "BottleneckDelay", "BottleneckLoss", 
    "DASHPolicy", "ClientResolution", "RequestDuration", "TCPOutputPacket", 
    "TCPOutputDelay", "TCPOutputJitter", "TCPOutputPloss", "TCPInputPacket", 
    "TCPInputDelay", "TCPInputJitter", "TCPInputPloss", "TCPInputRetrans", 
    "StdInputRate", "0_InputRateVariation", "5_InputRateVariation", 
    "10_InputRateVariation", "25_InputRateVariation", "50_InputRateVariation", 
    "75_InputRateVariation", "90_InputRateVariation", "95_InputRateVariation", 
    "100_InputRateVariation", "StdInterATimesReq", "0_InterATimesReq", 
    "5_InterATimesReq", "10_InterATimesReq", "25_InterATimesReq", 
    "50_InterATimesReq", "75_InterATimesReq", "90_InterATimesReq", 
    "95_InterATimesReq", "100_InterATimesReq", "StartUpDelay", 
    "AvgDownloadRate", "StdDownloadRate", "AvgBufferLevel", "StdBufferLevel", 
    "StallEvents", "RebufferingRatio", "StallLabel", "TotalStallingTime", 
    "AvgTimeStallingEvents", "AvgQualityIndex", "AvgVideoBitRate", 
    "AvgVideoQualityVariation", "AvgDownloadBitRate"
]

# Columnas que quieres conservar
columnas_deseadas = [
    "TCPOutputPacket", "TCPOutputDelay", "TCPOutputJitter", "TCPOutputPloss", 
    "TCPInputPacket", "TCPInputDelay", "TCPInputJitter", "TCPInputPloss", 
    "TCPInputRetrans"
]

# Leer el archivo TXT
df = pd.read_csv('dataset.txt', sep='\s+', header=None, names=columnas_completas)

# Seleccionar solo las columnas deseadas
df_seleccionado = df[columnas_deseadas]

# Guardar como CSV
df_seleccionado.to_csv('datos_tcp.csv', index=False)

print("Archivo convertido exitosamente a datos_tcp.csv con las columnas seleccionadas")