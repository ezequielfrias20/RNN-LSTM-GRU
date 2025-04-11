import matplotlib.pyplot as plt # Gráficas
import numpy as np
from typing import Optional, List, Dict, Union


def generar_grafico(
    x: Union[List, np.ndarray],
    y: Union[List, np.ndarray],
    tipo: str = "linea",
    titulo: str = "",
    xlabel: str = "",
    ylabel: str = "",
    etiquetas: Optional[List[str]] = None,
    colores: Optional[List[str]] = None,
    estilo: str = "ggplot",
    guardar: bool = False,
    nombre_archivo: str = "grafico.png",
    tamaño: tuple = (10, 6),
    **kwargs
) -> None:
    """
    Genera gráficos personalizados con matplotlib.

    Parámetros:
    -----------
    x : Array-like
        Datos para el eje X (puede ser lista o numpy array).
    y : Array-like o List[Array-like]
        Datos para el eje Y. Si hay múltiples series, pasar lista de arrays.
    tipo : str ('linea', 'barras', 'dispersion', 'histograma', 'boxplot')
        Tipo de gráfico a generar.
    titulo : str
        Título del gráfico.
    xlabel, ylabel : str
        Etiquetas para los ejes.
    etiquetas : List[str] (opcional)
        Nombres de las series (para leyenda).
    colores : List[str] (opcional)
        Colores para cada serie (ej: ['red', 'blue']).
    estilo : str
        Estilo de matplotlib (ej: 'ggplot', 'seaborn', 'default').
    guardar : bool
        Si True, guarda el gráfico en un archivo.
    nombre_archivo : str
        Nombre del archivo si guardar=True.
    tamaño : tuple
        Tamaño del gráfico (ancho, alto).
    **kwargs
        Argumentos adicionales para personalización (ej: alpha, bins, edgecolor).
    """
    plt.style.use(estilo)
    plt.figure(figsize=tamaño)

    # Configurar colores predeterminados
    if colores is None:
        colores = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Generar gráfico según el tipo
    if tipo == "linea":
        if isinstance(y[0], (list, np.ndarray)):  # Múltiples series
            for i, serie in enumerate(y):
                plt.plot(x, serie, label=etiquetas[i] if etiquetas else None, color=colores[i], **kwargs)
        else:
            plt.plot(x, y, label=etiquetas[0] if etiquetas else None, color=colores[0], **kwargs)
    
    elif tipo == "barras":
        plt.bar(x, y, color=colores, label=etiquetas, **kwargs)
    
    elif tipo == "dispersion":
        plt.scatter(x, y, color=colores[0], label=etiquetas[0] if etiquetas else None, **kwargs)
    
    elif tipo == "histograma":
        plt.hist(y, bins=kwargs.get('bins', 10), color=colores[0], edgecolor='black', alpha=0.7)
    
    elif tipo == "boxplot":
        plt.boxplot(y, labels=etiquetas, patch_artist=True, boxprops=dict(facecolor=colores[0]))
    
    else:
        raise ValueError("Tipo de gráfico no soportado. Usar: 'linea', 'barras', 'dispersion', 'histograma', 'boxplot'")

    # Añadir elementos comunes
    plt.title(titulo)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if etiquetas:
        plt.legend()
    plt.grid(True)

    if guardar:
        plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
    plt.show()