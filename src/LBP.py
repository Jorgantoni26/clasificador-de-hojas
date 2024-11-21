import cv2
import numpy as np
from skimage import feature, color
import sqlite3
import os

def extract_texture(image):
    # Convertir la imagen a escala de grises
    gray_image = color.rgb2gray(image)
    # Calcular el patrón local binario (LBP)
    lbp = feature.local_binary_pattern(gray_image, P=24, R=3, method='uniform')
    # Calcular el histograma de LBP
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 24 + 3), range=(0, 24 + 2))
    # Normalizar el histograma
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def process_images_in_menta(folder_path):
    # Conectar a la base de datos
    conn = sqlite3.connect("C:/Users/jorga/Downloads/clasificador-de-hojas/data/database/momentos.db")
    cursor = conn.cursor()
    
    # Procesar cada archivo en la carpeta "menta"
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpeg") or filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            
            if image is not None:
                # Extraer el histograma de textura
                texture_histogram = extract_texture(image)
                
                # Convertir el histograma en una cadena separada por comas
                histogram_str = ','.join(map(str, texture_histogram))
                
                # Insertar el histograma en la columna "textura" de la tabla "menta"
                cursor.execute("UPDATE menta SET textura = ? WHERE nombre_archivo = ?", (histogram_str, filename))
    
    # Confirmar los cambios y cerrar la conexión
    conn.commit()
    conn.close()

# Ruta de la carpeta "menta"
folder_path = "C:/Users/jorga/Downloads/clasificador-de-hojas/data/imagenes/menta_contorneada"
process_images_in_menta(folder_path)
