import cv2
import numpy as np
import sqlite3

# Base de datos y ruta de la imagen
histograma_bd = "histograma_bd.db"
ruta_imagen = 'IMG_20241006_192650.jpg'

# Conectar a la base de datos
conexion = sqlite3.connect(histograma_bd)
cursor = conexion.cursor()

# Crear la tabla si no existe
cursor.execute('''
    CREATE TABLE IF NOT EXISTS histograma (
        rgb_id INTEGER PRIMARY KEY,
        rojo INTEGER,
        verde INTEGER,
        azul INTEGER
    )
''')
conexion.commit()

# Leer la imagen
imagen = cv2.imread(ruta_imagen)
if imagen is None:
    print("Error al cargar la imagen.")
    exit()

# Calcular el histograma para cada canal (sin normalizar)
hist_rojo = cv2.calcHist([imagen], [2], None, [256], [0, 256]).flatten()
hist_verde = cv2.calcHist([imagen], [1], None, [256], [0, 256]).flatten()
hist_azul = cv2.calcHist([imagen], [0], None, [256], [0, 256]).flatten()

# Insertar o actualizar cada valor en la base de datos
for i in range(256):
    cursor.execute('''
        INSERT OR REPLACE INTO histograma (rgb_id, rojo, verde, azul) 
        VALUES (?, ?, ?, ?)
    ''', (i, int(hist_rojo[i]), int(hist_verde[i]), int(hist_azul[i])))

# Guardar los cambios y cerrar la conexión
conexion.commit()
print("Histograma almacenado en la base de datos en filas separadas para cada nivel de intensidad.")
conexion.close()
