import cv2
import numpy as np
import os
import sqlite3

# Carpetas de entrada y salida
carpeta_entrada = "C:/Users/jorga/Downloads/clasificador-de-hojas/data/imagenes/parra"
carpeta_salida = "C:/Users/jorga/Downloads/clasificador-de-hojas/data/imagenes/parra_contorneada"
print(os.path.exists("C:/Users/jorga/Downloads/MENTA_JORGE_MANZANO/HOJAS RECORTADAS"))

# Conectar a la base de datos SQLite o crearla si no existe
conn = sqlite3.connect("C:/Users/jorga/Downloads/clasificador-de-hojas/data/database/momentos.db")
cursor = conn.cursor()

# Crear la tabla para almacenar los momentos de Hu si no existe
cursor.execute("""
    CREATE TABLE IF NOT EXISTS parra (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nombre_archivo TEXT,
        M1 REAL, M2 REAL, M3 REAL, M4 REAL, M5 REAL, M6 REAL, M7 REAL
    )
""")
conn.commit()

# Función para extraer los momentos de Hu
def extraer_momentos_hu(img):
    # Convertir a escala de grises y aplicar un umbral
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gris, 128, 255, cv2.THRESH_BINARY)
    momentos = cv2.moments(thresh)
    momentos_hu = cv2.HuMoments(momentos).flatten()  # Array de 7 valores
    return momentos_hu

# Recorrer cada imagen en la carpeta de entrada
for idx, filename in enumerate(os.listdir(carpeta_entrada)):
    # Leer la imagen
    img_path = os.path.join(carpeta_entrada, filename)
    img = cv2.imread(img_path)

    if img is None:
        continue

    # Convertir a escala de grises y aplicar detección de bordes
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gris, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Crear una máscara vacía para dibujar el contorno de la hoja
    mask = np.zeros_like(gris)

    # Filtrar contornos por área mínima
    min_contour_area = 1000
    leaf_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    if leaf_contours:
        # Dibujar el contorno en la máscara
        cv2.drawContours(mask, leaf_contours, -1, 255, thickness=cv2.FILLED)

        # Crear una versión RGBA de la imagen original y aplicar la máscara
        isolated_leaf_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        isolated_leaf_rgba[:, :, 3] = mask

        # Guardar la imagen con fondo transparente
        nuevo_nombre = f"parra_{idx+1:02}.png"
        output_path = os.path.join(carpeta_salida, nuevo_nombre)
        cv2.imwrite(output_path, isolated_leaf_rgba)

        # Extraer momentos de Hu y almacenarlos en la base de datos
        momentos_hu = extraer_momentos_hu(img)
        print(f"Momentos de Hu para {filename}: {momentos_hu}")
        try:
            cursor.execute("""
            INSERT INTO parra (nombre_archivo, M1, M2, M3, M4, M5, M6, M7)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (nuevo_nombre, *momentos_hu))
            conn.commit()
            print(f"Datos de {filename} guardados en la base de datos.")
        except sqlite3.Error as e:
            print(f"Error al guardar en la base de datos para {filename}: {e}")

        conn.commit()

print("Proceso completado. Las imágenes y momentos de Hu se guardaron.")
conn.close()