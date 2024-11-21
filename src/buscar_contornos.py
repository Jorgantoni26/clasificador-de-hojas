import cv2
import numpy as np
import os

# Carpetas de entrada y salida
input_folder = "ruta_carpeta_origen"
output_folder = "ruta_carpeta_salida"



#recorrer cada imagen en la carpeta de entrada
for idx, filename in enumerate(os.listdir(input_folder)):
    # Leer la imagen
    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path)

    if img is None:
        continue

    # Convertir a escala de grises y aplicar detección de bordes
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Crear una máscara vacía para dibujar el contorno de la hoja
    mask = np.zeros_like(gray)

    # Filtrar contornos por área 
    min_contour_area = 1000  # Área mínima para considerar que es una hoja
    leaf_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    # Si hay contornos correctos, dibujar el contorno más grande en la máscara
    if leaf_contours:
        # Dibujar el contorno en la máscara
        cv2.drawContours(mask, leaf_contours, -1, 255, thickness=cv2.FILLED)

        # Crear una versión RGBA de la imagen original
        isolated_leaf_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

        # Aplicar la máscara para hacer transparente el fondo
        isolated_leaf_rgba[:, :, 3] = mask

        # Guardar la imagen con fondo transparente
        output_path = os.path.join(output_folder, f"hoja_aislada_{idx+1}.png")
        cv2.imwrite(output_path, isolated_leaf_rgba)

print("Proceso completado. Las imágenes se guardaron en la carpeta de salida.")
