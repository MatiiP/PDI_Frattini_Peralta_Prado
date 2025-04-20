import cv2
import numpy as np

def local_histogram_equalization(image, window_size):
    """
    Realiza la ecualización local del histograma en una imagen.

    Args:
        image (np.ndarray): Imagen de entrada (se convertirá a escala de grises).
        window_size (tuple): Tupla (M, N) que define el tamaño de la ventana.

    Returns:
        np.ndarray: Imagen procesada con ecualización local del histograma.
    """
    if len(image.shape) > 2 and image.shape[2] > 1:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy() # Trabajar con una copia

    M, N = window_size
    if M % 2 == 0 or N % 2 == 0:
        raise ValueError("El tamaño de la ventana (M, N) debe tener dimensiones impares.")

    # Calcular el padding necesario
    pad_M = M // 2
    pad_N = N // 2

    # Añadir padding a la imagen
    padded_image = cv2.copyMakeBorder(gray_image, pad_M, pad_M, pad_N, pad_N, cv2.BORDER_REPLICATE)

    # Crear imagen de salida
    output_image = np.zeros_like(gray_image)

    # Iterar sobre cada píxel de la imagen original
    rows, cols = gray_image.shape
    for i in range(rows):
        for j in range(cols):
            # Extraer la ventana local (vecindario) de la imagen con padding
            # La esquina superior izquierda de la ventana en la imagen con padding es (i, j)
            window = padded_image[i : i + M, j : j + N]

            # Calcular el histograma de la ventana
            hist = cv2.calcHist([window], [0], None, [256], [0, 256])

            # Calcular la función de distribución acumulativa (CDF)
            cdf = hist.cumsum()

            # --- Inicio de la Edición: Normalización de la CDF ---
            # Eliminar la línea confusa:
            # cdf_normalized = cdf * hist.max() / cdf.max() # Esta línea era confusa/incorrecta

            # Normalizar la CDF al rango [0, 255] para usarla como tabla de mapeo
            # Mascarar ceros para evitar división por cero si todos los píxeles son iguales
            cdf_m = np.ma.masked_equal(cdf, 0)
            # Aplicar la fórmula de normalización: (cdf(v) - cdf_min) * 255 / (total_pixels - cdf_min)
            # Aquí M*N es el número total de píxeles en la ventana
            total_pixels_in_window = M * N
            cdf_m = (cdf_m - cdf_m.min()) * 255 / (total_pixels_in_window - cdf_m.min())
            # Rellenar los valores enmascarados (correspondientes a intensidades no presentes) con 0
            cdf_final = np.ma.filled(cdf_m, 0).astype('uint8')
            # --- Fin de la Edición ---

            # Obtener el valor del píxel central en la ventana original (imagen con padding)
            center_pixel_intensity = padded_image[i + pad_M, j + pad_N]

            # Mapear la intensidad del píxel central usando la CDF normalizada (cdf_final)
            output_image[i, j] = cdf_final[center_pixel_intensity]

    return output_image
