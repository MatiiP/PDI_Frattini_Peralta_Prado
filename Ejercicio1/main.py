import cv2
import matplotlib.pyplot as plt
from ejercicio1 import local_histogram_equalization 

# Cargar la Imagen
image_path = 'Imagen_con_detalles_escondidos.tif'
try:
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original_image is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen en: {image_path}")
except FileNotFoundError as e:
    print(e)
    exit()
except Exception as e:
    print(f"Ocurrió un error al cargar la imagen: {e}")
    exit()
    
# Tamaños de Ventana a Probar
window_sizes = [(3, 3), (15, 15), (51, 51)] 
processed_images = {}

# Ecualización Local
for size in window_sizes:
    print(f"Procesando con ventana {size}...")
    try:
        processed_images[size] = local_histogram_equalization(original_image, size)
        print(f"Procesamiento con ventana {size} completado.")
    except ValueError as e:
        print(f"Error con tamaño de ventana {size}: {e}")
    except Exception as e:
        print(f"Error inesperado procesando con ventana {size}: {e}")

# Resultados 
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Imagen Original')
plt.axis('off')

plot_index = 2
for size, img in processed_images.items():
    if plot_index <= 4: # No nos excedemos de los subplots disponibles
        plt.subplot(2, 2, plot_index)
        plt.imshow(img, cmap='gray')
        plt.title(f'Ecualización Local - Ventana {size}')
        plt.axis('off')
        plot_index += 1

plt.tight_layout()
plt.show()

# Análisis
print("\n--- Análisis de Resultados ---")
print("Observando las imágenes procesadas:")
print("- Con ventanas pequeñas (ej. 3x3): Se realza mucho el ruido y las texturas muy finas. El contraste local aumenta significativamente, pero puede generar un aspecto 'granulado'.")
print("- Con ventanas medianas (ej. 15x15): Se logra un buen balance. Se realzan detalles en diferentes zonas que antes eran poco visibles (como texturas en áreas oscuras o claras) sin introducir excesivo ruido. Los detalles ocultos en la Figura 1 (dependiendo de la imagen específica) deberían volverse más aparentes.")
print("- Con ventanas grandes (ej. 51x51 o más): El resultado tiende a parecerse más a una ecualización global del histograma. Se pierde la capacidad de realzar detalles muy locales, ya que el histograma se calcula sobre una región más amplia, promediando más las características.")
print("\nInfluencia del tamaño de la ventana:")
print("El tamaño de la ventana es crucial. Determina la escala de los detalles que se realzarán.")
print("  - Ventanas más pequeñas -> Mayor realce de detalles finos y ruido, análisis muy local.")
print("  - Ventanas más grandes -> Menor realce de detalles finos, resultado más suave, más cercano a la ecualización global.")
print("La elección óptima del tamaño de la ventana depende de la escala de los detalles de interés en la imagen y del nivel de ruido aceptable.")
print("\nDetalles ocultos específicos (dependerá de la imagen 'Figura 1'):")
print("Busca en las imágenes procesadas (especialmente con ventana mediana) si aparecen texturas, objetos o patrones en áreas que parecían uniformes en la imagen original (zonas muy oscuras o muy brillantes).")
