import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tkinter as tk
from tkinter import filedialog
import os

def seleccionar_y_mostrar_png():
    # 1. Configurar la ventana oculta de Tkinter
    root = tk.Tk()
    root.withdraw() # Oculta la ventana principal de Tkinter
    root.attributes('-topmost', True) # Trae el explorador al frente

    # 2. Abrir el explorador de archivos de Windows
    ruta_archivo = filedialog.askopenfilename(
        title="Selecciona un archivo PNG",
        filetypes=[("Archivos PNG", "*.png"), ("Todos los archivos", "*.*")]
    )

    # 3. Verificar si el usuario seleccionó un archivo o canceló
    if not ruta_archivo:
        print("No se seleccionó ningún archivo.")
        return

    try:
        # 4. Leer y mostrar la imagen con Matplotlib
        img = mpimg.imread(ruta_archivo)
        
        # Crear la figura
        plt.figure(num=f"Visor PNG - {os.path.basename(ruta_archivo)}")
        plt.imshow(img)
        plt.axis('off') # Opcional: oculta los ejes (coordenadas)
        
        print(f"Mostrando: {ruta_archivo}")
        plt.show()
        
    except Exception as e:
        print(f"Error al abrir la imagen: {e}")

