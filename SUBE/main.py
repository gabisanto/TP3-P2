from pathlib import Path

import pandas as pd

from funciones import AnalizadorSUBE


def mostrar_menu() -> None:
    """
    Muestra las opciones de análisis disponibles.
    """
    print("\n--- MENÚ PRINCIPAL ---")
    print("1. Mostrar métricas generales")
    print("2. Mostrar métricas por tipo de transporte")
    print("3. Mostrar comparación anual")
    print("4. Generar gráficos")
    print("0. Salir")


def main() -> None:
    """
    Función principal del programa.
    Prepara los datos y permite consultar los resultados.
    """
    carpeta_proyecto = Path(__file__).resolve().parent
    carpeta_crudo = carpeta_proyecto / "Crudo"
    carpeta_graficos = carpeta_proyecto / "Graficos"

    analizador = AnalizadorSUBE(
        carpeta_crudo=str(carpeta_crudo),
        carpeta_graficos=str(carpeta_graficos)
    )

    try:
        print("Iniciando análisis de datos SUBE...")

        # Carga de los archivos originales.
        analizador.leer_archivos()

        # Información antes de realizar la limpieza.
        analizador.mostrar_informacion_inicial()

        # Limpieza y transformación automática.
        analizador.limpiar_y_transformar()
        analizador.agregar_columna_normalizada()

        # Información posterior al procesamiento.
        analizador.mostrar_informacion_final()

        print("\nLos datos fueron preparados correctamente.")

        opcion = ""

        while opcion != "0":
            mostrar_menu()
            opcion = input("Seleccione una opción: ").strip()

            if opcion == "1":
                metricas = analizador.calcular_metricas()
                analizador.mostrar_metricas(metricas)

            elif opcion == "2":
                metricas_transporte = (
                    analizador.calcular_metricas_por_transporte()
                )

                print("\n--- MÉTRICAS POR TIPO DE TRANSPORTE ---")
                print(metricas_transporte)

            elif opcion == "3":
                analizador.mostrar_variacion_anual()

            elif opcion == "4":
                analizador.generar_graficos()

            elif opcion == "0":
                print("\nPrograma finalizado.")

            else:
                print(
                    "\nOpción inválida. "
                    "Ingrese un número entre 0 y 4."
                )

    except FileNotFoundError as error:
        print(f"\nError de archivo: {error}")

    except pd.errors.ParserError as error:
        print(f"\nError al interpretar el CSV: {error}")

    except Exception as error:
        print(f"\nOcurrió un error inesperado: {error}")


if __name__ == "__main__":
    main()