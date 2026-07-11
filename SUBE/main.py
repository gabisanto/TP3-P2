from pathlib import Path

import pandas as pd

from funciones import AnalizadorSUBE


def main() -> None:
    """
    Función principal del programa.
    Coordina la ejecución completa del análisis.
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

        analizador.leer_archivos()
        analizador.mostrar_informacion_inicial()

        analizador.limpiar_y_transformar()
        analizador.agregar_columna_normalizada()
        analizador.mostrar_informacion_final()

        metricas = analizador.calcular_metricas()
        analizador.mostrar_metricas(metricas)

        metricas_transporte = (
            analizador.calcular_metricas_por_transporte()
        )

        print("\n--- MÉTRICAS POR TIPO DE TRANSPORTE ---")
        print(metricas_transporte)

        variacion_anual = analizador.calcular_variacion_anual()

        print("\n--- COMPARACIÓN ANUAL ---")
        print(variacion_anual)

        analizador.generar_graficos()

        print("\nAnálisis finalizado correctamente.")

    except FileNotFoundError as error:
        print(f"\nError de archivo: {error}")

    except pd.errors.ParserError as error:
        print(f"\nError al interpretar el CSV: {error}")

    except Exception as error:
        print(f"\nOcurrió un error inesperado: {error}")


if __name__ == "__main__":
    main()