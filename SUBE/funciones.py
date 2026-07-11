from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class AnalizadorSUBE:
    """
    Administra la lectura, limpieza, transformación y análisis
    de los datos de usos de tarjeta SUBE.
    """

    def __init__(self, carpeta_crudo: str, carpeta_graficos: str):
        """
        Inicializa el analizador.

        Parámetros:
        carpeta_crudo: ubicación de los archivos CSV originales.
        carpeta_graficos: ubicación donde se guardarán los gráficos.
        """
        self.carpeta_crudo = Path(carpeta_crudo)
        self.carpeta_graficos = Path(carpeta_graficos)
        self.df = pd.DataFrame()

        # Lista utilizada para indicar los archivos que serán procesados.
        self.archivos = [
            "dat-ab-usos-2024.csv",
            "dat-ab-usos-2025.csv"
        ]

        # Lista de columnas necesarias para el análisis.
        self.columnas_utilizadas = [
            "DIA_TRANSPORTE",
            "AMBA",
            "TIPO_TRANSPORTE",
            "PROVINCIA",
            "MUNICIPIO",
            "CANTIDAD",
            "DATO_PRELIMINAR"
        ]

        # Diccionario utilizado para renombrar columnas.
        self.renombres = {
            "DIA_TRANSPORTE": "fecha",
            "AMBA": "amba",
            "TIPO_TRANSPORTE": "tipo_transporte",
            "PROVINCIA": "provincia",
            "MUNICIPIO": "municipio",
            "CANTIDAD": "cantidad",
            "DATO_PRELIMINAR": "dato_preliminar"
        }

        # Tupla que establece el orden correcto de los días.
        self.orden_dias = (
            "Lunes",
            "Martes",
            "Miércoles",
            "Jueves",
            "Viernes",
            "Sábado",
            "Domingo"
        )

        self.carpeta_graficos.mkdir(parents=True, exist_ok=True)

    def leer_archivos(self) -> pd.DataFrame:
        """
        Lee los archivos CSV de 2024 y 2025 y los une
        en un único DataFrame.
        """
        dataframes = []

        for nombre_archivo in self.archivos:
            ruta_archivo = self.carpeta_crudo / nombre_archivo

            if not ruta_archivo.exists():
                raise FileNotFoundError(
                    f"No se encontró el archivo: {ruta_archivo}"
                )

            try:
                df_anual = pd.read_csv(
                    ruta_archivo,
                    usecols=self.columnas_utilizadas,
                    encoding="utf-8",
                    low_memory=False
                )
            except UnicodeDecodeError:
                # Alternativa para archivos con otra codificación.
                df_anual = pd.read_csv(
                    ruta_archivo,
                    usecols=self.columnas_utilizadas,
                    encoding="latin-1",
                    low_memory=False
                )

            dataframes.append(df_anual)

        self.df = pd.concat(dataframes, ignore_index=True)

        return self.df

    def mostrar_informacion_inicial(self) -> None:
        """
        Muestra información general antes de realizar la limpieza.
        """
        print("\n--- INFORMACIÓN INICIAL DEL DATASET ---")
        print(f"Cantidad de filas: {self.df.shape[0]}")
        print(f"Cantidad de columnas: {self.df.shape[1]}")

        print("\nTipos de datos:")
        print(self.df.dtypes)

        print("\nValores nulos:")
        print(self.df.isnull().sum())

        print("\nPrimeras filas:")
        print(self.df.head())

    def limpiar_y_transformar(self) -> pd.DataFrame:
        """
        Limpia y transforma los datos para el análisis estadístico.
        """
        if self.df.empty:
            raise ValueError(
                "Primero se deben leer los archivos con leer_archivos()."
            )

        # Transformación 1: renombrar las columnas.
        self.df.rename(columns=self.renombres, inplace=True)

        # Transformación 2: convertir la fecha a tipo datetime.
        self.df["fecha"] = pd.to_datetime(
            self.df["fecha"],
            errors="coerce"
        )

        # Transformación 3: convertir la cantidad a valor numérico.
        self.df["cantidad"] = pd.to_numeric(
            self.df["cantidad"],
            errors="coerce"
        )

        # Limpieza de espacios y unificación de mayúsculas.
        columnas_texto = [
            "amba",
            "tipo_transporte",
            "provincia",
            "municipio",
            "dato_preliminar"
        ]

        for columna in columnas_texto:
            self.df[columna] = (
                self.df[columna]
                .astype("string")
                .str.strip()
                .str.upper()
            )

        # Eliminación de registros sin fecha o sin cantidad.
        self.df.dropna(
            subset=["fecha", "cantidad"],
            inplace=True
        )

        # Eliminación de registros con cantidades negativas.
        self.df = self.df[self.df["cantidad"] >= 0].copy()

        # Filtrado de datos preliminares.
        self.df = self.df[
            self.df["dato_preliminar"] == "NO"
        ].copy()

        # Creación de nuevas columnas.
        self.df["anio"] = self.df["fecha"].dt.year
        self.df["mes_numero"] = self.df["fecha"].dt.month
        self.df["mes"] = self.df["fecha"].dt.month_name(locale="Spanish")
        self.df["dia_semana_numero"] = self.df["fecha"].dt.dayofweek

        nombres_dias = {
            0: "Lunes",
            1: "Martes",
            2: "Miércoles",
            3: "Jueves",
            4: "Viernes",
            5: "Sábado",
            6: "Domingo"
        }

        self.df["dia_semana"] = self.df[
            "dia_semana_numero"
        ].map(nombres_dias)

        # Nueva columna para diferenciar AMBA e interior.
        clasificacion_geografica = {
            "SI": "AMBA",
            "NO": "Interior"
        }

        self.df["zona"] = self.df["amba"].map(
            clasificacion_geografica
        )

        # Se eliminan filas que no pudieron clasificarse.
        self.df.dropna(subset=["zona"], inplace=True)

        return self.df

    def mostrar_informacion_final(self) -> None:
        """
        Muestra información general luego de la limpieza.
        """
        print("\n--- INFORMACIÓN DESPUÉS DE LA LIMPIEZA ---")
        print(f"Cantidad de filas: {self.df.shape[0]}")
        print(f"Cantidad de columnas: {self.df.shape[1]}")

        print("\nValores nulos:")
        print(self.df.isnull().sum())

        print("\nTipos de transporte:")
        print(self.df["tipo_transporte"].value_counts())

    def calcular_metricas(self) -> dict:
        """
        Calcula métricas estadísticas sobre los viajes diarios.

        Retorna:
        Un diccionario con los principales resultados.
        """
        viajes_diarios = (
            self.df
            .groupby("fecha", as_index=False)["cantidad"]
            .sum()
        )

        # Tupla con métricas principales.
        medidas_generales = (
            viajes_diarios["cantidad"].mean(),
            viajes_diarios["cantidad"].median(),
            viajes_diarios["cantidad"].max(),
            viajes_diarios["cantidad"].min()
        )

        fecha_maximo = viajes_diarios.loc[
            viajes_diarios["cantidad"].idxmax(),
            "fecha"
        ]

        fecha_minimo = viajes_diarios.loc[
            viajes_diarios["cantidad"].idxmin(),
            "fecha"
        ]

        metricas = {
            "promedio_diario": medidas_generales[0],
            "mediana_diaria": medidas_generales[1],
            "maximo_diario": medidas_generales[2],
            "minimo_diario": medidas_generales[3],
            "fecha_maximo": fecha_maximo,
            "fecha_minimo": fecha_minimo,
            "total_viajes": self.df["cantidad"].sum(),
            "cantidad_registros": len(self.df)
        }

        return metricas

    def mostrar_metricas(self, metricas: dict) -> None:
        """
        Muestra las métricas estadísticas en pantalla.
        """
        print("\n--- MÉTRICAS GENERALES ---")
        print(f"Registros analizados: {metricas['cantidad_registros']:,}")
        print(f"Total de usos: {metricas['total_viajes']:,.0f}")
        print(
            f"Promedio diario: "
            f"{metricas['promedio_diario']:,.2f}"
        )
        print(
            f"Mediana diaria: "
            f"{metricas['mediana_diaria']:,.2f}"
        )
        print(
            f"Máximo diario: "
            f"{metricas['maximo_diario']:,.0f}"
        )
        print(
            f"Fecha del máximo: "
            f"{metricas['fecha_maximo'].date()}"
        )
        print(
            f"Mínimo diario: "
            f"{metricas['minimo_diario']:,.0f}"
        )
        print(
            f"Fecha del mínimo: "
            f"{metricas['fecha_minimo'].date()}"
        )

    def calcular_metricas_por_transporte(self) -> pd.DataFrame:
        """
        Calcula el total, promedio y mediana por tipo de transporte.
        """
        datos_diarios = (
            self.df
            .groupby(
                ["fecha", "tipo_transporte"],
                as_index=False
            )["cantidad"]
            .sum()
        )

        metricas_transporte = (
            datos_diarios
            .groupby("tipo_transporte")["cantidad"]
            .agg(
                total="sum",
                promedio="mean",
                mediana="median",
                maximo="max",
                minimo="min"
            )
            .sort_values("total", ascending=False)
        )

        return metricas_transporte

    def calcular_variacion_anual(self) -> pd.DataFrame:
        """
        Calcula el total de usos por año y la variación porcentual.
        """
        totales_anuales = (
            self.df
            .groupby("anio", as_index=False)["cantidad"]
            .sum()
            .sort_values("anio")
        )

        totales_anuales["variacion_porcentual"] = (
            totales_anuales["cantidad"]
            .pct_change()
            .mul(100)
        )

        return totales_anuales

    def agregar_columna_normalizada(self) -> None:
        """
        Ejemplo opcional de uso de NumPy.

        Normaliza la columna cantidad utilizando normalización
        mínimo-máximo. Los valores quedan entre 0 y 1.
        """
        minimo = self.df["cantidad"].min()
        maximo = self.df["cantidad"].max()

        if maximo == minimo:
            self.df["cantidad_normalizada"] = 0.0
        else:
            self.df["cantidad_normalizada"] = np.where(
                maximo != minimo,
                (self.df["cantidad"] - minimo) / (maximo - minimo),
                0
            )

    def grafico_evolucion_mensual(self) -> None:
        """
        Genera un gráfico de líneas con la evolución mensual
        para 2024 y 2025.
        """
        datos = (
            self.df
            .groupby(
                ["anio", "mes_numero"],
                as_index=False
            )["cantidad"]
            .sum()
        )

        plt.figure(figsize=(11, 6))

        for anio in sorted(datos["anio"].unique()):
            datos_anio = datos[datos["anio"] == anio]

            plt.plot(
                datos_anio["mes_numero"],
                datos_anio["cantidad"],
                marker="o",
                label=str(anio)
            )

        plt.title("Evolución mensual de los usos de tarjeta SUBE")
        plt.xlabel("Mes")
        plt.ylabel("Cantidad de usos")
        plt.xticks(range(1, 13))
        plt.legend(title="Año")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        ruta = self.carpeta_graficos / "01_evolucion_mensual.png"
        plt.savefig(ruta, dpi=300)
        plt.close()

    def grafico_dias_semana(self) -> None:
        """
        Genera un gráfico de barras con el promedio de usos
        según el día de la semana.
        """
        viajes_por_fecha = (
            self.df
            .groupby(
                ["fecha", "dia_semana"],
                as_index=False
            )["cantidad"]
            .sum()
        )

        promedio_dias = (
            viajes_por_fecha
            .groupby("dia_semana")["cantidad"]
            .mean()
            .reindex(self.orden_dias)
        )

        plt.figure(figsize=(10, 6))
        promedio_dias.plot(kind="bar")

        plt.title("Promedio de usos según el día de la semana")
        plt.xlabel("Día de la semana")
        plt.ylabel("Promedio de usos diarios")
        plt.xticks(rotation=45)
        plt.tight_layout()

        ruta = self.carpeta_graficos / "02_dias_semana.png"
        plt.savefig(ruta, dpi=300)
        plt.close()

    def grafico_distribucion_geografica(self) -> None:
        """
        Genera un gráfico de barras comparando AMBA e interior.
        """
        datos_zona = (
            self.df
            .groupby("zona")["cantidad"]
            .sum()
            .sort_values(ascending=False)
        )

        plt.figure(figsize=(8, 6))
        datos_zona.plot(kind="bar")

        plt.title("Usos de tarjeta SUBE: AMBA e interior")
        plt.xlabel("Zona geográfica")
        plt.ylabel("Cantidad de usos")
        plt.xticks(rotation=0)
        plt.tight_layout()

        ruta = (
            self.carpeta_graficos /
            "03_distribucion_geografica.png"
        )
        plt.savefig(ruta, dpi=300)
        plt.close()

    def grafico_ranking_provincias(self, cantidad: int = 10) -> None:
        """
        Genera un ranking de las provincias con mayor cantidad
        de usos registrados.
        """
        ranking = (
            self.df
            .groupby("provincia")["cantidad"]
            .sum()
            .sort_values(ascending=False)
            .head(cantidad)
            .sort_values()
        )

        plt.figure(figsize=(11, 7))
        ranking.plot(kind="barh")

        plt.title(
            f"{cantidad} provincias con mayor cantidad de usos SUBE"
        )
        plt.xlabel("Cantidad de usos")
        plt.ylabel("Provincia")
        plt.tight_layout()

        ruta = self.carpeta_graficos / "04_ranking_provincias.png"
        plt.savefig(ruta, dpi=300)
        plt.close()

    def grafico_tipo_transporte(self) -> None:
        """
        Genera un gráfico de barras con la participación
        de cada tipo de transporte.
        """
        datos_transporte = (
            self.df
            .groupby("tipo_transporte")["cantidad"]
            .sum()
            .sort_values(ascending=False)
        )

        plt.figure(figsize=(10, 6))
        datos_transporte.plot(kind="bar")

        plt.title("Usos registrados por tipo de transporte")
        plt.xlabel("Tipo de transporte")
        plt.ylabel("Cantidad de usos")
        plt.xticks(rotation=45)
        plt.tight_layout()

        ruta = self.carpeta_graficos / "05_tipo_transporte.png"
        plt.savefig(ruta, dpi=300)
        plt.close()

    def grafico_histograma_viajes_diarios(self) -> None:
        """
        Genera un histograma para mostrar la distribución
        de los usos diarios.
        """
        viajes_diarios = (
            self.df
            .groupby("fecha")["cantidad"]
            .sum()
        )

        plt.figure(figsize=(10, 6))
        plt.hist(
            viajes_diarios,
            bins=30,
            edgecolor="black"
        )

        plt.title("Distribución de los usos diarios de tarjeta SUBE")
        plt.xlabel("Cantidad de usos por día")
        plt.ylabel("Frecuencia")
        plt.tight_layout()

        ruta = self.carpeta_graficos / "06_histograma_diario.png"
        plt.savefig(ruta, dpi=300)
        plt.close()

    def generar_graficos(self) -> None:
        """
        Ejecuta todos los métodos de generación de gráficos.
        """
        self.grafico_evolucion_mensual()
        self.grafico_dias_semana()
        self.grafico_distribucion_geografica()
        self.grafico_ranking_provincias()
        self.grafico_tipo_transporte()
        self.grafico_histograma_viajes_diarios()

        print(
            f"\nLos gráficos fueron guardados en: "
            f"{self.carpeta_graficos.resolve()}"
        )