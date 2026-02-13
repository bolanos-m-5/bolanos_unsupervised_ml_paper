import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class AnalisisExploratorio:
    def __init__(self, df):
        """
        Inicializa con un DataFrame que contenga al menos:
        - Customer
        - Channel
        - Team
        - nos (para cálculos porcentuales)
        """
        self.df = df

    def _remove_outliers(self, data, lower_percentile=0.05, upper_percentile=0.95):
        lower_bound = data.quantile(lower_percentile)
        upper_bound = data.quantile(upper_percentile)
        return data[(data >= lower_bound) & (data <= upper_bound)]
    

    def resumen_unicos(self, columnas=None):
        """
        Muestra cantidad de valores únicos para las columnas especificadas.
        
        Args:
            columnas: Lista de columnas para analizar. Si None, usa las columnas por defecto.
                     Ejemplo: ['customer', 'channel', 'team', 'category_smo']
        
        Returns:
            dict: Diccionario con el conteo de valores únicos por columna
        """
        # Columnas por defecto si no se especifica ninguna
        if columnas is None:
            columnas = ['customer', 'channel', 'team', 'category_smo']
        
        resultado = {}
        
        for columna in columnas:
            if columna in self.df.columns:
                # Crear clave descriptiva
                clave = f"{columna}_unicos"
                resultado[clave] = self.df[columna].nunique()
            else:
                print(f"Advertencia: La columna '{columna}' no existe en el dataset")
                
        return resultado

    def plot_distribucion_por_columna(self, columna, titulo):
        """
        Crea un gráfico de barras con la distribución de valores de una columna.
        """
        distribucion = self.df[columna].value_counts()

        plt.figure(figsize=(10, 6))
        ax = sns.countplot(data=self.df, x=columna, order=distribucion.index)
        plt.title(titulo)
        plt.xticks(rotation=45)

        # Etiquetas
        for p in ax.patches:
            ax.annotate(f'{p.get_height()}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=10, color='black')

        plt.show()
    
    def plot_distribuciones_multiples(self, columnas, ncols=2, figsize=(16, 6)):
        """
        Crea múltiples gráficos de distribución lado a lado.
        
        Args:
            columnas: Lista de columnas para graficar
                     Ejemplo: ['channel', 'team', 'category_smo']
            ncols: Número de columnas en la grilla de subplots (default: 2)
            figsize: Tamaño de la figura completa (default: (16, 6))
        """
        n_plots = len(columnas)
        nrows = (n_plots + ncols - 1) // ncols  # Calcular filas necesarias
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        
        # Asegurar que axes sea un array 2D
        if nrows == 1 and ncols == 1:
            axes = [[axes]]
        elif nrows == 1:
            axes = [axes]
        elif ncols == 1:
            axes = [[ax] for ax in axes]
        
        # Aplanar para iterar fácilmente
        axes_flat = [ax for row in axes for ax in row]
        
        for idx, columna in enumerate(columnas):
            ax = axes_flat[idx]
            
            # Contar clientes únicos por cada valor de la columna
            distribucion = self.df.groupby(columna)['customer'].nunique().sort_values(ascending=False)
            
            sns.barplot(x=distribucion.index, y=distribucion.values, ax=ax)
            ax.set_title(f'Distribución de clientes únicos por {columna}')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            ax.set_ylabel('Clientes únicos')
            
            # Etiquetas
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.0f}',
                           (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='bottom', fontsize=10, color='black')
        
        # Ocultar subplots vacíos
        for idx in range(n_plots, len(axes_flat)):
            axes_flat[idx].axis('off')
        
        plt.tight_layout()
        plt.show()

    def plot_nos_por_columna(self, columna):
        """
        Calcula y grafica el porcentaje de NOS por columna.
        """
        column_nos_distribution = self.df.groupby(columna)['nos_excl_NIT'].sum()
        total_nos = column_nos_distribution.sum()
        column_nos_percentage = (column_nos_distribution / total_nos) * 100
        column_nos_percentage = column_nos_percentage.sort_values(ascending=False)

        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=column_nos_percentage.index, y=column_nos_percentage.values)
        plt.title(f'Porcentaje de NOS_excl_NIT por {columna}')
        plt.xticks(rotation=45)

        for p in ax.patches:
            ax.annotate(f'{p.get_height():.1f}%',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=10, color='black')

        plt.ylabel('Porcentaje de NOS_excl_NIT (%)')
        plt.show()
    
    def plot_nos_multiples(self, columnas, ncols=2, figsize=(16, 6)):
        """
        Crea múltiples gráficos de porcentaje de NOS lado a lado.
        
        Args:
            columnas: Lista de columnas para graficar
                     Ejemplo: ['channel', 'team', 'category_smo']
            ncols: Número de columnas en la grilla de subplots (default: 2)
            figsize: Tamaño de la figura completa (default: (16, 6))
        """
        n_plots = len(columnas)
        nrows = (n_plots + ncols - 1) // ncols  # Calcular filas necesarias
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        
        # Asegurar que axes sea un array 2D
        if nrows == 1 and ncols == 1:
            axes = [[axes]]
        elif nrows == 1:
            axes = [axes]
        elif ncols == 1:
            axes = [[ax] for ax in axes]
        
        # Aplanar para iterar fácilmente
        axes_flat = [ax for row in axes for ax in row]
        
        for idx, columna in enumerate(columnas):
            ax = axes_flat[idx]
            
            # Calcular porcentajes
            column_nos_distribution = self.df.groupby(columna)['nos_excl_NIT'].sum()
            total_nos = column_nos_distribution.sum()
            column_nos_percentage = (column_nos_distribution / total_nos) * 100
            column_nos_percentage = column_nos_percentage.sort_values(ascending=False)
            
            sns.barplot(x=column_nos_percentage.index, y=column_nos_percentage.values, ax=ax)
            ax.set_title(f'Porcentaje de NOS_excl_NIT por {columna}')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            ax.set_ylabel('Porcentaje de NOS_excl_NIT (%)')
            
            # Etiquetas
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.1f}%',
                           (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='bottom', fontsize=10, color='black')
        
        # Ocultar subplots vacíos
        for idx in range(n_plots, len(axes_flat)):
            axes_flat[idx].axis('off')
        
        plt.tight_layout()
        plt.show()

    def tablas_nos_multiples(self, columnas):
        """
        Genera tablas con el porcentaje de NOS_excl_NIT y distribución de clientes para múltiples columnas.
        
        Args:
            columnas: Lista de columnas para analizar
                     Ejemplo: ['channel', 'team', 'category_smo']
        
        Returns:
            dict: Diccionario con las tablas por cada columna
        """
        resultados = {}
        
        for columna in columnas:
            # Calcular distribución de NOS por la columna
            column_nos_distribution = self.df.groupby(columna)['nos_excl_NIT'].sum()
            total_nos = column_nos_distribution.sum()
            column_nos_percentage = (column_nos_distribution / total_nos) * 100
            
            # Calcular distribución de clientes por la columna
            column_customers_distribution = self.df.groupby(columna)['customer'].nunique()
            total_customers = column_customers_distribution.sum()
            column_customers_percentage = (column_customers_distribution / total_customers) * 100
            
            # Crear DataFrame con los resultados completos
            tabla_resultado = pd.DataFrame({
                f'{columna}': column_nos_percentage.index,
                'Num_Clientes': column_customers_distribution.values,
                'Clientes (%)': column_customers_percentage.values.round(1),
                'NOS (%)': column_nos_percentage.values.round(1),
            }).sort_values('NOS (%)', ascending=False).reset_index(drop=True)
            
            resultados[columna] = tabla_resultado
        
        return resultados
    
    def descriptive_analytics(self, columnas, remove_outliers=False):
        """
        Genera estadísticas descriptivas (promedio, desviación estándar, mediana) 
        para las columnas especificadas.
        
        Args:
            columnas: Lista de nombres de columnas para analizar
                     Ejemplo: ['nos_excl_NIT', 'rate_nos', 'volume']
            remove_outliers: Si True, aplica filtro de outliers (default: False)
        
        Returns:
            pd.DataFrame: DataFrame con estadísticas descriptivas
        """
        estadisticas = []
        
        for columna in columnas:
            if columna in self.df.columns:
                # Filtrar valores faltantes e infinitos
                data = self.df[columna].replace([float('inf'), float('-inf')], pd.NA).dropna()
                
                # Convertir a numérico y eliminar valores no convertibles
                data = pd.to_numeric(data, errors='coerce').dropna()
                
                # Aplicar filtro de outliers solo si se solicita
                if remove_outliers and len(data) > 0:
                    try:
                        data = self._remove_outliers(data)
                    except:
                        print(f"No se pudo aplicar filtro de outliers a {columna}")
                
                if len(data) > 0:
                    try:
                        estadisticas.append({
                            'Variable': columna,
                            'Media': round(data.mean(), 2),
                            'Mediana': round(data.median(), 2),
                            'Desv_Std': round(data.std(), 2),
                            'Min': round(data.min(), 2),
                            'Max': round(data.max(), 2),
                            'Count': len(data),
                            'Count_Original': len(self.df[columna].dropna())
                        })
                    except Exception as e:
                        print(f"Error calculando estadísticas para {columna}: {e}")
                        estadisticas.append({
                            'Variable': columna,
                            'Media': 'Error en cálculo',
                            'Mediana': 'Error en cálculo',
                            'Desv_Std': 'Error en cálculo',
                            'Min': 'Error en cálculo',
                            'Max': 'Error en cálculo',
                            'Count': len(data),
                            'Count_Original': len(self.df[columna].dropna())
                        })
                else:
                    estadisticas.append({
                        'Variable': columna,
                        'Media': 'Sin datos válidos',
                        'Mediana': 'Sin datos válidos',
                        'Desv_Std': 'Sin datos válidos',
                        'Min': 'Sin datos válidos',
                        'Max': 'Sin datos válidos',
                        'Count': 0,
                        'Count_Original': len(self.df[columna].dropna())
                    })
            else:
                print(f"Advertencia: La columna '{columna}' no existe en el dataset")
        
        tabla_descriptiva = pd.DataFrame(estadisticas)
        return tabla_descriptiva

    def tabla_correlaciones(self, columnas, metodo='pearson', diagnostico=True):
        """
        Genera una tabla de correlaciones entre las columnas especificadas.
        
        Args:
            columnas: Lista de nombres de columnas para analizar
                     Ejemplo: ['nos_excl_NIT', 'rate_nos', 'volume']
            metodo: Método de correlación ('pearson', 'spearman', 'kendall')
            diagnostico: Si True, muestra información de diagnóstico
        
        Returns:
            pd.DataFrame: DataFrame con matriz de correlaciones
        """
        # Filtrar solo las columnas que existen
        columnas_validas = [col for col in columnas if col in self.df.columns]
        columnas_faltantes = [col for col in columnas if col not in self.df.columns]
        
        if columnas_faltantes:
            print(f"Advertencia: Las siguientes columnas no existen: {columnas_faltantes}")
        
        if len(columnas_validas) < 2:
            print("Error: Se necesitan al menos 2 columnas válidas para calcular correlaciones")
            return pd.DataFrame()
        
        # Filtrar datos válidos
        data_filtrada = self.df[columnas_validas].copy()
        
        # Reemplazar infinitos con NaN
        for col in columnas_validas:
            data_filtrada[col] = data_filtrada[col].replace([float('inf'), float('-inf')], pd.NA)
            data_filtrada[col] = pd.to_numeric(data_filtrada[col], errors='coerce')
        
        # Eliminar filas con cualquier NaN
        data_filtrada = data_filtrada.dropna()
        
        if diagnostico:
            print(f"Datos válidos después de limpieza: {len(data_filtrada)} filas")
            print("\nEstadísticas por variable:")
            for col in columnas_validas:
                if col in data_filtrada.columns and len(data_filtrada[col]) > 0:
                    print(f"{col}: Min={data_filtrada[col].min():.2f}, Max={data_filtrada[col].max():.2f}, Std={data_filtrada[col].std():.2f}")
        
        if len(data_filtrada) < 2:
            print("Error: Muy pocos datos válidos para calcular correlaciones")
            return pd.DataFrame()
        
        # Verificar varianza cero (datos constantes)
        if diagnostico:
            print("\nVerificación de varianza:")
            for col in columnas_validas:
                if col in data_filtrada.columns:
                    var = data_filtrada[col].var()
                    if var == 0 or pd.isna(var):
                        print(f"ADVERTENCIA: {col} tiene varianza cero (datos constantes)")
        
        # Calcular correlaciones
        correlaciones = data_filtrada.corr(method=metodo)
        
        # Redondear a 3 decimales para mejor legibilidad
        correlaciones = correlaciones.round(3)
        
        return correlaciones

    def graficar_rates_sin_outliers(self):
        plt.figure(figsize=(12, 10))

        if 'rate_nos' in self.df:
            filtered_nos = self._remove_outliers(self.df['rate_nos'].dropna())
            plt.subplot(4, 1, 1)
            sns.histplot(filtered_nos, bins=30, kde=True)
            plt.title('Distribución de Rate NOS - Sin Outliers')

        if 'rate_nsrd' in self.df:
            filtered_nsrd = self._remove_outliers(self.df['rate_nsrd'].dropna())
            plt.subplot(4, 1, 2)
            sns.histplot(filtered_nsrd, bins=30, kde=True)
            plt.title('Distribución de Rate NSRD - Sin Outliers')

        if 'rate_sd' in self.df:
            filtered_sd = self._remove_outliers(self.df['rate_sd'].dropna())
            plt.subplot(4, 1, 3)
            sns.histplot(filtered_sd, bins=30, kde=True)
            plt.title('Distribución de Rate SD - Sin Outliers')

        if 'nos_su' in self.df:
            filtered_nos_su = self._remove_outliers(self.df['nos_su'].dropna())
            plt.subplot(4, 1, 4)
            sns.histplot(filtered_nos_su, bins=30, kde=True)
            plt.title('Distribución de NOS SU - Sin Outliers')

        plt.tight_layout()
        plt.show()

    def graficar_growth_rates_sin_outliers(self):
        plt.figure(figsize=(12, 6))

        if 'variation_rate_nos' in self.df:
            filtered_nos = self._remove_outliers(self.df['variation_rate_nos'].dropna())
            plt.subplot(2, 1, 1)
            sns.histplot(filtered_nos, bins=30, kde=True)
            plt.title('Distribución de Variation Rate NOS - Sin Outliers')

        if 'variation_rate_volume' in self.df:
            filtered_volume = self._remove_outliers(self.df['variation_rate_volume'].dropna())
            plt.subplot(2, 1, 2)
            sns.histplot(filtered_volume, bins=30, kde=True)
            plt.title('Distribución de Variation Rate Volume - Sin Outliers')

        plt.tight_layout()
        plt.show()
    
    def calcular_distribucion_por_rango(self, bins=None, labels=None):
        # Sumar NOS por cliente
        client_nos_distribution = self.df.groupby('customer')['nos_excl_NIT'].sum()

        # Total NOS
        total_nos = client_nos_distribution.sum()

        # Porcentaje por cliente
        client_percentage = (client_nos_distribution / total_nos) * 100

        # Definir bins y labels si no se pasan
        if bins is None:
            bins = [0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25, 30, 100]
        if labels is None:
            labels = ['0-2.5%', '2.5-5%', '5-7.5%', '7.5-10%', '10-12.5%',
                      '12.5-15%', '15-17.5%', '17.5-20%', '20-22.5%',
                      '22.5-25%', '25-30%', '30%+']

        # Clasificación en bins
        client_bins = pd.cut(client_percentage, bins=bins, labels=labels, right=False)

        # Conteo por rango
        distribution_counts = client_bins.value_counts().sort_index()

        # Crear DataFrame de resultados
        distribution_table = pd.DataFrame({
            'Rango de Porcentaje de NOS_excl_NIT': distribution_counts.index,
            'Cantidad de Clientes': distribution_counts.values
        })

        self.distribution_table = distribution_table  # guardarlo para uso externo si se desea

        print(distribution_table)


    def graficar_top_customers(self, top_n=10):
        # Sumar NOS por cliente
        client_nos_distribution = self.df.groupby('customer')['nos_excl_NIT'].sum()
        total_nos = client_nos_distribution.sum()
        client_percentage = (client_nos_distribution / total_nos) * 100

        client_percentage_df = client_percentage.reset_index()
        client_percentage_df.columns = ['customer', 'Percentage of NOS_excl_NIT']

        # Ordenar y tomar top n
        top_customers = client_percentage_df.sort_values(by='Percentage of NOS_excl_NIT', ascending=False).head(top_n)

        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x='Percentage of NOS_excl_NIT', y='customer', data=top_customers)
        plt.title(f'Top {top_n} Clientes por Porcentaje de NOS_excl_NIT')
        plt.xlabel('Porcentaje de NOS_excl_NIT (%)')
        plt.ylabel('Cliente')

        for p in ax.patches:
            ax.annotate(f'{p.get_width():.1f}%', 
                        (p.get_width(), p.get_y() + p.get_height() / 2),
                        ha='left', va='center', fontsize=10, color='black')

        plt.show()

