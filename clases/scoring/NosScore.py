import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class ScoreDynamic:
    """
    Clase DINÁMICA para calcular el SCORE de desempeño basado en métricas NOS.
    Permite calcular scores por CUALQUIER dimensión: cliente, canal, equipo, sub_sector, producto, etc.
    
    DESCRIPCIÓN GENERAL:
    ===================
    Esta clase implementa un algoritmo de scoring compuesto que evalúa el desempeño
    comparando métricas individuales contra benchmarks del año fiscal.
    A diferencia de la clase Score original, esta versión es COMPLETAMENTE PARAMETRIZABLE.
    
    MÉTRICAS ANALIZADAS:
    ===================
    1. Rate NOS: Eficiencia en la conversión de ventas a NOS
    2. NOS SU: Eficiencia de NOS por unidad de sales unit
    3. Variation Rate Volume: Crecimiento en volumen de ventas
    4. Variation Rate NOS: Crecimiento en la métrica NOS
    
    METODOLOGÍA:
    ===========
    - Normalización relativa: Cada métrica se compara contra el promedio del año
    - Ponderación: Diferentes pesos según importancia estratégica
    - Penalizaciones: Ajustes negativos por métricas de crecimiento adversas
    - Escalamiento: Score final en escala 0-10 para interpretación intuitiva
    
    CASOS DE USO:
    ============
    - Scoring por cliente, canal, equipo, sub_sector, producto o cualquier combinación
    - Ranking por desempeño integral en cualquier dimensión
    - Identificación de entidades con alto potencial
    - Detección de entidades en riesgo (scores bajos)
    - Análisis comparativo cross-temporal y cross-dimensional
    
    DIFERENCIAS CON Score ORIGINAL:
    ==============================
    - Parametrizable: Se especifica la(s) dimensión(es) de análisis
    - Flexible: Soporta múltiples columnas de agrupación
    - Retrocompatible: Funciona igual que Score si se usa 'customer' como dimensión
    """
    
    def __init__(self, year_data: pd.DataFrame, df: pd.DataFrame, dimension_cols: list = None):
        """
        Inicializa la clase ScoreDynamic con datos de benchmark y datos detallados.
        
        Parameters:
        -----------
        year_data : pd.DataFrame
            DataFrame con datos agregados por año fiscal que sirven como benchmark.
            Debe contener columnas: ['fiscal_year', 'rate_nos', 'nos_su', 
            'variation_rate_volume', 'variation_rate_nos']
            
        df : pd.DataFrame  
            DataFrame detallado con datos por dimensión(es) específica(s).
            Debe contener las mismas métricas que year_data más las columnas
            de dimensión especificadas en dimension_cols
            
        dimension_cols : list, optional
            Lista de columnas que definen la dimensión de análisis.
            Por defecto: ['customer', 'channel', 'team']
            Ejemplos:
                - ['customer'] : Score por cliente
                - ['channel'] : Score por canal
                - ['sub_sector'] : Score por subsector
                - ['team', 'channel'] : Score por combinación equipo-canal
                - ['customer', 'channel', 'team'] : Score por cliente-canal-equipo
            
        Notes:
        ------
        - Se trabaja con una copia del DataFrame original para preservar datos
        - Los datos year_data deben estar previamente calculados y agregados
        - El DataFrame df debe estar filtrado al período de análisis deseado
        - Las columnas en dimension_cols deben existir en df
        """
        self.year_data = year_data
        self.df = df.copy()  # trabajar con copia para no modificar externo
        
        # Configuración de dimensiones (por defecto: customer, channel, team)
        if dimension_cols is None:
            self.dimension_cols = ['customer', 'channel', 'team']
        else:
            self.dimension_cols = dimension_cols if isinstance(dimension_cols, list) else [dimension_cols]
        
        # Validar que las columnas de dimensión existan en el DataFrame
        missing_cols = [col for col in self.dimension_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Las siguientes columnas de dimensión no existen en el DataFrame: {missing_cols}")

    def calcular_score(self, fiscal_year=2025):
        """
        Calcula el SCORE para las dimensiones especificadas.
        
        Parameters:
        -----------
        fiscal_year : int, default=2025
        Año fiscal para el cual calcular el benchmark
            
        Returns:
        --------
        pd.DataFrame
        DataFrame con el SCORE calculado para cada entidad (según dimensiones)
        """

        # filrtar en df solo valores del fiscal year analizado 
        self.df = self.df[self.df['fiscal_year'] == fiscal_year]
        # Obtener valores generales para el año fiscal dado
        general_values = self.year_data[self.year_data['fiscal_year'] == fiscal_year].iloc[0]

        # Agregar columnas de valores generales como referencias
        self.df['general_rate_nos'] = general_values['rate_nos']
        self.df['general_nos_su'] = general_values['nos_su']
        self.df['general_growth_volume'] = general_values['variation_rate_volume']
        self.df['general_growth_rate_nos'] = general_values['variation_rate_nos']

        # Inicializar SCORE en cero para todas las entidades
        self.df['SCORE'] = 0.0

        # PARÁMETROS DE CONFIGURACIÓN DEL MODELO
        # ======================================
        
        # Límites para ajustes normalizados (previenen valores extremos)
        max_adjustment_nos = 3          # Límite superior para rate_nos
        min_adjustment_nos = -3         # Límite inferior para rate_nos
        max_adjustment_growth = 3       # Límite superior para métricas de crecimiento
        min_adjustment_growth = -3      # Límite inferior para métricas de crecimiento
        max_adjustment_other = 3        # Límite superior para otras métricas
        min_adjustment_other = -3       # Límite inferior para otras métricas

        # Pesos específicos para cada componente (suman importancia relativa)
        peso_rate_nos = 1.0            # Peso máximo: Rate NOS es la métrica más crítica
        peso_growth_volume = 0.8       # Segundo peso: Crecimiento en volumen importante para expansión
        peso_growth_rate_nos = 0.8     # Tercer peso: Crecimiento en NOS rate importante para eficiencia
        peso_nos_su = 0.9              # Cuarto peso: Eficiencia NOS SU importante para rentabilidad

        # CÁLCULO DE COMPONENTES DEL SCORE
        # ================================

        # COMPONENTE 1: RATE NOS (Métrica de eficiencia principal)
        # Fórmula: ajuste = (valor_entidad - benchmark) / benchmark
        diff_nos = self.df['rate_nos'] - self.df['general_rate_nos']
        adj_nos = (diff_nos / (self.df['general_rate_nos'] + 1e-10)).clip(
            lower=min_adjustment_nos, upper=max_adjustment_nos
        )
        self.df['SCORE'] += peso_rate_nos * adj_nos

        # COMPONENTE 2: GROWTH RATE VOLUME (Métrica de crecimiento en volumen)
        # Lógica especial: Si crecimiento negativo, se usa valor completo como penalización
        diff_growth_volume = np.where(
            self.df['variation_rate_volume'] < 0,  
            self.df['variation_rate_volume'],  # Penalización completa si negativo
            self.df['variation_rate_volume'] - self.df['general_growth_volume']  # Diferencia vs benchmark si positivo
        )
        adj_growth = (diff_growth_volume / (abs(self.df['general_growth_volume']) + 1e-10)).clip(
            lower=min_adjustment_growth, upper=max_adjustment_growth
        )
        self.df['SCORE'] += peso_growth_volume * adj_growth

        # COMPONENTE 3: NOS SU EFFICIENCY (Métrica de eficiencia de ventas)
        # Misma lógica que growth volume
        diff_nos_su = np.where(
            self.df['nos_su'] < 0,  
            self.df['nos_su'],  # Penalización completa si negativo
            self.df['nos_su'] - self.df['general_nos_su']  # Diferencia vs benchmark si positivo
        )
        adj_nos_su = (diff_nos_su / (abs(self.df['general_nos_su']) + 1e-10)).clip(
            lower=min_adjustment_other, upper=max_adjustment_other
        )
        self.df['SCORE'] += peso_nos_su * adj_nos_su

        # COMPONENTE 4: GROWTH RATE NOS (Métrica de crecimiento en rate)
        diff_growth_nos = np.where(
            self.df['variation_rate_nos'] < 0,  
            self.df['variation_rate_nos'],  # Penalización completa si negativo
            self.df['variation_rate_nos'] - self.df['general_growth_rate_nos']  # Diferencia vs benchmark si positivo
        )
        adj_growth_nos = (diff_growth_nos / (abs(self.df['general_growth_rate_nos']) + 1e-10)).clip(
            lower=min_adjustment_growth, upper=max_adjustment_growth
        )
        self.df['SCORE'] += peso_growth_rate_nos * adj_growth_nos

        # PENALIZACIONES SIMPLIFICADAS
        # Solo penalizar casos críticos donde NOS es negativo (situación inaceptable)
        self.df['SCORE'] += np.where(self.df['nos'] < 0, -5.0, 0)

        # NORMALIZACIÓN POR RANKING PERCENTIL (distribución gradual)
        self.df['SCORE'] = self.df['SCORE'].rank(method='min', pct=True) * 10
        
        # ORDENAMIENTO Y SELECCIÓN DE RESULTADOS
        # ======================================
        self.df.sort_values(by='SCORE', ascending=False, inplace=True)

        # Seleccionar columnas relevantes para el análisis final (dinámicamente)
        # Columnas base que siempre se incluyen (solo las relevantes para SCORE)
        columnas_base = [
            # Métricas usadas en el cálculo del SCORE
            'rate_nos', 'general_rate_nos',
            'variation_rate_volume', 'general_growth_volume',
            'nos_su', 'general_nos_su', 
            'variation_rate_nos', 'general_growth_rate_nos',
            # SCORE final calculado
            'SCORE'
        ]
        
        # Construir lista de columnas finales
        columnas_finales = self.dimension_cols.copy()  # Empezar con columnas de dimensión
        columnas_finales.append('quarter')  # Agregar quarter siempre
        
        # Agregar columnas base
        for col in columnas_base:
            if col in self.df.columns and col not in columnas_finales:
                columnas_finales.append(col)
        
        self.df = self.df[columnas_finales]
        
        dimension_str = ' × '.join(self.dimension_cols)
        print(f"SCORE calculado exitosamente para {len(self.df)} entidades ({dimension_str}).")
        print(f"Rango de SCORE: {self.df['SCORE'].min():.2f} - {self.df['SCORE'].max():.2f}")
        print(f"SCORE promedio: {self.df['SCORE'].mean():.2f}")

        return self.df
    
    def graficar_score(self, bins=30):
        """
        Genera un histograma de la distribución del SCORE calculado.
        
        Parameters:
        -----------
        bins : int, default=30
            Número de bins para el histograma
            
        Returns:
        --------
        None
            Muestra el gráfico directamente
            
        Raises:
        -------
        Exception
            Si el SCORE no ha sido calculado previamente
            
        Notes:
        ------
        - Incluye curva de densidad (KDE) para mejor interpretación
        - Útil para identificar la distribución de desempeño
        - Permite detectar outliers o concentraciones de scores
        """
        if 'SCORE' not in self.df.columns:
            raise Exception("El SCORE no ha sido calculado. Ejecuta primero calcular_score().")
        
        dimension_str = ' × '.join(self.dimension_cols)
        
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df['SCORE'], bins=bins, kde=True, color='skyblue')
        plt.title(f'Distribución del SCORE de Desempeño\n({dimension_str})')
        plt.xlabel('SCORE (0-10)')
        plt.ylabel('Frecuencia')
        plt.grid(True, alpha=0.3)
        
        # Agregar estadísticas al gráfico
        mean_score = self.df['SCORE'].mean()
        median_score = self.df['SCORE'].median()
        plt.axvline(mean_score, color='red', linestyle='--', alpha=0.7, label=f'Media: {mean_score:.2f}')
        plt.axvline(median_score, color='orange', linestyle='--', alpha=0.7, label=f'Mediana: {median_score:.2f}')
        plt.legend()
        
        plt.show()

    def tabla_score_por_rango(self):
        """
        Crea una tabla resumen con la distribución por rangos de SCORE.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame con columnas ['Rango SCORE', 'Cantidad', 'Porcentaje (%)']
            
        Raises:
        -------
        Exception
            Si el SCORE no ha sido calculado previamente
            
        Notes:
        ------
        - Divide el SCORE en intervalos de 1 punto (0-1, 1-2, ..., 9-10)
        - Incluye conteo absoluto y porcentaje relativo
        - Útil para análisis de segmentación
        - Los rangos se interpretan como:
          * 8-10: Top performers (estrellas)
          * 6-8: Good performers (sólidos)
          * 4-6: Average performers (promedio)
          * 2-4: Underperformers (en riesgo)
          * 0-2: Poor performers (críticos)
        """
        if 'SCORE' not in self.df.columns:
            raise Exception("El SCORE no ha sido calculado. Ejecuta primero calcular_score().")
        
        # Definir los bins (intervalos) de 0 a 10 en pasos de 1
        bins = list(range(0, 11))  # [0,1,2,...,10]
        
        # Cortar la columna SCORE en estos bins
        self.df['score_rango'] = pd.cut(self.df['SCORE'], bins=bins, right=False, include_lowest=True)
        
        # Contar entidades por rango
        conteo = self.df['score_rango'].value_counts().sort_index()
        
        # Calcular porcentaje
        porcentaje = 100 * conteo / conteo.sum()
        
        # Crear DataFrame resumen
        resumen = pd.DataFrame({
            'Cantidad': conteo,
            'Porcentaje (%)': porcentaje.round(2)
        }).reset_index()
        
        resumen.rename(columns={'score_rango': 'Rango SCORE'}, inplace=True)
        
        return resumen