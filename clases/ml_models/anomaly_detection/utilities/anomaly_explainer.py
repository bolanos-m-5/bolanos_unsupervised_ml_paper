"""
Anomaly Explainer Utility

Clase utilitaria para explicar por qué ciertos registros son considerados anómalos.
Encapsula toda la lógica de análisis IQR y explicaciones.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


class AnomalyExplainer:
    """
    Utilidad para generar explicaciones de anomalías usando método IQR.
    """
    
    def __init__(self, iqr_multiplier: float = 1.5):
        """
        Inicializar el explicador.
        
        Parameters:
        - iqr_multiplier: Multiplicador para el rango intercuartílico
        """
        self.iqr_multiplier = iqr_multiplier
        self.bounds = None
        
    def calculate_iqr_bounds(self, test_df: pd.DataFrame, features: List[str]) -> None:
        """
        Calcular límites IQR para cada feature.
        
        Parameters:
        - test_df: DataFrame de prueba para calcular estadísticas
        - features: Lista de features a analizar
        """
        Q1 = test_df[features].quantile(0.25)
        Q3 = test_df[features].quantile(0.75)
        IQR = Q3 - Q1
        
        self.bounds = {
            'lower': Q1 - self.iqr_multiplier * IQR,
            'upper': Q3 + self.iqr_multiplier * IQR,
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR
        }
    
    def explain_single_anomaly(self, row: pd.Series, features: List[str]) -> dict:
        """
        Explicar una sola anomalía.
        
        Parameters:
        - row: Serie con los datos del registro
        - features: Lista de features a analizar
        
        Returns:
        - Dict con explicación del registro
        """
        if self.bounds is None:
            raise ValueError("IQR bounds not calculated. Call calculate_iqr_bounds first.")
        
        outlier_features = []
        
        for feature in features:
            value = row[feature]
            if pd.isna(value):
                continue
                
            lower_bound = self.bounds['lower'][feature]
            upper_bound = self.bounds['upper'][feature]
            
            if value < lower_bound:
                outlier_features.append(
                    f"{feature}: {value:.3f} < {lower_bound:.3f} (low)"
                )
            elif value > upper_bound:
                outlier_features.append(
                    f"{feature}: {value:.3f} > {upper_bound:.3f} (high)"
                )
        
        return {
            'outlier_features': outlier_features,
            'n_outlier_features': len(outlier_features),
            'explanation': '; '.join(outlier_features) if outlier_features else 'No obvious outliers'
        }
    
    def explain_anomalies(self, result_df: pd.DataFrame, 
                         features: List[str],
                         id_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Explicar todas las anomalías detectadas con formato UX mejorado.
        
        Parameters:
        - result_df: DataFrame con resultados de predicción
        - features: Lista de features usadas en el modelo
        - id_columns: Columnas identificadoras a incluir
        
        Returns:
        - DataFrame con explicaciones en formato amigable
        """
        # Filtrar solo anomalías
        anomalies = result_df[result_df['anomaly'] == True]
        
        if len(anomalies) == 0:

            return pd.DataFrame()
        
        # Calcular bounds si no están calculados
        if self.bounds is None:
            self.calculate_iqr_bounds(result_df, features)
        
        # Calcular medianas para cada feature
        medians = result_df[features].median()
        
        explanations = []
        
        for idx, row in anomalies.iterrows():
            explanation_row = {}
            
            # Agregar columnas ID primero
            if id_columns:
                for col in id_columns:
                    if col in row:
                        explanation_row[col] = row[col]
            
            # Agregar score si está disponible
            if 'anomaly_score' in row:
                explanation_row['anomaly_score'] = row['anomaly_score']
            
            # Contar outliers
            outlier_count = 0
            outlier_vars = []
            
            # Para cada feature, agregar valor, mediana y status
            for feature in features:
                value = row[feature]
                median_val = medians[feature]
                
                # Agregar valor y mediana
                explanation_row[f'{feature}'] = value
                explanation_row[f'median_{feature}'] = median_val
                
                # Determinar si es outlier
                if pd.notna(value):
                    lower_bound = self.bounds['lower'][feature]
                    upper_bound = self.bounds['upper'][feature]
                    
                    if value < lower_bound:
                        explanation_row[f'{feature}_status'] = 'bajo'
                        outlier_count += 1
                        outlier_vars.append(feature)
                    elif value > upper_bound:
                        explanation_row[f'{feature}_status'] = 'alto'
                        outlier_count += 1
                        outlier_vars.append(feature)
                    else:
                        explanation_row[f'{feature}_status'] = 'normal'
                else:
                    explanation_row[f'{feature}_status'] = 'N/A'
            
            # Agregar resumen de outliers
            explanation_row['outlier_count'] = outlier_count
            explanation_row['outlier_variables'] = f"outlier en {outlier_count} variables: {', '.join(outlier_vars)}" if outlier_vars else "sin outliers claros"
            
            explanations.append(explanation_row)
        
        # Crear DataFrame y ordenar por anomaly_score descendente
        explanations_df = pd.DataFrame(explanations)
        explanations_df = explanations_df.sort_values('anomaly_score', ascending=False)
        
        return explanations_df
    
    def print_explanation_summary(self, explanations_df: pd.DataFrame, top_n: int = 5) -> None:
        """
        Imprimir resumen de explicaciones con nuevo formato.
        
        Parameters:
        - explanations_df: DataFrame con explicaciones
        - top_n: Número de casos top a mostrar
        """
        if len(explanations_df) == 0:
            return
            
        print(f"📋 EXPLICACIONES DE ANOMALÍAS ({len(explanations_df)} casos)")
        print("=" * 50)
        
        for idx, exp in explanations_df.head(top_n).iterrows():
            # Mostrar ID si está disponible
            id_info = ""
            if 'customer' in exp:
                id_info = f"Cliente: {exp['customer']}"
            
            print(f"📍 {id_info}")
            print(f"   Anomaly Score: {exp['anomaly_score']:.4f}")
            print(f"   {exp['outlier_variables']}")
            print()
            print()
    
    def get_feature_outlier_stats(self, explanations_df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """
        Obtener estadísticas de qué features son más frecuentemente outliers.
        
        Parameters:
        - explanations_df: DataFrame con explicaciones
        - features: Lista de features
        
        Returns:
        - DataFrame con estadísticas por feature
        """
        if len(explanations_df) == 0:
            return pd.DataFrame()
        
        feature_stats = []
        
        for feature in features:
            # Contar cuántas veces cada feature aparece como outlier
            feature_mentions = explanations_df['explanation'].str.contains(feature, na=False).sum()
            feature_stats.append({
                'feature': feature,
                'outlier_count': feature_mentions,
                'outlier_percentage': feature_mentions / len(explanations_df) * 100
            })
        
        stats_df = pd.DataFrame(feature_stats)
        stats_df = stats_df.sort_values('outlier_count', ascending=False)
        
        return stats_df