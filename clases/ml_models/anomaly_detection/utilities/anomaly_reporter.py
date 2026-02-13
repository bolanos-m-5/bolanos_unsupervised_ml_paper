"""
Anomaly Reporter Utility

Clase utilitaria para generar reportes y métricas de anomalías.
Encapsula toda la lógica de reporting y presentación de resultados.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


class AnomalyReporter:
    """
    Utilidad para generar reportes de detección de anomalías.
    """
    
    @staticmethod
    def basic_anomaly_report(result_df: pd.DataFrame) -> Dict:
        """
        Generar reporte básico de anomalías detectadas.
        
        Parameters:
        - result_df: DataFrame con resultados de predicción
        
        Returns:
        - Dict con métricas básicas
        """
        total_records = len(result_df)
        anomalies_count = result_df['anomaly'].sum()
        normal_count = total_records - anomalies_count
        
        metrics = {
            'total_records': total_records,
            'anomalies_count': anomalies_count,
            'normal_count': normal_count,
            'anomaly_percentage': (anomalies_count / total_records * 100) if total_records > 0 else 0,
            'normal_percentage': (normal_count / total_records * 100) if total_records > 0 else 0
        }
        
        # Estadísticas de scores si están disponibles
        if 'anomaly_score' in result_df.columns:
            scores = result_df['anomaly_score'].dropna()
            if len(scores) > 0:
                metrics.update({
                    'score_min': scores.min(),
                    'score_max': scores.max(),
                    'score_mean': scores.mean(),
                    'score_median': scores.median(),
                    'score_std': scores.std()
                })
        
        return metrics
    
    @staticmethod
    def print_basic_report(metrics: Dict) -> None:
        """
        Imprimir reporte básico en consola.
        
        Parameters:
        - metrics: Dict con métricas del reporte básico
        """
        print(f"\nAnomalías: {metrics['anomalies_count']}/{metrics['total_records']} ({metrics['anomaly_percentage']:.1f}%)")
        
        # Información de scores si está disponible
        if 'score_min' in metrics:
            print(f"   Score range: {metrics['score_min']:.4f} - {metrics['score_max']:.4f}")
    
    @staticmethod
    def segmented_report(result_df: pd.DataFrame, segment_column: str) -> pd.DataFrame:
        """
        Generar reporte segmentado por una columna específica.
        
        Parameters:
        - result_df: DataFrame con resultados
        - segment_column: Columna para segmentar el análisis
        
        Returns:
        - DataFrame con métricas por segmento
        """
        if segment_column not in result_df.columns:
            raise ValueError(f"Column '{segment_column}' not found in DataFrame")
        
        segmented_stats = []
        
        for segment_value in result_df[segment_column].unique():
            segment_data = result_df[result_df[segment_column] == segment_value]
            segment_metrics = AnomalyReporter.basic_anomaly_report(segment_data)
            segment_metrics['segment'] = segment_value
            segment_metrics['segment_column'] = segment_column
            segmented_stats.append(segment_metrics)
        
        stats_df = pd.DataFrame(segmented_stats)
        stats_df = stats_df.sort_values('anomaly_percentage', ascending=False)
        
        return stats_df
    
    @staticmethod
    def print_segmented_report(segmented_stats: pd.DataFrame, top_n: int = 10) -> None:
        """
        Imprimir reporte segmentado en consola.
        
        Parameters:
        - segmented_stats: DataFrame con estadísticas segmentadas
        - top_n: Número de segmentos top a mostrar
        """
        if len(segmented_stats) == 0:
            return
        
        for _, row in segmented_stats.head(top_n).iterrows():
            print(f"   {row['segment']}: {row['anomalies_count']}/{row['total_records']} ({row['anomaly_percentage']:.1f}%)")
    
    @staticmethod
    def anomaly_distribution_report(result_df: pd.DataFrame, score_bins: int = 10) -> Dict:
        """
        Analizar distribución de scores de anomalías.
        
        Parameters:
        - result_df: DataFrame con resultados
        - score_bins: Número de bins para la distribución
        
        Returns:
        - Dict con información de distribución
        """
        if 'anomaly_score' not in result_df.columns:
            return {"error": "No anomaly scores available"}
        
        scores = result_df['anomaly_score'].dropna()
        if len(scores) == 0:
            return {"error": "No valid scores found"}
        
        # Crear bins y contar distribución
        bins = pd.cut(scores, bins=score_bins, include_lowest=True)
        distribution = bins.value_counts().sort_index()
        
        # Encontrar threshold sugerido (percentil 95)
        suggested_threshold = scores.quantile(0.95)
        high_score_count = (scores >= suggested_threshold).sum()
        
        return {
            'distribution': distribution,
            'suggested_threshold': suggested_threshold,
            'high_score_count': high_score_count,
            'high_score_percentage': high_score_count / len(scores) * 100,
            'total_scores': len(scores)
        }
    
    @staticmethod
    def comprehensive_report(result_df: pd.DataFrame, 
                           segment_columns: Optional[list] = None,
                           explanations_df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Generar reporte comprensivo que combina todas las métricas.
        
        Parameters:
        - result_df: DataFrame con resultados
        - segment_columns: Lista de columnas para análisis segmentado
        - explanations_df: DataFrame con explicaciones (opcional)
        
        Returns:
        - Dict con reporte completo
        """
        comprehensive = {}
        
        # Reporte básico
        comprehensive['basic'] = AnomalyReporter.basic_anomaly_report(result_df)
        
        # Reportes segmentados
        if segment_columns:
            comprehensive['segmented'] = {}
            for col in segment_columns:
                if col in result_df.columns:
                    comprehensive['segmented'][col] = AnomalyReporter.segmented_report(result_df, col)
        
        # Distribución de scores
        comprehensive['score_distribution'] = AnomalyReporter.anomaly_distribution_report(result_df)
        
        # Estadísticas de explicaciones
        if explanations_df is not None and len(explanations_df) > 0:
            comprehensive['explanations_summary'] = {
                'total_explained': len(explanations_df),
                'avg_outlier_features': explanations_df['n_outlier_features'].mean(),
                'max_outlier_features': explanations_df['n_outlier_features'].max(),
                'most_common_explanations': explanations_df['explanation'].value_counts().head()
            }
        
        return comprehensive