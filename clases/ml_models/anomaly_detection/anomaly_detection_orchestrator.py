"""
Anomaly Detection Core - Orchestrator Version

Versión completamente rediseñada como orquestador que coordina
clases utilitarias especializadas para cada responsabilidad.

Este core es ahora súper simple y solo maneja la coordinación.
"""

import pandas as pd
from typing import List, Dict, Optional

# Importar utilidades especializadas
from .utilities import (
    DataValidator,
    DataScaler, 
    AnomalyPredictor,
    AnomalyExplainer,
    AnomalyReporter
)


class AnomalyDetectionOrchestrator:
    """
    🎯 ORQUESTADOR PRINCIPAL para detección de anomalías.
    
    Este core se enfoca ÚNICAMENTE en coordinar llamadas a utilidades especializadas.
    Cada utilidad maneja una responsabilidad específica:
    
    - DataValidator: Validación y limpieza
    - DataScaler: Escalado de datos
    - AnomalyPredictor: Entrenamiento y predicción
    - AnomalyExplainer: Explicaciones IQR
    - AnomalyReporter: Reportes y métricas
    """
    
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, features: List[str]):
        """
        Inicializar el orquestador.
        
        Parameters:
        - train_df: DataFrame de entrenamiento
        - test_df: DataFrame de prueba
        - features: Lista de features para el modelo
        """
        # Datos originales
        self.original_train_df = train_df.copy()
        self.original_test_df = test_df.copy()
        self.features = features
        
        # Utilidades especializadas
        self.validator = DataValidator()
        self.scaler = DataScaler()
        self.predictor = AnomalyPredictor()
        self.explainer = AnomalyExplainer()
        self.reporter = AnomalyReporter()
        
        # Estado del pipeline
        self.train_df_clean = None
        self.test_df_clean = None
        self.result_df = None
        self._optimizer = None
        self.id_columns = None
        
        self._initialize_data()
    
    def _initialize_data(self):
        """Inicializar y validar datos usando utilidades."""
        # Validar y limpiar usando DataValidator
        self.train_df_clean, self.test_df_clean = self.validator.validate_and_clean_pipeline(
            self.original_train_df, 
            self.original_test_df, 
            self.features
        )
    
    # =====================================
    # MÉTODOS DE COORDINACIÓN BÁSICOS
    # =====================================
    
    def fit_model(self, n_estimators: int = 100, contamination: float = 0.05, max_samples: str = 'auto'):
        """
        🤖 Entrenar modelo coordinando DataScaler y AnomalyPredictor.
        """
        # 1. Escalar datos de entrenamiento
        X_train_scaled = self.scaler.fit_transform_train(self.train_df_clean, self.features)
        
        # 2. Entrenar predictor
        self.predictor.fit_model(X_train_scaled, n_estimators, contamination, max_samples)
    
    def predict_anomalies(self, add_scores: bool = True) -> pd.DataFrame:
        """
        🔮 Predecir anomalías coordinando DataScaler y AnomalyPredictor.
        """
        # 1. Escalar datos de prueba
        X_test_scaled, valid_mask = self.scaler.transform_test(self.test_df_clean, self.features)
        
        # 2. Hacer predicciones
        predictions, scores = self.predictor.predict_anomalies(X_test_scaled, add_scores)
        
        # 3. Crear DataFrame resultado
        self.result_df = self.predictor.create_result_dataframe(
            self.test_df_clean, valid_mask, predictions, scores
        )
        
        return self.result_df
    
    def explain_anomalies(self, id_columns: Optional[List[str]] = None, verbose: bool = True) -> pd.DataFrame:
        """
        📋 Explicar anomalías usando AnomalyExplainer.
        
        Parameters:
        - id_columns: Columnas identificadoras para incluir en explicaciones
        - verbose: Si True, imprime resumen automáticamente en consola
        """
        if self.result_df is None:
            raise ValueError("No predictions available. Call predict_anomalies first.")
        
        # Usar explainer para generar explicaciones
        explanations_df = self.explainer.explain_anomalies(
            self.result_df, self.features, id_columns
        )
        
        # Imprimir resumen solo si verbose=True
        if verbose:
            self.explainer.print_explanation_summary(explanations_df)
        
        return explanations_df
    
    def report_anomalies(self, segment_columns: Optional[List[str]] = None, verbose: bool = True):
        """
        📊 Generar reportes usando AnomalyReporter.
        
        Parameters:
        - segment_columns: Columnas para reportes segmentados
        - verbose: Si True, imprime reportes automáticamente en consola
        """
        if self.result_df is None:
            raise ValueError("No predictions available. Call predict_anomalies first.")
        
        # Reporte básico
        metrics = self.reporter.basic_anomaly_report(self.result_df)
        if verbose:
            self.reporter.print_basic_report(metrics)
        
        # Reportes segmentados si se solicitan
        if segment_columns and verbose:
            for col in segment_columns:
                if col in self.result_df.columns:
                    segmented_stats = self.reporter.segmented_report(self.result_df, col)
                    self.reporter.print_segmented_report(segmented_stats)
        
        return metrics
    
    # =====================================
    # PROPIEDADES Y ACCESO A UTILIDADES
    # =====================================
    
    @property
    def optimizer(self):
        """Lazy loading del optimizer para métodos avanzados."""
        if self._optimizer is None:
            from .anomaly_optimizer import AnomalyOptimizer
            self._optimizer = AnomalyOptimizer(self)
        return self._optimizer
    
    def get_model_info(self) -> Dict:
        """Obtener información del modelo a través del predictor."""
        return self.predictor.get_model_info()
    
    def get_feature_scales(self) -> pd.DataFrame:
        """Obtener información de escalado a través del scaler."""
        return self.scaler.get_feature_scales(self.features)
    
    # =====================================
    # MÉTODOS DE CONVENIENCIA (UNA LLAMADA)
    # =====================================
    
    def fit_with_optimization(self) -> Dict:
        """
        ENTRENAR CON OPTIMIZACIÓN AUTOMÁTICA.
        
        Usa feature selection + optimización de hiperparámetros en una sola operación.
        Encuentra las mejores features y parámetros simultáneamente.
        
        Returns:
        - Dict con resultados de optimización
        """
        # 🔍 Optimización completa: features + hiperparámetros
        optimization_results = self.optimizer.evaluate_feature_combinations()
        
        # 🏆 Obtener mejores resultados
        best_combination = optimization_results['best_combination']
        best_features = best_combination['features']
        best_params = best_combination['params']
        
        # 🔄 Actualizar features del orquestador
        self.features = best_features
        
        # 🤖 Entrenar modelo final con mejores parámetros
        self.fit_model(**best_params)
        
        return {
            'best_combination': best_combination,
            'best_features': best_features,
            'best_params': best_params,
            'composite_score': best_combination['score'],  # Ahora es mean_score simplificado
            'all_results': optimization_results.get('all_results', pd.DataFrame()),
            'summary_by_size': optimization_results.get('summary_by_size', pd.DataFrame()),
            'total_evaluations': len(optimization_results.get('all_results', [])),
            'optimization_details': optimization_results
        }
    
    def analyze_complete(self, verbose: bool = True) -> Dict:
        """
        🎯 Análisis completo coordinando todas las utilidades.
        
        Parameters:
        - verbose: Si True, imprime automáticamente explicaciones y reportes
        """
        if self.predictor.model is None:
            raise ValueError("Model not trained. Call fit_model first.")
        
        # 1. Predicciones
        predictions = self.predict_anomalies(add_scores=True)
        
        # 2. Reportes - CONTROLAR VERBOSIDAD
        metrics = self.report_anomalies(verbose=verbose)
        
        # 3. Explicaciones (usar id_columns guardadas) - CONTROLAR VERBOSIDAD
        explanations = self.explain_anomalies(id_columns=self.id_columns, verbose=verbose)
        
        return {
            'predictions': predictions,
            'metrics': metrics,
            'explanations': explanations
        }
    
    def one_call_complete_analysis(self, 
                                   segment_columns: Optional[List[str]] = None,
                                   id_columns: Optional[List[str]] = None,
                                   optimize_model: bool = True,
                                   national_values: Optional[Dict] = None,
                                   verbose: bool = True) -> Dict:
        """
        ⚡ UNA SOLA LLAMADA: Análisis completo coordinando todo.
        
        Parameters:
        - segment_columns: Columnas para segmentar reportes
        - id_columns: Columnas identificadoras para incluir en resultados
        - optimize_model: Si True, ejecuta optimización automática
        - national_values: Dict con valores nacionales de referencia
        - verbose: Si True, imprime explicaciones detalladas
        """
        results = {}
        
        # Guardar id_columns para usar en explicaciones
        self.id_columns = id_columns
        
        # 1. Entrenar (con o sin optimización)
        if optimize_model:
            results['optimization'] = self.fit_with_optimization()
        else:
            self.fit_model()
        
        # 2. Análisis principal - PASAR PARÁMETRO VERBOSE
        results['main_analysis'] = self.analyze_complete(verbose=verbose)
        
        # 3. Análisis segmentado opcional (básico)
        if segment_columns:
            results['segmented_reports'] = {}
            for col in segment_columns:
                if col in self.result_df.columns:
                    segmented_stats = self.reporter.segmented_report(self.result_df, col)
                    results['segmented_reports'][col] = segmented_stats
                    # Solo imprimir si verbose=True
                    if verbose:
                        self.reporter.print_segmented_report(segmented_stats)
        
        # 4. Resumen de anomalías (solo features + is_anomaly + anomaly_score)
        columns_to_include = self.features + ['is_anomaly', 'anomaly_score']
        
        # Incluir id_columns si se especificaron
        if id_columns:
            columns_to_include = id_columns + columns_to_include
        
        # Filtrar columnas existentes
        available_columns = [col for col in columns_to_include if col in self.result_df.columns]
        
        # Crear resumen ordenado por anomaly_score descendente
        # Filtrar solo anomalías detectadas (is_anomaly == True)
        anomaly_summary = self.result_df[self.result_df['anomaly'] == True][available_columns].copy()
        anomaly_summary = anomaly_summary.sort_values('anomaly_score', ascending=False)
        
        # 5. Agregar valores nacionales como referencia si se proporcionaron
        if national_values is not None:
            # Convertir DataFrame a dict si es necesario
            if isinstance(national_values, pd.DataFrame):
                if len(national_values) > 0:
                    national_dict = national_values[self.features].iloc[0].to_dict()
                else:
                    national_dict = {}
            else:
                national_dict = national_values
            
            # Agregar columnas nacionales
            for feature in self.features:
                if feature in national_dict:
                    national_col_name = f'national_{feature}'
                    anomaly_summary[national_col_name] = national_dict[feature]
        
        results['anomaly_summary'] = anomaly_summary
        
        return results
        
    def export_optimization_results(self, file_prefix: str = "isolation_forest_optimization") -> Dict:
        """
        📊 EXPORTAR RESULTADOS
        
        Extrae y formatea los resultados de optimización en formato apropiado para paper académico.
        
        Parameters:
        - file_prefix: Prefijo para archivos exportados
        
        Returns:
        - Dict con tabla académica y estadísticas
        """
        if not hasattr(self, 'optimizer') or self.optimizer is None:
            return {}
            
        # Intentar extraer resultados desde diferentes ubicaciones
        optimization_data = {}
        
        # Primero intentar desde los últimos resultados si están disponibles
        if hasattr(self.optimizer, 'all_results') and self.optimizer.all_results is not None:
            optimization_data['all_results'] = self.optimizer.all_results
            
        # Si no, ejecutar optimización para obtener resultados
        if 'all_results' not in optimization_data:
            full_results = self.optimizer.evaluate_feature_combinations()
            optimization_data = full_results
            
        if 'all_results' not in optimization_data or optimization_data['all_results'] is None:
            return {}
            
        df_results = optimization_data['all_results']
        
        # Top 10 configuraciones más importantes para paper académico
        # Usar mean_score como métrica principal (composite_score es ahora simplemente mean_score)
        top_results = df_results.nlargest(10, 'mean_score')
        
        # Crear tabla académica formateada
        academic_table = pd.DataFrame({
            'Rank': range(1, len(top_results) + 1),
            'Features_Selected': top_results['feature_combination'].apply(lambda x: ', '.join(x)),
            'N_Features': top_results['n_features'],
            'Contamination': top_results['best_params'].apply(lambda x: x.get('contamination', 'N/A')),
            'N_Estimators': top_results['best_params'].apply(lambda x: x.get('n_estimators', 'N/A')),
            'Max_Samples': top_results['best_params'].apply(lambda x: x.get('max_samples', 'N/A')),
            'Mean_Score': top_results['mean_score'].round(4),
            # 'Score_Std': eliminado porque simplificamos el optimizer - ya no calculamos std
            'Mean_Score_Final': top_results['mean_score'].round(4)  # Score principal (antes composite_score)
        })
        
        # Estadísticas académicas
        stats = {
            'total_combinations_evaluated': len(df_results),
            'feature_range': f"{df_results['n_features'].min()}-{df_results['n_features'].max()}",
            'best_mean_score': df_results['mean_score'].max(),
            'mean_mean_score': df_results['mean_score'].mean().round(4),
            'std_mean_score': df_results['mean_score'].std().round(4),
        }
        
        # Análisis de features más frecuentes en top 5
        top5_features = []
        for features_list in top_results.head(5)['feature_combination']:
            top5_features.extend(features_list)
            
        from collections import Counter
        feature_frequency = Counter(top5_features)
        most_frequent_features = dict(feature_frequency.most_common())
        
        
        return {
            'academic_table': academic_table,
            'full_results': df_results,
            'statistics': stats,
            'feature_frequency': most_frequent_features,
        }
    
    def get_optimization_dataframe(self, results_data=None, top_n: int = 10) -> pd.DataFrame:
        """Get optimization results as a clean DataFrame."""
        # Get optimization data
        if results_data is not None and 'all_results' in results_data:
            opt_results = results_data['all_results']
        elif hasattr(self, '_optimizer') and self._optimizer is not None:
            if hasattr(self._optimizer, 'all_results') and self._optimizer.all_results is not None:
                opt_results = self._optimizer.all_results
            else:
                return pd.DataFrame()
        else:
            return pd.DataFrame()
        
        if opt_results.empty:
            return pd.DataFrame()
        
        # Clean and format data
        opt_clean = opt_results.copy()
        
        # Extract parameters from best_params dict if present
        if 'best_params' in opt_clean.columns:
            opt_clean['contamination'] = opt_clean['best_params'].apply(
                lambda x: x.get('contamination', None) if isinstance(x, dict) else None
            )
            opt_clean['n_estimators'] = opt_clean['best_params'].apply(
                lambda x: x.get('n_estimators', None) if isinstance(x, dict) else None
            )
            opt_clean['max_samples'] = opt_clean['best_params'].apply(
                lambda x: x.get('max_samples', None) if isinstance(x, dict) else None
            )
        
        # Format feature combinations
        if 'feature_combination' in opt_clean.columns:
            opt_clean['features'] = opt_clean['feature_combination'].apply(
                lambda x: ', '.join(x) if isinstance(x, list) else str(x)
            )
        
        # Create final DataFrame
        optimization_df = pd.DataFrame({
            'rank': range(1, len(opt_clean) + 1),
            'mean_score': opt_clean['mean_score'].round(6),
            'n_features': opt_clean['n_features'],
            'contamination': opt_clean.get('contamination', None),
            'n_estimators': opt_clean.get('n_estimators', None), 
            'max_samples': opt_clean.get('max_samples', None),
            'features': opt_clean.get('features', opt_clean.get('feature_combination', 'N/A'))
        })
        
        # Sort by score and limit to top_n
        optimization_df = optimization_df.sort_values('mean_score', ascending=False)
        optimization_df['rank'] = range(1, len(optimization_df) + 1)
        
        return optimization_df.head(top_n)
    
    