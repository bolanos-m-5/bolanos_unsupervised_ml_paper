"""
Anomaly Detection Optimizer - Simplified Academic Version

Clase simplificada para optimización de hiperparámetros y evaluación
de combinaciones de features. Enfoque académico con métodos esenciales.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import ParameterGrid
from itertools import combinations
from typing import Dict, Optional, List


class AnomalyOptimizer:
    """
    Optimizador simplificado para detección de anomalías.
    
    Funcionalidades principales:
    - Optimización de hiperparámetros
    - Evaluación de combinaciones de features (feature selection)
    """
    
    def __init__(self, core_detector=None):
        """
        Inicializar optimizador.
        
        Parameters:
        - core_detector: Instancia del detector principal
        """
        self.core_detector = core_detector
    
    def _evaluate_single_feature_combination(self, features: List[str], param_grid: Dict) -> Dict:
        """
        🔧 Evaluar una combinación específica de features con mean_score.
        
        Parameters:
        - features: Lista de features a usar
        - param_grid: Grid de parámetros a probar
        
        Returns:
        - Dict con mejores parámetros y scores para esta combinación
        """
        # Preparar datos con solo las features seleccionadas
        X_train = self.core_detector.train_df_clean[features].dropna()
        X_test = self.core_detector.test_df_clean[features].dropna()
        
        # Usar los métodos del DataScaler del orchestrator
        X_train_scaled = self.core_detector.scaler.scaler.fit_transform(X_train)
        X_test_scaled = self.core_detector.scaler.scaler.transform(X_test)
        
        all_results = []
        
        # Probar cada configuración de parámetros
        for params in ParameterGrid(param_grid):
            model = IsolationForest(random_state=42, **params)
            model.fit(X_train_scaled)
            
            # Solo calcular mean_score (es lo más importante)
            test_scores = -model.decision_function(X_test_scaled)
            mean_score = test_scores.mean()
            
            all_results.append({
                **params,
                'mean_score': mean_score
            })
        
        # Simplemente usar el mejor mean_score
        results_df = pd.DataFrame(all_results)
        best_idx = results_df['mean_score'].idxmax()
        best_result = results_df.loc[best_idx]
        
        return {
            'best_params': {k: v for k, v in best_result.items() if k in param_grid.keys()},
            'mean_score': best_result['mean_score']
        }
    
    def evaluate_feature_combinations(self, feature_sizes: List[int] = None, 
                                     param_grid: Optional[Dict] = None) -> Dict:
        """
        🔍 Evaluar diferentes combinaciones de features + hiperparámetros.
        
        Esta función prueba subsets de features de diferentes tamaños
        para encontrar la mejor combinación de variables y parámetros.
        
        Parameters:
        - feature_sizes: Lista de tamaños de combinaciones a probar (ej: [2, 3, 4])
        - param_grid: Grid de hiperparámetros (usa default si None)
        - max_combinations: Máximo número de combinaciones por tamaño
        
        Returns:
        - Dict con mejores combinaciones y resultados
        """
        if not self.core_detector:
            raise ValueError("Core detector no asignado")
        
        all_features = self.core_detector.features.copy()
        
        # Tamaños por defecto
        if feature_sizes is None:
            max_size = len(all_features)
            feature_sizes = list(range(2, min(max_size + 1, 6)))  # 2 a 5 features
        
        # Grid por defecto - VALORES CONSERVADORES
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'contamination': [0.01,0.03, 0.05],
                'max_samples': ['auto', 50, 170]
            }
        
        print(f"🔍 Evaluando combinaciones de features: {feature_sizes}")
        print(f"   Features disponibles: {all_features}")
        
        all_results = []
        
        for size in feature_sizes:
            print(f"\n🎯 Probando combinaciones de {size} features...")
            
            # Generar todas las combinaciones de este tamaño
            feature_combinations = list(combinations(all_features, size))
            
            for i, feature_combo in enumerate(feature_combinations):
                
                # Evaluar esta combinación con grid search
                combo_results = self._evaluate_single_feature_combination(
                    list(feature_combo), param_grid
                )
                
                # Agregar metadata
                combo_results['feature_combination'] = list(feature_combo)
                combo_results['n_features'] = len(feature_combo)
                
                all_results.append(combo_results)
        
        # Encontrar la mejor combinación global - SIMPLIFICADO
        results_df = pd.DataFrame(all_results)
        
        # Usar mean_score directamente (más simple y confiable)
        best_idx = results_df['mean_score'].idxmax()
        best_result = results_df.loc[best_idx]
        
        print(f"\n🏆 MEJOR COMBINACIÓN ENCONTRADA:")
        print(f"   Features: {best_result['feature_combination']}")
        print(f"   Parámetros: {best_result['best_params']}")
        print(f"   Mean Score: {best_result['mean_score']:.4f}")
        
        return {
            'best_combination': {
                'features': best_result['feature_combination'],
                'params': best_result['best_params'],
                'score': best_result['mean_score'],
                'n_features': best_result['n_features']
            },
            'all_results': results_df,
            'summary_by_size': results_df.groupby('n_features')['mean_score'].agg(['mean', 'max', 'count'])
        }


