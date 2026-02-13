"""
Feature selection methods for clustering analysis - CLEANED VERSION.
Contains only essential methods: SFS, RFE, and Grid Search.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import silhouette_score

from .estimator import HierarchicalClusteringEstimator


class ClusteringFeatureSelector:
    """
    Feature selection methods for clustering analysis.
    Supports SFS, RFE, and grid search combination.
    """
    
    def __init__(self, data):
        """
        Initialize feature selector with RobustScaler.
        
        Parameters:
        - data: DataFrame with features
        """
        self.data_original = data
        self.scaler = None
        self.data_scaled = None
        self._setup_data()
    
    def _setup_data(self):
        """Setup data with RobustScaler."""
        if not isinstance(self.data_original, pd.DataFrame):
            raise ValueError("Data must be a DataFrame with column names")
        
        self.scaler = RobustScaler()
        scaled_array = self.scaler.fit_transform(self.data_original)
        self.data_scaled = pd.DataFrame(
            scaled_array,
            columns=self.data_original.columns,
            index=self.data_original.index
        )
    
    def sequential_forward_selection(self, n_features_to_select='auto', method='ward', 
                                   n_clusters=4, metric='euclidean', criterion='silhouette', 
                                   tol=0.001, verbose=False, min_features=2, required_features=None):
        """
        Sequential Forward Selection using sklearn's SequentialFeatureSelector - SIMPLIFIED.
        
        Parameters:
        - n_features_to_select: Number of features to select ('auto', int, or float)
        - min_features: Minimum number of features to select
        - required_features: List of features that must ALWAYS be included
        """
        required_features = required_features or []
        
        # Handle case with only required features
        selectable_features = [f for f in self.data_original.columns if f not in required_features]
        
        if len(selectable_features) == 0:
            # Only required features available
            selected_features = required_features
            data_selected = self.data_scaled[selected_features]
            
            estimator_final = HierarchicalClusteringEstimator(
                method=method, n_clusters=n_clusters, metric=metric, criterion_metric=criterion
            )
            estimator_final.fit(data_selected)
            scores = self._calculate_all_metrics(data_selected, estimator_final.labels_)
            
            return {
                'selected_features': selected_features,
                'scores': scores,
                'selector': None,
                'estimator': estimator_final
            }
        
        # Use sklearn directly for feature selection
        estimator = HierarchicalClusteringEstimator(
            method=method, n_clusters=n_clusters, metric=metric, criterion_metric=criterion
        )
        
        # Determine number of features to select from selectable ones
        if n_features_to_select == 'auto':
            # Priorizar min_features del usuario, solo usar mitad si no se especifica
            needed_features = max(min_features - len(required_features), 1)
            max_possible = len(selectable_features)
            n_to_select = min(needed_features, max_possible)
        elif isinstance(n_features_to_select, float):
            n_to_select = max(1, int(n_features_to_select * len(selectable_features)))
        else:
            n_to_select = max(1, min(n_features_to_select - len(required_features), len(selectable_features)))
        
        # Apply SFS to selectable features only
        data_selectable = self.data_scaled[selectable_features]
        
        sfs = SequentialFeatureSelector(
            estimator=estimator,
            n_features_to_select=n_to_select,
            direction='forward',
            cv=None,
            n_jobs=1,
            tol=tol if n_features_to_select == 'auto' else None
        )
        
        sfs.fit(data_selectable)
        auto_selected = list(data_selectable.columns[sfs.get_support()])
        selected_features = required_features + auto_selected
        
        # Calculate final metrics
        data_selected = self.data_scaled[selected_features]
        estimator_final = HierarchicalClusteringEstimator(
            method=method, n_clusters=n_clusters, metric=metric, criterion_metric=criterion
        )
        estimator_final.fit(data_selected)
        scores = self._calculate_all_metrics(data_selected, estimator_final.labels_)
        
        return {
            'selected_features': selected_features,
            'scores': scores,
            'selector': sfs,
            'estimator': estimator_final
        }
    
    def recursive_feature_elimination(self, n_features_to_select='auto', method='ward',
                                    n_clusters=4, metric='euclidean', criterion='silhouette',
                                    tol=0.001, verbose=False, min_features=2, required_features=None):
        """
        Recursive Feature Elimination using sklearn's SequentialFeatureSelector - SIMPLIFIED.
        
        Parameters:
        - n_features_to_select: Number of features to select ('auto', int, or float)
        - min_features: Minimum number of features to select
        - required_features: List of features that must ALWAYS be included
        """
        required_features = required_features or []
        
        # Handle case with only required features
        selectable_features = [f for f in self.data_original.columns if f not in required_features]
        
        if len(selectable_features) == 0:
            # Only required features available
            selected_features = required_features
            data_selected = self.data_scaled[selected_features]
            
            estimator_final = HierarchicalClusteringEstimator(
                method=method, n_clusters=n_clusters, metric=metric, criterion_metric=criterion
            )
            estimator_final.fit(data_selected)
            scores = self._calculate_all_metrics(data_selected, estimator_final.labels_)
            
            return {
                'selected_features': selected_features,
                'scores': scores,
                'selector': None,
                'estimator': estimator_final
            }
        
        # Use sklearn directly for feature elimination
        estimator = HierarchicalClusteringEstimator(
            method=method, n_clusters=n_clusters, metric=metric, criterion_metric=criterion
        )
        
        # Determine number of features to select from selectable ones
        if n_features_to_select == 'auto':
            # Priorizar min_features del usuario, solo usar mitad si no se especifica
            needed_features = max(min_features - len(required_features), 1)
            max_possible = len(selectable_features)
            n_to_select = min(needed_features, max_possible)
        elif isinstance(n_features_to_select, float):
            n_to_select = max(1, int(n_features_to_select * len(selectable_features)))
        else:
            n_to_select = max(1, min(n_features_to_select - len(required_features), len(selectable_features)))
        
        # Apply RFE to selectable features only
        data_selectable = self.data_scaled[selectable_features]
        
        rfe = SequentialFeatureSelector(
            estimator=estimator,
            n_features_to_select=n_to_select,
            direction='backward',
            cv=None,
            n_jobs=1,
            tol=tol if n_features_to_select == 'auto' else None
        )
        
        rfe.fit(data_selectable)
        auto_selected = list(data_selectable.columns[rfe.get_support()])
        selected_features = required_features + auto_selected
        
        # Calculate final metrics
        data_selected = self.data_scaled[selected_features]
        estimator_final = HierarchicalClusteringEstimator(
            method=method, n_clusters=n_clusters, metric=metric, criterion_metric=criterion
        )
        estimator_final.fit(data_selected)
        scores = self._calculate_all_metrics(data_selected, estimator_final.labels_)
        
        return {
            'selected_features': selected_features,
            'scores': scores,
            'selector': rfe,
            'estimator': estimator_final
        }
    
    def sfs_rfe_grid_search(self, param_grid=None, min_features=3, required_features=None, 
                           top_n=10, verbose=False):
        """
        SFS + RFE con Grid Search de hiperparámetros.
        Ejecuta SFS y RFE para cada combinación de hiperparámetros.
        
        Parameters:
        - param_grid: Diccionario con hiperparámetros a probar
        - min_features: Mínimo número de features
        - required_features: Features obligatorias
        - top_n: Número de mejores resultados a retornar
        - verbose: Mostrar progreso
        
        Returns:
        - Diccionario con resultados de grid search
        """
        required_features = required_features or []
        
        # Usar grid financiero completo si no se proporciona uno
        if param_grid is None:
            from .parameter_grids import ClusteringParameterGrid
            grid_builder = ClusteringParameterGrid(min_clusters=3, max_clusters=7)
            financial_grid = grid_builder.get_financial_grid()
            
            param_grid = {
                'valid_combinations': financial_grid['valid_combinations'],
                'n_clusters': financial_grid['n_clusters']
            }
        
        all_results = []
        experiment_count = 0
        
        # Procesar grid financiero
        if 'valid_combinations' in param_grid:
            # Nuevo formato: usar combinaciones válidas predefinidas
            for method, metric in param_grid['valid_combinations']:
                for n_clusters in param_grid['n_clusters']:
                    experiment_count += 1
                    
                    # Ejecutar SFS
                    try:
                        sfs_result = self.sequential_forward_selection(
                            method=method, n_clusters=n_clusters, metric=metric,
                            min_features=min_features, required_features=required_features,
                            verbose=False
                        )
                        
                        sfs_score = sfs_result['scores']['silhouette']
                        sfs_features = sfs_result['selected_features']
                        
                        all_results.append({
                            'method_type': 'SFS',
                            'hyperparams_method': method,
                            'hyperparams_metric': metric,
                            'hyperparams_n_clusters': n_clusters,
                            'selected_features': sfs_features,
                            'n_features': len(sfs_features),
                            'silhouette_score': sfs_score,
                            'algorithm': f"SFS-{method}-{metric}-C{n_clusters}",
                            'full_result': sfs_result
                        })
                            
                    except Exception:
                        pass
                    
                    # Ejecutar RFE
                    try:
                        rfe_result = self.recursive_feature_elimination(
                            method=method, n_clusters=n_clusters, metric=metric,
                            min_features=min_features, required_features=required_features,
                            verbose=False
                        )
                        
                        rfe_score = rfe_result['scores']['silhouette']
                        rfe_features = rfe_result['selected_features']
                        
                        all_results.append({
                            'method_type': 'RFE',
                            'hyperparams_method': method,
                            'hyperparams_metric': metric,  
                            'hyperparams_n_clusters': n_clusters,
                            'selected_features': rfe_features,
                            'n_features': len(rfe_features),
                            'silhouette_score': rfe_score,
                            'algorithm': f"RFE-{method}-{metric}-C{n_clusters}",
                            'full_result': rfe_result
                        })
                            
                    except Exception:
                        pass
        
        # Procesar resultados
        if not all_results:
            return {'error': 'No successful experiments'}
        
        # Convertir a DataFrame y ordenar por silhouette score
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values('silhouette_score', ascending=False).reset_index(drop=True)
        
        # Extraer mejor resultado
        best = results_df.iloc[0]
        
        # Separar mejores resultados por método
        sfs_results = results_df[results_df['method_type'] == 'SFS'].head(top_n)
        rfe_results = results_df[results_df['method_type'] == 'RFE'].head(top_n)
        
        return {
            # Mejor resultado global
            'best_overall': {
                'method_type': best['method_type'],
                'selected_features': best['selected_features'],
                'hyperparams': {
                    'method': best['hyperparams_method'],
                    'metric': best['hyperparams_metric'],
                    'n_clusters': best['hyperparams_n_clusters']
                },
                'scores': {'silhouette': best['silhouette_score']},
                'algorithm': best['algorithm']
            },
            
            # Resultados detallados
            'grid_search_results': {
                'all_results_df': results_df,
                'sfs_results': sfs_results,
                'rfe_results': rfe_results,
                'total_experiments': len(results_df)
            },
            
            # Comparación SFS vs RFE
            'method_comparison': {
                'best_sfs': {
                    'features': sfs_results.iloc[0]['selected_features'],
                    'hyperparams': {
                        'method': sfs_results.iloc[0]['hyperparams_method'],
                        'metric': sfs_results.iloc[0]['hyperparams_metric'],
                        'n_clusters': sfs_results.iloc[0]['hyperparams_n_clusters']
                    },
                    'silhouette': sfs_results.iloc[0]['silhouette_score']
                },
                'best_rfe': {
                    'features': rfe_results.iloc[0]['selected_features'],
                    'hyperparams': {
                        'method': rfe_results.iloc[0]['hyperparams_method'],
                        'metric': rfe_results.iloc[0]['hyperparams_metric'],
                        'n_clusters': rfe_results.iloc[0]['hyperparams_n_clusters']
                    },
                    'silhouette': rfe_results.iloc[0]['silhouette_score']
                },
                'winner': sfs_results.iloc[0]['algorithm'] if sfs_results.iloc[0]['silhouette_score'] > rfe_results.iloc[0]['silhouette_score'] else rfe_results.iloc[0]['algorithm']
            },
            
            # Top N resultados para análisis
            'top_results': results_df.head(top_n)
        }
    
    def _calculate_all_metrics(self, data, labels):
        """Calculate clustering metrics (Silhouette Score only)."""
        try:
            sil = silhouette_score(data, labels)
            return {'silhouette': sil}
        except Exception:
            return {'silhouette': -1.0}