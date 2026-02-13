"""
Feature selection methods for clustering analysis.
"""

import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster

from .estimator import HierarchicalClusteringEstimator


class ClusteringFeatureSelector:
    """
    Feature selection methods for clustering analysis.
    Supports SFS, RFE, and exhaustive search.
    """
    
    def __init__(self, data, scaler='robust'):
        """
        Initialize feature selector.
        
        Parameters:
        - data: DataFrame with features
        - scaler: 'robust', 'standard', or None
        """
        self.data_original = data
        self.scaler_type = scaler
        self.scaler = None
        self.data_scaled = None
        self._setup_data()
    
    def _setup_data(self):
        """Unified data scaling logic."""
        if not isinstance(self.data_original, pd.DataFrame):
            raise ValueError("Data must be a DataFrame with column names")
        
        if self.scaler_type == 'robust':
            self.scaler = RobustScaler()
            scaled_array = self.scaler.fit_transform(self.data_original)
            self.data_scaled = pd.DataFrame(
                scaled_array,
                columns=self.data_original.columns,
                index=self.data_original.index
            )
            print("✓ Data scaled using RobustScaler")
        elif self.scaler_type == 'standard':
            self.scaler = StandardScaler()
            scaled_array = self.scaler.fit_transform(self.data_original)
            self.data_scaled = pd.DataFrame(
                scaled_array,
                columns=self.data_original.columns,
                index=self.data_original.index
            )
            print("✓ Data scaled using StandardScaler")
        else:
            self.data_scaled = self.data_original.copy()
            print("⚠️ No scaling applied")
    
    def sequential_forward_selection(self, n_features_to_select='auto', method='ward', 
                                   n_clusters=4, metric='euclidean', criterion='silhouette', 
                                   tol=0.001, verbose=True, min_features=2, required_features=None):
        """
        Sequential Forward Selection using sklearn's SequentialFeatureSelector.
        
        Parameters:
        - min_features: Mínimo número de features a seleccionar (por defecto 2)
        - required_features: Lista de features que SIEMPRE deben estar incluidas (ej: ['SCORE'])
        """
        required_features = required_features or []
        
        if verbose:
            print(f"\n🔍 SEQUENTIAL FORWARD SELECTION")
            print(f"   Features disponibles: {list(self.data_original.columns)}")
            print(f"   Features obligatorias: {required_features if required_features else 'Ninguna'}")
            print(f"   Method: {method}, Metric: {metric}, Clusters: {n_clusters}")
            print(f"   Mínimo features: {min_features}")
        
        # Separar features obligatorias de las que se seleccionarán automáticamente
        selectable_features = [f for f in self.data_original.columns if f not in required_features]
        
        # ===== NUEVO: ALMACENAR TODOS LOS RESULTADOS =====
        all_results = []
        
        if len(selectable_features) == 0:
            # Solo hay features obligatorias
            selected_features = required_features
            
            # Calcular score para features obligatorias
            data_selected = self.data_scaled[selected_features]
            estimator_final = HierarchicalClusteringEstimator(
                method=method, n_clusters=n_clusters, metric=metric, criterion_metric=criterion
            )
            estimator_final.fit(data_selected)
            scores = self._calculate_all_metrics(data_selected, estimator_final.labels_)
            
            all_results.append({
                'selected_features': selected_features,
                'silhouette_score': scores['silhouette'],
                'n_features': len(selected_features),
                'method': 'sfs',
                'clustering_method': method,
                'metric': metric,
                'n_clusters': n_clusters
            })
            
        else:
            # ===== PROBAR DIFERENTES NÚMEROS DE FEATURES (2 hasta total disponible) =====
            max_features = len(self.data_original.columns)
            
            for n_features_test in range(min_features, max_features + 1):
                try:
                    # Calcular cuántas features adicionales necesitamos
                    n_required = len(required_features)
                    n_additional_needed = max(0, n_features_test - n_required)
                    
                    if n_additional_needed <= 0:
                        # Solo usar features obligatorias
                        test_selected_features = required_features
                    elif n_additional_needed >= len(selectable_features):
                        # Usar todas las features
                        test_selected_features = required_features + selectable_features
                    else:
                        # Ejecutar selección para este número específico
                        data_selectable = self.data_scaled[selectable_features]
                        
                        estimator = HierarchicalClusteringEstimator(
                            method=method, n_clusters=n_clusters, 
                            metric=metric, criterion_metric=criterion
                        )
                        
                        sfs = SequentialFeatureSelector(
                            estimator=estimator,
                            n_features_to_select=n_additional_needed,
                            direction='forward',
                            cv=None,
                            n_jobs=1,
                            tol=None  # Sin tolerancia para número exacto
                        )
                        
                        sfs.fit(data_selectable)
                        auto_selected = list(data_selectable.columns[sfs.get_support()])
                        
                        # Combinar features obligatorias + seleccionadas
                        test_selected_features = required_features + auto_selected
                    
                    # Calcular métricas para esta configuración
                    data_selected = self.data_scaled[test_selected_features]
                    estimator_final = HierarchicalClusteringEstimator(
                        method=method, n_clusters=n_clusters, metric=metric, criterion_metric=criterion
                    )
                    estimator_final.fit(data_selected)
                    scores = self._calculate_all_metrics(data_selected, estimator_final.labels_)
                    
                    all_results.append({
                        'selected_features': test_selected_features,
                        'silhouette_score': scores['silhouette'],
                        'n_features': len(test_selected_features),
                        'method': 'sfs',
                        'clustering_method': method,
                        'metric': metric,
                        'n_clusters': n_clusters
                    })
                    
                except Exception as e:
                    if verbose:
                        print(f"   ⚠️ Error con {n_features_test} features: {e}")
                    continue
        
        # Ordenar resultados por score y seleccionar el mejor
        if all_results:
            all_results_df = pd.DataFrame(all_results)
            all_results_df = all_results_df.sort_values('silhouette_score', ascending=False).reset_index(drop=True)
            
            best_result = all_results_df.iloc[0]
            selected_features = best_result['selected_features']
            scores = {'silhouette': best_result['silhouette_score']}
            
            if verbose:
                print(f"✅ Selected features: {selected_features}")
                print(f"   Silhouette: {scores['silhouette']:.4f}")
                print(f"   Tested {len(all_results)} configurations")
        else:
            # Fallback al método original
            data_selectable = self.data_scaled[selectable_features]
            
            estimator = HierarchicalClusteringEstimator(
                method=method, n_clusters=n_clusters, 
                metric=metric, criterion_metric=criterion
            )
            
            sfs = SequentialFeatureSelector(
                estimator=estimator,
                n_features_to_select=n_features_to_select,
                direction='forward',
                cv=None,
                n_jobs=1,
                tol=tol if n_features_to_select == 'auto' else None
            )
            
            sfs.fit(data_selectable)
            auto_selected = list(data_selectable.columns[sfs.get_support()])
            selected_features = required_features + auto_selected
            
            data_selected = self.data_scaled[selected_features]
            estimator_final = HierarchicalClusteringEstimator(
                method=method, n_clusters=n_clusters, metric=metric, criterion_metric=criterion
            )
            estimator_final.fit(data_selected)
            scores = self._calculate_all_metrics(data_selected, estimator_final.labels_)
            
            all_results_df = pd.DataFrame([{
                'selected_features': selected_features,
                'silhouette_score': scores['silhouette'],
                'n_features': len(selected_features),
                'method': 'sfs',
                'clustering_method': method,
                'metric': metric,
                'n_clusters': n_clusters
            }])

        return {
            'selected_features': selected_features,
            'scores': scores,
            'selector': sfs if 'sfs' in locals() else None,
            'estimator': estimator_final if 'estimator_final' in locals() else None,
            'all_results': all_results_df.to_dict('records')  # ✅ NUEVO: Todos los resultados
        }
    
    def recursive_feature_elimination(self, n_features_to_select='auto', method='ward',
                                    n_clusters=4, metric='euclidean', criterion='silhouette',
                                    tol=0.001, verbose=True, min_features=2, required_features=None):
        """
        Recursive Feature Elimination using sklearn's SequentialFeatureSelector.
        
        Parameters:
        - min_features: Mínimo número de features a seleccionar (por defecto 2)
        - required_features: Lista de features que SIEMPRE deben estar incluidas (ej: ['SCORE'])
        """
        required_features = required_features or []
        
        if verbose:
            print(f"\n🔍 RECURSIVE FEATURE ELIMINATION")
            print(f"   Features disponibles: {list(self.data_original.columns)}")
            print(f"   Features obligatorias: {required_features if required_features else 'Ninguna'}")
            print(f"   Method: {method}, Metric: {metric}, Clusters: {n_clusters}")
            print(f"   Mínimo features: {min_features}")
        
        # Separar features obligatorias de las que se seleccionarán automáticamente
        selectable_features = [f for f in self.data_original.columns if f not in required_features]
        
        # ===== NUEVO: ALMACENAR TODOS LOS RESULTADOS =====
        all_results = []
        
        if len(selectable_features) == 0:
            # Solo hay features obligatorias
            selected_features = required_features
            
            # Calcular score para features obligatorias
            data_selected = self.data_scaled[selected_features]
            estimator_final = HierarchicalClusteringEstimator(
                method=method, n_clusters=n_clusters, metric=metric, criterion_metric=criterion
            )
            estimator_final.fit(data_selected)
            scores = self._calculate_all_metrics(data_selected, estimator_final.labels_)
            
            all_results.append({
                'selected_features': selected_features,
                'silhouette_score': scores['silhouette'],
                'n_features': len(selected_features),
                'method': 'rfe',
                'clustering_method': method,
                'metric': metric,
                'n_clusters': n_clusters
            })
            
        else:
            # ===== PROBAR DIFERENTES NÚMEROS DE FEATURES (2 hasta total disponible) =====
            max_features = len(self.data_original.columns)
            
            for n_features_test in range(min_features, max_features + 1):
                try:
                    # Calcular cuántas features adicionales necesitamos
                    n_required = len(required_features)
                    n_additional_needed = max(0, n_features_test - n_required)
                    
                    if n_additional_needed <= 0:
                        # Solo usar features obligatorias
                        test_selected_features = required_features
                    elif n_additional_needed >= len(selectable_features):
                        # Usar todas las features
                        test_selected_features = required_features + selectable_features
                    else:
                        # Ejecutar selección para este número específico
                        data_selectable = self.data_scaled[selectable_features]
                        
                        estimator = HierarchicalClusteringEstimator(
                            method=method, n_clusters=n_clusters,
                            metric=metric, criterion_metric=criterion
                        )
                        
                        rfe = SequentialFeatureSelector(
                            estimator=estimator,
                            n_features_to_select=n_additional_needed,
                            direction='backward',
                            cv=None,
                            n_jobs=1,
                            tol=None  # Sin tolerancia para número exacto
                        )
                        
                        rfe.fit(data_selectable)
                        auto_selected = list(data_selectable.columns[rfe.get_support()])
                        
                        # Combinar features obligatorias + seleccionadas
                        test_selected_features = required_features + auto_selected
                    
                    # Calcular métricas para esta configuración
                    data_selected = self.data_scaled[test_selected_features]
                    estimator_final = HierarchicalClusteringEstimator(
                        method=method, n_clusters=n_clusters, metric=metric, criterion_metric=criterion
                    )
                    estimator_final.fit(data_selected)
                    scores = self._calculate_all_metrics(data_selected, estimator_final.labels_)
                    
                    all_results.append({
                        'selected_features': test_selected_features,
                        'silhouette_score': scores['silhouette'],
                        'n_features': len(test_selected_features),
                        'method': 'rfe',
                        'clustering_method': method,  
                        'metric': metric,
                        'n_clusters': n_clusters
                    })
                    
                except Exception as e:
                    if verbose:
                        print(f"   ⚠️ Error con {n_features_test} features: {e}")
                    continue
        
        # Ordenar resultados por score y seleccionar el mejor
        if all_results:
            all_results_df = pd.DataFrame(all_results)
            all_results_df = all_results_df.sort_values('silhouette_score', ascending=False).reset_index(drop=True)
            
            best_result = all_results_df.iloc[0]
            selected_features = best_result['selected_features']
            scores = {'silhouette': best_result['silhouette_score']}
            
            if verbose:
                print(f"✅ Selected features: {selected_features}")
                print(f"   Silhouette: {scores['silhouette']:.4f}")
                print(f"   Tested {len(all_results)} configurations")
        else:
            # Fallback al método original
            data_selectable = self.data_scaled[selectable_features]
            
            estimator = HierarchicalClusteringEstimator(
                method=method, n_clusters=n_clusters,
                metric=metric, criterion_metric=criterion
            )
            
            rfe = SequentialFeatureSelector(
                estimator=estimator,
                n_features_to_select=n_features_to_select,
                direction='backward',
                cv=None,
                n_jobs=1,
                tol=tol if n_features_to_select == 'auto' else None
            )
            
            rfe.fit(data_selectable)
            auto_selected = list(data_selectable.columns[rfe.get_support()])
            selected_features = required_features + auto_selected
            
            data_selected = self.data_scaled[selected_features]
            estimator_final = HierarchicalClusteringEstimator(
                method=method, n_clusters=n_clusters, metric=metric, criterion_metric=criterion
            )
            estimator_final.fit(data_selected)
            scores = self._calculate_all_metrics(data_selected, estimator_final.labels_)
            
            all_results_df = pd.DataFrame([{
                'selected_features': selected_features,
                'silhouette_score': scores['silhouette'],
                'n_features': len(selected_features),
                'method': 'rfe',
                'clustering_method': method,
                'metric': metric,
                'n_clusters': n_clusters
            }])

        return {
            'selected_features': selected_features,
            'scores': scores,
            'selector': rfe if 'rfe' in locals() else None,
            'estimator': estimator_final if 'estimator_final' in locals() else None,
            'all_results': all_results_df.to_dict('records')  # ✅ NUEVO: Todos los resultados
        }
    
    def exhaustive_search(self, min_features=2, max_features=None, param_grid=None, 
                         top_n=10, verbose=True):
        """
        Exhaustive search for optimal feature combinations.
        """
        all_features = list(self.data_original.columns)
        
        if max_features is None:
            max_features = len(all_features)
        
        if param_grid is None:
            param_grid = {
                'method': ['ward', 'average', 'complete'],
                'metric': ['euclidean'],
                'n_clusters': [3, 4, 5]
            }
        
        if verbose:
            print(f"\n🔍 EXHAUSTIVE FEATURE SEARCH")
            print(f"   Testing combinations: {min_features} to {max_features} features")
            
            # Calculate total combinations
            total_combinations = sum(
                len(list(combinations(all_features, r))) 
                for r in range(min_features, max_features + 1)
            )
            total_configs = total_combinations * len(param_grid['method']) * len(param_grid['n_clusters'])
            print(f"   Total configurations to test: {total_configs}")
        
        all_results = []
        
        for n_features in range(min_features, max_features + 1):
            feature_combos = list(combinations(all_features, n_features))
            
            for feature_combo in feature_combos:
                feature_list = list(feature_combo)
                data_subset = self.data_scaled[feature_list]
                
                for method in param_grid['method']:
                    for metric in param_grid.get('metric', ['euclidean']):
                        if method == 'ward' and metric != 'euclidean':
                            continue
                        
                        for n_clusters in param_grid['n_clusters']:
                            try:
                                Z = linkage(data_subset.values, method=method, metric=metric)
                                labels = fcluster(Z, n_clusters, criterion='maxclust')
                                
                                if len(np.unique(labels)) < 2:
                                    continue
                                
                                scores = self._calculate_all_metrics(data_subset.values, labels)
                                
                                all_results.append({
                                    'features': ', '.join(feature_list),
                                    'feature_list': feature_list,
                                    'n_features': n_features,
                                    'method': method,
                                    'metric': metric,
                                    'n_clusters': n_clusters,
                                    **scores
                                })
                            except Exception as e:
                                continue
        
        results_df = pd.DataFrame(all_results)
        if len(results_df) == 0:
            return {'error': 'No valid combinations found'}
        
        results_df = results_df.sort_values('silhouette', ascending=False).reset_index(drop=True)
        
        best = results_df.iloc[0]
        
        if verbose:
            print(f"✅ Best combination: {best['feature_list']}")
            print(f"   Silhouette: {best['silhouette']:.4f}")
        
        return {
            'best_features': best['feature_list'],
            'best_params': {
                'method': best['method'],
                'metric': best['metric'],
                'n_clusters': best['n_clusters']
            },
            'best_scores': {
                'silhouette': best['silhouette']
            },
            'all_results': results_df.head(top_n)
        }
    
    def compare_methods(self, n_features_to_select='auto', method='ward', n_clusters=4,
                       metric='euclidean', criterion='silhouette', tol=0.001, 
                       min_features=3, required_features=None, verbose=True):
        """
        Compare SFS vs RFE with same parameters.
        
        Parameters:
        - min_features: Mínimo número de features a seleccionar (default 3)
        - required_features: Lista de features obligatorias que siempre se incluyen
        """
        required_features = required_features or []
        
        if verbose:
            print("\n🔬 COMPARING FEATURE SELECTION METHODS")
            print(f"   Mínimo features: {min_features}")
            print(f"   Features obligatorias: {required_features if required_features else 'Ninguna'}")
        
        sfs_result = self.sequential_forward_selection(
            n_features_to_select=n_features_to_select, method=method, n_clusters=n_clusters,
            metric=metric, criterion=criterion, tol=tol, min_features=min_features, 
            required_features=required_features, verbose=False
        )
        
        rfe_result = self.recursive_feature_elimination(
            n_features_to_select=n_features_to_select, method=method, n_clusters=n_clusters,
            metric=metric, criterion=criterion, tol=tol, min_features=min_features,
            required_features=required_features, verbose=False
        )
        
        sfs_score = sfs_result['scores']['silhouette']
        rfe_score = rfe_result['scores']['silhouette']
        
        if sfs_score > rfe_score:
            winner = 'SFS'
            best_result = sfs_result
        else:
            winner = 'RFE'
            best_result = rfe_result
        
        if verbose:
            print(f"\n🏆 Winner: {winner}")
            print(f"   SFS: {sfs_score:.4f} | RFE: {rfe_score:.4f}")
            print(f"   Best features: {best_result['selected_features']}")
        
        return {
            'winner': winner,
            'sfs': sfs_result,
            'rfe': rfe_result,
            'best': best_result
        }
    
    def sfs_rfe_grid_search(self, param_grid=None, min_features=3, required_features=None, 
                           top_n=10, verbose=True):
        """
        🚀 NUEVO: SFS + RFE con Grid Search de hiperparámetros.
        Ejecuta SFS y RFE para CADA combinación de hiperparámetros.
        
        Parameters:
        - param_grid: Diccionario con hiperparámetros a probar (usa grid financiero por defecto)
        - min_features: Mínimo número de features
        - required_features: Features obligatorias
        - top_n: Número de mejores resultados a retornar
        - verbose: Mostrar progreso
        
        Returns:
        - Diccionario con resultados de grid search
        """
        required_features = required_features or []
        
        # 🚀 USAR GRID FINANCIERO COMPLETO si no se proporciona uno
        if param_grid is None:
            from .parameter_grids import ClusteringParameterGrid
            grid_builder = ClusteringParameterGrid(min_clusters=3, max_clusters=7)
            financial_grid = grid_builder.get_financial_grid()
            
            param_grid = {
                'valid_combinations': financial_grid['valid_combinations'],
                'n_clusters': financial_grid['n_clusters']
            }
            
            if verbose:
                print("🎯 USANDO GRID FINANCIERO COMPLETO:")
                grid_builder.print_grid_info()
        
        if verbose:
            print("\n🚀 SFS + RFE + GRID SEARCH EXPANDIDO")
            print(f"   Features obligatorias: {required_features if required_features else 'Ninguna'}")
            print(f"   Mínimo features: {min_features}")
            
            if 'valid_combinations' in param_grid:
                print(f"   Combinaciones método-métrica válidas:")
                for method, metric in param_grid['valid_combinations']:
                    print(f"     • {method} + {metric}")
                print(f"   Clusters: {param_grid['n_clusters']}")
                
                # Calcular total de experimentos
                total_grid_combinations = len(param_grid['valid_combinations']) * len(param_grid['n_clusters'])
                total_experiments = total_grid_combinations * 2  # SFS + RFE
                print(f"   Total experimentos: {total_experiments} (SFS + RFE × {total_grid_combinations} grid combinations)")
            else:
                # Grid legacy format
                print(f"   Grid de hiperparámetros (formato legacy):")
                for param, values in param_grid.items():
                    print(f"     {param}: {values}")
                
                total_grid_combinations = (
                    len(param_grid.get('method', ['ward'])) * 
                    len(param_grid.get('metric', ['euclidean'])) * 
                    len(param_grid.get('n_clusters', [4]))
                )
                total_experiments = total_grid_combinations * 2  # SFS + RFE
                print(f"   Total experimentos: {total_experiments}")
        
        all_results = []
        experiment_count = 0
        
        # 🎯 PROCESAR GRID FINANSIERO COMPLETO O LEGACY
        if 'valid_combinations' in param_grid:
            # Nuevo formato: usar combinaciones válidas predefinidas
            for method, metric in param_grid['valid_combinations']:
                for n_clusters in param_grid['n_clusters']:
                    experiment_count += 1
                    
                    if verbose:
                        print(f"   🔧 Experimento {experiment_count}: {method}-{metric}-C{n_clusters}")
                    
                    # ===== EJECUTAR SFS =====
                    try:
                        sfs_result = self.sequential_forward_selection(
                            method=method, n_clusters=n_clusters, metric=metric,
                            min_features=min_features, required_features=required_features,
                            verbose=False  # Silenciar prints individuales
                        )
                        
                        # Agregar información del experimento a los resultados de SFS
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
                        
                        if verbose:
                            print(f"     SFS: {sfs_score:.4f} - Features: {sfs_features}")
                            
                    except Exception as e:
                        if verbose:
                            print(f"     SFS FAILED: {e}")
                    
                    # ===== EJECUTAR RFE =====
                    try:
                        rfe_result = self.recursive_feature_elimination(
                            method=method, n_clusters=n_clusters, metric=metric,
                            min_features=min_features, required_features=required_features,
                            verbose=False  # Silenciar prints individuales
                        )
                        
                        # Agregar información del experimento a los resultados de RFE
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
                        
                        if verbose:
                            print(f"     RFE: {rfe_score:.4f} - Features: {rfe_features}")
                            
                    except Exception as e:
                        if verbose:
                            print(f"     RFE FAILED: {e}")
                            
        else:
            # Procesamiento del formato legacy (por compatibilidad)
            for method in param_grid.get('method', ['ward']):
                for metric in param_grid.get('metric', ['euclidean']):
                    # Ward solo funciona con euclidean
                    if method == 'ward' and metric != 'euclidean':
                        continue
                    
                    for n_clusters in param_grid.get('n_clusters', [4]):
                        experiment_count += 1
                        
                        if verbose:
                            print(f"   🔧 Experimento {experiment_count}: {method}-{metric}-{n_clusters}")
                        
                        # [Resto del código legacy igual que antes...]
                        
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
        
        if verbose:
            print(f"\n🏆 MEJOR RESULTADO GLOBAL:")
            print(f"   Algoritmo: {best['algorithm']}")
            print(f"   Silhouette: {best['silhouette_score']:.4f}")
            print(f"   Features: {best['selected_features']}")
            print(f"   Hiperparámetros: method={best['hyperparams_method']}, metric={best['hyperparams_metric']}, n_clusters={best['hyperparams_n_clusters']}")
            
            print(f"\n📊 RESUMEN:")
            print(f"   Total experimentos exitosos: {len(results_df)}")
            print(f"   Mejor SFS: {sfs_results.iloc[0]['silhouette_score']:.4f} ({sfs_results.iloc[0]['algorithm']})")
            print(f"   Mejor RFE: {rfe_results.iloc[0]['silhouette_score']:.4f} ({rfe_results.iloc[0]['algorithm']})")
        
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
        except Exception as e:
            return {'silhouette': -1.0}