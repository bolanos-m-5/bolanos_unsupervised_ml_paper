"""
Main clustering analysis and visualization.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

from .estimator import HierarchicalClusteringEstimator
from .feature_selector import ClusteringFeatureSelector


class ClusteringAnalyzer:
    """
    Main clustering analysis and visualization class.
    Combines parameter optimization, feature selection, and analysis.
    """
    
    def __init__(self, data, scaler='robust', min_clusters=2, max_clusters=10, 
                 filter_outliers=False, outlier_threshold=2, exclude_columns=None,
                 iqr_multiplier=3.0):
        """
        Initialize clustering analyzer.
        
        Parameters:
        - data: DataFrame with features
        - scaler: 'robust', 'standard', or None
        - min_clusters, max_clusters: range for parameter optimization
        - filter_outliers: Si True, filtra outliers extremos antes del análisis
        - outlier_threshold: Umbral de outliers (1, 2, 3, o 4) para filtrado
        - exclude_columns: Lista de columnas a excluir del análisis de outliers
        - iqr_multiplier: Multiplicador IQR (1.5=estricto, 3=normal, 5=permisivo)
        """
        self.data_original = data
        self.scaler_type = scaler
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        
        # Configuración de filtrado de outliers
        self.filter_outliers = filter_outliers
        self.outlier_threshold = outlier_threshold
        self.exclude_columns = exclude_columns or []
        self.iqr_multiplier = iqr_multiplier
        
        # Almacenar info de outliers removidos
        self.outlier_info = None
        self.data_clean = data.copy()
        
        # Filtrar outliers si está habilitado
        if filter_outliers:
            self._filter_extreme_outliers()
        
        # Components
        self.feature_selector = ClusteringFeatureSelector(self.data_clean, scaler)
        self.data_scaled = self.feature_selector.data_scaled
        
        # Results storage
        self.optimization_results = None
        self.best_estimator = None
        self.best_features = None
        self.labels_ = None
        self.linkage_matrix_ = None
        self.feature_selection_results = None
    
    def _filter_extreme_outliers(self):
        """
        Filtrar outliers extremos usando IQR configurable.
        Método interno llamado automáticamente si filter_outliers=True.
        """
        # Variables a analizar (excluir columnas especificadas)
        variables_analizar = [col for col in self.data_clean.columns 
                            if col not in self.exclude_columns]
        
        print(f"   Variables analizadas: {len(variables_analizar)}")
        print(f"   Variables excluidas: {self.exclude_columns}")
        print(f"   Multiplicador IQR: {self.iqr_multiplier}x")
        
        # Contar outliers por registro
        outlier_count = pd.Series(0, index=self.data_clean.index)
        outlier_details = {}  # Para guardar detalles por variable
        
        for col in variables_analizar:
            Q1 = self.data_clean[col].quantile(0.25)
            Q3 = self.data_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Outliers extremos (configurable con iqr_multiplier)
            lower_extreme = Q1 - self.iqr_multiplier * IQR
            upper_extreme = Q3 + self.iqr_multiplier * IQR
            
            is_outlier = (self.data_clean[col] < lower_extreme) | (self.data_clean[col] > upper_extreme)
            outlier_count += is_outlier.astype(int)
            
            # Guardar detalles para reporte
            num_outliers = is_outlier.sum()
            if num_outliers > 0:
                outlier_details[col] = {
                    'count': num_outliers,
                    'lower_bound': lower_extreme,
                    'upper_bound': upper_extreme,
                    'min_value': self.data_clean[col].min(),
                    'max_value': self.data_clean[col].max()
                }
        
        # Filtrar registros con muchos outliers
        casos_removidos = self.data_clean[outlier_count >= self.outlier_threshold]
        self.data_clean = self.data_clean[outlier_count < self.outlier_threshold].copy()
        
        # Guardar información
        self.outlier_info = {
            'original_size': len(self.data_original),
            'filtered_size': len(self.data_clean),
            'removed_count': len(casos_removidos),
            'removed_pct': len(casos_removidos) / len(self.data_original) * 100,
            'removed_indices': casos_removidos.index.tolist(),
            'threshold': self.outlier_threshold,
            'iqr_multiplier': self.iqr_multiplier,
            'outlier_details': outlier_details
        }
        
        print(f"   Dataset original: {self.outlier_info['original_size']} registros")
        print(f"   Casos removidos: {self.outlier_info['removed_count']} ({self.outlier_info['removed_pct']:.1f}%)")
        print(f"   Dataset limpio: {self.outlier_info['filtered_size']} registros")
        print(f"   Umbral usado: {self.outlier_threshold}+ outliers")
        
        # Mostrar variables con más outliers
        if outlier_details:
            print(f"\n   📊 Variables con outliers detectados:")
            for col, details in sorted(outlier_details.items(), key=lambda x: x[1]['count'], reverse=True)[:5]:
                print(f"      {col}: {details['count']} outliers (rango válido: [{details['lower_bound']:.2f}, {details['upper_bound']:.2f}])")
        
        print("=" * 60)
    
    def get_outlier_report(self):
        """
        Obtener reporte detallado de outliers filtrados.
        
        Returns:
        - Dict con información de outliers o None si no se filtró
        """
        return self.outlier_info
    
    def optimize_parameters(self, min_clusters=None, max_clusters=None, verbose=True):
        """
        Optimize clustering parameters using predefined financial grid.
        
        Parameters:
        - min_clusters: minimum number of clusters (uses self.min_clusters if None)
        - max_clusters: maximum number of clusters (uses self.max_clusters if None)
        - verbose: print progress and results
        """
        # Use provided cluster range or defaults
        min_clusters = min_clusters or self.min_clusters
        max_clusters = max_clusters or self.max_clusters
        
        # Predefined financial grid - only valid method-metric combinations
        valid_combinations = [
            # Ward only works with euclidean
            ('ward', 'euclidean'),
            # Complete works with all distances  
            ('complete', 'euclidean'),
            ('complete', 'cosine'),
            ('complete', 'correlation'),
            # Average works with all distances
            ('average', 'euclidean'), 
            ('average', 'cosine'),
            ('average', 'correlation')
        ]
        
        cluster_range = list(range(min_clusters, max_clusters + 1))
        
        results = []
        
        for method, metric in valid_combinations:
            for n_clusters in cluster_range:
                try:
                    # Calculate linkage matrix
                    Z = linkage(self.data_scaled.values, method=method, metric=metric)
                    labels = fcluster(Z, n_clusters, criterion='maxclust')
                    
                    if len(np.unique(labels)) < 2:
                        continue
                    
                    # Calculate Silhouette Score
                    sil = silhouette_score(self.data_scaled.values, labels)
                    
                    results.append({
                        'method': method,
                        'metric': metric,
                        'n_clusters': n_clusters,
                        'silhouette_score': sil
                    })
                    
                except Exception as e:
                    continue
        
        if not results:
            raise ValueError("No valid parameter combinations found")
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('silhouette_score', ascending=False).reset_index(drop=True)
        
        best = results_df.iloc[0]
        
        # Store results
        self.optimization_results = {
            'best_params': {
                'method': best['method'],
                'metric': best['metric'],
                'n_clusters': best['n_clusters']
            },
            'best_scores': {
                'silhouette': round(best['silhouette_score'], 4)
            },
            'all_results': results_df
        }
        
        return self.optimization_results
    
    def select_features_auto(self, method='ward', metric='euclidean', n_clusters=4, 
                           selection_method='compare', min_features=3, required_features=None, verbose=True):
        """
        Automatic feature selection using best method.
        
        Parameters:
        - method, metric, n_clusters: clustering parameters
        - selection_method: 'sfs', 'rfe', 'compare', 'exhaustive', 'exhaustive_hyperopt', or 'sfs_rfe_grid'
        - min_features: Mínimo número de features a seleccionar (default 3)
        - required_features: Lista de features obligatorias (ej: ['SCORE'])
        
        Returns:
        - Dictionary with feature selection results
        """
        required_features = required_features or []
        
        if verbose:
            print(f"\n🎯 AUTOMATIC FEATURE SELECTION")
            print(f"   Mínimo features requeridas: {min_features}")
            if required_features:
                print(f"   ⭐ Features obligatorias: {required_features}")
        
        feature_results = {
            'method_used': selection_method,
            'clustering_params': {'method': method, 'metric': metric, 'n_clusters': n_clusters},
            'required_features': required_features
        }
        
        if selection_method == 'sfs':
            result = self.feature_selector.sequential_forward_selection(
                method=method, metric=metric, n_clusters=n_clusters, 
                min_features=min_features, required_features=required_features, verbose=verbose
            )
            self.best_features = result['selected_features']
            feature_results.update({
                'best_features': self.best_features,
                'best_method': 'sfs',
                'comparison_scores': {'sfs': result.get('final_scores', {})},
                'sfs_results': result
            })
            
        elif selection_method == 'rfe':
            result = self.feature_selector.recursive_feature_elimination(
                method=method, metric=metric, n_clusters=n_clusters, 
                min_features=min_features, required_features=required_features, verbose=verbose
            )
            self.best_features = result['selected_features']
            feature_results.update({
                'best_features': self.best_features,
                'best_method': 'rfe',
                'comparison_scores': {'rfe': result.get('final_scores', {})},
                'rfe_results': result
            })
            
        elif selection_method == 'compare':
            result = self.feature_selector.compare_methods(
                method=method, metric=metric, n_clusters=n_clusters, 
                min_features=min_features, required_features=required_features, verbose=verbose
            )
            self.best_features = result['best']['selected_features']
            feature_results.update({
                'best_features': self.best_features,
                'best_method': result['winner'],  # Usar 'winner' en lugar de 'best_method'
                'comparison_scores': {
                    'sfs_score': result['sfs']['scores']['silhouette'],
                    'rfe_score': result['rfe']['scores']['silhouette']
                },
                'comparison_results': result
            })
            
        elif selection_method == 'exhaustive':
            result = self.feature_selector.exhaustive_search(
                param_grid={'method': [method], 'metric': [metric], 'n_clusters': [n_clusters]},
                verbose=verbose
            )
            if 'error' not in result:
                self.best_features = result['best_features']
                feature_results.update({
                    'best_features': self.best_features,
                    'best_method': 'exhaustive',
                    'comparison_scores': {'exhaustive': result.get('best_scores', {})},
                    'exhaustive_results': result
                })
            else:
                self.best_features = list(self.data_original.columns)
                feature_results.update({
                    'best_features': self.best_features,
                    'best_method': 'fallback',
                    'comparison_scores': {},
                    'error': result['error']
                })
        
        elif selection_method == 'exhaustive_hyperopt':
            # ⭐ NUEVO: Búsqueda exhaustiva con optimización de hiperparámetros
            param_grid = {
                'method': ['ward', 'average', 'complete'],
                'metric': ['euclidean'],  # Ward requiere euclidean
                'n_clusters': list(range(min(n_clusters-1, 3), max(n_clusters+2, 8)))  # Rango dinámico
            }
            result = self.feature_selector.exhaustive_search(
                param_grid=param_grid,
                min_features=min_features,
                max_features=min(len(self.data_original.columns), min_features + 3),  # Controlar combinaciones
                top_n=20,  # Top 20 para no saturar memoria
                verbose=verbose
            )
            if 'error' not in result:
                self.best_features = result['best_features']
                best_params = result.get('best_params', {})
                feature_results.update({
                    'best_features': self.best_features,
                    'best_method': 'exhaustive_hyperopt',
                    'comparison_scores': {'exhaustive_hyperopt': result.get('best_scores', {})},
                    'exhaustive_results': result,
                    'total_combinations_tested': len(result.get('all_results', [])),
                    'best_hyperparams': {
                        'method': best_params.get('method', method),
                        'metric': best_params.get('metric', metric), 
                        'n_clusters': best_params.get('n_clusters', n_clusters)
                    }
                })
            else:
                self.best_features = list(self.data_original.columns)
                feature_results.update({
                    'best_features': self.best_features,
                    'best_method': 'fallback',
                    'comparison_scores': {},
                    'error': result['error']
                })
        
        elif selection_method == 'sfs_rfe_grid':
            # 🚀 NUEVO: SFS + RFE con Grid Search de hiperparámetros EXPANDIDO
            from .parameter_grids import ClusteringParameterGrid
            
            # Crear grid builder con rango de clusters optimizado
            grid_builder = ClusteringParameterGrid(
                min_clusters=max(2, self.min_clusters), 
                max_clusters=min(self.max_clusters, 8)  # Limitar para evitar cálculo excesivo
            )
            financial_grid = grid_builder.get_financial_grid()
            
            param_grid = {
                'valid_combinations': financial_grid['valid_combinations'],
                'n_clusters': financial_grid['n_clusters']
            }
            
            if verbose:
                print(f"   🎯 SFS + RFE GRID SEARCH EXPANDIDO activado")
                print(f"   Usando grid financiero con {len(financial_grid['valid_combinations'])} combinaciones método-métrica")
                print(f"   Total experimentos: {financial_grid['total_combinations'] * 2} (SFS + RFE)")
            
            result = self.feature_selector.sfs_rfe_grid_search(
                param_grid=param_grid,
                min_features=min_features,
                required_features=required_features,
                top_n=25,  # Incrementar top_n para grid más grande
                verbose=verbose
            )
            
            if 'error' not in result:
                best_overall = result['best_overall']
                self.best_features = best_overall['selected_features']
                
                feature_results.update({
                    'best_features': self.best_features,
                    'best_method': 'sfs_rfe_grid_expanded',
                    'comparison_scores': {
                        'sfs_rfe_grid': best_overall['scores'],
                        'best_sfs': {'silhouette': result['method_comparison']['best_sfs']['silhouette']},
                        'best_rfe': {'silhouette': result['method_comparison']['best_rfe']['silhouette']}
                    },
                    'sfs_rfe_grid_results': result,
                    'winner_algorithm': best_overall['algorithm'],
                    'best_hyperparams': best_overall['hyperparams'],
                    'total_experiments': result['grid_search_results']['total_experiments'],
                    'grid_used': 'financial_expanded'
                })
            else:
                self.best_features = list(self.data_original.columns)
                feature_results.update({
                    'best_features': self.best_features,
                    'best_method': 'fallback',
                    'comparison_scores': {},
                    'error': result['error']
                })
        
        else:
            # Método no reconocido, usar todas las features
            self.best_features = list(self.data_original.columns)
            feature_results.update({
                'best_features': self.best_features,
                'best_method': 'none',
                'comparison_scores': {},
                'error': f"Unknown selection method: {selection_method}"
            })
        
        # Store feature results for later use
        self.feature_selection_results = feature_results
        
        return feature_results
    
    def fit_predict(self, method=None, metric=None, n_clusters=None, features=None, verbose=True):
        """
        Execute final clustering with optimized parameters.
        
        Parameters:
        - method, metric, n_clusters: clustering parameters (uses optimized if None)
        - features: features to use (uses selected features if None)
        """
        # ✅ PRIORIDAD 1: Si ya tenemos best_estimator configurado, usarlo
        if hasattr(self, 'best_estimator') and self.best_estimator is not None and method is None:
            method = self.best_estimator.method
            metric = self.best_estimator.metric 
            n_clusters = self.best_estimator.n_clusters
            if verbose:
                print(f"\n🎯 USING PRE-CONFIGURED BEST ESTIMATOR")
                print(f"   Method: {method}, Metric: {metric}, Clusters: {n_clusters}")
        
        # PRIORIDAD 2: Usar optimization_results si existen
        elif self.optimization_results and method is None:
            method = self.optimization_results['best_params']['method']
            metric = self.optimization_results['best_params']['metric']
            n_clusters = self.optimization_results['best_params']['n_clusters']
        
        # PRIORIDAD 3: Defaults como fallback
        else:
            method = method or 'ward'
            metric = metric or 'euclidean'
            n_clusters = n_clusters or 4
        
        # Use selected features if not provided
        if features is None:
            features = self.best_features or list(self.data_scaled.columns)
        
        # Select data
        data_to_cluster = self.data_scaled[features]
        
        if verbose:
            print(f"\n🚀 EXECUTING FINAL CLUSTERING")
            print(f"   Method: {method}, Metric: {metric}, Clusters: {n_clusters}")
            print(f"   Features: {features}")
        
        # Create and fit estimator (actualizar self.best_estimator)
        self.best_estimator = HierarchicalClusteringEstimator(
            method=method, n_clusters=n_clusters, metric=metric
        )
        
        self.labels_ = self.best_estimator.fit_predict(data_to_cluster)
        self.linkage_matrix_ = self.best_estimator.linkage_matrix_
        
        # Calculate final metrics
        final_scores = {
            'silhouette': round(silhouette_score(data_to_cluster, self.labels_), 4)
        }
        
        if verbose:
            print(f"\n📈 FINAL CLUSTERING SCORES")
            print(f"   Silhouette Score: {final_scores['silhouette']:.4f}")
            print(f"   Cluster distribution: {np.bincount(self.labels_)}")
        
        return {
            'labels': self.labels_,
            'scores': final_scores,
            'features_used': features,
            'parameters': {'method': method, 'metric': metric, 'n_clusters': n_clusters}
        }
    
    def get_cluster_centroids(self, features=None, include_counts=True, round_decimals=4):
        """
        Calculate cluster centroids (means) for each variable of the optimal clustering.
        
        Parameters:
        - features: list of features to include (uses best_features if None)
        - include_counts: whether to include cluster size counts
        - round_decimals: number of decimals to round results
        
        Returns:
        - DataFrame with centroids (means) per cluster and feature
        """
        if self.labels_ is None:
            raise ValueError("No clustering performed yet. Run fit_predict() first.")
        
        # Determine features to use
        if features is None:
            if self.best_features is not None:
                features = self.best_features
            else:
                # Use all available features except potentially problematic ones
                features = [col for col in self.data_clean.columns 
                           if col not in getattr(self, 'exclude_columns', [])]
        
        # Get the data with optimal features - USE CLEAN DATA that matches labels_
        data_for_centroids = self.data_clean[features].copy()
        data_for_centroids['cluster'] = self.labels_
        
        # Calculate centroids (means) by cluster
        centroids = data_for_centroids.groupby('cluster')[features].mean()
        
        # Add cluster counts if requested
        if include_counts:
            cluster_counts = data_for_centroids.groupby('cluster').size()
            centroids['n_customers'] = cluster_counts
            
            # Reorder columns to put n_customers first
            cols = ['n_customers'] + [col for col in centroids.columns if col != 'n_customers']
            centroids = centroids[cols]
        
        # Round results
        if round_decimals is not None:
            numeric_cols = centroids.select_dtypes(include=[np.number]).columns
            centroids[numeric_cols] = centroids[numeric_cols].round(round_decimals)
        
        return centroids
    
    def get_cluster_summary_stats(self, features=None):
        """
        Get comprehensive summary statistics for each cluster.
        
        Parameters:
        - features: list of features to analyze (uses best_features if None)
        
        Returns:
        - Dictionary with centroids, percentages, and cluster info
        """
        if self.labels_ is None:
            raise ValueError("No clustering performed yet. Run fit_predict() first.")
        
        # Get centroids
        centroids = self.get_cluster_centroids(features=features, include_counts=True)
        
        # Calculate additional statistics
        total_samples = len(self.labels_)
        cluster_info = []
        
        for cluster_id in sorted(centroids.index):
            n_customers = int(centroids.loc[cluster_id, 'n_customers'])
            percentage = (n_customers / total_samples) * 100
            
            cluster_info.append({
                'cluster_id': cluster_id,
                'n_customers': n_customers,
                'percentage': round(percentage, 2),
                'size_category': 'Large' if percentage > 25 else 'Medium' if percentage > 15 else 'Small'
            })
        
        return {
            'centroids': centroids,
            'cluster_info': cluster_info,
            'total_samples': total_samples,
            'n_clusters': len(centroids),
            'features_analyzed': centroids.columns.tolist()
        }
    
    def plot_dendrogram(self, figsize=(12, 8), title=None, max_d=None):
        """
        Plot hierarchical clustering dendrogram coloreado con clusters óptimos.
        
        Parameters:
        - figsize: Tamaño de la figura
        - title: Título del gráfico
        - max_d: Threshold manual para colores (si None, usa número de clusters óptimo)
        """
        if self.linkage_matrix_ is None:
            raise ValueError("No clustering performed yet. Run fit_predict() first.")
        
        plt.figure(figsize=figsize)
        
        # Determinar el número de clusters de múltiples fuentes posibles
        n_clusters = None
        
        # Opción 1: Usar max_d si se proporciona
        if max_d is not None:
            color_threshold = max_d
        else:
            # Opción 2: Usar self.best_estimator.n_clusters si está disponible
            if hasattr(self, 'best_estimator') and self.best_estimator is not None and hasattr(self.best_estimator, 'n_clusters'):
                n_clusters = self.best_estimator.n_clusters
            # Opción 3: Usar self.labels_ para contar clusters únicos
            elif hasattr(self, 'labels_') and self.labels_ is not None:
                n_clusters = len(np.unique(self.labels_))
            # Opción 4: Buscar en optimization_results
            elif hasattr(self, 'optimization_results') and self.optimization_results is not None:
                n_clusters = self.optimization_results['best_params'].get('n_clusters', None)
            
            # Calcular threshold basado en el número de clusters encontrado
            if n_clusters is not None and n_clusters > 1:
                # Obtener las distancias del linkage y calcular threshold
                distances = self.linkage_matrix_[:, 2]
                sorted_distances = np.sort(distances)
                # Threshold está entre los últimos n_clusters-1 merges
                threshold_idx = len(sorted_distances) - n_clusters + 1
                if threshold_idx >= 0 and threshold_idx < len(sorted_distances):
                    color_threshold = sorted_distances[threshold_idx]
                else:
                    color_threshold = np.max(self.linkage_matrix_[:, 2]) * 0.7
            else:
                # Valor por defecto que muestra algunos colores
                color_threshold = np.max(self.linkage_matrix_[:, 2]) * 0.7
        
        # Crear dendrograma con colores y sin etiquetas de muestra
        dendro = dendrogram(self.linkage_matrix_, 
                           color_threshold=color_threshold,   # ✅ Colorear clusters
                           above_threshold_color='gray',      # Color para clusters grandes
                           no_labels=True,                    # ✅ Sin etiquetas de muestra
                           leaf_rotation=0,                   # Sin rotación (no hay etiquetas)
                           count_sort=True)                   # Ordenar por tamaño
        
        # Mejorar título y etiquetas
        n_clusters_shown = len(set(dendro['color_list'])) - 1  # -1 para excluir gray
        
        # Debug information
        source_info = ""
        if n_clusters is not None:
            source_info = f" | Target: {n_clusters} clusters"
        
        plt.title(title or f'Dendrograma de Clustering Jerárquico\n({n_clusters_shown} clusters coloreados{source_info})')
        plt.xlabel('Muestras agrupadas por similitud')
        plt.ylabel('Distancia euclidiana')
        
        # Agregar línea horizontal en el threshold si está definido
        if color_threshold:
            cluster_label = n_clusters if n_clusters is not None else "Auto"
            plt.axhline(y=color_threshold, color='red', linestyle='--', alpha=0.7, 
                       label=f'Threshold (clusters={cluster_label})')
            plt.legend()
        
        # Print debug info
        print(f"🎨 DENDROGRAMA INFO:")
        print(f"   • Clusters objetivo: {n_clusters}")
        print(f"   • Clusters mostrados: {n_clusters_shown}")
        print(f"   • Color threshold: {color_threshold:.4f}")
        
        plt.tight_layout()
        plt.show()
        
        return dendro
    
    def plot_cluster_profiles(self, figsize=(15, 8), plot_type='bar'):
        """
        Plot cluster profiles showing centroids.
        
        Parameters:
        - plot_type: 'bar' or 'radar'
        """
        if self.labels_ is None:
            raise ValueError("No clustering performed yet. Run fit_predict() first.")
        
        # Get features used
        features = self.best_features or list(self.data_scaled.columns)
        data_plot = self.data_scaled[features]
        
        # Calculate centroids
        centroids = []
        for cluster in np.unique(self.labels_):
            centroid = data_plot[self.labels_ == cluster].mean()
            centroids.append(centroid)
        
        centroids_df = pd.DataFrame(centroids, index=[f'Cluster {i}' for i in np.unique(self.labels_)])
        
        if plot_type == 'bar':
            fig, axes = plt.subplots(1, len(np.unique(self.labels_)), figsize=figsize, sharey=True)
            
            if len(np.unique(self.labels_)) == 1:
                axes = [axes]
            
            for idx, (cluster, ax) in enumerate(zip(np.unique(self.labels_), axes)):
                centroid = centroids_df.iloc[idx]
                bars = ax.bar(range(len(centroid)), centroid.values)
                ax.set_title(f'Cluster {cluster}\n(n={sum(self.labels_ == cluster)})')
                ax.set_xticks(range(len(centroid)))
                ax.set_xticklabels(centroid.index, rotation=45)
                ax.grid(True, alpha=0.3)
                
                # Color bars by value
                for bar, val in zip(bars, centroid.values):
                    if val > 0:
                        bar.set_color('skyblue')
                    else:
                        bar.set_color('lightcoral')
            
            plt.tight_layout()
            plt.show()
            
        elif plot_type == 'radar':
            angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle
            
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(centroids_df)))
            
            for idx, (cluster_name, centroid) in enumerate(centroids_df.iterrows()):
                values = centroid.tolist()
                values += values[:1]  # Complete the circle
                
                ax.plot(angles, values, 'o-', linewidth=2, label=cluster_name, color=colors[idx])
                ax.fill(angles, values, alpha=0.25, color=colors[idx])
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(features)
            ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
            plt.title('Cluster Profiles - Radar Chart')
            plt.tight_layout()
            plt.show()
        
        return centroids_df
    
    def assign_clusters_to_original_data(self, original_df):
        """
        Assign cluster labels to original dataframe.
        
        Parameters:
        - original_df: original dataframe to add cluster labels
        
        Returns:
        - DataFrame with cluster column added
        """
        if self.labels_ is None:
            raise ValueError("No clustering performed yet. Run fit_predict() first.")
        
        if len(original_df) != len(self.labels_):
            raise ValueError("Length mismatch between original data and cluster labels")
        
        result_df = original_df.copy()
        result_df['cluster'] = self.labels_
        
        return result_df
    
    def get_cluster_summary(self):
        """Get summary statistics for each cluster."""
        if self.labels_ is None:
            raise ValueError("No clustering performed yet. Run fit_predict() first.")
        
        features = self.best_features or list(self.data_scaled.columns)
        data_summary = self.data_scaled[features]
        
        summary_stats = []
        
        for cluster in np.unique(self.labels_):
            cluster_data = data_summary[self.labels_ == cluster]
            
            stats = {
                'cluster': cluster,
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(data_summary) * 100
            }
            
            # Add feature statistics
            for feature in features:
                stats[f'{feature}_mean'] = cluster_data[feature].mean()
                stats[f'{feature}_std'] = cluster_data[feature].std()
            
            summary_stats.append(stats)
        
        return pd.DataFrame(summary_stats)
    
    def full_analysis(self, min_clusters=None, max_clusters=None, select_features=True, 
                     selection_method='compare', min_features=3, required_features=None,
                     plot_results=True, verbose=True):
        """
        Run complete financial clustering analysis pipeline.
        
        Parameters:
        - min_clusters, max_clusters: cluster range to test (uses class defaults if None)
        - select_features: whether to perform feature selection
        - selection_method: feature selection method ('sfs', 'rfe', 'compare', 'exhaustive', 'exhaustive_hyperopt', 'sfs_rfe_grid')
        - min_features: mínimo número de features a seleccionar (default 3)
        - required_features: lista de features obligatorias que siempre se incluyen (ej: ['SCORE'])
        - plot_results: whether to generate plots
        - verbose: whether to print detailed output
        
        Returns:
        - Dictionary with all results
        """
        results = {}
        
        # Step 1: Parameter optimization with financial grid
        results['optimization'] = self.optimize_parameters(
            min_clusters=min_clusters,
            max_clusters=max_clusters, 
            verbose=verbose
        )
        
        # Step 2: Feature selection
        if select_features:
            best_params = self.optimization_results['best_params']
            results['feature_selection'] = self.select_features_auto(
                selection_method=selection_method, min_features=min_features,
                required_features=required_features, verbose=verbose, **best_params
            )
            
            # ⭐ ACTUALIZAR: Si cualquier método devolvió hiperparámetros óptimos, usarlos para clustering final
            if 'best_hyperparams' in results['feature_selection']:
                optimized_params = results['feature_selection']['best_hyperparams']
                if verbose:
                    print(f"\n🚀 USING OPTIMIZED PARAMETERS: {optimized_params}")
                
                # Actualizar self con los parámetros optimizados para fit_predict Y plots
                self.best_estimator = HierarchicalClusteringEstimator(
                    method=optimized_params['method'],
                    metric=optimized_params['metric'],
                    n_clusters=optimized_params['n_clusters']
                )
                
                # ✅ IMPORTANTE: Actualizar también min/max_clusters para plots
                self.min_clusters = optimized_params['n_clusters']
                self.max_clusters = optimized_params['n_clusters']
        
        # Step 3: Final clustering
        results['clustering'] = self.fit_predict(verbose=verbose)
        
        # Step 4: Summary
        results['summary'] = self.get_cluster_summary()
        
        # Step 5: Plots
        if plot_results:
            if verbose:
                print("\n📊 GENERATING VISUALIZATIONS")
            
            self.plot_dendrogram()
            results['centroids'] = self.plot_cluster_profiles(plot_type='bar')
            self.plot_cluster_profiles(plot_type='radar')
        
        if verbose:
            print("\n🎉 FINANCIAL ANALYSIS COMPLETED!")
        
        return results
    
    def asignar_clusters_a_datos(self, df_original):
        """
        Assign clusters to original data (backward compatibility method).
        
        Parameters:
        - df_original: DataFrame with original data
        
        Returns:
        - DataFrame with added 'cluster' column
        """
        if self.labels_ is None:
            raise ValueError("No clustering performed yet. Run fit_predict() first.")
        
        df_copy = df_original.copy()
        df_copy['cluster'] = self.labels_
        return df_copy
    
    def get_cluster_summary(self):
        """Get summary statistics for each cluster."""
        if self.labels_ is None:
            return None
        
        features = self.best_features or list(self.data_scaled.columns)
        data_for_summary = self.data_scaled[features]
        
        summary = {}
        for cluster in np.unique(self.labels_):
            cluster_mask = self.labels_ == cluster
            cluster_data = data_for_summary[cluster_mask]
            
            summary[f'cluster_{cluster}'] = {
                'size': sum(cluster_mask),
                'percentage': sum(cluster_mask) / len(self.labels_) * 100,
                'centroid': cluster_data.mean().to_dict(),
                'std': cluster_data.std().to_dict()
            }
        
        return summary
    
    def analyze_grid_expanded_results(self, results, display_top_n=15, verbose=True):
        """
        Generate comprehensive cluster analysis report with statistics and interpretation.
        
        Parameters:
        - df_with_clusters: DataFrame with cluster assignments (optional)
        - selected_features: List of features to focus on (optional, uses best_features if available)
        - verbose: Print detailed analysis
        
        Returns:
        - Dictionary with comprehensive cluster analysis
        """
        if self.labels_ is None:
            raise ValueError("No clustering performed yet. Run fit_predict() first.")
        
        # Use provided dataframe or create one
        if df_with_clusters is None:
            df_with_clusters = self.data_scaled.copy()
            df_with_clusters['cluster'] = self.labels_
        
        # Determine features to analyze
        if selected_features is None:
            selected_features = self.best_features or list(self.data_scaled.columns)
        
        # Calculate statistics
        cluster_means = df_with_clusters.groupby('cluster')[selected_features].mean()
        cluster_stds = df_with_clusters.groupby('cluster')[selected_features].std()
        cluster_counts = df_with_clusters['cluster'].value_counts().sort_index()
        
        # Create detailed report
        report = {
            'cluster_statistics': {
                'means': cluster_means,
                'stds': cluster_stds,
                'counts': cluster_counts,
                'percentages': (cluster_counts / len(df_with_clusters) * 100).round(2)
            },
            'selected_features': selected_features,
            'total_samples': len(df_with_clusters),
            'n_clusters': len(cluster_counts)
        }
        
        # Create interpretation table
        interpretation_table = pd.DataFrame()
        interpretation_table['num_clientes'] = cluster_counts
        interpretation_table['porcentaje_%'] = report['cluster_statistics']['percentages']
        
        # Add means and stds for selected features
        for feature in selected_features:
            interpretation_table[f'{feature}_media'] = cluster_means[feature]
            interpretation_table[f'{feature}_std'] = cluster_stds[feature]
        
        report['interpretation_table'] = interpretation_table
        
        if verbose:
            self._print_cluster_report(report, selected_features)
        
        return report
    
    def _print_cluster_report(self, report, selected_features):
        """Print formatted cluster report."""
        stats = report['cluster_statistics']
        
        print(f"\n📋 Clusters: {report['n_clusters']} | Samples: {report['total_samples']} | Features: {selected_features}")
        
        for cluster_id in stats['means'].index:
            n_clientes = stats['counts'][cluster_id]
            pct = stats['percentages'][cluster_id]
            
            print(f"\n   Cluster {cluster_id} ({n_clientes} clientes, {pct:.1f}%):")
            
            for feature in selected_features:
                media = stats['means'].loc[cluster_id, feature]
                std = stats['stds'].loc[cluster_id, feature]
                
                # Add interpretative context
                if feature == 'SCORE':
                    if media > 0.5:
                        nivel = "ALTO"
                    elif media < -0.5:
                        nivel = "BAJO"
                    else:
                        nivel = "MEDIO"
                    print(f"      • {feature}: Media={media:.4f}, STD={std:.4f} ({nivel} rendimiento)")
                elif 'rate' in feature.lower():
                    if media > 0.1:
                        tendencia = "POSITIVA"
                    elif media < -0.1:
                        tendencia = "NEGATIVA"
                    else:
                        tendencia = "NEUTRAL"
                    print(f"      • {feature}: Media={media:.4f}, STD={std:.4f} (tendencia {tendencia})")
                else:
                    print(f"      • {feature}: Media={media:.4f}, STD={std:.4f}")
            
            print("-" * 120)
        
        return None
    
    def analyze_grid_expanded_results(self, results, display_top_n=15, verbose=True):
        """
        Analiza automáticamente los resultados del grid expandido con múltiples métricas.
        
        Parameters:
        - results: Resultados del full_analysis con selection_method='sfs_rfe_grid'
        - display_top_n: Número de mejores configuraciones a mostrar
        - verbose: Si True, muestra análisis detallado
        
        Returns:
        - dict: Resumen del análisis con mejores resultados y insights
        """
        import pandas as pd
        
        feature_results = results.get('feature_selection', {})
        analysis_summary = {
            'best_method': feature_results.get('best_method', 'N/A'),
            'success': False,
            'insights': {},
            'comparison_data': None,
            'top_results': None,
            'metric_analysis': {},
            'method_analysis': {}
        }
        
        if feature_results.get('best_method') == 'sfs_rfe_grid_expanded':
            analysis_summary['success'] = True
            
            if verbose:
                print("🏆 RESULTADOS GRID EXPANDIDO CON MÚLTIPLES MÉTRICAS:")
                print("=" * 70)
            
            # Extraer información clave
            winner_algorithm = feature_results.get('winner_algorithm', 'N/A')
            best_features = feature_results.get('best_features', [])
            best_hyperparams = feature_results.get('best_hyperparams', {})
            total_experiments = feature_results.get('total_experiments', 0)
            grid_used = feature_results.get('grid_used', 'N/A')
            
            analysis_summary['insights'] = {
                'winner_algorithm': winner_algorithm,
                'best_features': best_features,
                'best_hyperparams': best_hyperparams,
                'total_experiments': total_experiments,
                'grid_used': grid_used
            }
            
            if verbose:
                print(f"📈 MEJOR RESULTADO CON GRID EXPANDIDO:")
                print(f"   Algoritmo ganador: {winner_algorithm}")
                print(f"   Features seleccionadas: {best_features}")
                print(f"   Hiperparámetros óptimos: {best_hyperparams}")
                print(f"   Grid utilizado: {grid_used}")
                print(f"   Total experimentos: {total_experiments}")
            
            # Scores de comparación
            comparison_scores = feature_results.get('comparison_scores', {})
            global_score = comparison_scores.get('sfs_rfe_grid', {}).get('silhouette', 0)
            best_sfs_score = comparison_scores.get('best_sfs', {}).get('silhouette', 0)
            best_rfe_score = comparison_scores.get('best_rfe', {}).get('silhouette', 0)
            
            analysis_summary['insights']['scores'] = {
                'global_score': global_score,
                'best_sfs_score': best_sfs_score,
                'best_rfe_score': best_rfe_score
            }
            
            if verbose:
                print(f"\n📊 COMPARACIÓN DE SCORES CON MÚLTIPLES MÉTRICAS:")
                print(f"   🥇 Mejor global (grid expandido): {global_score:.4f}")
                print(f"   🔵 Mejor SFS global: {best_sfs_score:.4f}")
                print(f"   🔴 Mejor RFE global: {best_rfe_score:.4f}")
            
            # Obtener resultados detallados
            sfs_rfe_results = feature_results.get('sfs_rfe_grid_results', {})
            grid_results = sfs_rfe_results.get('grid_search_results', {})
            
            if 'all_results_df' in grid_results:
                top_n_results = grid_results['all_results_df'].head(display_top_n)
                analysis_summary['top_results'] = top_n_results
                
                if verbose:
                    print(f"\n🏆 TOP {display_top_n} CONFIGURACIONES (Múltiples Métricas):")
                    display(top_n_results[['algorithm', 'silhouette_score', 'hyperparams_method', 'hyperparams_metric', 'hyperparams_n_clusters', 'selected_features']].round(4))
                
                # Analizar diversidad de métricas
                top_metrics = top_n_results['hyperparams_metric'].value_counts()
                analysis_summary['metric_analysis'] = top_metrics.to_dict()
                
                if verbose:
                    print(f"\n🔍 ANÁLISIS DE MÉTRICAS EN TOP {display_top_n}:")
                    for metric, count in top_metrics.items():
                        pct = (count / len(top_n_results)) * 100
                        print(f"   • {metric}: {count} configuraciones ({pct:.1f}%)")
                
                # Analizar métodos más exitosos
                top_methods = top_n_results['hyperparams_method'].value_counts()
                analysis_summary['method_analysis'] = top_methods.to_dict()
                
                if verbose:
                    print(f"\n🔍 ANÁLISIS DE MÉTODOS EN TOP {display_top_n}:")
                    for method, count in top_methods.items():
                        pct = (count / len(top_n_results)) * 100
                        print(f"   • {method}: {count} configuraciones ({pct:.1f}%)")
            
            # Comparación detallada SFS vs RFE
            method_comparison = sfs_rfe_results.get('method_comparison', {})
            
            if 'best_sfs' in method_comparison and 'best_rfe' in method_comparison:
                best_sfs = method_comparison['best_sfs']
                best_rfe = method_comparison['best_rfe']
                
                comparison_data = pd.DataFrame({
                    'Método': ['SFS (Mejor con Grid)', 'RFE (Mejor con Grid)'],
                    'Silhouette_Score': [best_sfs['silhouette'], best_rfe['silhouette']],
                    'Features': [str(best_sfs['features']), str(best_rfe['features'])],
                    'Clustering_Method': [best_sfs['hyperparams']['method'], best_rfe['hyperparams']['method']],
                    'Métrica_Distancia': [best_sfs['hyperparams']['metric'], best_rfe['hyperparams']['metric']],
                    'N_Clusters': [best_sfs['hyperparams']['n_clusters'], best_rfe['hyperparams']['n_clusters']]
                })
                
                analysis_summary['comparison_data'] = comparison_data
                analysis_summary['insights']['method_comparison'] = {
                    'best_sfs': best_sfs,
                    'best_rfe': best_rfe
                }
                
                if verbose:
                    print(f"\n🔵 COMPARACIÓN DETALLADA SFS vs RFE:")
                    print("📋 TABLA COMPARATIVA CON MÚLTIPLES MÉTRICAS:")
                    display(comparison_data)
        
        else:
            if verbose:
                print("❌ Error en el grid expandido")
                print(f"Método detectado: {feature_results.get('best_method', 'N/A')}")
                if 'error' in feature_results:
                    print(f"Error: {feature_results['error']}")
            
            analysis_summary['error'] = feature_results.get('error', 'Método no es sfs_rfe_grid_expanded')
        
        return analysis_summary
