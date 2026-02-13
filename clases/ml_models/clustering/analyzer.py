"""
Clustering analysis and visualization - CLEANED VERSION.
Contains only essential functions used in production pipeline.
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
    
    def __init__(self, data, min_clusters=2, max_clusters=10, 
                 filter_outliers=False, outlier_threshold=2, exclude_columns=None,
                 iqr_multiplier=3.0):
        """
        Initialize clustering analyzer with RobustScaler.
        
        Parameters:
        - data: DataFrame with features
        - min_clusters, max_clusters: range for parameter optimization
        - filter_outliers: Si True, filtra outliers extremos antes del análisis
        - outlier_threshold: Umbral de outliers (1, 2, 3, o 4) para filtrado
        - exclude_columns: Lista de columnas a excluir del análisis de outliers
        - iqr_multiplier: Multiplicador IQR (1.5=estricto, 3=normal, 5=permisivo)
        """
        self.data_original = data
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
        self.feature_selector = ClusteringFeatureSelector(self.data_clean)
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
        
        # Contar outliers por registro
        outlier_count = pd.Series(0, index=self.data_clean.index)
        outlier_details = {}
        
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
                    'upper_bound': upper_extreme
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
            'threshold': self.outlier_threshold,
            'iqr_multiplier': self.iqr_multiplier,
            'outlier_details': outlier_details
        }
    
    def get_outlier_report(self):
        """
        Obtener reporte detallado de outliers filtrados.
        
        Returns:
        - Dict con información de outliers o None si no se filtró
        """
        return self.outlier_info
    
    def optimize_parameters(self, min_clusters=None, max_clusters=None, verbose=False):
        """
        Optimize clustering parameters using predefined financial grid.
        """
        min_clusters = min_clusters or self.min_clusters
        max_clusters = max_clusters or self.max_clusters
        
        # Predefined financial grid - only valid method-metric combinations
        valid_combinations = [
            ('ward', 'euclidean'),
            ('complete', 'euclidean'),
            ('complete', 'cosine'),
            ('complete', 'correlation'),
            ('average', 'euclidean'), 
            ('average', 'cosine'),
            ('average', 'correlation')
        ]
        
        cluster_range = list(range(min_clusters, max_clusters + 1))
        results = []
        
        for method, metric in valid_combinations:
            for n_clusters in cluster_range:
                try:
                    Z = linkage(self.data_scaled.values, method=method, metric=metric)
                    labels = fcluster(Z, n_clusters, criterion='maxclust')
                    
                    if len(np.unique(labels)) < 2:
                        continue
                    
                    sil = silhouette_score(self.data_scaled.values, labels)
                    
                    results.append({
                        'method': method,
                        'metric': metric,
                        'n_clusters': n_clusters,
                        'silhouette_score': sil
                    })
                    
                except Exception:
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
                           selection_method='sfs_rfe_grid', min_features=3, required_features=None, verbose=False):
        """
        Automatic feature selection using best method.
        
        Available selection_method options:
        - 'sfs': Sequential Forward Selection
        - 'rfe': Recursive Feature Elimination  
        - 'sfs_rfe_grid': Grid search with both SFS and RFE (recommended)
        """
        required_features = required_features or []
        
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
            })
        
        elif selection_method == 'sfs_rfe_grid':
            # SFS + RFE con Grid Search de hiperparámetros EXPANDIDO
            from .parameter_grids import ClusteringParameterGrid
            
            # Crear grid builder con rango de clusters optimizado
            grid_builder = ClusteringParameterGrid(
                min_clusters=max(2, self.min_clusters), 
                max_clusters=min(self.max_clusters, 8)
            )
            financial_grid = grid_builder.get_financial_grid()
            
            param_grid = {
                'valid_combinations': financial_grid['valid_combinations'],
                'n_clusters': financial_grid['n_clusters']
            }
            
            result = self.feature_selector.sfs_rfe_grid_search(
                param_grid=param_grid,
                min_features=min_features,
                required_features=required_features,
                top_n=25,
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
                })
            else:
                self.best_features = list(self.data_original.columns)
                feature_results.update({
                    'best_features': self.best_features,
                    'best_method': 'fallback',
                    'error': result['error']
                })
        
        else:
            # Método no reconocido, usar todas las features
            self.best_features = list(self.data_original.columns)
            feature_results.update({
                'best_features': self.best_features,
                'best_method': 'none',
                'error': f"Unknown selection method: {selection_method}"
            })
        
        # Store feature results for later use
        self.feature_selection_results = feature_results
        
        return feature_results
    
    def fit_predict(self, method=None, metric=None, n_clusters=None, features=None, verbose=False):
        """
        Execute final clustering with optimized parameters.
        """
        # Prioridad en la obtención de parámetros
        if hasattr(self, 'best_estimator') and self.best_estimator is not None and method is None:
            method = self.best_estimator.method
            metric = self.best_estimator.metric 
            n_clusters = self.best_estimator.n_clusters
        elif self.optimization_results and method is None:
            method = self.optimization_results['best_params']['method']
            metric = self.optimization_results['best_params']['metric']
            n_clusters = self.optimization_results['best_params']['n_clusters']
        else:
            method = method or 'ward'
            metric = metric or 'euclidean'
            n_clusters = n_clusters or 4
        
        # Use selected features if not provided
        if features is None:
            features = self.best_features or list(self.data_scaled.columns)
        
        # Select data
        data_to_cluster = self.data_scaled[features]
        
        # Create and fit estimator
        self.best_estimator = HierarchicalClusteringEstimator(
            method=method, n_clusters=n_clusters, metric=metric
        )
        
        self.labels_ = self.best_estimator.fit_predict(data_to_cluster)
        self.linkage_matrix_ = self.best_estimator.linkage_matrix_
        
        # Calculate final metrics
        final_scores = {
            'silhouette': round(silhouette_score(data_to_cluster, self.labels_), 4)
        }
        
        return {
            'labels': self.labels_,
            'scores': final_scores,
            'features_used': features,
            'parameters': {'method': method, 'metric': metric, 'n_clusters': n_clusters}
        }
    
    def get_cluster_centroids(self, features=None, include_counts=True, round_decimals=4):
        """
        Calculate cluster centroids (means) for each variable of the optimal clustering.
        
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
    
    def plot_dendrogram(self, figsize=(12, 8), title=None, max_d=None):
        """
        Plot hierarchical clustering dendrogram coloreado con clusters óptimos.
        """
        if self.linkage_matrix_ is None:
            raise ValueError("No clustering performed yet. Run fit_predict() first.")
        
        plt.figure(figsize=figsize)
        
        # Determinar el número de clusters de múltiples fuentes posibles
        n_clusters = None
        
        if max_d is not None:
            color_threshold = max_d
        else:
            # Buscar número de clusters desde múltiples fuentes
            if hasattr(self, 'best_estimator') and self.best_estimator is not None and hasattr(self.best_estimator, 'n_clusters'):
                n_clusters = self.best_estimator.n_clusters
            elif hasattr(self, 'labels_') and self.labels_ is not None:
                n_clusters = len(np.unique(self.labels_))
            elif hasattr(self, 'optimization_results') and self.optimization_results is not None:
                n_clusters = self.optimization_results['best_params'].get('n_clusters', None)
            
            # Calcular threshold basado en el número de clusters encontrado
            if n_clusters is not None and n_clusters > 1:
                distances = self.linkage_matrix_[:, 2]
                sorted_distances = np.sort(distances)
                threshold_idx = len(sorted_distances) - n_clusters + 1
                if threshold_idx >= 0 and threshold_idx < len(sorted_distances):
                    color_threshold = sorted_distances[threshold_idx]
                else:
                    color_threshold = np.max(self.linkage_matrix_[:, 2]) * 0.7
            else:
                color_threshold = np.max(self.linkage_matrix_[:, 2]) * 0.7
        
        # Crear dendrograma con colores y sin etiquetas de muestra
        dendro = dendrogram(self.linkage_matrix_, 
                           color_threshold=color_threshold,
                           above_threshold_color='gray',
                           no_labels=True,
                           leaf_rotation=0,
                           count_sort=True)
        
        # Mejorar título y etiquetas
        n_clusters_shown = len(set(dendro['color_list'])) - 1  # -1 para excluir gray
        
        plt.title(title or f'Dendrograma de Clustering Jerárquico\n({n_clusters_shown} clusters coloreados)')
        plt.xlabel('Muestras agrupadas por similitud')
        plt.ylabel('Distancia euclidiana')
        
        # Agregar línea horizontal en el threshold si está definido
        if color_threshold:
            cluster_label = n_clusters if n_clusters is not None else "Auto"
            plt.axhline(y=color_threshold, color='red', linestyle='--', alpha=0.7, 
                       label=f'Threshold (clusters={cluster_label})')
            plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        return dendro
    
    def plot_cluster_profiles(self, figsize=(15, 8), plot_type='bar'):
        """
        Plot cluster profiles showing centroids.
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
    
    def full_analysis(self, min_clusters=None, max_clusters=None, select_features=True, 
                     selection_method='sfs_rfe_grid', min_features=4, required_features=None,
                     plot_results=True, verbose=False):
        """
        Run complete financial clustering analysis pipeline.
        
        Available selection_method options:
        - 'sfs': Sequential Forward Selection
        - 'rfe': Recursive Feature Elimination
        - 'sfs_rfe_grid': Grid search with both SFS and RFE (recommended, default)
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
            
            # Si cualquier método devolvió hiperparámetros óptimos, usarlos para clustering final
            if 'best_hyperparams' in results['feature_selection']:
                optimized_params = results['feature_selection']['best_hyperparams']
                
                # Actualizar self con los parámetros optimizados
                self.best_estimator = HierarchicalClusteringEstimator(
                    method=optimized_params['method'],
                    metric=optimized_params['metric'],
                    n_clusters=optimized_params['n_clusters']
                )
                
                self.min_clusters = optimized_params['n_clusters']
                self.max_clusters = optimized_params['n_clusters']
        
        # Step 3: Final clustering
        results['clustering'] = self.fit_predict(verbose=verbose)
        
        # Step 4: Summary (simplified)
        results['summary'] = {
            'n_clusters': len(np.unique(self.labels_)),
            'cluster_sizes': dict(pd.Series(self.labels_).value_counts().sort_index()),
            'silhouette_score': results['clustering']['scores']['silhouette']
        }
        
        # Step 5: Plots
        if plot_results:
            self.plot_dendrogram()  
            results['centroids'] = self.plot_cluster_profiles(plot_type='bar')
            self.plot_cluster_profiles(plot_type='radar')
        
        return results
    
    def analyze_grid_expanded_results(self, results, display_top_n=15, verbose=False):
        """
        Analiza automáticamente los resultados del grid expandido con múltiples métricas.
        """
        feature_results = results.get('feature_selection', {})
        analysis_summary = {
            'best_method': feature_results.get('best_method', 'N/A'),
            'success': False,
            'insights': {},
            'top_results': None,
            'metric_analysis': {},
            'method_analysis': {}
        }
        
        if feature_results.get('best_method') == 'sfs_rfe_grid_expanded':
            analysis_summary['success'] = True
            
            # Extraer información clave
            winner_algorithm = feature_results.get('winner_algorithm', 'N/A')
            best_features = feature_results.get('best_features', [])
            best_hyperparams = feature_results.get('best_hyperparams', {})
            total_experiments = feature_results.get('total_experiments', 0)
            
            analysis_summary['insights'] = {
                'winner_algorithm': winner_algorithm,
                'best_features': best_features,
                'best_hyperparams': best_hyperparams,
                'total_experiments': total_experiments
            }
            
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
            
            # Obtener resultados detallados
            sfs_rfe_results = feature_results.get('sfs_rfe_grid_results', {})
            grid_results = sfs_rfe_results.get('grid_search_results', {})
            
            if 'all_results_df' in grid_results:
                top_n_results = grid_results['all_results_df'].head(display_top_n)
                analysis_summary['top_results'] = top_n_results
                
                # Analizar diversidad de métricas y métodos
                if len(top_n_results) > 0:
                    top_metrics = top_n_results['hyperparams_metric'].value_counts()
                    analysis_summary['metric_analysis'] = top_metrics.to_dict()
                    
                    top_methods = top_n_results['hyperparams_method'].value_counts()
                    analysis_summary['method_analysis'] = top_methods.to_dict()
        
        else:
            analysis_summary['error'] = feature_results.get('error', 'Método no es sfs_rfe_grid_expanded')
        
        return analysis_summary