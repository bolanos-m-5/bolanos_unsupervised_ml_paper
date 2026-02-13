"""
sklearn-compatible hierarchical clustering estimator.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster


class HierarchicalClusteringEstimator(BaseEstimator, ClusterMixin):
    """
    sklearn-compatible wrapper for hierarchical clustering.
    Enables use with sklearn's feature selection tools.
    
    Parameters:
    - method: linkage method ('ward', 'average', 'single', 'complete')
    - n_clusters: number of clusters
    - metric: distance metric ('euclidean', 'manhattan', 'cosine', etc.)
    - criterion_metric: metric for scoring (default: 'silhouette')
    """
    
    def __init__(self, method='ward', n_clusters=3, metric='euclidean', criterion_metric='silhouette'):
        self.method = method
        self.n_clusters = n_clusters
        self.metric = metric
        self.criterion_metric = criterion_metric
        self.labels_ = None
        self.linkage_matrix_ = None
    
    def fit(self, X, y=None):
        """Fit hierarchical clustering to data."""
        if self.method == 'ward' and self.metric != 'euclidean':
            raise ValueError("Ward method requires euclidean metric")
        
        # Convertir a numpy array si es DataFrame
        import numpy as np
        X_array = np.array(X) if hasattr(X, 'values') else X
        
        try:
            self.linkage_matrix_ = linkage(X_array, method=self.method, metric=self.metric)
            self.labels_ = fcluster(self.linkage_matrix_, self.n_clusters, criterion='maxclust')
        except Exception as e:
            # Si falla, simplemente pasar el error sin validaciones complejas
            raise e
                
        return self
    
    def fit_predict(self, X, y=None):
        """Fit and return cluster labels."""
        self.fit(X, y)
        return self.labels_
    
    def score(self, X, y=None):
        """
        Score the clustering using Silhouette Score.
        Higher is better (sklearn convention).
        """
        if self.labels_ is None:
            self.fit(X)
        
        if len(np.unique(self.labels_)) < 2:
            return -np.inf
        
        try:
            return silhouette_score(X, self.labels_)
        except:
            return -np.inf
    
    def get_params(self, deep=True):
        """Get parameters (required by sklearn)."""
        return {
            'method': self.method,
            'n_clusters': self.n_clusters,
            'metric': self.metric,
            'criterion_metric': self.criterion_metric
        }
    
    def set_params(self, **params):
        """Set parameters (required by sklearn)."""
        for key, value in params.items():
            setattr(self, key, value)
        return self