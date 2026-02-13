"""
Clustering module for hierarchical clustering analysis.

This module provides:
- HierarchicalClusteringEstimator: sklearn-compatible hierarchical clustering
- ClusteringFeatureSelector: Feature selection methods (SFS, RFE, exhaustive)  
- ClusteringAnalyzer: Main clustering analysis and evaluation
- ClusteringParameterGrid: Parameter grid builder for different data types
"""

from .estimator import HierarchicalClusteringEstimator
from .feature_selector import ClusteringFeatureSelector
from .analyzer import ClusteringAnalyzer
from .parameter_grids import ClusteringParameterGrid

__all__ = [
    'HierarchicalClusteringEstimator',
    'ClusteringFeatureSelector', 
    'ClusteringAnalyzer',
    'ClusteringParameterGrid'
]