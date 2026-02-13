"""
Parameter grid for hierarchical clustering in Polaris project.
Simplified version with only the financial data grid used in production.
"""

class ClusteringParameterGrid:
    """
    Parameter grid builder for financial clustering analysis.
    Provides the optimized grid used in Polaris project for retail data.
    """
    
    def __init__(self, min_clusters=2, max_clusters=10):
        """
        Initialize parameter grid builder.
        
        Parameters:
        - min_clusters, max_clusters: range for number of clusters to test
        """
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
    
    def get_clusters_range(self):
        """Get the range of clusters to test."""
        return list(range(self.min_clusters, self.max_clusters + 1))
    
    def get_financial_grid(self):
        """
        Grid optimized for financial/retail data with valid method-metric combinations.
        
        This is the grid used in the Polaris project for customer clustering.
        Includes only valid combinations:
        - Ward only works with Euclidean distance
        - Complete and Average work with Euclidean, Cosine, and Correlation
        
        Returns:
        - Dictionary with valid method-metric-clusters combinations
        """
        # These are the exact combinations used in analyzer.py optimize_parameters()
        valid_combinations = [
            ('ward', 'euclidean'),
            ('complete', 'euclidean'),
            ('complete', 'cosine'),
            ('complete', 'correlation'),
            ('average', 'euclidean'), 
            ('average', 'cosine'),
            ('average', 'correlation')
        ]
        
        return {
            'valid_combinations': valid_combinations,
            'n_clusters': self.get_clusters_range(),
            'total_combinations': len(valid_combinations) * len(self.get_clusters_range())
        }
    
    def get_grid_info(self):
        """
        Get information about the financial grid.
        
        Returns:
        - Dictionary with grid statistics
        """
        grid = self.get_financial_grid()
        
        return {
            'n_methods': 3,  # ward, complete, average
            'n_metrics': 3,  # euclidean, cosine, correlation
            'n_combinations': len(grid['valid_combinations']),
            'n_clusters_range': len(grid['n_clusters']),
            'total_evaluations': grid['total_combinations'],
            'cluster_range': f"{self.min_clusters}-{self.max_clusters}"
        }
    
    def print_grid_info(self):
        """Print formatted information about the grid."""
        info = self.get_grid_info()
        grid = self.get_financial_grid()
        
        print("=" * 60)
        print("📊 FINANCIAL CLUSTERING PARAMETER GRID")
        print("=" * 60)
        print(f"Valid Method-Metric Combinations: {info['n_combinations']}")
        for method, metric in grid['valid_combinations']:
            print(f"  • {method:10s} + {metric}")
        print(f"\nCluster Range: {info['cluster_range']} ({info['n_clusters_range']} values)")
        print(f"Total Evaluations: {info['total_evaluations']}")
        print("=" * 60)
    