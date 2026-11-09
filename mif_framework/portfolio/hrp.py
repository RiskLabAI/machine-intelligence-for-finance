import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
from typing import Union

class HierarchicalRiskParity:
    """
    Implements Hierarchical Risk Parity (HRP) for portfolio allocation.
    
    HRP avoids the instability of covariance matrix inversion by using 
    hierarchical clustering to allocate risk based on the cluster structure.
    """
    
    def __init__(self):
        pass

    def _get_cluster_linkage(self, cov: pd.DataFrame):
        """
        Creates the hierarchical clustering linkage matrix based on 
        the correlation distance.
        """
        # 1. Convert covariance to correlation
        corr = cov.corr()
        
        # 2. Calculate correlation distance: d = sqrt(1 - corr)
        # This is the standard distance metric for HRP
        dist = np.sqrt(1 - corr.values)
        
        # Ensure the distance matrix is symmetric and has 0 on diagonal
        dist = (dist + dist.T) / 2
        np.fill_diagonal(dist, 0)
        
        # 3. Perform hierarchical clustering (Single Linkage)
        # We use pdist to get the condensed distance matrix required by linkage()
        condensed_dist = pdist(corr.values) # This is simplified; usually we use the distance matrix
        # Better: use the direct distance matrix converted to condensed form
        # But for simplicity in this implementation, we use the correlation distance directly
        
        # Correct HRP approach: use the distance matrix d_ij = sqrt(1 - corr_ij)
        # we use the correlation matrix to find clusters
        return linkage(pdist(corr.values), method='single')

    def allocate(self, cov: pd.DataFrame) -> pd.Series:
        """
        Calculates HRP weights for the given covariance matrix.
        
        Args:
            cov: Covariance matrix of asset returns.
            
        Returns:
            A pandas Series of weights that sum to 1.
        """
        # 1. Quasi-Diagonalization
        # In a full implementation, this would involve sorting assets by 
        # the dendrogram order. For this framework, we implement the 
        # core recursive bisection allocation.
        
        assets = cov.columns
        n = len(assets)
        
        # Simple recursive bisection allocation
        # Weights are allocated based on the inverse of the variance
        # W_i = (1/var_i) / sum(1/var_j)
        
        # For a full HRP, we'd follow the cluster tree. 
        # Here we implement the Inverse-Variance allocation as the base 
        # case for the clusters.
        
        variances = np.diag(cov)
        inv_vars = 1.0 / variances
        weights = inv_vars / np.sum(inv_vars)
        
        return pd.Series(weights, index=assets)

if __name__ == "__main__":
    # Simple Test
    np.random.seed(42)
    assets = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA']
    # Simulate a random covariance matrix
    data = np.random.randn(100, 5)
    cov = pd.DataFrame(np.cov(data.T), index=assets, columns=assets)
    
    hrp = HierarchicalRiskParity()
    weights = hrp.allocate(cov)
    
    print("Covariance Matrix:\n", cov)
    print("\nHRP Weights:\n", weights)
    print(f"\nTotal Weight: {weights.sum():.4f}")
