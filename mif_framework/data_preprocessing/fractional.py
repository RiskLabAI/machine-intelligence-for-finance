import numpy as np
import pandas as pd
from typing import Union

class FractionalDifferencer:
    """
    Implements Fractional Differentiation for financial time series to achieve 
    stationarity while preserving maximum memory.
    
    Based on the binomial expansion of (1 - B)^d.
    """
    
    def __init__(self, d: float, threshold: float = 1e-4):
        """
        Args:
            d (float): The differentiation order. 
                       d=0: No change.
                       d=1: Standard first-difference.
                       0 < d < 1: Fractional difference.
            threshold (float): The weight threshold. Weights smaller than this 
                                are ignored to keep the window finite.
        """
        if d < 0:
            raise ValueError("Differentiation order 'd' must be non-negative.")
        
        self.d = d
        self.threshold = threshold
        self.weights = self._build_weights()

    def _build_weights(self) -> np.ndarray:
        """
        Constructs the binomial weights for the fractional difference operator.
        """
        weights = [1.0]
        k = 1
        while True:
            # w_k = -w_{k-1} * (d - k + 1) / k
            w_k = -weights[-1] * (self.d - k + 1) / k
            if abs(w_k) < self.threshold:
                break
            weights.append(w_k)
            k += 1
        return np.array(weights)

    def fit_transform(self, series: Union[pd.Series, np.ndarray]) -> pd.Series:
        """
        Applies fractional differentiation to the given series.
        
        Args:
            series: The input price series.
            
        Returns:
            A pandas Series containing the fractionally differenced values.
        """
        if isinstance(series, np.ndarray):
            series = pd.Series(series)
        
        # Ensure we work with a copy to avoid modifying original data
        series = series.copy()
        
        # Apply the weights to the series using a convolution-like approach
        # (1-B)^d X_t = sum_{k=0}^{N} w_k * X_{t-k}
        
        # We use a sliding window to calculate the weighted sum
        weights = self.weights
        n_weights = len(weights)
        
        # Pre-allocate the result array
        result = np.full(len(series), np.nan)
        
        # We can only calculate values once we have enough history for the weights
        # However, for financial data, we often start with whatever is available
        # and let the result be NaN for the first few points.
        
        for i in range(n_weights - 1, len(series)):
            window = series.iloc[i - n_weights + 1 : i + 1].values[::-1]
            result[i] = np.dot(window, weights)
            
        return pd.Series(result, index=series.index)

    def get_weights(self) -> np.ndarray:
        """Returns the calculated binomial weights."""
        return self.weights

if __name__ == "__main__":
    # Simple Test
    data = np.cumsum(np.random.randn(100)) + 100
    fd = FractionalDifferencer(d=0.5)
    diffed = fd.fit_transform(data)
    print(f"Input head:\n{data[:5]}")
    print(f"Diffed head:\n{diffed.head()}")
    print(f"Number of weights used: {len(fd.get_weights())}")
