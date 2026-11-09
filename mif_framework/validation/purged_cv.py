import numpy as np
import pandas as pd
from typing import List, Tuple

class PurgedKFold:
    """
    Implements Purged K-Fold Cross-Validation for financial time series.
    
    Financial data has temporal dependence. Standard K-Fold leads to leakage
    because samples near the fold boundary share information.
    
    This implementation removes (purges) samples from the training set that
    overlap in time with the test set.
    """
    
    def __init__(self, n_splits: int = 5, purge_window: int = 0, embargo_window: int = 0):
        """
        Args:
            n_splits (int): Number of folds.
            purge_window (int): Number of samples to remove before the test set.
            embargo_window (int): Number of samples to remove after the test set.
        """
        self.n_splits = n_splits
        self.purge_window = purge_window
        self.embargo_window = embargo_window

    def split(self, X: pd.DataFrame, y: pd.Series) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generates indices for purged training and testing sets.
        
        Args:
            X: Features DataFrame.
            y: Labels Series.
            
        Returns:
            A list of tuples (train_indices, test_indices).
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        fold_size = n_samples // self.n_splits
        
        splits = []
        
        for i in range(self.n_splits):
            # Define the test fold boundaries
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i != self.n_splits - 1 else n_samples
            
            test_indices = indices[test_start:test_end]
            
            # Define training indices (everything except the test fold)
            train_indices = np.concatenate([indices[:test_start], indices[test_end:]])
            
            # --- PURGING ---
            # Remove training samples that overlap with the test fold start
            if test_start > self.purge_window:
                # Purge samples just before the test start
                purge_start = test_start - self.purge_window
                # We remove these from the first part of the train set
                mask = np.ones(len(train_indices), dtype=bool)
                # Find the index in train_indices that corresponds to the purge window
                # Since train_indices are sorted, we can find the boundary
                idx_before_test = np.where(train_indices < test_start)[0][-1]
                purge_boundary = max(0, idx_before_test - self.purge_window + 1)
                mask[purge_boundary : idx_before_test + 1] = False
                train_indices = train_indices[mask]
                
            # --- EMBARGO ---
            # Remove training samples that follow the test fold to prevent leakage
            if test_end + self.embargo_window < n_samples:
                # Find the index in train_indices that corresponds to the start of the second train part
                idx_after_test = np.where(train_indices >= test_end)[0][0]
                embargo_boundary = idx_after_test + self.embargo_window
                # Remove the samples starting from the test end up to the embargo window
                mask = np.ones(len(train_indices), dtype=bool)
                mask[idx_after_test : embargo_boundary] = False
                train_indices = train_indices[mask]
            elif test_end < n_samples:
                # If we are near the end, we might not have a full embargo window, 
                # but we still remove what we can.
                idx_after_test = np.where(train_indices >= test_end)[0][0]
                mask = np.ones(len(train_indices), dtype=bool)
                mask[idx_after_test:] = False
                train_indices = train_indices[mask]

            splits.append((train_indices, test_indices))
            
        return splits

if __name__ == "__main__":
    # Simple Test
    X = pd.DataFrame(np.random.randn(100, 2))
    y = pd.Series(np.random.randint(0, 2, 100))
    
    pkf = PurgedKFold(n_splits=5, purge_window=5, embargo_window=5)
    splits = pkf.split(X, y)
    
    for i, (train, test) in enumerate(splits):
        print(f"Fold {i}: Train size={len(train)}, Test size={len(test)}")
        # Verify that no train index is within the purge/embargo window of the test index
        test_min = test.min()
        test_max = test.max()
        # Check for leakages
        leakage = [t for t in train if (test_min - 5 <= t <= test_min) or (test_max <= t <= test_max + 5)]
        print(f"  Leakage detected: {len(leakage)}")
