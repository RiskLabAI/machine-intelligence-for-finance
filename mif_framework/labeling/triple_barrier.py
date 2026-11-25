import numpy as np
import pandas as pd
from typing import Union, Optional
from mif_framework.feature_engineering.volatility import VolatilityScaler

class TripleBarrierLabeler:
    """
    Implements the Triple Barrier Method for financial labeling.
    Now supports Dynamic Volatility Scaling for adaptive barriers.
    """
    
    def __init__(self, pt_sl: float = 0.02, target_multiplier: float = 1.0, t1: int = 20, vol_window: int = 20):
        """
        Args:
            pt_sl (float): Base percentage width of barriers.
            target_multiplier (float): Multiplier for the barriers.
            t1 (int): The time limit (vertical barrier) in terms of bars.
            vol_window (int): Look-back window for volatility scaling.
        """
        self.pt_sl = pt_sl
        self.target_multiplier = target_multiplier
        self.t1 = t1
        self.vol_scaler = VolatilityScaler(window=vol_window)

    def label_series(self, price_series: pd.Series, use_dynamic_vol: bool = False) -> pd.Series:
        """
        Labels the price series using the Triple Barrier Method.
        
        Args:
            price_series: The input price series.
            use_dynamic_vol (bool): If True, scales barriers by current volatility.
            
        Returns:
            A pandas Series of labels: 1 (upper), -1 (lower), 0 (time limit).
        """
        labels = np.full(len(price_series), np.nan)
        
        # Get dynamic barriers if requested
        barriers_df = None
        if use_dynamic_vol:
            barriers_df = self.vol_scaler.get_dynamic_barriers(price_series)
        
        for i in range(len(price_series) - self.t1):
            start_price = price_series.iloc[i]
            
            # Determine barriers for this specific timestamp
            if use_dynamic_vol and not np.isnan(barriers_df['upper'].iloc[i]):
                upper_barrier = barriers_df['upper'].iloc[i]
                lower_barrier = barriers_df['lower'].iloc[i]
            else:
                # Fallback to static barriers
                upper_barrier = start_price * (1 + self.pt_sl * self.target_multiplier)
                lower_barrier = start_price * (1 - self.pt_sl * self.target_multiplier)
            
            # Check subsequent prices until t1
            for j in range(i + 1, i + self.t1 + 1):
                current_price = price_series.iloc[j]
                if current_price >= upper_barrier:
                    labels[i] = 1
                    break
                elif current_price <= lower_barrier:
                    labels[i] = -1
                    break
            else:
                labels[i] = 0
                
        return pd.Series(labels, index=price_series.index)

if __name__ == "__main__":
    # Test Dynamic vs Static
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(500))
    price_series = pd.Series(prices)
    
    labeler = TripleBarrierLabeler(pt_sl=0.02)
    
    static_labels = labeler.label_series(price_series, use_dynamic_vol=False)
    dynamic_labels = labeler.label_series(price_series, use_dynamic_vol=True)
    
    print(f"Static Labels Dist:\n{static_labels.value_counts()}")
    print(f"Dynamic Labels Dist:\n{dynamic_labels.value_counts()}")
