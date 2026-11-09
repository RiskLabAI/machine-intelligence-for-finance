import pandas as pd
import numpy as np
from typing import Literal

class FinancialSampler:
    """
    Implements event-based sampling for financial time series.
    Moves away from chronological time to 'Information Time'.
    """
    
    def __init__(self, sampling_type: Literal['tick', 'volume', 'dollar'], threshold: float):
        """
        Args:
            sampling_type: The type of bar to create.
            threshold: The value at which a new bar is closed.
        """
        self.sampling_type = sampling_type
        self.threshold = threshold

    def sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Samples the input dataframe into bars.
        
        Required columns in df:
        - 'price': The asset price.
        - 'volume': The volume traded in the tick.
        
        Returns:
            A DataFrame where each row is a closed bar.
        """
        if 'price' not in df.columns or 'volume' not in df.columns:
            raise ValueError("DataFrame must contain 'price' and 'volume' columns.")
            
        df = df.copy()
        
        if self.sampling_type == 'tick':
            # Every N ticks = 1 bar
            df['cum_val'] = np.arange(len(df))
        elif self.sampling_type == 'volume':
            # Every N volume = 1 bar
            df['cum_val'] = df['volume'].cumsum()
        elif self.sampling_type == 'dollar':
            # Every N dollar value = 1 bar
            df['dollar_vol'] = df['price'] * df['volume']
            df['cum_val'] = df['dollar_vol'].cumsum()
        else:
            raise ValueError("Invalid sampling_type. Use 'tick', 'volume', or 'dollar'.")
            
        # Determine which bar each tick belongs to
        df['bar_id'] = (df['cum_val'] // self.threshold).astype(int)
        
        # Aggregate ticks into bars
        # We use OHLC (Open, High, Low, Close) as the standard output for financial bars
        bars = df.groupby('bar_id').agg({
            'price': ['first', 'max', 'min', 'last'],
            'volume': 'sum'
        })
        
        bars.columns = ['open', 'high', 'low', 'close', 'volume']
        return bars

if __name__ == "__main__":
    # Simple Test
    np.random.seed(42)
    test_df = pd.DataFrame({
        'price': np.random.uniform(100, 110, 1000),
        'volume': np.random.randint(1, 100, 1000)
    })
    
    sampler = FinancialSampler(sampling_type='volume', threshold=5000)
    volume_bars = sampler.sample(test_df)
    print(f"Original ticks: {len(test_df)}")
    print(f"Volume bars created: {len(volume_bars)}")
    print(volume_bars.head())
