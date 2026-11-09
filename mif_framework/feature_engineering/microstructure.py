import numpy as np
import pandas as pd
from typing import Union

class MicrostructureFeatures:
    """
    Generates market microstructure features from tick data.
    Based on Chapter 11 of Machine Intelligence for Finance.
    """
    
    def __init__(self):
        pass

    def calculate_volatility_range(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Calculates the high-low range volatility (Parkinson-like).
        Required columns: 'high', 'low'
        """
        # range = log(high/low)
        log_range = np.log(df['high'] / df['low'])
        return log_range.rolling(window=window).mean()

    def calculate_order_imbalance(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculates a simple order imbalance proxy.
        Requires columns: 'buy_vol', 'sell_vol'
        """
        if 'buy_vol' not in df.columns or 'sell_vol' not in df.columns:
            # Fallback: If we only have total volume, we can't calculate imbalance
            # This is a placeholder for the full implementation
            return pd.Series(0, index=df.index)
            
        imbalance = (df['buy_vol'] - df['sell_vol']) / (df['buy_vol'] + df['sell_vol'])
        return imbalance

    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds all microstructure features to the dataframe.
        """
        df = df.copy()
        df['micro_vol'] = self.calculate_volatility_range(df)
        df['imbalance'] = self.calculate_order_imbalance(df)
        return df

if __name__ == "__main__":
    # Simple Test
    np.random.seed(42)
    df = pd.DataFrame({
        'high': np.random.uniform(101, 105, 100),
        'low': np.random.uniform(95, 99, 100),
        'buy_vol': np.random.randint(1, 100, 100),
        'sell_vol': np.random.randint(1, 100, 100)
    })
    
    mf = MicrostructureFeatures()
    df_features = mf.add_all_features(df)
    print(df_features.tail())
