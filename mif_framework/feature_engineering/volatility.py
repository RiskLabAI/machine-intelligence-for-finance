import numpy as np
import pandas as pd
from typing import Union

class VolatilityScaler:
    """
    Implements Dynamic Volatility Scaling for financial labels.
    
    Instead of using a fixed percentage for profit/loss barriers, 
    this class calculates the daily volatility (e.g., via standard deviation 
    of returns) to scale barriers based on current market conditions.
    """
    
    def __init__(self, window: int = 20, multiplier: float = 1.0):
        """
        Args:
            window (int): Look-back window for calculating volatility.
            multiplier (float): Scaling factor for the volatility.
        """
        self.window = window
        self.multiplier = multiplier

    def calculate_volatility(self, price_series: pd.Series) -> pd.Series:
        """
        Calculates the rolling volatility of returns.
        """
        # Calculate log returns
        returns = np.log(price_series / price_series.shift(1))
        
        # Calculate rolling standard deviation
        vol = returns.rolling(window=self.window).std()
        
        return vol

    def get_dynamic_barriers(self, price_series: pd.Series) -> pd.DataFrame:
        """
        Generates a DataFrame of upper and lower barriers for each point in time.
        """
        vol = self.calculate_volatility(price_series)
        
        # Upper Barrier = Price * (1 + vol * multiplier)
        # Lower Barrier = Price * (1 - vol * multiplier)
        upper = price_series * (1 + vol * self.multiplier)
        lower = price_series * (1 - vol * self.multiplier)
        
        return pd.DataFrame({
            'upper': upper,
            'lower': lower,
            'volatility': vol
        }, index=price_series.index)

if __name__ == "__main__":
    # Simple Test
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(1000))
    price_series = pd.Series(prices)
    
    scaler = VolatilityScaler(window=20)
    barriers = scaler.get_dynamic_barriers(price_series)
    
    print("Price vs Dynamic Barriers head:\n")
    combined = pd.concat([price_series, barriers], axis=1)
    combined.columns = ['price', 'upper', 'lower', 'vol']
    print(combined.tail())
