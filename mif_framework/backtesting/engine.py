import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any

class MIFBacktester:
    """
    Implements a professional Walk-Forward Backtester for the MIF Framework.
    
    This engine simulates real-world trading by applying transaction costs
    and calculating key risk-adjusted performance metrics.
    """
    
    def __init__(self, initial_capital: float = 100000.0, commission: float = 0.0001, slippage: float = 0.0001):
        """
        Args:
            initial_capital: Starting balance in dollars.
            commission: Percentage fee per trade (e.g., 0.0001 = 1bp).
            slippage: Estimated price impact per trade.
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage

    def run_backtest(self, prices: pd.Series, signals: pd.Series) -> Dict[str, Any]:
        """
        Runs the backtest based on price series and predicted signals.
        
        Args:
            prices: The close price series.
            signals: Predicted labels (-1, 0, 1).
            
        Returns:
            A dictionary containing the equity curve and performance metrics.
        """
        # Calculate raw returns
        returns = prices.pct_change().fillna(0)
        
        # Position is the signal from the previous step (avoid look-ahead bias)
        positions = signals.shift(1).fillna(0)
        
        # Strategy returns = position * returns
        strategy_returns = positions * returns
        
        # Calculate Transaction Costs
        # Cost is applied whenever the position changes
        trades = positions.diff().fillna(0).abs()
        costs = trades * (self.commission + self.slippage)
        
        # Net returns
        net_returns = strategy_returns - costs
        
        # Cumulative Equity Curve
        equity_curve = (1 + net_returns).cumprod() * self.initial_capital
        
        # Metrics Calculation
        total_return = (equity_curve.iloc[-1] / self.initial_capital) - 1
        
        # Annualized Sharpe Ratio (assuming 252 trading days)
        # We use the average return / std of return
        if len(net_returns) > 1:
            sharpe = (net_returns.mean() / net_returns.std()) * np.sqrt(252) if net_returns.std() != 0 else 0
        else:
            sharpe = 0
            
        # Max Drawdown
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        return {
            "equity_curve": equity_curve,
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "net_returns": net_returns
        }

if __name__ == "__main__":
    # Simple Test
    np.random.seed(42)
    prices = pd.Series(100 + np.cumsum(np.random.randn(1000)), index=pd.date_range("2020-01-01", periods=1000))
    # Random signals for testing
    signals = pd.Series(np.random.choice([-1, 0, 1], 1000), index=prices.index)
    
    backtester = MIFBacktester()
    results = backtester.run_backtest(prices, signals)
    
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
