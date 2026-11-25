import pandas as pd
import numpy as np
import os
import sys

# Ensure the local framework can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mif_framework.data_preprocessing.sampling import FinancialSampler
from mif_framework.data_preprocessing.fractional import FractionalDifferencer
from mif_framework.labeling.triple_barrier import TripleBarrierLabeler
from mif_framework.validation.purged_cv import PurgedKFold
from mif_framework.portfolio.hrp import HierarchicalRiskParity
from mif_framework.feature_engineering.volatility import VolatilityScaler
from mif_framework.feature_engineering.microstructure import MicrostructureFeatures
from mif_framework.models.predictor import MIFPredictor
from mif_framework.models.meta_labeling import MetaLabeler
from mif_framework.backtesting.engine import MIFBacktester

class MIFFullSystem:
    """
    The Complete Machine Intelligence for Finance Production Pipeline.
    Integrates Preprocessing, Labeling, ML Prediction, Meta-Filtering, and Backtesting.
    """
    
    def __init__(self, config=None):
        if config is None:
            config = {
                'sampling_type': 'dollar',
                'sampling_threshold': 10000,
                'diff_order': 0.5,
                'pt_sl': 0.02,
                't1': 20,
                'vol_window': 20
            }
        self.config = config
        self.sampler = FinancialSampler(sampling_type=config['sampling_type'], threshold=config['sampling_threshold'])
        self.differencer = FractionalDifferencer(d=config['diff_order'])
        self.labeler = TripleBarrierLabeler(pt_sl=config['pt_sl'], t1=config['t1'], vol_window=config['vol_window'])
        self.cv = PurgedKFold(n_splits=5, purge_window=5, embargo_window=5)
        self.hrp = HierarchicalRiskParity()
        self.feature_gen = MicrostructureFeatures()
        self.base_model = MIFPredictor(model_type='ensemble')
        self.meta_model = MetaLabeler()
        self.backtester = MIFBacktester()

    def run_end_to_end(self, raw_data: pd.DataFrame):
        """
        Executes the full professional quant pipeline.
        """
        print("\n--- Executing Full MIF Production Pipeline ---")
        
        # 1. Data Preprocessing
        print("[1/6] Sampling into Information Time...")
        bars = self.sampler.sample(raw_data)
        
        print("[2/6] Applying Fractional Differentiation...")
        stationary_prices = self.differencer.fit_transform(bars['close'])
        
        # 2. Feature Engineering
        print("[3/6] Generating Alpha Features...")
        bars_with_features = self.feature_gen.add_all_features(bars)
        bars_with_features['stat_price'] = stationary_prices
        
        # 3. Labeling (Adaptive Triple Barrier)
        print("[4/6] Creating Dynamic Labels...")
        labels = self.labeler.label_series(bars['close'], use_dynamic_vol=True)
        
        # 4. Machine Learning (Base + Meta)
        print("[5/6] Training AI (Base + Meta-Labeling)...")
        X = bars_with_features.drop(columns=['open', 'high', 'low', 'close', 'volume'], errors='ignore').fillna(0)
        y = labels.fillna(0).astype(int)
        
        splits = self.cv.split(X, y)
        train_idx, test_idx = splits[0] 
        
        self.base_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        base_preds = self.base_model.predict(X.iloc[test_idx])
        
        self.meta_model.fit(X.iloc[test_idx], y.iloc[test_idx], base_preds)
        meta_conf = self.meta_model.predict_confidence(X.iloc[test_idx])
        
        final_signals = np.where(meta_conf == 1, base_preds, 0)
        final_signals_series = pd.Series(final_signals, index=X.iloc[test_idx].index)
        
        # 5. Backtesting
        print("[6/6] Running Walk-Forward Backtest...")
        test_prices = bars['close'].iloc[test_idx]
        perf = self.backtester.run_backtest(test_prices, final_signals_series)
        
        print("\n--- FINAL SYSTEM PERFORMANCE ---")
        print(f"Total Return: {perf['total_return']:.2%}")
        print(f"Sharpe Ratio: {perf['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {perf['max_drawdown']:.2%}")
        
        return perf

if __name__ == "__main__":
    np.random.seed(42)
    n_ticks = 10000
    raw_data = pd.DataFrame({
        'price': 100 + np.cumsum(np.random.randn(n_ticks)),
        'volume': np.random.randint(1, 100, n_ticks),
        'high': 102 + np.random.randn(n_ticks),
        'low': 98 + np.random.randn(n_ticks),
        'buy_vol': np.random.randint(1, 100, n_ticks),
        'sell_vol': np.random.randint(1, 100, n_ticks)
    })
    
    system = MIFFullSystem()
    system.run_end_to_end(raw_data)
