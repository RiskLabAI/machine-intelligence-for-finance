import numpy as np
import pandas as pd
from typing import Tuple
from mif_framework.models.predictor import MIFPredictor

class MetaLabeler:
    """
    Implements Meta-Labeling to filter the predictions of a primary model.
    
    The Primary Model decides the SIDE (Buy/Sell).
    The Meta-Model decides the SIZE (Should we actually take this trade?).
    """
    
    def __init__(self):
        # The meta-model is a binary classifier
        self.meta_model = MIFPredictor(model_type='ensemble')

    def create_meta_labels(self, primary_preds: pd.Series, true_labels: pd.Series) -> pd.Series:
        """
        Creates binary labels for the meta-model.
        1 if primary prediction matches true label, 0 otherwise.
        """
        # Only label trades where the primary model actually predicted a side (1 or -1)
        meta_labels = np.where(primary_preds != 0, 
                              (primary_preds == true_labels).astype(int), 
                              np.nan)
        
        return pd.Series(meta_labels, index=true_labels.index)

    def fit(self, X: pd.DataFrame, y_true: pd.Series, primary_preds: pd.Series):
        """
        Trains the meta-model to predict the success of the primary model.
        """
        # Create binary labels for the meta-model (0 or 1)
        meta_y = self.create_meta_labels(primary_preds, y_true)
        
        # Only train on samples where the primary model actually made a bet
        mask = ~np.isnan(meta_y)
        if mask.sum() == 0:
            print("Warning: No trades were made by primary model. Meta-model cannot be trained.")
            return

        # IMPORTANT: The MIFPredictor expects labels in [-1, 0, 1]
        # Meta-labels are currently [0, 1]. 
        # We map: 0 (Incorrect) -> -1, 1 (Correct) -> 1
        meta_y_mapped = meta_y[mask].map({0: -1, 1: 1})
        
        self.meta_model.fit(X[mask], meta_y_mapped)
        print("Meta-Model trained successfully.")

    def predict_confidence(self, X: pd.DataFrame) -> pd.Series:
        """
        Predicts if the primary model's prediction is likely to be correct.
        Returns 1 for 'Correct/High Confidence' and -1 for 'Incorrect/Low Confidence'.
        """
        return self.meta_model.predict(X)

if __name__ == "__main__":
    # Simple Test
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(200, 5))
    y_true = pd.Series(np.random.choice([-1, 0, 1], 200))
    primary_preds = pd.Series(np.random.choice([-1, 0, 1], 200))
    
    meta = MetaLabeler()
    meta.fit(X, y_true, primary_preds)
    conf = meta.predict_confidence(X)
    
    print(f"Meta-confidence head:\n{conf.head()}")
    print(f"Confidence distribution:\n{conf.value_counts()}")
