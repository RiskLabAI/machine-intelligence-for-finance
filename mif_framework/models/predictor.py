import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report
from typing import Union, Tuple

class MIFPredictor:
    """
    Predictive brain for the MIF Framework.
    Implements an ensemble of Gradient Boosted Decision Trees (XGBoost and LightGBM)
    to predict the triple-barrier labels.
    """
    
    def __init__(self, model_type: str = 'ensemble'):
        """
        Args:
            model_type: 'xgb', 'lgbm', or 'ensemble'.
        """
        self.model_type = model_type
        self.xgb_model = XGBClassifier(
            n_estimators=100, 
            learning_rate=0.05, 
            max_depth=5, 
            random_state=42, 
            eval_metric='mlogloss'
        )
        self.lgbm_model = LGBMClassifier(
            n_estimators=100, 
            learning_rate=0.05, 
            max_depth=5, 
            random_state=42, 
            verbose=-1
        )

    def _prepare_labels(self, y: pd.Series) -> np.ndarray:
        """
        ML models expect labels starting from 0.
        We handle both ternary (-1, 0, 1) and binary (-1, 1) labels.
        """
        unique_vals = np.unique(y)
        # Create a mapping based on whatever values are actually present
        mapping = {val: i for i, val in enumerate(unique_vals)}
        return y.map(mapping).values

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Trains the selected model(s).
        """
        y_transformed = self._prepare_labels(y)
        
        if self.model_type == 'xgb':
            self.xgb_model.fit(X, y_transformed)
        elif self.model_type == 'lgbm':
            self.lgbm_model.fit(X, y_transformed)
        elif self.model_type == 'ensemble':
            self.xgb_model.fit(X, y_transformed)
            self.lgbm_model.fit(X, y_transformed)
        else:
            raise ValueError("Invalid model_type. Use 'xgb', 'lgbm', or 'ensemble'.")
        
        # Store the mapping for inverse transformation during prediction
        self.label_mapping = {i: val for val, i in {val: i for i, val in enumerate(np.unique(y))}.items()}
        print(f"Model {self.model_type} trained successfully.")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predicts labels for the given features.
        Returns labels in their original format.
        """
        if self.model_type == 'xgb':
            preds = self.xgb_model.predict(X)
        elif self.model_type == 'lgbm':
            preds = self.lgbm_model.predict(X)
        elif self.model_type == 'ensemble':
            p1 = self.xgb_model.predict(X)
            p2 = self.lgbm_model.predict(X)
            preds = np.round((p1 + p2) / 2).astype(int)
        
        # Map back to original labels using the stored mapping
        return pd.Series([self.label_mapping[p] for p in preds], index=X.index)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Evaluates the model performance.
        """
        preds = self.predict(X)
        acc = accuracy_score(y, preds)
        report = classification_report(y, preds, output_dict=True)
        
        return {
            "accuracy": acc,
            "report": report
        }

if __name__ == "__main__":
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
    y = pd.Series(np.random.choice([-1, 0, 1], 100))
    
    predictor = MIFPredictor(model_type='ensemble')
    predictor.fit(X, y)
    preds = predictor.predict(X)
    
    print(f"Prediction sample:\n{preds.head()}")
    print(f"Accuracy:\n{predictor.evaluate(X, y)['accuracy']}")
