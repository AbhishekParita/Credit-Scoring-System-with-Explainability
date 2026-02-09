"""
Custom calibration classes for sklearn 1.6+ compatibility
"""
import numpy as np
from sklearn.calibration import IsotonicRegression
from sklearn.base import BaseEstimator, ClassifierMixin


class CalibratedClassifier(BaseEstimator, ClassifierMixin):
    """Manual calibration wrapper for sklearn 1.6+ compatibility"""
    def __init__(self, base_model):
        self.base_model = base_model
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.classes_ = None
        
    def fit(self, X, y):
        # Get base model predictions
        probs = self.base_model.predict_proba(X)[:, 1]
        # Fit isotonic regression
        self.calibrator.fit(probs, y)
        self.classes_ = self.base_model.classes_
        return self
        
    def predict_proba(self, X):
        # Get base predictions and calibrate
        probs = self.base_model.predict_proba(X)[:, 1]
        calibrated_probs = self.calibrator.predict(probs)
        # Return 2D array [prob_class_0, prob_class_1]
        return np.column_stack([1 - calibrated_probs, calibrated_probs])
        
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
