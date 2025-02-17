import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# supported algorithms
algos = {
    'logistic_regression': LogisticRegression
    }

class Classification:
    def __init__(self, algorithm: str, **kwargs):
        """
        Initializes the Classification model class.

        Parameters:
            algorithm (str): The algorithm to use.
            kwargs: Parameters for the specified algorithm.
        """
        
        # validate the algorithm and initialize it
        self.method = algorithm.lower()
        if self.method in algos:
            self.model = algos[self.method](**kwargs)
        else:
            raise ValueError(f"Invalid method '{self.method}'. Choose from {list(algos.keys())}.")

    def fit(self
            , X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series
            , sample_weight = None
            ):
        """Fits the model to the data."""
        self.model.fit(X=X, y=y, sample_weight=sample_weight)
        return self
    
    def predict(self, X: np.ndarray | pd.DataFrame):
        """
        bla bla bla
        """
        if self.method == "logistic_regression":
            return self.model.predict(X=X)
        
    def score(self, X: np.ndarray | pd.DataFrame):
        """
        bla bla bla
        
        X (np.ndarray | pd.DataFrame): Features or predictors.
        """
        if self.method == "logistic_regression":
            return self.model.predict_proba(X=X)
        