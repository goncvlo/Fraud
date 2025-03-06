import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# supported algorithms
algos = {
    'logistic_regression': LogisticRegression
    , 'support_vector_machine': SVC
    , 'decision_tree': DecisionTreeClassifier
    , 'random_forest': RandomForestClassifier
    , 'xgboost': XGBClassifier
    }

class Classification:
    def __init__(self, algorithm: str, **kwargs):
        """
        Initializes the Classification model class.

        Parameters:
            algorithm (str): Algorithm to use.
            kwargs: Hyper-params for the specified algorithm.
        """
        
        # validate the algorithm and initialize it
        self.method = algorithm.lower()
        if self.method in algos:
            self.model = algos[self.method](**kwargs)
        else:
            raise ValueError(f"Invalid method '{self.method}'. Choose from {list(algos.keys())}.")

    def fit(self
            , X: np.ndarray | pd.DataFrame
            , y: np.ndarray | pd.Series
            , sample_weight = None
            ):
        """
        Feeds data to the model.
        
        Args:
            X (np.ndarray | pd.DataFrame): Features or predictors.
            y (np.ndarray | pd.Series): Target values.
        """
        
        self.model.fit(X=X, y=y, sample_weight=sample_weight)
        return self
    
    def predict(self, X: np.ndarray | pd.DataFrame):
        """
        Predicts target value for the given observations.

        Args:
            X (np.ndarray | pd.DataFrame): Features or predictors.
        Returns:
            y (np.ndarray | pd.Series): Predicted target values.
        """

        return self.model.predict(X=X)
        
    def score(self, X: np.ndarray | pd.DataFrame):
        """
        Computes probability score for the given observations.

        Args:
            X (np.ndarray | pd.DataFrame): Features or predictors.
        Returns:
            p (np.ndarray | pd.Series): Estimated class probabilities.
        """

        return self.model.predict_proba(X=X)
        