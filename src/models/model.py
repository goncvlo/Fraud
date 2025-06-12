import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import tensorflow as tf
from tensorflow import keras
    

class NeuralNetworkClassifier:
    def __init__(self, hyperparams: dict, input_size: int = 30):
        """
        Set algorithm hyperparameters and model.

        Args:
            hyperparameters (dict): Hyperparams for the specified algorithm.
            input_size (int): Number of (normalized) features.
        """
        self.hyperparams = hyperparams

        # build and assign model
        model = keras.Sequential()
        model.add(keras.Input(shape=(input_size,)))
        for i in range(self.hyperparams["n_layers"]):
            n_units = self.hyperparams[f"units_{i}"]
            activation = self.hyperparams[f"activation_{i}"]
            model.add(keras.layers.Dense(n_units, activation=activation))
        # output layer
        model.add(keras.layers.Dense(1, activation="sigmoid"))

        # model compilation
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.hyperparams["learning_rate"]),
            loss=self.hyperparams["loss"],
            metrics=[self.hyperparams["scoring_metric"]]
            #, verbose=0
            )
        self.model = model

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Model training."""
        history = self.model.fit(
            X.to_numpy(), y.to_numpy(),
            batch_size=self.hyperparams["batch_size"],
            epochs=self.hyperparams["epochs"],
            verbose=0
            )
        self.training_history = history
    
    def predict(self, X: pd.DataFrame):
        """Predicts class labels (0 or 1) for the given observations."""
        probas = self.model.predict(X.to_numpy())
        return (probas > 0.5).astype(int).flatten()

    def predict_proba(self, X: pd.DataFrame):
        """Computes class probabilities for the positive class (label 1)."""
        prob_1 = self.model.predict(X.to_numpy()).flatten()
        prob_0 = 1.0 - prob_1
        return np.stack([prob_0, prob_1], axis=1)


# supported algorithms
algorithms = {
    'LogisticRegression': LogisticRegression
    , 'SVC': SVC
    , 'DecisionTreeClassifier': DecisionTreeClassifier
    , 'RandomForestClassifier': RandomForestClassifier
    , 'GradientBoostingClassifier': GradientBoostingClassifier
    , 'MLPClassifier': MLPClassifier
    , 'XGBClassifier': XGBClassifier
    , 'LGBMClassifier': LGBMClassifier
    , "NeuralNetworkClassifier": NeuralNetworkClassifier
    }


class Classifier:
    def __init__(self, algorithm: str, **kwargs):
        """
        Set algorithm and model.

        Args:
            algorithm (str): Algorithm to use.
            kwargs (dict): Hyperparams for the specified algorithm.
        """
        
        # validate the algorithm and initialize it
        self.algorithm = algorithm
        if self.algorithm in algorithms:
            self.model = algorithms[self.algorithm](**kwargs)
        else:
            raise NotImplementedError(f"{algorithm} isn't supported. Select from {list(algorithms.keys())}.")

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Feeds data, predictors & target, into the algorithm."""
        self.model.fit(X=X, y=y, **kwargs)
    
    def predict(self, X: pd.DataFrame):
        """Predicts class labels (0 or 1) for the given observations."""
        return self.model.predict(X=X)
        
    def predict_proba(self, X: pd.DataFrame):
        """Computes class probabilities for the positive class (label 1)."""
        return self.model.predict_proba(X=X)
    