import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
import optuna
from sklearn.model_selection import TunedThresholdClassifierCV

from src.models.model import Classifier


class BayesianSearch:
    def __init__(self, config: dict, algorithm: str):
        """Set CV settings, scoring metric and hyperparameters space."""
        self.cross_validator = config['cross_validator']
        self.scoring_metric = config['scoring_metric']
        self.algorithm = algorithm
        self.param_grid = config['param_grid'][self.algorithm]

    def fit(
            self
            , train: dict[str, pd.DataFrame]
            , trial: optuna.trial.Trial
            ) -> float:
        
        # set suggested hyper-parameters
        hyperparams = self._suggest_hyperparams(trial)
        clf = Classifier(algorithm=self.algorithm, **hyperparams).model
    
        # strat kfold cv on training data
        cv_scores = cross_val_score(
            estimator=clf,
            X=train['X'], y=train['y'],
            scoring=self.scoring_metric,
            cv=self.cross_validator
            )
        
        return np.mean(cv_scores)
    
    def _suggest_hyperparams(self, trial: optuna.trial.Trial) -> dict:
        """Suggest hyperparameters based on the config, using trial."""
        tunable_params = {
            hp: (
                trial.suggest_float(name=hp, low=bounds[0], high=bounds[1])
                if hp in self.param_grid['float_params']
                else trial.suggest_int(name=hp, low=bounds[0], high=bounds[1])
            )
            for hp, bounds in self.param_grid['tunable'].items()
        }
        return {**tunable_params, **self.param_grid['fixed']}
    

class ClassifierThreshold():
    def __init__(self, config: dict):
        """Set CV settings and scoring metric."""
        self.cross_validator = config['cross_validator']
        self.scoring_metric = config['scoring_metric']
    
    def fit(self, clf: Classifier, X: pd.DataFrame, y: pd.Series):
        """Fits TunedThresholdClassifierCV to get best threshold."""

        tuned_clf = TunedThresholdClassifierCV(
            estimator=clf.model
            , scoring=self.scoring_metric
            , response_method='predict_proba'
            , cv=self.cross_validator
            , refit=True
            , random_state=123
            , store_cv_results=True
        )
        tuned_clf.fit(X=X, y=y)
        
        self.model = tuned_clf
        self.best_threshold = tuned_clf.best_threshold_
