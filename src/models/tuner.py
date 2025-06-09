import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, TunedThresholdClassifierCV
from sklearn.utils.class_weight import compute_sample_weight
import optuna

from src.models.model import Classifier
from src.models.evaluation import Evaluation


class HyperParamSearch:
    def __init__(self, config: dict, algorithm: str):
        """Set CV settings, scoring metric and hyperparameters space."""
        self.cross_validator = config['cross_validator']
        self.scoring_metric = config['scoring_metric']
        self.algorithm = algorithm
        self.param_grid = config['param_grid'][self.algorithm]

    def fit(
            self
            , X: pd.DataFrame, y: pd.Series # training set
            , trial: optuna.trial.Trial
            ) -> float:
        
        # set suggested hyper-parameters
        hyperparams = self._suggest_hyperparams(trial)
        clf = Classifier(algorithm=self.algorithm, **hyperparams).model
    
        # strat kfold cv on training data
        cv_scores = cross_val_score(
            estimator=clf,
            X=X, y=y,
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
    

class LabelWeightSearch:
    def __init__(self, config: dict, estimator: Classifier):
        self.cross_validator = config["cross_validator"]
        self.model = estimator
        self.scoring_metric = config["scoring_metric"]

    def fit(self, X: pd.DataFrame, y: pd.Series, trial: optuna.trial.Trial):

        w0 = trial.suggest_int("weight_0", 1.0, 100.0)
        w1 = trial.suggest_int("weight_1", 1.0, 100.0)
        
        skf = StratifiedKFold(n_splits=self.cross_validator, shuffle=True, random_state=42)
        scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            sample_weight = compute_sample_weight(
                class_weight={0: w0, 1: w1}, y=y_train
                )
            self.model.fit(X_train, y_train, sample_weight=sample_weight)

            score = Evaluation(clf=self.model)
            score = score.fit(metric=self.scoring_metric, test=(X_val, y_val))
            scores.append(score)
        
        return np.mean(scores)
    

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
