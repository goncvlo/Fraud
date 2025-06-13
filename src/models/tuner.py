import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, TunedThresholdClassifierCV
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.base import clone
import optuna
import tensorflow as tf
from tensorflow import keras

from src.models.model import Classifier
from src.models.evaluation import Evaluation


class HyperParamSearch:
    def __init__(self, config: dict, algorithm: str):
        """Set CV settings, scoring metric and hyperparameters space."""
        self.cross_validator = config["cross_validator"]
        self.scoring_metric = config["scoring_metric"]
        self.algorithm = algorithm
        self.param_grid = config["param_grid"][self.algorithm]

    def fit(
            self
            , X: pd.DataFrame, y: pd.Series # training set
            , trial: optuna.trial.Trial
            , X_val: pd.DataFrame = None, y_val: pd.Series = None
            ) -> float:
        
        # set suggested hyper-parameters
        hyperparams = self._suggest_hyperparams(trial)
        clf = Classifier(algorithm=self.algorithm, **hyperparams)

        # holdout validation for algorithm comparasion
        if X_val is not None and y_val is not None:
            clf.fit(X, y)
            scorer = Evaluation(clf=clf, threshold=0.5)
            # LOG ALL METRICS WHEN SETTING EXPERIMENT TRACKER !!
            score = scorer.fit(metric=self.scoring_metric, validation=(X_val, y_val))
            return score

        # strat kfold cv on training data
        else:
            cv_scores = cross_val_score(
                estimator=clf.model,
                X=X, y=y,
                scoring=self.scoring_metric,
                cv=self.cross_validator
                )
            return np.mean(cv_scores)
    
    def _suggest_hyperparams(self, trial: optuna.trial.Trial) -> dict:
        tunable_params = {}
        tunable_config = self.param_grid["tunable"]
        
        for hp, bounds in tunable_config.items():
            if hp == "hidden_layer_sizes":
                n_layers_bounds = bounds["n_layers"]
                n_units_bounds = bounds["n_units"]

                n_layers = trial.suggest_int("n_layers", n_layers_bounds[0], n_layers_bounds[1])
                hidden_layer_sizes = tuple(
                    trial.suggest_int(f"layer_{i}_units", n_units_bounds[0], n_units_bounds[1])
                    for i in range(n_layers)
                )
                tunable_params[hp] = hidden_layer_sizes

            elif hp == "n_layers":
                tunable_params[hp] = trial.suggest_int(hp, bounds[0], bounds[1])
                for i in range(tunable_params[hp]):
                    tunable_params[f"units_{i}"] = trial.suggest_int(f"units_{i}", tunable_config["n_units"][0], tunable_config["n_units"][1])
                    tunable_params[f"activation_{i}"] = trial.suggest_categorical(f"activation_{i}", tunable_config["activation_function"])

            elif hp in ["n_units", "activation_function"]:
                pass
            elif hp in self.param_grid["float_params"]:
                tunable_params[hp] = trial.suggest_float(hp, bounds[0], bounds[1])
            elif hp in self.param_grid["categ_params"]:
                tunable_params[hp] = trial.suggest_categorical(hp, bounds)
            else:
                tunable_params[hp] = trial.suggest_int(hp, bounds[0], bounds[1])

        return {**tunable_params, **self.param_grid["fixed"]}
    

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
            model = clone(self.model) # avoid retaining internal state after fitted > fresh estimator
            model.fit(X_train, y_train, sample_weight=sample_weight)

            score = Evaluation(clf=model)
            score = score.fit(metric=self.scoring_metric, test=(X_val, y_val))
            scores.append(score)
        
        return np.mean(scores)
    

class ClassifierThreshold():
    def __init__(self, config: dict):
        """Set CV settings and scoring metric."""
        self.cross_validator = config["cross_validator"]
        self.scoring_metric = config["scoring_metric"]
    
    def fit(self, clf: Classifier, X: pd.DataFrame, y: pd.Series):
        """Fits TunedThresholdClassifierCV to get best threshold."""

        tuned_clf = TunedThresholdClassifierCV(
            estimator=clf.model
            , scoring=self.scoring_metric
            , response_method="predict_proba"
            , cv=self.cross_validator
            , refit=True
            , random_state=42
            , store_cv_results=True
        )
        tuned_clf.fit(X=X, y=y)
        
        self.model = tuned_clf
        self.best_threshold = tuned_clf.best_threshold_
