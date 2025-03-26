import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, TunedThresholdClassifierCV
from src.models.classification import Classification

class GridSearch:
    def __init__(self, config: dict):
        """Set CV settings, scoring metric and hyperparameters for grid search."""
        self.cross_validator = config['cross_validator']
        self.scoring_metric = config['scoring_metric']
        self.param_grid = config['model_selection']['param_grid']
        self.best_algorithm = None
        self.best_hyperparams = None
        self.best_score = -np.inf

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Performs grid search for multiple algorithms."""
        
        results = {}
        # perform grid search for each algorithm
        for algorithm, param_grid in self.param_grid.items():

            clf = GridSearchCV(
                estimator=Classification(algorithm=algorithm).model
                , param_grid=param_grid
                , scoring=self.scoring_metric
                , refit=True
                , cv=self.cross_validator
                , return_train_score=True
                )
            clf.fit(X, y)

            # check if current is best algorithm
            if clf.best_score_ > self.best_score:
                self.best_score = clf.best_score_
                self.best_algorithm = algorithm
                self.best_hyperparams = clf.best_params_
            
            results[algorithm] = {}
            for key, value in zip(
                ['best_hyperparams', 'hyperparams', 'best_score', 'cv_results']
                , [clf.best_params_, self.param_grid[algorithm], clf.best_score_, clf.cv_results_]
                ):
                results[algorithm][key] = value
        
        self.results = results

class ClassificationThreshold():
    def __init__(self, config: dict):
        """Set CV settings and scoring metric."""
        self.cross_validator = config['cross_validator']
        self.scoring_metric = config['scoring_metric']
    
    def fit(self, clf: Classification, X: pd.DataFrame, y: pd.Series):
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
        return tuned_clf.best_threshold_
    