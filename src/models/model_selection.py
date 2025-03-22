import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, TunedThresholdClassifierCV
from sklearn.feature_selection import chi2, f_classif, SequentialFeatureSelector
from src.models.classification import Classification
from typing import Optional

class GridSearch:
    def __init__(self, model_selection: dict):
        """Set CV settings, scoring metric and hyperparameters for grid search."""
        self.cross_validator = model_selection['cross_validator']
        self.scoring_metric = model_selection['scoring_metric']
        self.param_grid = model_selection['param_grid']
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

class FeatureSelection:
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        """Set predictors and target, X and y, respectively."""
        self.X = X
        self.y = y

    def categorical(self, features: list[str]):
        """
        Checks which categorical (ordinal) features are related with the target.

        Args:
            features (list[str]): categorical features to be tested.
        Returns:
            (list): Categorical features which are related with the target.
        """

        # target is categorical - assumption is, distinct target values is up to 5
        if self.y.nunique()<=5:
            chi2_stats, p_values = chi2(X=self.X[features], y=self.y)
        else:
            f_stats, p_values = f_classif(X=self.X[features], y=self.y)

        return features[p_values<0.05]

    def wrapper(self, clf: Classification, model_selection: dict):
        """
        Forward feature selection.

        Args:
            clf (Classification): Algorithm on which to perform selection.
            model_selection (dict): Config dictionary of model selection.
        Returns:
            (list): Top features whose contribution doesn't exceed tol.
        """
    
        # fit algorithm into feature selector
        clf = SequentialFeatureSelector(
            estimator=clf.model
            , n_features_to_select='auto'
            , tol=model_selection['tolerance']
            , direction='forward'
            , scoring=model_selection['scoring_metric']
            , cv=model_selection['cross_validator']
        )
        clf.fit(X=self.X, y=self.y)

        return self.X.columns[clf.get_support()]

class ClassificationThreshold():
    def __init__(self, model_selection: dict):
        """Set CV settings and scoring metric."""
        self.cross_validator = model_selection['cross_validator']
        self.scoring_metric = model_selection['scoring_metric']
    
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