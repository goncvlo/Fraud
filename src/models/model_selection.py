import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import chi2, f_classif, SequentialFeatureSelector
import mlflow
import os
from src.models.classification import Classification
from typing import Optional

# set mlflow tracking
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.autolog(disable=True)

class GridSearch:
    def __init__(self, hyper_params: dict, cv: Optional[int], scoring_metric: str):
        """
        Set grid search hyperparameters, CV settings, and scoring metric.

        Args:
            hyper_params (dict): Hyperparameters for each algorithm. 
            cv (Optional[int]): CV iterator (None for KFold, int for StratKFold or other).
            scoring_metric (str): Metric to evaluate.
        """
        self.hyper_params = hyper_params
        self.cv = cv
        self.scoring_metric = scoring_metric
        self.best_algorithm = None
        self.best_hyperparams = None
        self.best_score = -np.inf

    def _log_mlflow(self, clf: GridSearchCV, X: pd.DataFrame, y: pd.Series):
        """Logs the best model, hyperparameters, and CV results to MLflow."""
        # log hyperparameters as an artifact
        mlflow.log_dict(
            dictionary=self.hyper_params[clf.best_estimator_.__class__.__name__],
            artifact_file="hyper_params.yml"
        )
        # log best parameters, model, and score
        mlflow.log_params(params=clf.best_params_)
        signature = mlflow.models.infer_signature(X, y)
        mlflow.sklearn.log_model(clf.best_estimator_, artifact_path='best_model', signature=signature)
        mlflow.log_metric(self.scoring_metric, clf.best_score_)
        # log CV results as .csv and remove temp file
        pd.DataFrame(clf.cv_results_).to_csv('cv_results_.csv')
        mlflow.log_artifact('cv_results_.csv')
        os.remove('cv_results_.csv')

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Performs grid search for multiple algorithms and logs into MLflow.

        Args:
            X (pd.DataFrame): Features or predictors.
            y (pd.Series): Target values.
        """
        # set MLflow experiment and run name based on hyperparameters
        run_name = 'parameters' if len(self.hyper_params) == 1 else 'algorithms'
        mlflow.set_experiment(experiment_name='model_evaluation')

        # start parent MLflow run
        with mlflow.start_run(run_name=run_name):
            # perform grid search for each algorithm
            for algo, params in self.hyper_params.items():
                with mlflow.start_run(run_name=algo, nested=True):
                    clf = GridSearchCV(
                        estimator=Classification(algorithm=algo).model,
                        param_grid=params,
                        scoring=self.scoring_metric,
                        refit=True,
                        cv=self.cv,
                        return_train_score=True
                    )
                    clf.fit(X, y)

                    # log the results to MLflow
                    self._log_mlflow(clf, X, y)

                    # get best algo and its hyperparams
                    if clf.best_score_ > self.best_score:
                        self.best_score = clf.best_score_
                        self.best_algorithm = algo
                        self.best_hyperparams = clf.best_params_

class FeatureSelection:
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        """
        Set predictors and target, X and y, respect.

        Args:
            X (np.ndarray | pd.DataFrame): Features or predictors.
            y (np.ndarray | pd.Series): Target values.
        """
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

    def wrapper(self, algorithm: str, algorithm_params: dict, cv: Optional[int], scoring_metric: str, tolerance: float):
        """
        Forward feature selection.

        Args:
            algorithm (str): Algorithm on which to perform selection.
            algorithm_params (dict): Hyperparameters of the algorithm. 
            cv (Optional[int]): CV iterator (None for KFold, int for StratKFold or other).
            scoring_metric (str): Metric to evaluate.
            tol (float): Tolerance for the scoring_metric.
        Returns:
            (list): Top features whose contribution doesn't exceed tol.
        """
    
        # fit algorithm into feature selector
        clf = SequentialFeatureSelector(
            estimator=Classification(algorithm=algorithm, **algorithm_params).model
            , n_features_to_select='auto'
            , tol=tolerance
            , direction='forward'
            , scoring=scoring_metric
            , cv=cv
        )
        clf.fit(X=self.X, y=self.y)

        return self.X.columns[clf.get_support()]
