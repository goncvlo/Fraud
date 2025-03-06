import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SequentialFeatureSelector
import mlflow
import os
from src.models.classification import Classification

# set mlflow tracking
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.autolog(disable=True)

def grid_search(
        X: np.ndarray | pd.DataFrame
        , y: np.ndarray | pd.Series
        , hyper_params: dict
        , cv: None | int
        , scoring_metric: str
        ):
    """
    Logs grid search results, for multiple algorithms, into mlflow.

    Args:
        X (np.ndarray | pd.DataFrame): Features or predictors.
        y (np.ndarray | pd.Series): Target values.
        hyper_params (dict): Different hyper-params for each algorithm. 
        cv (None | int): CV Iterator - None for KFold, Int for StratKFold or other.
        scoring_metric (str): metric to be evaluated.
    """

    # set mlflow tracking
    mlflow.set_experiment(experiment_name='model_evaluation')
    with mlflow.start_run(
        run_name='algorithm_evaluation'
        , description='Evaluate the performance of different algorithms with simple hyper-params.'
        ):

        # set grid search for each algorithm
        for algo in hyper_params.keys():
            with mlflow.start_run(run_name=algo, nested=True):
                
                # fit algorithm using cross-validation
                clf = GridSearchCV(
                    estimator=Classification(algorithm=algo).model
                    , param_grid=hyper_params[algo]
                    , scoring=scoring_metric
                    , refit=True
                    , cv=cv
                    , return_train_score=True
                )
                clf.fit(X=X, y=y)

                # logging hyper-params
                mlflow.log_dict(
                    dictionary=hyper_params[algo]
                    , artifact_file="hyper_params.yml"
                    )
                # logging best params, model and score
                mlflow.log_params(params=clf.best_params_)
                signature = mlflow.models.infer_signature(
                    model_input = X
                    , model_output = y
                    )
                mlflow.sklearn.log_model(
                    sk_model=clf.best_estimator_
                    , artifact_path='model_instance'
                    , signature=signature
                    )
                mlflow.log_metric(key=scoring_metric, value=clf.best_score_)
                # logging CV results
                pd.DataFrame(clf.cv_results_).to_csv('cv_results_.csv')
                mlflow.log_artifact('cv_results_.csv')
                os.remove('cv_results_.csv')

                mlflow.end_run()
        mlflow.end_run()

def feature_selector(
        X: np.ndarray | pd.DataFrame
        , y: np.ndarray | pd.Series
        , algorithm: str
        , algorithm_params: dict
        , tol: float
        , cv: None | int
        , scoring_metric: str
        )->list:
    """
    Forward feature selection.

    Args:
        X (np.ndarray | pd.DataFrame): Features or predictors.
        y (np.ndarray | pd.Series): Target values.
        algorithm (str): Algorithm to be tested.
        algorithm_params (dict): Hyper-params for the selected algorithm.
        tol (float): Tolerance for the scoring_metric.
        cv (None | int): CV Iterator - None for KFold, Int for StratKFold or other.
        scoring_metric (str): metric to be evaluated.
    Returns:
        (list): Most important features whose contribution doesn't exceed tol.
    """
    
    # fit algorithm into feature selector
    model = Classification(algorithm=algorithm, **algorithm_params).model
    clf = SequentialFeatureSelector(
        estimator=model
        , n_features_to_select='auto'
        , tol=tol
        , direction='forward'
        , scoring=scoring_metric
        , cv=cv
    )
    clf.fit(X=X, y=y)

    return X.columns[clf.get_support()]
