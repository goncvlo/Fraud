import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
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
        , cv: None | int = 5 # None for KFold, Int for StratKFold or iterable
        , scoring_metric: str = 'accuracy' # metric to be evaluated
        ):

    # set mlflow tracking
    mlflow.set_experiment(experiment_name='model_evaluation')
    with mlflow.start_run(run_name='algorithm_evaluation'):

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
                # logging params, model and score
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