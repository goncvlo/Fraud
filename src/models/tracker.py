import os
import subprocess
import time
import webbrowser
from datetime import datetime
from typing import Union

import mlflow
import pandas as pd
from lightgbm import LGBMModel
from xgboost import XGBModel
from sklearn.base import BaseEstimator

from src.models.tuner import HyperParamSearch, LabelWeightSearch
from src.models.model import Classifier
from src.models.evaluator import Evaluation


class Logging:

    def __init__(self, experiment_name: str, run_name: str):
        self.experiment_name = experiment_name
        self.run_name = run_name
        # launch MLflow UI
        launch_mlflow()

    def log_run(
        self,
        clf: Classifier,
        train: tuple[pd.DataFrame, pd.Series],
        test: tuple[pd.DataFrame, pd.Series],
    ):

        with get_run(self.experiment_name, self.run_name) as parent:
            now = datetime.now().strftime("%d%b%Y_%H%M").upper()
            with mlflow.start_run(run_name=f"TEST_{now}", nested=True):
                
                # get test and training sets performance
                eval = Evaluation(clf=clf, threshold=0.5)
                metrics_test = eval.fit(train=train, test=test)
                
                # logging
                mlflow.log_params(params=clf.model.get_params())
                self._log_model(clf=clf, name=clf.algorithm, input_sample=train[0].head())
                self._log_df(metrics_test, "metrics_test", "stats")
                mlflow.log_metrics(metrics_test[metrics_test.index=="test"].iloc[0, :].to_dict())

                for df, name in zip(
                    [pd.concat(train, axis=1), pd.concat(test, axis=1)],
                    ["train", "test"]
                    ):
                    dataset = mlflow.data.from_pandas(df=df, targets="label", name=name)
                    mlflow.log_input(dataset=dataset, context=name, tags={"dataset": name})

    def log_tuner(self, tuner: HyperParamSearch):

        with get_run(self.experiment_name, self.run_name) as parent:
            now = datetime.now().strftime("%d%b%Y_%H%M").upper()
            with mlflow.start_run(run_name=f"SEARCH_{now}", nested=True):

                # best trial details - scoring_metric direction is maximize
                best_trial = (
                    tuner.trials_runs
                    .sort_values(by=["value"], ascending=[False])
                    .iloc[0,:]
                    )
                metrics = {
                    "duration_secs": best_trial.duration.total_seconds(),
                    "trial_index": best_trial.number,
                    tuner.metric: best_trial.value,
                }
                params = (
                    best_trial
                    .filter(like="params_")
                    .rename(lambda c: c.replace("params_", ""))
                    .to_dict()
                )

                # logging
                mlflow.log_metrics(metrics=metrics)
                mlflow.log_params(params=params)
                mlflow.log_dict(dictionary=tuner.param_grid, artifact_file="param_grid.yaml")
                self._log_df(tuner.trials_runs, "trials_runs", "stats")

                # convert trials results dictionary to dataframe
                metrics_val = []
                for split_id, split_data in tuner.trials_metrics.items():
                    for dataset, metric in split_data.items():
                        row = {"trial_index": split_id, "metric": dataset}
                        row.update(metric)
                        metrics_val.append(row)
                metrics_val = pd.DataFrame(metrics_val)
                self._log_df(metrics_val, "metrics_val", "stats")

    def _log_df(self, df: pd.DataFrame, name: str, path: str):

        df.to_csv(f"{name}.csv")
        mlflow.log_artifact(f"{name}.csv", artifact_path=path)
        os.remove(f"{name}.csv")

    def _log_model(
        self,
        clf: Classifier,
        name: str,
        input_sample: pd.DataFrame,
    ):

        output_sample = clf.predict(input_sample)
        input_sample = input_sample.astype(
            {
                col: "float"
                for col in input_sample.select_dtypes(include="int").columns
            }
        )
        signature = mlflow.models.signature.infer_signature(
            input_sample, output_sample
        )

        if isinstance(clf.model, XGBModel):
            mlflow.xgboost.log_model(
                clf.model, name=name, input_example=input_sample, signature=signature,
                model_format="json",
            )

        elif isinstance(clf.model, LGBMModel):
            mlflow.lightgbm.log_model(
                clf.model, name=name, input_example=input_sample, signature=signature
            )
        
        elif isinstance(clf.model, BaseEstimator):
            mlflow.sklearn.log_model(
                clf.model, name=name, input_example=input_sample, signature=signature
                )

        else:
            raise NotImplementedError(
                f"Logging not implemented for model type: {type(clf.model)}"
            )


def launch_mlflow():
    """Launch MLflow UI on localhost and open it in a browser."""
    subprocess.Popen(["mlflow", "ui"])
    time.sleep(2)
    webbrowser.open(f"http://127.0.0.1:5000")
    mlflow.autolog(disable=True)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")


def get_run(experiment_name: str, run_name: str):
    client = mlflow.tracking.MlflowClient()

    # get or create experiment
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    # set the experiment context for mlflow
    mlflow.set_experiment(experiment_name)

    # search for run by name
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["start_time DESC"],
    )

    if runs:
        # resume existing run (will inherit correct experiment)
        existing_run_id = runs[0].info.run_id
        return mlflow.start_run(run_id=existing_run_id)
    else:
        # create new run in the selected experiment
        return mlflow.start_run(run_name=run_name)
    