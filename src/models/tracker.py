import os
import pandas as pd
import time
from typing import Union
from datetime import datetime
import mlflow
import subprocess
import webbrowser
import optuna

from src.models.tuner import HyperParamSearch, LabelWeightSearch


def log_run(
        experiment_name: str,
        study: optuna.study.Study,
        tuner: Union[HyperParamSearch, LabelWeightSearch]
        ):

    with get_or_create_run(run_name=tuner.algorithm, experiment_name=experiment_name) as parent:
        today_date = datetime.now().strftime("%d%b%Y").upper()
        with mlflow.start_run(run_name=today_date, nested=True):

            if isinstance(tuner, HyperParamSearch):

                log_artifact(artifact=study.best_trial.params, artifact_name="params")
                log_artifact(artifact=study.best_trial.duration.total_seconds(), artifact_name="duration_secs")
                log_artifact(artifact=study.best_trial.number, artifact_name="trial_index")
                log_artifact(artifact=study.best_trial.value, artifact_name=tuner.scoring_metric)

                log_artifact(artifact=study.trials_dataframe(), artifact_name="bayes_search", artifact_path="stats")
                log_artifact(artifact=tuner.results, artifact_name="results")
                log_artifact(artifact=tuner.param_grid, artifact_name="param_grid")


def launch_mlflow():
    """Launch MLflow UI on localhost and open it in a browser."""
    subprocess.Popen(["mlflow", "ui"])
    time.sleep(2)
    webbrowser.open(f"http://127.0.0.1:5000")
    mlflow.autolog(disable=True)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")


def log_artifact(artifact: Union[pd.DataFrame, dict, float, int], artifact_name: str, artifact_path: str = "default"):
    """Save object locally, log it as an artifact, and delete it."""

    if isinstance(artifact, pd.DataFrame):
        artifact.to_csv(f"{artifact_name}.csv")
        mlflow.log_artifact(f"{artifact_name}.csv", artifact_path=artifact_path)
        os.remove(f"{artifact_name}.csv")
    
    if isinstance(artifact, dict):
        if artifact_name=="params":
            mlflow.log_params(params=artifact)
        else:
            mlflow.log_dict(dictionary=artifact, artifact_file=f"{artifact_name}.yaml")
    
    if isinstance(artifact, (float, int)):
        mlflow.log_metric(key=artifact_name, value=artifact)


def get_or_create_run(run_name: str, experiment_name: str):
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
        order_by=["start_time DESC"]
    )
    
    if runs:
        # resume existing run (will inherit correct experiment)
        existing_run_id = runs[0].info.run_id
        return mlflow.start_run(run_id=existing_run_id)
    else:
        # create new run in the selected experiment
        return mlflow.start_run(run_name=run_name)
