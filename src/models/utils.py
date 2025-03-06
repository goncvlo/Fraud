import mlflow

# set mlflow tracking
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.autolog(disable=True)

def get_run(
        experiment_names: list[str]
        , order_by_metric: bool
        , metric_name: str
        , num: int = 0
        ):
    """
    Get (best) algorithm and its best hyper-params.

    Args:
        experiment_names (list[str]): Search runs within given experiments.
        order_by_metric (bool): Whether to desc sort  the results based on metric_name.
        metric_name (str): Metric for which the results were evaluated on.
        num (int): Row number to return results - 0 for top-1.
    Returns:
        algorithm_name (str)
        hyper_params (dict)
    """

    if order_by_metric:
        runs_info = mlflow.search_runs(
            search_all_experiments=True
            , order_by=[f"metrics.{metric_name} DESC"]
            , experiment_names=experiment_names
            )
    else:
        runs_info = mlflow.search_runs(
            search_all_experiments=True
            , experiment_names=experiment_names
            )
    # remove parent id
    runs_info = runs_info[~runs_info['tags.mlflow.parentRunId'].isna()]

    # algorithm and hyper-params selection
    algorithm_name = runs_info['tags.mlflow.runName'][num]
    hyper_params = (
        runs_info[[col for col in runs_info.columns if 'params.' in col]]
        .iloc[num,:].dropna().to_dict()
        )
    hyper_params = {
        key.replace('params.', ''): (
            int(value) if value.isdigit() else
            float(value) if '.' in value else
            value
        )
        for key, value in hyper_params.items()
    }

    return algorithm_name, hyper_params
