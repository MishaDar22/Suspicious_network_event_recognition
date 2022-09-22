import typing as tp
import mlflow


def get_or_create_experiment(name: str):
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        mlflow.create_experiment(name)
        return mlflow.get_experiment_by_name(name)
    return experiment


def _eid(name: str) -> tp.Optional[str]:
    return get_or_create_experiment(name).experiment_id


def run_experiment(exp_name: str, algorithm_name: str, params: dict, metrics: dict, features: tp.List[str],
                   template_for_run_name: str = "Top {} features") -> None:

    with mlflow.start_run(experiment_id=_eid(exp_name),
                          run_name=template_for_run_name.format(len(features))):
        tags = dict()
        tags["algorithm"] = algorithm_name
        mlflow.set_tags(tags)

        # TRACK PARAMS
        mlflow.log_params(params)

        # TRACK METRICS
        mlflow.log_metrics(metrics)


