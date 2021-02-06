"""PyTorch Lightnint用簡易モジュール群."""
# default packages
import argparse
import logging
import re
import time
import typing as t

# third party packages
import mlflow
import mlflow.tracking as mlf_tracking
import pytorch_lightning.loggers.base as pl_base

# logger
_logger = logging.getLogger(__name__)


class MLFlowLogger(pl_base.LightningLoggerBase):
    def __init__(
        self,
        tracking_uri: str,
        experiment_name: str = "default",
        artifact_location: t.Optional[str] = None,
        run_name: t.Optional[str] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self._tracking_uri = tracking_uri
        self._experiment_name = experiment_name
        self._artifact_location = artifact_location
        self._run_name = run_name
        self._prefix = prefix
        self._client = mlf_tracking.MlflowClient(tracking_uri)

        run = mlflow.active_run()
        if run is None:
            expt = self._client.get_experiment_by_name(self._experiment_name)
            if expt is None:
                _logger.warning(
                    f"Experiment with name {self._experiment_name} not found."
                    "Creating it."
                )
                expt = self._client.create_experiment(
                    name=self._experiment_name,
                    artifact_location=self._artifact_location,
                )
            run = mlflow.start_run(
                run_id=None,
                experiment_id=expt,
                run_name=self._run_name,
            )

        run = mlflow.active_run()
        self._run_id = run.info.run_id
        self._experiment_id = run.info.experiment_id

    @property
    def experiment(self) -> mlf_tracking.MlflowClient:
        return self._client

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def experiment_id(self) -> str:
        return self._experiment_id

    def log_hyperparams(
        self, params: t.Union[t.Dict[str, t.Any], argparse.Namespace]
    ) -> None:
        params = self._convert_params(params)
        params = self._flatten_dict(params)
        for k, v in params.items():
            self._client.log_param(self.run_id, k, v)

    def log_metrics(
        self, metrics: t.Dict[str, float], step: t.Optional[int] = None
    ) -> None:
        metrics = self._add_prefix(metrics)
        timestamp_ms = int(time.time() * 1000)
        for k, v in metrics.items():
            if isinstance(v, str):
                _logger.warning(f"Discarding metric with string value {k}={v}.")
                continue

            new_k = re.sub("[^a-zA-Z0-9_/. -]+", "", k)
            if k != new_k:
                _logger.warning(
                    "MLFlow only allows '_', '/', '.' and ' ' special characters"
                    " in metric name."
                    f" Replacing {k} with {new_k}."
                )
                k = new_k

            self._client.log_metric(self.run_id, k, v, timestamp_ms, step)

    def finalize(self, status: str = "FINISHED") -> None:
        super().finalize(status)

    @property
    def svae_dir(self) -> t.Optional[str]:
        LOCAL_FILE_URI_PREFIX = "file:"
        if self._tracking_uri.startswith(LOCAL_FILE_URI_PREFIX):
            return self._tracking_uri.lstrip(LOCAL_FILE_URI_PREFIX)

        return None

    @property
    def name(self) -> str:
        return self.experiment_id

    @property
    def version(self) -> str:
        return self.run_id
