"""PyTorch Lightnint用簡易モジュール群."""
# default packages
import argparse
import typing as t

# third party packages
import mlflow.tracking as mlf_tracking
import pytorch_lightning.loggers.base as pl_base
import pytorch_lightning.utilities as pl_utilities


class MLFlowLogger(pl_base.LightningLoggerBase):
    def __init__(self) -> None:
        super().__init__()

    @property
    @pl_base.rank_zero_experiment
    def experiment(self) -> mlf_tracking.MlflowClient:
        pass

    @property
    def run_id(self) -> str:
        pass

    @property
    def experimnt_id(self) -> str:
        pass

    @pl_utilities.rank_zero_only
    def log_hyperparams(
        self, params: t.Union[t.Dict[str, t.Any], argparse.Namespace]
    ) -> None:
        pass

    @pl_utilities.rank_zero_only
    def log_metrics(
        self, metrics: t.Dict[str, float], step: t.Optional[int] = None
    ) -> None:
        pass

    @pl_utilities.rank_zero_only
    def finalize(self, status: str = "FINISHED") -> None:
        pass

    @property
    def svae_dir(self) -> t.Optional[str]:
        pass

    @property
    def name(self) -> str:
        pass

    @property
    def version(self) -> str:
        pass
