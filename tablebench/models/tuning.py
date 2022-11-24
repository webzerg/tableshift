from dataclasses import dataclass
from typing import Callable, Optional

from ray import tune, air

from tablebench.configs.hparams import search_space
from tablebench.core import TabularDataset
from tablebench.models import get_model_config, get_estimator
from tablebench.models.training import train


@dataclass
class TuneConfig:
    num_samples: int = 1
    tune_metric_name: str = "metric"
    tune_metric_higher_is_better: bool = True
    report_split: str = "ood_test"

    @property
    def mode(self):
        return "max" if self.tune_metric_higher_is_better else "min"


def run_tuner(train_fn: Callable, param_space: dict,
              tune_config: TuneConfig):
    """Run an air Tuner using the specified train_fn and params."""
    tuner = tune.Tuner(
        train_fn,
        param_space=param_space,
        tune_config=tune.tune_config.TuneConfig(
            num_samples=tune_config.num_samples),
        run_config=air.RunConfig(local_dir="./ray-results",
                                 name="test_experiment"))

    results = tuner.fit()
    return results


def run_tuning_experiment(model: str, dset: TabularDataset, device: str,
                          tune_config: Optional[TuneConfig]):
    def _train_fn(run_config=None):
        # Get the default configs
        config = get_model_config(model, dset)
        if run_config:
            # Override the defaults with run_config, if provided.
            config.update(run_config)
        estimator = get_estimator(model, **config)
        train(estimator, dset, device=device, config=config,
              tune_report_split=tune_config.report_split)

    if not tune_config:
        # TODO(jpgard): return a metrics dict.
        _train_fn()
    else:
        results = run_tuner(_train_fn, search_space[model], tune_config)

        best_result = results.get_best_result(
            tune_config.tune_metric_name, tune_config.mode)

        print("Best trial config: {}".format(best_result.config))
        print("Best trial final {}: {}".format(
            tune_config.tune_metric_name,
            best_result.metrics[tune_config.tune_metric_name]))
        return results
