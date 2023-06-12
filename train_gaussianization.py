import os

import torch

from sacred import Experiment
from sacred.observers import MongoObserver

from gaussianization.utils.sacred.sacred_helper import mark_time_gauged, debug, fake_params, get_experiment_dir, \
    apply_cr_to_lines
from gaussianization import Gaussianization, GaussianizationHParams

ex_name = os.path.basename(__file__)[:-3]
ex = Experiment(ex_name, additional_cli_options=[mark_time_gauged, debug])
ex.observers.append(MongoObserver())

default_device = "gpu" if torch.cuda.is_available() else "cpu"
config = GaussianizationHParams.defaults() | dict(accelerator=default_device, num_threads=1)
ex.add_config(config)
ex.captured_out_filter = apply_cr_to_lines

# These keys do not affect the outcome. This is read by launcher script
ignore_keys = {"accelerator", "num_threads", "max_epochs", "num_workers", "devices", "check_val_every_n_epoch"}


def ensure_valid_config(config: dict):
    GaussianizationHParams(**{
        key: value
        for key, value in config.items()
        if key in GaussianizationHParams.parameters()
    })


@ex.automain
@fake_params(set(config) | set(GaussianizationHParams.parameters()))
def main(num_threads, **hparams):
    ensure_valid_config(hparams)
    if num_threads is not None:
        torch.set_num_threads(num_threads)

    log_dir = get_experiment_dir(ex, False)
    model = Gaussianization(hparams, log_dir)
    validation_loss = model.fit()
    model.trainer.save_checkpoint(log_dir / "checkpoints/manual-last.ckpt")
    return validation_loss
