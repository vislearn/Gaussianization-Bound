from inspect import Signature, signature, Parameter
from pathlib import Path
from pickle import dump
from typing import Union, Iterable

import matplotlib.pyplot as plt
from incense import ExperimentLoader
from incense.experiment import Experiment as IExperiment
from pandas import DataFrame
from pyrsistent import PMap, PList, PVector
# from pytorch_lightning.loggers import Logger
# from pytorch_lightning.utilities import rank_zero_only
from sacred import Experiment as SExperiment, cli_option
from sacred.utils import PathType


@cli_option("-t", "--mark-time-gauged", is_flag=True)
def mark_time_gauged(args, run):
    """
    Mark this run as being executed without other processes competing for resources.
    """
    run.info["time-gauged"] = args


@cli_option("-d", "--connect-debug", is_flag=True)
def debug(args, run):
    """
    Launch the remote PyCharm debugger.
    """
    import pydevd_pycharm
    pydevd_pycharm.settrace('localhost', port=52858, stdoutToServer=True, stderrToServer=True, suspend=True)


def apply_cr_to_lines(text: str) -> str:
    """
    Output filter to remove text before last carriage return in each line.
    """
    lines = []
    for line in text.split("\n"):
        try:
            last_cr = line.rindex("\r")
            line = line[last_cr + 1:]
        except ValueError:
            pass
        lines.append(line)
    return "\n".join(lines)


def get_experiment_dir(ex: Union[SExperiment, IExperiment], create=True) -> Union[Path, None]:
    if isinstance(ex, SExperiment):
        # noinspection PyProtectedMember
        ex_id = ex.current_run._id
        ex_name = ex.path
    elif ex is None:
        print("No experiment")
        return None
    else:
        ex_id = ex.id
        ex_name = ex.experiment.name
    if ex_id is None:
        print("No experiment id")
        return None
    else:
        info_file = Path.home() / "experiment-storage.txt"
        if info_file.is_file():
            with open(info_file) as file:
                ex_folder = Path(file.read().strip()) / ex_name
        else:
            ex_folder = Path.home() / "Documents/Experiment-Storage" / ex_name
        if not ex_folder.is_dir():
            raise FileNotFoundError(f"No folder for experiment with path {ex_folder} was found.")
        path = ex_folder / str(ex_id)
        if create and not path.is_dir():
            path.mkdir()
        return path


def store_file(ex: SExperiment, path: PathType):
    # Instead of add_artifact: Don't store this in the database which gets huge
    if "stored_files" not in ex.current_run.info:
        ex.current_run.info["stored_files"] = []
    ex.current_run.info["stored_files"].append(str(path))


def savefig(ex: SExperiment, name: str):
    path = get_experiment_dir(ex)
    if path is not None:
        plt.tight_layout()
        out_path = path / name
        plt.savefig(out_path, bbox_inches="tight")
        if name.endswith(".png"):
            # Only now it's worth because it's shown in Omniboard
            ex.add_artifact(out_path)
        store_file(ex, out_path)


def store_models(ex: SExperiment, config):
    store_pickle(ex, config.stored_states, "model-states.p", False)


def store_loggers(ex: SExperiment, loggers):
    store_pickle(ex, loggers, "loggers.p", True)


def store_pickle(ex: SExperiment, data, file_name, store_to_db=False):
    path = get_experiment_dir(ex)
    if path is not None:
        file_path = path / file_name
        with open(file_path, "wb") as file:
            dump(data, file)
        store_file(ex, file_path)
        # Maybe drop this, pollutes database
        if store_to_db:
            ex.add_artifact(file_path)


def fake_params(param_names):
    def parameterized(function):
        parameters = []
        original_signature = signature(function)
        explicit_names = set()
        for name, original_parameter in original_signature.parameters.items():
            assert isinstance(original_parameter, Parameter)
            if original_parameter.kind == Parameter.VAR_KEYWORD:
                for key in param_names:
                    if key not in explicit_names:
                        parameters.append(Parameter(key, Parameter.POSITIONAL_OR_KEYWORD))
            else:
                explicit_names.add(name)
                parameters.append(original_parameter)
        function.__signature__ = Signature(parameters)
        return function

    return parameterized


def intersection(list_a: Iterable, list_b: Iterable):
    return [a for a in list_a if a in list_b]


def load_experiments(loader, experiment_name, conditions=None,
                     status="COMPLETED"):
    if conditions is None:
        conditions = {}
    conditions["experiment.name"] = experiment_name
    if status is not None:
        conditions["status"] = status
    experiments = list(loader.find(conditions))

    def recursive_walk(prefix, value):
        if isinstance(value, PMap):
            value = dict(value)
        if isinstance(value, (PList, PVector)):
            value = list(value)
        if isinstance(value, dict):
            for key, sub_value in value.items():
                new_prefix = key if prefix is None else prefix + "." + key
                yield from recursive_walk(new_prefix, sub_value)
        else:
            yield prefix, value

    data = []
    for ex in experiments:
        data.append({
            "name": ex.experiment.name,
            **dict(recursive_walk(None, ex.config)),
            "obj": ex,
            "id": ex.id,
            "result": ex.result,
        })

    df = DataFrame(data).set_index("id")
    sort_keys = intersection(
        ["name", "result", "id"],
        df.columns
    )
    if len(sort_keys) > 0:
        df.sort_values(sort_keys)
    return df


"""
class SacredLogger(Logger):
    def __init__(self, experiment, prefix=""):
        super().__init__()
        self.experiment = experiment
        self.prefix = prefix

    @property
    def name(self):
        return "sacred_logger"

    @property
    def version(self):
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        for k, v in metrics.items():
            self.experiment.log_scalar(f"{self.prefix}{k}", v, step)

    def log_hyperparams(self, params, *args, **kwargs):
        pass
"""

if __name__ == '__main__':
    print("Cleaning up interrupted runs that pretend still to run.")
    loader = ExperimentLoader()
    loader._database.runs.update_many({
        "status": "RUNNING"
    }, {
        "$set": {
            "status": "INTERRUPTED"
        }
    })
