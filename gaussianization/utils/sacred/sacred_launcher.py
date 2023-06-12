import json
import os
import subprocess
from ast import literal_eval
from dataclasses import dataclass
from subprocess import Popen
from argparse import ArgumentError
from collections import namedtuple, Counter
from concurrent.futures import as_completed
from concurrent.futures.thread import ThreadPoolExecutor
from importlib.util import spec_from_file_location, module_from_spec
from itertools import product
from math import log10
from os import PathLike
from pathlib import Path
from typing import Union, Dict, List
from datetime import timedelta
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from incense import ExperimentLoader
from sacred import Experiment
from tqdm import tqdm
from yaml import full_load

ConfigWithCount = namedtuple("ConfigWithCount", ["config", "count"])

@dataclass
class RunResult:
    config: Dict
    return_code: int|str
    stdout: bytes
    stderr: bytes

    @property
    def ex_id(self):
        try:
            stderr_unicode: str = self.stderr.decode()
            offset = stderr_unicode.index("Started run with ID \"")
            id_end = stderr_unicode.index("\"", offset)
            return int(stderr_unicode[offset:id_end])
        except ValueError:
            return None

    @property
    def result(self):
        try:
            stderr_unicode: str = self.stderr.decode()
            offset = stderr_unicode.index("Result: ")
            id_end = stderr_unicode.index("\n", offset)
            return literal_eval(stderr_unicode[offset:id_end])
        except:
            return None


def send_bot_message(message: str):
    try:
        with open(Path.home() / "Documents/Development/Telegram/iterative_news_bot.json", "r") as file:
            bot_info = json.load(file)
        url = f"https://api.telegram.org/bot{bot_info['token']}/sendMessage"
        for chat_id in bot_info["chats"]:
            params = {
                "chat_id": chat_id,
                "text": message
            }
            request = Request(url, urlencode(params).encode())
            urlopen(request).read().decode()
    except FileNotFoundError:
        pass


class ExperimentLauncher:
    def __init__(self, script_path: Union[PathLike, str], exp_name: str = None):
        if isinstance(script_path, str):
            script_path = Path(script_path)
        self.script_path = script_path
        if exp_name is None:
            exp_name = script_path.name[:-3]
        self.exp_name = exp_name

        self.exp_module = self.load_experiment(script_path)
        self.running_processes: List[Popen] = []

    def load_experiment(self, script_path):
        import sys
        sys.path.append(str(script_path.parent))

        spec = spec_from_file_location("exp_module", script_path)
        exp_module = module_from_spec(spec)
        spec.loader.exec_module(exp_module)
        return exp_module

    def is_valid_config(self, config: dict) -> bool:
        if hasattr(self.exp_module, "check_valid_config"):
            try:
                self.exp_module.check_valid_config(config)
            except ValueError:
                return False
        return True

    def adapt_config(self, config: dict) -> dict:
        return config

    def grid_values(self, config_files: List[Union[Path, str]] = None, choices: List[str] = None) -> dict:
        grid_values = {}

        def add_entry(name, value):
            if len(name) == 0:
                raise ValueError("Empty name!")
            if name in grid_values:
                raise ArgumentError(None, f"Specified argument {name!r} more than once!")
            if not isinstance(value, list):
                value = [value]
            grid_values[name] = value

        if config_files is not None:
            for config_file in config_files:
                with open(config_file, "r") as file:
                    value_dict = full_load(file)
                    for key, entry in value_dict.items():
                        add_entry(key, entry)
        if choices is not None:
            for value_choice in choices:
                if isinstance(value_choice, str):
                    name, raw_value = value_choice.split("=", 1)
                    # assert name in grid_values, name
                    assert len(name) > 0, name
                    assert len(raw_value) > 0, name
                    try:
                        parsed_value = json.loads(raw_value)
                    except json.JSONDecodeError:
                        print(raw_value)
                        raise
                elif len(value_choice) == 2:
                    name, parsed_value = value_choice
                else:
                    raise ValueError(
                        f"Unknown value {value_choice}. Can parse strings 'key=value' and tuples/lists/... (key, value).")
                add_entry(name, parsed_value)

        return grid_values

    def values_from_ex_config(self) -> dict:
        """
        Load default config values from experiment.
        """
        # Fill in remaining values from default parameters
        ex: Experiment = self.exp_module.ex
        if len(ex.configurations) != 1:
            raise NotImplementedError(f"{len(ex.configurations)} configurations found.")
        # Copy dict in case it is altered elsewhere later
        return dict(ex.configurations[0]())

    def add_default_values(self, config: dict):
        # Copy dict in case it is altered elsewhere later
        new_config = dict(config)
        for key, value in self.values_from_ex_config().items():
            if key not in config:
                new_config[key] = value
        return new_config

    def run_configuration(self, config, num_threads, mark_time_gauged=False, connect_debug=False, verbose=False):
        arguments = []
        if mark_time_gauged:
            arguments.append("--mark-time-gauged")
        if connect_debug:
            arguments.append("--connect-debug")

        all_config = {
            "num_threads": num_threads,
            **config
        }
        if len(all_config) > 0:
            arguments.append('with')
            for key, value in all_config.items():
                arguments.append(f'{key}={value!r}')

        out = None if verbose else subprocess.PIPE
        with Popen(['python', self.script_path, *arguments],
                   stdout=out, stderr=out,
                   # Signals to controller are not passed to runner
                   preexec_fn=os.setpgrp) as process:
            self.running_processes.append(process)
            stdout, stderr = process.communicate()
            self.running_processes.remove(process)
            return RunResult(config=config, return_code=process.poll(), stdout=stdout, stderr=stderr)

    def grid_spec_to_list(self, config_spec: Dict[str, list], filter_valid=True):
        config_keys = list(config_spec.keys())
        configs = []
        total_invalid_count = 0
        for config_values in product(*config_spec.values()):
            config = dict(zip(config_keys, config_values))
            if filter_valid and not self.is_valid_config(config):
                total_invalid_count += 1
                continue
            configs.append(config)
        return configs, {
            "invalid": total_invalid_count
        }

    def ex_ignore_keys(self):
        ignore_keys = self.exp_module.ignore_keys
        return ignore_keys

    def get_existing_count(self, config, loader, add_default_parameters=True, ignore_internal_parameters=True):
        if add_default_parameters:
            config = self.add_default_values(config)
        if ignore_internal_parameters is not False:
            if isinstance(ignore_internal_parameters, bool):
                ignore_keys = self.ex_ignore_keys()
            else:
                ignore_keys = ignore_internal_parameters
            config = {
                key: value
                for key, value in config.items()
                if key not in ignore_keys
            }

        # Necessary for loose MongoDB matching
        expanded_config = expand_nested_dicts(config, prefix="config.")
        query = {
            "status": {"$in": ["COMPLETED"]},
            "experiment.name": self.exp_name,
            "omniboard.tags": {"$ne": "invalid"},
            **expanded_config,
        }
        return loader._runs.count_documents(query)

    def missing_config_counts(self, configs: List[Dict[str, object]], loader: ExperimentLoader, repeat_each=1):
        config_with_counts = []
        total_missing_count = 0
        total_duplicate_count = 0
        total_finished_count = 0
        with tqdm(configs, desc="Preparing") as pbar:
            for config in pbar:
                if any(config == existing_config for count, existing_config in config_with_counts):
                    total_duplicate_count += 1

                existing_count = self.get_existing_count(config, loader)
                missing_count = repeat_each - existing_count
                if missing_count > 0:
                    total_missing_count += missing_count
                    config_with_counts.append(ConfigWithCount(config, missing_count))
                    pbar.set_description(f"Identified {len(config_with_counts)} configs -> {total_missing_count} runs")
                else:
                    total_finished_count += 1

        return config_with_counts, {
            "missing": total_missing_count,
            "duplicate": total_duplicate_count,
            "finished": total_finished_count
        }

    def start_runs(self, configs: List[dict | ConfigWithCount], num_parallel_runs=None, num_threads=1,
                   connect_debug=False, mark_time_gauged=False, verbose=False):
        if num_parallel_runs is None:
            num_parallel_runs = max(1, os.cpu_count() // num_threads - 1)

        # Map all to ConfigWithCount
        configs = [
            config
            if isinstance(config, ConfigWithCount) else
            ConfigWithCount(config=config, count=1)
            for config in configs
        ]
        max_repeat = max(repeat for config, repeat in configs)

        pool = ThreadPoolExecutor(num_parallel_runs)
        futures = []
        for run_at_least in range(max_repeat, 0, -1):
            for config_spec in configs:
                if config_spec.count >= run_at_least:
                    futures.append(pool.submit(
                        self.run_configuration,
                        config=config_spec.config, num_threads=num_threads,
                        mark_time_gauged=mark_time_gauged, connect_debug=connect_debug,
                        verbose=verbose
                    ))
        return pool, futures

    def run_configs_and_wait(self, configs: List[dict | ConfigWithCount], num_parallel_runs=None, num_threads=1,
                    connect_debug=False, mark_time_gauged=False, verbose=False, send_message=True) -> List[RunResult]:
        pool, futures = self.start_runs(
            configs,
            num_parallel_runs=num_parallel_runs, num_threads=num_threads,
            connect_debug=connect_debug, mark_time_gauged=mark_time_gauged,
            verbose=verbose
        )
        interrupted_count = 0
        while True:
            try:
                results = self.fetch_results(futures, send_message=send_message)
                break
            except KeyboardInterrupt:
                interrupted_count += 1
                if interrupted_count == 1:
                    # Cancel future runs
                    pool.shutdown(wait=False, cancel_futures=True)
                    # Pool shutdown does not mark futures as_completed
                    # https://github.com/python/cpython/issues/87893
                    for f in tqdm(futures, desc="Cancelling future runs"):
                        if f.cancelled():
                            f.set_running_or_notify_cancel()
                    print("Stopped all pending experiments.")
                    print("Hit Ctrl-C again to cancel running experiments.")
                elif interrupted_count == 2:
                    # Cancel current runs
                    for process in tqdm(self.running_processes, desc="Killing processes"):
                        process.kill()
                    print("Stopped all running experiments.")
        if interrupted_count > 2:
            raise KeyboardInterrupt
        # Wait for remaining processes
        pool.shutdown(wait=True)

        status_counts = status_count_counter(results)
        print(f"Done running {sum([config.count for config in configs])} experiments: {status_counts}")
        if len(set(status_counts) - {0}) > 0:
            print(f"Total: {sum(value for key, value in status_counts.items() if key != 0)} FAILED!")
        else:
            print("All succeeded :D")
        if send_message:
            send_bot_message(f"Launcher done: {status_counts}")
            return results

    def fetch_results(self, futures, timeout=None, send_message=True):
        last_elapsed = 60
        results = []
        with tqdm(as_completed(futures, timeout=timeout), total=len(futures), smoothing=0) as pbar:
            for future in pbar:
                if future.cancelled():
                    result = RunResult(None, "cancelled", None, None)
                else:
                    result: RunResult = future.result()
                results.append(result)

                status_counts = status_count_counter(results)
                result_code = result.return_code
                elapsed = pbar.format_dict["elapsed"]
                if send_message:
                    elapsed_delta = timedelta(seconds=elapsed)
                    if result_code != 0 and status_counts[result_code] == 10 ** int(
                            log10(status_counts[result_code])):
                        send_bot_message(
                            f"Code {result_code}: {status_counts[result_code]} "
                            f"failed after {elapsed_delta}."
                        )
                    elif result_code == 0 and elapsed > last_elapsed * 2:
                        send_bot_message(
                            f"{status_counts[result_code]} succeeded after {elapsed_delta}."
                        )
                        last_elapsed = elapsed
                pbar.set_description(str(status_counts))
        return results


def status_count_counter(results: List[RunResult]) -> Counter:
    return Counter(result.return_code for result in results)


def expand_nested_dicts(to_expand, prefix="", new_dict=None):
    if new_dict is None:
        new_dict = {}
    for key, value in to_expand.items():
        new_key = prefix + key
        if isinstance(value, dict):
            expand_nested_dicts(value, f"{new_key}.", new_dict)
        else:
            new_dict[new_key] = value
    return new_dict
