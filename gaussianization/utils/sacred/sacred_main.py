import sys
from argparse import ArgumentParser
from pathlib import Path
from random import shuffle

from incense import ExperimentLoader

from gaussianization.utils.sacred.sacred_launcher import ExperimentLauncher


def launcher_main(launcher: ExperimentLauncher, args=None, adapt_config_list=None):
    parser = ArgumentParser()
    parser.add_argument("-v", "--verbose", default=False, action="store_true")
    parser.add_argument("-p", "--parallel-exps", default=None, type=int)
    parser.add_argument("-r", "--repeat-exps", default=1, type=int)
    parser.add_argument("-i", "--ignore-until", default=2511, type=lambda s: float("inf") if s == "all" else int(s))
    parser.add_argument("-c", "--cores-per-exp", default=1, type=int)
    parser.add_argument("--connect-debug", default=False, action="store_true")
    parser.add_argument("--dry-run", default=False, action="store_true")
    parser.add_argument("--mark-time-gauged", default=False, action="store_true")
    parser.add_argument("--shuffle-configs", default=False, action="store_true",
                        help="Execute configurations with the same number of missing experiments in random order.")
    mux_group = parser.add_mutually_exclusive_group()
    mux_group.add_argument("-f", "--config-files", nargs="*", type=Path)
    mux_group.add_argument("--values", nargs="*")

    args = parser.parse_args(args)

    # Specify dict[str -> list[value]|value]
    grid_values = launcher.grid_values(args.config_files, args.values)
    print("Perform grid of experiments:")
    for key, value in grid_values.items():
        if len(value) == 1:
            print(f"  {key} = {value[0]!r}")
        elif len(value) > 10:
            print(f"  {key} from {value[:9]!r} and {len(value) - 9} more")
        else:
            print(f"  {key} from {value!r}")
    print()

    # Map to list[dict[str -> value]]
    loader = ExperimentLoader()
    config_list, stats_expand = launcher.grid_spec_to_list(grid_values)
    if adapt_config_list is not None:
        config_list = adapt_config_list(config_list)
    configs, stats_missing = launcher.missing_config_counts(config_list, loader, repeat_each=args.repeat_exps)

    stats = stats_expand | stats_missing
    if stats["invalid"] > 0:
        print(f"{stats['invalid']} configurations were invalid.")
    if stats["duplicate"] > 0:
        print(f"{stats['duplicate']} configurations were skipped due to duplicates.")
    if stats["finished"] > 0:
        print(f"{stats['finished']} configurations were already run often enough.")
    if stats["missing"] == 0:
        print("All the requested experiments were already run. 🏝")
        return

    if args.dry_run:
        print("Would start:")
        for missing_count, config in configs:
            print(f"{missing_count}x {config}")
        return

    if args.shuffle_configs:
        shuffle(configs)

    num_threads = args.cores_per_exp
    mark_time_gauged = args.mark_time_gauged
    connect_debug = args.connect_debug
    verbose = args.verbose
    pool_size = args.parallel_exps
    launcher.run_configs_and_wait(configs, pool_size, num_threads, connect_debug, mark_time_gauged, verbose)

    # for dict[str -> value]:
    #     Add default values
    #     Add automatic values
    #     Remove irrelevant keys


if __name__ == '__main__':
    script_name = sys.argv[1]
    launcher = ExperimentLauncher(script_name)
    launcher_main(launcher, sys.argv[2:])
