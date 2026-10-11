"""Task runner for the common commands.

A Makefile would not run on Windows, where this project is developed, so the
command list lives here instead — one definition rather than two.

Usage:
    python run.py                 list the tasks
    python run.py test
    python run.py features --state DL --horizon 24
    python run.py train --tag my_experiment
"""

from __future__ import annotations

import subprocess
import sys

# name -> (what it does, the command, roughly how long)
TASKS: dict[str, tuple[str, list[str], str]] = {
    "install": ("Install dependencies", ["pip", "install", "-r", "requirements.txt"], ""),
    "test": ("Run the test suite", ["pytest", "tests/", "-q"], "~2m"),
    "quality": ("Data-quality report", ["python", "pipeline/data_quality.py"], "~1m"),

    "registry": ("Station registry from stations.csv", ["python", "pipeline/registry.py"], "~1s"),
    "normalise": ("546 raw CSVs into partitioned Parquet", ["python", "pipeline/normalise.py"], "~10m"),
    "hourly": ("15-minute readings into an hourly table", ["python", "pipeline/build_hourly.py"], "~2m"),
    "weather": ("ERA5 weather, free, no key needed", ["python", "pipeline/fetch_weather.py"], "~80m"),
    "fires": ("NASA FIRMS history, needs FIRMS_MAP_KEY", ["python", "pipeline/fetch_fires.py"], "~50m"),
    "fire-features": ("Per-station fire exposure", ["python", "pipeline/build_fire_features.py"], "~1m"),

    "features": ("Build the model's feature table", ["python", "pipeline/features.py"], "~4m"),
    "train": ("Backtest and save a model", ["python", "pipeline/train.py", "--folds", "3"], "~40m"),
    "diagnose": ("Is a feature real, or just seasonal?", ["python", "pipeline/diagnose_fires.py"], "~1m"),
    "compare": ("Compare two tagged models", ["python", "pipeline/compare_models.py"], "~2m"),
}

# Tasks that run one after another.
CHAINS: dict[str, tuple[str, list[str]]] = {
    "check": ("Data quality then tests", ["quality", "test"]),
    "foundation": ("Everything needed before modelling", ["registry", "normalise", "hourly"]),
}


def show_help() -> int:
    print(__doc__.strip())
    print("\nTasks:")
    width = max(len(n) for n in list(TASKS) + list(CHAINS))
    for name, (description, _, duration) in TASKS.items():
        print(f"  {name:<{width}}  {description}" + (f"  ({duration})" if duration else ""))
    print("\nChains:")
    for name, (description, steps) in CHAINS.items():
        print(f"  {name:<{width}}  {description}: {' -> '.join(steps)}")
    print("\nAnything after the task name is passed straight through, e.g.")
    print("  python run.py features --state DL --horizon 6 --fire-features core")
    return 0


def run(command: list[str], extra: list[str]) -> int:
    full = [*command, *extra]
    print(f"$ {' '.join(full)}", flush=True)
    return subprocess.call(full)


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help", "help"):
        return show_help()

    name, extra = argv[0], argv[1:]

    if name in CHAINS:
        for step in CHAINS[name][1]:
            code = run(TASKS[step][1], extra if step == CHAINS[name][1][-1] else [])
            if code:
                print(f"\n{step} failed with exit code {code}; stopping.")
                return code
        return 0

    if name not in TASKS:
        print(f"Unknown task {name!r}. Run `python run.py` to list them.")
        return 2
    return run(TASKS[name][1], extra)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
