from pathlib import Path
from datetime import datetime

from ..config import Config


def _next_dated_name(base_dir: Path, suffix: str | None = None) -> str:
    today = datetime.now().strftime("%Y-%m-%d")

    existing = [
        p.name for p in base_dir.iterdir()
        if p.is_dir() and p.name.startswith(today)
    ]

    indices = []
    for name in existing:
        tail = name.replace(today + "_", "")
        tail = tail.split("_")[0]
        try:
            indices.append(int(tail))
        except ValueError:
            continue

    next_idx = max(indices) + 1 if indices else 1

    if suffix:
        return f"{today}_{next_idx:03d}_{suffix}"
    
    return f"{today}_{next_idx:03d}"


def prepare_batch_directory(parent_dir: Path, experiment_name: str | None = None):
    """
    Creates and returns a parent directory for a batch experiment inside the given parent directory.
    Otherwise, a timestamped name is generated: YYYY-MM-DD_<index>_batch

    The resulting directory structure is:
        runs/
        └── <batch_name>/
            ├── config.yaml        # copy of the base configuration used for the batch
            ├── summary.csv        # optional aggregated results (to be populated later)
            ├── plots/             # plots summarizing the batch experiment
            └── <run_name_1>/      # created later by individual runs
                ├── config.yaml
                └── plots/
            └── <run_name_2>/
                ├── config.yaml
                └── plots/
            └── ...

    Returns:
        dict:
            {
                "batch_dir": Path to the batch directory,
                "config": Path to the batch-level config.yaml,
                "summary": Path to the batch-level summary.csv, #TODO
                "plots": Path to the batch-level plots directory
            }
    """
    if experiment_name is None:
        experiment_name = _next_dated_name(parent_dir)

    batch_dir = parent_dir / experiment_name
    batch_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = batch_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    return {
        "batch_dir": batch_dir,
        "config": batch_dir / "config.yaml",
        "summary": batch_dir / "summary.csv",
        "plots": plots_dir,
    }


def prepare_run_directory(parent_dir: Path, run_name: str | None = None):
    """
    Creates a run directory inside the given parent folder.
    """

    if run_name is None:
        run_name = _next_dated_name(parent_dir)

    run_dir = parent_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    return {
        "run_dir": run_dir,
        "config": run_dir / "config.yaml",
        "plots": plots_dir,
    }