from pathlib import Path
from datetime import datetime

from ..config import Config


def prepare_run_directory(cfg:Config):
    """
    Creates and returns a directory structure for a run.

    If cfg.output.run_name is provided:
        runs/<run_name>/
    else:
        runs/YYYY-MM-DD_<index>/

    Structure:
        runs/<run_name>/
            config.yaml
            graph.pkl
            plots/
            csv_files/

    Returns:
        dict with paths
    """

    root = Path(__file__).resolve().parents[3]  # repo root
    runs_dir = root / "runs"
    runs_dir.mkdir(exist_ok=True)

    # Case 1: user-defined run name
    if getattr(cfg.output, "run_name", None):
        run_name = cfg.output.run_name
        run_dir = runs_dir / run_name

    # Case 2: auto-generate run name
    else:
        today = datetime.now().strftime("%Y-%m-%d")

        # find existing runs for today
        existing = [
            p.name for p in runs_dir.iterdir()
            if p.is_dir() and p.name.startswith(today)
        ]

        # extract indices
        indices = []
        for name in existing:
            try:
                suffix = name.replace(today + "_", "")
                indices.append(int(suffix))
            except:
                continue

        next_idx = max(indices) + 1 if indices else 1
        run_name = f"{today}_{next_idx:03d}"
        run_dir = runs_dir / run_name

    # create structure
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    csv_dir = run_dir / "csv_files"
    csv_dir.mkdir(exist_ok=True)

    paths = {
        "run_dir": run_dir,
        "config": run_dir / "config.yaml",
        "plots": plots_dir,
        "csv_files": csv_dir,
    }

    # TODO: save node/edge lists? network files?

    return paths