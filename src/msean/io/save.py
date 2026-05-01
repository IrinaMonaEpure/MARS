from pathlib import Path
from datetime import datetime

from ..config import Config


def _get_runs_dir() -> Path:
    root = Path(__file__).resolve().parents[3]
    runs_dir = root / "runs"
    runs_dir.mkdir(exist_ok=True)
    return runs_dir


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


def prepare_batch_directory(cfg: Config):
    """
    Creates and returns a parent directory for a batch experiment inside the `runs/` folder.
    If `cfg.output.save_folder` is provided, that value is used for the directory name.
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
                "plots": Path to the batch-level plots directory #TODO
            }
    """
    runs_dir = _get_runs_dir()

    if getattr(cfg.output, "save_folder", None):
        batch_name = cfg.output.save_folder
    else:
        batch_name = _next_dated_name(runs_dir, suffix="batch")

    batch_dir = runs_dir / batch_name
    batch_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = batch_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    return {
        "batch_dir": batch_dir,
        "config": batch_dir / "config.yaml",
        "summary": batch_dir / "summary.csv",
        "plots": plots_dir,
    }


def prepare_run_directory(cfg:Config, parent_dir: Path | None = None, run_name: str | None = None):
    """
    Creates a run directory either inside runs/ or inside a batch parent folder.
    """

    if run_name is None:
        if getattr(cfg.output, "run_name", None):
            run_name = cfg.output.run_name
        else:
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