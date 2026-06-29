import pickle
import argparse
import numpy as np
from pathlib import Path
import time
from datetime import datetime

from msean import load_config
from msean.measurements.batch import batch_experiment
from msean.measurements.properties import PropertyEnum

parser = argparse.ArgumentParser()

parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--results_dir", type=str, required=True)

args = parser.parse_args()

seed = args.seed
results_dir = Path(args.results_dir)
results_dir.mkdir(parents=True, exist_ok=True)

# Load configuration file
root = Path(__file__).resolve().parents[1] # go up from scripts/ to project root
cfg = load_config(root / "configs" / "default_server_10k_normal.yaml")
runs_dir = root / "runs"

# Intialize random number generator for experiment reproducibility
rng = np.random.default_rng(seed)

# Run batch experiment
alpha_list = [(1/2)**(i/2) for i in range(17, 21)]

start_time = time.time()

print(
    f"Seed {seed} started at "
    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, "
    f"N: {cfg.network.n_nodes}, "
    f"K: {cfg.network.n_affiliations}",
    flush=True,
)

results, paths = batch_experiment(
    cfg=cfg,
    parent_dir=runs_dir,
    rng=rng,
    param_name="connection.alpha",
    param_values=alpha_list,
    properties=[
        PropertyEnum.AVERAGE_DEGREE,
        PropertyEnum.AVERAGE_DEGREE_PER_LAYER,
        PropertyEnum.DENSITY,
        PropertyEnum.DENSITY_PER_LAYER,
        PropertyEnum.AVERAGE_LOCAL_CLUSTERING,
        PropertyEnum.DEGREE_DISTRIBUTION,
        PropertyEnum.DEGREE_DISTRIBUTION_PER_LAYER,
        PropertyEnum.TRIANGLES,
        PropertyEnum.TRIANGLES_PER_LAYER,
        PropertyEnum.TRIANGLE_DIMENSIONS,
        PropertyEnum.LOCAL_CLUSTERING_DISTRIBUTION,
        PropertyEnum.GLOBAL_CLUSTERING,
        PropertyEnum.EDGE_LENGTH_DISTRIBUTION
    ],
    print_seed=seed
)

elapsed = time.time() - start_time

print(
    f"Seed {seed} finished in "
    f"{elapsed/60:.1f} minutes",
    flush=True,
)

# Save results file
with open(results_dir / f"results_{seed}.pkl", "wb") as f:
    pickle.dump(results, f)
