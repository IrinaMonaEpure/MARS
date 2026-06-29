import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from msean.measurements.properties import PropertyEnum


def weighted_percentile(values, weights, q):
    mask = weights > 0
    values = values[mask]
    weights = weights[mask]

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]

    cdf = np.cumsum(weights) / np.sum(weights)
    return values[np.searchsorted(cdf, q / 100)]


def stats_from_dist(dist):
    k = dist[:, 0]
    freq = dist[:, 1]

    n_degree_0 = freq[k == 0][0] if np.any(k == 0) else 0
    max_degree = k[freq > 0].max()
    p10 = weighted_percentile(k, freq, 10)
    p90 = weighted_percentile(k, freq, 90)

    return n_degree_0, max_degree, p10, p90


parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", type=str, required=True)
parser.add_argument("--output", type=str, required=True)

args = parser.parse_args()

rows = []

for path in sorted(Path(args.results_dir).glob("results_*.pkl")):
    seed = int(path.stem.split("_")[-1])

    with open(path, "rb") as f:
        results = pickle.load(f)

    for alpha, alpha_results in results.items():

        # Overall graph
        dist = alpha_results[PropertyEnum.DEGREE_DISTRIBUTION]
        n0, max_k, p10, p90 = stats_from_dist(dist)

        rows.append({
            "seed": seed,
            "alpha": alpha,
            "scope": "overall",
            "layer": None,
            "n_degree_0": n0,
            "max_degree": max_k,
            "p10": p10,
            "p90": p90,
        })

        # Per layer
        layer_dists = alpha_results[
            PropertyEnum.DEGREE_DISTRIBUTION_PER_LAYER
        ]

        for layer_idx, layer_dist in enumerate(layer_dists):
            n0, max_k, p10, p90 = stats_from_dist(layer_dist)

            rows.append({
                "seed": seed,
                "alpha": alpha,
                "scope": "layer",
                "layer": layer_idx,
                "n_degree_0": n0,
                "max_degree": max_k,
                "p10": p10,
                "p90": p90,
            })

df = pd.DataFrame(rows)

summary = (
    df
    .groupby(["alpha", "scope", "layer"], dropna=False)
    .agg(
        mean_degree_0=("n_degree_0", "mean"),
        se_degree_0=("n_degree_0", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        mean_max_degree=("max_degree", "mean"),
        se_max_degree=("max_degree", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        min_observed_max_degree=("max_degree", "min"),
        max_observed_max_degree=("max_degree", "max"),
        mean_p10=("p10", "mean"),
        se_p10=("p10", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        mean_p90=("p90", "mean"),
        se_p90=("p90", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        n_seeds=("seed", "nunique"),
    )
    .reset_index()
)

output = Path(args.output)
summary.to_csv(output, index=False)

print(f"Saved summary to {output}")