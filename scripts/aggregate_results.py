import argparse
import pickle
from pathlib import Path

import numpy as np

from msean.measurements.batch import combine_distributions


DISTRIBUTION_RESOLUTIONS = {
    "DEGREE_DISTRIBUTION": 1.0,
    "DEGREE_DISTRIBUTION_PER_LAYER": 1.0,
    "EMBEDDEDNESS_DISTRIBUTION": 1.0,
    "LOCAL_CLUSTERING_DISTRIBUTION": 0.01,
    "EDGE_LENGTH_DISTRIBUTION": 0.01,
    "EXCESS_CLOSURE_DISTRIBUTION": 0.01,
    "AVERAGE_ALTER_DISTANCE_DISTRIBUTION": 0.01,
}


def aggregate_mean_std_se(values):
    means = np.array([v[0] for v in values], dtype=float)

    mean = means.mean(axis=0)
    std = means.std(axis=0)
    se = std / np.sqrt(len(means))

    return mean, std, se


def aggregate_distribution_results(values, prop):
    resolution = DISTRIBUTION_RESOLUTIONS[prop.name]

    distributions = [
        value[:, [0, 1]]
        for value in values
    ]

    return combine_distributions(
        distributions,
        resolution=resolution,
    )


def aggregate_per_layer_distribution_results(values, prop):
    n_layers = len(values[0])

    return [
        aggregate_distribution_results(
            [value[layer_idx] for value in values],
            prop,
        )
        for layer_idx in range(n_layers)
    ]


def aggregate_results(all_results):
    combined = {}

    param_values = all_results[0].keys()

    for param_val in param_values:
        combined[param_val] = {}

        props = all_results[0][param_val].keys()

        for prop in props:
            values = [
                result[param_val][prop]
                for result in all_results
            ]

            first = values[0]

            if isinstance(first, list):
                combined[param_val][prop] = (
                    aggregate_per_layer_distribution_results(
                        values,
                        prop,
                    )
                )

            elif isinstance(first, np.ndarray):
                combined[param_val][prop] = (
                    aggregate_distribution_results(
                        values,
                        prop,
                    )
                )

            elif isinstance(first, tuple) and len(first) == 3:
                combined[param_val][prop] = (
                    aggregate_mean_std_se(values)
                )

            else:
                raise TypeError(
                    f"Unsupported result type for {prop}: {type(first)}"
                )

    return combined


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    result_files = sorted(results_dir.glob("results_*.pkl"))

    if not result_files:
        raise FileNotFoundError(
            f"No results_*.pkl files found in {results_dir}"
        )

    print(f"Found {len(result_files)} result files.", flush=True)

    all_results = []

    for path in result_files:
        print(f"Loading {path}", flush=True)

        with open(path, "rb") as f:
            all_results.append(pickle.load(f))

    combined = aggregate_results(all_results)

    output_path = Path(args.output)

    with open(output_path, "wb") as f:
        pickle.dump(combined, f)

    print(f"Saved combined results to {output_path}", flush=True)


if __name__ == "__main__":
    main()