import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from msean.measurements.properties import PropertyEnum


def weighted_percentile(
    values: np.ndarray,
    weights: np.ndarray,
    q: float,
) -> float:
    """
    Calculate a weighted percentile from values and their frequencies.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    mask = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0)
    )

    values = values[mask]
    weights = weights[mask]

    if values.size == 0:
        return np.nan

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]

    cumulative_weights = np.cumsum(weights)
    cutoff = (q / 100) * cumulative_weights[-1]

    idx = np.searchsorted(
        cumulative_weights,
        cutoff,
        side="left",
    )

    idx = min(idx, len(values) - 1)

    return float(values[idx])


def stats_from_distribution(dist: np.ndarray) -> dict:
    """
    Calculate weighted descriptive statistics from a distribution.

    Expected format:

        [[value, frequency], ...]

    Additional columns are ignored.
    """
    dist = np.asarray(dist, dtype=float)

    empty_result = {
        "mean": np.nan,
        "std": np.nan,
        "p10": np.nan,
        "p25": np.nan,
        "median": np.nan,
        "p75": np.nan,
        "p90": np.nan,
    }

    if (
        dist.ndim != 2
        or dist.shape[1] < 2
        or len(dist) == 0
    ):
        return empty_result

    values = dist[:, 0]
    weights = dist[:, 1]

    mask = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0)
    )

    values = values[mask]
    weights = weights[mask]

    if values.size == 0:
        return empty_result

    total_weight = np.sum(weights)

    if total_weight <= 0:
        return empty_result

    mean = np.sum(values * weights) / total_weight

    variance = (
        np.sum(weights * (values - mean) ** 2)
        / total_weight
    )

    return {
        "mean": float(mean),
        "std": float(np.sqrt(variance)),
        "p10": weighted_percentile(values, weights, 10),
        "p25": weighted_percentile(values, weights, 25),
        "median": weighted_percentile(values, weights, 50),
        "p75": weighted_percentile(values, weights, 75),
        "p90": weighted_percentile(values, weights, 90),
    }


def standard_error(series: pd.Series) -> float:
    """
    Calculate the standard error across valid seed results.
    """
    valid = series.dropna()

    if len(valid) <= 1:
        return np.nan

    return float(
        valid.std(ddof=1) / np.sqrt(len(valid))
    )


def config_key_to_columns(config_key) -> dict:
    """
    Convert a scalar or tuple configuration key to CSV columns.

    Examples
    --------
    0.5
        -> {"param_0": 0.5}

    (0.5, 0.1)
        -> {"param_0": 0.5, "param_1": 0.1}
    """
    if isinstance(config_key, tuple):
        return {
            f"param_{i}": value
            for i, value in enumerate(config_key)
        }

    return {
        "param_0": config_key,
    }


def parse_property(property_name: str) -> PropertyEnum:
    """
    Convert a command-line property name to PropertyEnum.
    """
    normalized_name = property_name.strip().upper()

    try:
        return PropertyEnum[normalized_name]

    except KeyError as exc:
        valid_properties = ", ".join(
            prop.name
            for prop in PropertyEnum
        )

        raise ValueError(
            f"Unknown property '{property_name}'. "
            f"Available properties are: {valid_properties}"
        ) from exc


parser = argparse.ArgumentParser()

parser.add_argument(
    "--results_dir",
    type=str,
    required=True,
    help="Directory containing results_*.pkl files.",
)

parser.add_argument(
    "--output",
    type=str,
    required=True,
    help="Path of the summary CSV to create.",
)

parser.add_argument(
    "--property",
    type=str,
    required=True,
    help=(
        "Overall distribution property, such as "
        "DEGREE_DISTRIBUTION, "
        "EDGE_LENGTH_DISTRIBUTION, or "
        "AVERAGE_ALTER_DISTANCE_DISTRIBUTION."
    ),
)

args = parser.parse_args()

property_enum = parse_property(args.property)

unsupported_properties = {
    PropertyEnum.DEGREE_DISTRIBUTION_PER_LAYER,
}

if property_enum in unsupported_properties:
    raise ValueError(
        f"{property_enum.name} is a per-layer property. "
        "This script supports only overall distributions."
    )

results_dir = Path(args.results_dir)

result_paths = sorted(
    results_dir.glob("results_*.pkl")
)

if not result_paths:
    raise FileNotFoundError(
        f"No results_*.pkl files found in {results_dir}"
    )

rows = []

for path in result_paths:

    try:
        seed = int(path.stem.split("_")[-1])

    except ValueError as exc:
        raise ValueError(
            f"Could not extract a seed from filename: {path.name}"
        ) from exc

    with open(path, "rb") as file:
        results = pickle.load(file)

    for config_key, config_results in results.items():

        if property_enum not in config_results:
            raise KeyError(
                f"{property_enum.name} was not found in "
                f"{path.name} for configuration {config_key}."
            )

        dist = config_results[property_enum]

        # Reject per-layer/list-style outputs.
        if isinstance(dist, list):
            raise ValueError(
                f"{property_enum.name} returned a list of distributions. "
                "This script supports only one overall distribution "
                "per configuration."
            )

        stats = stats_from_distribution(dist)

        rows.append({
            "seed": seed,
            **config_key_to_columns(config_key),
            **stats,
        })


per_seed_df = pd.DataFrame(rows)

if per_seed_df.empty:
    raise ValueError(
        f"No usable data found for {property_enum.name}."
    )

param_cols = sorted(
    [
        col
        for col in per_seed_df.columns
        if col.startswith("param_")
    ],
    key=lambda col: int(col.split("_")[1]),
)

summary = (
    per_seed_df
    .groupby(
        param_cols,
        as_index=False,
        dropna=False,
    )
    .agg(
        mean=("mean", "mean"),
        se_mean=("mean", standard_error),

        std=("std", "mean"),
        se_std=("std", standard_error),

        p10=("p10", "mean"),
        se_p10=("p10", standard_error),

        p25=("p25", "mean"),
        se_p25=("p25", standard_error),

        median=("median", "mean"),
        se_median=("median", standard_error),

        p75=("p75", "mean"),
        se_p75=("p75", standard_error),

        p90=("p90", "mean"),
        se_p90=("p90", standard_error),

        n_seeds=("seed", "nunique"),
    )
)

output = Path(args.output)
output.parent.mkdir(
    parents=True,
    exist_ok=True,
)

summary.to_csv(
    output,
    index=False,
)

per_seed_output = output.with_name(
    f"{output.stem}_per_seed{output.suffix}"
)

per_seed_df.to_csv(
    per_seed_output,
    index=False,
)

print(f"Property: {property_enum.name}")
print(f"Saved summary to {output}")
print(f"Saved per-seed values to {per_seed_output}")