from typing import List, Tuple
from copy import deepcopy
from pathlib import Path
import numpy as np
import networkx as nx

from msean.measurements import (
    PropertyEnum,
    get_degree_dist,
    get_degree_dist_layers,
    get_embeddedness_dist,
    get_local_clustering_dist,
    get_edge_len_dist,
    get_excess_closure_dist,
    get_density,
    get_density_layers,
    get_global_clustering,
    get_avg_local_clustering,
    get_avg_degree,
    get_avg_degree_layers,
    get_triangles
)

from msean.config import Config, set_nested, save_config
from msean.generation import generate_n_graphs
from msean.io.save import prepare_batch_directory, prepare_run_directory

PROPERTY_CALL = {
    PropertyEnum.DEGREE_DISTRIBUTION: get_degree_dist,
    PropertyEnum.DEGREE_DISTRIBUTION_PER_LAYER: get_degree_dist_layers,
    PropertyEnum.EMBEDDEDNESS_DISTRIBUTION: get_embeddedness_dist,
    PropertyEnum.LOCAL_CLUSTERING_DISTRIBUTION: get_local_clustering_dist,
    PropertyEnum.EDGE_LENGTH_DISTRIBUTION: get_edge_len_dist,
    PropertyEnum.EXCESS_CLOSURE_DISTRIBUTION: get_excess_closure_dist,
    PropertyEnum.DENSITY: get_density,
    PropertyEnum.DENSITY_PER_LAYER: get_density_layers,
    PropertyEnum.GLOBAL_CLUSTERING: get_global_clustering,
    PropertyEnum.AVERAGE_LOCAL_CLUSTERING: get_avg_local_clustering,
    PropertyEnum.AVERAGE_DEGREE: get_avg_degree,
    PropertyEnum.AVERAGE_DEGREE_PER_LAYER: get_avg_degree_layers,
    PropertyEnum.TRIANGLES: get_triangles
}

GLOBAL_PROPERTIES = [
    PropertyEnum.DENSITY,
    PropertyEnum.DENSITY_PER_LAYER,
    PropertyEnum.GLOBAL_CLUSTERING,
    PropertyEnum.AVERAGE_LOCAL_CLUSTERING,
    PropertyEnum.AVERAGE_DEGREE,
    PropertyEnum.AVERAGE_DEGREE_PER_LAYER,
    PropertyEnum.TRIANGLES
]

DISTRIBUTION_PROPERTIES = [
    PropertyEnum.DEGREE_DISTRIBUTION,
    PropertyEnum.DEGREE_DISTRIBUTION_PER_LAYER,
    PropertyEnum.EMBEDDEDNESS_DISTRIBUTION,
    PropertyEnum.LOCAL_CLUSTERING_DISTRIBUTION,
    PropertyEnum.EDGE_LENGTH_DISTRIBUTION,
    PropertyEnum.EXCESS_CLOSURE_DISTRIBUTION
]

def batch_experiment(cfg: Config, parent_dir: Path, rng: np.random.Generator, param_name: str, param_values: List, properties: List[PropertyEnum]):
    """
    Runs a batch experiment by varying a single configuration parameter over a specified range,
    generating a graph for each parameter value, and computing selected properties.

    For each parameter value:
        - A deep copy of the base configuration is created
        - The parameter specified by `param_name` is updated
        - A graph and its layers are generated
        - A dedicated run directory is created inside the batch directory
        - Selected properties are computed and stored

    Directory structure created:
        runs/
        └── <batch_name>/
            ├── config.yaml
            ├── summary.csv
            ├── plots/
            ├── <param>_<value_1>/
            │   ├── config.yaml
            │   └── plots/
            ├── <param>_<value_2>/
            │   ├── config.yaml
            │   └── plots/
            └── ...

    where:
        <param> is the last component of `param_name`
        <value_i> are the parameter values in the specified range

    Arguments:
        cfg (Config):
            Base configuration object.
        parent_dir (Path):
            Folder where output files should be saved.
        rng(np.random.Generator):
            Random number generator.
        param_name (str):
            Name of the parameter to vary, using dot notation
            (e.g., "connection.xi").
        param_vals (List):
            List of parameter values.
        properties (List[PropertyEnum]):
            List of properties to compute for each generated graph.

    Returns:
        results: dict
            Nested dictionary of results with structure:
                {
                    param_val_1: {
                        PropertyEnum.PROPERTY_A: value,
                        PropertyEnum.PROPERTY_B: value,
                        ...
                    },
                    param_val_2: {
                        PropertyEnum.PROPERTY_A: value,
                        PropertyEnum.PROPERTY_B: value,
                        ...
                    },
                    ...
                }
                where:
                    - Keys are parameter values (ints from the specified range)
                    - Values are dictionaries mapping PropertyEnum → computed result

            batch_paths: dict
                See output format of msean.io.save.prepare_batch_directory(cfg).
    """

    # Prepare batch (parent) output directory
    batch_paths = prepare_batch_directory(parent_dir)
    save_config(cfg, batch_paths["config"])

    results = {}

    short_param_name = param_name.split(".")[-1]

    for param_val in param_values:
        cfg_i = deepcopy(cfg)
        set_nested(cfg_i, param_name, param_val)

        # Prepare run directory inside batch directory 
        run_name = f"{short_param_name}_{param_val}"
        run_paths = prepare_run_directory(
            parent_dir=batch_paths["batch_dir"],
            run_name=run_name
        )

        save_config(cfg_i, run_paths["config"])

        G_list, layers_list = generate_n_graphs(cfg_i, rng)

        param_results = measure_properties(G_list, layers_list, cfg_i, properties)
        results[param_val] = param_results

    return results, batch_paths

def measure_properties(G_list: List[nx.Graph], layers_list: List[List[nx.Graph]], cfg: Config, properties: List[PropertyEnum]):
    # replace for prop in properties, return dict results[param_val]

    param_results = {}

    for prop in properties:
        if prop == PropertyEnum.DEGREE_DISTRIBUTION_PER_LAYER:
            # Degree distribution per layer requires individual layers as input
            aggregate_value = aggregate_distribution_per_layer(layers_list, prop)
        elif prop in [PropertyEnum.AVERAGE_DEGREE_PER_LAYER, PropertyEnum.DENSITY_PER_LAYER]:
            # Require individual layers as input
            aggregate_value = aggregate_global_property_per_layer(layers_list, prop)
        elif prop in GLOBAL_PROPERTIES:
            aggregate_value = aggregate_global_property(G_list, prop)
        elif prop in DISTRIBUTION_PROPERTIES:
            aggregate_value = aggregate_distribution(G_list, layers_list, cfg, prop)
        else:
            raise ValueError(f"Unknown property: {prop}")

        param_results[prop] = aggregate_value

    return param_results


def aggregate_global_property(G_list: List[nx.Graph], property: PropertyEnum):
    # average over results from multiple runs, simple mean for global properties
    values = np.array([
        PROPERTY_CALL[property](G)
        for G in G_list
    ], dtype=float)

    mean = np.mean(values)
    std = np.std(values)
    se = std / np.sqrt(len(values))

    return mean, std, se

def aggregate_global_property_per_layer(layers_list: List[List[nx.Graph]], property: PropertyEnum):
    # average over results from multiple runs, simple mean for global properties
    values = np.array([
        PROPERTY_CALL[property](layers)
        for layers in layers_list
    ], dtype=float)

    mean_values = np.mean(values, axis=0)
    std_values = np.std(values, axis=0)
    se_values = std_values / np.sqrt(values.shape[0])

    return mean_values, std_values, se_values

def aggregate_distribution(G_list: List[nx.Graph], layers_list: List[List[nx.Graph]], cfg: Config, property: PropertyEnum, resolution: float = None):
    if resolution is None:
        if property in [
            PropertyEnum.LOCAL_CLUSTERING_DISTRIBUTION,
            PropertyEnum.EDGE_LENGTH_DISTRIBUTION,
            PropertyEnum.EXCESS_CLOSURE_DISTRIBUTION
        ]:
            # Range is between 0 and 1, or 0 and sqrt(2) for edge length
            resolution = 0.05
        else:
            resolution = 1.0

    distributions = []
    for G, layers in zip(G_list, layers_list):
        if property == PropertyEnum.EDGE_LENGTH_DISTRIBUTION:
            # Passing cfg to specify whether toroidal (and if so, what is the metric space size)
            distributions.append(PROPERTY_CALL[property](G, cfg))
        elif property == PropertyEnum.EXCESS_CLOSURE_DISTRIBUTION:
            # Also requires individual layers
            distributions.append(PROPERTY_CALL[property](G, layers))
        else:
            distributions.append(PROPERTY_CALL[property](G))

    if property == PropertyEnum.EXCESS_CLOSURE_DISTRIBUTION:
        aggregated_distribution = combine_distributions(
            distributions,
            resolution,
            min_val=0.0,
            max_val=1.0,
        )
    else:
        aggregated_distribution = combine_distributions(distributions, resolution)

    return aggregated_distribution

def combine_distributions(
        distributions: List[np.array],
        resolution: float,
        min_val: float = None,
        max_val: float = None,
    ):

    # Remove invalid rows
    clean_distributions = []

    for arr in distributions:
        arr = np.asarray(arr, dtype=float)

        mask = (
            np.isfinite(arr[:, 0]) &
            np.isfinite(arr[:, 1])
        )

        arr = arr[mask]

        clean_distributions.append(arr)

    distributions = clean_distributions

    # Calculate the minimum and maximum value over all distributions
    if min_val is None:
        min_val = min(arr[:, 0].min() for arr in distributions)

    if max_val is None:
        max_val = max(arr[:, 0].max() for arr in distributions)

    # Create a new range of values
    vals = np.arange(
        min_val,
        max_val + 2 * resolution,
        resolution
    )

    # Flatten distributions from 2D to 1D
    flat_arrays = np.array([
        flatten_distribution(arr, vals, resolution)
        for arr in distributions
    ])

    n_runs = flat_arrays.shape[0]
    mean_freqs = flat_arrays.mean(axis=0)
    std_freqs = flat_arrays.std(axis=0)
    se_freqs = std_freqs / np.sqrt(n_runs)

    result = np.column_stack((vals, mean_freqs, std_freqs, se_freqs))

    return result

def aggregate_distribution_per_layer(layers_list: List[List[nx.Graph]], property: PropertyEnum, resolution: float = 1.0):
    distribution_sets = []
    for layer_set in layers_list:
        distribution_sets.append(PROPERTY_CALL[property](layer_set))

    distribution_per_layer = []

    for i in range(len(distribution_sets[0])):
        distribution_per_layer.append(combine_distributions([layer_set[i] for layer_set in distribution_sets], resolution))

    return distribution_per_layer


def flatten_distribution(arr: np.array, vals: np.array, resolution: float):
    out = np.zeros(len(vals))

    x = arr[:, 0]
    y = arr[:, 1]

    # Map x values to nearest bin index
    idx = np.floor((x - vals[0] + resolution / 2) / resolution).astype(int)

    assert np.all((idx >= 0) & (idx < len(vals))), "Some values fall outside vals range"

    # Accumulate
    np.add.at(out, idx, y)

    return out