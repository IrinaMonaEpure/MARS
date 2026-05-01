from typing import List, Tuple
from copy import deepcopy
import numpy as np

from msean.measurements import (
    PropertyEnum,
    get_degree_dist,
    get_degree_dist_layers,
    get_embeddedness_dist,
    get_clustering_dist,
    get_edge_len_dist,
    get_density,
    get_clustering,
    get_avg_degree,
)

from msean.config import Config, set_nested, save_config
from msean.generation import gen
from msean.io.save import prepare_batch_directory, prepare_run_directory

# TODO: Specify type of property: per layer or not
PROPERTY_CALL = {
    PropertyEnum.DEGREE_DISTRIBUTION: get_degree_dist,
    PropertyEnum.DEGREE_DISTRIBUTION_PER_LAYER: get_degree_dist_layers,
    PropertyEnum.EMBEDDEDNESS_DISTRIBUTION: get_embeddedness_dist,
    PropertyEnum.CLUSTERING_DISTRIBUTION: get_clustering_dist,
    PropertyEnum.EDGE_LENGTH_DISTRIBUTION: get_edge_len_dist,
    PropertyEnum.DENSITY: get_density,
    PropertyEnum.CLUSTERING: get_clustering,
    PropertyEnum.AVERAGE_DEGREE: get_avg_degree
}

def batch_experiment(cfg: Config, rng: np.random.Generator, param_name: str, param_range: Tuple[int, int, int], properties: List[PropertyEnum]):
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
        rng(np.random.Generator):
            Random number generator.
        param_name (str):
            Name of the parameter to vary, using dot notation
            (e.g., "connection.xi").
        param_range (Tuple[int, int, int]):
            (start, stop, step) defining the range of parameter values.
            The range is inclusive: range(start, stop + step, step)
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
    batch_paths = prepare_batch_directory(cfg)
    save_config(cfg, batch_paths["config"])

    results = {}

    # List of parameter values
    start, stop, step = param_range
    param_values = range(start, stop + step, step) # the limits are inclusive

    short_param_name = param_name.split(".")[-1]

    for param_val in param_values:
        cfg_i = deepcopy(cfg)
        set_nested(cfg_i, param_name, param_val)

        # Prepare run directory inside batch directory 
        run_name = f"{short_param_name}_{param_val}"
        run_paths = prepare_run_directory(
            cfg_i,
            parent_dir=batch_paths["batch_dir"],
            run_name=run_name
        )

        save_config(cfg_i, run_paths["config"])

        G, layers = gen(cfg_i, rng)

        results[param_val] = {}

        for prop in properties:
            # Degree distribution per layer requires individual layers as input
            if prop == PropertyEnum.DEGREE_DISTRIBUTION_PER_LAYER:
                value = PROPERTY_CALL[prop](layers)
            else:
                value = PROPERTY_CALL[prop](G)

            results[param_val][prop] = value

    return results, batch_paths