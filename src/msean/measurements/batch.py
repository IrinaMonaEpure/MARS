from typing import List, Tuple
from copy import deepcopy

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

from msean.config import Config, set_nested
from msean.generation import gen

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

def batch_experiment(cfg: Config, param_name: str, param_range: Tuple, properties: List[PropertyEnum]):
    results = {}

    for param_val in range(param_range):
        cfg_i = deepcopy(cfg)
        set_nested(cfg_i, param_name, param_val)
        G = gen(cfg_i)

        for prop in properties:
            results.setdefault(param_val, {})[prop] = PROPERTY_CALL[prop](G)

    return results