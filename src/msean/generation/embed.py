import numpy as np

from msean.config import Config

# Embedding nodes - steps 1 and 2a

def distribute_nodes_uniformly(N:int, cfg:Config, rng:np.random.Generator, label_prefix:str='') -> dict:
    """
    Distribute N nodes uniformly in the 2D plane mu_min <= x <= mu_max, mu_min <= y <=mu_max.
    
    Arguments:
        N (int): The number of nodes.
        rng (np.random.Generator): Random number generator.
        cfg (msean.Config): Model configuration.

    Returns:
        dict: Mapping of nodes 0 to N-1 to (x,y)-coordinates.
    """

    mu_min = 0 # minimum value of x and y coordinates.
    mu_max = mu_min + cfg.metric_space.square_side # maximum value of x and y coordinates

    return {f'{label_prefix}{i}':
            (rng.uniform(mu_min, mu_max), rng.uniform(mu_min, mu_max)) for i in range(N)}
