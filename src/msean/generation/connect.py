import math
import numpy as np

from msean.config import Config


SQRT2 = math.sqrt(2)


def d(pos1, pos2, cfg:Config):
    """
    Compute the Euclidean distance between two points in 2D.

    If cfg.metric_space.toroidal is True, distances are computed on a square torus
    with side length cfg.metric_space.square_side, meaning opposite edges wrap around.
    """
    
    x_dist = abs(pos1[0]-pos2[0])
    y_dist = abs(pos1[1]-pos2[1])

    if cfg.metric_space.toroidal:
        x_dist = min(x_dist, cfg.metric_space.square_side - x_dist)
        y_dist = min(y_dist, cfg.metric_space.square_side - y_dist)

    return math.sqrt(x_dist**2 + y_dist**2)

def choose_affiliation(node_pos, affiliation_embedding, cfg:Config, rng:np.random.Generator):
    """
    Sample an affiliation for a node based on spatial proximity.

    Each affiliation is assigned a weight according to an exponential decay
    function of the distance between the node and the affiliation. These
    weights are normalized to form a probability distribution, from which
    a single affiliation is sampled.

    The connection (weighting) function is:
        gamma(d) = exp(-ξ * (d / r_0)^s)

    where d is the Euclidean distance between the node and an affiliation.
    When ξ = 1 and s = 1, this reduces to:
        gamma(d) = exp(-d / r_0)

    Parameters
    ----------
    node_pos : tuple[float, float]
        The (x, y) position of the node in the embedding space.

    affiliation_embedding : dict
        Mapping from affiliation identifiers to their (x, y) positions.

    cfg : Config
        Configuration object containing connection parameters:
            - cfg.connection.xi (ξ): decay strength
            - cfg.connection.r_0: characteristic distance scale
            - cfg.connection.s: shape parameter

    rng : np.random.Generator
        NumPy random number generator used for sampling.

    Returns
    -------
    ai : hashable
        The identifier of the sampled affiliation.
    """

    w_dict = {}

    for aff in affiliation_embedding:
        affiliation_pos = affiliation_embedding[aff]
        w_dict[aff] = math.exp(-1 * d(node_pos, affiliation_pos, cfg)/(cfg.connection.alpha * SQRT2))

    z = sum([w_dict[k] for k in w_dict])

    affs = [k for k in w_dict]
    probs = [w_dict[k]/z for k in affs]

    ai = rng.choice(affs, p=probs)

    return ai
