import math
import numpy as np

from msean.config import Config

# Choosing an affiliation for a particular node - steps 2b

def d(pos1, pos2, toroidal=False):
    """Euclidean distance"""

    # TODO The toroidal stuff is the doughnut thing where it loops around if it goes over an edge... but I am not convinced we need this
    
    x_dist = abs(pos1[0]-pos2[0])
    if toroidal:
        x_dist = min(x_dist, 1-x_dist)
    y_dist = abs(pos1[1]-pos2[1])
    if toroidal:
        y_dist = min(y_dist, 1-y_dist)
    return math.sqrt(x_dist**2 + y_dist**2)

def choose_affiliation(node_pos, affiliation_embedding, cfg:Config, rng:np.random.Generator):
    """
    i. Calculate the connection probability weighting based on the distance between u and every affiliation ai, gamma(d(u, ai))
#   ii. Setting the probability of choosing ai to its weighting over the sum of all weightings, i.e. gamma(d(u, ai)) / Σ{j} gamma(d(u, aj))
#   iii. Sampling one affiliation from the set of possible affiliations according to its probability.
    """

    w_dict = {}

    for aff in affiliation_embedding:
        affiliation_pos = affiliation_embedding[aff]
        w_dict[aff] = math.exp(- cfg.connection.xi * ((d(node_pos, affiliation_pos)/cfg.connection.r_0)**cfg.connection.s))

    Z = sum([w_dict[k] for k in w_dict])

    affs = [k for k in w_dict]
    probs = [w_dict[k]/Z for k in affs]

    ai = rng.choice(affs, p=probs)

    return ai
