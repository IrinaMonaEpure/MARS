from enum import Enum
import numpy as np
import networkx as nx
from typing import List

from msean.generation import d
from msean.config import Config

class PropertyEnum(Enum):
    DEGREE_DISTRIBUTION = 1
    DEGREE_DISTRIBUTION_PER_LAYER = 2
    EMBEDDEDNESS_DISTRIBUTION = 3
    LOCAL_CLUSTERING_DISTRIBUTION = 4
    EDGE_LENGTH_DISTRIBUTION = 5
    EXCESS_CLOSURE_DISTRIBUTION = 6
    DENSITY = 7
    DENSITY_PER_LAYER = 8
    GLOBAL_CLUSTERING = 9
    AVERAGE_LOCAL_CLUSTERING = 10
    AVERAGE_DEGREE = 11
    AVERAGE_DEGREE_PER_LAYER = 12
    TRIANGLES = 13


# Distributions

def get_degree_dist(G:nx.Graph):
    """
    Returns the degree distribution as a 2D numpy array:
    [[degree, frequency], ...]
    """
    degrees = [d for _, d in G.degree()]
    
    # Count frequencies
    unique_degrees, counts = np.unique(degrees, return_counts=True)
    
    # Stack into 2D array
    deg_dist = np.column_stack((unique_degrees, counts))
    
    return deg_dist
    
def get_degree_dist_layers(layers:List[nx.Graph]):
    """
    Returns a list of degree distribution arrays, one per layer.
    Each array has shape (n_unique_degrees, 2):
    [[degree, frequency], ...]
    """
    return [get_degree_dist(layer) for layer in layers]

def get_embeddedness_dist(G:nx.Graph):
    """
    Returns the embeddedness distribution as a 2D numpy array:
    [[embeddedness, frequency], ...]
    """
    gdeg = nx.generalized_degree(G)

    triangle_multiplicity_list = [
        k
        for v in gdeg
        for k in gdeg[v]
        for _ in range(gdeg[v][k])
    ]

    triangle_multiplicity_dist = {
        k: int(triangle_multiplicity_list.count(k) / 2)
        for k in set(triangle_multiplicity_list)
    }

    vals = np.array(sorted(triangle_multiplicity_dist.keys()))
    freqs = np.array([triangle_multiplicity_dist[k] for k in vals])

    return np.column_stack((vals, freqs))

def get_local_clustering_dist(G:nx.Graph, res:int=2):
    """
    Returns the local clustering coefficient distribution as a 2D numpy array:
    [[clustering coefficient, frequency], ...]
    Records the clustering coefficient up to res decimal places.
    """
    clus_dict = nx.clustering(G)
    clus = [round(clus_dict[k], res) for k in clus_dict]
    
    # Count frequencies
    unique_coefs, counts = np.unique(clus, return_counts=True)
    
    # Stack into 2D array
    clus_dist = np.column_stack((unique_coefs, counts))
    
    return clus_dist

def get_edge_len_dist(G:nx.Graph, cfg: Config, res:int=2):
    """
    Returns the edge length distribution as a 2D numpy array:
    [[edge length, frequency], ...]
    Records the edge length up to res decimal places.
    """
    edge_len = [round(d(G.nodes[e[0]]['embedding'], G.nodes[e[1]]['embedding'], cfg), res) for e in G.edges()]
    
    # Count frequencies
    unique_lens, counts = np.unique(edge_len, return_counts=True)
    
    # Stack into 2D array
    edge_len_dist = np.column_stack((unique_lens, counts))
    
    return edge_len_dist

def get_excess_closure_dist(G:nx.Graph, layers:List[nx.Graph], res:int=2):
    node_labels = list(G.nodes())

    T_pure = _T_pure(layers, node_labels)
    T_unique = _T(G, node_labels)
    p = _P(G, layers, node_labels)

    c_pure = np.divide(
        T_pure,
        p,
        out=np.zeros_like(T_pure, dtype=float)
    )
    c_unique = np.divide(
        T_unique,
        p,
        out=np.zeros_like(T_unique, dtype=float)
    )
    c_excess = np.divide(
        c_unique - c_pure,
        1 - c_pure,
        out=np.zeros_like(c_unique),
        where=(1 - c_pure) != 0,
    )

    c_excess = c_excess[np.isfinite(c_excess)]
    c_excess = c_excess[(c_excess >= 0) & (c_excess <= 1)]

    np.round(c_excess, decimals=res, out=c_excess)

    # Count frequencies
    val, freq = np.unique(c_excess, return_counts=True)
    
    # Stack into 2D array
    c_excess_dist = np.column_stack((val, freq))
    
    return c_excess_dist


# Global properties

def get_density(G:nx.Graph):
    """
    Returns the density as float.
    """
    return nx.density(G)

def get_density_layers(layers:List[nx.Graph]):
    """
    Returns a list of density values per layer.
    """
    return [nx.density(l) for l in layers]

def get_global_clustering(G:nx.Graph):
    """
    Returns the global clustering coefficient.
    """
    return nx.transitivity(G)

def get_avg_local_clustering(G: nx.Graph):
    """
    Returns the average local clustering.
    """
    return nx.average_clustering(G)

def get_avg_degree(G:nx.Graph):
    """
    Returns the average degree of the graph.
    """
    return np.mean([d for _, d in G.degree()])

def get_avg_degree_layers(layers:List[nx.Graph]):
    """
    Returns a list of average degrees per layer.
    """
    return [np.mean([d for _, d in l.degree()]) for l in layers]

def get_triangles(G:nx.Graph):
    """
    Returns the number of triangles in the graph.
    """
    return sum(nx.triangles(G).values()) // 3


# Utils for excess closure

def _T(G:nx.Graph, node_labels:List[str]):
    """
    Return array of numbers of triangles around each node in node_labels.
    """
    triangles = nx.triangles(G)

    return np.array(
        [triangles[node] for node in node_labels],
        dtype=int
    )

def _T_pure(layers:List[nx.Graph], node_labels:List[str]):
    """
    Sum of _T over all individual layers.
    """
    T_pure = np.zeros(len(node_labels), dtype=int)

    for l in layers:
        T_pure += _T(l, node_labels)

    return T_pure

def _P(G:nx.Graph, layers:List[nx.Graph], node_labels:List[str]):
    """
    Array containing the number of possible different alter tie pairs around each node.
    """
    k = np.array([d for _, d in G.degree()])
    binom_k_2 = k * (k - 1) / 2

    n = len(node_labels)
    sum_Al = np.zeros((n, n), dtype=int)
    for l in layers:
        sum_Al += nx.to_numpy_array(l, nodelist=node_labels, dtype=int)

    binom_sumAl_2 = sum_Al * (sum_Al - 1) / 2
    sum_binom_sumAl_2 = binom_sumAl_2.sum(axis=0)

    return binom_k_2 - sum_binom_sumAl_2

def excess_closure_by_node(G, layers):
    node_labels = list(G.nodes())

    t_pure = _T_pure(layers, node_labels)
    t_unique = _T(G, node_labels)
    p = _P(G, layers, node_labels)

    c_pure = np.divide(
        t_pure, p,
        out=np.zeros_like(t_pure, dtype=float),
        where=p != 0,
    )

    c_unique = np.divide(
        t_unique, p,
        out=np.zeros_like(t_unique, dtype=float),
        where=p != 0,
    )

    c_excess = np.divide(
        c_unique - c_pure,
        1 - c_pure,
        out=np.zeros_like(c_unique, dtype=float),
        where=(1 - c_pure) != 0,
    )

    print("min/max c_pure:", c_pure.min(), c_pure.max())
    print("min/max c_unique:", c_unique.min(), c_unique.max())
    print("min/max c_excess:", c_excess.min(), c_excess.max())
    print("any c_pure > c_unique?", np.any(c_pure > c_unique))
    print("any c_pure > 1?", np.any(c_pure > 1))
    print("any c_unique > 1?", np.any(c_unique > 1))

    c_excess = np.clip(c_excess, 0.0, 1.0)

    return dict(zip(node_labels, c_excess))