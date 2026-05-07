from enum import Enum

import numpy as np
import networkx as nx
from msean.generation import d

class PropertyEnum(Enum):
    DEGREE_DISTRIBUTION = 1
    DEGREE_DISTRIBUTION_PER_LAYER = 2
    EMBEDDEDNESS_DISTRIBUTION = 3
    CLUSTERING_DISTRIBUTION = 4
    EDGE_LENGTH_DISTRIBUTION = 5
    DENSITY = 6
    CLUSTERING = 7
    AVERAGE_DEGREE = 8


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
    
def get_degree_dist_layers(layers: list):
    """
    Returns a list of degree distribution arrays, one per layer.
    Each array has shape (n_unique_degrees, 2):
    [[degree, frequency], ...]
    """
    return [get_degree_dist(layer) for layer in layers]

def get_embeddedness_dist(G: nx.Graph):
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

def get_clustering_dist(G:nx.Graph, res:int=3):
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

def get_edge_len_dist(G:nx.Graph, res:int=3):
    """
    Returns the edge length distribution as a 2D numpy array:
    [[edge length, frequency], ...]
    Records the clustering coefficient up to res decimal places.
    """
    edge_len = [round(d(G.nodes[e[0]]['embedding'], G.nodes[e[1]]['embedding']), res) for e in G.edges()]
    
    # Count frequencies
    unique_lens, counts = np.unique(edge_len, return_counts=True)
    
    # Stack into 2D array
    edge_len_dist = np.column_stack((unique_lens, counts))
    
    return edge_len_dist



# Global properties

def get_density(G:nx.Graph):
    """
    Returns the density as float.
    """
    return nx.density(G)

def get_clustering(G:nx.Graph):
    """
    Returns the global clustering coefficient.
    """
    return nx.transitivity(G)

def get_avg_degree(G:nx.Graph):
    """
    Returns the average degree of the graph.
    """
    return np.mean([d for _, d in G.degree()])