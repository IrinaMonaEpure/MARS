import numpy as np
import networkx as nx

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
    return [get_degree_dist(G) for G in layers]

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