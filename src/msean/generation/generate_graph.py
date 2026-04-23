import itertools
import networkx as nx
import numpy as np

from msean.generation import distribute_nodes_uniformly, choose_affiliation
from msean.config import Config


def gen(cfg:Config):
    """
    Generate a multilayer spatial affiliation network.

    The model embeds nodes and affiliation centers in a 2D unit square and
    constructs edges based on shared affiliations across multiple layers.

    Network generation proceeds as follows:

    1. Node embedding:
       N nodes are placed uniformly at random in the unit square [0, 1]^2.

    2. Layer construction (repeated for each of L layers):
       a. K_l affiliation nodes are embedded uniformly in the same space.
       b. Each node is assigned to exactly one affiliation.
          where d is the Euclidean distance between the node and an affiliation.
       c. All nodes assigned to the same affiliation are connected, forming
          a clique within that layer.

    3. Aggregation:
       The final graph G is obtained as the union of all layer graphs, i.e.
       two nodes are connected in G if they are connected in at least one layer.

    Parameters
    ----------
    cfg : Config
        Configuration object specifying network and connection parameters:
            - cfg.network.n_nodes : int
                Number of nodes N.
            - cfg.network.n_layers : int
                Number of layers L.
            - cfg.network.n_affiliations : list[int]
                Number of affiliations K_l for each layer.
            - cfg.seed : int
                Random seed for reproducibility.

    Returns
    -------
    G : networkx.Graph
        The aggregated graph obtained by composing all layers.

    layers : list[networkx.Graph]
        A list of graphs, where each graph corresponds to a single layer
        before aggregation.
    """

    rng = np.random.default_rng(cfg.seed)
    # I add a prefix to node names so you don't get a node and affiliation called the same thing.
    node_embedding = distribute_nodes_uniformly(cfg.network.n_nodes, cfg, rng, label_prefix='u')
    node_labels = list(node_embedding.keys())
    layers = []

    for l in range(cfg.network.n_layers):
        affiliation_embedding = distribute_nodes_uniformly(cfg.network.n_affiliations[l], cfg, rng, label_prefix='A')

        # affiliation -> list of nodes it was assigned to
        aff_connections = {}

        for node in node_labels:
            ai = choose_affiliation(node_embedding[node], affiliation_embedding, cfg, rng)
            aff_connections.setdefault(ai, []).append(node)
    
        Gl = nx.Graph()
        Gl.add_nodes_from(node_labels)

        for members in aff_connections.values():
            if len(members) > 1:
                Gl.add_edges_from(itertools.combinations(members, 2))
        
        layers.append(Gl)

    G = nx.compose_all(layers)
    nx.set_node_attributes(G, node_embedding, 'embedding')
    return G, layers
