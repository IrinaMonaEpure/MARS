import numpy as np

# Embedding nodes - steps 1 and 2a

def distribute_nodes_uniformly(N:int, mu_min:float=0, mu_max:float=1, label_prefix:str='') -> dict:
    """
    Distribute N nodes uniformly in the 2D plane mu_min <= x <= mu_max, mu_min <= y <=mu_max.
    
    Arguments:
        N (int): The number of nodes.
        mu_min (float): The minimum value of x and y coordinates.
        mu_max (float): The maximum value of x and y coordinates.

    Returns:
        dict: Mapping of nodes 0 to N-1 to (x,y)-coordinates.
    """
    return {f'{label_prefix}{i}': (np.random.uniform(mu_min, mu_max), np.random.uniform(mu_min, mu_max)) for i in range(N)}
