import networkx as nx
import matplotlib.pyplot as plt

def plot_degree_dist(G:nx.Graph, show:bool = False):
    """
    Plots the degree distribution of a network G as a line graph on a log-log scale.

    Arguments:
        G (nx.Graph): The network G.
        show (bool): If True, the plot is displayed. Default is False, in which case plt.show() should be called elsewhere to display the figure.

    Returns:
        None
    """
    # Create a new figure for the degree distribution.
    _ = plt.figure()

    # Plot the degree distribution
    degrees = [G.degree(n) for n in G.nodes()]
    # Find the frequencies and bins from a histogram plot, but clears the histogram plot as we want a line graph instead.
    freqs, bin_boundaries, _ = plt.hist(degrees) 
    plt.clf()
    # Extract the midpoints of the bins of the histogram.
    bin_size = bin_boundaries[1] - bin_boundaries[0]
    bin_mps = [bin_boundaries[i] + 0.5*bin_size for i in range(len(freqs))]
    # Plot the frequencies against the midpoints.
    plt.plot(bin_mps, freqs, color='navy')

    # Specify the settings of the figure.
    plt.title("Total Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    # The figure uses a log-log scale.
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1, plt.xlim()[1])
    plt.ylim(1, plt.ylim()[1])

    if show: plt.show()

def plot_degree_dist_layers(layers:list, labels:list, show:bool = False):
    """
    Plots the degree distribution of each layer of a multiplex network as a line graph on a log-log scale.

    Arguments:
        layers (list): a list of nx.Graphs representing each layer of the multiplex network.
        labels (list): a list of labels, where the i-th label is the label for the i-th layer of the network.
        show (bool): if True, the plot is displayed. Default is False, in which case plt.show() should be called elsewhere to display the figure.

    Returns:
        None
    """
    # Create a new figure for the degree distributions of the layers.
    _ = plt.figure()

    # Initialise lists to store the x and ys to plot for each layer.
    xs = []
    ys = []

    for G in layers:
        # Find the degree distribution.
        degrees = [G.degree(n) for n in G.nodes()]
        
        # Find the frequencies and bins from a histogram plot for each layer.
        # To plot the line graph, we are interested in the midpoints of each bin.
        freqs, bin_boundaries, _ = plt.hist(degrees)
        bin_size = bin_boundaries[1] - bin_boundaries[0]
        bin_mps = [bin_boundaries[i] + 0.5*bin_size for i in range(len(freqs))]

        # Add the midpoints and frequencies to a list 
        xs.append(bin_mps)
        ys.append(freqs)

    # Clear the histogram plots and plot the line graphs for each layer.
    plt.clf()
    for x, y, label in zip(xs, ys, labels):
        plt.plot(x, y, label=label)

    # Specify the settings of the figure.
    plt.title("Node Degree by Layer")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    # The figure uses a log-log scale.
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.xlim(1, plt.xlim()[1])
    plt.ylim(1, plt.ylim()[1])

    if show: plt.show()

def plot_embeddedness(G:nx.Graph, show:bool = False):
    """
    Plots the embeddedness distribution of edges in a network G. 

    Arguments: 
        G (nx.Graph): the network.
        show (bool): if True, the plot is displayed. Default is False, in which case plt.show() must be called elsewhere in the function. 

    Returns:
        None
    """
    # Create a new figure for the degree distributions of the layers.
    _ = plt.figure()

    # Calculate the embeddedness (triangle multiplicity) of each edge.
    gdeg = nx.generalized_degree(G)
    triangle_multiplicity_list = [k for v in gdeg for k in gdeg[v] for _ in range(gdeg[v][k])]
    triangle_multiplicity_dist = {k : int(triangle_multiplicity_list.count(k)/2) for k in set(triangle_multiplicity_list)}
    triangle_multiplicity_hist = [k for k in triangle_multiplicity_dist for _ in range(triangle_multiplicity_dist[k])]
    
    # Find the frequencies and bins from a histogram plot, but clear the histogram plot as we want a line graph instead.
    freqs, bin_boundaries, _ = plt.hist(triangle_multiplicity_hist) 
    plt.clf()
    # Extract the midpoints of the bins of the histogram.
    bin_size = bin_boundaries[1] - bin_boundaries[0]
    bin_mps = [bin_boundaries[i] + 0.5*bin_size for i in range(len(freqs))]
    # Plot the frequencies against the midpoints.
    plt.plot(bin_mps, freqs, color='navy')

    # Specify the settings of the figure.
    plt.title("Edge Embeddedness")
    plt.xlabel("Embeddedness")
    plt.ylabel("Frequency")
    # The figure uses a log-log scale.
    plt.xlim(1, plt.xlim()[1])
    plt.ylim(1, plt.ylim()[1])

    if show: plt.show()