import networkx as nx
import numpy as np
import random

def barabasi_albert_graph(n, preferential=True, directed=False):
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_edge(0, 1)
    while G.number_of_nodes() < n:
        if (preferential):
            edge_list_flatten = np.array(G.edges()).flatten()
            G.add_edge(G.number_of_nodes(), random.choice(edge_list_flatten))
        else:
            node_list = np.array(G.nodes())
            G.add_edge(G.number_of_nodes(), random.choice(node_list))

    return G