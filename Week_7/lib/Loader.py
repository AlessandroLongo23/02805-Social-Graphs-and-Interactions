import json
import networkx as nx

class Loader():
    def __init__(self):
        pass

    def load_network(self, graph_file):
        """Load the network from JSON and create an undirected NetworkX graph."""
        print("Loading network...")
        with open(graph_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        G = nx.Graph()
        
        for node in data['nodes']:
            G.add_node(node['name'])
        
        for edge in data['edges']:
            G.add_edge(edge[0], edge[1])
        
        print(f"Network loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    def load_artist_genres(self, genres_file):
        """Load artist-genre mappings."""
        print("Loading genre data...")
        with open(genres_file, 'r', encoding='utf-8') as f:
            artist_genres = json.load(f)
        print(f"Genre data loaded for {len(artist_genres)} artists\n")
        return artist_genres

    def load_term_frequencies(self, tf_file):
        """Load term frequency data from JSON."""
        print(f"Loading term frequencies from {tf_file}...")
        with open(tf_file, 'r', encoding='utf-8') as f:
            genre_tf = json.load(f)
        print(f"Loaded TF data for {len(genre_tf)} genres\n")
        return genre_tf