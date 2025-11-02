import json
import networkx as nx

class Loader():
    def __init__(self):
        pass

    def load_network(self, graph_file, directed=False):
        """Load the network from JSON and create an undirected NetworkX graph."""
        with open(graph_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if directed:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        
        for node in data['nodes']:
            G.add_node(node['name'], url=node['url'], length_of_content=node['length_of_content'])
        
        for edge in data['edges']:
            G.add_edge(edge[0], edge[1])
        
        print(f"Network loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    def load_community_results(self, community_file):
        """Load community detection results."""
        with open(community_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        print(f"Community data loaded: {results['community_stats']['num_communities']} communities")
        return results

    def load_partition(self, partition_file):
        """Load the full partition from saved file."""
        with open(partition_file, 'r', encoding='utf-8') as f:
            partition = json.load(f)
        
        print(f"Partition loaded for {len(partition)} artists")
        return partition

    def load_artist_genres(self, genres_file):
        """Load artist-genre mappings."""
        with open(genres_file, 'r', encoding='utf-8') as f:
            artist_genres = json.load(f)
        print(f"Genre data loaded for {len(artist_genres)} artists\n")
        return artist_genres

    def load_term_frequencies(self, tf_file):
        """Load term frequency data from JSON."""
        with open(tf_file, 'r', encoding='utf-8') as f:
            genre_tf = json.load(f)
        print(f"Loaded TF data for {len(genre_tf)} genres\n")
        return genre_tf