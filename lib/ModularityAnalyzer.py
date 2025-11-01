import json
import networkx as nx
import numpy as np
import random
from collections import defaultdict, Counter
from lib.Loader import Loader

class ModularityAnalyzer():
    def __init__(self):
        pass

    def filter_network_by_genres(self, G, artist_genres):
        """Keep only nodes that have genre information."""
        nodes_with_genres = set(artist_genres.keys())
        nodes_in_graph = set(G.nodes())
        
        # Find intersection
        valid_nodes = nodes_with_genres.intersection(nodes_in_graph)
        
        # Create subgraph
        G_filtered = G.subgraph(valid_nodes).copy()
        
        print(f"\nFiltered network: {G_filtered.number_of_nodes()} nodes, {G_filtered.number_of_edges()} edges")
        print(f"Nodes removed: {G.number_of_nodes() - G_filtered.number_of_nodes()}")
        
        return G_filtered

    def calculate_modularity(self, G, partition):
        """
        Calculate modularity using equation 9.12 from Network Science book.
        
        M = 1/(2L) * Σ(A_ij - k_i*k_j/(2L)) * δ(c_i, c_j)
        
        Where:
        - L is the total number of links
        - A_ij is the adjacency matrix (1 if there's a link, 0 otherwise)
        - k_i is the degree of node i
        - δ(c_i, c_j) is 1 if nodes i and j are in the same community, 0 otherwise
        
        Args:
            G: NetworkX graph (undirected)
            partition: Dictionary mapping node names to community labels
        
        Returns:
            Modularity value M
        """
        # Total number of edges (links)
        L = G.number_of_edges()
        
        if L == 0:
            return 0.0
        
        # Get degrees
        degrees = dict(G.degree())
        
        # Calculate modularity
        modularity = 0.0
        
        for node_i in G.nodes():
            for node_j in G.nodes():
                # Check if nodes are in the same community
                if partition[node_i] == partition[node_j]:
                    # A_ij: 1 if edge exists, 0 otherwise
                    A_ij = 1 if G.has_edge(node_i, node_j) else 0
                    
                    # Expected number of edges
                    k_i = degrees[node_i]
                    k_j = degrees[node_j]
                    expected = (k_i * k_j) / (2 * L)
                    
                    modularity += (A_ij - expected)
        
        # Normalize by 2L
        modularity = modularity / (2 * L)
        
        return modularity

    def assign_genres_first(self, artist_genres):
        """Assign each artist their first genre."""
        partition = {}
        for artist, genres in artist_genres.items():
            if genres:
                partition[artist] = genres[0]
        return partition

    def assign_genres_first_non_rock(self, artist_genres):
        """Assign first non-rock genre, or first genre if all are rock or only one genre."""
        partition = {}
        for artist, genres in artist_genres.items():
            if not genres:
                continue
            
            # Find first non-rock genre
            non_rock_genres = [g for g in genres if 'rock' not in g.lower()]
            
            if non_rock_genres:
                partition[artist] = non_rock_genres[0]
            else:
                # All genres contain 'rock' or only one genre
                partition[artist] = genres[0]
        
        return partition

    def assign_genres_random(self, artist_genres, seed=42):
        """Assign a random genre from each artist's genre list."""
        random.seed(seed)
        partition = {}
        for artist, genres in artist_genres.items():
            if genres:
                partition[artist] = random.choice(genres)
        return partition

    def analyze_partition(self, partition):
        """Analyze the partition to show community statistics."""
        community_counts = Counter(partition.values())
        
        print(f"  Number of communities: {len(community_counts)}")
        print(f"  Largest community: {community_counts.most_common(1)[0] if community_counts else 'N/A'}")
        print(f"  Top 10 communities:")
        for i, (genre, count) in enumerate(community_counts.most_common(10), 1):
            print(f"    {i:2d}. {genre:<30} {count:>4} artists")