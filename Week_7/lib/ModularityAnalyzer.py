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

# def main():
#     modularity_analyzer = ModularityAnalyzer()
#     loader = Loader()

#     print("="*70)
#     print("MODULARITY ANALYSIS: How Community-Like Are the Genres?")
#     print("="*70)
    
#     # Load data
#     graph_file = "../data/rock/performers_graph.json"
#     genres_file = "./Week_7/artist_genres.json"
    
#     G = loader.load_network(graph_file)
#     artist_genres = loader.load_artist_genres(genres_file)
    
#     # Filter network to only nodes with genre information
#     G_filtered = modularity_analyzer.filter_network_by_genres(G, artist_genres)
    
#     # Filter artist_genres to only include nodes in the filtered graph
#     valid_artists = set(G_filtered.nodes())
#     artist_genres_filtered = {k: v for k, v in artist_genres.items() if k in valid_artists}
    
#     print("\n" + "="*70)
#     print("CONCEPT OF MODULARITY")
#     print("="*70)
#     print("""
# Modularity measures how well a network is divided into communities. It compares 
# the actual number of edges within communities to the expected number if edges 
# were randomly distributed while preserving node degrees.

# According to equation 9.12 from the Network Science book:
#     M = 1/(2L) * Σ(A_ij - k_i*k_j/(2L)) * δ(c_i, c_j)

# Where:
# - M ranges from -1 to 1
# - M > 0.3 typically indicates significant community structure
# - M ≈ 0 suggests no better than random assignment
# - M < 0 suggests anti-community structure (nodes in same group avoid each other)

# Reference: http://networksciencebook.com/chapter/9
# """)
    
#     # Strategy 1: First genre
#     print("\n" + "="*70)
#     print("STRATEGY 1: First Genre Assignment")
#     print("="*70)
#     print("Each artist is assigned their FIRST genre from the list.\n")
    
#     partition_first = modularity_analyzer.assign_genres_first(artist_genres_filtered)
#     modularity_analyzer.analyze_partition(partition_first)
    
#     print("\nCalculating modularity...")
#     modularity_first = modularity_analyzer.calculate_modularity(G_filtered, partition_first)
#     print(f"\n>>> MODULARITY (First Genre): {modularity_first:.4f}")
    
#     if modularity_first > 0.3:
#         print("    Interpretation: Strong community structure")
#     elif modularity_first > 0:
#         print("    Interpretation: Weak to moderate community structure")
#     else:
#         print("    Interpretation: No meaningful community structure")
    
#     # Strategy 2: First non-rock genre
#     print("\n" + "="*70)
#     print("STRATEGY 2: First Non-Rock Genre Assignment")
#     print("="*70)
#     print("Each artist is assigned their FIRST NON-ROCK genre (if available).\n")
#     print("Rationale: Since most artists have 'rock' as their first genre,")
#     print("           this may reveal more specific genre communities.\n")
    
#     partition_non_rock = modularity_analyzer.assign_genres_first_non_rock(artist_genres_filtered)
#     modularity_analyzer.analyze_partition(partition_non_rock)
    
#     print("\nCalculating modularity...")
#     modularity_non_rock = modularity_analyzer.calculate_modularity(G_filtered, partition_non_rock)
#     print(f"\n>>> MODULARITY (First Non-Rock Genre): {modularity_non_rock:.4f}")
    
#     if modularity_non_rock > 0.3:
#         print("    Interpretation: Strong community structure")
#     elif modularity_non_rock > 0:
#         print("    Interpretation: Weak to moderate community structure")
#     else:
#         print("    Interpretation: No meaningful community structure")
    
#     # Strategy 3: Random genre
#     print("\n" + "="*70)
#     print("STRATEGY 3: Random Genre Assignment")
#     print("="*70)
#     print("Each artist is assigned a RANDOM genre from their list.\n")
#     print("Rationale: This provides a baseline and accounts for genre diversity.\n")
    
#     partition_random = modularity_analyzer.assign_genres_random(artist_genres_filtered)
#     modularity_analyzer.analyze_partition(partition_random)
    
#     print("\nCalculating modularity...")
#     modularity_random = modularity_analyzer.calculate_modularity(G_filtered, partition_random)
#     print(f"\n>>> MODULARITY (Random Genre): {modularity_random:.4f}")
    
#     if modularity_random > 0.3:
#         print("    Interpretation: Strong community structure")
#     elif modularity_random > 0:
#         print("    Interpretation: Weak to moderate community structure")
#     else:
#         print("    Interpretation: No meaningful community structure")
    
#     # Compare results
#     print("\n" + "="*70)
#     print("COMPARISON AND DISCUSSION")
#     print("="*70)
    
#     print(f"\n{'Strategy':<40} {'Modularity':>12}")
#     print("-"*55)
#     print(f"{'First Genre':<40} {modularity_first:>12.4f}")
#     print(f"{'First Non-Rock Genre':<40} {modularity_non_rock:>12.4f}")
#     print(f"{'Random Genre':<40} {modularity_random:>12.4f}")
    
#     print("\n" + "="*70)
#     print("KEY FINDINGS:")
#     print("="*70)
    
#     # Determine which is best
#     best_modularity = max(modularity_first, modularity_non_rock, modularity_random)
    
#     if best_modularity == modularity_first:
#         best_strategy = "First Genre"
#     elif best_modularity == modularity_non_rock:
#         best_strategy = "First Non-Rock Genre"
#     else:
#         best_strategy = "Random Genre"
    
#     print(f"\n1. Best performing strategy: {best_strategy} (M = {best_modularity:.4f})")
    
#     improvement_non_rock = ((modularity_non_rock - modularity_first) / abs(modularity_first) * 100 
#                            if modularity_first != 0 else float('inf'))
#     improvement_random = ((modularity_random - modularity_first) / abs(modularity_first) * 100 
#                          if modularity_first != 0 else float('inf'))
    
#     print(f"\n2. Improvement from first genre to first non-rock: {improvement_non_rock:+.1f}%")
#     print(f"   Improvement from first genre to random: {improvement_random:+.1f}%")
    
#     print("\n3. Are genres good communities?")
#     if best_modularity > 0.3:
#         print("   YES - Genres show strong community structure in the network.")
#         print("   Artists within the same genre tend to be more connected than expected by chance.")
#     elif best_modularity > 0.1:
#         print("   SOMEWHAT - Genres show weak to moderate community structure.")
#         print("   There is some clustering by genre, but it's not very pronounced.")
#     else:
#         print("   NO - Genres do not correspond well to network communities.")
#         print("   The network structure doesn't align well with genre classifications.")
    
#     print("\n4. Interpretation:")
#     print("   - If modularity increased with non-rock genres: The 'rock' label is too")
#     print("     broad and masks more specific subgenre communities.")
#     print("   - If random assignment performed similarly: Multiple genres per artist")
#     print("     reflect the network's interconnected nature.")
#     print("   - Low modularity overall suggests: The network is formed by shared")
#     print("     influences/collaborations that cross genre boundaries.")
    
#     print("\n" + "="*70)
#     print("Analysis complete!")
#     print("="*70)
    
#     # Save results
#     results = {
#         "network_stats": {
#             "total_nodes": G_filtered.number_of_nodes(),
#             "total_edges": G_filtered.number_of_edges()
#         },
#         "modularity_scores": {
#             "first_genre": modularity_first,
#             "first_non_rock_genre": modularity_non_rock,
#             "random_genre": modularity_random
#         },
#         "best_strategy": best_strategy,
#         "best_modularity": best_modularity
#     }
    
#     with open("Week_7/modularity_results.json", 'w', encoding='utf-8') as f:
#         json.dump(results, f, indent=2)
    
#     print("\nResults saved to: Week_7/modularity_results.json")

# if __name__ == "__main__":
#     main()

