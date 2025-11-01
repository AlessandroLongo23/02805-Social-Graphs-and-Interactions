import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import Counter
import numpy as np

def load_network(graph_file):
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

def load_artist_genres(genres_file):
    """Load artist-genre mappings."""
    print("Loading genre data...")
    with open(genres_file, 'r', encoding='utf-8') as f:
        artist_genres = json.load(f)
    print(f"Genre data loaded for {len(artist_genres)} artists")
    return artist_genres

def filter_network_by_genres(G, artist_genres):
    """Keep only nodes that have genre information."""
    nodes_with_genres = set(artist_genres.keys())
    nodes_in_graph = set(G.nodes())
    valid_nodes = nodes_with_genres.intersection(nodes_in_graph)
    G_filtered = G.subgraph(valid_nodes).copy()
    
    print(f"\nFiltered network: {G_filtered.number_of_nodes()} nodes, {G_filtered.number_of_edges()} edges")
    return G_filtered

def detect_communities_louvain(G, seed=42):
    """
    Detect communities using the Louvain algorithm.
    
    The Louvain algorithm is a greedy optimization method that attempts to optimize
    the modularity of a partition of the network.
    
    Reference: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.louvain.louvain_communities.html
    
    Args:
        G: NetworkX graph
        seed: Random seed for reproducibility
    
    Returns:
        communities: List of sets, each containing nodes in a community
        partition: Dictionary mapping node names to community IDs
    """
    print("\nDetecting communities using Louvain algorithm...")
    
    # Run Louvain community detection
    communities = nx.community.louvain_communities(G, seed=seed)
    
    # Convert to partition dictionary (node -> community_id)
    partition = {}
    for comm_id, comm_nodes in enumerate(communities):
        for node in comm_nodes:
            partition[node] = comm_id
    
    print(f"Number of communities detected: {len(communities)}")
    
    # Analyze community sizes
    comm_sizes = [len(comm) for comm in communities]
    print(f"Largest community size: {max(comm_sizes)}")
    print(f"Smallest community size: {min(comm_sizes)}")
    print(f"Average community size: {np.mean(comm_sizes):.1f}")
    
    # Show top 10 largest communities
    sorted_communities = sorted(enumerate(communities), key=lambda x: len(x[1]), reverse=True)
    print(f"\nTop 10 largest communities:")
    for i, (comm_id, comm_nodes) in enumerate(sorted_communities[:10], 1):
        print(f"  {i:2d}. Community {comm_id:3d}: {len(comm_nodes):4d} nodes")
    
    return communities, partition

def calculate_modularity_networkx(G, partition):
    """
    Calculate modularity using NetworkX's built-in function.
    This serves as validation for our manual implementation.
    """
    # Convert partition dict to list of sets
    communities = {}
    for node, comm_id in partition.items():
        if comm_id not in communities:
            communities[comm_id] = set()
        communities[comm_id].add(node)
    
    community_list = list(communities.values())
    modularity = nx.community.modularity(G, community_list)
    
    return modularity

def compare_with_genre_modularity(structural_modularity):
    """Load and compare with previous genre-based modularity results."""
    try:
        with open("Week_7/modularity_results.json", 'r', encoding='utf-8') as f:
            genre_results = json.load(f)
        
        print("\n" + "="*70)
        print("COMPARISON WITH GENRE-BASED COMMUNITIES")
        print("="*70)
        
        print(f"\n{'Method':<40} {'Modularity':>12}")
        print("-"*55)
        print(f"{'Structural (Louvain)':<40} {structural_modularity:>12.4f}")
        print(f"{'Genre - First Genre':<40} {genre_results['modularity_scores']['first_genre']:>12.4f}")
        print(f"{'Genre - First Non-Rock':<40} {genre_results['modularity_scores']['first_non_rock_genre']:>12.4f}")
        print(f"{'Genre - Random':<40} {genre_results['modularity_scores']['random_genre']:>12.4f}")
        
        best_genre_mod = max(genre_results['modularity_scores'].values())
        improvement = ((structural_modularity - best_genre_mod) / best_genre_mod * 100 
                      if best_genre_mod != 0 else float('inf'))
        
        print(f"\n{'='*70}")
        print("INTERPRETATION")
        print("="*70)
        
        print(f"\nStructural communities show {improvement:+.1f}% modularity vs best genre method")
        
        if structural_modularity > best_genre_mod + 0.05:
            print("\n✓ The Louvain algorithm found SIGNIFICANTLY better communities than genres.")
            print("  This suggests that:")
            print("  - Network structure is driven by factors beyond just genre")
            print("  - Collaborations, influences, and connections cross genre boundaries")
            print("  - The 'true' communities are based on the actual network topology")
        elif structural_modularity > best_genre_mod:
            print("\n✓ The Louvain algorithm found slightly better communities than genres.")
            print("  This suggests genres partially capture network structure, but not completely.")
        else:
            print("\n✓ Genre-based communities are comparable to structural communities.")
            print("  This suggests genres are a good proxy for network structure.")
        
        if structural_modularity > 0.3:
            print(f"\nModularity of {structural_modularity:.4f} indicates STRONG community structure.")
        elif structural_modularity > 0.1:
            print(f"\nModularity of {structural_modularity:.4f} indicates MODERATE community structure.")
        else:
            print(f"\nModularity of {structural_modularity:.4f} indicates WEAK community structure.")
            
    except FileNotFoundError:
        print("\nNote: Run modularity_analysis.py first to compare with genre-based results.")

def visualize_communities(G, partition, output_file="Week_7/community_network_visualization.png", 
                         max_colored_communities=10, figsize=(20, 20)):
    """
    Visualize the network with nodes colored by community.
    
    Args:
        G: NetworkX graph
        partition: Dictionary mapping nodes to community IDs
        output_file: Path to save the visualization
        max_colored_communities: Number of largest communities to color uniquely
        figsize: Figure size tuple
    """
    print("\n" + "="*70)
    print("VISUALIZING NETWORK")
    print("="*70)
    
    # Count community sizes
    comm_sizes = Counter(partition.values())
    largest_communities = [comm_id for comm_id, _ in comm_sizes.most_common(max_colored_communities)]
    
    print(f"\nColoring the {max_colored_communities} largest communities")
    print(f"Remaining nodes will be shown in light gray\n")
    
    # Create color map
    # Use distinct colors for largest communities
    colors_list = list(mcolors.TABLEAU_COLORS.keys())
    if max_colored_communities > len(colors_list):
        # Add more colors if needed
        colors_list.extend(list(mcolors.CSS4_COLORS.keys()))
    
    node_colors = []
    for node in G.nodes():
        comm_id = partition[node]
        if comm_id in largest_communities:
            # Assign unique color to large communities
            color_idx = largest_communities.index(comm_id)
            node_colors.append(colors_list[color_idx % len(colors_list)])
        else:
            # Light gray for small communities
            node_colors.append('lightgray')
    
    # Calculate layout using ForceAtlas2
    print("Calculating network layout (this may take a while for large networks)...")
    print("Using ForceAtlas2 layout...")
    
    try:
        # Try to use ForceAtlas2 layout if available
        if G.number_of_nodes() > 1000:
            print(f"Note: Network has {G.number_of_nodes()} nodes, using optimized layout...")
            pos = nx.forceatlas2_layout(G, max_iter=500, seed=42)
        else:
            pos = nx.forceatlas2_layout(G, max_iter=500, seed=42)
    except AttributeError:
        # Fallback to spring layout if ForceAtlas2 is not available
        print("ForceAtlas2 not available, falling back to spring layout...")
        if G.number_of_nodes() > 1000:
            pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)
        else:
            pos = nx.spring_layout(G, k=0.3, iterations=100, seed=42)
    
    # Create visualization
    print("Creating visualization...")
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw edges first (lighter and thinner)
    nx.draw_networkx_edges(G, pos, alpha=0.1, width=0.5, edge_color='gray', ax=ax)
    
    # Draw nodes
    node_size = 50 if G.number_of_nodes() > 500 else 100
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_size, 
                          alpha=0.8, ax=ax)
    
    # Add labels only for very small networks
    if G.number_of_nodes() < 100:
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    
    ax.set_title(f"Network Visualization - Communities Detected by Louvain Algorithm\n"
                f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
                f"{len(set(partition.values()))} communities", 
                fontsize=16, fontweight='bold', pad=20)
    
    # Create legend for colored communities
    legend_elements = []
    for i, comm_id in enumerate(largest_communities[:max_colored_communities]):
        color = colors_list[i % len(colors_list)]
        size = comm_sizes[comm_id]
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor=color, markersize=10,
                                         label=f'Community {comm_id} ({size} nodes)'))
    
    if len(comm_sizes) > max_colored_communities:
        other_count = sum(size for comm_id, size in comm_sizes.items() 
                         if comm_id not in largest_communities)
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                         markerfacecolor='lightgray', markersize=10,
                                         label=f'Other communities ({other_count} nodes)'))
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nVisualization saved to: {output_file}")
    
    plt.show()
    
    # Print structure observations
    print("\n" + "="*70)
    print("NETWORK STRUCTURE OBSERVATIONS")
    print("="*70)
    print("""
Based on the visualization, consider:

1. CLUSTERING: Are communities visibly clustered together or scattered?
   - Tight clusters suggest strong within-community connections
   - Scattered nodes suggest communities span different network regions

2. BRIDGES: Are there visible "bridge" nodes connecting communities?
   - Bridge nodes may be artists that cross genre/style boundaries
   - High betweenness centrality nodes often appear as bridges

3. CORE-PERIPHERY: Is there a dense core with sparse periphery?
   - Core nodes are likely influential/popular artists
   - Peripheral nodes may be niche or newer artists

4. MODULARITY: Do the colored communities form distinct visual groups?
   - Clear separation validates high modularity scores
   - Overlapping colors suggest interconnected communities
""")

def save_community_results(G, communities, partition, structural_modularity, output_file):
    """Save community detection results to JSON."""
    
    # Prepare results
    results = {
        "network_stats": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges()
        },
        "community_stats": {
            "num_communities": len(communities),
            "modularity": structural_modularity,
            "largest_community_size": max(len(c) for c in communities),
            "smallest_community_size": min(len(c) for c in communities),
            "average_community_size": np.mean([len(c) for c in communities])
        },
        "top_10_communities": []
    }
    
    # Add top 10 communities with their members
    sorted_communities = sorted(enumerate(communities), key=lambda x: len(x[1]), reverse=True)
    for comm_id, comm_nodes in sorted_communities[:10]:
        results["top_10_communities"].append({
            "community_id": comm_id,
            "size": len(comm_nodes),
            "members": list(comm_nodes)[:20]  # Include first 20 members to keep file manageable
        })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nCommunity results saved to: {output_file}")
    
    # Also save the full partition separately for confusion matrix analysis
    partition_file = output_file.replace('.json', '_partition.json')
    with open(partition_file, 'w', encoding='utf-8') as f:
        json.dump(partition, f, indent=2)
    
    print(f"Full partition saved to: {partition_file}")

def extract_backbone_edges(G, method='edge_betweenness', alpha=0.4):
    """
    Extract backbone edges from the network.
    
    Args:
        G: NetworkX graph
        method: 'edge_betweenness' or 'inverse_betweenness'
        alpha: Threshold for edge selection (keep top alpha % of edges)
    
    Returns:
        G_backbone: Graph containing only backbone edges
    """
    print(f"\nExtracting backbone using {method} method with alpha={alpha}...")
    
    # Calculate edge betweenness centrality
    print("Calculating edge betweenness centrality (this may take a while)...")
    edge_betweenness = nx.edge_betweenness_centrality(G, weight=None)
    
    # Create weighted graph based on method
    G_weighted = G.copy()
    
    if method == 'edge_betweenness':
        # Higher betweenness = higher weight (keep important connecting edges)
        for edge, betweenness in edge_betweenness.items():
            G_weighted[edge[0]][edge[1]]['weight'] = betweenness
    elif method == 'inverse_betweenness':
        # Lower betweenness = higher weight (keep local structure)
        max_betweenness = max(edge_betweenness.values())
        for edge, betweenness in edge_betweenness.items():
            # Avoid division by zero
            G_weighted[edge[0]][edge[1]]['weight'] = max_betweenness - betweenness + 0.001
    
    # Sort edges by weight and keep top alpha %
    sorted_edges = sorted(G_weighted.edges(data=True), 
                         key=lambda x: x[2]['weight'], 
                         reverse=True)
    
    n_edges_to_keep = int(len(sorted_edges) * alpha)
    backbone_edges = sorted_edges[:n_edges_to_keep]
    
    # Create backbone graph
    G_backbone = nx.Graph()
    G_backbone.add_nodes_from(G.nodes())
    for u, v, data in backbone_edges:
        G_backbone.add_edge(u, v, **data)
    
    # Remove isolated nodes
    isolated = list(nx.isolates(G_backbone))
    G_backbone.remove_nodes_from(isolated)
    
    print(f"Backbone extracted: {G_backbone.number_of_nodes()} nodes, "
          f"{G_backbone.number_of_edges()} edges")
    print(f"Removed {len(isolated)} isolated nodes")
    
    return G_backbone

def visualize_backbone_communities(G_backbone, partition, output_file, 
                                   max_colored_communities=10):
    """
    Visualize the backbone network with community colors.
    
    Args:
        G_backbone: Backbone graph
        partition: Dictionary mapping nodes to community IDs
        output_file: Path to save visualization
        max_colored_communities: Number of largest communities to color
    """
    print("\n" + "="*70)
    print("VISUALIZING BACKBONE WITH COMMUNITIES")
    print("="*70)
    
    # Filter partition to only include nodes in backbone
    partition_filtered = {node: comm_id for node, comm_id in partition.items() 
                         if node in G_backbone.nodes()}
    
    # Count community sizes in backbone
    comm_sizes = Counter(partition_filtered.values())
    largest_communities = [comm_id for comm_id, _ in comm_sizes.most_common(max_colored_communities)]
    
    print(f"\nColoring the {max_colored_communities} largest communities in backbone")
    print(f"Backbone has {G_backbone.number_of_nodes()} nodes and {G_backbone.number_of_edges()} edges\n")
    
    # Create color map
    colors_list = list(mcolors.TABLEAU_COLORS.keys())
    if max_colored_communities > len(colors_list):
        colors_list.extend(list(mcolors.CSS4_COLORS.keys()))
    
    node_colors = []
    for node in G_backbone.nodes():
        if node not in partition_filtered:
            node_colors.append('lightgray')
        else:
            comm_id = partition_filtered[node]
            if comm_id in largest_communities:
                color_idx = largest_communities.index(comm_id)
                node_colors.append(colors_list[color_idx % len(colors_list)])
            else:
                node_colors.append('lightgray')
    
    # Calculate layout
    print("Calculating ForceAtlas2 layout for backbone...")
    try:
        if G_backbone.number_of_nodes() > 1000:
            pos = nx.forceatlas2_layout(G_backbone, max_iter=500, seed=42)
        else:
            pos = nx.forceatlas2_layout(G_backbone, max_iter=500, seed=42)
    except AttributeError:
        print("ForceAtlas2 not available, using spring layout...")
        if G_backbone.number_of_nodes() > 1000:
            pos = nx.spring_layout(G_backbone, k=0.15, iterations=50, seed=42)
        else:
            pos = nx.spring_layout(G_backbone, k=0.3, iterations=100, seed=42)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(20, 20))
    
    # Draw edges
    nx.draw_networkx_edges(G_backbone, pos, alpha=0.15, width=1, edge_color='gray', ax=ax)
    
    # Draw nodes
    node_size = 80 if G_backbone.number_of_nodes() > 500 else 120
    nx.draw_networkx_nodes(G_backbone, pos, node_color=node_colors, 
                          node_size=node_size, alpha=0.9, ax=ax)
    
    ax.set_title(f"Backbone Network Visualization - Communities by Louvain\n"
                f"{G_backbone.number_of_nodes()} nodes, {G_backbone.number_of_edges()} edges",
                fontsize=16, fontweight='bold', pad=20)
    
    # Create legend
    legend_elements = []
    for i, comm_id in enumerate(largest_communities[:max_colored_communities]):
        color = colors_list[i % len(colors_list)]
        size = comm_sizes.get(comm_id, 0)
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                         markerfacecolor=color, markersize=10,
                                         label=f'Community {comm_id} ({size} nodes)'))
    
    if len(comm_sizes) > max_colored_communities:
        other_count = sum(size for comm_id, size in comm_sizes.items()
                         if comm_id not in largest_communities)
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                         markerfacecolor='lightgray', markersize=10,
                                         label=f'Other communities ({other_count} nodes)'))
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nBackbone visualization saved to: {output_file}")
    plt.show()
    
    print("\n" + "="*70)
    print("BACKBONE OBSERVATIONS")
    print("="*70)
    print("""
The backbone network reveals the most important structural connections:

1. COMMUNITY CORRESPONDENCE: Do the communities align with backbone structure?
   - If yes: Communities are defined by their connections to each other
   - If no: Communities may be based on dense internal connections

2. HUB IDENTIFICATION: Which nodes remain central in the backbone?
   - These are the true "bridge" artists across the network
   - They connect different parts of the musical landscape

3. STRUCTURAL CLARITY: Is the network structure clearer now?
   - Backbone removes noise and highlights key pathways
   - May reveal hierarchical or modular organization
""")

def main():
    print("="*70)
    print("STRUCTURAL COMMUNITY DETECTION (Louvain Algorithm)")
    print("="*70)
    
    # Load data
    graph_file = "data/rock/performers_graph.json"
    genres_file = "Week_7/artist_genres.json"
    
    G = load_network(graph_file)
    artist_genres = load_artist_genres(genres_file)
    
    # Filter to nodes with genres
    G_filtered = filter_network_by_genres(G, artist_genres)
    
    # Make sure network is connected or work with largest component
    if not nx.is_connected(G_filtered):
        print("\nNetwork is not fully connected. Working with largest connected component...")
        largest_cc = max(nx.connected_components(G_filtered), key=len)
        G_filtered = G_filtered.subgraph(largest_cc).copy()
        print(f"Largest component: {G_filtered.number_of_nodes()} nodes, {G_filtered.number_of_edges()} edges")
    
    # Detect communities using Louvain
    communities, partition = detect_communities_louvain(G_filtered, seed=42)
    
    # Calculate modularity
    print("\nCalculating modularity of detected communities...")
    structural_modularity = calculate_modularity_networkx(G_filtered, partition)
    
    print("\n" + "="*70)
    print(f"MODULARITY OF STRUCTURAL COMMUNITIES: {structural_modularity:.4f}")
    print("="*70)
    
    if structural_modularity > 0.4:
        print("✓ EXCELLENT - Very strong community structure detected")
    elif structural_modularity > 0.3:
        print("✓ GOOD - Strong community structure detected")
    elif structural_modularity > 0.2:
        print("✓ MODERATE - Moderate community structure detected")
    else:
        print("⚠ WEAK - Weak community structure detected")
    
    # Compare with genre-based modularity
    compare_with_genre_modularity(structural_modularity)
    
    # Visualize network
    visualize_communities(G_filtered, partition, 
                         output_file="Week_7/community_network_visualization.png",
                         max_colored_communities=10)
    
    # Save results
    save_community_results(G_filtered, communities, partition, structural_modularity,
                          "Week_7/community_detection_results.json")
    
    # Extract and visualize backbone
    print("\n" + "="*70)
    print("BACKBONE EXTRACTION AND VISUALIZATION")
    print("="*70)
    print("\nExtracting backbone edges to reveal core network structure...")
    print("This removes less important edges and highlights key connections.")
    
    # Try edge betweenness method
    try:
        G_backbone = extract_backbone_edges(G_filtered, 
                                           method='edge_betweenness', 
                                           alpha=0.1)
        
        # Visualize backbone with communities
        visualize_backbone_communities(G_backbone, partition,
                                      output_file="Week_7/backbone_communities_visualization.png",
                                      max_colored_communities=10)
    except Exception as e:
        print(f"\nBackbone extraction skipped: {e}")
        print("For large networks, edge betweenness calculation can be slow.")
        print("You can run this separately if needed.")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nFiles created:")
    print("1. community_detection_results.json - Community statistics")
    print("2. community_detection_results_partition.json - Full partition data")
    print("3. community_network_visualization.png - Full network with communities")
    print("4. backbone_communities_visualization.png - Backbone network (if generated)")
    print("\nNext steps:")
    print("1. Examine the visualizations to understand network structure")
    print("2. Compare structural communities with genre communities")
    print("3. Run confusion matrix analysis (next exercise)")
    print("4. Compare full network vs backbone structure")

if __name__ == "__main__":
    main()

