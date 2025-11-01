import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from lib.Loader import Loader

class ConfusionMatrixAnalyzer():
    def __init__(self):
        pass

    def load_community_results(self, community_file):
        """Load community detection results."""
        print("Loading community detection results...")
        with open(community_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # Reconstruct partition dictionary from top communities
        # Note: This only includes top 10, so we need to load full results
        print(f"Community data loaded: {results['community_stats']['num_communities']} communities")
        return results

    def load_partition(self, partition_file):
        """Load the full partition from saved file."""
        print("\nLoading partition data...")
        with open(partition_file, 'r', encoding='utf-8') as f:
            partition = json.load(f)
        
        print(f"Partition loaded for {len(partition)} artists")
        return partition

    def create_confusion_matrix(self, artist_genres, partition, top_n_genres=7, top_n_communities=7):
        """
        Create a confusion matrix D(G x C) comparing genres with communities.
        
        Args:
            artist_genres: Dictionary mapping artists to list of genres
            partition: Dictionary mapping artists to community IDs
            top_n_genres: Number of top genres to include
            top_n_communities: Number of top communities to include
        
        Returns:
            confusion_matrix: numpy array of shape (G, C)
            genre_labels: List of genre names
            community_labels: List of community IDs
        """
        print("\n" + "="*70)
        print("CREATING CONFUSION MATRIX")
        print("="*70)
        
        # Find top genres (count all genre occurrences)
        all_genres = []
        for genres in artist_genres.values():
            all_genres.extend(genres)
        
        genre_counts = Counter(all_genres)
        top_genres = [genre for genre, _ in genre_counts.most_common(top_n_genres)]
        
        print(f"\nTop {top_n_genres} genres:")
        for i, genre in enumerate(top_genres, 1):
            print(f"  {i}. {genre:<30} ({genre_counts[genre]} occurrences)")
        
        # Find top communities
        community_counts = Counter(partition.values())
        top_communities = [comm_id for comm_id, _ in community_counts.most_common(top_n_communities)]
        
        print(f"\nTop {top_n_communities} communities:")
        for i, comm_id in enumerate(top_communities, 1):
            print(f"  {i}. Community {comm_id:<5} ({community_counts[comm_id]} nodes)")
        
        # Create confusion matrix
        confusion_matrix = np.zeros((top_n_genres, top_n_communities), dtype=int)
        
        # Count overlaps
        artists_counted = set()
        
        for artist, genres in artist_genres.items():
            # Check if artist is in partition
            if artist not in partition:
                continue
            
            comm_id = partition[artist]
            
            # Check if community is in top communities
            if comm_id not in top_communities:
                continue
            
            comm_idx = top_communities.index(comm_id)
            
            # Check if artist has any of the top genres
            artist_top_genres = [g for g in genres if g in top_genres]
            
            if not artist_top_genres:
                continue
            
            # For each genre this artist has (from top genres), increment the count
            for genre in artist_top_genres:
                genre_idx = top_genres.index(genre)
                confusion_matrix[genre_idx, comm_idx] += 1
            
            artists_counted.add(artist)
        
        print(f"\nTotal artists included in confusion matrix: {len(artists_counted)}")
        
        return confusion_matrix, top_genres, top_communities

    def visualize_confusion_matrix(self, confusion_matrix, genre_labels, community_labels, 
                                output_file="confusion_matrix.png"):
        """Visualize the confusion matrix as a heatmap."""
        
        print("\n" + "="*70)
        print("VISUALIZING CONFUSION MATRIX")
        print("="*70)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='YlOrRd', 
                    xticklabels=[f"Comm {c}" for c in community_labels],
                    yticklabels=genre_labels, 
                    cbar_kws={'label': 'Number of Artists'},
                    linewidths=0.5, linecolor='gray',
                    ax=ax)
        
        ax.set_xlabel('Communities', fontsize=12, fontweight='bold')
        ax.set_ylabel('Genres', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix: Genres vs Communities\n'
                    'Cell (i,j) = Number of artists with genre i in community j', 
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nConfusion matrix saved to: {output_file}")
        plt.show()

    def analyze_confusion_matrix(self, confusion_matrix, genre_labels, community_labels):
        """Analyze and interpret the confusion matrix."""
        
        print("\n" + "="*70)
        print("CONFUSION MATRIX ANALYSIS")
        print("="*70)
        
        # Print the matrix
        print("\nConfusion Matrix D(genre, community):")
        print("\n" + " "*25, end="")
        for comm in community_labels:
            print(f"Comm {comm:3d}", end="  ")
        print()
        print("-" * (25 + len(community_labels) * 10))
        
        for i, genre in enumerate(genre_labels):
            print(f"{genre:<25}", end="")
            for j in range(len(community_labels)):
                print(f"{confusion_matrix[i, j]:7d}  ", end="")
            print()
        
        # Calculate row sums (total artists per genre across all communities)
        row_sums = confusion_matrix.sum(axis=1)
        col_sums = confusion_matrix.sum(axis=0)
        
        print("\n" + "="*70)
        print("ROW ANALYSIS (Genre Distribution Across Communities)")
        print("="*70)
        
        for i, genre in enumerate(genre_labels):
            print(f"\n{genre}:")
            if row_sums[i] == 0:
                print("  No artists in top communities")
                continue
            
            # Find dominant community for this genre
            dominant_comm_idx = np.argmax(confusion_matrix[i, :])
            dominant_comm = community_labels[dominant_comm_idx]
            dominant_count = confusion_matrix[i, dominant_comm_idx]
            dominant_pct = (dominant_count / row_sums[i]) * 100
            
            print(f"  Total artists: {row_sums[i]}")
            print(f"  Dominant community: Community {dominant_comm} ({dominant_count} artists, {dominant_pct:.1f}%)")
            
            # Show distribution
            print(f"  Distribution: ", end="")
            for j, comm in enumerate(community_labels):
                if confusion_matrix[i, j] > 0:
                    pct = (confusion_matrix[i, j] / row_sums[i]) * 100
                    print(f"C{comm}:{pct:.0f}% ", end="")
            print()
        
        print("\n" + "="*70)
        print("COLUMN ANALYSIS (Community Composition by Genre)")
        print("="*70)
        
        for j, comm in enumerate(community_labels):
            print(f"\nCommunity {comm}:")
            if col_sums[j] == 0:
                print("  No artists with top genres")
                continue
            
            # Find dominant genre for this community
            dominant_genre_idx = np.argmax(confusion_matrix[:, j])
            dominant_genre = genre_labels[dominant_genre_idx]
            dominant_count = confusion_matrix[dominant_genre_idx, j]
            dominant_pct = (dominant_count / col_sums[j]) * 100
            
            print(f"  Total artist-genre pairs: {col_sums[j]}")
            print(f"  Dominant genre: {dominant_genre} ({dominant_count} artists, {dominant_pct:.1f}%)")
            
            # Show top 3 genres
            top_genres_idx = np.argsort(confusion_matrix[:, j])[::-1][:3]
            print(f"  Top genres: ", end="")
            for idx in top_genres_idx:
                if confusion_matrix[idx, j] > 0:
                    pct = (confusion_matrix[idx, j] / col_sums[j]) * 100
                    print(f"{genre_labels[idx]}:{pct:.0f}% ", end="")
            print()
        
        # Calculate alignment metrics
        print("\n" + "="*70)
        print("ALIGNMENT METRICS")
        print("="*70)
        
        # 1. Purity: For each community, what % belongs to dominant genre?
        purities = []
        for j in range(len(community_labels)):
            if col_sums[j] > 0:
                purity = np.max(confusion_matrix[:, j]) / col_sums[j]
                purities.append(purity)
        
        avg_purity = np.mean(purities) if purities else 0
        print(f"\nAverage Community Purity: {avg_purity:.3f}")
        print("  (Higher = communities are more homogeneous in terms of genre)")
        
        # 2. Completeness: For each genre, what % is in dominant community?
        completeness_scores = []
        for i in range(len(genre_labels)):
            if row_sums[i] > 0:
                completeness = np.max(confusion_matrix[i, :]) / row_sums[i]
                completeness_scores.append(completeness)
        
        avg_completeness = np.mean(completeness_scores) if completeness_scores else 0
        print(f"\nAverage Genre Completeness: {avg_completeness:.3f}")
        print("  (Higher = genres are more concentrated in single communities)")
        
        # 3. Normalized mutual information approximation
        total = confusion_matrix.sum()
        if total > 0:
            # Simplified alignment score
            diagonal_strength = np.trace(confusion_matrix) / total if confusion_matrix.shape[0] == confusion_matrix.shape[1] else 0
            max_overlap = np.max(confusion_matrix, axis=1).sum() / total
            
            print(f"\nMax Overlap Score: {max_overlap:.3f}")
            print("  (Ratio of artists in dominant communities for their genres)")
        
        print("\n" + "="*70)
        print("INTERPRETATION")
        print("="*70)
        
        if avg_purity > 0.6 and avg_completeness > 0.6:
            print("\n✓ STRONG ALIGNMENT: Communities correspond well to genres")
            print("  - Communities are genre-homogeneous (high purity)")
            print("  - Genres are concentrated in specific communities (high completeness)")
            print("  - The network structure reflects genre boundaries")
        elif avg_purity > 0.4 or avg_completeness > 0.4:
            print("\n~ MODERATE ALIGNMENT: Communities partially correspond to genres")
            print("  - Some communities align with genres, but there's significant mixing")
            print("  - Network structure is influenced by genre but not determined by it")
            print("  - Other factors (collaborations, era, style) also shape communities")
        else:
            print("\n✗ WEAK ALIGNMENT: Communities do not correspond well to genres")
            print("  - Communities span multiple genres")
            print("  - Genres are scattered across communities")
            print("  - Network structure is driven by factors other than genre")
            print("  - Consider: collaborations, influences, era, geographic location, etc.")
        
        print("\nKey Insights:")
        print("  - Genres with high concentration: Artists mainly collaborate within genre")
        print("  - Genres with low concentration: Artists collaborate across genre boundaries")
        print("  - Communities with mixed genres: Cross-genre musical movements or eras")
        print("  - Communities with pure genres: Genre-specific sub-networks")

    def save_confusion_matrix_results(self, confusion_matrix, genre_labels, community_labels, 
                                    output_file="confusion_matrix_results.json"):
        """Save confusion matrix and analysis results."""
        
        results = {
            "confusion_matrix": confusion_matrix.tolist(),
            "genre_labels": genre_labels,
            "community_labels": [int(c) for c in community_labels],
            "row_sums": confusion_matrix.sum(axis=1).tolist(),
            "col_sums": confusion_matrix.sum(axis=0).tolist(),
            "metrics": {
                "average_purity": float(np.mean([np.max(confusion_matrix[:, j]) / confusion_matrix[:, j].sum() 
                                            for j in range(confusion_matrix.shape[1]) 
                                            if confusion_matrix[:, j].sum() > 0])),
                "average_completeness": float(np.mean([np.max(confusion_matrix[i, :]) / confusion_matrix[i, :].sum() 
                                                    for i in range(confusion_matrix.shape[0]) 
                                                    if confusion_matrix[i, :].sum() > 0]))
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nConfusion matrix results saved to: {output_file}")

# def main():
#     loader = Loader()
#     confusion_matrix_analyzer = ConfusionMatrixAnalyzer()

#     print("="*70)
#     print("CONFUSION MATRIX: GENRES VS COMMUNITIES")
#     print("="*70)
    
#     # Load data
#     genres_file = "./Week_7/artist_genres.json"
#     community_file = "./Week_7/community_detection_results.json"
#     partition_file = "./Week_7/community_detection_results_partition.json"
    
#     artist_genres = loader.load_artist_genres(genres_file)
    
#     # Load partition
#     print("\n" + "="*70)
#     print("NOTE: This script needs the full partition data from community_detection.py")
#     print("Make sure you've run community_detection.py first.")
#     print("="*70)
    
#     try:
#         partition = confusion_matrix_analyzer.load_partition(partition_file)
#     except FileNotFoundError:
#         print(f"\nError: Partition file not found at {partition_file}")
#         print("Please run community_detection.py first to generate the partition.")
#         return
#     except Exception as e:
#         print(f"\nError loading partition: {e}")
#         return
    
#     # Create confusion matrix
#     confusion_matrix, genre_labels, community_labels = confusion_matrix_analyzer.create_confusion_matrix(
#         artist_genres, partition, top_n_genres=7, top_n_communities=7
#     )
    
#     # Visualize
#     confusion_matrix_analyzer.visualize_confusion_matrix(confusion_matrix, genre_labels, community_labels)
    
#     # Analyze
#     confusion_matrix_analyzer.analyze_confusion_matrix(confusion_matrix, genre_labels, community_labels)
    
#     # Save results
#     confusion_matrix_analyzer.save_confusion_matrix_results(confusion_matrix, genre_labels, community_labels)
    
#     print("\n" + "="*70)
#     print("ANALYSIS COMPLETE!")
#     print("="*70)

# if __name__ == "__main__":
#     main()

