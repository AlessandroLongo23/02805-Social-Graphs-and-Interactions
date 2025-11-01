import json
import os
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import math
import sys
from bs4 import BeautifulSoup

# Import TFIDFAnalyzer from the main tfidf script
sys.path.append(os.path.dirname(__file__))
from TFIDFAnalyzer import TFIDFAnalyzer

class TFIDFCommunities():
    def __init__(self):
        pass

    def extract_text_from_html(self, html_content):
        """Extract clean text from HTML content."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "meta", "link"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Break into lines and remove leading/trailing space
        lines = (line.strip() for line in text.splitlines())
        
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        
        # Drop blank lines
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text


    def extract_texts_from_performers(self, performers_dir='data/rock/performers'):
        """
        Extract cleaned text from all performer JSON files.
        
        Returns:
            Dictionary mapping artist names to their cleaned Wikipedia text
        """
        print("Extracting texts from performer JSON files...")
        
        artist_texts = {}
        
        if not os.path.exists(performers_dir):
            print(f"ERROR: Performers directory not found at {performers_dir}")
            return {}
        
        files = [f for f in os.listdir(performers_dir) if f.endswith('.json')]
        
        for i, filename in enumerate(files, 1):
            if i % 100 == 0:
                print(f"  Processed {i}/{len(files)} files...")
            
            filepath = os.path.join(performers_dir, filename)
            # Decode URL-encoded filename to get artist name
            artist_name = filename.replace('.json', '').replace('%20', ' ').replace('%2F', '/')
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract text from nested structure: data['parse']['text']['*']
                if 'parse' in data and 'text' in data['parse'] and '*' in data['parse']['text']:
                    html_content = data['parse']['text']['*']
                    text = self.extract_text_from_html(html_content)
                    
                    if text:
                        artist_texts[artist_name] = text
                    
            except Exception as e:
                print(f"  Warning: Could not process {filename}: {e}")
                continue
        
        print(f"Extracted texts for {len(artist_texts)} artists\n")
        return artist_texts


    def create_community_documents(self, artist_texts, partition, top_n=15):
        """
        Create aggregated documents for the top N communities.
        
        For each community, aggregate all text from all its members.
        
        Args:
            artist_texts: Dictionary mapping artists to their texts
            partition: Dictionary mapping artists to community IDs
            top_n: Number of top communities to process
        
        Returns:
            Dictionary mapping community IDs to aggregated text
        """
        print(f"Creating documents for top {top_n} communities...")
        
        # Count community sizes
        community_sizes = {}
        for artist, comm_id in partition.items():
            community_sizes[comm_id] = community_sizes.get(comm_id, 0) + 1
        
        # Get top N communities by size
        top_communities = sorted(community_sizes.items(), 
                                key=lambda x: x[1], 
                                reverse=True)[:top_n]
        
        print("\nTop Communities by Size:")
        for comm_id, size in top_communities:
            print(f"  Community {comm_id}: {size} artists")
        
        # Create aggregated documents
        community_docs = {}
        
        for comm_id, size in top_communities:
            # Get all artists in this community
            members = [artist for artist, cid in partition.items() if cid == comm_id]
            
            # Aggregate their texts
            all_text = []
            for artist in members:
                if artist in artist_texts:
                    all_text.append(artist_texts[artist])
            
            community_docs[f"Community_{comm_id}"] = " ".join(all_text)
            
            print(f"  Community {comm_id}: Aggregated {len(all_text)}/{size} texts "
                f"({len(' '.join(all_text).split())} words)")
        
        print(f"\nCreated {len(community_docs)} community documents\n")
        return community_docs


    def clean_and_tokenize(self, text):
        """
        Clean and tokenize text (similar to term_frequency_analysis.py).
        
        Args:
            text: Raw text string
        
        Returns:
            List of cleaned tokens
        """
        import re
        import string
        
        # Lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove digits
        text = re.sub(r'\d+', '', text)
        
        # Tokenize
        tokens = text.split()
        
        # Remove stopwords (common English words)
        stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to',
            'was', 'will', 'with', 'his', 'her', 'or', 'had', 'have', 'but', 'not',
            'they', 'this', 'which', 'were', 'when', 'who', 'would', 'there', 'their',
            'what', 'so', 'if', 'out', 'than', 'then', 'them', 'these', 'more', 'up',
            'about', 'can', 'could', 'did', 'do', 'been', 'also', 'one', 'two',
            'first', 'second', 'other', 'into', 'through', 'during', 'before', 'after',
            'ref', 'cite', 'web', 'url', 'http', 'https', 'www', 'com', 'org', 'net',
            'retrieved', 'archived', 'original', 'external', 'links'
        }
        
        tokens = [t for t in tokens if t not in stopwords]
        
        # Remove very short words
        tokens = [t for t in tokens if len(t) > 2]
        
        return tokens


    def compute_community_tf(self, community_docs, min_count=3):
        """
        Compute term frequencies for community documents.
        
        Args:
            community_docs: Dictionary mapping community names to text
            min_count: Minimum word count to include
        
        Returns:
            Dictionary mapping community names to TF dictionaries
        """
        print("Computing term frequencies for communities...")
        
        community_tf = {}
        
        for comm_name, text in community_docs.items():
            print(f"  Processing {comm_name}...")
            
            # Tokenize and clean
            tokens = self.clean_and_tokenize(text)
            
            # Count frequencies
            word_counts = {}
            for token in tokens:
                word_counts[token] = word_counts.get(token, 0) + 1
            
            # Filter by minimum count
            filtered_counts = {w: c for w, c in word_counts.items() if c >= min_count}
            
            community_tf[comm_name] = filtered_counts
            
            print(f"    {len(tokens)} tokens → {len(filtered_counts)} unique words "
                f"(min_count={min_count})")
        
        print(f"\nTerm frequencies computed for {len(community_tf)} communities\n")
        return community_tf


    def create_combined_community_grid(self, community_tfidf, output_file, n_cols=3):
        """
        Create grid of community TF-IDF word clouds.
        
        Args:
            community_tfidf: Dictionary of TF-IDF scores
            output_file: Output file path
            n_cols: Number of columns
        """
        print("\n" + "="*70)
        print("CREATING COMMUNITY TF-IDF WORD CLOUD GRID")
        print("="*70)
        
        communities = list(community_tfidf.keys())
        n_comms = len(communities)
        n_rows = (n_comms + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4))
        
        if n_comms == 1:
            axes = np.array([axes])
        else:
            axes = axes.flatten()
        
        colormaps = ['plasma', 'inferno', 'magma', 'cividis', 'viridis',
                    'twilight_shifted', 'RdPu', 'YlOrRd', 'PuBu', 'GnBu',
                    'BuPu', 'OrRd', 'YlGnBu', 'YlOrBr', 'RdYlGn']
        
        for i, comm_name in enumerate(communities):
            print(f"Adding {comm_name} to grid ({i+1}/{n_comms})")
            
            tfidf_dict = community_tfidf[comm_name]
            
            if not tfidf_dict:
                axes[i].text(0.5, 0.5, f'{comm_name}\n(No data)', 
                            ha='center', va='center', fontsize=14)
                axes[i].axis('off')
                continue
            
            # Normalize scores
            max_score = max(tfidf_dict.values())
            min_score = min(tfidf_dict.values())
            
            normalized_scores = {}
            for word, score in tfidf_dict.items():
                if max_score > min_score:
                    normalized = 1 + 99 * (score - min_score) / (max_score - min_score)
                else:
                    normalized = 50
                normalized_scores[word] = normalized
            
            # Create word cloud
            colormap = colormaps[i % len(colormaps)]
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                colormap=colormap,
                max_words=100,
                relative_scaling=0.5,
                collocations=False
            ).generate_from_frequencies(normalized_scores)
            
            # Plot
            axes[i].imshow(wordcloud, interpolation='bilinear')
            axes[i].set_title(comm_name.replace('_', ' '), fontsize=14, fontweight='bold')
            axes[i].axis('off')
        
        # Hide empty subplots
        for j in range(n_comms, len(axes)):
            axes[j].axis('off')
        
        plt.suptitle('TF-IDF Word Clouds by Structural Community\n(Size = Characteristic Importance)', 
                    fontsize=20, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"\nCombined community TF-IDF grid saved to {output_file}")
        plt.show()


# def main():
#     analyzer = TFIDFAnalyzer()
#     communities_generator = TFIDFCommunities()
    
#     print("="*70)
#     print("TF-IDF ANALYSIS FOR STRUCTURAL COMMUNITIES")
#     print("="*70)
#     print("""
# This script applies TF-IDF analysis to structural communities instead of genres.

# Process:
# 1. Load community partition from Louvain algorithm
# 2. Extract text for all artists
# 3. Aggregate text for each community (all members' texts combined)
# 4. Calculate TF-IDF for communities
# 5. Create word clouds to characterize communities by their vocabulary

# This helps answer: What are these communities about?
# """)
    
#     # Paths
#     partition_file = "community_detection_results_partition.json"
#     performers_dir = "data/rock/performers"
#     output_dir = "tfidf_communities"
    
#     # Check if partition file exists
#     if not os.path.exists(partition_file):
#         print(f"ERROR: Community partition file not found at {partition_file}")
#         print("Please run community_detection.py first!")
#         return
    
#     # Load partition
#     print(f"Loading community partition from {partition_file}...")
#     with open(partition_file, 'r', encoding='utf-8') as f:
#         partition = json.load(f)
#     print(f"Loaded partition with {len(partition)} artists\n")
    
#     # Extract texts
#     artist_texts = communities_generator.extract_texts_from_performers(performers_dir)
    
#     if not artist_texts:
#         print("ERROR: No texts extracted!")
#         return
    
#     # Create community documents
#     community_docs = communities_generator.create_community_documents(artist_texts, partition, top_n=15)
    
#     # Compute TF
#     community_tf = communities_generator.compute_community_tf(community_docs, min_count=5)
    
#     # Calculate IDF and TF-IDF
#     analyzer = TFIDFAnalyzer(community_tf)
#     idf_scores = analyzer.calculate_idf(log_base='e')
#     community_tfidf = analyzer.calculate_tfidf(idf_scores, tf_variant='log')
    
#     # Display comparison
#     analyzer.display_comparison(community_tfidf, n=10)
    
#     # Create output directory
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Save TF-IDF scores
#     tfidf_file = os.path.join(output_dir, 'community_tfidf_scores.json')
#     with open(tfidf_file, 'w', encoding='utf-8') as f:
#         json.dump(community_tfidf, f, indent=2, ensure_ascii=False)
#     print(f"\nCommunity TF-IDF scores saved to {tfidf_file}")
    
#     # Save IDF scores
#     idf_file = os.path.join(output_dir, 'community_idf_scores.json')
#     with open(idf_file, 'w', encoding='utf-8') as f:
#         json.dump(idf_scores, f, indent=2, ensure_ascii=False)
#     print(f"Community IDF scores saved to {idf_file}")
    
#     # Save top words
#     top_words = {}
#     for comm_name, tfidf_dict in community_tfidf.items():
#         sorted_words = sorted(tfidf_dict.items(), 
#                              key=lambda x: x[1], 
#                              reverse=True)[:20]
#         top_words[comm_name] = [{"word": w, "score": float(s)} for w, s in sorted_words]
    
#     top_file = os.path.join(output_dir, 'top_community_tfidf_words.json')
#     with open(top_file, 'w', encoding='utf-8') as f:
#         json.dump(top_words, f, indent=2, ensure_ascii=False)
#     print(f"Top community TF-IDF words saved to {top_file}")
    
#     # Create word clouds
#     wordcloud_dir = os.path.join(output_dir, 'wordclouds')
#     analyzer.create_tfidf_wordclouds(community_tfidf, wordcloud_dir)
    
#     # Create combined grid
#     combined_file = os.path.join(wordcloud_dir, 'all_communities_tfidf_combined.png')
#     analyzer.create_combined_community_grid(community_tfidf, combined_file, n_cols=3)
    
#     # Final summary
#     print("\n" + "="*70)
#     print("COMMUNITY TF-IDF ANALYSIS COMPLETE!")
#     print("="*70)
#     print(f"\nGenerated files in {output_dir}/:")
#     print("  - community_tfidf_scores.json - TF-IDF scores for all communities")
#     print("  - community_idf_scores.json - IDF scores across communities")
#     print("  - top_community_tfidf_words.json - Top 20 words per community")
#     print("  - wordclouds/ - Individual and combined word clouds")
    
#     print("\n" + "="*70)
#     print("INTERPRETATION QUESTIONS")
#     print("="*70)
#     print("""
# Compare community word clouds with genre word clouds:

# QUESTIONS TO CONSIDER:
# 1. Are community word clouds more or less meaningful than genre word clouds?
# 2. Can you identify what makes each community distinct from its vocabulary?
# 3. Look at the confusion matrix from Part 2:
#    - Do communities with high genre purity show genre-specific vocabulary?
#    - Do mixed communities show vocabulary from multiple genres?
# 4. Which approach better characterizes the network structure:
#    - Genre labels (external metadata)?
#    - Community detection + TF-IDF (data-driven)?

# This demonstrates how TF-IDF can characterize structural communities
# when you DON'T have genre labels!
# """)


# if __name__ == "__main__":
#     main()

