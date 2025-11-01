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
        artist_texts = {}
        
        if not os.path.exists(performers_dir):
            print(f"ERROR: Performers directory not found at {performers_dir}")
            return {}
        
        files = [f for f in os.listdir(performers_dir) if f.endswith('.json')]
        
        for filename in files:
            filepath = os.path.join(performers_dir, filename)
            artist_name = filename.replace('.json', '').replace('%20', ' ').replace('%2F', '/')
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
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
        plt.show()