import json
import os
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import math

class TFIDFAnalyzer:
    """
    TF-IDF (Term Frequency - Inverse Document Frequency) Analyzer
    
    TF-IDF is a numerical statistic that reflects how important a word is 
    to a document in a collection of documents (corpus).
    
    Components:
    - TF (Term Frequency): How often a word appears in a document
    - IDF (Inverse Document Frequency): How rare/common a word is across all documents
    - TF-IDF = TF × IDF: Words that are frequent in one doc but rare overall get high scores
    """
    
    def __init__(self, genre_tf_dict):
        """
        Initialize with term frequency data.
        
        Args:
            genre_tf_dict: Dictionary mapping genres to TF dictionaries
        """
        self.genre_tf = genre_tf_dict
        self.genres = list(genre_tf_dict.keys())
        self.n_documents = len(self.genres)
        
        # Build vocabulary (all unique words across all genres)
        self.vocabulary = set()
        for tf_dict in genre_tf_dict.values():
            self.vocabulary.update(tf_dict.keys())
        
        print(f"Initialized TF-IDF Analyzer:")
        print(f"  - {self.n_documents} documents (genres)")
        print(f"  - {len(self.vocabulary)} unique words in vocabulary\n")
    
    def calculate_idf(self, log_base='e'):
        """
        Calculate IDF (Inverse Document Frequency) for each word.
        
        IDF measures how rare or common a word is across all documents.
        
        Formula: IDF(word) = log(N / df(word))
        Where:
        - N = total number of documents
        - df(word) = number of documents containing the word
        
        Args:
            log_base: Logarithm base ('e' for natural log, 10, 2, etc.)
        
        Returns:
            Dictionary mapping words to IDF scores
        """
        print(f"Calculating IDF scores (log base: {log_base})...")
        
        # Count document frequency for each word
        document_frequency = {}
        for word in self.vocabulary:
            df = sum(1 for tf_dict in self.genre_tf.values() if word in tf_dict)
            document_frequency[word] = df
        
        # Calculate IDF
        idf_scores = {}
        
        for word, df in document_frequency.items():
            if log_base == 'e':
                idf = math.log(self.n_documents / df)
            elif log_base == 10:
                idf = math.log10(self.n_documents / df)
            elif log_base == 2:
                idf = math.log2(self.n_documents / df)
            else:
                idf = math.log(self.n_documents / df) / math.log(log_base)
            
            idf_scores[word] = idf
        
        print(f"IDF calculation complete!")
        print(f"  - Min IDF: {min(idf_scores.values()):.4f} (common words)")
        print(f"  - Max IDF: {max(idf_scores.values()):.4f} (rare words)\n")
        
        return idf_scores
    
    def calculate_tfidf(self, idf_scores, tf_variant='raw'):
        """
        Calculate TF-IDF scores for each word in each genre.
        
        TF-IDF = TF × IDF
        
        Args:
            idf_scores: Dictionary of IDF scores
            tf_variant: Type of TF to use:
                - 'raw': raw count
                - 'log': 1 + log(count)
                - 'normalized': count / max_count_in_doc
        
        Returns:
            Dictionary mapping genres to TF-IDF dictionaries
        """
        print(f"Calculating TF-IDF scores (TF variant: {tf_variant})...")
        
        genre_tfidf = {}
        
        for genre, tf_dict in self.genre_tf.items():
            tfidf_dict = {}
            
            # Get max count for normalization
            max_count = max(tf_dict.values()) if tf_dict else 1
            
            for word, count in tf_dict.items():
                # Calculate TF based on variant
                if tf_variant == 'raw':
                    tf = count
                elif tf_variant == 'log':
                    tf = 1 + math.log(count)
                elif tf_variant == 'normalized':
                    tf = count / max_count
                else:
                    tf = count  # default to raw
                
                # Calculate TF-IDF
                idf = idf_scores.get(word, 0)
                tfidf = tf * idf
                
                tfidf_dict[word] = tfidf
            
            genre_tfidf[genre] = tfidf_dict
        
        print(f"TF-IDF calculation complete!\n")
        
        return genre_tfidf
    
    def get_top_words(self, genre_tfidf, n=10):
        """
        Get top N words by TF-IDF score for each genre.
        
        Args:
            genre_tfidf: Dictionary mapping genres to TF-IDF dictionaries
            n: Number of top words to return
        
        Returns:
            Dictionary mapping genres to list of (word, score) tuples
        """
        top_words = {}
        
        for genre, tfidf_dict in genre_tfidf.items():
            sorted_words = sorted(tfidf_dict.items(), 
                                 key=lambda x: x[1], 
                                 reverse=True)[:n]
            top_words[genre] = sorted_words
        
        return top_words
    
    def display_comparison(self, genre_tfidf, n=10):
        """
        Display comparison between TF and TF-IDF top words.
        
        Args:
            genre_tfidf: Dictionary of TF-IDF scores
            n: Number of top words to show
        """
        print("="*70)
        print("TF vs TF-IDF COMPARISON")
        print("="*70)
        
        for genre in self.genres:
            print(f"\n{genre.upper()}")
            print("-" * 70)
            
            # Top words by TF (raw frequency)
            tf_top = sorted(self.genre_tf[genre].items(), 
                           key=lambda x: x[1], 
                           reverse=True)[:n]
            
            # Top words by TF-IDF
            tfidf_top = sorted(genre_tfidf[genre].items(), 
                              key=lambda x: x[1], 
                              reverse=True)[:n]
            
            print(f"{'TOP BY TF (Frequency)':<40} {'TOP BY TF-IDF (Importance)'}")
            print(f"{'Word':<20} {'Count':>15}  {'Word':<20} {'Score':>15}")
            print("-" * 70)
            
            for i in range(n):
                tf_word, tf_count = tf_top[i] if i < len(tf_top) else ("", 0)
                tfidf_word, tfidf_score = tfidf_top[i] if i < len(tfidf_top) else ("", 0)
                
                print(f"{tf_word:<20} {tf_count:>15}  {tfidf_word:<20} {tfidf_score:>15.4f}")


def load_term_frequencies(tf_file):
    """Load term frequency data from JSON."""
    print(f"Loading term frequencies from {tf_file}...")
    with open(tf_file, 'r', encoding='utf-8') as f:
        genre_tf = json.load(f)
    print(f"Loaded TF data for {len(genre_tf)} genres\n")
    return genre_tf


def create_tfidf_wordcloud(tfidf_dict, title, output_file, 
                           colormap='viridis', max_words=100):
    """
    Create word cloud from TF-IDF scores.
    
    Args:
        tfidf_dict: Dictionary of TF-IDF scores
        title: Title for the word cloud
        output_file: Path to save image
        colormap: Matplotlib colormap
        max_words: Maximum number of words
    """
    if not tfidf_dict:
        print(f"  No data for {title}")
        return
    
    # Rescale TF-IDF scores for word cloud
    # WordCloud expects frequencies, so we normalize to reasonable range
    max_score = max(tfidf_dict.values())
    min_score = min(tfidf_dict.values())
    
    # Normalize to range [1, 100] for better visualization
    normalized_scores = {}
    for word, score in tfidf_dict.items():
        if max_score > min_score:
            normalized = 1 + 99 * (score - min_score) / (max_score - min_score)
        else:
            normalized = 50  # All same score
        normalized_scores[word] = normalized
    
    # Create word cloud from frequencies
    wordcloud = WordCloud(
        width=1200,
        height=600,
        background_color='white',
        colormap=colormap,
        max_words=max_words,
        relative_scaling=0.5,
        min_font_size=10,
        collocations=False
    ).generate_from_frequencies(normalized_scores)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=10)
    ax.axis('off')
    
    # Save
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def create_tfidf_wordclouds(genre_tfidf, output_dir):
    """
    Create TF-IDF word clouds for all genres.
    
    Args:
        genre_tfidf: Dictionary mapping genres to TF-IDF dictionaries
        output_dir: Directory to save word clouds
    """
    print("\n" + "="*70)
    print("CREATING TF-IDF WORD CLOUDS")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Color maps for variety
    colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis',
                 'twilight', 'RdYlBu', 'Spectral', 'coolwarm', 'rainbow',
                 'turbo', 'spring', 'summer', 'autumn', 'winter']
    
    for i, (genre, tfidf_dict) in enumerate(genre_tfidf.items(), 1):
        print(f"Creating word cloud {i}/{len(genre_tfidf)}: {genre}")
        
        colormap = colormaps[i % len(colormaps)]
        filename = f"{genre.replace(' ', '_')}_tfidf_wordcloud.png"
        filepath = os.path.join(output_dir, filename)
        
        create_tfidf_wordcloud(
            tfidf_dict,
            f"{genre.title()} - TF-IDF Word Cloud",
            filepath,
            colormap=colormap
        )
        
        print(f"  Saved to {filepath}")
    
    print(f"\nAll TF-IDF word clouds saved to {output_dir}/")


def create_combined_tfidf_grid(genre_tfidf, output_file, n_cols=3):
    """
    Create grid of TF-IDF word clouds.
    
    Args:
        genre_tfidf: Dictionary of TF-IDF scores
        output_file: Output file path
        n_cols: Number of columns
    """
    print("\n" + "="*70)
    print("CREATING COMBINED TF-IDF WORD CLOUD GRID")
    print("="*70)
    
    genres = list(genre_tfidf.keys())
    n_genres = len(genres)
    n_rows = (n_genres + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4))
    
    if n_genres == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()
    
    colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis',
                 'twilight', 'RdYlBu', 'Spectral', 'coolwarm', 'rainbow',
                 'turbo', 'spring', 'summer', 'autumn', 'winter']
    
    for i, genre in enumerate(genres):
        print(f"Adding {genre} to grid ({i+1}/{n_genres})")
        
        tfidf_dict = genre_tfidf[genre]
        
        if not tfidf_dict:
            axes[i].text(0.5, 0.5, f'{genre.title()}\n(No data)', 
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
        axes[i].set_title(genre.title(), fontsize=14, fontweight='bold')
        axes[i].axis('off')
    
    # Hide empty subplots
    for j in range(n_genres, len(axes)):
        axes[j].axis('off')
    
    plt.suptitle('TF-IDF Word Clouds by Genre\n(Size = Characteristic Importance)', 
                fontsize=20, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"\nCombined TF-IDF grid saved to {output_file}")
    plt.show()


def save_tfidf_results(genre_tfidf, idf_scores, output_dir):
    """
    Save TF-IDF results to JSON files.
    
    Args:
        genre_tfidf: Dictionary of TF-IDF scores
        idf_scores: Dictionary of IDF scores
        output_dir: Output directory
    """
    # Save TF-IDF scores
    tfidf_file = os.path.join(output_dir, 'tfidf_scores.json')
    with open(tfidf_file, 'w', encoding='utf-8') as f:
        json.dump(genre_tfidf, f, indent=2, ensure_ascii=False)
    print(f"TF-IDF scores saved to {tfidf_file}")
    
    # Save IDF scores
    idf_file = os.path.join(output_dir, 'idf_scores.json')
    with open(idf_file, 'w', encoding='utf-8') as f:
        json.dump(idf_scores, f, indent=2, ensure_ascii=False)
    print(f"IDF scores saved to {idf_file}")
    
    # Save top words
    top_words = {}
    for genre, tfidf_dict in genre_tfidf.items():
        sorted_words = sorted(tfidf_dict.items(), 
                             key=lambda x: x[1], 
                             reverse=True)[:20]
        top_words[genre] = [{"word": w, "score": float(s)} for w, s in sorted_words]
    
    top_file = os.path.join(output_dir, 'top_tfidf_words.json')
    with open(top_file, 'w', encoding='utf-8') as f:
        json.dump(top_words, f, indent=2, ensure_ascii=False)
    print(f"Top TF-IDF words saved to {top_file}")


def main():
    print("="*70)
    print("TF-IDF ANALYSIS & WORD CLOUDS")
    print("="*70)
    print("""
TF-IDF (Term Frequency - Inverse Document Frequency) Overview:

TF-IDF identifies words that are CHARACTERISTIC of a document by balancing:
1. TF: How often a word appears in THIS document (frequency)
2. IDF: How rare the word is ACROSS ALL documents (specificity)

Result: High TF-IDF = frequent in this genre, rare in others = CHARACTERISTIC!

This fixes the problem from Part 4 where generic words (band, album, music)
dominated all genre word clouds.
""")
    
    # Configuration
    selected_method = 'all_genres'  # Change to try different methods
    
    print(f"Using TF analysis method: {selected_method}")
    print("(Change 'selected_method' in script to try: all_genres, first_genre, first_non_rock)\n")
    
    # Paths
    tf_dir = f"Week_7/tf_analysis_{selected_method}"
    tf_file = f"{tf_dir}/term_frequencies.json"
    output_dir = f"{tf_dir}/tfidf_analysis"
    
    # Check if TF file exists
    if not os.path.exists(tf_file):
        print(f"ERROR: Term frequency file not found at {tf_file}")
        print("Please run term_frequency_analysis.py first!")
        return
    
    # Load TF data
    genre_tf = load_term_frequencies(tf_file)
    
    # Initialize TF-IDF analyzer
    analyzer = TFIDFAnalyzer(genre_tf)
    
    # Calculate IDF
    idf_scores = analyzer.calculate_idf(log_base='e')  # Natural logarithm
    
    # Calculate TF-IDF
    genre_tfidf = analyzer.calculate_tfidf(idf_scores, tf_variant='log')  # 1 + log(TF)
    
    # Display comparison
    analyzer.display_comparison(genre_tfidf, n=10)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    save_tfidf_results(genre_tfidf, idf_scores, output_dir)
    
    # Create word clouds
    wordcloud_dir = os.path.join(output_dir, 'wordclouds')
    create_tfidf_wordclouds(genre_tfidf, wordcloud_dir)
    
    # Create combined grid
    combined_file = os.path.join(wordcloud_dir, 'all_genres_tfidf_combined.png')
    create_combined_tfidf_grid(genre_tfidf, combined_file, n_cols=3)
    
    # Final summary
    print("\n" + "="*70)
    print("TF-IDF ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nGenerated files in {output_dir}/:")
    print("  - tfidf_scores.json - TF-IDF scores for all words")
    print("  - idf_scores.json - IDF scores for vocabulary")
    print("  - top_tfidf_words.json - Top 20 words per genre")
    print("  - wordclouds/ - Individual and combined word clouds")
    
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    print("""
Compare the TF-IDF word clouds with the TF word clouds from Part 4:

TF WORD CLOUDS (Part 4):
- Dominated by generic words: band, album, music, song
- Similar across all genres
- Shows what's FREQUENT but not CHARACTERISTIC

TF-IDF WORD CLOUDS (Part 6):
- Shows genre-specific vocabulary
- Different for each genre
- Reveals what makes each genre UNIQUE
- Generic words are downweighted

INTERPRETATION QUESTIONS TO ANSWER:
1. Are the TF-IDF top 10 words more descriptive than TF top 10? YES!
2. Do TF-IDF word clouds help understand genres? Examine them!
3. Do you see genre-specific terms (e.g., specific artist names, 
   subgenre terms, cultural references)?
4. How did rescaling work? We normalized TF-IDF to [1,100] range

NEXT: Optional - Apply same analysis to structural communities!
""")


if __name__ == "__main__":
    main()

