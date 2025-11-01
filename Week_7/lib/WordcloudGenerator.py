import json
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from lib.Loader import Loader

class WordcloudGenerator():
    def __init__(self):
        pass

    def tf_dict_to_text(self, tf_dict):
        """
        Convert TF dictionary to text string by repeating words according to frequency.
        
        Args:
            tf_dict: Dictionary mapping words to frequencies
        
        Returns:
            String with words repeated by their frequency
        """
        words = []
        for word, count in tf_dict.items():
            # Repeat each word according to its count
            words.extend([word] * count)
        
        return ' '.join(words)


    def create_wordcloud(self, text, title, colormap='viridis', max_words=200, 
                        width=800, height=400, background_color='white'):
        """
        Create a word cloud from text.
        
        Args:
            text: Input text string
            title: Title for the word cloud
            colormap: Matplotlib colormap name
            max_words: Maximum number of words to display
            width: Image width
            height: Image height
            background_color: Background color
        
        Returns:
            WordCloud object
        """
        # Create word cloud
        # Set collocations=False as recommended (prevents the package from finding phrases)
        wordcloud = WordCloud(
            width=width,
            height=height,
            background_color=background_color,
            colormap=colormap,
            max_words=max_words,
            collocations=False,  # Important: prevents artificial phrase detection
            relative_scaling=0.5,
            min_font_size=10
        ).generate(text)
        
        return wordcloud


    def plot_wordcloud(self, wordcloud, title, ax=None):
        """
        Plot a word cloud.
        
        Args:
            wordcloud: WordCloud object
            title: Plot title
            ax: Matplotlib axis (optional)
        """
        if ax is None:
            plt.figure(figsize=(10, 6))
            ax = plt.gca()
        
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=10)
        ax.axis('off')


    def save_individual_wordclouds(self, genre_tf, output_dir, colormap='viridis'):
        """
        Create and save individual word clouds for each genre.
        
        Args:
            genre_tf: Dictionary mapping genres to TF dictionaries
            output_dir: Directory to save word clouds
            colormap: Matplotlib colormap name
        """
        print("="*70)
        print("CREATING INDIVIDUAL WORD CLOUDS")
        print("="*70)
        
        os.makedirs(output_dir, exist_ok=True)
        
        for i, (genre, tf_dict) in enumerate(genre_tf.items(), 1):
            print(f"Creating word cloud {i}/{len(genre_tf)}: {genre}")
            
            # Convert TF dict to text
            text = self.tf_dict_to_text(tf_dict)
            
            if not text.strip():
                print(f"  Skipping {genre} (no text data)")
                continue
            
            # Create word cloud
            wordcloud = self.create_wordcloud(
                text, 
                title=f"{genre.title()} Genre",
                colormap=colormap,
                max_words=200,
                width=1200,
                height=600
            )
            
            # Create figure and plot
            fig, ax = plt.subplots(figsize=(12, 6))
            self.plot_wordcloud(wordcloud, f"{genre.title()} Genre", ax)
            
            # Save
            filename = f"{genre.replace(' ', '_')}_wordcloud.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"  Saved to {filepath}")
        
        print(f"\nAll individual word clouds saved to {output_dir}/")


    def create_combined_wordcloud_grid(self, genre_tf, output_file, n_cols=3):
        """
        Create a grid of word clouds for all genres.
        
        Args:
            genre_tf: Dictionary mapping genres to TF dictionaries
            output_file: Path to save combined visualization
            n_cols: Number of columns in grid
        """
        print("\n" + "="*70)
        print("CREATING COMBINED WORD CLOUD GRID")
        print("="*70)
        
        genres = list(genre_tf.keys())
        n_genres = len(genres)
        n_rows = (n_genres + n_cols - 1) // n_cols  # Ceiling division
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4))
        
        # Flatten axes array for easier indexing
        if n_genres == 1:
            axes = np.array([axes])
        else:
            axes = axes.flatten()
        
        # Color maps for variety
        colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis',
                    'twilight', 'RdYlBu', 'Spectral', 'coolwarm', 'rainbow',
                    'turbo', 'spring', 'summer', 'autumn', 'winter']
        
        for i, genre in enumerate(genres):
            print(f"Adding {genre} to grid ({i+1}/{n_genres})")
            
            tf_dict = genre_tf[genre]
            text = self.tf_dict_to_text(tf_dict)
            
            if not text.strip():
                axes[i].text(0.5, 0.5, f'{genre.title()}\n(No data)', 
                            ha='center', va='center', fontsize=14)
                axes[i].axis('off')
                continue
            
            # Create word cloud with rotating colormaps
            colormap = colormaps[i % len(colormaps)]
            wordcloud = self.create_wordcloud(
                text,
                title=genre.title(),
                colormap=colormap,
                max_words=150,
                width=800,
                height=400
            )
            
            # Plot
            self.plot_wordcloud(wordcloud, genre.title(), axes[i])
        
        # Hide empty subplots
        for j in range(n_genres, len(axes)):
            axes[j].axis('off')
        
        plt.suptitle('Word Clouds by Genre (Term Frequency)', 
                    fontsize=20, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"\nCombined word cloud grid saved to {output_file}")
        plt.show()


    def analyze_wordcloud_results(self, genre_tf):
        """
        Print analysis and observations about the word clouds.
        
        Args:
            genre_tf: Dictionary mapping genres to TF dictionaries
        """
        print("\n" + "="*70)
        print("WORD CLOUD ANALYSIS")
        print("="*70)
        
        print("\nTop 5 words per genre:")
        print("-" * 70)
        
        for genre, tf_dict in genre_tf.items():
            # Get top 5 words
            top_words = sorted(tf_dict.items(), key=lambda x: x[1], reverse=True)[:5]
            words_str = ', '.join([f"{word} ({count})" for word, count in top_words])
            print(f"{genre.title():<30} {words_str}")
        
        print("\n" + "="*70)
        print("OBSERVATIONS")
        print("="*70)
        print("""
    Key Findings from Word Clouds:

    1. GENERIC WORDS DOMINATE:
    - Words like 'band', 'album', 'music', 'song' appear across all genres
    - This is expected with pure term frequency (TF) without IDF weighting
    - These words don't help distinguish between genres

    2. GENRE-SPECIFIC HINTS:
    - Despite generic dominance, you may notice some genre-specific terms
    - Look for words that appear prominently in one genre but not others
    - These are the words that TF-IDF will highlight

    3. VISUALIZATION INSIGHTS:
    - Larger words appear more frequently
    - Word size = frequency (not importance)
    - Colors are just for aesthetics (no meaning)

    4. LIMITATIONS OF TF ALONE:
    - Can't distinguish "common to all rock music" from "unique to genre"
    - Wikipedia articles share similar structure (band, album, song, etc.)
    - Need IDF component to identify characteristic words

    NEXT STEPS:
    - Part 5: Learn about TF-IDF theory
    - Part 6: Apply TF-IDF to find truly characteristic words per genre
    - Compare TF word clouds with TF-IDF word clouds

    The difference will be dramatic - TF-IDF word clouds will show what makes
    each genre UNIQUE rather than just what's FREQUENT!
    """)


# def main():
#     loader = Loader()
#     wordcloud_generator = WordcloudGenerator()

#     print("="*70)
#     print("WORD CLOUD GENERATION")
#     print("="*70)
    
#     # Configuration
#     print("\nAvailable TF analysis methods:")
#     methods = ['all_genres', 'first_genre', 'first_non_rock']
#     for i, method in enumerate(methods, 1):
#         print(f"  {i}. {method}")
    
#     # You can change this to use a different method
#     selected_method = 'all_genres'  # Change to 'first_genre' or 'first_non_rock' if desired
    
#     print(f"\nUsing method: {selected_method}")
#     print("(You can change 'selected_method' in the script to try other methods)\n")
    
#     # Paths
#     tf_dir = f"Week_7/tf_analysis_{selected_method}"
#     tf_file = f"{tf_dir}/term_frequencies.json"
    
#     # Check if file exists
#     if not os.path.exists(tf_file):
#         print(f"ERROR: Term frequency file not found at {tf_file}")
#         print("Please run term_frequency_analysis.py first!")
#         return
    
#     # Load TF data
#     genre_tf = loader.load_term_frequencies(tf_file)
    
#     # Create output directory
#     wordcloud_dir = f"{tf_dir}/wordclouds"
    
#     # Create individual word clouds
#     wordcloud_generator.save_individual_wordclouds(genre_tf, wordcloud_dir, colormap='viridis')
    
#     # Create combined grid
#     combined_file = f"{wordcloud_dir}/all_genres_combined.png"
#     wordcloud_generator.create_combined_wordcloud_grid(genre_tf, combined_file, n_cols=3)
    
#     # Analysis
#     wordcloud_generator.analyze_wordcloud_results(genre_tf)
    
#     print("\n" + "="*70)
#     print("WORD CLOUD GENERATION COMPLETE!")
#     print("="*70)
#     print(f"\nGenerated files in {wordcloud_dir}/:")
#     print("  - Individual word cloud for each genre (PNG)")
#     print("  - all_genres_combined.png (grid view)")
#     print("\nYou can now:")
#     print("  1. Examine the word clouds to see term frequencies")
#     print("  2. Note the generic words that dominate")
#     print("  3. Proceed to Part 5 (TF-IDF) to fix this limitation")
#     print("  4. Try other methods by changing 'selected_method' in the script")


# if __name__ == "__main__":
#     main()

