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

    def create_combined_wordcloud_grid(self, genre_tf, n_cols=3):
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
