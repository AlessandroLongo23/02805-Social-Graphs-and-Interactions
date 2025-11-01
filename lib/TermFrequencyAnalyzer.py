import json
import re
import os
from collections import Counter, defaultdict
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import random
from lib.Loader import Loader

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class GenreAssignmentMethod:
    """Enum-like class for genre assignment methods."""
    ALL_GENRES = "all_genres"  # Count page for each genre it has
    FIRST_GENRE = "first_genre"  # Use only first genre
    SECOND_GENRE = "second_genre"  # Use only second genre (if exists)
    RANDOM_GENRE = "random_genre"  # Pick random genre from list
    FIRST_NON_ROCK = "first_non_rock"  # Use first non-rock genre


class TermFrequencyAnalyzer():
    def __init__(self):
        pass

    def get_top_genres(self, artist_genres, n=15):
        """Get the top N most common genres."""
        print(f"Identifying top {n} genres...")
        
        all_genres = []
        for genres in artist_genres.values():
            all_genres.extend(genres)
        
        genre_counts = Counter(all_genres)
        top_genres = [genre for genre, _ in genre_counts.most_common(n)]
        
        print(f"Top {n} genres:")
        for i, genre in enumerate(top_genres, 1):
            print(f"  {i:2d}. {genre:<30} ({genre_counts[genre]} occurrences)")
        
        return top_genres


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


    def load_performer_text(self, performers_dir, artist_name):
        """Load Wikipedia text for a specific performer."""
        # URL encode the artist name for filename
        filename = artist_name.replace(" ", "%20").replace("/", "%2F") + ".json"
        filepath = os.path.join(performers_dir, filename)
        
        if not os.path.exists(filepath):
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'parse' in data and 'text' in data['parse'] and '*' in data['parse']['text']:
                html_content = data['parse']['text']['*']
                text = self.extract_text_from_html(html_content)
                return text
        except Exception as e:
            print(f"  Warning: Error loading {artist_name}: {e}")
        
        return None


    def clean_and_tokenize(self, text, remove_stopwords=True, lemmatize=False, min_word_length=3):
        """
        Clean and tokenize text.
        
        Args:
            text: Raw text string
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to lemmatize words
            min_word_length: Minimum word length to keep
        
        Returns:
            List of cleaned tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove wiki-specific syntax
        text = re.sub(r'\[edit\]', '', text)
        text = re.sub(r'\[\d+\]', '', text)  # Remove reference numbers
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove punctuation and non-alphabetic tokens
        tokens = [token for token in tokens if token.isalpha()]
        
        # Filter by minimum length
        tokens = [token for token in tokens if len(token) >= min_word_length]
        
        # Remove stopwords
        if remove_stopwords:
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token not in stop_words]
        
        # Lemmatize
        if lemmatize:
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        return tokens


    def assign_artist_to_genres(self, artist, genres, method, top_genres):
        """
        Assign artist to one or more genres based on the method.
        
        Args:
            artist: Artist name
            genres: List of genres for this artist
            method: Genre assignment method
            top_genres: List of top genres to consider
        
        Returns:
            List of genres to assign this artist to
        """
        # Filter to only top genres
        valid_genres = [g for g in genres if g in top_genres]
        
        if not valid_genres:
            return []
        
        if method == GenreAssignmentMethod.ALL_GENRES:
            return valid_genres
        
        elif method == GenreAssignmentMethod.FIRST_GENRE:
            return [valid_genres[0]]
        
        elif method == GenreAssignmentMethod.SECOND_GENRE:
            if len(valid_genres) > 1:
                return [valid_genres[1]]
            return [valid_genres[0]]  # Fallback to first
        
        elif method == GenreAssignmentMethod.RANDOM_GENRE:
            return [random.choice(valid_genres)]
        
        elif method == GenreAssignmentMethod.FIRST_NON_ROCK:
            non_rock = [g for g in valid_genres if 'rock' not in g.lower()]
            if non_rock:
                return [non_rock[0]]
            return [valid_genres[0]]  # Fallback to first
        
        return [valid_genres[0]]  # Default


    def aggregate_text_by_genre(self, artist_genres, performers_dir, top_genres, 
                                method=GenreAssignmentMethod.ALL_GENRES,
                                lemmatize=False, min_word_freq=5):
        """
        Aggregate and tokenize text for each genre.
        
        Args:
            artist_genres: Dictionary mapping artists to genres
            performers_dir: Directory containing performer JSON files
            top_genres: List of top genres
            method: Genre assignment method
            lemmatize: Whether to lemmatize words
            min_word_freq: Minimum frequency to keep a word
        
        Returns:
            Dictionary mapping genres to term frequency dictionaries
        """
        print("\n" + "="*70)
        print(f"AGGREGATING TEXT BY GENRE (Method: {method})")
        print("="*70)
        
        # Initialize genre text collections
        genre_tokens = defaultdict(list)
        artists_per_genre = defaultdict(set)
        
        # Process each artist
        total_artists = len(artist_genres)
        processed = 0
        skipped = 0
        
        for artist, genres in artist_genres.items():
            # Determine which genre(s) to assign this artist to
            assigned_genres = self.assign_artist_to_genres(artist, genres, method, top_genres)
            
            if not assigned_genres:
                continue
            
            # Load and process text
            text = self.load_performer_text(performers_dir, artist)
            
            if text:
                # Tokenize and clean
                tokens = self.clean_and_tokenize(text, 
                                        remove_stopwords=True, 
                                        lemmatize=lemmatize,
                                        min_word_length=3)
                
                # Add to each assigned genre
                for genre in assigned_genres:
                    genre_tokens[genre].extend(tokens)
                    artists_per_genre[genre].add(artist)
                
                processed += 1
            else:
                skipped += 1
            
            if (processed + skipped) % 50 == 0:
                print(f"  Processed {processed + skipped}/{total_artists} artists...")
        
        print(f"\nProcessed {processed} artists, skipped {skipped} (no text found)")
        print(f"\nGenre statistics:")
        for genre in top_genres:
            if genre in artists_per_genre:
                print(f"  {genre:<30} {len(artists_per_genre[genre]):>4} artists, "
                    f"{len(genre_tokens[genre]):>8} tokens")
        
        # Calculate term frequencies and filter rare words
        print(f"\nCalculating term frequencies (filtering words with freq < {min_word_freq})...")
        genre_tf = {}
        
        for genre in top_genres:
            if genre in genre_tokens:
                # Count tokens
                token_counts = Counter(genre_tokens[genre])
                
                # Filter rare words
                filtered_counts = {word: count for word, count in token_counts.items() 
                                if count >= min_word_freq}
                
                genre_tf[genre] = filtered_counts
                print(f"  {genre:<30} {len(filtered_counts):>6} unique words "
                    f"(filtered from {len(token_counts)})")
        
        return genre_tf


    def display_top_words(self, genre_tf, n=15):
        """Display top N words for each genre."""
        print("\n" + "="*70)
        print(f"TOP {n} WORDS PER GENRE")
        print("="*70)
        
        for genre, tf_dict in genre_tf.items():
            print(f"\n{genre.upper()}")
            print("-" * 50)
            
            # Get top N words
            top_words = Counter(tf_dict).most_common(n)
            
            for i, (word, count) in enumerate(top_words, 1):
                print(f"  {i:2d}. {word:<20} {count:>6}")


    def save_term_frequencies(self, genre_tf, output_file):
        """Save term frequency data to JSON."""
        # Convert Counter objects to regular dicts for JSON serialization
        json_data = {genre: dict(tf_dict) for genre, tf_dict in genre_tf.items()}
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        

    def create_tf_summary(self, genre_tf, top_genres, method):
        """Create a summary visualization of term frequencies."""
        print("\nCreating term frequency summary visualization...")
        
        # Calculate statistics
        genre_stats = []
        for genre in top_genres:
            if genre in genre_tf:
                tf_dict = genre_tf[genre]
                genre_stats.append({
                    'genre': genre,
                    'unique_words': len(tf_dict),
                    'total_words': sum(tf_dict.values()),
                    'avg_freq': sum(tf_dict.values()) / len(tf_dict) if tf_dict else 0
                })
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: Number of unique words per genre
        genres = [s['genre'] for s in genre_stats]
        unique_counts = [s['unique_words'] for s in genre_stats]
        
        axes[0].barh(range(len(genres)), unique_counts, color='steelblue')
        axes[0].set_yticks(range(len(genres)))
        axes[0].set_yticklabels(genres, fontsize=10)
        axes[0].set_xlabel('Number of Unique Words', fontsize=12)
        axes[0].set_title('Vocabulary Size by Genre', fontsize=14, fontweight='bold')
        axes[0].invert_yaxis()
        
        # Plot 2: Total word count per genre
        total_counts = [s['total_words'] for s in genre_stats]
        
        axes[1].barh(range(len(genres)), total_counts, color='coral')
        axes[1].set_yticks(range(len(genres)))
        axes[1].set_yticklabels(genres, fontsize=10)
        axes[1].set_xlabel('Total Word Count', fontsize=12)
        axes[1].set_title('Total Words by Genre', fontsize=14, fontweight='bold')
        axes[1].invert_yaxis()
        
        plt.suptitle(f'Term Frequency Analysis\nMethod: {method}', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()