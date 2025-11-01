import json
import os
from bs4 import BeautifulSoup
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm

class GenresExtractor():
    def __init__(self):
        pass

    def normalize_genre(self, genre):
        """Normalize genre names by converting to lowercase and handling common variations."""
        genre = genre.lower().strip()
        
        # Map common variations to standard forms
        genre_mappings = {
            "rock'n'roll": "rock and roll",
            "rock & roll": "rock and roll",
            "rock 'n' roll": "rock and roll",
            "rock n roll": "rock and roll",
            "rock-and-roll": "rock and roll",
            "hard rock": "hard rock",
            "heavy metal": "heavy metal",
            "alternative rock": "alternative rock",
            "punk rock": "punk rock",
            "blues rock": "blues rock",
            "folk rock": "folk rock",
            "progressive rock": "progressive rock",
            "glam rock": "glam rock",
            "pop rock": "pop rock",
        }
        
        return genre_mappings.get(genre, genre)

    def extract_genres_from_html(self, html_content):
        """
        Extract genres from Wikipedia infobox HTML content.
        
        Args:
            html_content: HTML string from Wikipedia page
        
        Returns:
            A list of genre strings (lowercase and normalized)
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        genres = []
        
        # Strategy 1: Look for infobox rows with "Genre" or "Genres" label
        # Wikipedia infoboxes typically use <tr> with <th> for labels and <td> for values
        for row in soup.find_all('tr'):
            # Find the header cell
            header = row.find('th')
            if header:
                header_text = header.get_text(strip=True).lower()
                
                # Check if this is a genre row
                if 'genre' in header_text and 'musical' not in header_text:
                    # Find the data cell
                    data_cell = row.find('td')
                    if data_cell:
                        # Extract text from links (genres are usually linked)
                        genre_links = data_cell.find_all('a')
                        if genre_links:
                            for link in genre_links:
                                genre_text = link.get_text(strip=True)
                                if genre_text and not genre_text.startswith('['):
                                    genres.append(self.normalize_genre(genre_text))
                        else:
                            # If no links, just get the text
                            genre_text = data_cell.get_text(strip=True)
                            # Split by common delimiters
                            for delimiter in [',', ';', '\n']:
                                if delimiter in genre_text:
                                    for g in genre_text.split(delimiter):
                                        g = g.strip()
                                        if g and not g.startswith('['):
                                            genres.append(self.normalize_genre(g))
                                    break
                            else:
                                if genre_text and not genre_text.startswith('['):
                                    genres.append(self.normalize_genre(genre_text))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_genres = []
        for genre in genres:
            if genre not in seen:
                seen.add(genre)
                unique_genres.append(genre)
        
        return unique_genres

    def extract_genres_from_performers(self, performers_dir):
        """
        Extract genres from all performer JSON files in the specified directory.
        
        Args:
            performers_dir: Path to directory containing performer JSON files
        
        Returns:
            A dictionary mapping artist names to lists of genres
        """
        artist_genres = {}
        
        # Get all JSON files in the directory
        json_files = [f for f in os.listdir(performers_dir) if f.endswith('.json')]
        
        print(f"Processing {len(json_files)} performer files...")
        
        no_genre_count = 0
        processed_count = 0
        
        for filename in tqdm(json_files):
            filepath = os.path.join(performers_dir, filename)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract artist name and HTML content
                if 'parse' in data and 'title' in data['parse']:
                    artist_name = data['parse']['title']
                    
                    # Get HTML content
                    if 'text' in data['parse'] and '*' in data['parse']['text']:
                        html_content = data['parse']['text']['*']
                        
                        # Extract genres
                        genres = self.extract_genres_from_html(html_content)
                        
                        if genres:
                            artist_genres[artist_name] = genres
                            processed_count += 1
                        else:
                            no_genre_count += 1
            
            except Exception as e:
                print(f"  Error processing {filename}: {e}")
        
        print(f"\nSuccessfully extracted genres for {processed_count} artists")
        print(f"No genres found for {no_genre_count} artists")
        
        return artist_genres

    def analyze_genres(self, artist_genres):
        """
        Analyze and display statistics about the extracted genres.
        
        Args:
            artist_genres: Dictionary mapping artist names to lists of genres
        """
        # Count nodes with genres
        num_nodes_with_genres = len(artist_genres)
        print(f"\n{'='*60}")
        print(f"GENRE EXTRACTION STATISTICS")
        print(f"{'='*60}")
        print(f"Number of nodes with genres: {num_nodes_with_genres}")
        
        # Calculate average number of genres per node
        total_genres = sum(len(genres) for genres in artist_genres.values())
        avg_genres = total_genres / num_nodes_with_genres if num_nodes_with_genres > 0 else 0
        print(f"Average number of genres per node: {avg_genres:.2f}")
        
        # Count distinct genres
        all_genres = []
        for genres in artist_genres.values():
            all_genres.extend(genres)
        
        genre_counts = Counter(all_genres)
        num_distinct_genres = len(genre_counts)
        print(f"Total number of distinct genres: {num_distinct_genres}")
        
        # Get top 15 genres
        top_15_genres = genre_counts.most_common(15)
        
        print(f"\n{'='*60}")
        print(f"TOP 15 GENRES")
        print(f"{'='*60}")
        for i, (genre, count) in enumerate(top_15_genres, 1):
            print(f"{i:2d}. {genre:<30} {count:>4} artists")
        
        # Create histogram
        plt.figure(figsize=(14, 8))
        genres_list, counts_list = zip(*top_15_genres)
        
        bars = plt.barh(range(len(genres_list)), counts_list, color='steelblue')
        plt.yticks(range(len(genres_list)), genres_list)
        plt.xlabel('Number of Artists', fontsize=12)
        plt.ylabel('Genre', fontsize=12)
        plt.title('Top 15 Genres by Artist Count', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()  # Highest count at the top
        
        # Add count labels on bars
        for i, (bar, count) in enumerate(zip(bars, counts_list)):
            plt.text(count, i, f' {count}', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.show()
        
        return genre_counts, top_15_genres

    def save_artist_genres(self, artist_genres, output_file):
        """Save the artist-genre dictionary to a JSON file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(artist_genres, f, indent=2, ensure_ascii=False)
