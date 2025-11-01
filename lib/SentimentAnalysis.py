import os
import json
import urllib.parse
import re
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

class SentimentAnalysis:
    def __init__(self):
        pass

    def calculate_sentiment(self, text, labmt_words):
        """Calculates the sentiment of a text using the LabMT wordlist."""
        # Tokenize: extract words (alphanumeric sequences, handling apostrophes)
        words = re.findall(r"\b[\w']+\b", text.lower())
        if not words:
            return 0
        
        total_happiness = 0
        word_count = 0
        for word in words:
            if word in labmt_words:
                total_happiness += labmt_words[word]
                word_count += 1
                
        return total_happiness / word_count if word_count > 0 else 0

    def _get_text_from_json(self, node_name):
        """
        Load Wikipedia text from JSON file.
        Node names need to be URL-encoded to match filenames.
        """
        # URL-encode the node name to match filename
        encoded_name = urllib.parse.quote(node_name.encode('utf-8'))
        encoded_name = encoded_name.replace('/', '%2F')
        
        # Try multiple possible paths
        possible_paths = [
            os.path.join('data', 'rock', 'performers', f'{encoded_name}.json'),
            os.path.join('..', 'data', 'rock', 'performers', f'{encoded_name}.json'),
            os.path.join(os.path.dirname(__file__), '..', 'data', 'rock', 'performers', f'{encoded_name}.json'),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Extract HTML content from Wikipedia API response
                        html_content = data.get('parse', {}).get('text', {}).get('*', '')
                        # Extract plain text from HTML (remove tags)
                        text = re.sub(r'<[^>]+>', ' ', html_content)
                        # Clean up whitespace
                        text = re.sub(r'\s+', ' ', text).strip()
                        return text
                except (json.JSONDecodeError, KeyError) as e:
                    return ''
        
        return ''

    def calculate_all_sentiments(self, graph, labmt_words):
        """
        Calculates sentiment for all nodes in the graph.
        
        Args:
            graph (nx.Graph): The graph to analyze.
            labmt_words (dict): The LabMT wordlist.
            
        Returns:
            dict: Dictionary mapping node names to sentiment scores.
        """
        node_sentiments = {}
        for node, _ in graph.nodes(data=True):
            text = self._get_text_from_json(node)
            node_sentiments[node] = self.calculate_sentiment(text, labmt_words)
        return node_sentiments
    
    def calculate_statistics(self, sentiments):
        """
        Calculates statistical measures for sentiment distribution.
        
        Args:
            sentiments: List or array of sentiment scores.
            
        Returns:
            dict: Dictionary with statistical measures.
        """
        sentiments_array = np.array(sentiments)
        
        # Calculate mode (handle edge cases)
        try:
            mode_result = stats.mode(sentiments_array)
            mode_value = float(mode_result.mode[0])
        except:
            # If mode calculation fails (e.g., all values unique), use median as fallback
            mode_value = np.median(sentiments_array)
        
        stats_dict = {
            'mean': np.mean(sentiments_array),
            'median': np.median(sentiments_array),
            'mode': mode_value,
            'variance': np.var(sentiments_array),
            'std': np.std(sentiments_array),
            'percentile_25': np.percentile(sentiments_array, 25),
            'percentile_75': np.percentile(sentiments_array, 75),
        }
        return stats_dict
    
    def print_statistics(self, stats_dict):
        """Prints statistical measures in a formatted way."""
        print("\n=== Sentiment Statistics ===")
        print(f"Mean: {stats_dict['mean']:.4f}")
        print(f"Median: {stats_dict['median']:.4f}")
        print(f"Mode: {stats_dict['mode']:.4f}")
        print(f"Variance: {stats_dict['variance']:.4f}")
        print(f"Standard Deviation: {stats_dict['std']:.4f}")
        print(f"25th Percentile: {stats_dict['percentile_25']:.4f}")
        print(f"75th Percentile: {stats_dict['percentile_75']:.4f}")
        print("=" * 30)
    
    def get_top_artists(self, node_sentiments, top_n=10, happiest=True):
        """
        Gets the top N happiest or saddest artists.
        
        Args:
            node_sentiments (dict): Dictionary mapping node names to sentiment scores.
            top_n (int): Number of artists to return.
            happiest (bool): If True, return happiest; if False, return saddest.
            
        Returns:
            list: List of tuples (artist_name, sentiment_score) sorted by sentiment.
        """
        sorted_artists = sorted(node_sentiments.items(), key=lambda x: x[1], reverse=happiest)
        return sorted_artists[:top_n]
    
    def print_top_artists(self, node_sentiments, top_n=10):
        """Prints the top N happiest and saddest artists."""
        happiest = self.get_top_artists(node_sentiments, top_n, happiest=True)
        saddest = self.get_top_artists(node_sentiments, top_n, happiest=False)
        
        print(f"\n=== Top {top_n} Happiest Pages ===")
        for i, (artist, sentiment) in enumerate(happiest, 1):
            print(f"{i}. {artist}: {sentiment:.4f}")
        
        print(f"\n=== Top {top_n} Saddest Pages ===")
        for i, (artist, sentiment) in enumerate(saddest, 1):
            print(f"{i}. {artist}: {sentiment:.4f}")
        print()
    
    def sentiment_distribution(self, graph, labmt_words):
        """
        Calculates and plots the sentiment distribution of a graph.
        Includes statistics and visual indicators.
        
        Args:
            graph (nx.Graph): The graph to analyze.
            labmt_words (dict): The LabMT wordlist.
            
        Returns:
            tuple: (sentiments_list, node_sentiments_dict, stats_dict)
        """
        # Calculate all sentiments
        node_sentiments = self.calculate_all_sentiments(graph, labmt_words)
        sentiments = list(node_sentiments.values())
        
        # Calculate statistics
        stats_dict = self.calculate_statistics(sentiments)
        self.print_statistics(stats_dict)
        
        # Print top artists
        self.print_top_artists(node_sentiments, top_n=10)
        
        # Plot histogram with statistics
        plt.figure(figsize=(12, 7))
        plt.hist(sentiments, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        
        # Add vertical lines for mean and mean±std
        mean = stats_dict['mean']
        std = stats_dict['std']
        plt.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.4f}')
        plt.axvline(mean + std, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Mean + Std: {mean + std:.4f}')
        plt.axvline(mean - std, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Mean - Std: {mean - std:.4f}')
        
        plt.title('Sentiment Distribution of the Network', fontsize=14, weight='bold')
        plt.xlabel('Sentiment Score', fontsize=12)
        plt.ylabel('Number of Nodes', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return sentiments, node_sentiments, stats_dict
    
    def community_sentiment_distribution(self, graph, partition, labmt_words, overall_sentiments, overall_stats):
        """
        Calculates and compares the sentiment of communities vs overall network.
        Includes vertical lines for means.
        
        Args:
            graph (nx.Graph): The graph to analyze.
            partition (dict): The community partition.
            labmt_words (dict): The LabMT wordlist.
            overall_sentiments (list): The overall sentiment distribution.
            overall_stats (dict): Statistics for overall distribution.
        """
        community_sentiments = {}
        for node, community_id in partition.items():
            if community_id not in community_sentiments:
                community_sentiments[community_id] = []
            
            # Load text from JSON file (node name is the artist name)
            text = self._get_text_from_json(node)
            sentiment = self.calculate_sentiment(text, labmt_words)
            community_sentiments[community_id].append(sentiment)
        
        # Calculate average sentiment for each community
        avg_community_sentiments = {comm: np.mean(sentiments) for comm, sentiments in community_sentiments.items()}
        
        # Sort communities by sentiment
        sorted_communities = sorted(avg_community_sentiments.items(), key=lambda item: item[1])
        
        # Get happiest and saddest communities
        saddest_community = sorted_communities[0]
        happiest_community = sorted_communities[-1]
        
        happiest_mean = happiest_community[1]
        saddest_mean = saddest_community[1]
        overall_mean = overall_stats['mean']
        
        print(f"Happiest Community: {happiest_community[0]} (Avg Sentiment: {happiest_mean:.4f})")
        print(f"Saddest Community: {saddest_community[0]} (Avg Sentiment: {saddest_mean:.4f})")
        print(f"Overall Network Mean: {overall_mean:.4f}")
        
        # Plot distributions
        plt.figure(figsize=(14, 8))
        plt.hist(overall_sentiments, bins=20, color='gray', alpha=0.5, label='Overall Network', density=True)
        plt.hist(community_sentiments[happiest_community[0]], bins=20, color='green', alpha=0.6, 
                label=f'Happiest Community ({happiest_community[0]})', density=True)
        plt.hist(community_sentiments[saddest_community[0]], bins=20, color='red', alpha=0.6, 
                label=f'Saddest Community ({saddest_community[0]})', density=True)
        
        # Add vertical lines for the three means
        plt.axvline(happiest_mean, color='green', linestyle='--', linewidth=2.5, 
                   label=f'Happiest Mean: {happiest_mean:.4f}')
        plt.axvline(overall_mean, color='blue', linestyle='--', linewidth=2.5, 
                   label=f'Overall Mean: {overall_mean:.4f}')
        plt.axvline(saddest_mean, color='red', linestyle='--', linewidth=2.5, 
                   label=f'Saddest Mean: {saddest_mean:.4f}')
        
        plt.title('Community Sentiment vs. Overall Network Sentiment', fontsize=14, weight='bold')
        plt.xlabel('Sentiment Score', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()