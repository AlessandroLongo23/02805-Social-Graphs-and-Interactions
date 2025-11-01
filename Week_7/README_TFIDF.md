# Part 5/6: TF-IDF Analysis and Word Clouds

This directory contains scripts for **TF-IDF (Term Frequency - Inverse Document Frequency)** analysis, which identifies **characteristic words** that distinguish genres and communities.

## What is TF-IDF?

**TF-IDF** is a numerical statistic that shows how important a word is to a document within a collection:

- **TF (Term Frequency)**: How often a word appears in THIS document
- **IDF (Inverse Document Frequency)**: How rare the word is ACROSS ALL documents
- **TF-IDF = TF × IDF**: Words that are frequent in one document but rare overall get high scores

### Why TF-IDF?

**Problem with raw TF (Part 4)**: Word clouds dominated by generic words (band, album, music) that appear in ALL genres.

**TF-IDF solution**: Downweights common words, highlights genre-specific vocabulary that makes each text unique!

---

## Part 5: Understanding TF-IDF (Theory)

### Questions to Answer:

1. **Alternative TF Definitions**:
   - Raw count: Just the frequency
   - Logarithmic: `1 + log(count)` - reduces impact of very frequent words
   - Normalized: `count / max_count` - normalizes across different document lengths
   
   **Logarithmic is often better** because it prevents very common words from dominating.

2. **What does IDF stand for?**
   - **Inverse Document Frequency** - measures how rare/common a word is across the corpus
   
3. **How does IDF work?**
   - IDF = log(N / df) where N = total documents, df = documents containing the word
   - **Rare words** (small df) → **high IDF** → emphasized
   - **Common words** (large df) → **low IDF** → de-emphasized

4. **Why logarithm in IDF?**
   - Without log: word appearing in 1 doc vs 10 docs would have 10× difference
   - With log: differences are compressed, more balanced scaling
   - Prevents extreme weights that could distort analysis

5. **Why does IDF reduce need for stopword removal?**
   - Stopwords appear in ALL documents → df = N → IDF = log(N/N) = log(1) = 0
   - TF-IDF = TF × 0 = 0 (automatically filtered out!)
   - Generic words get naturally downweighted even without manual removal

---

## Part 6: TF-IDF Word Clouds

### Scripts

#### 1. `tfidf_analysis.py` - Genre TF-IDF Analysis

**Purpose**: Calculate TF-IDF and create word clouds for the top 15 genres.

**Features**:
- Loads TF data from Part 3
- Calculates IDF scores across all genres
- Computes TF-IDF for each word in each genre
- Compares TF vs TF-IDF top words
- Creates individual and combined word clouds
- Saves results to JSON files

**Usage**:
```bash
cd Week_7
python tfidf_analysis.py
```

**Configuration**:
- Edit `selected_method` variable to choose genre assignment:
  - `'all_genres'` - Include all genres for each artist
  - `'first_genre'` - Use only first genre
  - `'first_non_rock'` - Use first non-"rock" genre

**Output**:
- `tf_analysis_{method}/tfidf_analysis/`
  - `tfidf_scores.json` - Full TF-IDF scores
  - `idf_scores.json` - IDF scores for vocabulary
  - `top_tfidf_words.json` - Top 20 words per genre
  - `wordclouds/` - Individual word clouds for each genre
  - `wordclouds/all_genres_tfidf_combined.png` - Grid view

**Key Questions Answered**:
- ✅ Are TF-IDF top 10 words more descriptive than TF? **Compare output!**
- ✅ Do word clouds help understand genres? **Visual analysis!**
- ✅ How did you rescale TF-IDF? **Normalized to [1, 100] range**
- ✅ Which log base? **Natural log (base e) by default**

---

#### 2. `tfidf_communities.py` - Community TF-IDF Analysis (Optional)

**Purpose**: Apply TF-IDF to the 15 largest structural communities from Louvain algorithm.

**Features**:
- Loads community partition from Part 2
- Aggregates all text for each community
- Calculates TF-IDF for communities
- Creates word clouds to characterize communities
- Enables comparison with genre-based analysis

**Usage**:
```bash
cd Week_7
python tfidf_communities.py
```

**Requirements**:
- Must run `community_detection.py` first (Part 2)
- Needs access to `data/rock/performers/*.json`

**Output**:
- `Week_7/tfidf_communities/`
  - `community_tfidf_scores.json` - Full TF-IDF scores
  - `community_idf_scores.json` - IDF scores
  - `top_community_tfidf_words.json` - Top 20 words per community
  - `wordclouds/` - Individual word clouds
  - `wordclouds/all_communities_tfidf_combined.png` - Grid view

**Key Questions**:
- ✅ Are community clouds more meaningful than genre clouds?
- ✅ Can you identify what makes each community distinct?
- ✅ How does confusion matrix relate to vocabulary overlap?
- ✅ Which better characterizes structure: genres or communities?

---

## Implementation Details

### TF Variant Used
We use **logarithmic TF**: `TF = 1 + log(count)`

**Why?**
- Reduces impact of extremely frequent words
- Better balance between common and rare words
- Prevents one very frequent word from dominating

### IDF Formula
`IDF(word) = log(N / df(word))`

Where:
- N = total number of documents (genres or communities)
- df(word) = number of documents containing the word
- log = natural logarithm (base e)

### TF-IDF Calculation
`TF-IDF(word, doc) = TF(word, doc) × IDF(word)`

### Word Cloud Rescaling
**Problem**: TF-IDF scores can be in various ranges (e.g., 0.1 to 50)

**Solution**: Normalize to [1, 100] range:
```python
normalized = 1 + 99 * (score - min) / (max - min)
```

This ensures:
- All scores are positive
- Reasonable range for visualization
- Relative importance preserved

---

## Workflow

### Complete Assignment Workflow:

```bash
# Part 1: Extract genres
python main.py

# Part 2: Community detection
python community_detection.py

# Part 3: Confusion matrix
python confusion_matrix_analysis.py

# Part 4: Term Frequency analysis
python term_frequency_analysis.py

# Part 5: TF word clouds
python wordcloud_generator.py

# Part 6a: TF-IDF for genres ⭐ NEW
python tfidf_analysis.py

# Part 6b: TF-IDF for communities (optional) ⭐ NEW
python tfidf_communities.py
```

---

## Expected Results

### Comparison: TF vs TF-IDF

**TF Word Clouds (Part 4)**:
- Generic words: band, album, music, song, released, formed
- Similar across all genres
- Shows what's **frequent**

**TF-IDF Word Clouds (Part 6)**:
- Genre-specific vocabulary
- Different for each genre
- Shows what's **characteristic**

### Example Genre Differences

**Heavy Metal** (TF-IDF):
- Characteristic: metal, heavy, thrash, doom, iron, maiden, metallica

**Hip Hop** (TF-IDF):
- Characteristic: hop, rap, rapper, beats, rhymes, mc, dj

**Country** (TF-IDF):
- Characteristic: country, nashville, bluegrass, folk, western, honky

---

## Dependencies

All scripts require:
```bash
pip install numpy matplotlib wordcloud
```

For community TF-IDF analysis (`tfidf_communities.py`):
```bash
pip install beautifulsoup4 lxml
```

Network analysis (if not already installed):
```bash
pip install networkx python-louvain
```

---

## Troubleshooting

### Issue: "Term frequency file not found"
**Solution**: Run `term_frequency_analysis.py` first

### Issue: "Community partition file not found"
**Solution**: Run `community_detection.py` first

### Issue: Import error for TFIDFAnalyzer
**Solution**: Ensure both scripts are in the same directory (Week_7/)

### Issue: Out of memory
**Solution**: Reduce `top_n` parameter (e.g., from 15 to 10 communities)

---

## Key Insights & Interpretation

### TF vs TF-IDF:
1. **TF shows frequency** → generic words dominate
2. **TF-IDF shows distinctiveness** → genre-specific words emerge

### Genres vs Communities:
1. **Genre word clouds**: Based on musical labels (metadata)
2. **Community word clouds**: Based on network structure (data-driven)

**Comparison helps answer**: Do structural communities align with musical genres?

### Analysis Questions:
- Are TF-IDF top words more descriptive? → **Examine output!**
- Do clouds help understand genres? → **Look for domain-specific terms**
- Communities vs genres? → **Compare with confusion matrix**

---

## Files Generated

```
Week_7/
├── tfidf_analysis.py              # Main TF-IDF script
├── tfidf_communities.py           # Communities TF-IDF script
├── README_TFIDF.md               # This file
│
├── tf_analysis_all_genres/
│   └── tfidf_analysis/
│       ├── tfidf_scores.json
│       ├── idf_scores.json
│       ├── top_tfidf_words.json
│       └── wordclouds/
│           ├── heavy_metal_tfidf_wordcloud.png
│           ├── hip_hop_tfidf_wordcloud.png
│           └── all_genres_tfidf_combined.png
│
└── tfidf_communities/
    ├── community_tfidf_scores.json
    ├── community_idf_scores.json
    ├── top_community_tfidf_words.json
    └── wordclouds/
        ├── Community_0_tfidf_wordcloud.png
        ├── Community_1_tfidf_wordcloud.png
        └── all_communities_tfidf_combined.png
```

---

## References

- [TF-IDF on Wikipedia](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- Network Science Book - Chapter 9 (Communities)
- Week 7 Assignment Notebook

---

## Questions?

If you encounter issues or have questions:
1. Check that all previous scripts (Parts 1-4) ran successfully
2. Verify file paths in the configuration section
3. Ensure all dependencies are installed
4. Check that performer JSON files exist in `data/rock/performers/`

---

**Ready to analyze!** Run the scripts and examine the word clouds to understand what makes each genre and community unique! 🎵✨

