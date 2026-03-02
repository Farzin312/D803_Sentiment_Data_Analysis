"""
Step 2: NLP Preprocessing Pipeline for Sentiment Analysis.

This script applies the following NLP preprocessing techniques to the
IMDB movie review dataset:

    1. Handling missing data
    2. Duplicate removal
    3. Lowercasing
    4. HTML tag removal
    5. Punctuation and special character removal (preserving ratings)
    6. Tokenization
    7. Selective Stopword removal (preserving negation)
    8. Lemmatization
    9. Text Embedding (TF-IDF vectorization for machine learning readiness)

The preprocessed dataset is saved as a CSV for downstream sentiment
analysis tasks.

Dataset: IMDB Dataset of 50K Movie Reviews
Source:  https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
"""

import os
import re
import time

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------------------------------------------------
# NLTK resource downloads (one-time)
# ---------------------------------------------------------------------------
NLTK_RESOURCES = ["punkt_tab", "stopwords", "wordnet", "omw-1.4"]
for resource in NLTK_RESOURCES:
    nltk.download(resource, quiet=True)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(__file__)
INPUT_FILE = os.path.join(SCRIPT_DIR, "IMDB_Dataset.csv")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "IMDB_Preprocessed.csv")
SAMPLE_OUTPUT_FILE = os.path.join(SCRIPT_DIR, "IMDB_Vectorized_Sample.csv")

# ---------------------------------------------------------------------------
# Initialise NLP tools
# ---------------------------------------------------------------------------
# Load standard English stopwords and remove negation words that are critical for sentiment
NEGATION_WORDS = {
    "no", "nor", "not", "none", "neither", "never", "nt",
    "isnt", "arent", "wasnt", "werent", "havent", "hasnt", "hadnt",
    "dont", "doesnt", "didnt", "wont", "wouldnt", "shant", "shouldnt",
    "cant", "cannot", "couldnt", "mustnt"
}
STOP_WORDS = set(stopwords.words("english")) - NEGATION_WORDS
LEMMATIZER = WordNetLemmatizer()


# ---------------------------------------------------------------------------
# Preprocessing functions
# ---------------------------------------------------------------------------
def remove_html_tags(text: str) -> str:
    """Remove HTML tags such as <br />, <p>, etc."""
    return re.sub(r"<[^>]+>", " ", text)


def remove_special_characters(text: str) -> str:
    """
    Remove non-alphanumeric characters, but preserve numbers and rating
    patterns (e.g., 10/10) which are critical for sentiment.
    """
    # Preserve patterns like 10/10 or 5/5 by converting them to words
    text = re.sub(r"(\d+)/(\d+)", r" \1outof\2 ", text)
    # Remove everything except alphanumeric characters and spaces
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_review(text: str) -> str:
    """
    Apply the full NLP preprocessing pipeline to a single review.
    """
    # 1. Lowercasing
    text = text.lower()

    # 2. HTML tag removal
    text = remove_html_tags(text)

    # 3. Punctuation and special character removal (preserving ratings)
    text = remove_special_characters(text)

    # 4. Tokenization
    tokens = word_tokenize(text)

    # 5. Stopword removal (excluding negation words)
    tokens = [token for token in tokens if token not in STOP_WORDS]

    # 6. Lemmatization
    tokens = [LEMMATIZER.lemmatize(token) for token in tokens]

    # 7. Rejoin tokens into a cleaned string
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("NLP Preprocessing Pipeline — IMDB Sentiment Dataset")
    print("=" * 60)

    # --- Load dataset ---
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run 01_build_dataset.py first.")
        return

    print(f"\nLoading dataset from: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    print(f"  Shape: {df.shape}")

    # --- A. Handle missing data ---
    df = df.dropna(subset=["review"])
    df["review"] = df["review"].fillna("")

    # --- B. Remove duplicates ---
    print("\n--- Step 1: Cleaning Data ---")
    dupes = df.duplicated(subset=["review"]).sum()
    print(f"  Duplicate reviews found: {dupes}")
    df = df.drop_duplicates(subset=["review"]).reset_index(drop=True)
    print(f"  Final rows after cleaning: {len(df)}")

    # --- C. Apply NLP preprocessing ---
    print("\n--- Step 2: Applying NLP Preprocessing ---")
    print("  Techniques: HTML removal, lowercasing, punctuation removal (preserving ratings),")
    print("              tokenization, selective stopword removal, lemmatization")
    
    start_time = time.time()
    df["review_clean"] = df["review"].apply(preprocess_review)
    elapsed = time.time() - start_time
    print(f"  Processing complete in {elapsed:.1f} seconds.")

    # --- D. Save full preprocessed dataset ---
    print("\n--- Step 3: Saving Preprocessed Data ---")
    output_df = df[["review", "review_clean", "sentiment"]]
    output_df.to_csv(OUTPUT_FILE, index=False)
    print(f"  Preprocessed dataset (49k+ items) saved to: {OUTPUT_FILE}")

    # --- E. Create Vectorized Sample (TF-IDF Text Embedding) ---
    print("\n--- Step 4: Generating TF-IDF Vectorized Sample (500 items) ---")
    print("  Ensuring readiness for machine learning tasks...")
    sample_df = output_df.head(500).copy()
    vectorizer = TfidfVectorizer(max_features=100) # Top 100 features for manageable CSV
    tfidf_matrix = vectorizer.fit_transform(sample_df["review_clean"])

    # Convert to DataFrame
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    tfidf_df["sentiment_label"] = sample_df["sentiment"].values

    tfidf_df.to_csv(SAMPLE_OUTPUT_FILE, index=False)
    print(f"  Vectorized sample (500 items) saved to: {SAMPLE_OUTPUT_FILE}")
    print(f"  Shape: {tfidf_df.shape} (500 items, 100 word-features + 1 sentiment-label)")

    print("\n[SUCCESS] Preprocessing completed. All datasets ready for submission.")
    print("=" * 60)


if __name__ == "__main__":
    main()
