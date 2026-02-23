"""
Step 2: NLP Preprocessing Pipeline for Sentiment Analysis.

This script applies the following NLP preprocessing techniques to the
IMDB movie review dataset:

    1. Handling missing data
    2. Lowercasing
    3. HTML tag and special character removal
    4. Punctuation removal
    5. Tokenization
    6. Stopword removal
    7. Lemmatization

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

# ---------------------------------------------------------------------------
# Initialise NLP tools
# ---------------------------------------------------------------------------
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


# ---------------------------------------------------------------------------
# Preprocessing functions
# ---------------------------------------------------------------------------
def remove_html_tags(text: str) -> str:
    """Remove HTML tags such as <br />, <p>, etc."""
    return re.sub(r"<[^>]+>", " ", text)


def remove_special_characters(text: str) -> str:
    """Remove non-alphabetic characters (keeps spaces)."""
    return re.sub(r"[^a-zA-Z\s]", "", text)


def preprocess_review(text: str) -> str:
    """
    Apply the full NLP preprocessing pipeline to a single review.

    Steps applied in order:
        1. Lowercasing
        2. HTML tag removal
        3. Special character and punctuation removal
        4. Tokenization
        5. Stopword removal
        6. Lemmatization
        7. Rejoin tokens into a cleaned string
    """
    # 1. Lowercasing
    text = text.lower()

    # 2. HTML tag removal (IMDB reviews contain <br /> tags)
    text = remove_html_tags(text)

    # 3. Punctuation and special character removal
    text = remove_special_characters(text)

    # 4. Tokenization
    tokens = word_tokenize(text)

    # 5. Stopword removal
    tokens = [token for token in tokens if token not in STOP_WORDS]

    # 6. Lemmatization
    tokens = [LEMMATIZER.lemmatize(token) for token in tokens]

    # 7. Rejoin
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("NLP Preprocessing Pipeline — IMDB Sentiment Dataset")
    print("=" * 60)

    # --- Load dataset ---
    print(f"\nLoading dataset from: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    # --- A. Handle missing data ---
    print("\n--- Step 1: Handling Missing Data ---")
    missing_before = df.isnull().sum().sum()
    print(f"  Missing values before: {missing_before}")

    # Drop rows with missing review text
    df = df.dropna(subset=["review"])

    # Fill any remaining NaN (shouldn't exist, but safety measure)
    df["review"] = df["review"].fillna("")

    missing_after = df.isnull().sum().sum()
    print(f"  Missing values after:  {missing_after}")
    print(f"  Rows after cleaning:   {len(df)}")

    # --- B. Remove duplicates ---
    print("\n--- Step 2: Removing Duplicate Reviews ---")
    dupes = df.duplicated(subset=["review"]).sum()
    print(f"  Duplicate reviews found: {dupes}")
    df = df.drop_duplicates(subset=["review"]).reset_index(drop=True)
    print(f"  Rows after dedup:        {len(df)}")

    # --- C. Apply NLP preprocessing ---
    print("\n--- Step 3: Applying NLP Preprocessing ---")
    print("  Techniques: lowercasing, HTML removal, punctuation removal,")
    print("              tokenization, stopword removal, lemmatization")
    print("  Processing reviews (this may take a few minutes)...")

    start_time = time.time()
    df["review_clean"] = df["review"].apply(preprocess_review)
    elapsed = time.time() - start_time
    print(f"  Completed in {elapsed:.1f} seconds.")

    # --- D. Show samples ---
    print("\n--- Sample: Original vs. Preprocessed ---")
    for i in range(3):
        print(f"\n  [Review {i + 1}]")
        original = df.iloc[i]["review"][:200]
        cleaned = df.iloc[i]["review_clean"][:200]
        print(f"  Original:     {original}...")
        print(f"  Preprocessed: {cleaned}...")

    # --- E. Summary statistics ---
    print("\n--- Preprocessing Summary ---")
    df["original_word_count"] = df["review"].apply(lambda x: len(x.split()))
    df["clean_word_count"] = df["review_clean"].apply(lambda x: len(x.split()))

    print(f"  Average original word count:     {df['original_word_count'].mean():.1f}")
    print(f"  Average preprocessed word count: {df['clean_word_count'].mean():.1f}")
    reduction = (1 - df["clean_word_count"].mean() / df["original_word_count"].mean()) * 100
    print(f"  Average word count reduction:    {reduction:.1f}%")
    print(f"  Sentiment distribution:")
    print(f"    {df['sentiment'].value_counts().to_dict()}")

    # --- F. Save preprocessed dataset ---
    output_df = df[["review", "review_clean", "sentiment"]]
    output_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n  Preprocessed dataset saved to: {OUTPUT_FILE}")
    print(f"  Total records: {len(output_df)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
