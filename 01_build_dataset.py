"""
Step 1: Download and convert raw IMDB movie review files into a single CSV dataset.

This script downloads the IMDB Large Movie Review Dataset (Maas et al., 2011)
from Stanford AI Lab and consolidates 50,000 labeled reviews into a structured
CSV file with 'review' and 'sentiment' columns.

Dataset source: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
Original source: https://ai.stanford.edu/~amaas/data/sentiment/
"""

import os
import tarfile
import urllib.request

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "aclImdb")
ARCHIVE_FILE = os.path.join(SCRIPT_DIR, "aclImdb_v1.tar.gz")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "IMDB_Dataset.csv")

DOWNLOAD_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"


def download_and_extract():
    """Download the IMDB dataset archive and extract it."""
    if os.path.isdir(DATA_DIR):
        print("  Dataset directory already exists, skipping download.")
        return

    print(f"  Downloading from {DOWNLOAD_URL} ...")
    urllib.request.urlretrieve(DOWNLOAD_URL, ARCHIVE_FILE)
    print(f"  Download complete ({os.path.getsize(ARCHIVE_FILE) / 1e6:.1f} MB).")

    print("  Extracting archive...")
    with tarfile.open(ARCHIVE_FILE, "r:gz") as tar:
        tar.extractall(path=SCRIPT_DIR)
    print("  Extraction complete.")

    # Clean up the archive to save disk space
    os.remove(ARCHIVE_FILE)
    print("  Removed archive file.")


def load_reviews(split: str) -> list[dict]:
    """Load reviews from a given split (train or test)."""
    reviews = []
    split_dir = os.path.join(DATA_DIR, split)
    for label in ("pos", "neg"):
        label_dir = os.path.join(split_dir, label)
        sentiment = "positive" if label == "pos" else "negative"
        for filename in sorted(os.listdir(label_dir)):
            if not filename.endswith(".txt"):
                continue
            filepath = os.path.join(label_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read().strip()
            reviews.append({"review": text, "sentiment": sentiment})
    return reviews


def main():
    print("=" * 60)
    print("Step 1: Build IMDB Dataset CSV")
    print("=" * 60)

    print("\n--- Downloading Dataset ---")
    download_and_extract()

    print("\nLoading training reviews...")
    train_reviews = load_reviews("train")
    print(f"  Loaded {len(train_reviews)} training reviews.")

    print("Loading test reviews...")
    test_reviews = load_reviews("test")
    print(f"  Loaded {len(test_reviews)} test reviews.")

    all_reviews = train_reviews + test_reviews
    df = pd.DataFrame(all_reviews)

    # Shuffle the dataset for randomness
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\nTotal reviews: {len(df)}")
    print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nDataset saved to: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
