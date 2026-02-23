# D803 — Sentiment Data Analysis (NLP Preprocessing)

## Overview

This project implements a Natural Language Processing (NLP) preprocessing pipeline for sentiment analysis using the **IMDB Dataset of 50K Movie Reviews**. The pipeline applies industry-standard text preprocessing techniques to prepare raw movie reviews for downstream sentiment classification tasks.

The dataset contains 50,000 movie reviews labeled as **positive** or **negative**, sourced from the Internet Movie Database (IMDB).

## Dataset

- **Name:** IMDB Dataset of 50K Movie Reviews
- **Source:** [Kaggle — IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Original Source:** [Stanford AI Lab — Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- **Size:** 50,000 reviews (25,000 positive, 25,000 negative)

## NLP Preprocessing Techniques Applied

| Technique | Description |
|---|---|
| Handling Missing Data | Detection and removal of null or empty reviews |
| Duplicate Removal | Identification and removal of duplicate review entries |
| Lowercasing | Conversion of all text to lowercase for uniformity |
| HTML Tag Removal | Stripping of HTML markup (e.g., `<br />`) from reviews |
| Punctuation & Special Character Removal | Removal of non-alphabetic characters |
| Tokenization | Splitting text into individual word tokens using NLTK |
| Stopword Removal | Filtering out common English stopwords (e.g., "the", "is", "and") |
| Lemmatization | Reducing words to their base/dictionary form (e.g., "running" → "run") |

## Prerequisites

- **Python 3.10+**
- **pip** (Python package manager)
- Internet connection (required for downloading the dataset and NLTK resources on first run)

## Setup and Execution

### 1. Clone the Repository

```bash
git clone https://github.com/Farzin312/D803_Sentiment_Data_Analysis.git
cd D803_Sentiment_Data_Analysis
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
```

### 3. Activate the Virtual Environment

**macOS / Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Download and Build the Dataset

This script downloads the IMDB dataset from Stanford AI Lab and converts it into a CSV file:

```bash
python 01_build_dataset.py
```

**Output:** `IMDB_Dataset.csv` — 50,000 reviews with `review` and `sentiment` columns.

### 6. Run the Preprocessing Pipeline

This script applies all NLP preprocessing techniques:

```bash
python 02_preprocess.py
```

**Output:** `IMDB_Preprocessed.csv` — Contains original reviews, preprocessed reviews, and sentiment labels.

## Project Structure

```
D803_Sentiment_Data_Analysis/
├── 01_build_dataset.py       # Downloads and builds the raw CSV dataset
├── 02_preprocess.py          # NLP preprocessing pipeline
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── .gitignore                # Git ignore rules
```

## Output Files (Generated at Runtime)

| File | Description |
|---|---|
| `IMDB_Dataset.csv` | Raw dataset with 50,000 movie reviews |
| `IMDB_Preprocessed.csv` | Preprocessed dataset ready for NLP tasks |

## References

Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). Learning word vectors for sentiment analysis. In *Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies* (pp. 142–150). Association for Computational Linguistics. https://aclanthology.org/P11-1015
