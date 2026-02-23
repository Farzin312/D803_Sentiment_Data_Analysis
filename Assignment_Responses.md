# QLN1 Task 1: Sentiment Data Analysis — Assignment Responses

---

## Part A: Data Source and Dataset

The dataset selected for this sentiment analysis project is the **IMDB Dataset of 50K Movie Reviews**, obtained from Kaggle (Maas et al., 2011). This dataset was originally compiled by the Stanford AI Lab and contains 50,000 movie reviews from the Internet Movie Database (IMDB), evenly split between 25,000 positive and 25,000 negative reviews. Each review consists of free-form English text written by moviegoers, paired with a binary sentiment label (positive or negative) derived from the reviewer's numerical rating.

**Dataset link:** https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

### Part A1: Dataset Description

The collected dataset contains **50,000 entries**, which far exceeds the minimum requirement of 500 items. Each entry includes two fields:

| Field | Description |
|---|---|
| `review` | The full text of the movie review (free-form English) |
| `sentiment` | Binary label: "positive" or "negative" |

**Sentiment distribution:**
- Positive reviews: 25,000
- Negative reviews: 25,000

The raw dataset file (`IMDB_Dataset.csv`) is included with this submission.

---

## Part B: Selected NLP Techniques

The following NLP preprocessing techniques were selected and applied to the dataset in sequential order:

1. **Handling Missing Data** — Detection and removal of any null or empty review entries to ensure data completeness.
2. **Duplicate Removal** — Identification and elimination of 418 duplicate reviews to prevent redundancy and potential bias in downstream analysis.
3. **Lowercasing** — Conversion of all text to lowercase to ensure uniform representation of words (e.g., "Movie" and "movie" are treated identically).
4. **HTML Tag Removal** — Stripping of embedded HTML markup (e.g., `<br />` tags) that is present in the raw IMDB review data.
5. **Punctuation and Special Character Removal** — Removal of all non-alphabetic characters, including punctuation marks, numbers, and special symbols, retaining only alphabetic characters and whitespace.
6. **Tokenization** — Splitting each review into individual word tokens using the NLTK `word_tokenize` function, which applies the Punkt tokenizer for accurate sentence and word boundary detection.
7. **Stopword Removal** — Filtering out common English stopwords (e.g., "the," "is," "and," "a") using the NLTK English stopword list, reducing noise and focusing on content-bearing words.
8. **Lemmatization** — Reducing each token to its base dictionary form using the NLTK WordNet Lemmatizer (e.g., "running" becomes "run," "better" becomes "better," "movies" becomes "movie").

These techniques were implemented in Python using the Natural Language Toolkit (NLTK) library. The preprocessing pipeline is contained in the script `02_preprocess.py`.

---

## Part C: Preprepared Data

The preprocessed dataset is saved as `IMDB_Preprocessed.csv` and contains the following three columns:

| Column | Description |
|---|---|
| `review` | The original, unmodified review text |
| `review_clean` | The fully preprocessed review text |
| `sentiment` | The sentiment label (positive or negative) |

**Preprocessing results summary:**
- Original records: **50,000**
- Duplicates removed: **418**
- Total preprocessed records: **49,582**
- Average original word count per review: **231.4 words**
- Average preprocessed word count per review: **119.1 words**
- Average word count reduction: **48.5%**

**Sample comparison (Original vs. Preprocessed):**

| | Original (excerpt) | Preprocessed (excerpt) |
|---|---|---|
| Review 1 | "When I first saw the ad for this, I was like 'Oh here we go. He's done High School Musical, but he can't coast along on that..." | "first saw ad like oh go he done high school musical cant coast along he making appearance disney show..." |
| Review 2 | "A Girl's Folly is a sort of half-comedy, half-mockumentary look at the motion picture business of the mid-1910's..." | "girl folly sort halfcomedy halfmockumentary look motion picture business mids get glimpse life early movie studio..." |

The preprocessed dataset file (`IMDB_Preprocessed.csv`) is included with this submission and is ready for use in downstream NLP sentiment classification tasks.

---

## Part D: Strengths and Limitations of Selected NLP Techniques

The NLP preprocessing techniques applied in this project each contribute distinct strengths to the preparation of textual data for sentiment analysis. Lowercasing ensures lexical consistency, preventing the model from treating capitalized and lowercase variants of the same word as separate features, which reduces vocabulary size and improves generalization. Tokenization, the foundational step in most NLP pipelines, segments continuous text into discrete units that can be individually analyzed and quantified. Siino et al. (2024) demonstrated that the selection and ordering of preprocessing techniques can influence classification accuracy by up to 25%, even when using state-of-the-art transformer models, underscoring that these steps remain consequential despite advances in deep learning. Stopword removal reduces the dimensionality of the feature space by eliminating high-frequency function words that carry little semantic meaning, allowing classifiers to focus on content-bearing terms. Lemmatization further consolidates the vocabulary by reducing inflected forms to their dictionary base, which improves feature consistency without the aggressive truncation associated with stemming (Jim et al., 2024).

Despite these strengths, each technique also introduces limitations that must be carefully considered. Stopword removal, while effective at reducing noise, can inadvertently discard words that carry sentiment-relevant information. For example, negation words such as "not" and "no" are commonly included in standard stopword lists, yet they are critical for determining sentiment polarity; removing them can reverse the intended meaning of a phrase. Shukla and Dwivedi (2024) found that certain preprocessing techniques increased classifier accuracy while others had no significant impact, and that the effectiveness of any given technique depended on the specific classifier being used. Additionally, lemmatization, although more linguistically principled than stemming, depends on accurate part-of-speech tagging to function optimally, and errors in tagging can lead to incorrect base-form reductions. HTML tag removal and special character stripping, while necessary for the IMDB dataset, are domain-specific steps that may not generalize to all text sources without modification.

Furthermore, the cumulative effect of applying multiple preprocessing steps sequentially introduces the risk of over-processing, where meaningful textual features are progressively stripped away. Tan et al. (2023) noted that while preprocessing is essential for improving the signal-to-noise ratio in text data, excessive cleaning can degrade model performance by eliminating contextual cues that contribute to accurate sentiment classification. The preprocessing pipeline must therefore balance thoroughness with preservation of semantic content, recognizing that the optimal combination of techniques varies by dataset and by the downstream model architecture (Siino et al., 2024).

---

## Part E: Rationale Behind Project Choices

The IMDB Dataset of 50K Movie Reviews was selected as the data source for several strategic reasons that align with the objectives of this sentiment analysis project. First, the dataset contains substantial free-form English text—averaging over 230 words per review—which provides sufficient linguistic complexity to meaningfully demonstrate the impact of NLP preprocessing techniques such as tokenization, stopword removal, and lemmatization. Second, the binary sentiment labels (positive and negative) are derived directly from user-assigned star ratings, ensuring reliable ground-truth annotations without the subjectivity inherent in manual labeling. Third, the dataset's balanced class distribution (approximately 50% positive and 50% negative) eliminates class imbalance as a confounding variable, allowing the evaluation of preprocessing techniques to focus on their direct effects on text quality rather than on distributional artifacts. Jim et al. (2024) emphasized that dataset selection is a critical determinant of sentiment analysis outcomes, noting that balanced, well-annotated datasets enable more reliable comparisons of preprocessing and classification methods. The IMDB dataset has been extensively validated in the academic literature and remains one of the most widely used benchmarks for sentiment analysis research (Malik & Bilal, 2024).

The selection of preprocessing techniques was driven by the need to systematically address the specific characteristics of the IMDB review data while adhering to established best practices in NLP. The inclusion of HTML tag removal addresses a domain-specific challenge, as the raw IMDB data contains embedded markup that would otherwise introduce noise into the feature space. Lowercasing, punctuation removal, and stopword filtering collectively reduce vocabulary dimensionality, which improves both computational efficiency and model generalization. Lemmatization was chosen over stemming because it produces linguistically valid base forms, which preserves interpretability and is particularly advantageous when the preprocessed text may be used with models that benefit from coherent input, such as word embedding algorithms. Siino et al. (2024) demonstrated that even simple classifiers, when paired with an appropriately designed preprocessing pipeline, can rival or exceed the performance of more complex models with minimal preprocessing, reinforcing the importance of deliberate technique selection.

Together, these decisions contribute to the overall effectiveness of the sentiment analysis pipeline by ensuring that the preprocessed data retains the semantic content necessary for accurate classification while eliminating noise, redundancy, and inconsistency. The 48.5% reduction in average word count achieved through preprocessing reflects a substantial decrease in feature dimensionality, which reduces computational cost and mitigates the risk of overfitting in downstream models. By selecting a well-established dataset, applying a principled sequence of preprocessing steps, and documenting the rationale behind each decision, this project establishes a reproducible and transparent foundation for sentiment analysis that can be extended to accommodate additional techniques or alternative datasets in future work (Shukla & Dwivedi, 2024).

---

## Part F: References

Jim, J. R., Talukder, M. A. R., Malakar, P., Kabir, M. M., Nur, K., & Mridha, M. F. (2024). Recent advancements and challenges of NLP-based sentiment analysis: A state-of-the-art review. *Natural Language Processing Journal*, *6*, Article 100059. https://doi.org/10.1016/j.nlp.2024.100059

Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). Learning word vectors for sentiment analysis. In *Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies* (pp. 142–150). Association for Computational Linguistics. https://aclanthology.org/P11-1015

Malik, N., & Bilal, M. (2024). Natural language processing for analyzing online customer reviews: A survey, taxonomy, and open research challenges. *PeerJ Computer Science*, *10*, Article e2203. https://doi.org/10.7717/peerj-cs.2203

Shukla, D., & Dwivedi, S. K. (2024). The study of the effect of preprocessing techniques for emotion detection on Amazon product review dataset. *Social Network Analysis and Mining*, *14*, Article 191. https://doi.org/10.1007/s13278-024-01352-4

Siino, M., Tinnirello, I., & La Cascia, M. (2024). Is text preprocessing still worth the time? A comparative survey on the influence of popular preprocessing methods on Transformers and traditional classifiers. *Information Systems*, *121*, Article 102342. https://doi.org/10.1016/j.is.2023.102342

Tan, K. L., Lee, C. P., & Lim, K. M. (2023). A survey of sentiment analysis: Approaches, datasets, and future research. *Applied Sciences*, *13*(7), Article 4550. https://doi.org/10.3390/app13074550

---
