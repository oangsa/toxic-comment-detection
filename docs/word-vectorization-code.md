# Word Vectorization Code

This page shows the code path behind word vectorization in this project.

It is based on the current implementation in `src/toxic_pipeline.py`.

## 1. Training-Time Idea

At training time, the vectorizer learns a vocabulary from the cleaned training comments and converts them into TF-IDF numbers.

A simplified version looks like this:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

WORD_TFIDF_CONFIG = {
    "ngram_range": (1, 2),
    "max_features": 10000,
    "min_df": 2,
    "max_df": 0.9,
    "sublinear_tf": True,
    "stop_words": custom_stop_words,
}

vectorizer = TfidfVectorizer(**WORD_TFIDF_CONFIG)
X_word_train = vectorizer.fit_transform(cleaned_train_texts)
```

What happens here:

- `fit` learns which words and bigrams become vocabulary features
- `transform` converts each cleaned comment into a sparse TF-IDF vector
- `fit_transform` does both in one step

## 2. Actual Project Helper

In this project, the word vectorizer is created by:

```python
def build_word_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(**WORD_TFIDF_CONFIG)
```

So the real project uses the same idea, just wrapped in a helper function.

## 3. Cleaning Before Vectorization

Before text becomes numbers, the project cleans it first.

This is the current cleaning logic:

```python
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = expand_contractions(text)
    text = text.lower()
    text = re.sub(r"http\\S+|www\\S+", " URL ", text)
    text = re.sub(r"[^a-z]", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text
```

That means:

- uppercase becomes lowercase
- URLs are replaced before cleaning
- punctuation and non-letters are removed
- extra spaces are collapsed

Example:

```python
raw_text = "Why are you not stupid?"
cleaned_text = clean_text(raw_text)
print(cleaned_text)
# why are you not stupid
```

## 4. Protection Layer Before Cleaning

The project also protects a few special phrases before the normal cleaning step:

```python
def protect_non_toxic_negations(text: str) -> str:
    if not isinstance(text, str):
        return ""

    protected_text = expand_contractions(text)
    for pattern in AFFECTIONATE_PROFANITY_PROTECTION_PATTERNS:
        protected_text = pattern.sub(AFFECTIONATE_PROFANITY_PLACEHOLDER, protected_text)
    for pattern in NON_TOXIC_NEGATION_PROTECTION_PATTERNS:
        protected_text = pattern.sub(NON_TOXIC_NEGATION_PLACEHOLDER, protected_text)
    return re.sub(r"\\s+", " ", protected_text).strip()
```

This happens before TF-IDF so that some misleading toxic-looking phrases are softened before tokenization.

## 5. Inference-Time Vectorization

This is the actual project function that turns one comment into model input:

```python
def vectorize_comment(comment: str, artifacts: dict[str, Any]) -> Any:
    artifacts = _normalize_prediction_artifacts(artifacts)
    raw_text = str(comment)
    protected_text = protect_non_toxic_negations(raw_text)
    clean = clean_text(protected_text)
    engineered = build_all_engineered_features(raw_text)[artifacts["feature_columns"]]
    scaler = artifacts["scaler"]

    word_vectorizer = artifacts["word_vectorizer"]

    x_word = word_vectorizer.transform([clean])
    x_eng = csr_matrix(scaler.transform(engineered.values))
    return hstack([x_word, x_eng], format="csr")
```

This is the most important “under the hood” code in the current project.

## 6. What Each Line Is Doing

Here is the same logic with comments:

```python
from scipy.sparse import csr_matrix, hstack

def vectorize_comment_explained(comment, artifacts):
    raw_text = str(comment)

    # Protect special non-toxic patterns before normal cleaning.
    protected_text = protect_non_toxic_negations(raw_text)

    # Normalize text into the form used by the trained vectorizer.
    clean = clean_text(protected_text)

    # Build numeric handcrafted features from the original raw text.
    engineered = build_all_engineered_features(raw_text)[artifacts["feature_columns"]]

    # Turn cleaned text into sparse TF-IDF numbers.
    x_word = artifacts["word_vectorizer"].transform([clean])

    # Scale the engineered numeric features.
    x_eng = csr_matrix(artifacts["scaler"].transform(engineered.values))

    # Join text features and engineered features into one model input row.
    x_all = hstack([x_word, x_eng], format="csr")
    return x_all
```

## 7. Minimal Demo You Can Run

If you want to show exactly how one comment becomes numbers, this small script matches the real saved artifacts:

```python
from src.toxic_pipeline import (
    clean_text,
    protect_non_toxic_negations,
    build_all_engineered_features,
    load_best_artifacts,
)
from scipy.sparse import csr_matrix, hstack

comment = "Why are you not stupid?"
artifacts = load_best_artifacts()

raw_text = str(comment)
protected_text = protect_non_toxic_negations(raw_text)
clean = clean_text(protected_text)

engineered = build_all_engineered_features(raw_text)[artifacts["feature_columns"]]
x_word = artifacts["word_vectorizer"].transform([clean])
x_eng = csr_matrix(artifacts["scaler"].transform(engineered.values))
x_all = hstack([x_word, x_eng], format="csr")

print("Raw text:", raw_text)
print("Protected text:", protected_text)
print("Cleaned text:", clean)
print("Word vector shape:", x_word.shape)
print("Engineered feature shape:", x_eng.shape)
print("Combined shape:", x_all.shape)
print("Word vector non-zero count:", x_word.nnz)
print("Combined non-zero count:", x_all.nnz)
```

## 8. Showing Which Words Became Numbers

If you want to show which vocabulary terms were activated:

```python
from src.toxic_pipeline import clean_text, load_best_artifacts

comment = "Why are you not stupid?"
artifacts = load_best_artifacts()
vectorizer = artifacts["word_vectorizer"]

clean = clean_text(comment)
X = vectorizer.transform([clean])

index_to_term = {index: term for term, index in vectorizer.vocabulary_.items()}

for index, value in zip(X.indices.tolist(), X.data.tolist()):
    print(index_to_term[index], round(value, 6))
```

Expected kind of output for the current saved vectorizer:

```text
not 0.301558
stupid 0.953448
```

## 9. Full Flow In One Block

If you want one short code block for the report, this is the best compact version:

```python
raw_text = "Why are you not stupid?"
protected_text = protect_non_toxic_negations(raw_text)
clean = clean_text(protected_text)

x_word = word_vectorizer.transform([clean])
x_eng = csr_matrix(scaler.transform(build_all_engineered_features(raw_text)[feature_columns].values))
x_all = hstack([x_word, x_eng], format="csr")

prediction = model.predict(x_all)[0]
probability = model.predict_proba(x_all)[0][1]
```

This is the real idea under the hood:

1. protect
2. clean
3. vectorize text into TF-IDF numbers
4. build engineered numeric features
5. join everything into one matrix row
6. send that row into Logistic Regression
