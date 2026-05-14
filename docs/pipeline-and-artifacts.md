# Pipeline and Artifacts

## Purpose

The runtime pipeline turns a raw user comment into a toxicity prediction using a mix of learned text features and hand-built engineered signals.

The core implementation lives in `src/toxic_pipeline.py`.

## End-to-End Prediction Flow

At inference time, the pipeline does the following:

1. Accept a raw input comment.
2. Apply lightweight text protection rules before vectorization.
3. Clean and normalize the text.
4. Build engineered numeric features from the original comment.
5. Transform text into word-level TF-IDF features.
6. Scale the engineered features.
7. Concatenate all feature blocks into one sparse matrix.
8. Run a Logistic Regression classifier and return:
   - a binary label
   - a toxic probability score

## Text Processing

The pipeline currently uses:

- lowercase normalization
- optional contraction expansion through `contractions`
- URL masking in cleaning
- non-alphabetic character stripping in cleaned text
- whitespace normalization

There are also two context-sensitive protection layers:

- non-toxic negation protection
  - examples: `do not like`, `not stupid`, `not trying to insult`
- affectionate profanity protection
  - examples: `I fucking love you`, `this is fucking amazing`

These protections reduce false positives in contexts where a toxic-looking word does not actually behave like an attack.

## Vectorization

### Word TF-IDF

The word vectorizer uses:

- `ngram_range=(1, 2)`
- `max_features=10000`
- `min_df=2`
- `max_df=0.9`
- `sublinear_tf=True`
- English stop words except `not`, `no`, and `never`

The current code intentionally removes the older `char_wb` character TF-IDF branch. That branch helped with spelling variation, but it also pushed some very short neutral inputs such as `you` toward clearly toxic predictions, so the active pipeline is now word TF-IDF plus engineered features only.

## Engineered Features

The code maintains a larger engineered feature pool and a smaller selected subset.

Current full feature pool:

- `Character Count`
- `Word Count`
- `Exclamation Count`
- `Profanity Count`
- `Strong Toxic Signal Flag`
- `Second-person Pronoun Count`
- `Repeated Character Pattern Count`
- `Average Word Length`
- `Uppercase Ratio`
- `Question Mark Count`
- `Repeated Punctuation Count`
- `Identity-group Term Count`
- `URL Count`
- `Negation Count`
- `Non-toxic Negation Pattern Count`
- `Short/Unclear Without Toxic Signal Flag`

Current code default selected subset:

- `Character Count`
- `Profanity Count`
- `Repeated Character Pattern Count`
- `Identity-group Term Count`
- `URL Count`
- `Negation Count`
- `Non-toxic Negation Pattern Count`

## Runtime Safeguards Added Recently

The code now includes two practical inference-time safeguards:

### Short neutral override

Very short neutral inputs such as `I`, `you`, or `you are` can trigger unstable model behavior because there is very little semantic context. The runtime now checks for short non-profane inputs and returns a low-risk non-toxic response instead of trusting the model blindly.

### Affectionate profanity protection

Positive intensifier phrases like `I fucking love you` used to be strongly overclassified as toxic because the model learned that `fuck` is usually toxic. The new protection layer replaces a narrow set of clearly positive profanity phrases before vectorization so direct insults still remain toxic while affectionate use is softened.

## Saved Artifacts

The standard artifact set is stored under `src/`:

- `best_model_final.pkl`
- `best_model_word_vectorizer.pkl`
- `best_model_scaler.pkl`
- `best_model_metadata.pkl`

There is also an Optuna-based fallback set:

- `optuna_feature_test_best_model.pkl`
- `optuna_feature_test_word_vectorizer.pkl`
- `optuna_feature_test_scaler.pkl`
- `optuna_feature_test_metadata.pkl`

The loader resolves the preferred files first and falls back when needed.

## Artifact Compatibility Note

This repository currently has an important but manageable mismatch:

- the latest code defaults select a 7-feature engineered subset
- the latest code has removed the character TF-IDF branch
- the saved historical artifacts may still expect the older hybrid feature layout

The loader still contains compatibility logic to reduce scaler dimensions when needed, but it now raises a clear error if a saved model still expects char features. Retraining is required so the saved artifacts and current source code reflect the same feature contract.

## CLI Entry Point

`src/main.py` provides a small interface for local use:

- interactive mode:
  - `python src/main.py`
- single comment mode:
  - `python src/main.py "sample comment"`

The CLI loads artifacts once in interactive mode and prints:

- `Prediction`
- `Toxic score`
