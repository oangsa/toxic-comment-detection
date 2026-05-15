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

The repository currently contains multiple saved artifact families under `src/`:

- grid-search winner
  - `grid_search_best_model.pkl`
  - `grid_search_word_vectorizer.pkl`
  - `grid_search_scaler.pkl`
- main Optuna search
  - `optuna_best_model.pkl`
  - `optuna_word_vectorizer.pkl`
  - `optuna_scaler.pkl`
- random search
  - `random_search_best_model.pkl`
  - `random_search_word_vectorizer.pkl`
  - `random_search_scaler.pkl`
- compact runtime/default set
  - `best_model_final.pkl`
  - `best_model_word_vectorizer.pkl`
  - `best_model_scaler.pkl`
  - `best_model_metadata.pkl`
- Optuna feature-subset set
  - `optuna_feature_test_best_model.pkl`
  - `optuna_feature_test_word_vectorizer.pkl`
  - `optuna_feature_test_scaler.pkl`
  - `optuna_feature_test_metadata.pkl`

Among the currently saved experiments, the grid-search artifact family has the strongest held-out F1 on the aligned word-only evaluation split.

## Artifact Compatibility Note

The preferred current models are word-only TF-IDF plus engineered features, and the latest saved comparison was evaluated that way.

There are still some historical `*_char_vectorizer.pkl` files in `src/`, but they should be treated as leftovers from older experiments rather than the active path.

The loader also contains compatibility logic to reduce scaler dimensions when a model uses a smaller engineered-feature subset than the full `16`-feature scaler layout.

The main remaining project decision is not feature-contract breakage but promotion:

- the runtime default artifact family is the compact `best_model_*` set
- the best saved F1 currently belongs to the `grid_search_*` family

Promoting one family as canonical would make the project story simpler.

## CLI Entry Point

`src/main.py` provides a small interface for local use:

- interactive mode:
  - `python src/main.py`
- single comment mode:
  - `python src/main.py "sample comment"`

The CLI loads artifacts once in interactive mode and prints:

- `Prediction`
- `Toxic score`
