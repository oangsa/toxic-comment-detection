# Model Tuning

## Tuning Goal

The project compared text-only and hybrid text-plus-feature approaches, then tuned the classifier to maximize classification quality while preserving a practical local inference workflow.

The main optimization target across the documented experiments was F1 score, with ROC-AUC, precision, and recall used as secondary checks.

## Dataset Preparation

The notebook history shows two nearby cleaned dataset counts:

- `158,237` rows in `data_cleaning_final.ipynb`
- `158,194` rows used in the later training notebooks

The later training notebooks report:

- train size: `126,555`
- test size: `31,639`
- toxic ratio in train: `0.096`

This indicates the project followed a cleaned-data workflow and then ran feature engineering and model selection on the cleaned set.

## Experiment Path

The repository shows a progression of experiments:

- baseline cleaning and EDA notebooks
- handcrafted feature creation notebooks
- `random_search_tuning.ipynb`
- `GridSearch.ipynb`
- `main_Optuna_automated.ipynb`
- `optuna_feature_test.ipynb`

This suggests the project evolved in stages:

1. establish a text baseline
2. add engineered features
3. tune linear models
4. refine the engineered feature subset
5. save final reusable artifacts

## Models Considered

The notebooks reference at least:

- Logistic Regression
- SGDClassifier

The final exported model family used in the current runtime is Logistic Regression.

## Current Saved-Artifact Comparison

Re-evaluating the saved artifact families on the shared word-only pipeline and the same `random_state=42` 20% stratified split gives:

- `Grid Search`
  - feature count: `16`
  - F1: `0.7682`
  - precision: `0.7755`
  - recall: `0.7611`
  - ROC-AUC: `0.9610`
- `Optuna Main`
  - feature count: `16`
  - F1: `0.7660`
  - precision: `0.7600`
  - recall: `0.7720`
  - ROC-AUC: `0.9642`
- `Random Search`
  - feature count: `16`
  - F1: `0.7622`
  - precision: `0.7690`
  - recall: `0.7555`
  - ROC-AUC: `0.9619`
- `Optuna Feature Test`
  - feature count: `7`
  - F1: `0.7600`
  - precision: `0.7686`
  - recall: `0.7516`
  - ROC-AUC: `0.9652`
- `Runtime Best Model`
  - feature count: `7`
  - F1: `0.7583`
  - precision: `0.7746`
  - recall: `0.7427`
  - ROC-AUC: `0.9627`

By the project’s primary tuning metric, `Grid Search` is the current saved winner.

## Winner Profile

The grid-search winner is a Logistic Regression family model trained on:

- word TF-IDF
- all `16` engineered features
- no character TF-IDF branch

It is not the smallest artifact family, but it currently gives the best F1 among the saved notebook outputs.

## Optuna Feature-Subset Results

The saved Optuna feature-selection metadata reports:

- best CV F1: `0.7682582718497781`
- best LR params:
  - `penalty='l2'`
  - `C=1.0567675307931639`
  - `tol=0.00023605043231416352`
  - `class_weight={0: 1, 1: 3}`

Feature-set comparison:

- `Text only`
  - feature count: `0`
  - F1: `0.757535`
  - precision: `0.778087`
  - recall: `0.738040`
  - ROC-AUC: `0.962072`
- `Text + all engineered`
  - feature count: `16`
  - F1: `0.761605`
  - precision: `0.765412`
  - recall: `0.757836`
  - ROC-AUC: `0.965151`
- `Text + Optuna subset`
  - feature count: `7`
  - F1: `0.759967`
  - precision: `0.768556`
  - recall: `0.751567`
  - ROC-AUC: `0.965192`

## Interpretation

The tuning story now has two useful takeaways:

- if the target is pure held-out F1 across the saved search methods, the current winner is grid search with all `16` engineered features
- if the target is a more compact engineered subset, the Optuna feature-selection notebook gives the strongest `7`-feature result and the best ROC-AUC among the saved families

So the engineered features still matter, but the “best” answer depends on whether you optimize for top F1 or a smaller selective feature set.

## Why Recent Inference Changes Were Still Needed

Even with strong test-set metrics, several practical edge cases appeared during manual probing:

- very short neutral comments like `I` or `You`
- affectionate profanity such as `I fucking love you`

These are examples of a broader lesson:

- a good overall F1 score does not guarantee intuitive behavior on short, context-light inputs

One especially important failure mode came from the old `char_wb` branch. A one-word input like `you` could be scored as highly toxic by the saved hybrid model, while the same pipeline without char features behaved much more reasonably. Because of that, the active runtime pipeline has now dropped character TF-IDF entirely.

## Recommended Next Tuning Step

The most useful next experiment would be to retrain the final artifacts with the latest code defaults so the model reflects:

- word TF-IDF plus engineered features only
- the updated selected-feature list
- the current short-input and positive-profanity safeguards

After retraining, rerun:

1. the text-only baseline
2. the grid-search style word-plus-all-engineered baseline
3. the current reduced subset
4. an updated manual edge-case evaluation sheet

That would give the cleanest apples-to-apples comparison between the current saved winner, the compact subset option, and the current safer runtime logic.
