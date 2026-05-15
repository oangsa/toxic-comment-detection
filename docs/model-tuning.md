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

The current runtime context uses this 7-column engineered subset:

- `Question Mark Count`
- `Profanity Count`
- `Repeated Punctuation Count`
- `Short/Unclear Without Toxic Signal Flag`
- `Second-person Pronoun Count`
- `URL Count`
- `Non-toxic Negation Pattern Count`

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

Single-feature additions over the text-only baseline:

| Feature | F1 | Precision | Recall | ROC-AUC | Delta vs text only |
|---|---:|---:|---:|---:|---:|
| Question Mark Count | 0.759134 | 0.778896 | 0.740350 | 0.962465 | 0.001599 |
| Profanity Count | 0.759085 | 0.774459 | 0.744309 | 0.962151 | 0.001550 |
| Repeated Punctuation Count | 0.758924 | 0.778819 | 0.740020 | 0.962375 | 0.001389 |
| Short/Unclear Without Toxic Signal Flag | 0.758796 | 0.778549 | 0.740020 | 0.962051 | 0.001261 |
| Second-person Pronoun Count | 0.758725 | 0.775862 | 0.742329 | 0.963205 | 0.001191 |
| URL Count | 0.758130 | 0.778977 | 0.738370 | 0.962039 | 0.000595 |
| Non-toxic Negation Pattern Count | 0.758084 | 0.778512 | 0.738700 | 0.962072 | 0.000549 |
| Uppercase Ratio | 0.758008 | 0.770805 | 0.745629 | 0.963933 | 0.000473 |
| Average Word Length | 0.757617 | 0.777894 | 0.738370 | 0.962095 | 0.000082 |
| Strong Toxic Signal Flag | 0.757586 | 0.765757 | 0.749588 | 0.961985 | 0.000051 |
| Repeated Character Pattern Count | 0.757571 | 0.777431 | 0.738700 | 0.962477 | 0.000036 |
| Exclamation Count | 0.757525 | 0.776968 | 0.739030 | 0.962462 | -0.000010 |
| Character Count | 0.757453 | 0.778281 | 0.737710 | 0.962463 | -0.000082 |
| Identity-group Term Count | 0.757406 | 0.777816 | 0.738040 | 0.962027 | -0.000128 |
| Word Count | 0.757370 | 0.778474 | 0.737380 | 0.962389 | -0.000164 |
| Negation Count | 0.757068 | 0.777469 | 0.737710 | 0.962223 | -0.000467 |

## Interpretation

The tuning story now has two useful takeaways:

- if the target is pure held-out F1 across the saved search methods, the current winner is grid search with all `16` engineered features
- if the target is a more compact engineered subset, the Optuna feature-selection notebook gives the strongest `7`-feature result and the best ROC-AUC among the saved families

So the engineered features still matter, but the “best” answer depends on whether you optimize for top F1 or a smaller selective feature set.

## Optuna Drop-One Ablation

The current `drop-one` table from `src/optuna_feature_test.ipynb` for the full `16`-feature model is:

| Removed feature | F1 without feature | Delta vs all features |
|---|---:|---:|
| Second-person Pronoun Count | 0.757581 | -0.004024 |
| Repeated Character Pattern Count | 0.760027 | -0.001578 |
| Question Mark Count | 0.760927 | -0.000678 |
| URL Count | 0.760974 | -0.000631 |
| Average Word Length | 0.761352 | -0.000252 |
| Short/Unclear Without Toxic Signal Flag | 0.761479 | -0.000126 |
| Negation Count | 0.761605 | 0.000000 |
| Exclamation Count | 0.761652 | 0.000047 |
| Repeated Punctuation Count | 0.761731 | 0.000126 |
| Character Count | 0.761731 | 0.000126 |
| Non-toxic Negation Pattern Count | 0.761810 | 0.000205 |
| Uppercase Ratio | 0.761825 | 0.000221 |
| Identity-group Term Count | 0.762015 | 0.000410 |
| Word Count | 0.762063 | 0.000458 |
| Profanity Count | 0.762094 | 0.000489 |
| Strong Toxic Signal Flag | 0.762811 | 0.001206 |

This shows that:

- `Second-person Pronoun Count` is the most important feature in that specific full-feature ablation
- some other engineered features are neutral or mildly harmful once the rest of the feature stack is already present

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
- the current 7-column runtime feature list shown above
- the current short-input and positive-profanity safeguards

After retraining, rerun:

1. the text-only baseline
2. the grid-search style word-plus-all-engineered baseline
3. the current reduced subset
4. an updated manual edge-case evaluation sheet

That would give the cleanest apples-to-apples comparison between the current saved winner, the compact subset option, and the current safer runtime logic.
