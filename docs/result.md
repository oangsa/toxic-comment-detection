# Results

This file collects the current evaluated results for the project in one place.

## Evaluation Setup

Unless noted otherwise, the results below were checked on:

- `data/train.csv`
- cleaned dataset size: `158,194`
- train/test split: `126,555 / 31,639`
- split rule: `test_size=0.20`, `random_state=42`, stratified on `toxic`
- current shared pipeline:
  - word-level TF-IDF only
  - no `char_wb` branch
  - engineered features from `src/toxic_pipeline.py`

## Saved Artifact Comparison

These results compare the currently saved artifact families on the same shared evaluation split.

| Saved artifact family | Engineered features | F1 | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Grid Search | 16 | 0.768232 | 0.775462 | 0.761135 | 0.961021 |
| Optuna Main | 16 | 0.765957 | 0.759987 | 0.772022 | 0.964234 |
| Random Search | 16 | 0.762190 | 0.768972 | 0.755526 | 0.961869 |
| Optuna Feature Test Artifact | 7 | 0.759967 | 0.768556 | 0.751567 | 0.965192 |
| Main Notebook / Runtime Best Model | 7 | 0.758295 | 0.774604 | 0.742659 | 0.962679 |

### Current Winner

By the project’s primary tuning metric, the current saved winner is:

- `Grid Search`
  - F1: `0.768232`

If the priority is ROC-AUC or a smaller engineered-feature set, the strongest compact option is:

- `Optuna Feature Test Artifact`
  - engineered features: `7`
  - ROC-AUC: `0.965192`

## Optuna Feature-Test Notebook Results

These results describe the current comparison inside `src/optuna_feature_test.ipynb`.

Best saved Optuna feature-test settings:

- best CV F1: `0.7682582718497781`
- Logistic Regression params:
  - `penalty='l2'`
  - `solver='liblinear'`
  - `C=1.0567675307931639`
  - `class_weight={0: 1, 1: 3}`
  - `tol=0.00023605043231416352`
- final model fit uses:
  - `max_iter=2000`

Selected engineered subset:

- `Character Count`
- `Profanity Count`
- `Second-person Pronoun Count`
- `Repeated Character Pattern Count`
- `Uppercase Ratio`
- `Identity-group Term Count`
- `URL Count`

Feature-set comparison:

| Setup | Feature count | F1 | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Text only | 0 | 0.757535 | 0.778087 | 0.738040 | 0.962072 |
| Text + all engineered | 16 | 0.761605 | 0.765412 | 0.757836 | 0.965151 |
| Text + Optuna subset | 7 | 0.759967 | 0.768556 | 0.751567 | 0.965192 |

### Optuna Notebook Interpretation

- Best F1 inside the current Optuna feature-test comparison: `Text + all engineered` at `0.761605`
- Best ROC-AUC inside the current Optuna feature-test comparison: `Text + Optuna subset` at `0.965192`

So the Optuna notebook still supports a compact `7`-feature subset story, but in the current word-only evaluation the full `16`-feature version edges it out on F1.

## Main Notebook / Runtime Result

The current runtime-oriented artifact family uses the selected `7`-feature subset:

- `Character Count`
- `Profanity Count`
- `Repeated Character Pattern Count`
- `Identity-group Term Count`
- `URL Count`
- `Negation Count`
- `Non-toxic Negation Pattern Count`

Saved runtime/main metadata reports:

- Test F1: `0.7584118438761777`
- Test ROC-AUC: `0.9626497598085998`
- Logistic Regression config:
  - `penalty='l2'`
  - `solver='liblinear'`
  - `C=1.0645493016479186`
  - `class_weight={0: 1, 1: 3}`
  - `tol=0.0002226958973431528`
  - `max_iter=2000`
  - `random_state=42`

Re-evaluating the saved runtime artifact on the shared split gives:

- F1: `0.758295`
- Precision: `0.774604`
- Recall: `0.742659`
- ROC-AUC: `0.962679`

The tiny difference from the saved metadata is just from recomputing through the current shared evaluation script rather than copying the old notebook printout directly.

## Practical Summary

- If you want the best currently saved F1, use the `Grid Search` artifact family.
- If you want the Optuna-centered feature-selection workflow, use `optuna_feature_test.ipynb`.
- If you want a compact `7`-feature runtime story, the Optuna feature-test artifact remains a strong option.
- If you want one canonical final model for the repo, the next clean step is to retrain and promote a single artifact family intentionally.
