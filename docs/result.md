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

- `Question Mark Count`
- `Profanity Count`
- `Repeated Punctuation Count`
- `Short/Unclear Without Toxic Signal Flag`
- `Second-person Pronoun Count`
- `URL Count`
- `Non-toxic Negation Pattern Count`

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

### Single-Feature Additions

The same notebook also reports the following single-feature results when each engineered feature is added individually on top of the text-only baseline:

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

### Drop-One Ablation

The same notebook also reports the following `drop-one` results for the full `16`-feature setup:

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

Main takeaway:

- removing `Second-person Pronoun Count` hurts the full model the most
- several other features are neutral or slightly harmful in the full `16`-feature stack

## Main Notebook / Runtime Result

The current runtime-oriented artifact family uses the selected `7`-feature subset:

- `Question Mark Count`
- `Profanity Count`
- `Repeated Punctuation Count`
- `Short/Unclear Without Toxic Signal Flag`
- `Second-person Pronoun Count`
- `URL Count`
- `Non-toxic Negation Pattern Count`

This is the current runtime feature contract reflected in `src/toxic_pipeline.py` and the compact default artifact family.

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
