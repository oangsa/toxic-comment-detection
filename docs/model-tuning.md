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
- `random_search_tuning_v2.ipynb`
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

## Final Saved Logistic Regression Configuration

The saved best-model metadata reports:

- `penalty='l2'`
- `solver='liblinear'`
- `C=1.0645493016479186`
- `class_weight={0: 1, 1: 3}`
- `tol=0.0002226958973431528`
- `max_iter=2000`
- `random_state=42`

This weighting favors recall on the minority toxic class, which is helpful on an imbalanced dataset but also contributes to some false-positive pressure on ambiguous short inputs.

## Saved Final Evaluation

The `best_model_final.pkl` metadata reports:

- Test F1: `0.7923909478517547`
- Test ROC-AUC: `0.9757904224394159`

These are the best numbers directly attached to the final saved artifact set.

## Optuna Feature-Subset Results

The saved Optuna metadata reports:

- best CV F1: `0.7941291754774195`
- best LR params:
  - `penalty='l2'`
  - `C=1.0645493016479186`
  - `tol=0.0002226958973431528`
  - `class_weight={0: 1, 1: 3}`

Feature-set comparison:

- `Text only`
  - feature count: `0`
  - F1: `0.789361`
  - precision: `0.785621`
  - recall: `0.793138`
  - ROC-AUC: `0.974891`
- `Text + all engineered`
  - feature count: `16`
  - F1: `0.791025`
  - precision: `0.785366`
  - recall: `0.796767`
  - ROC-AUC: `0.975549`
- `Text + Optuna subset`
  - feature count: `7`
  - F1: `0.792589`
  - precision: `0.787810`
  - recall: `0.797427`
  - ROC-AUC: `0.975790`

## Interpretation

The tuning story is fairly clear:

- the text baseline was already strong
- adding all engineered features helped slightly
- selecting a smaller tuned subset helped a bit more

So the engineered features appear useful, but only when kept focused.

## Why Recent Inference Changes Were Still Needed

Even with strong test-set metrics, several practical edge cases appeared during manual probing:

- very short neutral comments like `I` or `You`
- affectionate profanity such as `I fucking love you`

These are examples of a broader lesson:

- a good overall F1 score does not guarantee intuitive behavior on short, context-light inputs

That is why recent changes introduced inference-time protections without changing the broad model architecture.

## Recommended Next Tuning Step

The most useful next experiment would be to retrain the final artifacts with the latest code defaults so the model reflects:

- preserved pronoun tokens in the word vectorizer
- the updated selected-feature list
- any future safeguarded feature contracts you choose to keep

After retraining, rerun:

1. the text-only baseline
2. the all-features baseline
3. the current reduced subset
4. an updated manual edge-case evaluation sheet

That would give the cleanest apples-to-apples comparison between the older saved artifact set and the current safer runtime logic.
