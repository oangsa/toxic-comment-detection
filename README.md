# Toxic Comment Classification Project

This repository contains a toxic comment classification project built around a text-and-features pipeline:

- word-level TF-IDF
- a compact engineered feature layer
- a tuned Logistic Regression classifier

The project started as notebook-driven experimentation on the Jigsaw toxic comment dataset and now includes a reusable inference pipeline in `src/toxic_pipeline.py` and a simple CLI entry point in `src/main.py`.

## Overview

The main goal is to classify whether a comment is `Toxic` or `Not Toxic` while also making the model behavior more interpretable through feature engineering and visualization.

The current codebase includes:

- data cleaning and exploratory notebooks
- feature engineering experiments
- random search, grid search, and Optuna-based tuning notebooks
- saved model artifacts for local inference
- a visualization notebook showing how raw CSV text becomes cleaned text and then engineered features

The current runtime feature columns are:

- `Question Mark Count`
- `Profanity Count`
- `Repeated Punctuation Count`
- `Short/Unclear Without Toxic Signal Flag`
- `Second-person Pronoun Count`
- `URL Count`
- `Non-toxic Negation Pattern Count`

## Current Saved-Model Snapshot

Using the shared word-only pipeline and the same 20% stratified split, the saved model families currently compare as follows:

- `Grid Search`
  - engineered features: `16`
  - Test F1: `0.7682`
  - Precision: `0.7755`
  - Recall: `0.7611`
  - ROC-AUC: `0.9610`
- `Optuna Main`
  - engineered features: `16`
  - Test F1: `0.7660`
  - Precision: `0.7600`
  - Recall: `0.7720`
  - ROC-AUC: `0.9642`
- `Random Search`
  - engineered features: `16`
  - Test F1: `0.7622`
  - Precision: `0.7690`
  - Recall: `0.7555`
  - ROC-AUC: `0.9619`
- `Optuna Feature Test`
  - engineered features: `7`
  - Test F1: `0.7600`
  - Precision: `0.7686`
  - Recall: `0.7516`
  - ROC-AUC: `0.9652`
- `Runtime Best Model`
  - engineered features: `7`
  - Test F1: `0.7583`
  - Precision: `0.7746`
  - Recall: `0.7427`
  - ROC-AUC: `0.9627`

By the project’s primary tuning metric, the current saved winner is `grid_search_best_model.pkl`.

The historical training notebooks also report:

- dataset shape after cleaning: `158,194` rows
- train split: `126,555`
- test split: `31,639`
- toxic ratio in train: about `9.6%`

## Important Note About Current Code vs Saved Artifacts

The current notebooks and preferred saved artifacts now align on a word-only TF-IDF pipeline plus engineered features. The latest code also includes:

- the 7-column runtime feature contract shown above
- short neutral input protection
- affectionate profanity protection
- removal of the `char_wb` TF-IDF branch after false positives on very short inputs such as `you`

There are still some historical `*_char_vectorizer.pkl` files in `src/`, but the latest comparison above is based on the current word-only pipeline and does not use those character artifacts.

In practice:

- the strongest saved F1 result is the grid-search artifact set
- the compact 7-feature runtime artifact set is the current default inference path
- any future final export should be retrained and resaved intentionally so one artifact family is promoted as canonical

## Quick Start

Install dependencies:

```bash
pip install -r requirement.txt
```

Run the classifier:

```bash
python src/main.py
```

Classify a single comment:

```bash
python src/main.py "I do not like your car"
```

## Project Structure

- `data/`
  - source dataset files such as `train.csv`
- `docs/`
  - project documentation
- `src/`
  - notebooks, artifacts, inference code, and analysis utilities
- `src/toxic_pipeline.py`
  - shared preprocessing, feature engineering, vectorization, artifact loading, and prediction logic
- `src/main.py`
  - simple CLI wrapper for interactive prediction
- `src/data_visualizations.ipynb`
  - before/after cleaning views plus feature evidence plots

## Documentation

- [Documentation Index](docs/README.md)
- [Pipeline and Artifacts](docs/pipeline-and-artifacts.md)
- [Feature Engineering](docs/feature-engineering.md)
- [Model Tuning](docs/model-tuning.md)

## Notes on the Workflow

This repository is notebook-heavy by design. The typical project flow has been:

1. clean and inspect the dataset
2. engineer and visualize features
3. compare text-only and hybrid feature sets
4. tune Logistic Regression and related baselines
5. save reusable artifacts
6. expose a small local prediction interface

If you want the cleanest next step for the project, retrain and promote one final artifact family, then point the runtime loader and docs at that single winner so the documentation, code, and saved model stay aligned.
