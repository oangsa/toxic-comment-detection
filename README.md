# Toxic Comment Classification Project

This repository contains a toxic comment classification project built around a hybrid text pipeline:

- word-level TF-IDF
- character-level TF-IDF
- engineered toxicity features
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

## Current Model Snapshot

The saved `best_model_final.pkl` metadata reports:

- Test F1: `0.7924`
- Test ROC-AUC: `0.9758`
- Logistic Regression config:
  - `penalty='l2'`
  - `solver='liblinear'`
  - `C=1.0645493016479186`
  - `class_weight={0: 1, 1: 3}`
  - `tol=0.0002226958973431528`
  - `max_iter=2000`

The historical training notebooks also report:

- dataset shape after cleaning: `158,194` rows
- train split: `126,555`
- test split: `31,639`
- toxic ratio in train: about `9.6%`

## Important Note About Current Code vs Saved Artifacts

The latest inference code has been improved with:

- preserved pronoun tokens in word TF-IDF defaults
- a smaller default selected-feature list in code
- short neutral input protection
- affectionate profanity protection

These safeguards work immediately at inference time, but the currently saved artifacts were trained earlier and still reflect the older selected-feature metadata unless you retrain and resave them.

In practice:

- runtime protections already help reduce obvious false positives such as very short neutral inputs
- a full retrain is still recommended if you want the saved model artifacts to match the latest code defaults exactly

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

If you want the cleanest next step for the project, retrain the final artifacts with the latest `src/toxic_pipeline.py` defaults so the documentation, code, and saved model stay aligned.
