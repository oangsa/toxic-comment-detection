# Documentation Index

This folder collects the project write-up behind the toxic comment classifier.

The current runtime path described across these docs uses 7 engineered feature columns:

- `Question Mark Count`
- `Profanity Count`
- `Repeated Punctuation Count`
- `Short/Unclear Without Toxic Signal Flag`
- `Second-person Pronoun Count`
- `URL Count`
- `Non-toxic Negation Pattern Count`

## Files

- [word-vectorization-visual.md](word-vectorization-visual.md)
  - gives a visual explanation of how the current word TF-IDF vectorizer turns cleaned text into numeric features
- [word-vectorization-code.md](word-vectorization-code.md)
  - shows the actual code path behind cleaning, TF-IDF transformation, and final feature assembly
- [result.md](result.md)
  - collects the current evaluated metrics across saved artifacts, Optuna feature tests, and runtime/default artifacts
- [pipeline-and-artifacts.md](pipeline-and-artifacts.md)
  - explains the runtime pipeline, preprocessing flow, vectorization, and saved artifacts
- [feature-engineering.md](feature-engineering.md)
  - documents the engineered features, why they exist, and how recent rule-based safeguards fit in
- [model-tuning.md](model-tuning.md)
  - summarizes the tuning process, historical experiments, and the saved evaluation results

## Recommended Reading Order

1. Start with the root [README.md](../README.md) for the project overview.
2. Read [result.md](result.md) for the current metrics snapshot.
3. Read [pipeline-and-artifacts.md](pipeline-and-artifacts.md) to understand the prediction path.
4. Read [word-vectorization-visual.md](word-vectorization-visual.md) for a report-friendly example of text becoming numbers.
5. Read [word-vectorization-code.md](word-vectorization-code.md) for the actual code flow behind the vectorizer.
6. Read [feature-engineering.md](feature-engineering.md) for the interpretable feature layer.
7. Read [model-tuning.md](model-tuning.md) for the experiment and evaluation history.
