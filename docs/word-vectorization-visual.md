# Word Vectorization Visual

This page shows how the project’s current word TF-IDF vectorizer turns cleaned text into numbers.

It matches the active pipeline in `src/toxic_pipeline.py`:

- word-level TF-IDF only
- `ngram_range=(1, 2)`
- `max_features=10000`
- `min_df=2`
- `max_df=0.9`
- English stop words are removed, except `not`, `no`, and `never`

## Big Picture

The vectorizer does not feed raw words directly into Logistic Regression.

Instead, it converts each comment into a long numeric feature vector:

```text
Raw comment
    ->
Cleaned text
    ->
Known vocabulary terms
    ->
TF-IDF weights
    ->
Sparse numeric vector
```

## Simple Visual

Example raw comment:

```text
Why are you not stupid?
```

After project cleaning:

```text
why are you not stupid
```

Then the word vectorizer checks which vocabulary terms from the trained model appear in that cleaned text.

In the current saved `best_model_word_vectorizer.pkl`, this example activates:

- `not` at vocabulary index `5693`
- `stupid` at vocabulary index `8487`

So the comment becomes a sparse vector like this:

```text
Index:   ... 5693 ... 8487 ...
Term:    ...  not ... stupid ...
Value:   ... 0.3016 .. 0.9534 ..
```

Everything not present stays `0`.

That means the real vector is mostly zeros:

```text
[0, 0, 0, ..., 0.3016, ..., 0.9534, ..., 0]
```

## Why The Numbers Are Not Just Word Counts

TF-IDF stands for:

- `TF`: term frequency
- `IDF`: inverse document frequency

So the final number is not just “how many times the word appears”.

It is closer to:

```text
important word in this comment
and
not too common across the whole dataset
```

Because of that:

- repeated or distinctive words get larger weights
- very common words usually get smaller weights
- absent words get `0`

## Worked Example Table

Here is the same transformation in table form.

| Step | Output |
|---|---|
| Raw text | `Why are you not stupid?` |
| Cleaned text | `why are you not stupid` |
| Terms kept by vectorizer | `not`, `stupid` |
| Vocabulary indices | `5693`, `8487` |
| TF-IDF values | `0.3016`, `0.9534` |

## Another Few Real Examples

These examples were checked against the current saved word vectorizer.

| Raw text | Cleaned text | Active TF-IDF terms |
|---|---|---|
| `you are stupid` | `you are stupid` | `stupid` |
| `fucking idiot` | `fucking idiot` | `fucking`, `idiot` |
| `this is fucking amazing` | `this is fucking amazing` | `amazing`, `fucking` |
| `why are you not stupid` | `why are you not stupid` | `not`, `stupid` |

## What Happened To Words Like `you` and `are`?

Not every visible word becomes a feature.

Some words disappear because:

- they are removed by the stop-word list
- they did not survive the fitted `max_features` cutoff
- they were too rare or too common during training

That is why:

- `you are stupid` can end up activating only `stupid`
- `why are you not stupid` can end up activating only `not` and `stupid`

## Why This Helps The Model

This numeric representation gives the classifier something it can compute with.

Instead of reading language directly, Logistic Regression receives:

- a long numeric text vector from TF-IDF
- engineered numeric features such as `Profanity Count` or `URL Count`

Then it learns which dimensions push a comment toward:

- `Toxic`
- `Not Toxic`

## Short Report Version

If you need a short explanation for the report, you can use this:

> The word vectorizer converts cleaned comments into sparse TF-IDF vectors. Each known vocabulary term is assigned a numeric position, and the model stores a weight at that position based on how important the term is in the comment and how rare it is across the dataset. For example, the cleaned comment `why are you not stupid` activates the `not` and `stupid` dimensions, producing a numeric vector that the classifier can use for prediction.
