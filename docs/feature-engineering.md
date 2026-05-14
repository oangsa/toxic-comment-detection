# Feature Engineering

## Why Feature Engineering Was Added

A pure TF-IDF model already performs well on many toxic-comment tasks, but it can miss some structured cues that are easy for humans to notice:

- explicit profanity
- repeated aggressive spelling
- identity-related references
- uppercase shouting
- short unclear fragments
- negation patterns that flip meaning

The project therefore adds interpretable numeric features on top of the text features so the model can use both learned lexical patterns and explicit behavioral signals.

## Feature Families

### Length and structure

- `Character Count`
- `Word Count`
- `Average Word Length`

These features help distinguish very short fragments, long explanations, and dense message styles.

### Emphasis and formatting

- `Exclamation Count`
- `Question Mark Count`
- `Repeated Punctuation Count`
- `Uppercase Ratio`
- `Repeated Character Pattern Count`

These features approximate cues like shouting, stylized emphasis, or elongated spelling.

### Toxic cue features

- `Profanity Count`
- `Strong Toxic Signal Flag`
- `Second-person Pronoun Count`
- `Identity-group Term Count`

These were designed to capture patterns commonly associated with direct attacks or identity-targeted language.

### Context and de-escalation features

- `Negation Count`
- `Non-toxic Negation Pattern Count`
- `Short/Unclear Without Toxic Signal Flag`
- `URL Count`

These are more contextual. Some are useful because they can reduce over-triggering:

- `Non-toxic Negation Pattern Count` helps preserve phrases like `do not like` or `not stupid`
- `Short/Unclear Without Toxic Signal Flag` helps detect low-information comments that should be treated cautiously

## Current Selected Features

The current code default subset is:

- `Character Count`
- `Profanity Count`
- `Repeated Character Pattern Count`
- `Identity-group Term Count`
- `URL Count`
- `Negation Count`
- `Non-toxic Negation Pattern Count`

This subset is more conservative than some earlier experiments. In particular, recent code changes intentionally removed `Second-person Pronoun Count` and `Uppercase Ratio` from the default selected list because they were contributing to odd edge-case behavior on very short inputs.

## Historical Feature Selection Results

The Optuna feature selection notebook saved the following comparison:

- `Text only`
  - F1: `0.7894`
- `Text + all engineered` with 16 features
  - F1: `0.7910`
- `Text + Optuna subset` with 7 features
  - F1: `0.7926`

This is a modest but real gain, and it suggests the engineered layer is most useful when it stays selective rather than simply adding every possible numeric signal.

## Single-Feature Insights

The saved Optuna metadata indicates that several single engineered features improved F1 relative to the text-only baseline, especially:

- `Character Count`
- `Uppercase Ratio`
- `Repeated Character Pattern Count`
- `Short/Unclear Without Toxic Signal Flag`
- `Negation Count`

At the same time, some features provided little gain or even slightly reduced performance when added alone, which is why subset selection mattered.

## Drop-One Insights

The saved ablation results show that removing some features hurts the full engineered model more than removing others.

Historically, the biggest drops came from removing:

- `Uppercase Ratio`
- `Repeated Character Pattern Count`
- `Non-toxic Negation Pattern Count`
- `Character Count`

This is useful context for understanding the original training decisions, even though the latest code now takes a safer stance on some of these same features during inference.

## Recent Safeguard Changes

### Short neutral input protection

The team observed that very short inputs like `I` or `You` could be classified as toxic in unrealistic ways. This usually happens because there is too little semantic context and the model falls back on unstable shortcuts.

To reduce this issue, the pipeline now explicitly protects short, non-profane, neutral-looking inputs.

### Affectionate profanity protection

Another observed failure mode was phrases like `I fucking love you`. The model strongly associated profanity fragments with toxicity and did not sufficiently account for positive meaning.

The pipeline now protects a narrow set of clearly positive profanity contexts, such as:

- `I fucking love you`
- `this is fucking amazing`
- `fucking awesome`

This does not make profanity safe everywhere. Direct insults like `you fucking idiot` are still left exposed to the classifier.

## How to Explain the Feature Layer

If you need a short presentation summary, a good framing is:

1. Clean the raw text.
2. Extract normalized lexical features with TF-IDF.
3. Add a small set of engineered indicators for patterns humans can interpret directly.
4. Use tuning and ablation to keep only the features that provide measurable value.

## Visualization Support

The notebook `src/data_visualizations.ipynb` was added to support this feature-engineering story visually. It shows:

- raw CSV text before cleaning
- cleaned text after normalization
- token changes introduced by cleaning
- examples of comments that trigger particular feature families
- per-feature scorecards and class-separated distributions
