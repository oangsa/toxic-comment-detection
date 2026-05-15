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

## Current Runtime Feature Columns

The active runtime pipeline uses these 7 engineered feature columns:

- `Question Mark Count`
- `Profanity Count`
- `Repeated Punctuation Count`
- `Short/Unclear Without Toxic Signal Flag`
- `Second-person Pronoun Count`
- `URL Count`
- `Non-toxic Negation Pattern Count`

These are the columns referenced by the current runtime metadata and CLI prediction path.

## Broader Feature Families

The project still keeps a wider 16-feature engineering pool in code and notebooks for experimentation, ablation, and comparison work.

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

## Why These 7 Were Kept

The current runtime subset leans on the latest single-feature results more directly: each selected feature showed a positive delta versus the text-only baseline in the latest Optuna feature notebook. The wider 16-feature pool is still useful for research, but the runtime context now centers on these 7 columns.

## Current Feature Selection Results

The current saved Optuna feature-selection notebook reports:

- `Text only`
  - F1: `0.7575`
- `Text + all engineered` with 16 features
  - F1: `0.7616`
- `Text + Optuna subset` with 7 features
  - F1: `0.7600`

Within that notebook, the selective `7`-feature subset beats the text-only baseline, but the current `16`-feature version slightly beats the `7`-feature subset on F1. The compact subset still has the best ROC-AUC in that comparison, so the feature-selection story is now more about trading a small amount of F1 for a smaller, more targeted feature layer.

At the same time, the broader saved-model comparison across tuning methods now shows that the highest overall held-out F1 belongs to the grid-search artifact family using all `16` engineered features. This means:

- the Optuna feature notebook gives the best compact subset story
- the grid-search notebook gives the best overall saved F1 story

## Single-Feature Insights

The current single-feature table from `src/optuna_feature_test.ipynb` measures each engineered feature added individually on top of the text-only baseline:

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

This shows that several features help a little on their own, but the improvements are small and distributed. That is why feature selection and full-stack ablation are both important: a feature can look useful in isolation and still become redundant or harmful when combined with the rest.

## Drop-One Insights

The current `drop-one` ablation in `src/optuna_feature_test.ipynb` shows how the full `16`-feature configuration changes when each engineered feature is removed one at a time.

Current results:

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

What this means:

- `Second-person Pronoun Count` is the strongest single dependency in the current full-feature ablation
- `Repeated Character Pattern Count` is still meaningfully helpful
- several features become neutral or slightly harmful when used on top of the rest of the full engineered set

This is useful context for understanding why the runtime default subset and the Optuna-selected subset are not identical.

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
