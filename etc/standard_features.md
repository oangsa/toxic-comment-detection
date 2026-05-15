# Standard Features for Toxic Comment Detection

## Current Runtime Context

The active runtime pipeline combines word TF-IDF with these 7 engineered feature columns:

1. Question mark count
2. Profanity count
3. Repeated punctuation count (`!!`, `??`, `!?`)
4. Short/unclear without toxic signal flag
5. Second-person pronoun count
6. URL count
7. Non-toxic negation pattern count

## Historical Experiment Pool

The notebooks and older tuning runs still reference a broader 16-feature pool that also included signals such as character count, word count, uppercase ratio, repeated-character patterns, identity-group terms, and general negation count.

Character TF-IDF was part of the older hybrid baseline, but it has been removed from the active runtime pipeline after causing unstable false positives on very short inputs such as `you`.

Use the 7-feature list above as the current default project context.
