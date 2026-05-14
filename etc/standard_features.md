# Standard Features for Toxic Comment Detection

## Oangsa

1. Word TF-IDF (unigram + bigram) **[Text vectorization: Text -> Vector]**
2. Character count (`len(comment_text)`)
3. Word count
4. Exclamation count
5. Profanity count (dictionary-based)
6. Second-person pronoun count (`you`, `your`, `u`)
7. Repeated-character score (`soooo`, `idiottt`)
8. Average word length

## Ploy

1. Uppercase ratio
2. Question mark count
3. Repeated punctuation count (`!!`, `??`, `!?`)
4. Identity-group term count (for hate context)
5. URL flag/count
6. Negation count (`not`, `never`, `no`)
7. Sentiment polarity score (optional support feature)
8. Short neutral input protection at inference time

Character TF-IDF was part of the older hybrid baseline, but it has been removed from the active runtime pipeline after causing unstable false positives on very short inputs such as `you`.

This set is a balanced standard baseline for the current project direction: straightforward to build and usually strong for toxic classification tasks.
