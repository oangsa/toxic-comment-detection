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

1. Character TF-IDF (3-5 grams, `char_wb`) **[Text vectorization: Text -> Vector]**
2. Uppercase ratio
3. Question mark count
4. Repeated punctuation count (`!!`, `??`, `!?`)
5. Identity-group term count (for hate context)
6. URL flag/count
7. Negation count (`not`, `never`, `no`)
8. Sentiment polarity score (optional support feature)

This set is a balanced standard baseline: straightforward to build and usually strong for toxic classification tasks.
