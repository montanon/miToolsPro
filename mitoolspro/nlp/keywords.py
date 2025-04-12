from collections import Counter
from itertools import islice
from typing import List

from spacy.tokens import Doc


def get_freq_dist(
    doc: Doc,
    attribute: str = "lower_",
    n_grams: int = 1,
    drop_stopwords: bool = False,
    drop_punctuation: bool = False,
) -> Counter:
    tokens = (
        getattr(token, attribute)
        for token in doc
        if not token.is_space
        and not (drop_stopwords and token.is_stop)
        and not (drop_punctuation and token.is_punct)
    )

    if n_grams == 1:
        tokens = tokens
    else:
        tokens = list(tokens)
        tokens = zip(*(islice(tokens, i, None) for i in range(n_grams)))

    return Counter(tokens)


def get_tokens(
    doc: Doc,
    attribute: str = "lower_",
    drop_stopwords: bool = False,
    drop_punctuation: bool = True,
    lowercase: bool = True,
) -> List[str]:
    tokens = (
        getattr(token, attribute).lower()
        if lowercase and attribute != "lower_"
        else getattr(token, attribute)
        for token in doc
        if not token.is_space
        and not (drop_stopwords and token.is_stop)
        and not (drop_punctuation and token.is_punct)
    )
    return list(tokens)
