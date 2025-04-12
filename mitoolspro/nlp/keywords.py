from collections import Counter
from itertools import islice
from typing import Dict, List, Union

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
    n_grams: int = 1,
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
    if n_grams == 1:
        tokens = tokens
    else:
        tokens = list(tokens)
        tokens = zip(*(islice(tokens, i, None) for i in range(n_grams)))

    return list(tokens)


def get_bow(
    doc: Doc,
    lemmatize: bool = False,
    lowercase: bool = True,
    stop_words: Union[None, List[str], set] = None,
    drop_punctuation: bool = True,
) -> Dict[str, int]:
    if stop_words is None:

        def is_stop(token):
            return token.is_stop
    else:
        stop_set = {w.lower() for w in stop_words}

        def is_stop(token):
            return token.lower_ in stop_set

    counts = Counter()

    for token in doc:
        if token.is_space:
            continue
        if drop_punctuation and token.is_punct:
            continue
        if is_stop(token):
            continue

        term = token.lemma_ if lemmatize else token.text
        if lowercase:
            term = term.lower()

        counts[term] += 1

    return dict(counts.most_common())
