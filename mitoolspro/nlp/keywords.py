from collections import Counter
from itertools import islice

from spacy.tokens import Doc


def freq_dist(
    doc: Doc,
    attr: str = "lower_",
    n_grams: int = 1,
    drop_stop: bool = False,
    drop_punct: bool = False,
) -> Counter:
    tokens = (
        getattr(tok, attr)
        for tok in doc
        if not tok.is_space
        and not (drop_stop and tok.is_stop)
        and not (drop_punct and tok.is_punct)
    )

    if n_grams == 1:
        items = tokens
    else:
        toks = list(tokens)
        items = zip(*(islice(toks, i, None) for i in range(n_grams)))

    return Counter(items)
