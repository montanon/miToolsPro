import unicodedata
from typing import Dict, List, Optional, Union

from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer


def calculate_tfidf(
    texts: List[str],
    lowercase: bool = True,
    stop_words: Optional[Union[List[str], set[str]]] = None,
    ngram_range: tuple = (1, 1),
    max_features: Optional[int] = None,
    min_df: Union[int, float] = 1,
    max_df: Union[int, float] = 1.0,
    sorted: bool = True,
) -> Dict[str, Dict[str, float]]:
    vectorizer = TfidfVectorizer(
        lowercase=lowercase,
        stop_words=stop_words,
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
    )

    tfidf = vectorizer.fit_transform(texts)
    tfidf = DataFrame.sparse.from_spmatrix(
        tfidf, columns=vectorizer.get_feature_names_out()
    )

    if sorted:
        column_sums = tfidf.sum(axis=0)
        sorted_columns = column_sums.sort_values(ascending=False).index.tolist()
        tfidf = tfidf[sorted_columns]

    return tfidf


def _strip_accents(text: str) -> str:
    return "".join(
        ch
        for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )
