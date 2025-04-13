import re
from collections import Counter
from itertools import islice
from typing import Dict, List, Optional, Union

from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Span

from mitoolspro.nlp.spacy_utils import _strip_accents


@Language.factory("strip_accents")
def create_strip_accents(nlp: Language, name: str):
    def strip_accents_component(doc: Doc) -> Doc:
        accentless_text = _strip_accents(doc.text)
        new_doc = nlp.make_doc(accentless_text)
        return new_doc

    return strip_accents_component


def build_lemma_patterns(
    nlp: Language,
    categories: Dict[str, List[str]],
    strip_accents: bool = True,
    ignore_case: bool = False,
) -> Dict[str, List[Doc]]:
    lemma_docs: Dict[str, List[Doc]] = {}

    for cat, surface_list in categories.items():
        patterns = []
        for surface in surface_list:
            text = _strip_accents(surface) if strip_accents else surface
            lemmas = [tok.lemma_ for tok in nlp(text)]
            if ignore_case:
                lemmas = [lemma.lower() for lemma in lemmas]
            pattern_text = " ".join(lemmas)
            patterns.append(nlp(pattern_text))
        lemma_docs[cat] = patterns

    return lemma_docs


class SentenceLemmaTagger:
    def __init__(
        self,
        nlp: Language,
        categories: Dict[str, List[str]],
        strip_accents: bool = True,
        ignore_case: bool = True,
        keep_tags: bool = False,
    ):
        lemma_patterns = build_lemma_patterns(
            nlp,
            categories,
            strip_accents=strip_accents,
            ignore_case=ignore_case,
        )
        self.matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
        for cat, docs in lemma_patterns.items():
            self.matcher.add(cat, docs)

        for cat in categories:
            if not Span.has_extension(cat):
                Span.set_extension(cat, default=False)
            if keep_tags and not Span.has_extension(f"{cat}_tags"):
                Span.set_extension(f"{cat}_tags", default=[])
        self.ignore_case = ignore_case
        self.keep_tags = keep_tags

    def __call__(self, doc: Doc) -> Doc:
        if self.ignore_case:
            original_lemmas = [token.lemma_ for token in doc]
            for token in doc:
                token.lemma_ = token.lemma_.lower()
        for match_id, start, end in self.matcher(doc):
            category = doc.vocab.strings[match_id]
            sent = doc[start].sent
            sent._.set(category, True)
            if self.keep_tags:
                sent._.get(f"{category}_tags").append(doc[start:end].text)
        if self.ignore_case:
            for token, original_lemma in zip(doc, original_lemmas):
                token.lemma_ = original_lemma
        return doc


@Language.factory(
    "sentence_lemma_tagger",
    default_config={
        "categories": {},
        "strip_accents": True,
        "ignore_case": True,
        "keep_tags": False,
    },
)
def create_sentence_lemma_tagger(
    nlp: Language,
    name: str,
    categories: Dict[str, List[str]],
    strip_accents: bool,
    ignore_case: bool,
    keep_tags: bool,
):
    return SentenceLemmaTagger(nlp, categories, strip_accents, ignore_case, keep_tags)


class DocLemmaTagger:
    def __init__(
        self,
        nlp: Language,
        categories: Dict[str, List[str]],
        strip_accents: bool = True,
        ignore_case: bool = True,
        keep_tags: bool = False,
    ):
        lemma_patterns = build_lemma_patterns(
            nlp, categories, strip_accents, ignore_case
        )
        self.matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
        for cat, docs in lemma_patterns.items():
            self.matcher.add(cat, docs)
        for cat in categories:
            if not Doc.has_extension(cat):
                Doc.set_extension(cat, default=False)
            if keep_tags and not Doc.has_extension(f"{cat}_tags"):
                Doc.set_extension(f"{cat}_tags", default=[])
        self.ignore_case = ignore_case
        self.keep_tags = keep_tags

    def __call__(self, doc: Doc) -> Doc:
        if self.ignore_case:
            original_lemmas = [token.lemma_ for token in doc]
            for token in doc:
                token.lemma_ = token.lemma_.lower()
        for match_id, start, end in self.matcher(doc):
            category = doc.vocab.strings[match_id]
            setattr(doc._, category, True)
            if self.keep_tags:
                doc._.get(f"{category}_tags").append(doc[start:end].text)
        if self.ignore_case:
            for token, orig in zip(doc, original_lemmas):
                token.lemma_ = orig
        return doc


@Language.factory(
    "doc_lemma_tagger",
    default_config={
        "categories": {},
        "strip_accents": True,
        "ignore_case": True,
        "keep_tags": False,
    },
)
def create_doc_lemma_tagger(
    nlp: Language,
    name: str,
    categories: Dict[str, List[str]],
    strip_accents: bool,
    ignore_case: bool,
    keep_tags: bool,
):
    return DocLemmaTagger(nlp, categories, strip_accents, ignore_case, keep_tags)


def build_word_patterns(
    nlp: Language,
    categories: Dict[str, List[str]],
    strip_accents: bool = True,
    ignore_case: bool = False,
) -> Dict[str, List[Doc]]:
    word_docs = {}
    for cat, words in categories.items():
        patterns = []
        for word in words:
            text = _strip_accents(word) if strip_accents else word
            if ignore_case:
                text = text.lower()
            patterns.append(nlp.make_doc(text))
        word_docs[cat] = patterns
    return word_docs


class SentenceWordTagger:
    def __init__(
        self,
        nlp: Language,
        categories: Dict[str, List[str]],
        ignore_case: bool = True,
        strip_accents: bool = True,
        keep_tags: bool = False,
    ):
        attr = "LOWER" if ignore_case else "TEXT"
        self.matcher = PhraseMatcher(nlp.vocab, attr=attr)
        patterns = build_word_patterns(
            nlp,
            categories,
            ignore_case=ignore_case,
            strip_accents=strip_accents,
        )
        for cat, docs in patterns.items():
            self.matcher.add(cat, docs)
        for cat in categories:
            if not Span.has_extension(cat):
                Span.set_extension(cat, default=False)
            if keep_tags and not Span.has_extension(f"{cat}_tags"):
                Span.set_extension(f"{cat}_tags", default=[])
        self.ignore_case = ignore_case
        self.keep_tags = keep_tags

    def __call__(self, doc: Doc) -> Doc:
        for match_id, start, end in self.matcher(doc):
            category = doc.vocab.strings[match_id]
            sent = doc[start].sent
            sent._.set(category, True)
            if self.keep_tags:
                sent._.get(f"{category}_tags").append(doc[start:end].text)
        return doc


@Language.factory(
    "sentence_word_tagger",
    default_config={
        "categories": {},
        "strip_accents": True,
        "ignore_case": True,
        "keep_tags": False,
    },
)
def create_sentence_word_tagger(
    nlp: Language,
    name: str,
    categories: Dict[str, List[str]],
    ignore_case: bool,
    strip_accents: bool,
    keep_tags: bool,
):
    return SentenceWordTagger(nlp, categories, ignore_case, strip_accents, keep_tags)


class DocWordTagger:
    def __init__(
        self,
        nlp: Language,
        categories: Dict[str, List[str]],
        ignore_case: bool = True,
        strip_accents: bool = True,
        keep_tags: bool = False,
    ):
        attr = "LOWER" if ignore_case else "TEXT"
        self.matcher = PhraseMatcher(nlp.vocab, attr=attr)
        patterns = build_word_patterns(
            nlp, categories, ignore_case=ignore_case, strip_accents=strip_accents
        )
        for cat, docs in patterns.items():
            self.matcher.add(cat, docs)
        for cat in categories:
            if not Doc.has_extension(cat):
                Doc.set_extension(cat, default=False)
            if keep_tags and not Doc.has_extension(f"{cat}_tags"):
                Doc.set_extension(f"{cat}_tags", default=[])
        self.ignore_case = ignore_case
        self.keep_tags = keep_tags

    def __call__(self, doc: Doc) -> Doc:
        for match_id, start, end in self.matcher(doc):
            category = doc.vocab.strings[match_id]
            setattr(doc._, category, True)
            if self.keep_tags:
                doc._.get(f"{category}_tags").append(doc[start:end].text)
        return doc


@Language.factory(
    "doc_word_tagger",
    default_config={
        "categories": {},
        "strip_accents": True,
        "ignore_case": True,
        "keep_tags": False,
    },
)
def create_doc_word_tagger(
    nlp: Language,
    name: str,
    categories: Dict[str, List[str]],
    ignore_case: bool,
    strip_accents: bool,
    keep_tags: bool,
):
    return DocWordTagger(nlp, categories, ignore_case, strip_accents, keep_tags)


def build_regex_pattern_table(
    categories: Dict[str, List[str]],
    strip_accents: bool = True,
    ignore_case: bool = True,
) -> Dict[str, re.Pattern]:
    flags = re.IGNORECASE if ignore_case else 0
    pattern_table = {}
    for cat, patterns in categories.items():
        if strip_accents:
            patterns = [_strip_accents(pattern) for pattern in patterns]
        regex_str = "|".join(patterns)
        pattern_table[cat] = re.compile(regex_str, flags)
    return pattern_table


class SentenceRegexTagger:
    def __init__(
        self,
        nlp: Language,
        categories: Dict[str, List[str]],
        ignore_case: bool = True,
        strip_accents: bool = True,
        keep_tags: bool = False,
    ):
        self.pattern_table = build_regex_pattern_table(
            categories, ignore_case=ignore_case, strip_accents=strip_accents
        )
        self.strip_accents = strip_accents
        self.keep_tags = keep_tags

        for cat in categories:
            if not Span.has_extension(cat):
                Span.set_extension(cat, default=False)
            if keep_tags and not Span.has_extension(f"{cat}_tags"):
                Span.set_extension(f"{cat}_tags", default=[])

    def __call__(self, doc: Doc) -> Doc:
        for sent in doc.sents:
            text = _strip_accents(sent.text) if self.strip_accents else sent.text
            for cat, pattern in self.pattern_table.items():
                matches = list(pattern.finditer(text))
                if matches:
                    setattr(sent._, cat, True)
                    if self.keep_tags:
                        for match in matches:
                            sent._.get(f"{cat}_tags").append(match.group())
        return doc


@Language.factory(
    "sentence_regex_tagger",
    default_config={
        "categories": {},
        "ignore_case": True,
        "strip_accents": True,
        "keep_tags": False,
    },
)
def create_sentence_regex_tagger(
    nlp: Language,
    name: str,
    categories: Dict[str, List[str]],
    ignore_case: bool,
    strip_accents: bool,
    keep_tags: bool,
):
    return SentenceRegexTagger(
        nlp,
        categories,
        ignore_case=ignore_case,
        strip_accents=strip_accents,
        keep_tags=keep_tags,
    )


class DocRegexTagger:
    def __init__(
        self,
        nlp: Language,
        categories: Dict[str, List[str]],
        ignore_case: bool = True,
        strip_accents: bool = True,
        keep_tags: bool = False,
    ):
        self.pattern_table = build_regex_pattern_table(
            categories, strip_accents=strip_accents, ignore_case=ignore_case
        )
        self.strip_accents = strip_accents
        self.keep_tags = keep_tags

        for cat in categories:
            if not Doc.has_extension(cat):
                Doc.set_extension(cat, default=False)
            if keep_tags and not Doc.has_extension(f"{cat}_tags"):
                Doc.set_extension(f"{cat}_tags", default=[])

    def __call__(self, doc: Doc) -> Doc:
        text = _strip_accents(doc.text) if self.strip_accents else doc.text
        for cat, pattern in self.pattern_table.items():
            matches = list(pattern.finditer(text))
            if matches:
                setattr(doc._, cat, True)
                if self.keep_tags:
                    for match in matches:
                        doc._.get(f"{cat}_tags").append(match.group())
        return doc


@Language.factory(
    "doc_regex_tagger",
    default_config={
        "categories": {},
        "ignore_case": True,
        "strip_accents": True,
        "keep_tags": False,
    },
)
def create_doc_regex_tagger(
    nlp: Language,
    name: str,
    categories: Dict[str, List[str]],
    ignore_case: bool,
    strip_accents: bool,
    keep_tags: bool,
):
    return DocRegexTagger(
        nlp,
        categories,
        ignore_case=ignore_case,
        strip_accents=strip_accents,
        keep_tags=keep_tags,
    )


class DocBOWExtractor:
    def __init__(
        self,
        nlp: Language,
        lemmatize: bool = False,
        lowercase: bool = True,
        stop_words: Optional[Union[List[str], set[str]]] = None,
        drop_punctuation: bool = True,
        keep_stop_words: bool = False,
    ):
        if not Doc.has_extension("bow"):
            Doc.set_extension("bow", default=None)

        self.lemmatize = lemmatize
        self.lowercase = lowercase
        self.drop_punctuation = drop_punctuation
        self.stop_set = (
            {w.lower() for w in stop_words} if stop_words is not None else None
        )
        self.keep_stop_words = keep_stop_words

    def __call__(self, doc: Doc) -> Doc:
        counts = Counter()
        for token in doc:
            if token.is_space:
                continue
            if self.drop_punctuation and token.is_punct:
                continue
            if self.stop_set is None:
                if not self.keep_stop_words and token.is_stop:
                    continue
            else:
                if token.lower_ in self.stop_set:
                    continue
            term = token.lemma_ if self.lemmatize else token.text
            if self.lowercase:
                term = term.lower()
            counts[term] += 1
        doc._.bow = dict(counts.most_common())
        return doc


@Language.factory(
    "doc_bow_extractor",
    default_config={
        "lemmatize": False,
        "lowercase": True,
        "stop_words": None,
        "drop_punctuation": True,
        "keep_stop_words": False,
    },
)
def create_doc_bow_extractor(
    nlp: Language,
    name: str,
    lemmatize: bool,
    lowercase: bool,
    stop_words: Optional[Union[List[str], set[str]]],
    drop_punctuation: bool,
    keep_stop_words: bool,
):
    return DocBOWExtractor(
        nlp, lemmatize, lowercase, stop_words, drop_punctuation, keep_stop_words
    )


class DocFreqDistExtractor:
    def __init__(
        self,
        nlp: Language,
        n_grams: int = 1,
        lemmatize: bool = False,
        lowercase: bool = True,
        stop_words: Optional[Union[List[str], set]] = None,
        drop_punctuation: bool = True,
        keep_stop_words: bool = False,
        as_frequencies: bool = False,
    ):
        if not Doc.has_extension("freq_dist"):
            Doc.set_extension("freq_dist", default=None)
        self.n_grams = n_grams
        self.lemmatize = lemmatize
        self.lowercase = lowercase
        self.drop_punctuation = drop_punctuation
        self.keep_stop_words = keep_stop_words
        self.stop_set = (
            {w.lower() for w in stop_words} if stop_words is not None else None
        )
        self.as_frequencies = as_frequencies

    def __call__(self, doc: Doc) -> Doc:
        tokens = (
            self._get_term(token)
            for token in doc
            if not token.is_space
            and not (self.drop_punctuation and token.is_punct)
            and not (
                self.stop_set is None and not self.keep_stop_words and token.is_stop
            )
            and not (self.stop_set is not None and token.lower_ in self.stop_set)
        )
        if self.n_grams == 1:
            token_items = list(tokens)
        else:
            token_list = list(tokens)
            token_items = list(
                zip(*(islice(token_list, i, None) for i in range(self.n_grams)))
            )
        freq_dist = Counter(token_items)
        if self.as_frequencies:
            total = sum(freq_dist.values())
            doc._.freq_dist = {k: v / total for k, v in freq_dist.items()}
        else:
            doc._.freq_dist = dict(freq_dist.most_common())
        return doc

    def _get_term(self, token) -> str:
        term = token.lemma_ if self.lemmatize else token.text
        if self.lowercase:
            term = term.lower()
        return term


@Language.factory(
    "doc_freq_dist_extractor",
    default_config={
        "n_grams": 1,
        "lemmatize": False,
        "lowercase": True,
        "stop_words": None,
        "drop_punctuation": True,
        "keep_stop_words": False,
        "as_frequencies": False,
    },
)
def create_doc_freq_dist_extractor(
    nlp: Language,
    name: str,
    n_grams: int,
    lemmatize: bool,
    lowercase: bool,
    stop_words: Optional[Union[List[str], set]],
    drop_punctuation: bool,
    keep_stop_words: bool,
    as_frequencies: bool,
):
    return DocFreqDistExtractor(
        nlp,
        n_grams=n_grams,
        lemmatize=lemmatize,
        lowercase=lowercase,
        stop_words=stop_words,
        drop_punctuation=drop_punctuation,
        keep_stop_words=keep_stop_words,
        as_frequencies=as_frequencies,
    )


class DocTokenExtractor:
    def __init__(
        self,
        nlp: Language,
        attribute: str = "lower_",
        n_grams: Union[int, List[int]] = 1,
        drop_stopwords: bool = False,
        drop_punctuation: bool = True,
        lowercase: bool = True,
    ):
        if not Doc.has_extension("tokens"):
            Doc.set_extension("tokens", default=None)
        if isinstance(n_grams, int):
            self.n_grams = [n_grams]
        else:
            self.n_grams = n_grams
        self.attribute = attribute
        self.drop_stopwords = drop_stopwords
        self.drop_punctuation = drop_punctuation
        self.lowercase = lowercase

    def __call__(self, doc: Doc) -> Doc:
        base_tokens = [
            getattr(token, self.attribute).lower()
            if self.lowercase and self.attribute != "lower_"
            else getattr(token, self.attribute)
            for token in doc
            if not token.is_space
            and not (self.drop_stopwords and token.is_stop)
            and not (self.drop_punctuation and token.is_punct)
        ]
        result = {}
        for n in self.n_grams:
            if n == 1:
                result[n] = base_tokens
            else:
                ngram_tokens = list(
                    zip(*(islice(base_tokens, i, None) for i in range(n)))
                )
                result[n] = ngram_tokens
        doc._.tokens = result
        return doc


@Language.factory(
    "doc_token_extractor",
    default_config={
        "attribute": "lower_",
        "n_grams": 1,
        "drop_stopwords": False,
        "drop_punctuation": True,
        "lowercase": True,
    },
)
def create_doc_token_extractor(
    nlp: Language,
    name: str,
    attribute: str,
    n_grams: Union[int, List[int]],
    drop_stopwords: bool,
    drop_punctuation: bool,
    lowercase: bool,
):
    return DocTokenExtractor(
        nlp, attribute, n_grams, drop_stopwords, drop_punctuation, lowercase
    )
