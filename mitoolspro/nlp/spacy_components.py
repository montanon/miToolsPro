import re
from collections import Counter
from itertools import chain, islice
from typing import Any, Dict, List, Optional, Union

from rapidfuzz.fuzz import ratio
from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Span, Token

from mitoolspro.nlp.spacy_utils import _strip_accents


class StripAccents:
    def __init__(self, nlp: Language, name: str):
        self.nlp = nlp
        self.name = name

    def __call__(self, doc: Doc) -> Doc:
        accentless_text = _strip_accents(doc.text)
        new_doc = self.nlp.make_doc(accentless_text)
        return new_doc


@Language.factory("strip_accents")
def create_strip_accents(nlp: Language, name: str):
    return StripAccents(nlp, name)


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
            if keep_tags and not Span.has_extension(f"{cat}_matches"):
                Span.set_extension(f"{cat}_matches", default=[])
            if not Token.has_extension(cat):
                Token.set_extension(cat, default=False)
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
            for token in doc[start:end]:
                token._.set(category, True)
            if self.keep_tags:
                match_text = doc[start:end].text
                sent._.get(f"{category}_tags").append(match_text)
                sent_start = sent.start_char
                char_start = doc[start].idx - sent_start
                char_end = (doc[end - 1].idx + len(doc[end - 1].text)) - sent_start
                sent._.get(f"{category}_matches").append([char_start, char_end])
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
            if keep_tags and not Doc.has_extension(f"{cat}_matches"):
                Doc.set_extension(f"{cat}_matches", default=[])
            if not Token.has_extension(cat):
                Token.set_extension(cat, default=False)
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
            for token in doc[start:end]:
                token._.set(category, True)
            if self.keep_tags:
                doc._.get(f"{category}_tags").append(doc[start:end].text)
                doc._.get(f"{category}_matches").append([start, end])
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
            if keep_tags and not Span.has_extension(f"{cat}_matches"):
                Span.set_extension(f"{cat}_matches", default=[])
            if not Token.has_extension(cat):
                Token.set_extension(cat, default=False)
        self.ignore_case = ignore_case
        self.keep_tags = keep_tags

    def __call__(self, doc: Doc) -> Doc:
        for match_id, start, end in self.matcher(doc):
            category = doc.vocab.strings[match_id]
            sent = doc[start].sent
            sent._.set(category, True)
            for token in doc[start:end]:
                token._.set(category, True)
            if self.keep_tags:
                match_text = doc[start:end].text
                sent._.get(f"{category}_tags").append(match_text)
                sent_start = sent.start_char
                char_start = doc[start].idx - sent_start
                char_end = (doc[end - 1].idx + len(doc[end - 1].text)) - sent_start
                sent._.get(f"{category}_matches").append([char_start, char_end])
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
            if keep_tags and not Doc.has_extension(f"{cat}_matches"):
                Doc.set_extension(f"{cat}_matches", default=[])
            if not Token.has_extension(cat):
                Token.set_extension(cat, default=False)
        self.ignore_case = ignore_case
        self.keep_tags = keep_tags

    def __call__(self, doc: Doc) -> Doc:
        for match_id, start, end in self.matcher(doc):
            category = doc.vocab.strings[match_id]
            setattr(doc._, category, True)
            for token in doc[start:end]:
                token._.set(category, True)
            if self.keep_tags:
                doc._.get(f"{category}_tags").append(doc[start:end].text)
                doc._.get(f"{category}_matches").append([start, end])
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
            if keep_tags and not Span.has_extension(f"{cat}_matches"):
                Span.set_extension(f"{cat}_matches", default=[])
            if not Token.has_extension(cat):
                Token.set_extension(cat, default=False)

    def __call__(self, doc: Doc) -> Doc:
        for sent in doc.sents:
            text = _strip_accents(sent.text) if self.strip_accents else sent.text
            for cat, pattern in self.pattern_table.items():
                matches = list(pattern.finditer(text))
                if matches:
                    setattr(sent._, cat, True)
                    for match in matches:
                        if self.keep_tags:
                            sent._.get(f"{cat}_tags").append(match.group())
                            sent._.get(f"{cat}_matches").append(
                                [match.start(), match.end()]
                            )
                        for token in sent:
                            if (
                                token.idx - sent.start_char >= match.start()
                                and token.idx - sent.start_char + len(token.text)
                                <= match.end()
                            ):
                                token._.set(cat, True)
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
            if keep_tags and not Doc.has_extension(f"{cat}_matches"):
                Doc.set_extension(f"{cat}_matches", default=[])
            if not Token.has_extension(cat):
                Token.set_extension(cat, default=False)

    def __call__(self, doc: Doc) -> Doc:
        text = _strip_accents(doc.text) if self.strip_accents else doc.text
        for cat, pattern in self.pattern_table.items():
            matches = list(pattern.finditer(text))
            if matches:
                setattr(doc._, cat, True)
                for match in matches:
                    if self.keep_tags:
                        doc._.get(f"{cat}_tags").append(match.group())
                        doc._.get(f"{cat}_matches").append([match.start(), match.end()])
                    for token in doc:
                        if (
                            token.idx >= match.start()
                            and token.idx + len(token.text) <= match.end()
                        ):
                            token._.set(cat, True)
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
        n_grams: Union[int, List[int]] = 1,
        lemmatize: bool = False,
        lowercase: bool = True,
        stop_words: Optional[Union[List[str], set]] = None,
        drop_punctuation: bool = True,
        keep_stop_words: bool = False,
        as_frequencies: bool = False,
    ):
        if not Doc.has_extension("freq_dist"):
            Doc.set_extension("freq_dist", default=None)
        if isinstance(n_grams, int):
            self.n_grams = [n_grams]
        else:
            self.n_grams = sorted(n_grams)
        self.lemmatize = lemmatize
        self.lowercase = lowercase
        self.drop_punctuation = drop_punctuation
        self.keep_stop_words = keep_stop_words
        self.stop_set = (
            {w.lower() for w in stop_words} if stop_words is not None else None
        )
        self.as_frequencies = as_frequencies

    def __call__(self, doc: Doc) -> Doc:
        base_tokens = []
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
            base_tokens.append(term)
        result: Dict[int, Dict[str, float]] = {}
        for n in self.n_grams:
            if n == 1:
                items = base_tokens
            else:
                items = list(zip(*(islice(base_tokens, i, None) for i in range(n))))
            freq_dist = Counter(items)
            if self.as_frequencies:
                total = sum(freq_dist.values())
                freq_dist = {k: v / total for k, v in freq_dist.items()}
            else:
                freq_dist = dict(freq_dist.most_common())
            freq_dist = {
                " ".join(k) if isinstance(k, tuple) else k: v
                for k, v in freq_dist.items()
            }
            result[n] = freq_dist
        doc._.freq_dist = result if len(result) > 1 else result[self.n_grams[0]]
        return doc


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
    n_grams: Union[int, List[int]],
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
        keep_stop_words: bool = True,
        drop_punctuation: bool = True,
        lowercase: bool = True,
        lemmatize: bool = False,
    ):
        if not Doc.has_extension("tokens"):
            Doc.set_extension("tokens", default=None)
        if isinstance(n_grams, int):
            self.n_grams = [n_grams]
        else:
            self.n_grams = sorted(n_grams)
        self.attribute = attribute
        self.keep_stop_words = keep_stop_words
        self.drop_punctuation = drop_punctuation
        self.lowercase = lowercase
        self.lemmatize = lemmatize

    def _get_term(self, token) -> str:
        term = token.lemma_ if self.lemmatize else getattr(token, self.attribute)
        if self.lowercase:
            term = term.lower()
        return term

    def __call__(self, doc: Doc) -> Doc:
        base_tokens = [
            self._get_term(token)
            for token in doc
            if not token.is_space
            and (self.keep_stop_words or not token.is_stop)
            and not (self.drop_punctuation and token.is_punct)
        ]
        result: Dict[int, List[str]] = {}
        for n in self.n_grams:
            if n == 1:
                result[n] = base_tokens
            else:
                ngram_tokens = list(
                    zip(*(islice(base_tokens, i, None) for i in range(n)))
                )
                result[n] = [
                    " ".join(k) if isinstance(k, tuple) else k for k in ngram_tokens
                ]
        doc._.tokens = result if len(result) > 1 else result[self.n_grams[0]]
        return doc


@Language.factory(
    "doc_token_extractor",
    default_config={
        "attribute": "lower_",
        "n_grams": 1,
        "keep_stop_words": False,
        "drop_punctuation": True,
        "lowercase": True,
        "lemmatize": False,
    },
)
def create_doc_token_extractor(
    nlp: Language,
    name: str,
    attribute: str,
    n_grams: Union[int, List[int]],
    keep_stop_words: bool,
    drop_punctuation: bool,
    lowercase: bool,
    lemmatize: bool,
):
    return DocTokenExtractor(
        nlp, attribute, n_grams, keep_stop_words, drop_punctuation, lowercase, lemmatize
    )


class CacheSentenceIndices:
    def __init__(self, nlp: Language, name: str):
        self.nlp = nlp
        self.name = name
        if not Doc.has_extension("sents_list"):
            Doc.set_extension("sents_list", default=None)
        if not Span.has_extension("index"):
            Span.set_extension("index", default=None)

    def __call__(self, doc: Doc) -> Doc:
        sents = [(sent.text, sent.start, sent.end) for sent in doc.sents]
        doc._.sents_list = sents
        for i, sent in enumerate(doc.sents):
            sent._.index = i
        return doc


@Language.factory("sentence_indices")
def create_cache_sentence_indices(nlp: Language, name: str):
    return CacheSentenceIndices(nlp, name)


@Language.factory("regex_replacer")
def create_regex_replacer(
    nlp: Language,
    name: str,
    patterns: Dict[str, List[str]],
    ignore_case: bool = False,
    strip_accents: bool = False,
):
    return RegexReplacer(
        nlp, patterns, ignore_case=ignore_case, strip_accents=strip_accents
    )


class RegexReplacer:
    def __init__(
        self,
        nlp: Language,
        patterns: Dict[str, List[str]],
        ignore_case: bool = False,
        strip_accents: bool = False,
    ):
        self.nlp = nlp
        self.ignore_case = ignore_case
        self.strip_accents = strip_accents
        self.patterns = self._compile_patterns(patterns)
        if not Doc.has_extension("replacements"):
            Doc.set_extension("replacements", default=[])

    def _compile_patterns(
        self, patterns: Dict[str, List[str]]
    ) -> Dict[str, List[re.Pattern]]:
        flags = re.IGNORECASE if self.ignore_case else 0
        return {
            label: [re.compile(pat, flags=flags) for pat in pats]
            for label, pats in patterns.items()
        }

    def resolve_overlaps(
        self, matches: list[tuple[int, int, str, str, int]]
    ) -> list[tuple[int, int, str, str]]:
        # Sort by: longest match first, then highest priority, then earliest start
        matches.sort(key=lambda x: (-(x[1] - x[0]), x[4], x[0]))

        selected = []
        occupied = set()

        for start, end, label, text, priority in matches:
            if not any(i in occupied for i in range(start, end)):
                selected.append((start, end, label, text))
                occupied.update(range(start, end))

        return sorted(selected, key=lambda x: x[0])

    def _normalize(self, text: str) -> str:
        text = text.lower() if self.ignore_case else text
        text = _strip_accents(text) if self.strip_accents else text
        return text

    def __call__(self, doc: Doc) -> Doc:
        original_text = doc.text
        norm_text = self._normalize(original_text)

        matches = []
        for priority, (label, patterns) in enumerate(self.patterns.items()):
            for pattern in patterns:
                for match in pattern.finditer(norm_text):
                    start, end = match.start(), match.end()
                    matches.append((start, end, label, match.group(), priority))

        # Resolve overlaps: longer matches first, then by priority
        selected = self.resolve_overlaps(matches)

        # Perform replacements
        new_text = []
        last_idx = 0
        replacements = []

        for start, end, label, original in selected:
            prefix = original_text[last_idx:start]
            suffix = original_text[end] if end < len(original_text) else ""

            # Ensure spacing around replacement
            needs_left_pad = prefix and prefix[-1].isalnum()
            needs_right_pad = suffix and suffix[0].isalnum()

            replacement = label
            if needs_left_pad:
                replacement = " " + replacement
            if needs_right_pad:
                replacement = replacement + " "

            new_text.append(original_text[last_idx:start])
            new_text.append(replacement)
            replacements.append((start, end, label, original))
            last_idx = end

        new_text.append(original_text[last_idx:])
        replaced_text = "".join(new_text)

        new_doc = self.nlp.make_doc(replaced_text)
        new_doc._.replacements = replacements

        # Optional: set span-level metadata
        for start, end, label, original in replacements:
            try:
                ext_name = f"{label}_original"
                if not Span.has_extension(ext_name):
                    Span.set_extension(ext_name, default=None)
                span = new_doc.char_span(start, start + len(label))
                if span:
                    setattr(span._, ext_name, original)
            except Exception:
                pass  # extension errors or span alignment mismatches

        return new_doc


@Language.factory("entity_replacer")
def create_entity_replacer(
    nlp: Language,
    name: str,
    entities: Dict[str, list[str]],
    abbreviations: Dict[str, list[str]],
    stopwords: Optional[Union[List[str], set[str]]] = None,
    *,
    min_overlap: int = 2,
    fuzzy_threshold: float = 0.85,
    match_threshold: float = 0.75,
):
    return EntityReplacer(
        nlp,
        name,
        entities,
        abbreviations,
        stopwords,
        min_overlap=min_overlap,
        fuzzy_threshold=fuzzy_threshold,
        match_threshold=match_threshold,
    )


class EntityReplacer:
    def __init__(
        self,
        nlp: Language,
        name: str,
        entities: Dict[str, list[str]],
        abbreviations: Dict[str, list[str]],
        stopwords: str,
        *,
        min_overlap: int = 2,
        fuzzy_threshold: float = 0.8,
        match_threshold: float = 0.8,
    ):
        if not Doc.has_extension("entity_replacements"):
            Doc.set_extension("entity_replacements", default=[])
        self.nlp = nlp
        self.name = name
        self.entities = entities
        self.abbreviations = abbreviations
        self.stopwords = stopwords if stopwords is not None else set()
        self.min_overlap = min_overlap
        self.fuzzy_threshold = fuzzy_threshold
        self.match_threshold = match_threshold
        self.max_span_length = int(max(len(v) for v in entities.values()) * 1.5)
        self._build_entity_index()

    def _fuzzy_match(
        self, span_tokens: List[str], candidate: List[str]
    ) -> tuple[bool, float]:
        if not span_tokens:
            return False, 0.0
        total_characters = sum(len(token) for token in candidate)
        matched_token_count = 0
        matched_candidate_count = 0
        for candidate_token in candidate:
            if len(candidate_token) == 1:
                if candidate_token in span_tokens:
                    matched_candidate_count += 1
                    matched_token_count += 1
            else:
                best_score = max(
                    ratio(candidate_token, span_token) / 100.0
                    for span_token in span_tokens
                )
                if best_score >= self.fuzzy_threshold:
                    matched_candidate_count += 1
                    matched_token_count += len(candidate_token)
        if total_characters != 0:
            match_ratio = matched_token_count / total_characters
        else:
            match_ratio = 0
        match = (
            (matched_candidate_count >= self.min_overlap)
            and match_ratio >= self.match_threshold
            and match_ratio > self.match_threshold
        )
        return match, match_ratio

    def _find_matches(self, doc: Doc) -> List[Dict[str, Any]]:
        matches = []
        doc_tokens = [token.text for token in doc]
        norm_tokens = [self._tokenize_and_normalize(token.text) for token in doc]
        n = len(doc_tokens)
        for i in range(n):
            for j in range(i + 1, min(n + 1, i + self.max_span_length)):
                span = doc[i:j]
                span_tokens = list(chain.from_iterable(norm_tokens[i:j]))

                if not span_tokens:
                    continue

                for ent in self.entity_index:
                    candidate_tokens = ent["norm_tokens"]

                    if span_tokens == candidate_tokens:
                        matches.append(
                            {
                                "start": span.start,
                                "end": span.end,
                                "label": ent["label"],
                                "replaced": span.text,
                                "score": 1.0,
                            }
                        )
                        continue
                    elif (
                        span_tokens[0] not in candidate_tokens
                        or span_tokens[-1] not in candidate_tokens
                        or abs(len(span_tokens) - len(candidate_tokens)) > 3
                        or len(set(span_tokens) & set(candidate_tokens))
                        < self.min_overlap
                    ):
                        continue

                    match, score = self._fuzzy_match(span_tokens, candidate_tokens)
                    if match:
                        matches.append(
                            {
                                "start": span.start,
                                "end": span.end,
                                "label": ent["label"],
                                "replaced": span.text,
                                "score": score,
                            }
                        )

        matches.sort(key=lambda m: (m["end"] - m["start"], m["score"]), reverse=True)
        seen = set()
        final_matches = []
        for m in matches:
            if all(i not in seen for i in range(m["start"], m["end"])):
                seen.update(range(m["start"], m["end"]))
                final_matches.append(m)

        return final_matches

    def __call__(self, doc: Doc) -> Doc:
        matches = self._find_matches(doc)

        matches = sorted(matches, key=lambda m: m["start"])
        replacements = []
        new_text = []
        last_idx = 0
        original_text = doc.text

        for match in matches:
            span = doc[match["start"] : match["end"]]
            start_char = span.start_char
            end_char = span.end_char
            label = match["label"]
            replaced = span.text

            prefix = original_text[last_idx:start_char]
            suffix = original_text[end_char:] if end_char < len(original_text) else ""

            needs_left_pad = prefix and prefix[-1].isalnum()
            needs_right_pad = suffix and suffix[0].isalnum()

            replacement = label
            if needs_left_pad:
                replacement = " " + replacement
            if needs_right_pad:
                replacement = replacement + " "

            new_text.append(original_text[last_idx:start_char])
            new_text.append(replacement)
            replacements.append(
                {
                    "start": match["start"],
                    "end": match["end"],
                    "label": label,
                    "replaced": replaced,
                }
            )

            last_idx = end_char

        new_text.append(original_text[last_idx:])
        replaced_text = "".join(new_text)
        new_doc = self.nlp.make_doc(replaced_text)
        new_doc = doc.from_docs([new_doc])
        new_doc._.entity_replacements = replacements
        return new_doc

    def _build_entity_index(self):
        self.entity_index = []
        for label, values in self.entities.items():
            for original in values:
                norm_tokens = self._tokenize_and_normalize(original)
                self.entity_index.append(
                    {"label": label, "original": original, "norm_tokens": norm_tokens}
                )

    def _tokenize_and_normalize(self, text: str) -> list[str]:
        text = self._normalize_text(text)
        text = self._normalize_rut(text)
        text = self._expand_abbreviation(text)
        return [token for token in text.split() if token.lower() not in self.stopwords]

    def _normalize_text(self, text: str) -> str:
        return _strip_accents(text.lower())

    def _normalize_rut(self, text: str) -> str:
        if self._is_valid_rut(text):
            return re.sub(r"[^0-9kK]", "", text.lower())
        return text

    def _expand_abbreviation(self, text: str) -> str:
        for abbr, expansion in self.abbreviations.items():
            pattern = re.escape(abbr.lower())
            replacement = " ".join(expansion)
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def _is_valid_rut(self, text: str) -> bool:
        return re.match(r"^[0-9kK]+$", text) is not None
