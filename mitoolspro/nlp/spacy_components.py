import re
from typing import Dict, List

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
    lemmatizer = nlp.get_pipe("lemmatizer")
    lemma_docs: Dict[str, List[Doc]] = {}

    for cat, surface_list in categories.items():
        patterns = []
        for surface in surface_list:
            text = _strip_accents(surface) if strip_accents else surface
            lemmas = [tok.lemma_ for tok in lemmatizer(nlp.make_doc(text))]
            if ignore_case:
                lemmas = [lemma.lower() for lemma in lemmas]
            pattern_text = " ".join(lemmas)
            patterns.append(nlp.make_doc(pattern_text))
        lemma_docs[cat] = patterns

    return lemma_docs


class SentenceLemmaTagger:
    def __init__(
        self,
        nlp: Language,
        categories: Dict[str, List[str]],
        strip_accents: bool,
        ignore_case: bool,
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

    def __call__(self, doc: Doc) -> Doc:
        for match_id, start, end in self.matcher(doc):
            category = doc.vocab.strings[match_id]
            doc[start].sent._.set(category, True)
        return doc


@Language.factory(
    "sentence_lemma_tagger",
    default_config={
        "categories": {},
        "strip_accents": True,
        "ignore_case": True,
    },
)
def create_sentence_lemma_tagger(
    nlp: Language,
    name: str,
    categories: Dict[str, List[str]],
    strip_accents: bool,
    ignore_case: bool,
):
    return SentenceLemmaTagger(nlp, categories, strip_accents, ignore_case)


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
        self.ignore_case = ignore_case

    def __call__(self, doc: Doc) -> Doc:
        for match_id, start, end in self.matcher(doc):
            category = doc.vocab.strings[match_id]
            doc[start].sent._.set(category, True)
        return doc


@Language.factory(
    "sentence_word_tagger",
    default_config={
        "categories": {},
        "strip_accents": True,
        "ignore_case": True,
    },
)
def create_sentence_word_tagger(
    nlp: Language,
    name: str,
    categories: Dict[str, List[str]],
    ignore_case: bool,
    strip_accents: bool,
):
    return SentenceWordTagger(nlp, categories, ignore_case, strip_accents)


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
    ):
        self.pattern_table = build_regex_pattern_table(
            categories, ignore_case=ignore_case, strip_accents=strip_accents
        )
        self.strip_accents = strip_accents

        for cat in categories:
            if not Span.has_extension(cat):
                Span.set_extension(cat, default=False)

    def __call__(self, doc: Doc) -> Doc:
        for sent in doc.sents:
            text = _strip_accents(sent.text) if self.strip_accents else sent.text
            for cat, pattern in self.pattern_table.items():
                if pattern.search(text):
                    setattr(sent._, cat, True)
        return doc


@Language.factory(
    "sentence_regex_tagger",
    default_config={
        "categories": {},
        "ignore_case": True,
        "strip_accents": True,
    },
)
def create_sentence_regex_tagger(
    nlp: Language,
    name: str,
    categories: Dict[str, List[str]],
    ignore_case: bool,
    strip_accents: bool,
):
    return SentenceRegexTagger(
        nlp, categories, ignore_case=ignore_case, strip_accents=strip_accents
    )
