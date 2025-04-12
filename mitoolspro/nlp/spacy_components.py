from typing import Dict, List, Tuple

from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Span

from mitoolspro.nlp.spacy_utils import CategoryMatches, _strip_accents


@Language.factory("strip_accents")
def create_strip_accents(nlp: Language, name: str):
    def strip_accents_component(doc: Doc) -> Doc:
        accentless_text = _strip_accents(doc.text)
        new_doc = nlp.make_doc(accentless_text)
        return new_doc

    return strip_accents_component


def build_lemma_phrase_patterns(
    nlp: Language,
    categories: Dict[str, List[str]],
    strip_accents: bool = True,
    case_sensitive: bool = False,
) -> Dict[str, List[Doc]]:
    lemmatizer = nlp.get_pipe("lemmatizer")
    lemma_docs: Dict[str, List[Doc]] = {}

    for cat, surface_list in categories.items():
        patterns = []
        for surface in surface_list:
            text = _strip_accents(surface) if strip_accents else surface
            lemmas = [tok.lemma_ for tok in lemmatizer(nlp.make_doc(text))]
            if not case_sensitive:
                lemmas = [lemma.lower() for lemma in lemmas]
            pattern_text = " ".join(lemmas)
            patterns.append(nlp.make_doc(pattern_text))
        lemma_docs[cat] = patterns

    return lemma_docs


@Language.factory(
    "sentence_lemma_tagger",
    default_config={
        "categories": {},
        "strip_accents": True,
        "case_sensitive": False,
    },
)
def create_sentence_lemma_tagger(
    nlp: Language,
    name: str,
    categories: Dict[str, List[str]],
    strip_accents: bool,
    case_sensitive: bool,
):
    return SentenceLemmaTagger(nlp, categories, strip_accents, case_sensitive)


class SentenceLemmaTagger:
    def __init__(
        self,
        nlp: Language,
        categories: Dict[str, List[str]],
        strip_accents: bool,
        case_sensitive: bool,
    ):
        lemma_patterns = build_lemma_phrase_patterns(
            nlp,
            categories,
            strip_accents=strip_accents,
            case_sensitive=case_sensitive,
        )
        self.matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
        for cat, docs in lemma_patterns.items():
            self.matcher.add(cat, docs)

        for cat in categories:
            if not Span.has_extension(cat):
                Span.set_extension(cat, default=False)

    def __call__(self, doc: Doc) -> Doc:
        for match_id, start, end in self.matcher(doc):
            cat = doc.vocab.strings[match_id]
            doc[start].sent._.set(cat, True)
        return doc
