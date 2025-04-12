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


def surface_to_lemma_phrases(
    nlp: Language,
    categories: Dict[str, List[str]],
    strip_accents: bool = True,
) -> Dict[str, List[Doc]]:
    lemmatizer = nlp.get_pipe("lemmatizer")
    lemma_docs: Dict[str, List[Doc]] = {}

    for cat, surface_list in categories.items():
        patterns = []
        for surface in surface_list:
            text = _strip_accents(surface) if strip_accents else surface
            lemmas = [tok.lemma_ for tok in lemmatizer(nlp.make_doc(text))]
            pattern_text = " ".join(lemmas)
            patterns.append(nlp.make_doc(pattern_text))
        lemma_docs[cat] = patterns

    return lemma_docs


@Language.factory(
    "mark_lemma_context",
    default_config={"categories": {}, "strip_accents": True, "case_sensitive": True},
)
def create_mark_lemma_context(
    nlp: Language,
    name: str,
    categories: Dict[str, List[str]],
    strip_accents: bool,
    case_sensitive: bool,
):
    return LemmaContextComponent(nlp, categories, strip_accents, case_sensitive)


class LemmaContextComponent:
    def __init__(
        self,
        nlp: Language,
        categories: Dict[str, List[str]],
        strip_accents: bool,
        case_sensitive: bool,
    ):
        lemma_patterns = surface_to_lemma_phrases(
            nlp, categories, strip_accents=strip_accents
        )

        self.matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
        for cat, docs in lemma_patterns.items():
            self.matcher.add(cat, docs)

        for cat in categories:
            if not Span.has_extension(cat):
                Span.set_extension(cat, default=False)

    def __call__(self, doc: Doc) -> Doc:
        for match_id, start, _ in self.matcher(doc):
            cat = doc.vocab.strings[match_id]
            sent = doc[start].sent
            setattr(sent._, cat, True)
        return doc
