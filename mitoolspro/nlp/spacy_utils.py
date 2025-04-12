import unicodedata
from typing import Dict, List, Tuple

from spacy.language import Language
from spacy.tokens import Doc


def _strip_accents(text: str) -> str:
    return "".join(
        ch
        for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )


@Language.factory("strip_accents")
def create_strip_accents(nlp: Language, name: str):
    def strip_accents_component(doc: Doc) -> Doc:
        accentless_text = _strip_accents(doc.text)
        new_doc = nlp.make_doc(accentless_text)
        return new_doc

    return strip_accents_component


def build_lemma_match_tables(
    nlp: Language,
    categories: Dict[str, List[str]],
    *,
    strip_accents: bool = True,
) -> Tuple[Dict[str, set], Dict[str, List[Tuple[str, ...]]]]:
    single_matches: Dict[str, set] = {}
    multi_matches: Dict[str, List[Tuple[str, ...]]] = {}

    lemmatizer = nlp.get_pipe("lemmatizer")

    for category, matches in categories.items():
        single_match: set = set()
        multi_match: List[Tuple[str, ...]] = []

        for match in matches:
            text = _strip_accents(match) if strip_accents else match
            doc = lemmatizer(nlp.make_doc(text))  # fast: lemmatizer only
            lemmas = [token.lemma_ for token in doc]

            if len(lemmas) == 1:
                single_match.add(lemmas[0])
            else:
                multi_match.append(tuple(lemmas))

        single_matches[category] = single_match
        multi_matches[category] = multi_match

    return single_matches, multi_matches


@Language.factory("mark_lemma_context", default_config={"categories": {}})
def create_mark_lemma_context(
    nlp: Language, name: str, categories: Dict[str, List[str]]
):
    single_matches, multi_matches = build_lemma_match_tables(nlp, categories)

    def mark_lemma_context(doc: Doc) -> Doc:
        for sentence in doc.sents:
            open_categories = set(categories)
            tokens = [token.lemma_ for token in sentence]

            tokens_set = set(tokens)
            for category in tuple(open_categories):
                if single_matches[category] & tokens_set:
                    setattr(sentence._, category, True)
                    open_categories.remove(category)

            if open_categories:
                for i in range(len(tokens)):
                    if not open_categories:
                        break
                    for category in tuple(open_categories):
                        for sequence in multi_matches[category]:
                            end = i + len(sequence)
                            if tokens[i:end] == list(sequence):
                                setattr(sentence._, category, True)
                                open_categories.remove(category)
                                break

        return doc

    return mark_lemma_context
