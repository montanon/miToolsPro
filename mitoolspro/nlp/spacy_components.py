from typing import Dict, List

from spacy.language import Language
from spacy.tokens import Doc

from mitoolspro.nlp.spacy_utils import CategoryMatches, _strip_accents


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
) -> List[CategoryMatches]:
    return [
        CategoryMatches(name=name, matches=matches, strip_accents=strip_accents)
        for name, matches in categories.items()
    ]


@Language.factory("mark_lemma_context", default_config={"categories": {}})
def create_mark_lemma_context(
    nlp: Language, name: str, categories: Dict[str, List[str]]
):
    category_objects = build_lemma_match_tables(nlp, categories)

    def mark_lemma_context(doc: Doc) -> Doc:
        for sentence in doc.sents:
            open_categories = {cat.name for cat in category_objects}
            tokens = [token.lemma_ for token in sentence]

            tokens_set = set(tokens)
            for category in tuple(open_categories):
                cat_obj = next(c for c in category_objects if c.name == category)
                if cat_obj.single_matches & tokens_set:
                    setattr(sentence._, category, True)
                    open_categories.remove(category)

            if open_categories:
                for i in range(len(tokens)):
                    if not open_categories:
                        break
                    for category in tuple(open_categories):
                        cat_obj = next(
                            c for c in category_objects if c.name == category
                        )
                        for sequence in cat_obj.multi_matches:
                            end = i + len(sequence)
                            if tokens[i:end] == list(sequence):
                                setattr(sentence._, category, True)
                                open_categories.remove(category)
                                break

        return doc

    return mark_lemma_context
