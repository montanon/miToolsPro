import re
import unittest
from typing import Dict, List

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span

from mitoolspro.nlp.spacy_components import (
    SentenceLemmaTagger,
    SentenceRegexTagger,
    SentenceWordTagger,
    build_lemma_patterns,
    build_regex_pattern_table,
    build_word_patterns,
    create_sentence_lemma_tagger,
    create_sentence_regex_tagger,
    create_sentence_word_tagger,
    create_strip_accents,
)


class TestStripAccentsComponent(unittest.TestCase):
    def setUp(self):
        self.nlp = spacy.blank("en")
        self.nlp.add_pipe("strip_accents")

    def test_strip_accents_basic(self):
        doc = self.nlp("Café")
        self.assertEqual(doc.text, "Cafe")

    def test_strip_accents_multiple(self):
        doc = self.nlp("Héllò Wörld")
        self.assertEqual(doc.text, "Hello World")

    def test_strip_accents_no_change(self):
        doc = self.nlp("Hello World")
        self.assertEqual(doc.text, "Hello World")

    def test_strip_accents_empty(self):
        doc = self.nlp("")
        self.assertEqual(doc.text, "")


class TestBuildLemmaPatterns(unittest.TestCase):
    def setUp(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.categories = {"food": ["apples", "bananas"], "drink": ["coffee", "tea"]}

    def test_build_lemma_patterns_basic(self):
        patterns = build_lemma_patterns(self.nlp, self.categories)
        self.assertIn("food", patterns)
        self.assertIn("drink", patterns)
        self.assertEqual(len(patterns["food"]), 2)
        self.assertEqual(len(patterns["drink"]), 2)

    def test_build_lemma_patterns_strip_accents(self):
        categories = {"food": ["café"]}
        patterns = build_lemma_patterns(self.nlp, categories, strip_accents=True)
        self.assertEqual(patterns["food"][0].text, "cafe")

    def test_build_lemma_patterns_ignore_case(self):
        categories = {"food": ["APPLE"]}
        patterns = build_lemma_patterns(self.nlp, categories, ignore_case=True)
        self.assertEqual(patterns["food"][0].text, "apple")

    def test_build_lemma_patterns_empty(self):
        patterns = build_lemma_patterns(self.nlp, {})
        self.assertEqual(patterns, {})


class TestSentenceLemmaTagger(unittest.TestCase):
    def setUp(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.categories = {"food": ["apple", "banana"], "drink": ["coffee", "tea"]}
        self.nlp.add_pipe(
            "sentence_lemma_tagger",
            after="lemmatizer",
            config={
                "categories": self.categories,
                "strip_accents": True,
                "ignore_case": True,
            },
        )

    def test_sentence_lemma_tagger_basic(self):
        doc = self.nlp("I like apples and coffee.")
        self.assertTrue(doc[3].sent._.food)
        self.assertTrue(doc[5].sent._.drink)

    def test_sentence_lemma_tagger_multiple_sentences(self):
        doc = self.nlp("I like apples. I drink coffee.")
        self.assertTrue(doc[3].sent._.food)
        self.assertTrue(doc[7].sent._.drink)

    def test_sentence_lemma_tagger_no_match(self):
        doc = self.nlp("I like cars.")
        self.assertFalse(doc[2].sent._.food)
        self.assertFalse(doc[2].sent._.drink)

    def test_sentence_lemma_tagger_ignore_case(self):
        doc = self.nlp("I like APPLE.")
        self.assertTrue(doc[2].sent._.food)

    def test_sentence_lemma_tagger_keep_tags(self):
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe(
            "sentence_lemma_tagger",
            after="lemmatizer",
            config={
                "categories": self.categories,
                "strip_accents": True,
                "ignore_case": True,
                "keep_tags": True,
            },
        )
        doc = nlp("I like apples and bananas with coffee.")
        self.assertTrue(doc[3].sent._.food)
        self.assertTrue(doc[5].sent._.food)
        self.assertTrue(doc[7].sent._.drink)
        self.assertEqual(doc[3].sent._.food_tags, ["apples", "bananas"])
        self.assertEqual(doc[7].sent._.drink_tags, ["coffee"])


class TestBuildWordPatterns(unittest.TestCase):
    def setUp(self):
        self.nlp = spacy.blank("en")
        self.categories = {"food": ["apple", "banana"], "drink": ["coffee", "tea"]}

    def test_build_word_patterns_basic(self):
        patterns = build_word_patterns(self.nlp, self.categories)
        self.assertIn("food", patterns)
        self.assertIn("drink", patterns)
        self.assertEqual(len(patterns["food"]), 2)
        self.assertEqual(len(patterns["drink"]), 2)

    def test_build_word_patterns_strip_accents(self):
        categories = {"food": ["café"]}
        patterns = build_word_patterns(self.nlp, categories, strip_accents=True)
        self.assertEqual(patterns["food"][0].text, "cafe")

    def test_build_word_patterns_ignore_case(self):
        categories = {"food": ["APPLE"]}
        patterns = build_word_patterns(self.nlp, categories, ignore_case=True)
        self.assertEqual(patterns["food"][0].text, "apple")

    def test_build_word_patterns_empty(self):
        patterns = build_word_patterns(self.nlp, {})
        self.assertEqual(patterns, {})


class TestSentenceWordTagger(unittest.TestCase):
    def setUp(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.categories = {"food": ["apple", "banana"], "drink": ["coffee", "tea"]}
        self.nlp.add_pipe(
            "sentence_word_tagger", config={"categories": self.categories}
        )

    def test_sentence_word_tagger_basic(self):
        doc = self.nlp("I like apple and coffee.")
        self.assertTrue(doc[3].sent._.food)
        self.assertTrue(doc[5].sent._.drink)

    def test_sentence_word_tagger_multiple_sentences(self):
        doc = self.nlp("I like apple. I drink coffee.")
        self.assertTrue(doc[3].sent._.food)
        self.assertTrue(doc[7].sent._.drink)

    def test_sentence_word_tagger_no_match(self):
        doc = self.nlp("I like cars.")
        self.assertFalse(doc[2].sent._.food)
        self.assertFalse(doc[2].sent._.drink)

    def test_sentence_word_tagger_ignore_case(self):
        doc = self.nlp("I like APPLE.")
        self.assertTrue(doc[2].sent._.food)

    def test_sentence_word_tagger_keep_tags(self):
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe(
            "sentence_word_tagger",
            config={
                "categories": self.categories,
                "strip_accents": True,
                "ignore_case": True,
                "keep_tags": True,
            },
        )
        doc = nlp("I like apple and banana with coffee.")
        self.assertTrue(doc[3].sent._.food)
        self.assertTrue(doc[5].sent._.food)
        self.assertTrue(doc[7].sent._.drink)
        self.assertEqual(doc[3].sent._.food_tags, ["apple", "banana"])
        self.assertEqual(doc[7].sent._.drink_tags, ["coffee"])


class TestBuildRegexPatternTable(unittest.TestCase):
    def setUp(self):
        self.categories = {
            "email": [r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"],
            "phone": [r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"],
        }

    def test_build_regex_pattern_table_basic(self):
        patterns = build_regex_pattern_table(self.categories)
        self.assertIn("email", patterns)
        self.assertIn("phone", patterns)
        self.assertTrue(isinstance(patterns["email"], re.Pattern))
        self.assertTrue(isinstance(patterns["phone"], re.Pattern))

    def test_build_regex_pattern_table_ignore_case(self):
        patterns = build_regex_pattern_table(self.categories, ignore_case=True)
        self.assertTrue(patterns["email"].flags & re.IGNORECASE)

    def test_build_regex_pattern_table_strip_accents(self):
        categories = {"test": ["café"]}
        patterns = build_regex_pattern_table(categories, strip_accents=True)
        self.assertTrue(patterns["test"].search("cafe"))

    def test_build_regex_pattern_table_empty(self):
        patterns = build_regex_pattern_table({})
        self.assertEqual(patterns, {})


class TestSentenceRegexTagger(unittest.TestCase):
    def setUp(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.categories = {
            "email": [r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"],
            "phone": [r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"],
        }
        self.nlp.add_pipe(
            "sentence_regex_tagger", config={"categories": self.categories}
        )

    def test_sentence_regex_tagger_basic(self):
        doc = self.nlp("Contact me at test@example.com or 123-456-7890.")
        self.assertTrue(doc[4].sent._.email)
        self.assertTrue(doc[4].sent._.phone)

    def test_sentence_regex_tagger_multiple_sentences(self):
        doc = self.nlp("Email: test@example.com. Phone: 123-456-7890.")
        self.assertTrue(doc[1].sent._.email)
        self.assertTrue(doc[4].sent._.phone)

    def test_sentence_regex_tagger_no_match(self):
        doc = self.nlp("No contact information here.")
        self.assertFalse(doc[0].sent._.email)
        self.assertFalse(doc[0].sent._.phone)

    def test_sentence_regex_tagger_ignore_case(self):
        doc = self.nlp("Email: TEST@EXAMPLE.COM")
        self.assertTrue(doc[1].sent._.email)

    def test_sentence_regex_tagger_keep_tags(self):
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe(
            "sentence_regex_tagger",
            config={
                "categories": self.categories,
                "ignore_case": True,
                "strip_accents": True,
                "keep_tags": True,
            },
        )
        doc = nlp(
            "Contact me at test@example.com or 123-456-7890 and test2@example.com."
        )
        self.assertTrue(doc[4].sent._.email)
        self.assertTrue(doc[4].sent._.phone)
        self.assertEqual(
            doc[4].sent._.email_tags, ["test@example.com", "test2@example.com"]
        )
        self.assertEqual(doc[4].sent._.phone_tags, ["123-456-7890"])


if __name__ == "__main__":
    unittest.main()
