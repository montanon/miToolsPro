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


class TestDocLemmaTagger(unittest.TestCase):
    def setUp(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.categories = {"food": ["apple", "banana"], "drink": ["coffee", "tea"]}
        self.nlp.add_pipe(
            "doc_lemma_tagger",
            after="lemmatizer",
            config={
                "categories": self.categories,
                "strip_accents": True,
                "ignore_case": True,
            },
        )

    def test_doc_lemma_tagger_basic(self):
        doc = self.nlp("I like apples and coffee.")
        self.assertTrue(doc._.food)
        self.assertTrue(doc._.drink)

    def test_doc_lemma_tagger_multiple_sentences(self):
        doc = self.nlp("I like apples. I drink coffee.")
        self.assertTrue(doc._.food)
        self.assertTrue(doc._.drink)

    def test_doc_lemma_tagger_no_match(self):
        doc = self.nlp("I like cars.")
        self.assertFalse(doc._.food)
        self.assertFalse(doc._.drink)

    def test_doc_lemma_tagger_ignore_case(self):
        doc = self.nlp("I like APPLE.")
        self.assertTrue(doc._.food)

    def test_doc_lemma_tagger_keep_tags(self):
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe(
            "doc_lemma_tagger",
            after="lemmatizer",
            config={
                "categories": self.categories,
                "strip_accents": True,
                "ignore_case": True,
                "keep_tags": True,
            },
        )
        doc = nlp("I like apples and bananas with coffee.")
        self.assertTrue(doc._.food)
        self.assertTrue(doc._.drink)
        self.assertEqual(doc._.food_tags, ["apples", "bananas"])
        self.assertEqual(doc._.drink_tags, ["coffee"])


class TestDocWordTagger(unittest.TestCase):
    def setUp(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.categories = {"food": ["apple", "banana"], "drink": ["coffee", "tea"]}
        self.nlp.add_pipe(
            "doc_word_tagger",
            config={
                "categories": self.categories,
                "strip_accents": True,
                "ignore_case": True,
            },
        )

    def test_doc_word_tagger_basic(self):
        doc = self.nlp("I like apple and coffee.")
        self.assertTrue(doc._.food)
        self.assertTrue(doc._.drink)

    def test_doc_word_tagger_multiple_sentences(self):
        doc = self.nlp("I like apple. I drink coffee.")
        self.assertTrue(doc._.food)
        self.assertTrue(doc._.drink)

    def test_doc_word_tagger_no_match(self):
        doc = self.nlp("I like cars.")
        self.assertFalse(doc._.food)
        self.assertFalse(doc._.drink)

    def test_doc_word_tagger_ignore_case(self):
        doc = self.nlp("I like APPLE.")
        self.assertTrue(doc._.food)

    def test_doc_word_tagger_keep_tags(self):
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe(
            "doc_word_tagger",
            config={
                "categories": self.categories,
                "strip_accents": True,
                "ignore_case": True,
                "keep_tags": True,
            },
        )
        doc = nlp("I like apple and banana with coffee.")
        self.assertTrue(doc._.food)
        self.assertTrue(doc._.drink)
        self.assertEqual(doc._.food_tags, ["apple", "banana"])
        self.assertEqual(doc._.drink_tags, ["coffee"])


class TestDocRegexTagger(unittest.TestCase):
    def setUp(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.categories = {
            "email": [r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"],
            "phone": [r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"],
        }
        self.nlp.add_pipe(
            "doc_regex_tagger",
            config={
                "categories": self.categories,
                "ignore_case": True,
                "strip_accents": True,
            },
        )

    def test_doc_regex_tagger_basic(self):
        doc = self.nlp("Contact me at test@example.com or 123-456-7890.")
        self.assertTrue(doc._.email)
        self.assertTrue(doc._.phone)

    def test_doc_regex_tagger_multiple_sentences(self):
        doc = self.nlp("Email: test@example.com. Phone: 123-456-7890.")
        self.assertTrue(doc._.email)
        self.assertTrue(doc._.phone)

    def test_doc_regex_tagger_no_match(self):
        doc = self.nlp("No contact information here.")
        self.assertFalse(doc._.email)
        self.assertFalse(doc._.phone)

    def test_doc_regex_tagger_ignore_case(self):
        doc = self.nlp("Email: TEST@EXAMPLE.COM")
        self.assertTrue(doc._.email)

    def test_doc_regex_tagger_keep_tags(self):
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe(
            "doc_regex_tagger",
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
        self.assertTrue(doc._.email)
        self.assertTrue(doc._.phone)
        self.assertEqual(doc._.email_tags, ["test@example.com", "test2@example.com"])
        self.assertEqual(doc._.phone_tags, ["123-456-7890"])


class TestDocFreqDistExtractor(unittest.TestCase):
    def setUp(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.add_pipe(
            "doc_freq_dist_extractor",
            config={
                "n_grams": 1,
                "lemmatize": False,
                "lowercase": True,
                "stop_words": None,
                "drop_punctuation": True,
                "keep_stop_words": True,
            },
        )

    def test_doc_freq_dist_extractor_basic(self):
        doc = self.nlp("The quick brown fox jumps over the lazy dog.")
        expected = {
            "the": 2,
            "quick": 1,
            "brown": 1,
            "fox": 1,
            "jumps": 1,
            "over": 1,
            "lazy": 1,
            "dog": 1,
        }
        self.assertEqual(doc._.freq_dist, expected)

    def test_doc_freq_dist_extractor_filter_stop_words(self):
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe(
            "doc_freq_dist_extractor",
            config={
                "n_grams": 1,
                "lemmatize": False,
                "lowercase": True,
                "stop_words": None,
                "drop_punctuation": True,
                "keep_stop_words": False,
            },
        )
        doc = nlp("The quick brown fox jumps over the lazy dog.")
        expected = {
            "quick": 1,
            "brown": 1,
            "fox": 1,
            "jumps": 1,
            "lazy": 1,
            "dog": 1,
        }
        self.assertEqual(doc._.freq_dist, expected)

    def test_doc_freq_dist_extractor_lemmatize(self):
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe(
            "doc_freq_dist_extractor",
            config={
                "n_grams": 1,
                "lemmatize": True,
                "lowercase": True,
                "stop_words": None,
                "drop_punctuation": True,
                "keep_stop_words": False,
            },
        )
        doc = nlp("The foxes jumped over the dogs.")
        expected = {
            "fox": 1,
            "jump": 1,
            "dog": 1,
        }
        self.assertEqual(doc._.freq_dist, expected)

    def test_doc_freq_dist_extractor_custom_stop_words(self):
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe(
            "doc_freq_dist_extractor",
            config={
                "n_grams": 1,
                "lemmatize": False,
                "lowercase": True,
                "stop_words": ["the", "over"],
                "drop_punctuation": True,
                "keep_stop_words": False,
            },
        )
        doc = nlp("The quick brown fox jumps over the lazy dog.")
        expected = {
            "quick": 1,
            "brown": 1,
            "fox": 1,
            "jumps": 1,
            "lazy": 1,
            "dog": 1,
        }
        self.assertEqual(doc._.freq_dist, expected)

    def test_doc_freq_dist_extractor_keep_punctuation(self):
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe(
            "doc_freq_dist_extractor",
            config={
                "n_grams": 1,
                "lemmatize": False,
                "lowercase": True,
                "stop_words": None,
                "drop_punctuation": False,
                "keep_stop_words": False,
            },
        )
        doc = nlp("The quick brown fox jumps over the lazy dog.")
        expected = {
            "quick": 1,
            "brown": 1,
            "fox": 1,
            "jumps": 1,
            "lazy": 1,
            "dog": 1,
            ".": 1,
        }
        self.assertEqual(doc._.freq_dist, expected)

    def test_doc_freq_dist_extractor_case_sensitive(self):
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe(
            "doc_freq_dist_extractor",
            config={
                "n_grams": 1,
                "lemmatize": False,
                "lowercase": False,
                "stop_words": None,
                "drop_punctuation": True,
                "keep_stop_words": False,
            },
        )
        doc = nlp("The Quick Brown Fox jumps over the Lazy Dog.")
        expected = {
            "Quick": 1,
            "Brown": 1,
            "Fox": 1,
            "jumps": 1,
            "Lazy": 1,
            "Dog": 1,
        }
        self.assertEqual(doc._.freq_dist, expected)

    def test_doc_freq_dist_extractor_n_grams(self):
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe(
            "doc_freq_dist_extractor",
            config={
                "n_grams": 2,
                "lemmatize": False,
                "lowercase": True,
                "stop_words": None,
                "drop_punctuation": True,
                "keep_stop_words": False,
            },
        )
        doc = nlp("The quick brown fox jumps over the lazy dog.")
        expected = {
            ("quick", "brown"): 1,
            ("brown", "fox"): 1,
            ("fox", "jumps"): 1,
            ("jumps", "lazy"): 1,
            ("lazy", "dog"): 1,
        }
        self.assertEqual(doc._.freq_dist, expected)

    def test_doc_freq_dist_extractor_as_frequencies(self):
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe(
            "doc_freq_dist_extractor",
            config={
                "n_grams": 1,
                "lemmatize": False,
                "lowercase": True,
                "stop_words": None,
                "drop_punctuation": True,
                "keep_stop_words": False,
                "as_frequencies": True,
            },
        )
        doc = nlp("The quick brown fox jumps over the lazy dog.")
        expected = {
            "quick": 1 / 6,
            "brown": 1 / 6,
            "fox": 1 / 6,
            "jumps": 1 / 6,
            "lazy": 1 / 6,
            "dog": 1 / 6,
        }
        self.assertEqual(doc._.freq_dist, expected)

    def test_doc_freq_dist_extractor_as_frequencies_with_stop_words(self):
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe(
            "doc_freq_dist_extractor",
            config={
                "n_grams": 1,
                "lemmatize": False,
                "lowercase": True,
                "stop_words": None,
                "drop_punctuation": True,
                "keep_stop_words": True,
                "as_frequencies": True,
            },
        )
        doc = nlp("The quick brown fox jumps over the lazy dog.")
        expected = {
            "the": 2 / 9,
            "quick": 1 / 9,
            "brown": 1 / 9,
            "fox": 1 / 9,
            "jumps": 1 / 9,
            "over": 1 / 9,
            "lazy": 1 / 9,
            "dog": 1 / 9,
        }
        self.assertEqual(doc._.freq_dist, expected)

    def test_doc_freq_dist_extractor_as_frequencies_n_grams(self):
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe(
            "doc_freq_dist_extractor",
            config={
                "n_grams": 2,
                "lemmatize": False,
                "lowercase": True,
                "stop_words": None,
                "drop_punctuation": True,
                "keep_stop_words": False,
                "as_frequencies": True,
            },
        )
        doc = nlp("The quick brown fox jumps over the lazy dog.")
        expected = {
            ("quick", "brown"): 1 / 5,
            ("brown", "fox"): 1 / 5,
            ("fox", "jumps"): 1 / 5,
            ("jumps", "lazy"): 1 / 5,
            ("lazy", "dog"): 1 / 5,
        }
        self.assertEqual(doc._.freq_dist, expected)


if __name__ == "__main__":
    unittest.main()
