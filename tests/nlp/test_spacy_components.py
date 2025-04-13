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
            "quick brown": 1,
            "brown fox": 1,
            "fox jumps": 1,
            "jumps lazy": 1,
            "lazy dog": 1,
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
            "quick brown": 1 / 5,
            "brown fox": 1 / 5,
            "fox jumps": 1 / 5,
            "jumps lazy": 1 / 5,
            "lazy dog": 1 / 5,
        }
        self.assertEqual(doc._.freq_dist, expected)

    def test_doc_freq_dist_extractor_multiple_n_grams(self):
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe(
            "doc_freq_dist_extractor",
            config={
                "n_grams": [1, 2, 3],
                "lemmatize": False,
                "lowercase": True,
                "stop_words": None,
                "drop_punctuation": True,
                "keep_stop_words": False,
                "as_frequencies": False,
            },
        )
        doc = nlp("The quick brown fox jumps over the lazy dog.")
        expected = {
            1: {
                "quick": 1,
                "brown": 1,
                "fox": 1,
                "jumps": 1,
                "lazy": 1,
                "dog": 1,
            },
            2: {
                "quick brown": 1,
                "brown fox": 1,
                "fox jumps": 1,
                "jumps lazy": 1,
                "lazy dog": 1,
            },
            3: {
                "quick brown fox": 1,
                "brown fox jumps": 1,
                "fox jumps lazy": 1,
                "jumps lazy dog": 1,
            },
        }
        self.assertEqual(doc._.freq_dist, expected)

    def test_doc_freq_dist_extractor_multiple_n_grams_with_stopwords(self):
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe(
            "doc_freq_dist_extractor",
            config={
                "n_grams": [1, 2],
                "lemmatize": False,
                "lowercase": True,
                "stop_words": None,
                "drop_punctuation": True,
                "keep_stop_words": True,
                "as_frequencies": False,
            },
        )
        doc = nlp("The quick brown fox jumps over the lazy dog.")
        expected = {
            1: {
                "the": 2,
                "quick": 1,
                "brown": 1,
                "fox": 1,
                "jumps": 1,
                "over": 1,
                "lazy": 1,
                "dog": 1,
            },
            2: {
                "the quick": 1,
                "quick brown": 1,
                "brown fox": 1,
                "fox jumps": 1,
                "jumps over": 1,
                "over the": 1,
                "the lazy": 1,
                "lazy dog": 1,
            },
        }
        self.assertEqual(doc._.freq_dist, expected)

    def test_doc_freq_dist_extractor_multiple_n_grams_as_frequencies(self):
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe(
            "doc_freq_dist_extractor",
            config={
                "n_grams": [1, 2],
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
            1: {
                "quick": 1 / 6,
                "brown": 1 / 6,
                "fox": 1 / 6,
                "jumps": 1 / 6,
                "lazy": 1 / 6,
                "dog": 1 / 6,
            },
            2: {
                "quick brown": 1 / 5,
                "brown fox": 1 / 5,
                "fox jumps": 1 / 5,
                "jumps lazy": 1 / 5,
                "lazy dog": 1 / 5,
            },
        }
        self.assertEqual(doc._.freq_dist, expected)

    def test_doc_freq_dist_extractor_multiple_n_grams_empty_doc(self):
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe(
            "doc_freq_dist_extractor",
            config={
                "n_grams": [1, 2, 3],
                "lemmatize": False,
                "lowercase": True,
                "stop_words": None,
                "drop_punctuation": True,
                "keep_stop_words": False,
                "as_frequencies": False,
            },
        )
        doc = nlp("")
        expected = {1: {}, 2: {}, 3: {}}
        self.assertEqual(doc._.freq_dist, expected)

    def test_doc_freq_dist_extractor_multiple_n_grams_short_doc(self):
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe(
            "doc_freq_dist_extractor",
            config={
                "n_grams": [1, 2, 3],
                "lemmatize": False,
                "lowercase": True,
                "stop_words": None,
                "drop_punctuation": True,
                "keep_stop_words": False,
                "as_frequencies": False,
            },
        )
        doc = nlp("Hello world")
        expected = {
            1: {"hello": 1, "world": 1},
            2: {"hello world": 1},
            3: {},
        }
        self.assertEqual(doc._.freq_dist, expected)


class TestDocTokenExtractor(unittest.TestCase):
    def setUp(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.add_pipe(
            "doc_token_extractor",
            config={
                "attribute": "lower_",
                "n_grams": 1,
                "keep_stop_words": True,
                "drop_punctuation": True,
                "lowercase": True,
            },
        )

    def test_doc_token_extractor_basic(self):
        doc = self.nlp("The quick brown fox jumps over the lazy dog.")
        expected = [
            "the",
            "quick",
            "brown",
            "fox",
            "jumps",
            "over",
            "the",
            "lazy",
            "dog",
        ]
        self.assertEqual(doc._.tokens, expected)

    def test_doc_token_extractor_drop_stopwords(self):
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe(
            "doc_token_extractor",
            config={
                "attribute": "lower_",
                "n_grams": 1,
                "keep_stop_words": False,
                "drop_punctuation": True,
                "lowercase": True,
            },
        )
        doc = nlp("The quick brown fox jumps over the lazy dog.")
        expected = ["quick", "brown", "fox", "jumps", "lazy", "dog"]
        self.assertEqual(doc._.tokens, expected)

    def test_doc_token_extractor_keep_punctuation(self):
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe(
            "doc_token_extractor",
            config={
                "attribute": "lower_",
                "n_grams": 1,
                "keep_stop_words": True,
                "drop_punctuation": False,
                "lowercase": True,
            },
        )
        doc = nlp("The quick brown fox jumps over the lazy dog.")
        expected = [
            "the",
            "quick",
            "brown",
            "fox",
            "jumps",
            "over",
            "the",
            "lazy",
            "dog",
            ".",
        ]
        self.assertEqual(doc._.tokens, expected)

    def test_doc_token_extractor_case_sensitive(self):
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe(
            "doc_token_extractor",
            config={
                "attribute": "text",
                "n_grams": 1,
                "keep_stop_words": True,
                "drop_punctuation": True,
                "lowercase": False,
            },
        )
        doc = nlp("The Quick Brown Fox jumps over the Lazy Dog.")
        expected = [
            "The",
            "Quick",
            "Brown",
            "Fox",
            "jumps",
            "over",
            "the",
            "Lazy",
            "Dog",
        ]
        self.assertEqual(doc._.tokens, expected)

    def test_doc_token_extractor_n_grams(self):
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe(
            "doc_token_extractor",
            config={
                "attribute": "lower_",
                "n_grams": 2,
                "keep_stop_words": True,
                "drop_punctuation": True,
                "lowercase": True,
            },
        )
        doc = nlp("The quick brown fox jumps over the lazy dog.")
        expected = [
            ("the", "quick"),
            ("quick", "brown"),
            ("brown", "fox"),
            ("fox", "jumps"),
            ("jumps", "over"),
            ("over", "the"),
            ("the", "lazy"),
            ("lazy", "dog"),
        ]
        self.assertEqual(doc._.tokens, expected)

    def test_doc_token_extractor_custom_attribute(self):
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe(
            "doc_token_extractor",
            config={
                "attribute": "lemma_",
                "n_grams": 1,
                "keep_stop_words": True,
                "drop_punctuation": True,
                "lowercase": True,
            },
        )
        doc = nlp("The foxes jumped over the dogs.")
        expected = ["the", "fox", "jump", "over", "the", "dog"]
        self.assertEqual(doc._.tokens, expected)

    def test_doc_token_extractor_empty_doc(self):
        doc = self.nlp("")
        self.assertEqual(doc._.tokens, [])

    def test_doc_token_extractor_single_token(self):
        doc = self.nlp("Hello")
        self.assertEqual(doc._.tokens, ["hello"])

    def test_doc_token_extractor_multiple_n_grams(self):
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe(
            "doc_token_extractor",
            config={
                "attribute": "lower_",
                "n_grams": [1, 2, 3],
                "keep_stop_words": True,
                "drop_punctuation": True,
                "lowercase": True,
            },
        )
        doc = nlp("The quick brown fox jumps over the lazy dog.")
        expected = {
            1: ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"],
            2: [
                ("the", "quick"),
                ("quick", "brown"),
                ("brown", "fox"),
                ("fox", "jumps"),
                ("jumps", "over"),
                ("over", "the"),
                ("the", "lazy"),
                ("lazy", "dog"),
            ],
            3: [
                ("the", "quick", "brown"),
                ("quick", "brown", "fox"),
                ("brown", "fox", "jumps"),
                ("fox", "jumps", "over"),
                ("jumps", "over", "the"),
                ("over", "the", "lazy"),
                ("the", "lazy", "dog"),
            ],
        }
        self.assertEqual(doc._.tokens, expected)

    def test_doc_token_extractor_multiple_n_grams_with_stopwords(self):
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe(
            "doc_token_extractor",
            config={
                "attribute": "lower_",
                "n_grams": [1, 2],
                "keep_stop_words": False,
                "drop_punctuation": True,
                "lowercase": True,
            },
        )
        doc = nlp("The quick brown fox jumps over the lazy dog.")
        expected = {
            1: ["quick", "brown", "fox", "jumps", "lazy", "dog"],
            2: [
                ("quick", "brown"),
                ("brown", "fox"),
                ("fox", "jumps"),
                ("jumps", "lazy"),
                ("lazy", "dog"),
            ],
        }
        self.assertEqual(doc._.tokens, expected)

    def test_doc_token_extractor_multiple_n_grams_with_punctuation(self):
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe(
            "doc_token_extractor",
            config={
                "attribute": "lower_",
                "n_grams": [1, 2],
                "keep_stop_words": True,
                "drop_punctuation": False,
                "lowercase": True,
            },
        )
        doc = nlp("The quick brown fox jumps over the lazy dog.")
        expected = {
            1: [
                "the",
                "quick",
                "brown",
                "fox",
                "jumps",
                "over",
                "the",
                "lazy",
                "dog",
                ".",
            ],
            2: [
                ("the", "quick"),
                ("quick", "brown"),
                ("brown", "fox"),
                ("fox", "jumps"),
                ("jumps", "over"),
                ("over", "the"),
                ("the", "lazy"),
                ("lazy", "dog"),
                ("dog", "."),
            ],
        }
        self.assertEqual(doc._.tokens, expected)

    def test_doc_token_extractor_multiple_n_grams_custom_attribute(self):
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe(
            "doc_token_extractor",
            config={
                "attribute": "lemma_",
                "n_grams": [1, 2],
                "keep_stop_words": True,
                "drop_punctuation": True,
                "lowercase": True,
            },
        )
        doc = nlp("The foxes jumped over the dogs.")
        expected = {
            1: ["the", "fox", "jump", "over", "the", "dog"],
            2: [
                ("the", "fox"),
                ("fox", "jump"),
                ("jump", "over"),
                ("over", "the"),
                ("the", "dog"),
            ],
        }
        self.assertEqual(doc._.tokens, expected)

    def test_doc_token_extractor_multiple_n_grams_empty_doc(self):
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe(
            "doc_token_extractor",
            config={
                "attribute": "lower_",
                "n_grams": [1, 2, 3],
                "keep_stop_words": True,
                "drop_punctuation": True,
                "lowercase": True,
            },
        )
        doc = nlp("")
        expected = {1: [], 2: [], 3: []}
        self.assertEqual(doc._.tokens, expected)

    def test_doc_token_extractor_multiple_n_grams_short_doc(self):
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe(
            "doc_token_extractor",
            config={
                "attribute": "lower_",
                "n_grams": [1, 2, 3],
                "keep_stop_words": True,
                "drop_punctuation": True,
                "lowercase": True,
            },
        )
        doc = nlp("Hello world")
        expected = {
            1: ["hello", "world"],
            2: [("hello", "world")],
            3: [],
        }
        self.assertEqual(doc._.tokens, expected)


class TestPipelineFunctionality(unittest.TestCase):
    def setUp(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.texts = [
            "The quick brown fox jumps over the lazy dog.",
            "I like apples and coffee.",
            "Contact me at test@example.com or 123-456-7890.",
        ]

    def test_strip_accents_pipeline(self):
        nlp = spacy.blank("en")
        nlp.add_pipe("strip_accents")
        docs = list(nlp.pipe(self.texts))
        self.assertEqual(len(docs), 3)
        self.assertEqual(docs[0].text, "The quick brown fox jumps over the lazy dog.")

    def test_sentence_lemma_tagger_pipeline(self):
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe(
            "sentence_lemma_tagger",
            config={
                "categories": {"food": ["apple"], "drink": ["coffee"]},
                "strip_accents": True,
                "ignore_case": True,
            },
        )
        docs = list(nlp.pipe(self.texts))
        self.assertEqual(len(docs), 3)
        self.assertTrue(docs[1][2].sent._.food)
        self.assertTrue(docs[1][4].sent._.drink)

    def test_doc_lemma_tagger_pipeline(self):
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe(
            "doc_lemma_tagger",
            config={
                "categories": {"food": ["apple"], "drink": ["coffee"]},
                "strip_accents": True,
                "ignore_case": True,
            },
        )
        docs = list(nlp.pipe(self.texts))
        self.assertEqual(len(docs), 3)
        self.assertTrue(docs[1]._.food)
        self.assertTrue(docs[1]._.drink)

    def test_sentence_word_tagger_pipeline(self):
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe(
            "sentence_word_tagger",
            config={
                "categories": {"food": ["apples"], "drink": ["coffee"]},
                "strip_accents": True,
                "ignore_case": True,
            },
        )
        docs = list(nlp.pipe(self.texts))
        self.assertEqual(len(docs), 3)
        self.assertTrue(docs[1][2].sent._.food)
        self.assertTrue(docs[1][4].sent._.drink)

    def test_doc_word_tagger_pipeline(self):
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe(
            "doc_word_tagger",
            config={
                "categories": {"food": ["apples"], "drink": ["coffee"]},
                "strip_accents": True,
                "ignore_case": True,
            },
        )
        docs = list(nlp.pipe(self.texts))
        self.assertEqual(len(docs), 3)
        self.assertTrue(docs[1]._.food)
        self.assertTrue(docs[1]._.drink)

    def test_sentence_regex_tagger_pipeline(self):
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe(
            "sentence_regex_tagger",
            config={
                "categories": {
                    "email": [r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"],
                    "phone": [r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"],
                },
                "ignore_case": True,
                "strip_accents": True,
            },
        )
        docs = list(nlp.pipe(self.texts))
        self.assertEqual(len(docs), 3)
        self.assertTrue(docs[2][4].sent._.email)
        self.assertTrue(docs[2][6].sent._.phone)

    def test_doc_regex_tagger_pipeline(self):
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe(
            "doc_regex_tagger",
            config={
                "categories": {
                    "email": [r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"],
                    "phone": [r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"],
                },
                "ignore_case": True,
                "strip_accents": True,
            },
        )
        docs = list(nlp.pipe(self.texts))
        self.assertEqual(len(docs), 3)
        self.assertTrue(docs[2]._.email)
        self.assertTrue(docs[2]._.phone)

    def test_doc_bow_extractor_pipeline(self):
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe(
            "doc_bow_extractor",
            config={
                "lemmatize": False,
                "lowercase": True,
                "stop_words": None,
                "drop_punctuation": True,
                "keep_stop_words": True,
            },
        )
        docs = list(nlp.pipe(self.texts))
        self.assertEqual(len(docs), 3)
        self.assertIn("the", docs[0]._.bow)
        self.assertEqual(docs[0]._.bow["the"], 2)

    def test_doc_freq_dist_extractor_pipeline(self):
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
            },
        )
        docs = list(nlp.pipe(self.texts))
        self.assertEqual(len(docs), 3)
        self.assertIn("the", docs[0]._.freq_dist)
        self.assertEqual(docs[0]._.freq_dist["the"], 2)

    def test_doc_freq_dist_extractor_pipeline_multi_n_grams(self):
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe(
            "doc_freq_dist_extractor",
            config={
                "n_grams": [1, 2],
                "lemmatize": False,
                "lowercase": True,
                "stop_words": None,
                "drop_punctuation": True,
                "keep_stop_words": True,
            },
        )
        docs = list(nlp.pipe(self.texts))
        self.assertEqual(len(docs), 3)
        self.assertIn("the", docs[0]._.freq_dist[1])
        self.assertEqual(docs[0]._.freq_dist[1]["the"], 2)
        self.assertIn("the quick", docs[0]._.freq_dist[2])
        self.assertEqual(docs[0]._.freq_dist[2]["the quick"], 1)

    def test_doc_token_extractor_pipeline(self):
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe(
            "doc_token_extractor",
            config={
                "attribute": "lower_",
                "n_grams": 1,
                "keep_stop_words": True,
                "drop_punctuation": True,
                "lowercase": True,
            },
        )
        docs = list(nlp.pipe(self.texts))
        self.assertEqual(len(docs), 3)
        self.assertIn("the", docs[0]._.tokens)
        self.assertEqual(docs[0]._.tokens.count("the"), 2)

    def test_doc_token_extractor_pipeline_multi_n_grams(self):
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe(
            "doc_token_extractor",
            config={
                "attribute": "lower_",
                "n_grams": [1, 2],
                "keep_stop_words": True,
                "drop_punctuation": True,
                "lowercase": True,
            },
        )
        docs = list(nlp.pipe(self.texts))
        self.assertEqual(len(docs), 3)
        self.assertIn("the", docs[0]._.tokens[1])
        self.assertEqual(docs[0]._.tokens[1].count("the"), 2)
        self.assertIn("the quick", docs[0]._.tokens[2])
        self.assertEqual(docs[0]._.tokens[2].count("the quick"), 1)


if __name__ == "__main__":
    unittest.main()
