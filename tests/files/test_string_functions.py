import unittest
from unittest import TestCase

from thefuzz import fuzz

from mitoolspro.utils.string_functions import (
    clean_str,
    decode_string,
    encode_string,
    find_str_line_number_in_text,
    fuzz_ratio,
    fuzz_string_in_string,
    get_numbers_from_str,
    lcs_similarity,
    lowerstrip,
    remove_characters_from_string,
    remove_characters_from_strings,
    remove_chars,
    remove_multiple_spaces,
    replace_prefix,
    split_strings,
    str_is_number,
    stretch_string,
    strip_punctuation,
)


class TestStringEncodingDecoding(TestCase):
    def test_decode_string_valid_utf8_bytes(self):
        self.assertEqual(decode_string(b"hello"), "hello")

    def test_decode_string_valid_windows_1252_bytes(self):
        self.assertEqual(decode_string(b"\xe9", encoding="windows-1252"), "é")

    def test_decode_string_utf8_fallback(self):
        self.assertEqual(decode_string(b"\xe9"), "é")  # Decoded as utf-8

    def test_decode_string_invalid_encoding(self):
        result = decode_string(b"\xff\xff", encoding="ascii")
        self.assertEqual(result, "ÿÿ")  # Returns original bytes

    def test_decode_string_str_input(self):
        self.assertEqual(decode_string("already a string"), "already a string")

    def test_non_existent_encoding(self):
        self.assertEqual(decode_string(b"\x81", encoding="windows-1252"), "")

    def test_decode_string_invalid_encoding_type(self):
        with self.assertRaises(TypeError):
            decode_string(b"test", encoding=123)

    def test_encode_string_valid_utf8(self):
        self.assertEqual(encode_string("hello"), b"hello")

    def test_encode_string_valid_windows_1252(self):
        self.assertEqual(encode_string("é", encoding="windows-1252"), b"\xe9")

    def test_encode_string_utf8_fallback(self):
        self.assertEqual(encode_string("€", encoding="ascii"), b"\x80")

    def test_encode_string_invalid_encoding(self):
        result = encode_string("€", encoding="ascii")
        self.assertEqual(
            result, encode_string("€", encoding="windows-1252")
        )  # Returns utf-8 encoded value

    def test_encode_string_bytes_input(self):
        self.assertEqual(encode_string(b"already bytes"), "b'already bytes'")

    def test_encode_string_invalid_encoding_type(self):
        with self.assertRaises(TypeError):
            encode_string("test", encoding=123)

    def test_encode_string_non_utf8_str(self):
        self.assertEqual(encode_string("\x81", encoding="windows-1252"), b"\xc2\x81")

    def test_encode_string_invalid_input_type(self):
        self.assertEqual(encode_string(123), "123")


class TestStripPunctuation(TestCase):
    def test_strip_punctuation_all_true(self):
        self.assertEqual(strip_punctuation("!Hello, World!", all=True), "Hello World")
        self.assertEqual(
            strip_punctuation("...This, is! a test?", all=True), "This is a test"
        )
        self.assertEqual(
            strip_punctuation("No-punctuation-here.", all=True), "Nopunctuationhere"
        )
        self.assertEqual(strip_punctuation("!!!", all=True), "")
        self.assertEqual(strip_punctuation("", all=True), "")
        self.assertEqual(strip_punctuation("1234!@#$", all=True), "1234")

    def test_strip_punctuation_all_false(self):
        self.assertEqual(strip_punctuation("!Hello, World!", all=False), "Hello, World")
        self.assertEqual(
            strip_punctuation("...This, is! a test?", all=False), "This, is! a test"
        )
        self.assertEqual(
            strip_punctuation("No-punctuation-here.", all=False), "No-punctuation-here"
        )
        self.assertEqual(strip_punctuation("!!!", all=False), "")
        self.assertEqual(strip_punctuation("", all=False), "")
        self.assertEqual(strip_punctuation("1234!@#$", all=False), "1234")

    def test_strip_punctuation_whitespace_handling(self):
        self.assertEqual(
            strip_punctuation("   !Hello, World!   ", all=True), "Hello World"
        )
        self.assertEqual(
            strip_punctuation("   ...This, is! a test?   ", all=False),
            "This, is! a test",
        )
        self.assertEqual(
            strip_punctuation("   No-punctuation-here.   ", all=False),
            "No-punctuation-here",
        )

    def test_strip_punctuation_unicode_characters(self):
        self.assertEqual(strip_punctuation("¡Hola, Mundo!", all=True), "¡Hola Mundo")
        self.assertEqual(strip_punctuation("¡Hola, Mundo!", all=False), "¡Hola, Mundo")
        self.assertEqual(
            strip_punctuation("   ¡Hola, Mundo!   ", all=True), "¡Hola Mundo"
        )
        self.assertEqual(
            strip_punctuation("   ¡Hola, Mundo!   ", all=False), "¡Hola, Mundo"
        )

    def test_strip_punctuation_non_ascii(self):
        self.assertEqual(strip_punctuation("Hello ;,'World'!", all=True), "Hello World")
        self.assertEqual(strip_punctuation("Hello 'World'!", all=False), "Hello 'World")
        self.assertEqual(
            strip_punctuation("   Hello 'World'!   ", all=True), "Hello World"
        )
        self.assertEqual(
            strip_punctuation("   Hello 'World'!   ", all=False), "Hello 'World"
        )

    def test_strip_punctuation_numbers_and_special_cases(self):
        self.assertEqual(strip_punctuation("12345", all=True), "12345")
        self.assertEqual(strip_punctuation("12345", all=False), "12345")
        self.assertEqual(strip_punctuation("12345!!!", all=True), "12345")
        self.assertEqual(strip_punctuation("!!!12345", all=True), "12345")
        self.assertEqual(strip_punctuation("12345!!!", all=False), "12345")
        self.assertEqual(strip_punctuation("!!!12345", all=False), "12345")

    def test_invalid_input_type(self):
        with self.assertRaises(AttributeError):
            strip_punctuation(None)
        with self.assertRaises(AttributeError):
            strip_punctuation(12345)
        with self.assertRaises(AttributeError):
            strip_punctuation(["Hello, World!"], all=True)


class TestStrIsNumber(TestCase):
    def test_integer_string(self):
        self.assertTrue(str_is_number("123"))

    def test_float_string(self):
        self.assertTrue(str_is_number("123.456"))

    def test_negative_integer_string(self):
        self.assertTrue(str_is_number("-123"))

    def test_negative_float_string(self):
        self.assertTrue(str_is_number("-123.456"))

    def test_non_numeric_string(self):
        self.assertFalse(str_is_number("abc"))

    def test_empty_string(self):
        self.assertFalse(str_is_number(""))


class TestGetNumbersFromStr(TestCase):
    def test_integer_string(self):
        string = "abc 123 def 456"
        self.assertEqual(get_numbers_from_str(string), [123.0, 456.0])

    def test_float_string(self):
        string = "abc 123.456 def 789.012"
        self.assertEqual(get_numbers_from_str(string), [123.456, 789.012])

    def test_negative_number_string(self):
        string = "abc -123 def -456"
        self.assertEqual(get_numbers_from_str(string), [-123.0, -456.0])

    def test_non_numeric_string(self):
        string = "abc def"
        self.assertEqual(get_numbers_from_str(string), [])

    def test_empty_string(self):
        string = ""
        self.assertEqual(get_numbers_from_str(string), [])

    def test_indexed_return(self):
        string = "abc 123 def 456"
        self.assertEqual(get_numbers_from_str(string, 1), 456.0)


class TestRemoveMultipleSpaces(TestCase):
    def test_multiple_spaces(self):
        string = "abc   def   ghi"
        self.assertEqual(remove_multiple_spaces(string), "abc def ghi")

    def test_tabs(self):
        string = "abc\t\t\tdef\t\t\tghi"
        self.assertEqual(remove_multiple_spaces(string), "abc def ghi")

    def test_newlines(self):
        string = "abc\n\ndef\n\nghi"
        self.assertEqual(remove_multiple_spaces(string), "abc def ghi")

    def test_mixed_whitespace(self):
        string = "abc \t \n def \t \n ghi"
        self.assertEqual(remove_multiple_spaces(string), "abc def ghi")

    def test_no_extra_spaces(self):
        string = "abc def ghi"
        self.assertEqual(remove_multiple_spaces(string), "abc def ghi")


class TestFindStrLineNumberInText(TestCase):
    def test_substring_at_start(self):
        text = "abc\ndef\nghi"
        substring = "abc"
        self.assertEqual(find_str_line_number_in_text(text, substring), 0)

    def test_substring_in_middle(self):
        text = "abc\ndef\nghi"
        substring = "def"
        self.assertEqual(find_str_line_number_in_text(text, substring), 1)

    def test_substring_at_end(self):
        text = "abc\ndef\nghi"
        substring = "ghi"
        self.assertEqual(find_str_line_number_in_text(text, substring), 2)

    def test_substring_not_found(self):
        text = "abc\ndef\nghi"
        substring = "jkl"
        self.assertEqual(find_str_line_number_in_text(text, substring), None)


class TestLcsSimilarity(TestCase):
    def test_identical_strings(self):
        self.assertEqual(lcs_similarity("abc", "abc"), 1.0)

    def test_different_strings(self):
        self.assertEqual(lcs_similarity("abc", "def"), 0.0)

    def test_common_subsequence(self):
        self.assertEqual(lcs_similarity("abc", "adc"), 2 / 3)

    def test_empty_string(self):
        self.assertEqual(lcs_similarity("abc", ""), 0.0)
        self.assertEqual(lcs_similarity("", "abc"), 0.0)
        self.assertEqual(lcs_similarity("", ""), 0.0)


class TestCleanStr(TestCase):
    def test_clean_str_no_pattern(self):
        string = "Hello, World!"
        result = clean_str(string, None)
        self.assertEqual(result, string)

    def test_clean_str_with_pattern(self):
        string = "Hello, World!"
        pattern = ","
        result = clean_str(string, pattern)
        self.assertEqual(result, "Hello World!")

    def test_clean_str_with_pattern_and_sub_char(self):
        string = "Hello, World!"
        pattern = ","
        sub_char = ";"
        result = clean_str(string, pattern, sub_char)
        self.assertEqual(result, "Hello; World!")


class TestStretchString(TestCase):
    def test_normal_case(self):
        self.assertEqual(
            stretch_string("This is a sample string for testing purposes", 10),
            "This is a\nsample\nstring for\ntesting\npurposes",
        )

    def test_no_spaces(self):
        self.assertEqual(
            stretch_string("LongStringWithNoSpaces", 5),
            "LongS\ntring\nWithN\noSpac\nes",
        )

    def test_edge_cases(self):
        self.assertEqual(stretch_string("", 10), "")
        self.assertEqual(stretch_string("Short", 10), "Short")
        self.assertEqual(stretch_string("ExactlyTen", 10), "ExactlyTen")

    def test_whitespace_handling(self):
        self.assertEqual(
            stretch_string("  This   string has  weird spacing ", 10),
            "This\nstring has\nweird\nspacing",
        )

    def test_long_word(self):
        self.assertEqual(
            stretch_string("Supercalifragilisticexpialidocious", 10),
            "Supercalif\nragilistic\nexpialidoc\nious",
        )


class TestRemoveChars(TestCase):
    def test_basic_removal(self):
        self.assertEqual(remove_chars("Hello, World!", "lo"), "He, Wrd!")

    def test_no_chars_to_remove(self):
        self.assertEqual(remove_chars("Hello, World!", ""), "Hello, World!")

    def test_remove_all_characters(self):
        self.assertEqual(remove_chars("Hello, World!", "Helo, Wrd!"), "")

    def test_string_with_no_matching_characters(self):
        self.assertEqual(remove_chars("Hello, World!", "abc"), "Hello, World!")

    def test_empty_string_input(self):
        self.assertEqual(remove_chars("", "abc"), "")

    def test_special_characters(self):
        self.assertEqual(remove_chars("H@#llo, W$rld!", "@#$"), "Hllo, Wrld!")


class TestSplitStrings(TestCase):
    def test_split_strings(self):
        str_list = ["HelloWorld", "GoodByeWorld"]
        self.assertEqual(
            split_strings(str_list), ["Hello", "World", "Good", "Bye", "World"]
        )

    def test_split_strings_no_capital_letters(self):
        str_list = ["hello", "world"]
        self.assertEqual(split_strings(str_list), ["hello", "world"])

    def test_split_strings_empty_list(self):
        str_list = []
        self.assertEqual(split_strings(str_list), [])


class TestReplacePrefix(TestCase):
    def test_replace_prefix(self):
        string = "Hello World"
        prefix = "Hello"
        replacement = "Goodbye"
        self.assertEqual(replace_prefix(string, prefix, replacement), "Goodbye World")

    def test_replace_prefix_no_match(self):
        string = "Hello World"
        prefix = "Goodbye"
        replacement = "Hello"
        self.assertEqual(replace_prefix(string, prefix, replacement), "Hello World")

    def test_replace_prefix_empty_string(self):
        string = ""
        prefix = "Hello"
        replacement = "Goodbye"
        self.assertEqual(replace_prefix(string, prefix, replacement), "")


class TestFuzzStringInString(TestCase):
    def test_fuzz_string_in_string_exact_match(self):
        src_string = "Hello World"
        dst_string = "Hello World"
        self.assertTrue(fuzz_string_in_string(src_string, dst_string))

    def test_fuzz_string_in_string_no_match(self):
        src_string = "Hello World"
        dst_string = "Goodbye World"
        self.assertFalse(fuzz_string_in_string(src_string, dst_string))

    def test_fuzz_string_in_string_partial_match_below_threshold(self):
        src_string = "Hello World"
        dst_string = "Hello"
        self.assertFalse(fuzz_string_in_string(src_string, dst_string, 100))

    def test_fuzz_string_in_string_partial_match_above_threshold(self):
        src_string = "Hello World"
        dst_string = "Hello"
        self.assertTrue(fuzz_string_in_string(src_string, dst_string, 50))


class TestFuzzRatio(TestCase):
    def test_fuzz_ratio_exact_match(self):
        src_string = "Hello World"
        dst_string = "Hello World"
        self.assertEqual(fuzz_ratio(src_string, dst_string), 100)

    def test_fuzz_ratio_no_match(self):
        src_string = "Hello World"
        dst_string = "Goodbye World"
        self.assertEqual(
            fuzz_ratio(src_string, dst_string),
            fuzz.partial_ratio(src_string, dst_string),
        )

    def test_fuzz_ratio_partial_match(self):
        src_string = "Hello World"
        dst_string = "Hello"
        self.assertEqual(
            fuzz_ratio(src_string, dst_string),
            fuzz.partial_ratio(src_string, dst_string),
        )


class TestRemoveCharactersFromString(TestCase):
    def test_remove_characters_with_default_pattern(self):
        self.assertEqual(
            remove_characters_from_string("file/name*with?invalid%chars"),
            "filenamewithinvalidchars",
        )

    def test_remove_characters_with_custom_pattern(self):
        self.assertEqual(
            remove_characters_from_string("hello123world", r"\d"), "helloworld"
        )

    def test_remove_characters_empty_string(self):
        self.assertEqual(remove_characters_from_string(""), "")

    def test_remove_characters_no_matches(self):
        self.assertEqual(remove_characters_from_string("hello world"), "hello world")

    def test_remove_characters_none_pattern(self):
        self.assertEqual(
            remove_characters_from_string("hello world", None), "hello world"
        )


class TestRemoveCharactersFromStrings(TestCase):
    def test_remove_characters_from_strings_with_default_pattern(self):
        input_strings = ["file/name1", "file*name2", "file?name3"]
        expected = ["filename1", "filename2", "filename3"]
        self.assertEqual(list(remove_characters_from_strings(input_strings)), expected)

    def test_remove_characters_from_strings_with_custom_pattern(self):
        input_strings = ["hello123", "world456", "test789"]
        expected = ["hello", "world", "test"]
        self.assertEqual(
            list(remove_characters_from_strings(input_strings, r"\d")), expected
        )

    def test_remove_characters_from_strings_empty_list(self):
        self.assertEqual(list(remove_characters_from_strings([])), [])

    def test_remove_characters_from_strings_no_matches(self):
        input_strings = ["hello", "world", "test"]
        self.assertEqual(
            list(remove_characters_from_strings(input_strings)), input_strings
        )

    def test_remove_characters_from_strings_none_pattern(self):
        input_strings = ["hello", "world", "test"]
        self.assertEqual(
            list(remove_characters_from_strings(input_strings, None)), input_strings
        )


class TestLowerstrip(TestCase):
    def test_lowerstrip_all_true(self):
        self.assertEqual(lowerstrip("  Hello, World!  ", all=True), "hello world")

    def test_lowerstrip_all_false(self):
        self.assertEqual(lowerstrip("  Hello, World!  ", all=False), "hello, world")

    def test_lowerstrip_empty_string(self):
        self.assertEqual(lowerstrip("", all=True), "")
        self.assertEqual(lowerstrip("", all=False), "")

    def test_lowerstrip_whitespace_only(self):
        self.assertEqual(lowerstrip("   ", all=True), "")
        self.assertEqual(lowerstrip("   ", all=False), "")

    def test_lowerstrip_mixed_case(self):
        self.assertEqual(lowerstrip("  HeLLo, WoRLD!  ", all=True), "hello world")
        self.assertEqual(lowerstrip("  HeLLo, WoRLD!  ", all=False), "hello, world")

    def test_lowerstrip_with_numbers(self):
        self.assertEqual(lowerstrip("  Hello123, World!  ", all=True), "hello123 world")
        self.assertEqual(
            lowerstrip("  Hello123, World!  ", all=False), "hello123, world"
        )


if __name__ == "__main__":
    unittest.main()
