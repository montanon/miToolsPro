import unittest

from mitoolspro.document.fonts.fonts import FONT_FILES, FONT_MAPPING, select_font


class TestFonts(unittest.TestCase):
    def setUp(self):
        self.fontfamily = "arial"

    def test_select_font_normal(self):
        self.assertEqual(
            select_font(self.fontfamily, "Arial"),
            FONT_MAPPING[self.fontfamily]["normal"],
        )

    def test_select_font_bold(self):
        self.assertEqual(
            select_font(self.fontfamily, "Arial Bold"),
            FONT_MAPPING[self.fontfamily]["bold"],
        )

    def test_select_font_italic(self):
        test_cases = ["Arial Italic", "Arial Oblique"]
        for case in test_cases:
            self.assertEqual(
                select_font(self.fontfamily, case),
                FONT_MAPPING[self.fontfamily]["italic"],
            )

    def test_select_font_bold_italic(self):
        test_cases = ["Arial Bold Italic", "Arial Bold Oblique"]
        for case in test_cases:
            self.assertEqual(
                select_font(self.fontfamily, case),
                FONT_MAPPING[self.fontfamily]["bold-italic"],
            )

    def test_font_mappings_consistency(self):
        self.assertEqual(
            set(FONT_FILES[self.fontfamily].keys()),
            set(FONT_MAPPING[self.fontfamily].keys()),
            "Font files and mappings should have the same styles",
        )


if __name__ == "__main__":
    unittest.main()
