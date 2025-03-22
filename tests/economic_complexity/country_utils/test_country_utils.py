import unittest
from unittest import TestCase

import pandas as pd

from mitoolspro.economic_complexity.country_utils.rename import (
    name_converter,
)


class TestNameConverter(TestCase):
    def test_standard_country_conversion(self):
        self.assertEqual(
            name_converter.convert(names="United States", to="ISO3"), "USA"
        )
        self.assertEqual(name_converter.convert(names="France", to="ISO3"), "FRA")
        self.assertEqual(name_converter.convert(names="Germany", to="ISO3"), "DEU")

    def test_custom_country_conversion(self):
        self.assertEqual(name_converter.convert(names="Bonaire", to="ISO3"), "BES")
        self.assertEqual(
            name_converter.convert(names="Netherlands Antilles", to="ISO3"), "ANT"
        )
        self.assertEqual(name_converter.convert(names="Serbia", to="ISO3"), "SER")
        self.assertEqual(name_converter.convert(names="East Timor", to="ISO3"), "TLS")

    def test_multiple_countries_conversion(self):
        countries = ["United States", "Bonaire", "France", "East Timor"]
        expected = ["USA", "BES", "FRA", "TLS"]
        result = name_converter.convert(names=countries, to="ISO3")
        self.assertEqual(result, expected)

    def test_different_output_formats(self):
        self.assertEqual(
            name_converter.convert(names="Bonaire", to="continent"), "America"
        )
        self.assertEqual(
            name_converter.convert(names="Serbia", to="continent"), "Europe"
        )
        self.assertEqual(
            name_converter.convert(names="East Timor", to="continent"), "Asia"
        )

    def test_case_insensitive(self):
        self.assertEqual(name_converter.convert(names="BONAIRE", to="ISO3"), "BES")
        self.assertEqual(name_converter.convert(names="east timor", to="ISO3"), "TLS")

    def test_dataframe_conversion(self):
        df = pd.DataFrame(
            {"country": ["United States", "Bonaire", "France", "East Timor"]}
        )
        result = name_converter.convert(names=df["country"], to="ISO3")
        expected = ["USA", "BES", "FRA", "TLS"]
        self.assertEqual(result, expected)

    def test_invalid_country(self):
        result = name_converter.convert(
            names="NonExistentCountry", to="ISO3", not_found=None
        )
        self.assertEqual(result, "NonExistentCountry")

    def test_invalid_output_format(self):
        with self.assertRaises(KeyError):
            name_converter.convert(names="United States", to="InvalidFormat")


if __name__ == "__main__":
    unittest.main()
