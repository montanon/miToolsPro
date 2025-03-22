import unittest
from unittest import TestCase

import numpy as np

from mitoolspro.economic_complexity.objects import Product, ProductsBasket


class TestProduct(TestCase):
    def setUp(self):
        self.product = Product(code=1234, name="Test Product", pci=0.5, value=100.0)

    def test_product_attributes(self):
        self.assertEqual(self.product.code, 1234)
        self.assertEqual(self.product.name, "Test Product")
        self.assertEqual(self.product.pci, 0.5)
        self.assertEqual(self.product.value, 100.0)


class TestProductsBasket(TestCase):
    def setUp(self):
        self.products = [
            Product(code=1, name="Product 1", pci=0.5, value=100),
            Product(code=2, name="Product 2", pci=1.0, value=200),
            Product(code=3, name="Product 3", pci=1.5, value=300),
            Product(code=4, name="Product 4", pci=2.0, value=400),
            Product(code=5, name="Product 5", pci=2.5, value=500),
        ]
        self.basket = ProductsBasket(products=self.products)

    def test_duplicate_code_raises_error(self):
        duplicate_products = [
            Product(code=1, name="Product 1", pci=0.5, value=100),
            Product(code=1, name="Product 2", pci=1.0, value=200),
        ]
        with self.assertRaises(ValueError):
            ProductsBasket(products=duplicate_products)

    def test_duplicate_name_raises_error(self):
        duplicate_products = [
            Product(code=1, name="Same Name", pci=0.5, value=100),
            Product(code=2, name="Same Name", pci=1.0, value=200),
        ]
        with self.assertRaises(ValueError):
            ProductsBasket(products=duplicate_products)

    def test_statistics(self):
        self.assertEqual(self.basket.mean, 1.5)
        self.assertAlmostEqual(self.basket.std, 0.7905694150420949)
        self.assertEqual(self.basket.minimum, 0.5)
        self.assertEqual(self.basket.maximum, 2.5)
        self.assertEqual(self.basket.median, 1.5)

    def test_range(self):
        expected_range = {
            "min": 0.5,
            "mean": 1.5,
            "median": 1.5,
            "max": 2.5,
        }
        self.assertEqual(self.basket.range, expected_range)

    def test_len(self):
        self.assertEqual(len(self.basket), 5)

    def test_get_quantiles(self):
        quantiles = self.basket.get_quantiles(n=4)
        expected_quantiles = [0.5, 1.0, 1.5, 2.0, 2.5]
        np.testing.assert_array_almost_equal(quantiles, expected_quantiles)

    def test_get_quantiles_empty(self):
        empty_basket = ProductsBasket(products=[])
        self.assertEqual(empty_basket.get_quantiles(n=4), [])

    def test_products_closest_to_quantiles(self):
        closest_products = self.basket.products_closest_to_quantiles(n=4)
        expected_pcis = [0.5, 1.0, 1.5, 2.0, 2.5]
        actual_pcis = [product.pci for product in closest_products]
        self.assertEqual(actual_pcis, expected_pcis)

    def test_product_list(self):
        df = self.basket.product_list()
        self.assertEqual(len(df), 5)
        self.assertEqual(list(df.columns), ["Code", "Name", "PCI", "Value"])
        self.assertEqual(df["Code"].tolist(), [1, 2, 3, 4, 5])


if __name__ == "__main__":
    unittest.main()
