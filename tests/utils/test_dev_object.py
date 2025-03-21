import json
import tempfile
import unittest
from pathlib import Path
from threading import Thread
from unittest import TestCase

from mitoolspro.utils import (
    Dev,
    get_dev_var,
    store_dev_var,
)


class TestDev(TestCase):
    def setUp(self):
        self.dev = Dev()  # Reference the singleton instance
        self.dev.clear_vars()  # Clear all variables before each test
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_file = Path(self.temp_dir.name) / "dev_vars.json"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_singleton_instance(self):
        dev1 = Dev()
        dev2 = Dev()
        self.assertIs(dev1, dev2)  # Both instances should point to the same object

    def test_store_and_get_var(self):
        self.dev.store_var("test_key", 123)
        self.assertEqual(self.dev.get_var("test_key"), 123)
        with self.assertRaises(KeyError):
            self.dev.get_var("non_existent_key")  # Key should not exist

    def test_store_var_invalid_key(self):
        with self.assertRaises(ValueError):
            self.dev.store_var(123, "value")  # Key must be a string

    def test_delete_var(self):
        self.dev.store_var("test_key", 123)
        self.dev.delete_var("test_key")
        with self.assertRaises(KeyError):
            self.dev.delete_var("test_key")  # Key should not exist

    def test_clear_vars(self):
        self.dev.store_var("key1", "value1")
        self.dev.store_var("key2", "value2")
        self.dev.clear_vars()
        self.assertEqual(len(self.dev.variables), 0)  # No variables should remain

    def test_save_variables(self):
        self.dev.store_var("key1", "value1")
        self.dev.save_variables(self.test_file)
        with open(self.test_file) as f:
            dev_vars = json.load(f)
        dev_vars["key1"] == "value1"

    def test_load_variables(self):
        self.dev.store_var("key1", "value1")
        self.dev.save_variables(self.test_file)
        self.dev.load_variables(self.test_file)
        self.assertEqual(self.dev.get_var("key1"), "value1")

    def test_load_variables_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            self.dev.load_variables("non_existent_file.json")

    def test_store_var_using_global_function(self):
        store_dev_var("global_key", "global_value")
        self.assertEqual(get_dev_var("global_key"), "global_value")

    def test_thread_safety(self):
        def store_in_thread(dev, key, value):
            dev.store_var(key, value)

        thread1 = Thread(target=store_in_thread, args=(self.dev, "key1", "value1"))
        thread2 = Thread(target=store_in_thread, args=(self.dev, "key2", "value2"))
        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()
        # Both keys should exist in the dictionary
        self.assertEqual(self.dev.get_var("key1"), "value1")
        self.assertEqual(self.dev.get_var("key2"), "value2")

    def test_save_empty_variables(self):
        self.dev.clear_vars()
        with self.assertRaises(ValueError):
            self.dev.save_variables(self.test_file)

    def test_dictionary_access(self):
        self.dev["test_key"] = 123
        self.assertEqual(self.dev["test_key"], 123)

        with self.assertRaises(KeyError):
            _ = self.dev["non_existent"]

    def test_dictionary_deletion(self):
        self.dev["test_key"] = 123
        del self.dev["test_key"]
        with self.assertRaises(KeyError):
            self.dev.get_var("test_key")
        with self.assertRaises(KeyError):
            del self.dev["non_existent"]

    def test_iteration(self):
        test_data = {"key1": "value1", "key2": "value2", "key3": "value3"}
        for key, value in test_data.items():
            self.dev[key] = value
        keys = list(self.dev)
        self.assertEqual(set(keys), set(test_data.keys()))
        dev_items = dict(self.dev.items())
        self.assertEqual(dev_items, test_data)
        self.assertEqual(len(self.dev), len(test_data))


if __name__ == "__main__":
    unittest.main()
