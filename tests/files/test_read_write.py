import json
import pickle
import tempfile
import unittest
from pathlib import Path

from mitoolspro.files.read_write import (
    load_pkl,
    read_json,
    read_text,
    store_pkl,
    write_json,
    write_text,
)


class CustomClass:
    def __init__(self, value):
        self.value = value


class TestReadText(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_read_text(self):
        test_file = self.temp_path / "test.txt"
        test_content = "Hello, World! 🚀"
        test_file.write_text(test_content)
        self.assertEqual(read_text(test_file), test_content)

    def test_read_text_nonexistent(self):
        with self.assertRaises(FileNotFoundError):
            read_text(self.temp_path / "nonexistent.txt")


class TestWriteText(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_write_text(self):
        test_file = self.temp_path / "test.txt"
        test_content = "Hello, World! 🚀"
        write_text(test_content, test_file)
        self.assertEqual(test_file.read_text(), test_content)

    def test_write_text_invalid_encoding(self):
        test_file = self.temp_path / "test.txt"
        with self.assertRaises(LookupError):
            write_text("test", test_file, encoding="invalid_encoding")


class TestReadJson(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_read_json(self):
        test_file = self.temp_path / "test.json"
        test_data = {"name": "Test", "value": 42, "nested": {"key": "value"}}
        test_file.write_text(json.dumps(test_data))
        self.assertEqual(read_json(test_file), test_data)

    def test_read_json_nonexistent(self):
        with self.assertRaises(FileNotFoundError):
            read_json(self.temp_path / "nonexistent.json")

    def test_read_json_invalid(self):
        test_file = self.temp_path / "invalid.json"
        test_file.write_text("{invalid json}")
        with self.assertRaises(ValueError):
            read_json(test_file)


class TestWriteJson(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_write_json(self):
        test_file = self.temp_path / "test.json"
        test_data = {"name": "Test", "value": 42, "nested": {"key": "value"}}
        write_json(test_data, test_file)
        self.assertEqual(json.loads(test_file.read_text()), test_data)

    def test_write_json_indent(self):
        test_file = self.temp_path / "test.json"
        test_data = {"name": "Test", "value": 42}
        write_json(test_data, test_file, indent=2)
        self.assertEqual(json.loads(test_file.read_text()), test_data)

    def test_write_json_ensure_ascii(self):
        test_file = self.temp_path / "test.json"
        test_data = {"name": "Test", "value": "🚀"}
        write_json(test_data, test_file, ensure_ascii=False)
        self.assertEqual(json.loads(test_file.read_text()), test_data)

    def test_write_json_invalid(self):
        test_file = self.temp_path / "test.json"
        with self.assertRaises(ValueError):
            write_json({"invalid": lambda x: x}, test_file)


class TestStoreLoadPkl(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_store_load_pkl(self):
        test_file = self.temp_path / "test.pkl"
        test_data = {"name": "Test", "value": 42, "nested": {"key": "value"}}
        store_pkl(test_data, test_file)
        self.assertEqual(load_pkl(test_file), test_data)

    def test_store_load_pkl_nonexistent(self):
        with self.assertRaises(FileNotFoundError):
            load_pkl(self.temp_path / "nonexistent.pkl")

    def test_store_load_pkl_invalid(self):
        test_file = self.temp_path / "invalid.pkl"
        test_file.write_bytes(b"invalid pickle data")
        with self.assertRaises(ValueError):
            load_pkl(test_file)

    def test_store_load_pkl_custom_class(self):
        test_file = self.temp_path / "test.pkl"
        custom_obj = CustomClass(42)
        store_pkl(custom_obj, test_file)
        loaded_obj = load_pkl(test_file)
        self.assertEqual(loaded_obj.value, 42)


if __name__ == "__main__":
    unittest.main()
