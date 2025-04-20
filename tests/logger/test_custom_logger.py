import json
import logging
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

from mitoolspro.logger.custom_logger import (
    MiJSONFormatter,
    NonErrorFilter,
    cleanup_logging,
    setup_logging,
)


class TestSetupLogging(TestCase):
    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        self.config = {
            "version": 1,
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "json",
                }
            },
            "formatters": {
                "json": {
                    "()": "mitoolspro.logger.custom_logger.MiJSONFormatter",
                    "fmt_keys": {
                        "level": "levelname",
                        "message": "message",
                        "timestamp": "asctime",
                    },
                }
            },
            "root": {"level": "DEBUG", "handlers": ["console"]},
        }
        json.dump(self.config, self.temp_file)
        self.temp_file.close()

    def tearDown(self):
        Path(self.temp_file.name).unlink()

    def test_setup_logging_success(self):
        with patch("logging.config.dictConfig") as mock_dict_config:
            setup_logging(Path(self.temp_file.name))
            mock_dict_config.assert_called_once()

    def test_setup_logging_nonexistent_file(self):
        with self.assertRaises(FileNotFoundError):
            setup_logging(Path("nonexistent.json"))


class TestCleanupLogging(TestCase):
    def setUp(self):
        self.logger = logging.getLogger()
        self.handler = logging.StreamHandler()
        self.logger.addHandler(self.handler)

    def test_cleanup_logging(self):
        cleanup_logging()
        self.assertEqual(len(self.logger.handlers), 0)


class TestMiJSONFormatter(TestCase):
    def setUp(self):
        self.formatter = MiJSONFormatter(
            fmt_keys={
                "level": "levelname",
                "message": "message",
                "timestamp": "asctime",
            }
        )

    def _create_log_record(self, level, msg, exc_info=None, stack_info=None):
        record = logging.LogRecord(
            name="test",
            level=level,
            pathname="test.py",
            lineno=1,
            msg=msg,
            args=(),
            exc_info=exc_info,
        )
        record.created = datetime.now().timestamp()
        record.msecs = 0
        record.relativeCreated = 0
        record.asctime = datetime.fromtimestamp(record.created).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        if stack_info:
            record.stack_info = stack_info
        return record

    def test_format_basic_record(self):
        record = self._create_log_record(logging.INFO, "Test message")

        formatted = self.formatter.format(record)
        data = json.loads(formatted)

        self.assertIn("level", data)
        self.assertIn("message", data)
        self.assertIn("timestamp", data)
        self.assertEqual(data["message"], "Test message")

    def test_format_with_exception(self):
        try:
            raise ValueError("Test error")
        except ValueError as e:
            record = self._create_log_record(
                logging.ERROR, "Test error", exc_info=(type(e), e, e.__traceback__)
            )

        formatted = self.formatter.format(record)
        data = json.loads(formatted)

        self.assertIn("exc_info", data)
        self.assertIn("Test error", data["exc_info"])

    def test_format_with_stack_info(self):
        record = self._create_log_record(
            logging.INFO, "Test message", stack_info="Stack trace"
        )

        formatted = self.formatter.format(record)
        data = json.loads(formatted)

        self.assertIn("stack_info", data)
        self.assertEqual(data["stack_info"], "Stack trace")


class TestNonErrorFilter(unittest.TestCase):
    def setUp(self):
        self.filter = NonErrorFilter()

    def _create_log_record(self, level, msg):
        record = logging.LogRecord(
            name="test",
            level=level,
            pathname="test.py",
            lineno=1,
            msg=msg,
            args=(),
            exc_info=None,
        )
        return record

    def test_filter_info_level(self):
        record = self._create_log_record(logging.INFO, "Test message")
        self.assertTrue(self.filter.filter(record))

    def test_filter_error_level(self):
        record = self._create_log_record(logging.ERROR, "Test error")
        self.assertFalse(self.filter.filter(record))


if __name__ == "__main__":
    unittest.main()
