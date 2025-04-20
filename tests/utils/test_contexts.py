import logging
import tempfile
import time
import unittest
from io import StringIO
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

from mitoolspro.utils.contexts import Timing, retry, timeit


class TestTiming(TestCase):
    @patch("sys.stdout", new_callable=StringIO)
    def test_timing_milliseconds(self, mock_stdout):
        with Timing("Task A: ", unit="ms"):
            time.sleep(0.01)  # Sleep for 10 milliseconds
        output = mock_stdout.getvalue().strip()
        self.assertTrue("Task A: " in output)
        self.assertTrue("ms" in output)
        elapsed_time = float(output.split()[2])
        self.assertAlmostEqual(
            elapsed_time, 10, delta=5
        )  # Allow delta for timing imprecision

    @patch("sys.stdout", new_callable=StringIO)
    def test_timing_seconds(self, mock_stdout):
        with Timing("Task B: ", unit="s"):
            time.sleep(0.5)  # Sleep for 500 milliseconds
        output = mock_stdout.getvalue().strip()
        self.assertTrue("Task B: " in output)
        self.assertTrue("s" in output)
        elapsed_time = float(output.split()[2])
        self.assertAlmostEqual(elapsed_time, 0.5, delta=0.1)

    @patch("sys.stdout", new_callable=StringIO)
    def test_timing_minutes(self, mock_stdout):
        with Timing("Task C: ", unit="m"):
            time.sleep(1)  # Sleep for 1 second
        output = mock_stdout.getvalue().strip()
        self.assertTrue("Task C: " in output)
        self.assertTrue("m" in output)
        elapsed_time = float(output.split()[2])
        self.assertAlmostEqual(elapsed_time, 1 / 60, delta=0.005)

    @patch("sys.stdout", new_callable=StringIO)
    def test_on_exit_callback(self, mock_stdout):
        def custom_on_exit(elapsed_time_ns):
            return f" - Time in nanoseconds: {elapsed_time_ns}"

        with Timing("Task D: ", unit="ms", on_exit=custom_on_exit):
            time.sleep(0.02)  # Sleep for 20 milliseconds
        output = mock_stdout.getvalue().strip()
        self.assertTrue("Task D: " in output)
        self.assertTrue("ms" in output)
        self.assertTrue("Time in nanoseconds" in output)

    @patch("sys.stdout", new_callable=StringIO)
    def test_disabled_timing(self, mock_stdout):
        with Timing("Task E: ", unit="ms", enabled=False):
            time.sleep(0.01)  # Sleep for 10 milliseconds
        self.assertEqual(mock_stdout.getvalue().strip(), "")

    @patch("sys.stdout", new_callable=StringIO)
    def test_default_unit(self, mock_stdout):
        with Timing("Task F: "):
            time.sleep(0.01)  # Sleep for 10 milliseconds
        output = mock_stdout.getvalue().strip()
        self.assertTrue("Task F: " in output)
        self.assertTrue("ms" in output)

    def test_invalid_unit(self):
        with self.assertRaises(KeyError):
            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                with Timing("Task G: ", unit="invalid"):
                    time.sleep(0.01)  # Sleep for 10 milliseconds
                mock_stdout.getvalue().strip()

    def test_short_sleep(self):
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            with Timing("Short Task: ", unit="ns"):
                time.sleep(0.000001)  # Sleep for 1 microsecond
            output = mock_stdout.getvalue().strip()
            self.assertTrue("ns" in output)
            elapsed_time = float(output.split()[2])
            self.assertGreater(elapsed_time, 0)

    def test_timeit_factory(self):
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            with timeit("Test Task: ", unit="s"):
                time.sleep(0.1)  # Sleep for 100 milliseconds
            output = mock_stdout.getvalue().strip()
            self.assertTrue("Test Task: " in output)
            self.assertTrue("s" in output)
            elapsed_time = float(output.split()[2])
            self.assertAlmostEqual(elapsed_time, 0.1, delta=0.05)

    def test_timeit_disabled(self):
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            with timeit("Disabled Task: ", enabled=False):
                time.sleep(0.1)
            self.assertEqual(mock_stdout.getvalue().strip(), "")


class TestRetry(TestCase):
    def setUp(self):
        self.logger = logging.getLogger("mtp")
        self.logger.setLevel(logging.INFO)
        self.log_output = StringIO()
        self.handler = logging.StreamHandler(self.log_output)
        self.logger.addHandler(self.handler)

    def tearDown(self):
        self.logger.removeHandler(self.handler)
        self.handler.close()

    def test_successful_execution(self):
        @retry(max_attempts=3)
        def successful_func():
            return "success"

        result = successful_func()
        self.assertEqual(result, "success")
        logs = self.log_output.getvalue()
        self.assertIn("Attempt 1/3 for successful_func", logs)
        self.assertIn("successful_func succeeded", logs)

    def test_retry_on_failure(self):
        attempts = 0

        @retry(max_attempts=3, delay_seconds=0.1)
        def failing_func():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ValueError("Temporary failure")
            return "success"

        result = failing_func()
        self.assertEqual(result, "success")
        self.assertEqual(attempts, 3)
        logs = self.log_output.getvalue()
        self.assertIn("Attempt 1/3 for failing_func", logs)
        self.assertIn("Attempt 2/3 for failing_func", logs)
        self.assertIn("Attempt 3/3 for failing_func", logs)

    def test_max_attempts_exceeded(self):
        @retry(max_attempts=2, delay_seconds=0.1)
        def always_fail():
            raise ValueError("Always fails")

        with self.assertRaises(TimeoutError) as context:
            always_fail()

        self.assertIn(
            "Failed to execute always_fail after 2 attempts", str(context.exception)
        )
        logs = self.log_output.getvalue()
        self.assertIn("Attempt 1/2 for always_fail", logs)
        self.assertIn("Attempt 2/2 for always_fail", logs)

    def test_specific_exception_handling(self):
        @retry(max_attempts=2, delay_seconds=0.1, exceptions=ValueError)
        def raise_value_error():
            raise ValueError("Test error")

        @retry(max_attempts=2, delay_seconds=0.1, exceptions=ValueError)
        def raise_type_error():
            raise TypeError("Test error")

        with self.assertRaises(TimeoutError):
            raise_value_error()

        with self.assertRaises(TypeError):
            raise_type_error()

    def test_backoff_factor(self):
        start_time = time.time()

        @retry(max_attempts=3, delay_seconds=0.1, backoff_factor=2)
        def failing_func():
            raise ValueError("Fail")

        with self.assertRaises(TimeoutError):
            failing_func()

        elapsed_time = time.time() - start_time
        self.assertGreater(elapsed_time, 0.1 + 0.2)  # First delay + second delay

    def test_jitter(self):
        delays = []

        @retry(max_attempts=3, delay_seconds=0.1, jitter=True)
        def failing_func():
            delays.append(time.time())
            raise ValueError("Fail")

        with self.assertRaises(TimeoutError):
            failing_func()

        self.assertEqual(len(delays), 3)
        delay1 = delays[1] - delays[0]
        delay2 = delays[2] - delays[1]
        self.assertGreater(delay1, 0.09)  # 0.1 * 0.9
        self.assertLess(delay1, 0.11)  # 0.1 * 1.1
        self.assertGreater(delay2, 0.18)  # 0.2 * 0.9
        self.assertLess(delay2, 0.22)  # 0.2 * 1.1

    def test_no_jitter(self):
        delays = []

        @retry(max_attempts=3, delay_seconds=0.1, jitter=False, backoff_factor=1)
        def failing_func():
            delays.append(time.time())
            raise ValueError("Fail")

        with self.assertRaises(TimeoutError):
            failing_func()

        self.assertEqual(len(delays), 3)
        delay1 = delays[1] - delays[0]
        delay2 = delays[2] - delays[1]
        self.assertAlmostEqual(delay1, 0.1, delta=0.01)
        self.assertAlmostEqual(delay2, 0.1, delta=0.01)


if __name__ == "__main__":
    unittest.main()
