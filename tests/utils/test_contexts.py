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
    def test_timing_milliseconds(self):
        with self.assertLogs('mtp', level='INFO') as log:
            with Timing("Task A: ", unit="ms"):
                time.sleep(0.01)  # Sleep for 10 milliseconds
        output = log.output[0]
        self.assertTrue("Task A: " in output)
        self.assertTrue("ms" in output)
        # Extract elapsed time from log message
        parts = output.split()
        elapsed_time = float(parts[2])
        self.assertAlmostEqual(
            elapsed_time, 10, delta=5
        )  # Allow delta for timing imprecision

    def test_timing_seconds(self):
        with self.assertLogs('mtp', level='INFO') as log:
            with Timing("Task B: ", unit="s"):
                time.sleep(0.5)  # Sleep for 500 milliseconds
        output = log.output[0]
        self.assertTrue("Task B: " in output)
        self.assertTrue("s" in output)
        parts = output.split()
        elapsed_time = float(parts[2])
        self.assertAlmostEqual(elapsed_time, 0.5, delta=0.1)

    def test_timing_minutes(self):
        with self.assertLogs('mtp', level='INFO') as log:
            with Timing("Task C: ", unit="m"):
                time.sleep(1)  # Sleep for 1 second
        output = log.output[0]
        self.assertTrue("Task C: " in output)
        self.assertTrue("m" in output)
        parts = output.split()
        elapsed_time = float(parts[2])
        self.assertAlmostEqual(elapsed_time, 1 / 60, delta=0.005)

    def test_on_exit_callback(self):
        def custom_on_exit(elapsed_time_ns):
            return f" - Time in nanoseconds: {elapsed_time_ns}"

        with self.assertLogs('mtp', level='INFO') as log:
            with Timing("Task D: ", unit="ms", on_exit=custom_on_exit):
                time.sleep(0.02)  # Sleep for 20 milliseconds
        output = log.output[0]
        self.assertTrue("Task D: " in output)
        self.assertTrue("ms" in output)
        self.assertTrue("Time in nanoseconds" in output)

    def test_disabled_timing(self):
        with self.assertNoLogs('mtp', level='INFO'):
            with Timing("Task E: ", unit="ms", enabled=False):
                time.sleep(0.01)  # Sleep for 10 milliseconds

    def test_default_unit(self):
        with self.assertLogs('mtp', level='INFO') as log:
            with Timing("Task F: "):
                time.sleep(0.01)  # Sleep for 10 milliseconds
        output = log.output[0]
        self.assertTrue("Task F: " in output)
        self.assertTrue("ms" in output)

    def test_invalid_unit(self):
        with self.assertRaises(KeyError):
            with Timing("Task G: ", unit="invalid"):
                time.sleep(0.01)  # Sleep for 10 milliseconds

    def test_short_sleep(self):
        with self.assertLogs('mtp', level='INFO') as log:
            with Timing("Short Task: ", unit="ns"):
                time.sleep(0.000001)  # Sleep for 1 microsecond
        output = log.output[0]
        self.assertTrue("ns" in output)
        parts = output.split()
        elapsed_time = float(parts[2])
        self.assertGreater(elapsed_time, 0)

    def test_timeit_factory(self):
        with self.assertLogs('mtp', level='INFO') as log:
            with timeit("Test Task: ", unit="s"):
                time.sleep(0.1)  # Sleep for 100 milliseconds
        output = log.output[0]
        self.assertTrue("Test Task: " in output)
        self.assertTrue("s" in output)
        parts = output.split()
        elapsed_time = float(parts[2])
        self.assertAlmostEqual(elapsed_time, 0.1, delta=0.05)

    def test_timeit_disabled(self):
        with self.assertNoLogs('mtp', level='INFO'):
            with timeit("Disabled Task: ", enabled=False):
                time.sleep(0.1)


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
        self.assertGreater(delay1, 0.099)  # 0.1 * 0.9
        self.assertLess(delay1, 0.12)  # 0.1 * 1.1
        self.assertGreater(delay2, 0.199)  # 0.2 * 0.9
        self.assertLess(delay2, 0.242)  # 0.2 * 1.1

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
