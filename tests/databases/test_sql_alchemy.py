import os
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session

from mitoolspro.databases.sql_alchemy import DBQueueWriter

Base = declarative_base()


class TestModel(Base):
    __tablename__ = "test_table"
    id = Column(Integer, primary_key=True)
    name = Column(String)


class TestDBQueueWriter(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.writer = DBQueueWriter(self.db_path, Base)

    def tearDown(self):
        self.writer.close()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        os.rmdir(self.temp_dir)

    def test_initialization(self):
        self.assertTrue(os.path.exists(self.db_path))
        self.assertIsNotNone(self.writer.engine)
        self.assertIsNotNone(self.writer.Session)
        self.assertTrue(self.writer._thread.is_alive())

    def test_register_handler(self):
        mock_handler = MagicMock()
        self.writer.register_handler("test_op", mock_handler)
        self.assertEqual(self.writer._handlers["test_op"], mock_handler)

    def test_enqueue_and_process(self):
        mock_handler = MagicMock()
        self.writer.register_handler("test_op", mock_handler)

        test_args = ("arg1", "arg2")
        self.writer.enqueue("test_op", *test_args)
        self.writer.wait_until_done()

        mock_handler.assert_called_once()
        call_args = mock_handler.call_args[0]
        self.assertIsInstance(call_args[0], Session)
        self.assertEqual(call_args[1:], test_args)

    def test_unsupported_operation(self):
        with self.assertLogs(level="ERROR") as log:
            self.writer.enqueue("unsupported_op")
            self.writer.wait_until_done()
            self.assertIn("Unsupported operation: unsupported_op", log.output[0])

    def test_error_handling(self):
        def failing_handler(session, *args):
            raise ValueError("Test error")

        self.writer.register_handler("failing_op", failing_handler)

        with self.assertLogs(level="ERROR") as log:
            self.writer.enqueue("failing_op")
            self.writer.wait_until_done()
            self.assertIn("Error in DBTaskWriter: Test error", log.output[0])

    def test_close_and_cleanup(self):
        self.writer.close()
        self.assertFalse(self.writer._thread.is_alive())

    def test_multiple_operations(self):
        results = []

        def handler(session, value):
            results.append(value)

        self.writer.register_handler("test_op", handler)

        for i in range(5):
            self.writer.enqueue("test_op", i)

        self.writer.wait_until_done()
        self.assertEqual(results, [0, 1, 2, 3, 4])

    def test_database_operations(self):
        def insert_handler(session, name):
            test_model = TestModel(name=name)
            session.add(test_model)

        self.writer.register_handler("insert", insert_handler)

        test_names = ["test1", "test2", "test3"]
        for name in test_names:
            self.writer.enqueue("insert", name)

        self.writer.wait_until_done()

        session = self.writer.Session()
        try:
            results = session.query(TestModel).all()
            self.assertEqual(len(results), len(test_names))
            self.assertEqual([r.name for r in results], test_names)
        finally:
            session.close()


if __name__ == "__main__":
    unittest.main()
