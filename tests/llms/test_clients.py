import tempfile
import unittest
from pathlib import Path
from unittest import TestCase

from mitoolspro.exceptions import ArgumentValueError
from mitoolspro.llms.clients import OpenAIClient, OpenAITokensCounter
from mitoolspro.llms.objects import ModelRegistry, Prompt


class TestOpenAIClient(TestCase):
    def setUp(self):
        self.model = "gpt-4o-mini"
        self.client = OpenAIClient(api_key="test_key", model=self.model)

    def test_initialization(self):
        self.assertEqual(self.client.model, self.model)
        self.assertEqual(len(self.client.raw_responses), 0)
        self.assertIsNone(self.client.counter)
        self.assertFalse(self.client.beta)

        client_with_counter = OpenAIClient(
            api_key="test_key",
            model=self.model,
            counter=OpenAITokensCounter(Path("test.json"), model=self.model),
            beta=True,
        )
        self.assertIsNotNone(client_with_counter.counter)
        self.assertTrue(client_with_counter.beta)

    def test_parse_request(self):
        prompt = Prompt("Test prompt")
        request = self.client.parse_request(prompt)
        self.assertEqual(request["model"], self.model)
        self.assertEqual(len(request["messages"]), 1)
        self.assertEqual(request["messages"][0]["role"], "user")
        self.assertEqual(request["messages"][0]["content"], "Test prompt")

    def test_get_model_info(self):
        info = self.client.get_model_info()
        self.assertEqual(info["name"], "OpenAI")
        self.assertEqual(info["model"], self.model)

    def test_model_name(self):
        self.assertEqual(self.client.model_name(), self.model)


class TestOpenAITokensCounter(TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.file_path = Path(self.temp_dir.name) / "tokens.json"
        self.model = "gpt-4o-mini"
        self.model_registry = ModelRegistry.get_instance("openai")
        self.model_cost = self.model_registry.get_model_cost(self.model)
        self.counter = OpenAITokensCounter(self.file_path, model=self.model)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_initialization(self):
        self.assertEqual(self.counter.model, self.model)
        self.assertEqual(self.counter.model_cost["input"], self.model_cost["input"])
        self.assertEqual(self.counter.model_cost["output"], self.model_cost["output"])

        with self.assertRaises(ArgumentValueError):
            OpenAITokensCounter(file_path=self.file_path, model="invalid_model")

    def test_count_tokens_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.counter.count_tokens("test text")

    def test_different_models_costs(self):
        models_to_test = [
            "gpt-3.5-turbo",
            "gpt-4o",
            "o1-preview",
            "o1",
            "o1-mini",
        ]

        for model in models_to_test:
            counter = OpenAITokensCounter(self.file_path, model=model)
            self.assertEqual(
                counter.model_cost["input"],
                self.model_registry.get_model_cost(model)["input"],
            )
            self.assertEqual(
                counter.model_cost["output"],
                self.model_registry.get_model_cost(model)["output"],
            )


if __name__ == "__main__":
    unittest.main()
