import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from typing import Dict
from unittest import TestCase

from mitoolspro.exceptions import (
    ArgumentKeyError,
    ArgumentTypeError,
    ArgumentValueError,
)
from mitoolspro.llms.objects import (
    ModelRegistry,
    PersistentTokensCounter,
    Prompt,
    TokensCounter,
    TokenUsageStats,
)


class TestPrompt(TestCase):
    def test_initialization(self):
        prompt = Prompt("Translate to French.", {"task": "translation"})
        self.assertEqual(prompt.text, "Translate to French.")
        self.assertEqual(prompt.metadata, {"task": "translation"})
        prompt = Prompt("Summarize this text.")
        self.assertEqual(prompt.metadata, {})
        with self.assertRaises(ArgumentValueError):
            Prompt("")
        with self.assertRaises(ArgumentValueError):
            Prompt(123)  # Non-string text

    def test_representation(self):
        prompt = Prompt("Translate to French.")
        print(prompt)
        self.assertTrue(repr(prompt).startswith("Prompt(\ntext"))

    def test_format(self):
        prompt = Prompt("Translate to French: {text}")
        formatted_prompt = prompt.format(text="Hello")
        self.assertEqual(formatted_prompt.text, "Translate to French: Hello")
        with self.assertRaises(ArgumentKeyError):
            prompt.format(language="English")

    def test_update_metadata(self):
        prompt = Prompt("Translate to French.")
        prompt.update_metadata("task", "translation")
        self.assertEqual(prompt.metadata, {"task": "translation"})
        with self.assertRaises(ArgumentValueError):
            prompt.update_metadata(123, "translation")
        with self.assertRaises(ArgumentValueError):
            prompt.update_metadata("task", 456)

    def test_get_metadata(self):
        prompt = Prompt("Translate to French.", {"task": "translation"})
        self.assertEqual(prompt.get_metadata("task"), "translation")
        self.assertIsNone(prompt.get_metadata("language"))

    def test_to_dict(self):
        prompt = Prompt("Translate to French.", {"task": "translation"})
        expected = {"text": "Translate to French.", "metadata": {"task": "translation"}}
        self.assertEqual(prompt.to_dict(), expected)

    def test_from_dict(self):
        data = {"text": "Translate to French.", "metadata": {"task": "translation"}}
        prompt = Prompt.from_dict(data)
        self.assertEqual(prompt.text, "Translate to French.")
        self.assertEqual(prompt.metadata, {"task": "translation"})
        with self.assertRaises(ArgumentValueError):
            Prompt.from_dict({"metadata": {"task": "translation"}})

    def test_concatenation_with_prompt(self):
        prompt1 = Prompt("Translate to French.", {"task": "translation"})
        prompt2 = Prompt("Summarize this text.", {"task": "summarization"})
        combined = prompt1 + prompt2
        self.assertEqual(combined.text, "Translate to French.\nSummarize this text.")
        self.assertEqual(
            combined.metadata, {"task": "translation"}
        )  # Metadata from frist prompt

    def test_concatenation_with_string(self):
        prompt = Prompt("Translate to French.")
        combined = prompt + "Provide a summary."
        self.assertEqual(combined.text, "Translate to French.\nProvide a summary.")
        self.assertEqual(combined.metadata, {})
        prompt += "Provide a detailed summary."
        self.assertEqual(
            prompt.text, "Translate to French.\nProvide a detailed summary."
        )

    def test_invalid_concatenation(self):
        prompt = Prompt("Translate to French.")
        with self.assertRaises(ArgumentTypeError):
            prompt + 123  # Invalid type

    def test_static_concatenate(self):
        prompt1 = Prompt("Translate to French.", {"task": "translation"})
        prompt2 = Prompt("Summarize this text.", {"task": "summarization"})
        text = "Explain this in detail."
        combined = Prompt.concatenate([prompt1, prompt2, text], separator="\n---\n")
        self.assertEqual(
            combined.text,
            "Translate to French.\n---\nSummarize this text.\n---\nExplain this in detail.",
        )
        self.assertEqual(combined.metadata, {"task": "summarization"})
        with self.assertRaises(ArgumentValueError):
            Prompt.concatenate([])
        with self.assertRaises(ArgumentTypeError):
            Prompt.concatenate([prompt1, 123])

    def test_edge_cases(self):
        long_text = "A" * 10_000
        prompt = Prompt(long_text)
        self.assertEqual(prompt.text, long_text)
        prompt1 = Prompt("Task 1", {"task": "t1"})
        prompt2 = Prompt("Task 2", {"task": "t2"})
        combined = prompt1 + prompt2
        self.assertEqual(combined.metadata, {"task": "t1"})
        prompts = [Prompt("One"), Prompt("Two"), "Three"]
        combined = Prompt.concatenate(prompts, separator=" | ")
        self.assertEqual(combined.text, "One | Two | Three")


class TestTokensCounter(TokensCounter):
    def get_usage_stats(self, response: Dict) -> TokenUsageStats:
        total_tokens = len(response.get("text", "").split())
        prompt_tokens = total_tokens // 2
        completion_tokens = total_tokens - prompt_tokens
        registry = ModelRegistry.get_instance(self.source)
        model_costs = registry.get_model_costs(self.model)
        cost = (
            prompt_tokens * model_costs["input"]
            + completion_tokens * model_costs["output"]
        ) / 1_000_000
        return TokenUsageStats(
            source=self.source,
            model=self.model,
            model_cost=model_costs,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=cost,
            timestamp=datetime.now(),
        )

    def count_tokens(self, text: str) -> int:
        return len(text.split())


class TokensCounterTests(TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.source = "openai"
        self.model = "gpt-3.5-turbo"
        self.model_cost = ModelRegistry.get_instance(self.source).get_model_costs(
            self.model
        )
        self.counter = TestTokensCounter(self.source, self.model)
        self.usage_sample = self.counter.get_usage_stats(
            {"text": "This is a sample text with nine tokens total." * 10}
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_initialization(self):
        self.assertEqual(self.counter.source, self.source)
        self.assertEqual(self.counter.model, self.model)
        self.assertEqual(self.counter.prompt_tokens_count, 0)
        self.assertEqual(self.counter.completion_tokens_count, 0)
        self.assertEqual(self.counter.total_tokens_count, 0)
        self.assertEqual(self.counter.usage_history, [])

    def test_update_usage(self):
        self.counter.update(self.usage_sample)
        self.assertEqual(len(self.counter.usage_history), 1)
        self.assertEqual(self.counter.prompt_tokens_count, 40)
        self.assertEqual(self.counter.completion_tokens_count, 41)
        self.assertEqual(self.counter.total_tokens_count, 81)

    def test_count_tokens(self):
        text = "This is a sample text with nine tokens total."
        self.assertEqual(self.counter.count_tokens(text), 9)

    def test_would_exceed_context(self):
        self.counter.set_max_context_length(50)
        self.assertTrue(self.counter.would_exceed_context("word " * 51))
        self.assertFalse(self.counter.would_exceed_context("word " * 49))

    def test_cost_calculation(self):
        self.counter.update(self.usage_sample)
        self.assertEqual(self.counter._calculate_cost(), 0.000448)
        self.assertEqual(self.counter._calculate_input_cost(), 0.00012)
        self.assertEqual(self.counter._calculate_output_cost(), 0.000328)

    def test_cost_detail(self):
        self.counter.update(self.usage_sample)
        cost_detail = self.counter.cost_detail
        self.assertAlmostEqual(cost_detail["total"], 0.000448)
        self.assertAlmostEqual(cost_detail["prompt_tokens"], 0.00012)
        self.assertAlmostEqual(cost_detail["completion_tokens"], 0.000328)

    def test_json_serialization(self):
        self.counter.update(self.usage_sample)
        json_data = self.counter.json()
        data = json.loads(json_data)
        self.assertEqual(data["prompt_tokens_count"], 40)
        self.assertEqual(data["completion_tokens_count"], 41)
        self.assertEqual(data["total_tokens_count"], 81)
        self.assertEqual(data["model"], self.model)
        self.assertEqual(data["source"], self.source)

    def test_save_to_json(self):
        file_path = self.temp_path / "test_tokens_counter.json"
        self.counter.update(self.usage_sample)
        self.counter.save(file_path)
        with open(file_path, "r") as f:
            data = json.load(f)
        self.assertEqual(data["prompt_tokens_count"], 40)
        self.assertEqual(data["completion_tokens_count"], 41)
        self.assertEqual(data["total_tokens_count"], 81)
        self.assertEqual(data["model"], self.model)
        self.assertEqual(data["source"], self.source)

    def test_load_from_json(self):
        file_path = self.temp_path / "test_tokens_counter.json"
        self.counter.update(self.usage_sample)
        self.counter.save(file_path)

        loaded_counter = TestTokensCounter.load(file_path)
        self.assertEqual(loaded_counter.prompt_tokens_count, 40)
        self.assertEqual(loaded_counter.completion_tokens_count, 41)
        self.assertEqual(loaded_counter.total_tokens_count, 81)
        self.assertEqual(loaded_counter.model, self.model)
        self.assertEqual(loaded_counter.source, self.source)
        self.assertEqual(len(loaded_counter.usage_history), 1)
        self.assertAlmostEqual(loaded_counter.cost, 0.000448)

    def test_invalid_file_extension(self):
        with self.assertRaises(ArgumentValueError):
            self.counter.save(self.temp_path / "invalid_file.txt")

    def test_load_from_nonexistent_file(self):
        with self.assertRaises(FileNotFoundError):
            TestTokensCounter.load(self.temp_path / "nonexistent.json")

    def test_usage_dataframe(self):
        self.counter.update(self.usage_sample)
        df = self.counter.usage()
        self.assertEqual(df.shape, (1, 8))  # Updated for new TokenUsageStats fields
        self.assertEqual(df.loc[0, "total_tokens"], 81)
        self.assertEqual(df.loc[0, "prompt_tokens"], 40)
        self.assertEqual(df.loc[0, "completion_tokens"], 41)
        self.assertEqual(df.loc[0, "source"], self.source)
        self.assertEqual(df.loc[0, "model"], self.model)
        self.assertAlmostEqual(df.loc[0, "cost"], 0.000448)

    def test_different_model_costs(self):
        gpt4_counter = TestTokensCounter(source="openai", model="gpt-4o")
        gpt35_counter = TestTokensCounter(source="openai", model="gpt-3.5-turbo")
        registry = ModelRegistry.get_instance("openai")
        gpt4_costs = registry.get_model_costs("gpt-4o")
        gpt35_costs = registry.get_model_costs("gpt-3.5-turbo")
        # Create usage stats with same tokens but different models
        gpt4_usage = TokenUsageStats(
            source="openai",
            model="gpt-4o",
            model_cost=gpt4_costs,
            total_tokens=100,
            prompt_tokens=60,
            completion_tokens=40,
            cost=(60 * gpt4_costs["input"] + 40 * gpt4_costs["output"]) / 1_000_000,
            timestamp=datetime.now(),
        )
        gpt35_usage = TokenUsageStats(
            source="openai",
            model="gpt-3.5-turbo",
            model_cost=gpt35_costs,
            total_tokens=100,
            prompt_tokens=60,
            completion_tokens=40,
            cost=(60 * gpt35_costs["input"] + 40 * gpt35_costs["output"]) / 1_000_000,
            timestamp=datetime.now(),
        )
        gpt4_counter.update(gpt4_usage)
        gpt35_counter.update(gpt35_usage)
        # Verify different costs for same token counts
        self.assertNotEqual(gpt4_counter.cost, gpt35_counter.cost)
        self.assertEqual(
            gpt4_counter.total_tokens_count, gpt35_counter.total_tokens_count
        )


class TestPersistentTokensCounter(PersistentTokensCounter):
    def get_usage_stats(self, response: Dict) -> TokenUsageStats:
        total_tokens = len(response.get("text", "").split())
        prompt_tokens = total_tokens // 2
        completion_tokens = total_tokens - prompt_tokens

        registry = ModelRegistry.get_instance(self.source)
        model_costs = registry.get_model_costs(self.model)

        cost = (
            prompt_tokens * model_costs["input"]
            + completion_tokens * model_costs["output"]
        ) / 1_000_000

        return TokenUsageStats(
            source=self.source,
            model=self.model,
            model_cost=model_costs,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=cost,
            timestamp=datetime.now(),
        )

    def count_tokens(self, text: str) -> int:
        return len(text.split())


class PersistentTokensCounterTests(TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.model = "gpt-3.5-turbo"
        self.model2 = "gpt-4o"
        self.source = "openai"
        self.file_path = self.temp_path / "token_counter.json"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_singleton_behavior_same_model(self):
        counter1 = TestPersistentTokensCounter(
            self.file_path, source=self.source, model=self.model
        )
        counter2 = TestPersistentTokensCounter(
            self.file_path, source=self.source, model=self.model
        )
        self.assertIs(counter1, counter2)

    def test_singleton_behavior_different_model(self):
        counter1 = TestPersistentTokensCounter(
            self.file_path, source=self.source, model=self.model
        )
        counter2 = TestPersistentTokensCounter(
            self.file_path, source=self.source, model=self.model2
        )
        self.assertIs(counter1, counter2)
        self.assertEqual(counter2.model, self.model2)  # Model should be updated

    def test_persistence_with_model_change(self):
        counter1 = TestPersistentTokensCounter(
            self.file_path, source=self.source, model=self.model
        )
        response1 = {"text": "test " * 10}
        stats1 = counter1.get_usage_stats(response1)
        counter1.update(stats1)
        counter2 = TestPersistentTokensCounter(
            self.file_path, source=self.source, model=self.model2
        )
        response2 = {"text": "test " * 10}
        stats2 = counter2.get_usage_stats(response2)
        counter2.update(stats2)
        self.assertEqual(len(counter2.usage_history), 2)
        self.assertEqual(counter2.usage_history[0].model, self.model)
        self.assertEqual(counter2.usage_history[1].model, self.model2)
        self.assertNotEqual(
            counter2.usage_history[0].cost, counter2.usage_history[1].cost
        )

    def test_load_with_empty_history(self):
        counter = TestPersistentTokensCounter(
            self.file_path, source=self.source, model=self.model
        )
        counter.save()
        loaded = TestPersistentTokensCounter.load(self.file_path)
        self.assertEqual(loaded.source, self.source)
        self.assertEqual(loaded.model, self.model)
        self.assertEqual(len(loaded.usage_history), 0)

    def test_load_uses_latest_model(self):
        counter = TestPersistentTokensCounter(
            self.file_path, source=self.source, model=self.model
        )
        response1 = {"text": "test " * 10}
        stats1 = counter.get_usage_stats(response1)
        counter.update(stats1)
        counter = TestPersistentTokensCounter(
            self.file_path, source=self.source, model=self.model2
        )
        response2 = {"text": "test " * 10}
        stats2 = counter.get_usage_stats(response2)
        counter.update(stats2)
        loaded = TestPersistentTokensCounter.load(self.file_path)
        self.assertEqual(loaded.model, self.model2)
        self.assertEqual(loaded.source, self.source)

    def test_cost_calculation_across_models(self):
        counter = TestPersistentTokensCounter(
            self.file_path, source=self.source, model=self.model
        )
        response1 = {"text": "test " * 10}
        stats1 = counter.get_usage_stats(response1)
        counter.update(stats1)
        cost1 = counter.cost
        counter = TestPersistentTokensCounter(
            self.file_path, source=self.source, model=self.model2
        )
        response2 = {"text": "test " * 10}
        stats2 = counter.get_usage_stats(response2)
        counter.update(stats2)
        cost2 = counter.cost - cost1
        self.assertNotEqual(cost1, cost2)
        self.assertEqual(counter.total_tokens_count, 20)

    def test_dataframe_output(self):
        counter = TestPersistentTokensCounter(
            self.file_path, source=self.source, model=self.model
        )
        response1 = {"text": "test " * 10}
        stats1 = counter.get_usage_stats(response1)
        counter.update(stats1)

        counter = TestPersistentTokensCounter(
            self.file_path, source=self.source, model=self.model2
        )
        response2 = {"text": "test " * 10}
        stats2 = counter.get_usage_stats(response2)
        counter.update(stats2)

        df = counter.usage()
        self.assertEqual(len(df), 2)
        self.assertEqual(list(df["model"].unique()), [self.model, self.model2])
        self.assertEqual(list(df["source"].unique()), [self.source])

    def test_invalid_file_path(self):
        nonexistent_path = self.temp_path / "nonexistent" / "token_counter.json"
        with self.assertRaises(FileNotFoundError):
            TestPersistentTokensCounter.load(nonexistent_path)


class TestModelRegistry(TestCase):
    def setUp(self):
        self.registry = ModelRegistry.get_instance("openai")

    def test_singleton_behavior(self):
        registry1 = ModelRegistry.get_instance("openai")
        registry2 = ModelRegistry.get_instance("openai")
        self.assertIs(registry1, registry2)
        registry3 = ModelRegistry.get_instance("openai")
        self.assertIs(registry1, registry3)

    def test_initialization_with_supported_company(self):
        self.assertEqual(self.registry.source, "openai")

    def test_initialization_with_unsupported_company(self):
        with self.assertRaises(ArgumentValueError):
            ModelRegistry.get_instance("unsupported")

    def test_model_configurations_exist(self):
        self.assertTrue(hasattr(ModelRegistry, "OPENAI_MODELS"))
        self.assertTrue(hasattr(ModelRegistry, "ANTHROPIC_MODELS"))
        self.assertTrue(hasattr(ModelRegistry, "GOOGLE_MODELS"))

    def test_openai_models_initialization(self):
        self.assertIn("gpt-3.5-turbo", self.registry.models)
        self.assertIn("gpt-4o-mini", self.registry.models)
        self.assertIn("gpt-4o", self.registry.models)
        self.assertIn("o1-preview", self.registry.models)
        self.assertIn("o1", self.registry.models)
        self.assertIn("o1-mini", self.registry.models)

    def test_get_model_config_existing_model(self):
        costs = self.registry.get_model_costs("gpt-3.5-turbo")
        self.assertEqual(costs["input"], 3.0)
        self.assertEqual(costs["output"], 8.0)

    def test_get_model_config_nonexistent_model(self):
        with self.assertRaises(ValueError):
            self.registry.get_model_costs("nonexistent-model")

    def test_model_config_structure(self):
        for _, config in self.registry.models.items():
            self.assertIn("input", config)
            self.assertIn("output", config)
            self.assertIsInstance(config["input"], (int, float))
            self.assertIsInstance(config["output"], (int, float))

    def test_cost_values_are_positive(self):
        for _, config in self.registry.models.items():
            self.assertGreaterEqual(config["input"], 0)
            self.assertGreaterEqual(config["output"], 0)


if __name__ == "__main__":
    unittest.main()
