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
        self.source = "openai"
        self.model = "gpt-3.5-turbo"
        self.model_cost = ModelRegistry.get_instance(self.source).get_model_costs(
            self.model
        )
        self.counter = TestTokensCounter(self.source, self.model)
        self.usage_sample = self.counter.get_usage_stats(
            {"text": "This is a sample text with nine tokens total." * 10}
        )

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
        file_path = Path("test_tokens_counter.json")
        self.counter.update(self.usage_sample)
        self.counter.save(file_path)
        with open(file_path, "r") as f:
            data = json.load(f)
        self.assertEqual(data["prompt_tokens_count"], 40)
        self.assertEqual(data["completion_tokens_count"], 41)
        self.assertEqual(data["total_tokens_count"], 81)
        self.assertEqual(data["model"], self.model)
        self.assertEqual(data["source"], self.source)
        file_path.unlink()  # Cleanup

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

    def test_load_from_json(self):
        file_path = Path("test_tokens_counter.json")
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
        file_path.unlink()  # Cleanup

    def test_invalid_file_extension(self):
        with self.assertRaises(ArgumentValueError):
            self.counter.save("invalid_file.txt")

    def test_load_from_nonexistent_file(self):
        with self.assertRaises(FileNotFoundError):
            TestTokensCounter.load("nonexistent.json")

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
        cost = self._calculate_cost(total_tokens)
        return TokenUsageStats(
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
        self.test_dir = tempfile.TemporaryDirectory()
        self.file_path = Path(self.test_dir.name) / "token_counter.json"

    def tearDown(self):
        self.test_dir.cleanup()

    def test_singleton_behavior(self):
        counter1 = TestPersistentTokensCounter(
            self.file_path, cost_per_1M_input_tokens=0.02, cost_per_1M_output_tokens=0.2
        )
        counter2 = TestPersistentTokensCounter(
            self.file_path, cost_per_1M_input_tokens=0.05, cost_per_1M_output_tokens=0.5
        )
        self.assertIs(counter1, counter2)  # Both should refer to the same instance
        self.assertEqual(counter1.cost_per_1M_input_tokens, 0.02)
        self.assertEqual(
            counter1.cost_per_1M_output_tokens, 0.2
        )  # The first initialization value is retained

    def test_file_based_initialization(self):
        counter1 = TestPersistentTokensCounter(
            self.file_path, cost_per_1M_input_tokens=0.02
        )
        usage_stats = TokenUsageStats(
            total_tokens=1000,
            prompt_tokens=700,
            completion_tokens=300,
            cost=0.02,
            timestamp=datetime.now(),
        )
        counter1.update(usage_stats)
        counter2 = TestPersistentTokensCounter(self.file_path)
        self.assertEqual(len(counter2.usage_history), 1)
        self.assertEqual(counter2.count, 1000)

    def test_update_and_auto_save(self):
        counter = TestPersistentTokensCounter(
            self.file_path, cost_per_1M_input_tokens=0.02
        )
        usage_stats = TokenUsageStats(
            total_tokens=1000,
            prompt_tokens=700,
            completion_tokens=300,
            cost=0.02,
            timestamp=datetime.now(),
        )
        counter.update(usage_stats)
        new_counter = TestPersistentTokensCounter(self.file_path)
        self.assertEqual(new_counter.count, 1000)
        self.assertEqual(new_counter.prompt_tokens_count, 700)
        self.assertEqual(new_counter.completion_tokens_count, 300)

    def test_usage_dataframe(self):
        counter = TestPersistentTokensCounter(
            self.file_path, cost_per_1M_input_tokens=0.02
        )
        usage_stats1 = TokenUsageStats(
            total_tokens=1000,
            prompt_tokens=700,
            completion_tokens=300,
            cost=0.02,
            timestamp=datetime.now(),
        )
        usage_stats2 = TokenUsageStats(
            total_tokens=500,
            prompt_tokens=300,
            completion_tokens=200,
            cost=0.01,
            timestamp=datetime.now(),
        )
        counter.update(usage_stats1)
        counter.update(usage_stats2)
        df = counter.usage()
        self.assertEqual(len(df), 2)  # Two rows in the DataFrame
        self.assertEqual(df["total_tokens"].sum(), 1500)
        self.assertEqual(df["cost"].sum(), 0.03)

    def test_multiple_instances_with_different_paths(self):
        file_path2 = Path(self.test_dir.name) / "token_counter_2.json"
        counter1 = TestPersistentTokensCounter(
            self.file_path, cost_per_1M_input_tokens=0.02
        )
        counter2 = TestPersistentTokensCounter(
            file_path2, cost_per_1M_input_tokens=0.05
        )
        self.assertIsNot(counter1, counter2)  # Different instances
        self.assertNotEqual(
            counter1.cost_per_1M_input_tokens, counter2.cost_per_1M_input_tokens
        )

    def test_empty_file_initialization(self):
        self.assertFalse(self.file_path.exists())
        counter = TestPersistentTokensCounter(
            self.file_path, cost_per_1M_input_tokens=0.02
        )
        self.assertTrue(self.file_path.exists())
        self.assertEqual(len(counter.usage_history), 0)

    def test_data_integrity_after_reload(self):
        counter = TestPersistentTokensCounter(
            self.file_path, cost_per_1M_input_tokens=0.02
        )
        usage_stats = TokenUsageStats(
            total_tokens=2000,
            prompt_tokens=1500,
            completion_tokens=500,
            cost=3e-5,
            timestamp=datetime.now(),
        )
        counter.update(usage_stats)
        new_counter = TestPersistentTokensCounter(self.file_path)
        self.assertEqual(new_counter.total_tokens_count, 2000)
        self.assertEqual(new_counter.cost, 3e-5)

    def test_invalid_file_path(self):
        with self.assertRaises(FileNotFoundError):
            TestPersistentTokensCounter(
                "/invalid/path/token_counter.json", cost_per_1M_input_tokens=0.02
            )

    def test_save_and_load(self):
        counter = TestPersistentTokensCounter(
            self.file_path,
            cost_per_1M_input_tokens=0.02,
            cost_per_1M_output_tokens=0.05,
        )
        usage_stats = TokenUsageStats(
            total_tokens=1500,
            prompt_tokens=1000,
            completion_tokens=500,
            cost=0.02,
            timestamp=datetime.now(),
        )
        counter.update(usage_stats)
        counter.save()
        loaded_counter = TestPersistentTokensCounter.load(self.file_path)
        self.assertEqual(len(loaded_counter.usage_history), 1)
        self.assertEqual(loaded_counter.prompt_tokens_count, 1000)
        self.assertEqual(loaded_counter.completion_tokens_count, 500)
        self.assertEqual(loaded_counter.total_tokens_count, 1500)
        self.assertEqual(loaded_counter.cost, 4.5e-5)


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
