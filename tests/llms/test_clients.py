import tempfile
import unittest
from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock, patch

from mitoolspro.exceptions import ArgumentValueError
from mitoolspro.llms.clients import OpenAIClient, OpenAITokensCounter, OllamaClient
from mitoolspro.llms.objects import ModelRegistry, Prompt


class TestOpenAIClient(TestCase):
    def setUp(self):
        self.model = "gpt-4o-mini"
        self.client = OpenAIClient(api_key="test_key", model=self.model)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_initialization(self):
        self.assertEqual(self.client.model, self.model)
        self.assertEqual(len(self.client.raw_responses), 0)
        self.assertIsNone(self.client.counter)
        self.assertFalse(self.client.beta)

        temp_file = self.temp_path / "test.json"
        client_with_counter = OpenAIClient(
            api_key="test_key",
            model=self.model,
            counter=OpenAITokensCounter(temp_file, model=self.model),
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

    def test_parse_request_with_string(self):
        prompt_text = "Test string prompt"
        result = self.client.parse_request(prompt_text)
        expected = {
            "model": self.model,
            "messages": [{"role": "user", "content": "Test string prompt"}],
        }
        self.assertEqual(result, expected)

    @patch('mitoolspro.llms.clients.OpenAI')
    def test_get_response_normal_mode(self, mock_openai):
        # Setup mock
        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_response = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        
        client = OpenAIClient(api_key="test_key", model=self.model, beta=False)
        
        request = {"model": self.model, "messages": [{"role": "user", "content": "test"}]}
        
        result = client._get_response(request)
        
        mock_client.chat.completions.create.assert_called_once_with(**request)
        self.assertEqual(result, mock_response)

    @patch('mitoolspro.llms.clients.OpenAI')
    def test_get_response_beta_mode(self, mock_openai):
        # Setup mock
        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_response = Mock()
        mock_client.beta.chat.completions.parse.return_value = mock_response
        
        client = OpenAIClient(api_key="test_key", model=self.model, beta=True)
        
        request = {"model": self.model, "messages": [{"role": "user", "content": "test"}]}
        
        result = client._get_response(request)
        
        mock_client.beta.chat.completions.parse.assert_called_once_with(**request)
        self.assertEqual(result, mock_response)

    def test_parse_response(self):
        mock_response = Mock()
        mock_message = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = mock_message
        
        result = self.client.parse_response(mock_response)
        self.assertEqual(result, mock_message)

    @patch('mitoolspro.llms.clients.OpenAI')
    def test_request_without_counter(self, mock_openai):
        # Setup mock
        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = "Test response"
        mock_client.chat.completions.create.return_value = mock_response
        
        client = OpenAIClient(api_key="test_key", model=self.model)
        
        result = client.request("Test prompt")
        
        # Should add response to raw_responses
        self.assertEqual(len(client.raw_responses), 1)
        self.assertEqual(client.raw_responses[0], mock_response)
        self.assertEqual(result, "Test response")

    @patch('mitoolspro.llms.clients.OpenAI')
    def test_request_with_counter(self, mock_openai):
        # Setup mock
        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = "Test response"
        mock_client.chat.completions.create.return_value = mock_response
        
        # Setup mock counter
        mock_counter = Mock()
        mock_usage_stats = Mock()
        mock_counter.get_usage_stats.return_value = mock_usage_stats
        
        client = OpenAIClient(api_key="test_key", model=self.model, counter=mock_counter)
        
        result = client.request("Test prompt")
        
        # Should call counter methods
        mock_counter.get_usage_stats.assert_called_once_with(mock_response)
        mock_counter.update.assert_called_once_with(mock_usage_stats)
        self.assertEqual(result, "Test response")


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


class TestOllamaClient(TestCase):
    def setUp(self):
        self.model = "gemma3:12b"
        self.base_url = "http://localhost:11434"
        self.client = OllamaClient(model=self.model, base_url=self.base_url)

    def test_initialization_defaults(self):
        default_client = OllamaClient()
        self.assertEqual(default_client.model, "gemma3:12b")
        self.assertIsNone(default_client.counter)
        self.assertEqual(len(default_client.raw_responses), 0)

    def test_initialization_custom(self):
        custom_model = "llama3:8b"
        custom_url = "http://192.168.1.100:11434"
        client = OllamaClient(model=custom_model, base_url=custom_url)
        
        self.assertEqual(client.model, custom_model)
        self.assertIsNone(client.counter)
        self.assertEqual(len(client.raw_responses), 0)

    def test_parse_request_with_string(self):
        prompt_text = "Hello, how are you?"
        result = self.client.parse_request(prompt_text)
        
        expected = {
            "model": self.model,
            "prompt": prompt_text,
        }
        self.assertEqual(result, expected)

    def test_parse_request_with_prompt_object(self):
        prompt = Prompt(text="What is the weather like?")
        result = self.client.parse_request(prompt)
        
        expected = {
            "model": self.model,
            "prompt": "What is the weather like?",
        }
        self.assertEqual(result, expected)

    def test_get_model_info(self):
        result = self.client.get_model_info()
        expected = {"name": "Ollama", "model": self.model}
        self.assertEqual(result, expected)

    def test_model_name(self):
        result = self.client.model_name()
        self.assertEqual(result, self.model)

    def test_parse_response(self):
        mock_response = {
            "message": {
                "content": "This is a test response from Ollama"
            }
        }
        
        result = self.client.parse_response(mock_response)
        self.assertEqual(result, "This is a test response from Ollama")

    def test_parse_response_empty_content(self):
        mock_response = {
            "message": {
                "content": ""
            }
        }
        
        result = self.client.parse_response(mock_response)
        self.assertEqual(result, "")

    def test_client_attributes(self):
        # Test that all expected attributes exist
        self.assertTrue(hasattr(self.client, 'model'))
        self.assertTrue(hasattr(self.client, 'client'))
        self.assertTrue(hasattr(self.client, 'counter'))
        self.assertTrue(hasattr(self.client, 'raw_responses'))
        
        # Test types
        self.assertIsInstance(self.client.model, str)
        self.assertIsInstance(self.client.raw_responses, list)

    @patch('mitoolspro.llms.clients.OllamaRawClient')
    def test_get_response(self, mock_ollama_client):
        # Setup mock
        mock_client_instance = Mock()
        mock_ollama_client.return_value = mock_client_instance
        mock_response = {"message": {"content": "Ollama test response"}}
        mock_client_instance.chat.return_value = mock_response
        
        client = OllamaClient(model=self.model, base_url=self.base_url)
        
        request = {"model": self.model, "prompt": "test prompt"}
        
        result = client._get_response(request)
        
        mock_client_instance.chat.assert_called_once_with(
            messages=[{"role": "user", "content": "test prompt"}],
            model=self.model
        )
        self.assertEqual(result, mock_response)

    @patch('mitoolspro.llms.clients.OllamaRawClient')
    def test_request(self, mock_ollama_client):
        # Setup mock
        mock_client_instance = Mock()
        mock_ollama_client.return_value = mock_client_instance
        mock_response = {"message": {"content": "Ollama request response"}}
        mock_client_instance.chat.return_value = mock_response
        
        client = OllamaClient(model=self.model, base_url=self.base_url)
        
        result = client.request("Test request prompt")
        
        # Should add response to raw_responses
        self.assertEqual(len(client.raw_responses), 1)
        self.assertEqual(client.raw_responses[0], mock_response)
        self.assertEqual(result, "Ollama request response")

    @patch('mitoolspro.llms.clients.OllamaRawClient')
    def test_request_with_prompt_object(self, mock_ollama_client):
        # Setup mock
        mock_client_instance = Mock()
        mock_ollama_client.return_value = mock_client_instance
        mock_response = {"message": {"content": "Prompt object response"}}
        mock_client_instance.chat.return_value = mock_response
        
        client = OllamaClient(model=self.model, base_url=self.base_url)
        prompt = Prompt(text="Prompt object test")
        
        result = client.request(prompt)
        
        # Should parse prompt object correctly
        mock_client_instance.chat.assert_called_once_with(
            messages=[{"role": "user", "content": "Prompt object test"}],
            model=self.model
        )
        self.assertEqual(result, "Prompt object response")

    @patch('mitoolspro.llms.clients.OllamaRawClient')
    def test_request_with_kwargs(self, mock_ollama_client):
        # Setup mock
        mock_client_instance = Mock()
        mock_ollama_client.return_value = mock_client_instance
        mock_response = {"message": {"content": "Response with kwargs"}}
        mock_client_instance.chat.return_value = mock_response
        
        client = OllamaClient(model=self.model, base_url=self.base_url)
        
        result = client.request("Test prompt", temperature=0.7, max_tokens=100)
        
        # Should pass kwargs to the chat method
        mock_client_instance.chat.assert_called_once_with(
            messages=[{"role": "user", "content": "Test prompt"}],
            model=self.model,
            temperature=0.7,
            max_tokens=100
        )
        self.assertEqual(result, "Response with kwargs")


if __name__ == "__main__":
    unittest.main()
