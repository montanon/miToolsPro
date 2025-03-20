import os
from datetime import datetime
from pathlib import Path
from typing import Dict

from openai import OpenAI
from openai.types.chat import ChatCompletion

from mitoolspro.llms.objects import (
    LLMModel,
    PersistentTokensCounter,
    Prompt,
    TokenUsageStats,
)


class OpenAIClient(LLMModel):
    _roles = ["system", "user", "assistant", "tool", "function"]

    def __init__(
        self,
        api_key: str = None,
        model: str = "gpt-4o-mini",
        counter: "OpenAITokensCounter" = None,
        beta: bool = False,
    ):
        self.model = model
        self.client = OpenAI(
            api_key=api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        )
        self.raw_responses = []
        self.counter = counter
        self.beta = beta

    def parse_request(self, prompt: Prompt) -> Dict:
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt.text}],
        }

    def request(self, request: Prompt, **kwargs) -> Dict:
        request = self.parse_request(request)
        response = self._get_response(request, **kwargs)
        self.raw_responses.append(response)
        if self.counter is not None:
            usage = self.counter.get_usage_stats(response)
            self.counter.update(usage)
        return self.parse_response(response)

    def parse_response(self, response: Dict) -> str:
        return response.choices[0].message

    def get_model_info(self) -> Dict:
        return {"name": "OpenAI", "model": self.model}

    def _get_response(self, request: Dict, **kwargs) -> Dict:
        if not self.beta:
            return self.client.chat.completions.create(**request, **kwargs)
        else:
            return self.client.beta.chat.completions.parse(**request, **kwargs)

    def model_name(self) -> str:
        return self.model


class OpenAITokensCounter(PersistentTokensCounter):
    def __new__(cls, file_path: Path, model: str = "gpt-4o-mini"):
        return super().__new__(cls, file_path=file_path, source="openai", model=model)

    def __init__(self, file_path: Path, model: str = "gpt-4o-mini"):
        super().__init__(
            file_path=file_path,
            model=model,
            source="openai",
        )

    def get_usage_stats(self, response: ChatCompletion) -> TokenUsageStats:
        total_tokens = response.usage.total_tokens
        return TokenUsageStats(
            source="openai",
            model=self.model,
            model_cost=self.model_cost,
            total_tokens=total_tokens,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            cost=response.usage.prompt_tokens * (self.model_cost["input"] / 1_000_000)
            + response.usage.completion_tokens
            * (self.model_cost["output"] / 1_000_000),
            timestamp=datetime.fromtimestamp(response.created),
        )

    def count_tokens(self, text):
        raise NotImplementedError
