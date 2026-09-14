from __future__ import annotations

import logging
import json
from dataclasses import dataclass
from typing import Any

from SGPO.llm.base import LLMClient
from SGPO.utils.execution import limited
from openai import OpenAI

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_SELECTION_TIMEOUT_SECONDS = 30
DEFAULT_CODEGEN_TIMEOUT_SECONDS = 300
DEFAULT_MAX_RETRIES = 3
DEFAULT_SELECTION_MAX_RETRIES = 1


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    api_key: str
    base_url: str
    temperature: float = 0.1
    timeout_seconds: int = 120
    max_retries: int = 3


class OpenAICompatibleClient(LLMClient):
    def __init__(self, config):
        api_key = config["api_key"]
        if not api_key:
            raise RuntimeError("Missing API key. Set llm.api_key in conf.yaml.")

        self._config = config
        self._timeout_seconds = numeric_config(
            config,
            "timeout_seconds",
            fallback_key="request_timeout_seconds",
            default=DEFAULT_TIMEOUT_SECONDS,
        )
        self._codegen_timeout_seconds = numeric_config(
            config,
            "codegen_timeout_seconds",
            default=max(DEFAULT_CODEGEN_TIMEOUT_SECONDS, self._timeout_seconds),
        )
        self._selection_timeout_seconds = numeric_config(
            config,
            "selection_timeout_seconds",
            default=min(DEFAULT_SELECTION_TIMEOUT_SECONDS, self._timeout_seconds),
        )
        self._max_retries = int(config.get("max_retries", DEFAULT_MAX_RETRIES))
        self._selection_max_retries = int(
            config.get("selection_max_retries", min(DEFAULT_SELECTION_MAX_RETRIES, self._max_retries))
        )
        self._client = OpenAI(
            api_key=api_key,
            base_url=config["base_url"],
            timeout=self._timeout_seconds,
            max_retries=self._max_retries,
        )

    @property
    def timeout_seconds(self) -> float:
        return self._timeout_seconds

    @property
    def codegen_timeout_seconds(self) -> float:
        return self._codegen_timeout_seconds

    @property
    def selection_timeout_seconds(self) -> float:
        return self._selection_timeout_seconds

    @limited("llm")
    def complete_text(self, messages: list[dict[str, str]], timeout_seconds: float | None = None) -> str:
        response = self._client.chat.completions.create(
            model=self._config["model"],
            messages=messages,
            temperature=self._config["temperature"],
            timeout=timeout_seconds or self._timeout_seconds,
        )
        return response.choices[0].message.content or ""

    @limited("llm")
    def complete_json(self, messages: list[dict[str, str]], timeout_seconds: float | None = None) -> dict[str, Any]:
        response = self._client.chat.completions.create(
            model=self._config["model"],
            messages=messages,
            temperature=self._config["temperature"],
            response_format={"type": "json_object"},
            timeout=timeout_seconds or self._timeout_seconds,
        )
        content = response.choices[0].message.content or "{}"
        logger.info("LLM response received: %d chars", len(content))
        return extract_json_object(content)

    @limited("llm")
    def complete_selection_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        response = self._client.with_options(max_retries=self._selection_max_retries).chat.completions.create(
            model=self._config["model"],
            messages=messages,
            temperature=self._config["temperature"],
            response_format={"type": "json_object"},
            timeout=self._selection_timeout_seconds,
        )
        content = response.choices[0].message.content or "{}"
        logger.info("LLM selection response received: %d chars", len(content))
        return extract_json_object(content)

    def complete_codegen_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        return self.complete_json(messages, timeout_seconds=self._codegen_timeout_seconds)

    def complete_codegen_text(self, messages: list[dict[str, str]]) -> str:
        return self.complete_text(messages, timeout_seconds=self._codegen_timeout_seconds)


def numeric_config(
    config: dict[str, Any],
    key: str,
    default: float,
    fallback_key: str | None = None,
) -> float:
    value = config.get(key)
    if value is None and fallback_key:
        value = config.get(fallback_key)
    if value is None:
        return float(default)
    return float(value)


def extract_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("LLM response does not contain a JSON object.")
    return json.loads(text[start : end + 1])
