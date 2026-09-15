from __future__ import annotations

import logging
import json
import os
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
        api_key = resolve_api_key(config)
        if not api_key:
            env_name = config.get("api_key_env") or "configured environment variable"
            raise RuntimeError(
                f"Missing API key. Set llm.api_key in conf.yaml or environment variable {env_name}."
            )

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
        self._max_tokens = int(config.get("max_tokens", 4096))
        self._selection_max_tokens = int(config.get("selection_max_tokens", min(2048, self._max_tokens)))
        self._codegen_max_tokens = int(config.get("codegen_max_tokens", self._max_tokens))
        self._use_json_response_format = bool(config.get("use_json_response_format", True))
        self._thinking_mode = normalize_thinking_mode(config.get("thinking", "disabled"))
        self._selection_thinking_mode = normalize_thinking_mode(
            config.get("selection_thinking", self._thinking_mode)
        )
        self._codegen_thinking_mode = normalize_thinking_mode(
            config.get("codegen_thinking", self._thinking_mode)
        )
        self._client = OpenAI(
            api_key=api_key,
            base_url=config.get("base_url"),
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
        response = self._client.chat.completions.create(**self._completion_kwargs(
            messages, timeout_seconds or self._timeout_seconds, self._max_tokens,
            json_mode=False, thinking_mode=self._thinking_mode,
        ))
        return response.choices[0].message.content or ""

    @limited("llm")
    def complete_json(self, messages: list[dict[str, str]], timeout_seconds: float | None = None) -> dict[str, Any]:
        response = self._client.chat.completions.create(**self._completion_kwargs(
            messages, timeout_seconds or self._timeout_seconds, self._max_tokens,
            json_mode=True, thinking_mode=self._thinking_mode,
        ))
        content = response.choices[0].message.content or "{}"
        logger.info("LLM response received: %d chars", len(content))
        return extract_json_object(content)

    @limited("llm")
    def complete_selection_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        response = self._client.with_options(
            max_retries=self._selection_max_retries
        ).chat.completions.create(**self._completion_kwargs(
            messages, self._selection_timeout_seconds, self._selection_max_tokens,
            json_mode=True, thinking_mode=self._selection_thinking_mode,
        ))
        content = response.choices[0].message.content or "{}"
        logger.info("LLM selection response received: %d chars", len(content))
        return extract_json_object(content)

    @limited("llm")
    def complete_codegen_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        response = self._client.chat.completions.create(**self._completion_kwargs(
            messages, self._codegen_timeout_seconds, self._codegen_max_tokens,
            json_mode=True, thinking_mode=self._codegen_thinking_mode,
        ))
        content = response.choices[0].message.content or "{}"
        logger.info("LLM codegen response received: %d chars", len(content))
        return extract_json_object(content)

    @limited("llm")
    def complete_codegen_text(self, messages: list[dict[str, str]]) -> str:
        response = self._client.chat.completions.create(**self._completion_kwargs(
            messages, self._codegen_timeout_seconds, self._codegen_max_tokens,
            json_mode=False, thinking_mode=self._codegen_thinking_mode,
        ))
        return response.choices[0].message.content or ""

    def _completion_kwargs(
        self,
        messages: list[dict[str, str]],
        timeout_seconds: float,
        max_tokens: int,
        *,
        json_mode: bool,
        thinking_mode: str | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._config["model"],
            "messages": messages,
            "temperature": float(self._config.get("temperature", 0.1)),
            "max_tokens": max_tokens,
            "timeout": timeout_seconds,
        }
        if json_mode and self._use_json_response_format:
            kwargs["response_format"] = {"type": "json_object"}
        if thinking_mode:
            kwargs["extra_body"] = {"thinking": {"type": thinking_mode}}
        return kwargs


def resolve_api_key(config: dict[str, Any]) -> str:
    direct = str(config.get("api_key") or "").strip()
    if direct.startswith("${") and direct.endswith("}"):
        return os.environ.get(direct[2:-1].strip(), "").strip()
    if direct:
        return direct
    env_name = str(config.get("api_key_env") or "").strip()
    if not env_name:
        return ""
    from_environment = os.environ.get(env_name, "").strip()
    if from_environment:
        return from_environment
    if len(env_name) > 32:
        logger.warning("llm.api_key_env appears to contain a key; use llm.api_key or an environment variable name")
        return env_name
    return ""


def normalize_thinking_mode(value: Any) -> str | None:
    if isinstance(value, bool):
        return "enabled" if value else "disabled"
    text = str(value or "").strip().lower()
    if text in {"enabled", "disabled"}:
        return text
    return None


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
