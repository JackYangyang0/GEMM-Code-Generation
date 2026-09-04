from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMClient(ABC):
    @abstractmethod
    def complete_text(self, messages: list[dict[str, str]]) -> str:
        raise NotImplementedError

    @abstractmethod
    def complete_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        raise NotImplementedError
