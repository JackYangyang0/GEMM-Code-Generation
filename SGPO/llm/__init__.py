from .base import LLMClient

__all__ = [
    "LLMClient",
    "OpenAICompatibleClient",
    "build_get_strategy_messages",
    "get_strategy_ids_from_llm",
]


def __getattr__(name):
    if name == "OpenAICompatibleClient":
        from .openai_client import OpenAICompatibleClient
        return OpenAICompatibleClient
    if name in {"build_get_strategy_messages", "get_strategy_ids_from_llm"}:
        from . import strategy_selector
        return getattr(strategy_selector, name)
    raise AttributeError(name)
