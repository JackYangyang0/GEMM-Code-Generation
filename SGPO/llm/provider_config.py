"""Resolve one OpenAI-compatible provider without mixing credentials."""
from collections.abc import Mapping
from copy import deepcopy


PROVIDERS = ('qwen', 'deepseek', 'chatgpt', 'cmecloud')


def resolve_provider_config(config):
    if not isinstance(config, Mapping):
        raise ValueError('llm must be a configuration mapping')
    grouped = any(name in config for name in PROVIDERS)
    if not grouped:
        return deepcopy(dict(config))
    provider = config.get('provider')
    if provider not in PROVIDERS:
        raise ValueError('llm.provider must be one of: ' + ', '.join(PROVIDERS))
    selected = config.get(provider)
    if not isinstance(selected, Mapping):
        raise ValueError(f'llm.{provider} must be a configuration mapping')
    resolved = deepcopy(dict(selected))
    resolved['provider'] = provider
    for name in ('model', 'base_url'):
        if not isinstance(resolved.get(name), str) or not resolved[name].strip():
            raise ValueError(f'Set llm.{provider}.{name} before selecting this provider')
    # Do not send vendor-specific thinking parameters to an arbitrary endpoint.
    resolved.setdefault('thinking', None)
    return resolved
