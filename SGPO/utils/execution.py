"""Bounded chain concurrency with ordered results and isolated publication."""
from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from threading import BoundedSemaphore, local
from typing import Any


@dataclass(frozen=True)
class ExecutionConfig:
    chain_workers: int = 3
    llm_max_concurrency: int = 3
    compile_workers: int = 2
    gpu_verification_workers: int = 1


@dataclass
class ChainOutcome:
    value: Any = None
    error: Exception | None = None
    latest: dict = field(default_factory=dict)


_config = ExecutionConfig()
_local = local()
_gates = {"llm": BoundedSemaphore(3), "compile": BoundedSemaphore(2),
          "gpu_verification": BoundedSemaphore(1)}


def configure_execution(config: dict | None = None) -> ExecutionConfig:
    global _config, _gates
    values = config or {}
    defaults = ExecutionConfig()
    parsed = {}
    for name in defaults.__dataclass_fields__:
        value = values.get(name, getattr(defaults, name))
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"execution.{name} must be a positive integer")
        parsed[name] = value
    if parsed["gpu_verification_workers"] != 1:
        raise ValueError("execution.gpu_verification_workers must be 1 for isolated GPU timing")
    _config = ExecutionConfig(**parsed)
    _gates = {"llm": BoundedSemaphore(_config.llm_max_concurrency),
              "compile": BoundedSemaphore(_config.compile_workers),
              "gpu_verification": BoundedSemaphore(1)}
    return _config


def execution_config() -> ExecutionConfig:
    return _config


@contextmanager
def execution_slot(kind: str):
    depths = getattr(_local, "slot_depths", None)
    if depths is None:
        depths = _local.slot_depths = {}
    depth = depths.get(kind, 0)
    gate = _gates[kind]
    if depth == 0:
        gate.acquire()
    depths[kind] = depth + 1
    try:
        yield
    finally:
        depths[kind] -= 1
        if depth == 0:
            gate.release()


def limited(kind: str):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with execution_slot(kind):
                return function(*args, **kwargs)
        return wrapped
    return decorate


def chain_namespace() -> str | None:
    return getattr(_local, "chain_namespace", None)


def defer_latest(path, data) -> bool:
    latest = getattr(_local, "latest", None)
    if latest is None:
        return False
    latest[path] = copy.deepcopy(data)
    return True


@contextmanager
def chain_client(client):
    config = getattr(client, "_config", None)
    if not isinstance(config, dict):
        yield client
        return
    worker = type(client)(copy.deepcopy(config))
    try:
        yield worker
    finally:
        worker._client.close()


def parallel_chain_map(function, items, namespace):
    """Run independent chains; a failed chain does not cancel its siblings."""
    def invoke(item):
        previous_namespace = chain_namespace()
        previous_latest = getattr(_local, "latest", None)
        _local.chain_namespace = namespace(item)
        _local.latest = {}
        outcome = ChainOutcome()
        try:
            outcome.value = function(item)
        except Exception as exc:
            outcome.error = exc
        finally:
            outcome.latest = _local.latest
            _local.chain_namespace = previous_namespace
            _local.latest = previous_latest
        return outcome

    items = list(items)
    if len(items) <= 1 or _config.chain_workers == 1:
        return [invoke(item) for item in items]
    with ThreadPoolExecutor(max_workers=_config.chain_workers, thread_name_prefix="sgpo-chain") as pool:
        return list(pool.map(invoke, items))
