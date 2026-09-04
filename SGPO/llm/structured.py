from __future__ import annotations

from typing import Any, Callable, TypeVar

T = TypeVar("T")


def parse_with_retry(
    producer: Callable[[], dict[str, Any]],
    parser: Callable[[dict[str, Any]], T],
    retries: int = 1,
) -> T:
    last_error: Exception | None = None
    for _ in range(retries + 1):
        try:
            return parser(producer())
        except Exception as exc:  # pragma: no cover - returned in tests through final raise
            last_error = exc
    assert last_error is not None
    raise last_error
